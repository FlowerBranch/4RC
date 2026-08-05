"""Non-differentiable reporting for the bounded 4RC experiments.

Kept apart from ``losses.py`` because none of this returns a graph.  Like the loss
terms these are pure tensor functions -- the caller passes an already-gathered
visibility mask rather than a scene -- but they exist to make a run's outcome
readable, not to train anything.

The question these answer is whether supervising confidence buys anything
downstream.  The cluster scorer turns confidence into an occlusion call by an
*absolute* threshold, ``~(max over cameras of conf > tau)``, and scores it against
ground truth reduced with ``visibility.any(axis=0)``.  Because that fusion is a max,
a per-camera target composes into the any-camera decision exactly, so per-camera
visibility is the right thing to measure against here.

Two things make this worth reporting every run rather than inferring later.  The
confidence term only separates visible from occluded if the *position error* does,
which is a property of the data and worth failing fast on.  And the threshold is a
free parameter that lives in another repository, so the grid below is what tells
you which value is even workable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from arc.training.losses import (
    synchronized_differences,
    synchronized_pair_indices,
)

# Spans the observed track-confidence range: `expp1` output is `1 + exp(x)`, and
# archived runs sit in the low hundreds with a p05 near 36.  A grid rather than one
# value because the useful threshold is not known ahead of the run.
DEFAULT_CONFIDENCE_TAUS: tuple[float, ...] = (
    1.5,
    5.0,
    25.0,
    50.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
)


def confidence_occlusion_diagnostics(
    confidence: torch.Tensor,
    per_sample_error: torch.Tensor,
    visible_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    taus: Sequence[float] = DEFAULT_CONFIDENCE_TAUS,
) -> dict:
    """Summarize how well confidence separates visible from occluded samples.

    ``visible_mask`` is per-camera-per-time ground-truth visibility, the same
    definition the position loss masks on.  ``valid_mask`` restricts to samples with
    finite predictions and targets; it defaults to everything.

    Returns the confidence and error means on each side of the visibility split,
    plus, for each tau, the accuracy of calling ``confidence > tau`` "visible".  The
    per-side accuracies are named to match the scorer's ``occlusion_accuracy_for_vis1``
    and ``..._for_vis0`` so the numbers can be read side by side.
    """

    if confidence.shape != per_sample_error.shape:
        raise ValueError(
            "confidence and per_sample_error must have equal shape, got "
            f"{tuple(confidence.shape)} and {tuple(per_sample_error.shape)}"
        )
    if visible_mask.shape != confidence.shape:
        raise ValueError(
            f"visible_mask must have shape {tuple(confidence.shape)}, got "
            f"{tuple(visible_mask.shape)}"
        )
    if visible_mask.dtype != torch.bool:
        raise ValueError(f"visible_mask must be boolean, got {visible_mask.dtype}")
    if valid_mask is None:
        valid_mask = torch.ones_like(visible_mask)
    elif valid_mask.shape != confidence.shape:
        raise ValueError(
            f"valid_mask must have shape {tuple(confidence.shape)}, got "
            f"{tuple(valid_mask.shape)}"
        )
    elif valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be boolean, got {valid_mask.dtype}")
    if not valid_mask.any():
        raise ValueError("No valid samples for the confidence diagnostics")

    confidence = confidence.detach().float()
    per_sample_error = per_sample_error.detach().float()
    visible = visible_mask & valid_mask
    occluded = (~visible_mask) & valid_mask

    report: dict = {
        "sample_count": int(valid_mask.sum().item()),
        "visible_count": int(visible.sum().item()),
        "occluded_count": int(occluded.sum().item()),
        # Overall means over exactly the samples the confidence term reduces, so
        # they are the right pair to read the term's optimum conf*=alpha/err against.
        "mean_confidence": _masked_mean(confidence, valid_mask),
        "mean_error": _masked_mean(per_sample_error, valid_mask),
        "mean_confidence_visible": _masked_mean(confidence, visible),
        "mean_confidence_occluded": _masked_mean(confidence, occluded),
        "mean_error_visible": _masked_mean(per_sample_error, visible),
        "mean_error_occluded": _masked_mean(per_sample_error, occluded),
    }

    # If the errors do not separate, no confidence that is monotone in error can
    # separate either. Surfacing the gap directly makes that diagnosable from the
    # summary instead of from a downstream metric that moved for other reasons.
    if report["mean_error_visible"] is not None and report["mean_error_occluded"] is not None:
        report["error_separation"] = (
            report["mean_error_occluded"] - report["mean_error_visible"]
        )
    else:
        report["error_separation"] = None

    grid = []
    for tau in taus:
        predicted_visible = confidence > float(tau)
        grid.append(
            {
                "tau": float(tau),
                "occlusion_accuracy": _agreement(predicted_visible, visible_mask, valid_mask),
                "occlusion_accuracy_for_vis1": _agreement(
                    predicted_visible,
                    visible_mask,
                    visible,
                ),
                "occlusion_accuracy_for_vis0": _agreement(
                    predicted_visible,
                    visible_mask,
                    occluded,
                ),
                "predicted_visible_fraction": _masked_mean(
                    predicted_visible.float(),
                    valid_mask,
                ),
            }
        )
    report["tau_grid"] = grid
    return report


def synchronized_consistency_stats(
    displacements: torch.Tensor,
    slot_time_indices: torch.Tensor,
    *,
    metric_scale: float = 1.0,
) -> dict | None:
    """Per-pixel dP disagreement between synchronized observations, in metres.

    The headline de-hallucination number: same-time-index observation slots owe
    identical displacement fields, so any residual here is motion invented
    between viewpoints.  Returns ``None`` when the window has no synchronized
    pair (single camera, or a single time).
    """

    slot_time_indices = torch.as_tensor(slot_time_indices).reshape(-1)
    first, _ = synchronized_pair_indices(slot_time_indices)
    if not first:
        return None

    with torch.no_grad():
        difference = synchronized_differences(
            displacements.detach().float(),
            slot_time_indices,
            metric_scale=metric_scale,
        )
        norms = torch.linalg.vector_norm(difference, dim=-1).flatten()
        return {
            "pair_count": len(first),
            "mean_m": float(norms.mean().item()),
            "median_m": float(norms.median().item()),
            "p90_m": float(torch.quantile(norms, 0.9).item()),
        }


def temporal_injection_report(
    baseline_taps: Sequence[tuple],
    indexed_taps: Sequence[tuple],
    slot_time_indices: torch.Tensor,
) -> list[dict]:
    """How far the injected time-index signal travels through the encoder.

    Both inputs are backbone tap tuples ``(patches [1,S,P,C], camera [1,S,C],
    time [1,S,C])`` from two forwards over the same images: one without time
    indices (the released behaviour) and one with them.  Per tap this reports
    the relative movement of the time token (what the motion decoder's AdaLN
    reads) and of the patch tokens (the only channel by which frozen global
    attention could act on the index), plus how cleanly the per-frame deltas
    cluster by shared index — cosine near 1 for same-index pairs and near 0
    for different-index pairs means the signal survives as decodable structure.
    """

    slot_time_indices = torch.as_tensor(slot_time_indices).reshape(-1)
    same_first, same_second = synchronized_pair_indices(slot_time_indices)
    values = [int(value) for value in slot_time_indices.tolist()]
    different_pairs = [
        (earlier, later)
        for later in range(len(values))
        for earlier in range(later)
        if values[earlier] != values[later]
    ]

    def _pair_cosine(deltas: torch.Tensor, pairs) -> float | None:
        if not pairs:
            return None
        norms = deltas.norm(dim=-1, keepdim=True)
        if not (norms > 0).all():
            return None
        directions = deltas / norms
        cosines = [
            float((directions[a] * directions[b]).sum().item())
            for a, b in pairs
        ]
        return sum(cosines) / len(cosines)

    def _relative_change(delta: torch.Tensor, base: torch.Tensor) -> float | None:
        base_norm = float(base.norm().item())
        if base_norm == 0:
            return None
        return float(delta.norm().item()) / base_norm

    report = []
    with torch.no_grad():
        for baseline, indexed in zip(baseline_taps, indexed_taps):
            base_patches = baseline[0].detach().float()[0]
            base_time = baseline[2].detach().float()[0]
            time_delta = indexed[2].detach().float()[0] - base_time
            patch_delta = indexed[0].detach().float()[0] - base_patches
            frame_patch_delta = patch_delta.mean(dim=1)
            report.append(
                {
                    "time_token_relative_change": _relative_change(
                        time_delta, base_time
                    ),
                    "patch_token_relative_change": _relative_change(
                        patch_delta, base_patches
                    ),
                    "time_delta_cos_same_index": _pair_cosine(
                        time_delta, list(zip(same_first, same_second))
                    ),
                    "time_delta_cos_different_index": _pair_cosine(
                        time_delta, different_pairs
                    ),
                    "patch_delta_cos_same_index": _pair_cosine(
                        frame_patch_delta, list(zip(same_first, same_second))
                    ),
                    "patch_delta_cos_different_index": _pair_cosine(
                        frame_patch_delta, different_pairs
                    ),
                }
            )
    return report


def reconstruction_shift_report(
    baseline_depth: torch.Tensor,
    indexed_depth: torch.Tensor,
    baseline_pose_enc: torch.Tensor,
    indexed_pose_enc: torch.Tensor,
) -> dict:
    """How much the frozen reconstruction moves when time indices are injected.

    The embedding enters at block 13, below every geometry tap, so depth and
    pose shift even with all reconstruction weights frozen.  This quantifies
    that shift between a no-index forward and an indexed forward over the same
    images; it is the number that says whether an init scale is a perturbation
    or a demolition.
    """

    if baseline_depth.shape != indexed_depth.shape:
        raise ValueError(
            "Depth tensors must have equal shape, got "
            f"{tuple(baseline_depth.shape)} and {tuple(indexed_depth.shape)}"
        )
    if baseline_pose_enc.shape != indexed_pose_enc.shape:
        raise ValueError(
            "pose_enc tensors must have equal shape, got "
            f"{tuple(baseline_pose_enc.shape)} and {tuple(indexed_pose_enc.shape)}"
        )
    if baseline_pose_enc.shape[-1] != 9:
        raise ValueError(
            "pose_enc must have 9 trailing components, got "
            f"{tuple(baseline_pose_enc.shape)}"
        )

    with torch.no_grad():
        base_depth = baseline_depth.detach().float()
        depth_delta = (indexed_depth.detach().float() - base_depth).abs()
        relative = (depth_delta / base_depth.abs().clamp_min(1e-6)).flatten()
        pose_delta = (
            indexed_pose_enc.detach().float()
            - baseline_pose_enc.detach().float()
        ).abs()
        return {
            "depth_relative_change": {
                "mean": float(relative.mean().item()),
                "median": float(relative.median().item()),
                "p90": float(torch.quantile(relative, 0.9).item()),
            },
            "pose_enc_max_abs_change": {
                "translation": float(pose_delta[..., 0:3].max().item()),
                "quaternion": float(pose_delta[..., 3:7].max().item()),
                "fov": float(pose_delta[..., 7:9].max().item()),
            },
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not mask.any():
        return None
    return float(values[mask].mean().item())


def _agreement(
    predicted_visible: torch.Tensor,
    visible_mask: torch.Tensor,
    subset: torch.Tensor,
) -> float | None:
    if not subset.any():
        return None
    correct = predicted_visible[subset] == visible_mask[subset]
    return float(correct.float().mean().item())
