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

A tau belongs to a model state, not to the protocol: supervising confidence moves the
channel's absolute level, so the same threshold means different things in two runs and
a bare occlusion accuracy without its grid is not comparable to anything.  Every
accuracy here therefore travels with the basis that produced it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from arc.training.losses import (
    synchronized_differences,
    synchronized_pair_indices,
)

# Spans the observed track-confidence range: `expp1` output is `1 + exp(x)`, and
# archived runs sit in the low hundreds with a p05 near 36.  A grid rather than one
# value because the useful threshold is not known ahead of the run.
#
# Kept at its original values, and kept reported, because four archived cluster runs
# were scored on exactly these thresholds; changing them would silently strand those
# numbers.  It is no longer the grid to read an optimum off -- see the relative grid
# below for why.
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

# The absolute grid above is not comparable across runs, and the reason is structural
# rather than a matter of picking wider limits.  The confidence term's optimum is
# `conf* = alpha/err`, so as training drives the error down the whole channel level
# climbs -- by a different factor in every arm.  Measured across the archived weight
# sweep, mean visible confidence ran 6175 (weight 0.01) to 25452 (weight 0.001), and
# a threshold of 300 was 0.63x the run's own implied optimum in one arm against 0.35x
# in another.  Occlusion accuracy was still rising at 300 in all of them, so the best
# figure each reported was an artifact of where the grid stopped, not an optimum.
#
# So the grid is derived from the run's own state instead: multiples of
# `implied_optimal_confidence = alpha/mean_error`, the level the term is itself
# pulling confidence towards.  Two runs then sample the same points of their own
# curves and their occlusion accuracies can be read side by side.
#
# Geometric, because "enough resolution near the optimum" is a statement about ratios
# once the grid is scale-relative -- half-octave steps hold the same relative spacing
# everywhere, in every run.  The span is set by what the sweep showed: mean visible
# confidence sat 13x to 30x above the implied optimum, so the top at 256x clears it by
# 4-10x (that the accuracy has stopped rising there is checked per run and reported as
# `at_grid_edge`, never assumed), and the bottom at 1/32x lands near the old grid's
# low end so the two overlap.
DEFAULT_CONFIDENCE_TAU_MULTIPLES: tuple[float, ...] = tuple(
    float(2.0 ** (exponent / 2.0)) for exponent in range(-10, 17)
)

# Bumped when the shape of the report changes, so a reader can tell which keys to
# expect.  Version 1 is the absolute-grid-only block written by the archived runs.
CONFIDENCE_DIAGNOSTICS_VERSION = 2


def confidence_occlusion_diagnostics(
    confidence: torch.Tensor,
    per_sample_error: torch.Tensor,
    visible_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    confidence_alpha: float | None = None,
    taus: Sequence[float] = DEFAULT_CONFIDENCE_TAUS,
    multiples: Sequence[float] = DEFAULT_CONFIDENCE_TAU_MULTIPLES,
) -> dict:
    """Summarize how well confidence separates visible from occluded samples.

    ``visible_mask`` is per-camera-per-time ground-truth visibility, the same
    definition the position loss masks on.  ``valid_mask`` restricts to samples with
    finite predictions and targets; it defaults to everything.

    Returns the confidence and error means on each side of the visibility split,
    plus, for each tau, the accuracy of calling ``confidence > tau`` "visible".  The
    per-side accuracies are named to match the scorer's ``occlusion_accuracy_for_vis1``
    and ``..._for_vis0`` so the numbers can be read side by side.

    Two grids, because a tau belongs to a model state and not to the protocol.
    ``tau_grid`` is the fixed absolute sweep the archived runs were scored on, kept so
    those numbers stay readable.  ``relative_tau_grid`` is the one to read an optimum
    off: it is ``multiples`` times this run's own ``implied_optimal_confidence``, and
    needs ``confidence_alpha`` -- without it there is no run-relative scale and the
    relative grid is ``None`` rather than silently absolute.  Every reported accuracy
    is accompanied by the basis that produced it.

    ``best`` names the argmax of each grid outright, so no reader has to infer it from
    the rows, and flags ``at_grid_edge`` when that argmax sits at either end -- the
    grid then stopped before the curve turned and the "best" is a property of the
    limits, not of the model.  Read it against
    ``trivial_all_visible_occlusion_accuracy``, which is what calling everything
    visible already scores and therefore the bar any threshold has to clear at all.
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
        "confidence_diagnostics_version": CONFIDENCE_DIAGNOSTICS_VERSION,
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

    # Calling everything visible needs no model at all, so it is the number a
    # threshold has to beat before it has earned its place. At the released
    # checkpoint none does, which is only visible once this sits next to the grid.
    report["trivial_all_visible_occlusion_accuracy"] = (
        report["visible_count"] / report["sample_count"]
    )
    report["implied_optimal_confidence"] = _implied_optimal_confidence(
        confidence_alpha,
        report["mean_error"],
    )

    report["tau_grid_basis"] = (
        "absolute: DEFAULT_CONFIDENCE_TAUS, fixed in confidence units. Retained for "
        "comparability with runs scored before the relative grid existed; not "
        "comparable across runs whose confidence scale differs"
    )
    report["tau_grid"] = [
        _tau_row(confidence, float(tau), visible_mask, valid_mask, visible, occluded)
        for tau in taus
    ]

    implied_optimal = report["implied_optimal_confidence"]
    if implied_optimal is None:
        # No alpha, or an error of zero: there is no run-relative scale to build on.
        # Reporting the absolute grid under a relative name would be the exact
        # confusion this block exists to end, so say so instead.
        report["relative_tau_grid_basis"] = None
        report["relative_tau_grid"] = None
    else:
        report["relative_tau_grid_basis"] = (
            "relative: multiples of implied_optimal_confidence "
            f"(= alpha/mean_error = {implied_optimal:.6g}), the level this run's own "
            "confidence term is pulling towards. Chosen relative to the run because "
            "conf* = alpha/err makes the channel's absolute level a function of how "
            "far training got, so a fixed grid samples a different part of every "
            "run's curve"
        )
        report["relative_tau_grid"] = [
            {
                "multiple": float(multiple),
                **_tau_row(
                    confidence,
                    float(multiple) * implied_optimal,
                    visible_mask,
                    valid_mask,
                    visible,
                    occluded,
                ),
            }
            for multiple in multiples
        ]

    report["best"] = {
        "absolute": _best_row(report["tau_grid"], report["tau_grid_basis"]),
        "relative": _best_row(
            report["relative_tau_grid"],
            report["relative_tau_grid_basis"],
        ),
    }
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


def _implied_optimal_confidence(
    alpha: float | None,
    mean_error: float | None,
) -> float | None:
    """Where the confidence term wants confidence to sit, given the current error.

    The optimum of ``conf * err - alpha * log(conf)`` is ``conf* = alpha/err``, and
    ``mean_error`` is the same per-sample Huber error alpha was calibrated against --
    not the L2 metric error, which is a different quantity.  ``None`` rather than an
    infinity when the error has reached zero: there is no finite level to compare a
    threshold against, and a NaN propagating into the grid would be worse than a
    missing grid.
    """

    if alpha is None or mean_error is None:
        return None
    alpha = float(alpha)
    mean_error = float(mean_error)
    if not math.isfinite(alpha) or alpha <= 0:
        return None
    if not math.isfinite(mean_error) or mean_error <= 0:
        return None
    implied = alpha / mean_error
    return implied if math.isfinite(implied) else None


def _tau_row(
    confidence: torch.Tensor,
    tau: float,
    visible_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    visible: torch.Tensor,
    occluded: torch.Tensor,
) -> dict:
    """One threshold's worth of agreement, shared by both grids.

    Shared rather than duplicated so the two grids can never drift into measuring
    subtly different things and being compared anyway.
    """

    predicted_visible = confidence > tau
    return {
        "tau": tau,
        "occlusion_accuracy": _agreement(predicted_visible, visible_mask, valid_mask),
        "occlusion_accuracy_for_vis1": _agreement(predicted_visible, visible_mask, visible),
        "occlusion_accuracy_for_vis0": _agreement(predicted_visible, visible_mask, occluded),
        "predicted_visible_fraction": _masked_mean(predicted_visible.float(), valid_mask),
    }


def _best_row(grid: list[dict] | None, basis: str | None) -> dict | None:
    """The grid's argmax, named outright rather than left to the reader.

    ``at_grid_edge`` is true when the best row is the first or last one, which is the
    machine-readable form of "this grid stopped before the curve turned".  Both ends
    count: the archived runs' optimum was pinned to the top of the grid, and the
    released checkpoint's to the bottom, and in both cases the reported best is a
    property of the limits rather than of the model.  A flat grid ties at the first
    row and so reads as an edge, which is the honest answer -- it located nothing.
    """

    if not grid:
        return None
    scored = [
        (index, row)
        for index, row in enumerate(grid)
        if row["occlusion_accuracy"] is not None
    ]
    if not scored:
        return None
    index, best = max(scored, key=lambda pair: pair[1]["occlusion_accuracy"])
    return {
        "tau": best["tau"],
        "multiple": best.get("multiple"),
        "occlusion_accuracy": best["occlusion_accuracy"],
        "predicted_visible_fraction": best["predicted_visible_fraction"],
        "at_grid_edge": index in (0, len(grid) - 1),
        "basis": basis,
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
