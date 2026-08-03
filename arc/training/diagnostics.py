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
