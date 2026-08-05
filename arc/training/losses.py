"""Differentiable training loss terms for the bounded 4RC experiments.

Every term here is **pure tensor math**: tensors in, tensors out.  Nothing in this
module knows about ``DumpedKubricScene``, ``SparseCorrespondences``, NumPy, Sim(3)
or the filesystem.  Gathering, masking and alignment stay on the geometry side in
``sparse_tracking.py``, which keeps the dependency one-way and lets every term be
unit-tested from hand-built tensors with no scene fixture.

Adding a term later means writing one function here and giving it a weight in
:func:`compose_tracking_loss`; nothing else has to change.  Reporting that is not
differentiable belongs in ``diagnostics.py``, not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


def track_position_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    huber_delta: float,
) -> torch.Tensor:
    """Huber loss on absolute metric track positions.

    ``delta`` is a physical threshold in metres and the residuals are metric, so
    the quadratic/linear boundary sits at a fixed distance rather than at a fixed
    fraction of the scene.  That is fine for one scene and is the reason the
    reported numbers are interpretable; it is the open question for multi-scene
    training.

    The reduction is deliberately a single ``reduction="mean"`` over the masked
    ``(K,3)`` selection rather than a mean of per-sample means.  The two are equal
    in exact arithmetic but not bit-for-bit, and this path has archived runs to
    reproduce.  Use :func:`per_sample_huber_error` when a per-sample quantity is
    needed.
    """

    _validate_position_inputs(predicted, target, mask, huber_delta)
    selected_prediction = predicted[mask]
    selected_target = target[mask]
    loss = F.huber_loss(
        selected_prediction,
        selected_target,
        reduction="mean",
        delta=huber_delta,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Track position loss produced NaN or Inf")
    return loss


def per_sample_huber_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    huber_delta: float,
) -> torch.Tensor:
    """Per-sample Huber error, averaged over the three coordinates.

    Shape ``(M,S)`` for ``(M,S,3)`` inputs.  This is what the confidence term
    weights, so it is computed unmasked: the confidence term supervises occluded
    samples too, and they are exactly the ones the position mask removes.
    """

    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must have equal shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[-1] != 3:
        raise ValueError(
            f"predicted must have a trailing 3-vector axis, got {tuple(predicted.shape)}"
        )
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    elementwise = F.huber_loss(
        predicted,
        target,
        reduction="none",
        delta=huber_delta,
    )
    return elementwise.mean(dim=-1)


def track_metric_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean L2 residual in metres.

    A diagnostic rather than a training signal, but it is reported beside the
    terms and shares their masking convention, so it lives with them.
    """

    _validate_position_inputs(predicted, target, mask, 1.0)
    metric_error = torch.linalg.vector_norm(
        predicted[mask] - target[mask],
        dim=-1,
    ).mean()
    if not torch.isfinite(metric_error):
        raise FloatingPointError("Track metric error produced NaN or Inf")
    return metric_error


def track_confidence_loss(
    confidence: torch.Tensor,
    per_sample_error: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Confidence-weighted regression in the DUSt3R style.

    ``conf * err - alpha * log(conf)``, mirroring the (dead) reference
    implementation at ``eval/mv_recon/criterion.py:450``.  Minimising over ``conf``
    gives ``conf* = alpha / err``, so confidence becomes a monotone proxy for
    accuracy.  The track head's confidence activation is ``expp1`` (``1 + exp(x)``,
    strictly greater than 1 and unbounded above), which is what makes this form
    admissible and a probability-style classification loss inadmissible.

    ``per_sample_error`` is **detached**.  Confidence and xyz are split off the same
    final conv, so this keeps the term's gradient on the confidence row of that conv
    and leaves the position rows to :func:`track_position_loss` alone.

    ``mask`` must not be the position term's visibility mask.  Occluded samples are
    precisely where the error is large, so they are where a low optimal confidence
    is learned; excluding them would remove the signal this term exists for.
    """

    if confidence.shape != per_sample_error.shape:
        raise ValueError(
            "confidence and per_sample_error must have equal shape, got "
            f"{tuple(confidence.shape)} and {tuple(per_sample_error.shape)}"
        )
    if mask.shape != confidence.shape:
        raise ValueError(
            f"mask must have shape {tuple(confidence.shape)}, got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a boolean tensor, got {mask.dtype}")
    if not mask.any():
        raise ValueError("No samples selected for the track confidence loss")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")

    selected_confidence = confidence[mask].float()
    selected_error = per_sample_error[mask].detach().float()
    if not torch.isfinite(selected_confidence).all():
        raise FloatingPointError("Track confidence contains NaN or Inf")
    if not torch.isfinite(selected_error).all():
        raise FloatingPointError("Per-sample track error contains NaN or Inf")
    if not (selected_confidence > 0).all():
        raise ValueError(
            "Track confidence must be strictly positive before log; got a minimum "
            f"of {selected_confidence.min().item()}"
        )

    loss = (
        selected_confidence * selected_error
        - float(alpha) * torch.log(selected_confidence)
    ).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Track confidence loss produced NaN or Inf")
    return loss


def synchronized_pair_indices(
    slot_time_indices: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Slot pairs ``(a, b)``, ``a < b``, that share a semantic time index."""

    slot_time_indices = torch.as_tensor(slot_time_indices)
    if slot_time_indices.ndim != 1:
        raise ValueError(
            "slot_time_indices must be one-dimensional, got shape "
            f"{tuple(slot_time_indices.shape)}"
        )
    values = [int(value) for value in slot_time_indices.tolist()]
    first: list[int] = []
    second: list[int] = []
    for later in range(len(values)):
        for earlier in range(later):
            if values[earlier] == values[later]:
                first.append(earlier)
                second.append(later)
    return first, second


def synchronized_differences(
    displacements: torch.Tensor,
    slot_time_indices: torch.Tensor,
    *,
    metric_scale: float = 1.0,
) -> torch.Tensor:
    """Metric-scaled dP differences over every synchronized slot pair.

    ``displacements`` is the raw ``track_multi`` grid ``(1,Q,S,H,W,3)``; the
    result stacks one difference per same-time-index slot pair along dim 2.
    ``metric_scale`` lifts raw differences into metres (the Sim(3) scale times
    the scene's track upscaling factor; the rotation preserves norms, so one
    scalar is exact).
    """

    if (
        displacements.ndim != 6
        or displacements.shape[0] != 1
        or displacements.shape[-1] != 3
    ):
        raise ValueError(
            "displacements must have shape (1,Q,S,H,W,3), got "
            f"{tuple(displacements.shape)}"
        )
    slot_time_indices = torch.as_tensor(slot_time_indices)
    if slot_time_indices.numel() != displacements.shape[2]:
        raise ValueError(
            f"slot_time_indices must have one entry per observation slot "
            f"({displacements.shape[2]}), got {slot_time_indices.numel()}"
        )
    if not math.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError("metric_scale must be finite and positive")

    first, second = synchronized_pair_indices(slot_time_indices.reshape(-1))
    if not first:
        raise ValueError(
            "No synchronized observation pairs share a time index; at least "
            "two observations of one instant are needed"
        )
    device = displacements.device
    first_index = torch.tensor(first, device=device, dtype=torch.long)
    second_index = torch.tensor(second, device=device, dtype=torch.long)
    return (
        displacements.index_select(2, first_index)
        - displacements.index_select(2, second_index)
    ) * float(metric_scale)


def synchronized_consistency_loss(
    displacements: torch.Tensor,
    slot_time_indices: torch.Tensor,
    *,
    huber_delta: float,
    metric_scale: float = 1.0,
) -> torch.Tensor:
    """Huber penalty on dP disagreement between synchronized observations.

    The displacement field ``dP_q^{t_q -> tau}`` depends only on the target
    *time* ``tau``, never on which camera observed it, so observation slots with
    equal time index owe identical fields; their difference is supervised toward
    zero at every pixel.  This is the dense, self-supervised counterpart of the
    sparse position term, whose visibility mask has holes exactly where
    cross-view hallucination lives.  ``metric_scale`` keeps ``huber_delta``
    commensurate with the position term's metric threshold.
    """

    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    difference = synchronized_differences(
        displacements,
        slot_time_indices,
        metric_scale=metric_scale,
    )
    zero_target = torch.zeros(
        (),
        device=difference.device,
        dtype=difference.dtype,
    ).expand_as(difference)
    loss = F.huber_loss(
        difference,
        zero_target,
        reduction="mean",
        delta=huber_delta,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Synchronized consistency loss produced NaN or Inf"
        )
    return loss


def resolve_confidence_alpha(
    mean_confidence: float,
    mean_position_error: float,
) -> float:
    """Pick ``alpha`` so the term's optimum starts at the pretrained operating point.

    The optimum of ``conf * err - alpha * log(conf)`` is ``conf* = alpha / err``, so
    ``alpha = mean_confidence * mean_error`` puts ``conf*`` exactly at the released
    checkpoint's current mean confidence.  The term then *re-orders* confidence by
    accuracy without shifting its absolute level, which matters because the
    downstream occlusion call thresholds confidence absolutely.

    Without this, the reference ``alpha = 1`` against metric residuals of order one
    metre implies ``conf* < 1``, below the ``expp1`` floor, and confidence collapses.
    """

    if not math.isfinite(mean_confidence) or mean_confidence <= 0:
        raise ValueError(
            f"mean_confidence must be finite and positive, got {mean_confidence}"
        )
    if not math.isfinite(mean_position_error) or mean_position_error <= 0:
        raise ValueError(
            f"mean_position_error must be finite and positive, got {mean_position_error}"
        )
    alpha = float(mean_confidence) * float(mean_position_error)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"Resolved alpha is not finite and positive: {alpha}")
    return alpha


def compose_tracking_loss(
    terms: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of named loss terms, skipping ablated ones entirely.

    A term whose weight is ``0.0`` is not multiplied by zero, it is left out of the
    sum, so an ablated term adds no graph node and cannot perturb the remaining
    terms.  A single term at weight ``1.0`` is returned as-is, which is what keeps a
    confidence-disabled run bit-identical to the position-only path.

    Returns the total and a breakdown of each active term's **unweighted** value, so
    runs at different weights stay comparable.  The breakdown holds detached scalar
    *tensors*, not floats: this runs every training step, and a tensor-math function
    has no business forcing a device sync for a figure only the reporting layer
    reads.  Call ``.item()`` at the point of use.
    """

    unknown = set(weights) - set(terms)
    if unknown:
        raise KeyError(f"Weights given for unknown loss terms: {sorted(unknown)}")

    total: torch.Tensor | None = None
    breakdown: dict[str, torch.Tensor] = {}
    for name in terms:
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Weight for '{name}' must be finite and non-negative")
        if weight == 0.0:
            continue
        term = terms[name]
        if term.ndim != 0:
            raise ValueError(f"Loss term '{name}' must be a scalar, got {tuple(term.shape)}")
        breakdown[name] = term.detach()
        contribution = term if weight == 1.0 else weight * term
        total = contribution if total is None else total + contribution

    if total is None:
        raise ValueError("compose_tracking_loss needs at least one term with a nonzero weight")
    if not torch.isfinite(total):
        raise FloatingPointError("Composed tracking loss produced NaN or Inf")
    return total, breakdown


def _validate_position_inputs(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    huber_delta: float,
) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must have equal shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[-1] != 3:
        raise ValueError(
            f"predicted must have a trailing 3-vector axis, got {tuple(predicted.shape)}"
        )
    if mask.shape != predicted.shape[:-1]:
        raise ValueError(
            f"mask must have shape {tuple(predicted.shape[:-1])}, got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a boolean tensor, got {mask.dtype}")
    if not mask.any():
        raise ValueError("No samples selected for the track position loss")
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
