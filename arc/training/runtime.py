"""Runtime helpers shared by every bounded 4RC training entry point.

These grew inside ``overfit_temporal_tracking.py`` when it was the only driver.
A second driver would otherwise copy them, and a copy is exactly how the guards
here stop being guards: the freeze-mask assertion, the frozen-gradient check and
the finite-gradient check exist to make a silent no-op loud, and two divergent
copies make them quiet again.

Nothing here knows about ``argparse``.  ``build_optimizer`` takes scalars rather
than a ``Namespace`` precisely so a second driver with a different parser can
call it without inheriting the first driver's flag names.

The per-anchor supervision mechanism (``cut_features`` through
``accumulate_weighted``) lives here for the same reason: the overfit built and
measured it, the multi-scene trainer runs the identical structure, and a copy
would let the two drivers' memory behaviour drift apart silently.

Deliberately **not** re-exported from ``arc.training``'s ``__all__``:
``gradient_norm`` and ``move_views_to_cuda`` are device-specific, and the
package's public surface is device-neutral.  Import them from
``arc.training.runtime`` explicitly.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from arc.training.losses import compose_tracking_loss
from arc.training.sparse_tracking import sparse_targets


# Per freeze mode: (trainable tensor count, trainable parameters excluding the
# time-index embedding, whose row count is a flag). Measured on a meta-device
# Arc. A refactor that silently changes a freeze mask must fail here rather
# than quietly costing GPU weeks.
EXPECTED_TRAINABLE_SETS = {
    "temporal_tracking": (231, 314_551_588),
    "temporal_tracking_global_attention": (483, 711_268_132),
    # Pinned at DEFAULT_LATE_GLOBAL_BLOCKS; every other k derives from it via
    # LATE_GLOBAL_PER_BLOCK in expected_trainable_set.
    "temporal_tracking_late_global": (303, 427_899_172),
}
# One global-attention block, measured on a meta-device Arc. The 14 are exactly
# homogeneous, so k of them cost exactly k times this.
LATE_GLOBAL_PER_BLOCK = (18, 28_336_896)
DEFAULT_LATE_GLOBAL_BLOCKS = 4
# Every global-attention block the vitg encoder has (alt_start=13, depth=40).
# At this k the late mask equals temporal_tracking_global_attention exactly --
# asserted in the tests, since argument validation runs before a model exists.
MAX_LATE_GLOBAL_BLOCKS = 14
TIME_EMBEDDING_DIM = 1536
TIME_EMBEDDING_KEY = "backbone.pretrained.time_index_embedding.weight"


def expected_trainable_set(freeze_mode, late_global_blocks):
    """(tensor count, non-embedding parameter count) the mode must produce.

    The late-global entry is pinned at k = DEFAULT_LATE_GLOBAL_BLOCKS and other
    k derive from it, so the startup assertion stays live for every k instead
    of being skipped whenever k is not the default.
    """

    tensors, parameters = EXPECTED_TRAINABLE_SETS[freeze_mode]
    if freeze_mode != "temporal_tracking_late_global":
        return tensors, parameters
    per_block_tensors, per_block_parameters = LATE_GLOBAL_PER_BLOCK
    delta = late_global_blocks - DEFAULT_LATE_GLOBAL_BLOCKS
    return (
        tensors + per_block_tensors * delta,
        parameters + per_block_parameters * delta,
    )


def assert_trainable_parameter_set(
    model,
    *,
    freeze_mode: str,
    max_time_indices: int,
    late_global_blocks: int | None = None,
) -> dict:
    """Fail if the freeze mask is not exactly the set the mode promises.

    This is the guard the whole module exists for: a refactor that silently
    changes which parameters train must stop the run here rather than quietly
    cost GPU weeks.  Both drivers call it, so there is one expectation and one
    message -- a second copy beside a caller is how the two drift until only one
    of them is still checking anything.

    Returns the trainable-parameter report so a caller can log it without asking
    the model twice.
    """

    report = model.get_trainable_parameter_report()
    expected_tensors, expected_non_embedding = expected_trainable_set(
        freeze_mode,
        late_global_blocks,
    )
    expected_parameter_count = (
        expected_non_embedding + max_time_indices * TIME_EMBEDDING_DIM
    )
    note = "" if late_global_blocks is None else f", k={late_global_blocks}"
    if (
        report["tensor_count"] != expected_tensors
        or report["parameter_count"] != expected_parameter_count
    ):
        raise RuntimeError(
            f"Unexpected {freeze_mode}{note} parameter set: "
            f"{report['tensor_count']} tensors / "
            f"{report['parameter_count']} parameters; expected "
            f"{expected_tensors} / {expected_parameter_count}"
        )
    return report


def autocast_context(precision: str):
    if precision == "32":
        return nullcontext()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def gradient_norm(parameters) -> float:
    """L2 norm over every parameter that carries a gradient.

    The accumulator follows the first gradient's device rather than naming one.
    On CUDA that is bit-identical to seeding it with a zero scalar -- ``0 + x``
    is exact in float32, and the summation order is unchanged -- and it is what
    lets a CPU-only test drive a training step at all, which the guards below
    are worth nothing without.
    """

    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().square().sum()
        squared_norm = (
            contribution if squared_norm is None else squared_norm + contribution
        )
    return 0.0 if squared_norm is None else float(torch.sqrt(squared_norm).item())


def assert_frozen_gradients_absent(model) -> None:
    offenders = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(
            "Frozen parameters received gradients: " + ", ".join(offenders[:10])
        )


def assert_trainable_gradients_finite(model) -> None:
    missing = []
    non_finite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            non_finite.append(name)
    if missing or non_finite:
        details = []
        if missing:
            details.append(f"missing gradients: {missing[:10]}")
        if non_finite:
            details.append(f"non-finite gradients: {non_finite[:10]}")
        raise RuntimeError("Invalid trainable gradients; " + "; ".join(details))


def move_views_to_cuda(views: list[dict]) -> None:
    for view in views:
        for key in ("img", "time_index", "track_query_idx"):
            view[key] = view[key].to("cuda", non_blocking=True)


def shuffled_index_views(scene) -> list[dict] | None:
    """Views with every non-primary camera's time indices reversed.

    Reversal keeps every index a valid, trained embedding row (in-distribution)
    while breaking cross-camera synchronization, which is exactly the structure
    the temporal indices exist to encode.  Each view dict is shallow-copied so
    the scene's own views keep their true indices.  Returns ``None`` when there
    is nothing to break: a single camera, or a single time (whose reversal is
    the identity).
    """

    time_count = len(scene.times)
    if len(scene.cameras) < 2 or time_count < 2:
        return None
    primary_camera = scene.cameras[0]
    views = []
    for observation, view in zip(scene.observations, scene.views):
        copy = dict(view)
        if observation.camera != primary_camera:
            copy["time_index"] = torch.tensor(
                [time_count - 1 - observation.semantic_time_index],
                dtype=torch.long,
                device=view["time_index"].device,
            )
        views.append(copy)
    return views


def build_optimizer(
    model,
    *,
    lr: float,
    embedding_lr: float | None = None,
    encoder_lr: float | None = None,
) -> tuple[torch.optim.AdamW, dict[str, float], list]:
    """AdamW over named parameter groups with per-group learning rates.

    The embedding gets its own group so its rate is a free knob, and unfrozen
    encoder blocks get a lower default because the frozen geometry heads read
    the features those blocks produce.  Weight decay stays 0 everywhere: no
    regularizer is part of this proof, and decoupled decay must not drag the
    unsupervised confidence output around.

    Mode-agnostic by construction: the groups are selected by ``requires_grad``
    and the embedding's name, never by the freeze mode, so a preset that unfreezes
    a different block set needs no change here.  A test drives all three presets
    against this rather than leaving it as a claim.
    """

    embedding_parameters = [
        parameter
        for parameter in model.backbone.pretrained.time_index_embedding.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for module in (model.motion_decoder, model.track_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    encoder_parameters = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "time_index_embedding" not in name
    ]
    learning_rates = {
        "decoder": lr,
        "embedding": lr if embedding_lr is None else embedding_lr,
        "encoder_blocks": (lr * 0.1 if encoder_lr is None else encoder_lr),
    }
    groups = [
        {"params": head_parameters, "lr": learning_rates["decoder"]},
        {"params": embedding_parameters, "lr": learning_rates["embedding"]},
    ]
    if encoder_parameters:
        groups.append(
            {"params": encoder_parameters, "lr": learning_rates["encoder_blocks"]}
        )
    else:
        learning_rates["encoder_blocks"] = None

    grouped = sum(parameter.numel() for group in groups for parameter in group["params"])
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if grouped != trainable:
        raise RuntimeError(
            f"Parameter groups cover {grouped} parameters but {trainable} are "
            "trainable; a trainable parameter escaped every group"
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    return optimizer, learning_rates, encoder_parameters


def tracking_only(raw_predictions: dict, keep_confidence: bool = False) -> dict:
    """Drop unused reconstruction branches promptly to reduce retained memory.

    ``conf_track_multi`` is kept only when the confidence term needs it, so a
    position-only run retains exactly what it retained before.
    """

    kept = {
        "track_multi": raw_predictions["track_multi"],
        "track_query_idx": raw_predictions["track_query_idx"],
    }
    if keep_confidence:
        kept["conf_track_multi"] = raw_predictions["conf_track_multi"]
    return kept


# --- The per-anchor memory mechanism, shared by both drivers. Several anchors
# are supervised as one encoder pass plus one track-head pass per anchor, each
# backwarded onto a detached cut of the backbone taps; the summed cut gradients
# then flow through the encoder exactly once. The overfit measured the marginal
# cost of an extra anchor at a flat ~2.3 GiB against a 135 GiB primary arm at
# the 48-observation window -- which is what a widened Q axis cannot deliver.


def cut_features(feats):
    """Detach the backbone taps into leaves, returning the cut and its pairs.

    This is the encoder/head boundary that lets several anchors be supervised
    without holding several track-head graphs at once.  Each anchor's head pass
    runs on the leaves and backwards immediately, freeing its own graph and
    leaving its gradient on the cut; one
    ``torch.autograd.backward(originals, leaf_grads)`` afterwards pushes the
    summed gradient through the encoder exactly once.  Summing gradients at a
    cut is the chain rule, so this is identical to one combined backward -- not
    an approximation -- while the encoder is neither re-run nor re-differentiated
    per anchor.
    """

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def cut(value):
        if torch.is_tensor(value):
            if not value.requires_grad:
                return value
            leaf = value.detach().requires_grad_(True)
            pairs.append((value, leaf))
            return leaf
        if isinstance(value, (list, tuple)):
            return type(value)(cut(item) for item in value)
        return value

    return cut(feats), pairs


def backward_through_cut(pairs) -> None:
    """Push the anchors' accumulated gradients back through the encoder once."""

    if not pairs:
        return
    originals = [original for original, _ in pairs]
    gradients = [
        torch.zeros_like(leaf) if leaf.grad is None else leaf.grad
        for _, leaf in pairs
    ]
    torch.autograd.backward(originals, gradients)


def encode_and_reconstruct(model, views):
    """The per-step work that does not depend on which frame is the query."""

    images, _, time_indices = model._preprocess_input(views)
    feats = model.encode_features(images, time_indices=time_indices)
    return images, feats, model.reconstruct(feats, images)


def anchor_tracks(model, feats, images, scene, anchor_index):
    """One anchor's dense field, shaped as the Q=1 raw dict the loss expects."""

    slot = scene.anchor_observation_slots[anchor_index]
    track, track_conf = model.track_for_query(feats, images, slot)
    return {
        "track_multi": track[:, None],
        "conf_track_multi": track_conf[:, None],
        "track_query_idx": torch.tensor([slot], device=track.device),
    }


def anchor_sample_counts(scene, correspondences, anchor_count: int) -> list[int]:
    """Supervised sample count per anchor, from the scene alone.

    These are the weights that make per-anchor backward equal one combined
    ``reduction="mean"``.  They come from ``sparse_targets``, the same masking
    the loss itself applies, so the two cannot drift apart; and because nothing
    in that mask reads a prediction, they are fixed for the whole run.
    """

    counts = []
    for anchor_index in range(anchor_count):
        anchor = correspondences.select_query_slot(anchor_index)
        if anchor.count == 0:
            counts.append(0)
            continue
        _, _, _, mask = sparse_targets(scene, anchor)
        counts.append(int(mask.sum().item()))
    return counts


def weighted_anchor_total(
    result,
    *,
    position_weight: float,
    confidence_weight: float,
    sync_weight: float,
) -> torch.Tensor:
    """One anchor's contribution to the combined objective.

    ``compose_tracking_loss`` is linear in its terms, so backwarding each
    anchor's weighted total in turn accumulates exactly the gradient of the sum
    -- which is the combined loss, given the position and confidence weights are
    that anchor's share of the supervised samples and the sync weight is its
    share of the anchors.

    One scope limit on that equivalence.  The sync share is
    ``1 / active_anchor_count``, not ``1 / anchor_count``, so the objective
    reproduced is a stacked forward over the **active** anchors.  A declared
    anchor that supervises nothing is skipped entirely, and against the
    ``Q = anchor_count`` reference ``_evaluate`` computes, its dense field would
    have contributed a sync term this does not include.  Weighting by the active
    count is the defensible choice -- an anchor with no supervised query still
    has a displacement field, but nothing here has any evidence about it -- but
    it does mean the two numbers differ in that case.

    Terms are inserted in the same order ``sparse_tracking_loss`` uses, because
    ``compose_tracking_loss`` sums in insertion order and float addition is not
    associative: with a single anchor at weight 1.0 this must reproduce that
    function's own ``total_loss`` bit for bit.
    """

    terms = {"position": result.loss}
    weights = {"position": position_weight}
    if result.sync_loss is not None:
        terms["sync"] = result.sync_loss
        weights["sync"] = sync_weight
    if result.confidence_loss is not None:
        terms["confidence"] = result.confidence_loss
        weights["confidence"] = confidence_weight
    total, _ = compose_tracking_loss(terms, weights)
    return total


def accumulate_weighted(total: float | None, value, weight: float) -> float | None:
    """Running weighted sum of a reported scalar across anchors."""

    if value is None:
        return total
    contribution = float(value.detach().item()) * float(weight)
    return contribution if total is None else total + contribution


def _quantile_summary(values: torch.Tensor) -> dict[str, float]:
    """Mean and the 5/50/95 percentiles of one flat tensor."""

    quantiles = torch.tensor([0.05, 0.5, 0.95], device=values.device)
    p05, p50, p95 = torch.quantile(values, quantiles).tolist()
    return {
        "mean": float(values.mean().item()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


def confidence_stats(raw_predictions) -> dict | None:
    """Summarize the track head's confidence channel, per anchor.

    Nothing supervises it unless the confidence term is enabled: the position
    loss is a Huber on xyz, and xyz and confidence are split off the *same* final
    conv, so the channel gets a zero gradient while its trunk moves.
    ``score_joint.py`` thresholds this confidence **absolutely** to derive
    occlusion, which feeds OA and AJ, so an unsupervised mean shift is not
    harmless.  Log it so drift is attributable.

    **Why per anchor, and not pooled.**  That same scorer matches each query in
    whichever anchor fits it best, so whether anchor 1's confidence distribution
    sits where anchor 0's does is exactly the thing worth knowing.  Pooling the
    anchors averages that away.  It also removes a hard ceiling: ``torch.quantile``
    refuses inputs above ``2**24`` elements, and a pooled ``(1,Q,S,H,W)`` at the
    committed 48 observations crosses it at the *second* anchor (Q*S <= 88 at
    378x504).  A per-anchor slice is ``S*H*W`` **independent of Q**, so the cap
    binds only if ``S`` alone passed 88 observations -- well beyond what the card
    fits.  The diagnostic therefore stops capping the science at one anchor.

    At ``Q == 1`` the returned dict is exactly what the pooled version returned:
    the same four keys over the identical elements in the identical order.
    Per-anchor detail appears only when there is more than one anchor, which
    keeps ``run_summary.json`` unchanged for every single-anchor run.

    The pooled ``mean`` stays exact above ``Q == 1`` because the slices are equal
    sized.  Pooled *quantiles* are **not** recoverable from per-anchor quantiles
    and are reported as ``None`` rather than fabricated; ``torch.sort`` has no cap
    and reproduces this function's linear-interpolation convention exactly, if one
    is ever genuinely wanted.
    """

    confidence = raw_predictions.get("conf_track_multi")
    if confidence is None:
        return None
    values = confidence.detach().float()
    if values.numel() == 0:
        return None
    # The anchor axis is read off the tensor rather than assumed, so a head that
    # ever changes layout fails here instead of silently summarizing the wrong
    # axis. Every producer in this repo emits (1,Q,S,H,W).
    if values.ndim != 5 or values.shape[0] != 1:
        raise ValueError(
            "conf_track_multi must have shape (1,Q,S,H,W) for the anchor axis to "
            f"be unambiguous, got {tuple(values.shape)}"
        )

    per_anchor = [
        _quantile_summary(values[0, anchor].reshape(-1))
        for anchor in range(int(values.shape[1]))
    ]
    if len(per_anchor) == 1:
        return per_anchor[0]
    return {
        "mean": float(values.mean().item()),
        "p05": None,
        "p50": None,
        "p95": None,
        "per_anchor": per_anchor,
    }


def confidence_gradient_norms(model) -> dict[str, float]:
    """Attribute the final track conv's gradient to its confidence and xyz rows.

    xyz and confidence come off the same ``Conv2d(_, 4, 1)``: rows 0-2 are the
    position term's contribution and row 3 is the confidence term's.  Because the
    confidence term detaches the error, that split is exact -- no second backward
    pass is needed to attribute it.
    """

    output_conv = model.track_head.scratch.output_conv2[2]
    # The split is only meaningful for the 4-channel xyz+conf head. Fail loudly if
    # the head is ever rebuilt with a different output_dim rather than silently
    # reporting a norm over the wrong rows.
    if output_conv.out_channels != 4:
        raise RuntimeError(
            "Expected a 4-channel track output conv (3 xyz + 1 confidence), got "
            f"{output_conv.out_channels}"
        )
    norms = {}
    for label, rows in (
        ("track_head_output_conv_position_rows", slice(0, 3)),
        ("track_head_output_conv_confidence_row", slice(3, 4)),
    ):
        total = 0.0
        for parameter in (output_conv.weight, output_conv.bias):
            if parameter is None or parameter.grad is None:
                continue
            total += float(
                parameter.grad[rows].detach().float().norm().item() ** 2
            )
        norms[label] = float(total**0.5)
    return norms
