"""Runtime helpers shared by every bounded 4RC training entry point.

These grew inside ``overfit_temporal_tracking.py`` when it was the only driver.
A second driver would otherwise copy them, and a copy is exactly how the guards
here stop being guards: the freeze-mask assertion, the frozen-gradient check and
the finite-gradient check exist to make a silent no-op loud, and two divergent
copies make them quiet again.

Nothing here knows about ``argparse``.  ``build_optimizer`` takes scalars rather
than a ``Namespace`` precisely so a second driver with a different parser can
call it without inheriting the first driver's flag names.

Deliberately **not** re-exported from ``arc.training``'s ``__all__``:
``gradient_norm`` and ``move_views_to_cuda`` are device-specific, and the
package's public surface is device-neutral.  Import them from
``arc.training.runtime`` explicitly.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


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


def confidence_stats(raw_predictions) -> dict[str, float] | None:
    """Summarize the track head's confidence channel.

    Nothing supervises it unless the confidence term is enabled: the position
    loss is a Huber on xyz, and xyz and confidence are split off the *same* final
    conv, so the channel gets a zero gradient while its trunk moves.
    ``score_joint.py`` thresholds this confidence absolutely to derive occlusion,
    which feeds OA and AJ, so an unsupervised mean shift is not harmless.  Log it
    so drift is attributable.

    Note ``torch.quantile`` refuses inputs above 2**24 elements.  At the one-scene
    harness's sizes that is three orders of magnitude away; a driver running wider
    windows or more anchors has to guard it rather than assume it.
    """

    confidence = raw_predictions.get("conf_track_multi")
    if confidence is None:
        return None
    values = confidence.detach().float().flatten()
    if values.numel() == 0:
        return None
    quantiles = torch.tensor([0.05, 0.5, 0.95], device=values.device)
    p05, p50, p95 = torch.quantile(values, quantiles).tolist()
    return {
        "mean": float(values.mean().item()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
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
