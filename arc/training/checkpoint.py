"""Strict patch checkpoints for the temporal-tracking freeze presets."""

from __future__ import annotations

from pathlib import Path

import torch


_TIME_EMBEDDING_KEY = "backbone.pretrained.time_index_embedding.weight"
_DEPTH_EMBED_KEY = "backbone.pretrained.depth_patch_embed.weight"
_CAMERA_PROJ_KEY = "backbone.pretrained.camera_proj.weight"
# One tensor stands for the refiner's eight, as one weight stands for each
# injection above: the saver keys off requires_grad and set_freeze toggles
# the refiner as a whole, so a patch carries every refiner tensor or none.
_REFINER_KEY = "motion_decoder.refiner.read_proj.weight"


def _trainable_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _parse_payload(path: str | Path) -> tuple[str, int | None, int, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("freeze_mode"), str)
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise RuntimeError(
            "Not a temporal-tracking patch checkpoint. Patches written before "
            "the freeze_mode field also predate the motion-decoder gradient "
            "fix and are not worth loading; train a new one with "
            "train_temporal_tracking.py."
        )
    # A patch written before this field existed genuinely has no k, and None is
    # its true value: every mode that predates it is fully determined by its
    # name. Absence is therefore read, not rejected -- archived patches must
    # keep loading. The one hole that leaves is closed immediately below.
    late_global_blocks = payload.get("late_global_blocks")
    if late_global_blocks is not None and (
        isinstance(late_global_blocks, bool)
        or not isinstance(late_global_blocks, int)
    ):
        raise RuntimeError(
            "Patch late_global_blocks must be an integer or absent, got "
            f"{late_global_blocks!r}"
        )
    if payload["freeze_mode"] == "temporal_tracking_late_global" and (
        late_global_blocks is None
    ):
        raise RuntimeError(
            "Patch declares freeze mode 'temporal_tracking_late_global' but "
            "records no late_global_blocks, so the trained block set cannot be "
            "reconstructed."
        )
    # The iteration count reads the same way, with a sharper reason for its
    # default: a patch without the field trained exactly one iteration -- one
    # written before the field existed because the unrolled loop did not
    # exist either, and one from the retired one-scene driver's saver, which
    # recorded no count, because that driver never refined. So absence IS 1,
    # not unknown, and archived patches keep loading. A stored None reads as
    # absent, as late_global_blocks does.
    refine_iters = payload.get("refine_iters")
    if refine_iters is None:
        refine_iters = 1
    elif isinstance(refine_iters, bool) or not isinstance(refine_iters, int):
        raise RuntimeError(
            "Patch refine_iters must be an integer or absent, got "
            f"{refine_iters!r}"
        )
    elif refine_iters < 1:
        raise RuntimeError(
            f"Patch refine_iters must be at least 1, got {refine_iters}"
        )
    # Unlike the flags read_temporal_patch_metadata derives from the key set
    # alone, the count and the key set CAN disagree. The refiner trains
    # exactly when the run unrolled more than one iteration (the trainer
    # passes refine=args.refine_iters > 1), so a trained refiner with no
    # count would run at one iteration, where the refiner never executes,
    # and a count with no refiner would unroll K identical passes of a
    # zero-term refiner (field_embed and read_proj are zero-init): K times
    # the cost for the K=1 output, under a record claiming a refinement that
    # never trained. Checked here, where the key set is already in hand,
    # so the loader -- the trainer's --resume and every test load -- refuses
    # it as well as the metadata reader. No in-repo writer produces the
    # disagreement: the trainer, the only one, records the count beside the
    # tensors from the same args.refine_iters its freeze reads. This refuses
    # hand-edited and foreign payloads, which is how a test manufactures one.
    refine = _REFINER_KEY in payload["state_dict"]
    if refine != (refine_iters > 1):
        raise ValueError(
            f"This patch reads refine_iters={refine_iters} (an absent field "
            f"reads as 1) but {'carries' if refine else 'lacks'} the track "
            f"refiner's tensors ({_REFINER_KEY}); the two cannot both be true "
            "of one trained run, so it cannot be run as trained"
        )
    return (
        payload["freeze_mode"],
        late_global_blocks,
        refine_iters,
        payload["state_dict"],
    )


def read_temporal_patch_metadata(path: str | Path) -> dict:
    """Freeze mode, block count, embedding table size, geometry flags and the
    refinement arm, without a model.

    ``max_time_indices`` is derived from the stored embedding tensor's row
    count rather than a separate field, so it cannot disagree with the
    weights; it is ``None`` when the patch carries no embedding.
    ``late_global_blocks`` is ``None`` for every mode whose name already
    determines its parameter set. ``depth_input`` and ``camera_input`` are
    derived from the stored key set the same way: a geometry-arm patch
    carries its injection tensors because the saver keys off
    ``requires_grad``, a pre-flag or control patch simply lacks the keys and
    reads ``False``, and neither can disagree with the weights. A loader
    must pass both into ``set_freeze`` before loading, or the injection
    tensors surface as unexpected keys.

    ``refine`` is derived from the key set the same way -- the refiner's
    tensors are in the patch exactly when ``set_freeze(refine=True)`` trained
    them -- and ``refine_iters`` is the recorded iteration count, 1 when the
    field is absent: a patch without it trained one iteration, whether
    written before the field existed or by the retired one-scene driver,
    which never refined. Unlike the pairs above these two CAN disagree, since one is a
    key set and the other a stored number; ``_parse_payload`` refuses the
    disagreement with a ``ValueError`` before either caller sees the values.
    A loader passes ``refine`` into ``set_freeze`` and ``refine_iters`` into
    the forward.
    """

    freeze_mode, late_global_blocks, refine_iters, state_dict = _parse_payload(path)
    embedding = state_dict.get(_TIME_EMBEDDING_KEY)
    return {
        "freeze_mode": freeze_mode,
        "late_global_blocks": late_global_blocks,
        "max_time_indices": (
            None
            if not isinstance(embedding, torch.Tensor)
            else int(embedding.shape[0])
        ),
        "depth_input": _DEPTH_EMBED_KEY in state_dict,
        "camera_input": _CAMERA_PROJ_KEY in state_dict,
        "refine": _REFINER_KEY in state_dict,
        "refine_iters": refine_iters,
    }


def load_temporal_tracking_checkpoint(model, path: str | Path) -> None:
    """Strictly overlay a saved temporal-tracking patch onto a base Arc model."""

    # The iteration count is the caller's to act on (inference.py hands it to
    # the forward; the trainer's resume refuses a changed one before reaching
    # here). _parse_payload has already refused a count that disagrees with
    # the stored key set; the exact key match below refuses a refiner that
    # disagrees with the MODEL's freeze, as unexpected or missing tensors.
    freeze_mode, late_global_blocks, _, state_dict = _parse_payload(path)
    if getattr(model, "freeze", None) != freeze_mode:
        raise ValueError(
            f"This patch was trained under freeze mode '{freeze_mode}'; call "
            f"model.set_freeze('{freeze_mode}') before loading it "
            f"(model.freeze is {getattr(model, 'freeze', None)!r})"
        )
    # Before the key match, not after: under the late-global mode the mode name
    # matches for every k, so without this a k mismatch surfaces as a list of
    # missing block tensors rather than as the one number that is wrong.
    model_blocks = getattr(model, "late_global_blocks", None)
    if model_blocks != late_global_blocks:
        raise ValueError(
            f"This patch was trained with late_global_blocks={late_global_blocks}; "
            f"call model.set_freeze('{freeze_mode}', "
            f"late_global_blocks={late_global_blocks}) before loading it "
            f"(model.late_global_blocks is {model_blocks!r})"
        )

    parameters = _trainable_parameters(model)
    missing = set(parameters) - set(state_dict)
    unexpected = set(state_dict) - set(parameters)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        raise RuntimeError(
            "Incompatible temporal-tracking patch; " + "; ".join(details)
        )

    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state_dict[name]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"Temporal-tracking checkpoint value '{name}' is not a tensor"
                )
            if value.shape != parameter.shape:
                raise RuntimeError(
                    f"Temporal-tracking checkpoint shape mismatch for '{name}': "
                    f"{tuple(value.shape)} versus {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
