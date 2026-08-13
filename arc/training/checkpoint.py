"""Strict patch checkpoints for the temporal-tracking freeze presets."""

from __future__ import annotations

from pathlib import Path

import torch


_TIME_EMBEDDING_KEY = "backbone.pretrained.time_index_embedding.weight"


def _trainable_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_temporal_tracking_checkpoint(model, path: str | Path) -> Path:
    """Save exactly the parameters enabled by the model's temporal freeze mode.

    The saved set is keyed off ``requires_grad`` at call time, so the model's
    freeze mode must still be the one it was trained under -- re-asserting a
    narrower mode before saving would silently drop trained tensors from the
    patch. The mode is recorded so the loader can demand it back; whether a
    mode name is *valid* is ``set_freeze``'s business, the single authority on
    mode names.

    ``late_global_blocks`` is recorded for the same reason and read off the
    same object: under ``temporal_tracking_late_global`` the mode name alone
    does not determine the parameter set, so without it a k=4 patch would pass
    the loader's mode check against a k=8 model and fail as a wall of missing
    keys instead of as one sentence about k.
    """

    freeze_mode = getattr(model, "freeze", None)
    if not freeze_mode or freeze_mode == "none":
        raise ValueError(
            "Patch checkpoints need a temporal freeze mode; call "
            f"model.set_freeze(...) first (model.freeze is {freeze_mode!r})"
        )
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("Model has no trainable temporal-tracking parameters")
    payload = {
        "freeze_mode": freeze_mode,
        "late_global_blocks": getattr(model, "late_global_blocks", None),
        "state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in parameters.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _parse_payload(path: str | Path) -> tuple[str, int | None, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("freeze_mode"), str)
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise RuntimeError(
            "Not a temporal-tracking patch checkpoint. Patches written before "
            "the freeze_mode field also predate the motion-decoder gradient "
            "fix and are not worth loading; re-run the overfit to produce a "
            "new one."
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
            "reconstructed; re-run the overfit to produce a new one."
        )
    return payload["freeze_mode"], late_global_blocks, payload["state_dict"]


def read_temporal_patch_metadata(path: str | Path) -> dict:
    """Freeze mode, block count and embedding table size, without a model.

    ``max_time_indices`` is derived from the stored embedding tensor's row
    count rather than a separate field, so it cannot disagree with the
    weights; it is ``None`` when the patch carries no embedding.
    ``late_global_blocks`` is ``None`` for every mode whose name already
    determines its parameter set.
    """

    freeze_mode, late_global_blocks, state_dict = _parse_payload(path)
    embedding = state_dict.get(_TIME_EMBEDDING_KEY)
    return {
        "freeze_mode": freeze_mode,
        "late_global_blocks": late_global_blocks,
        "max_time_indices": (
            None
            if not isinstance(embedding, torch.Tensor)
            else int(embedding.shape[0])
        ),
    }


def load_temporal_tracking_checkpoint(model, path: str | Path) -> None:
    """Strictly overlay a saved temporal-tracking patch onto a base Arc model."""

    freeze_mode, late_global_blocks, state_dict = _parse_payload(path)
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
