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
        "state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in parameters.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _parse_payload(path: str | Path) -> tuple[str, dict]:
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
    return payload["freeze_mode"], payload["state_dict"]


def read_temporal_patch_metadata(path: str | Path) -> dict:
    """Freeze mode and embedding table size, without needing a model.

    ``max_time_indices`` is derived from the stored embedding tensor's row
    count rather than a separate field, so it cannot disagree with the
    weights; it is ``None`` when the patch carries no embedding.
    """

    freeze_mode, state_dict = _parse_payload(path)
    embedding = state_dict.get(_TIME_EMBEDDING_KEY)
    return {
        "freeze_mode": freeze_mode,
        "max_time_indices": (
            None
            if not isinstance(embedding, torch.Tensor)
            else int(embedding.shape[0])
        ),
    }


def load_temporal_tracking_checkpoint(model, path: str | Path) -> None:
    """Strictly overlay a saved temporal-tracking patch onto a base Arc model."""

    freeze_mode, state_dict = _parse_payload(path)
    if getattr(model, "freeze", None) != freeze_mode:
        raise ValueError(
            f"This patch was trained under freeze mode '{freeze_mode}'; call "
            f"model.set_freeze('{freeze_mode}') before loading it "
            f"(model.freeze is {getattr(model, 'freeze', None)!r})"
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
