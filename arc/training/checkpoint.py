"""Strict patch checkpoints for the ``temporal_tracking`` freeze preset."""

from __future__ import annotations

from pathlib import Path

import torch


FORMAT_VERSION = 1


def _trainable_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_temporal_tracking_checkpoint(model, path: str | Path) -> Path:
    """Save exactly the parameters enabled by ``freeze='temporal_tracking'``."""

    if getattr(model, "freeze", None) != "temporal_tracking":
        raise ValueError(
            "Temporal-tracking patch checkpoints require "
            "model.freeze == 'temporal_tracking'"
        )
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("Model has no trainable temporal-tracking parameters")
    payload = {
        "format_version": torch.tensor(FORMAT_VERSION, dtype=torch.int64),
        "state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in parameters.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_temporal_tracking_checkpoint(model, path: str | Path) -> None:
    """Strictly overlay a saved temporal-tracking patch onto a base Arc model."""

    if getattr(model, "freeze", None) != "temporal_tracking":
        raise ValueError(
            "Set model.freeze to 'temporal_tracking' before loading its patch"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise RuntimeError("Not a temporal-tracking patch checkpoint")
    version = int(torch.as_tensor(payload["format_version"]).item())
    if version != FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported temporal-tracking checkpoint version {version}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Temporal-tracking checkpoint has no state_dict")

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
