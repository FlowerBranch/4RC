"""The resumable half of a training checkpoint, and the one hazard it has.

``save_temporal_tracking_checkpoint`` writes a *patch*: the trainable parameters
and the freeze mode, which is what inference loads.  A run that a scheduler can
preempt needs more than that -- optimizer moments, the step counter, every RNG
stream and the grad scaler -- or a resume silently restarts Adam and the data
order instead of continuing them.

**Everything here is a plain Python type on purpose.** ``checkpoint.py`` reads
patches with ``torch.load(..., weights_only=True)``, and that rejects *every*
numpy object: ``np.random.get_state()``, any ``ndarray``, any numpy scalar. One
of them anywhere in the payload -- including inside a stashed ``vars(args)``,
including a ``Path`` -- does not merely lose that field, it makes the whole file
unreadable by ``load_temporal_tracking_checkpoint``, i.e. it stops being a
temporal patch at all. So numpy's state is decomposed to ``(str, list[int], int,
int, float)`` on the way in and rebuilt on the way out, and the tests assert the
patch readers still parse the superset.

Writes are atomic against a *concurrent writer*, not just a crash: the temp name
carries pid and a uuid, and the bytes are fsynced before the rename. A requeued
SLURM job can start on a second node while the original is still inside its grace
window writing a multi-GB file to shared storage, and a fixed ``.tmp`` name lets
those two interleave into one published file that ``torch.load`` either rejects or,
worse, accepts as a truncated prefix.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import torch


# Bumped when a field changes meaning. A resume that reads an older payload should
# say so plainly rather than half-restoring it.
TRAINER_STATE_VERSION = 1


def _encode_numpy_random_state(state) -> dict:
    """``np.random.get_state()`` as plain types.

    The tuple is ``(name, keys, pos, has_gauss, cached_gaussian)`` where ``keys``
    is a ``uint32`` ndarray -- the one field that would poison ``weights_only``.
    """

    name, keys, pos, has_gauss, cached = state
    return {
        "name": str(name),
        "keys": [int(value) for value in keys],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _decode_numpy_random_state(encoded: dict):
    return (
        encoded["name"],
        np.array(encoded["keys"], dtype=np.uint32),
        int(encoded["pos"]),
        int(encoded["has_gauss"]),
        float(encoded["cached_gaussian"]),
    )


def capture_rng_state() -> dict:
    """Every stream a training step can consume, in `weights_only`-safe form."""

    import random

    state = {
        "python": list(random.getstate()[1]),
        "python_version": random.getstate()[0],
        "python_gauss": random.getstate()[2],
        "torch": torch.get_rng_state(),
        "numpy": _encode_numpy_random_state(np.random.get_state()),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = list(torch.cuda.get_rng_state_all())
    return state


def restore_rng_state(state: dict) -> None:
    import random

    random.setstate(
        (
            state["python_version"],
            tuple(state["python"]),
            state["python_gauss"],
        )
    )
    torch.set_rng_state(state["torch"])
    np.random.set_state(_decode_numpy_random_state(state["numpy"]))
    cuda = state.get("torch_cuda")
    if cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(cuda))


def build_trainer_state(
    *,
    step: int,
    optimizer,
    base_learning_rates,
    scaler=None,
    settings: dict | None = None,
) -> dict:
    """The resumable fields, as plain types beside the patch's own keys."""

    return {
        "trainer_state_version": TRAINER_STATE_VERSION,
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "base_learning_rates": [float(value) for value in base_learning_rates],
        # None rather than {} when disabled: an empty dict is also what a scaler
        # whose CUDA context failed to initialise returns, and the two mean very
        # different things.
        "scaler": None if scaler is None or not scaler.is_enabled() else scaler.state_dict(),
        "rng": capture_rng_state(),
        "settings": dict(settings or {}),
    }


def save_atomically(payload: dict, path: str | Path) -> Path:
    """Write, fsync, then rename -- safe against a concurrent second writer.

    The temp name is unique per process and per call. A shared ``--output_dir``
    across a requeue overlap is the normal case, not an exotic one, and a fixed
    ``.tmp`` name there interleaves two writes into one file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            # The rename is atomic; the *write* is not, and a node that dies
            # between them would otherwise publish a name pointing at nothing.
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def read_trainer_state(path: str | Path) -> dict:
    """Load a payload written by :func:`save_atomically`, checking its version."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "trainer_state_version" not in payload:
        raise RuntimeError(
            f"{path} is not a trainer checkpoint; it carries no "
            "trainer_state_version. A bare temporal patch has no optimizer or "
            "step counter, so resuming from one would restart training while "
            "reporting that it continued."
        )
    version = payload["trainer_state_version"]
    if version != TRAINER_STATE_VERSION:
        raise RuntimeError(
            f"{path} was written by trainer state version {version}, this build "
            f"reads {TRAINER_STATE_VERSION}"
        )
    return payload
