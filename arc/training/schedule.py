"""Linear warmup into cosine decay, as a multiplier on each group's base rate.

Deliberately not a ``torch.optim.lr_scheduler``: the state that has to survive a
resume is one integer, and a scheduler object would put a second copy of it in
the checkpoint to disagree with the trainer's own step counter.  The rate at step
``s`` is a pure function of ``s`` here, so a resumed run recomputes it rather than
restoring it, and there is nothing to drift.

**The `base_lrs` capture is the load-bearing part.** ``optimizer.load_state_dict``
overwrites ``param_groups[i]["lr"]`` with the *decayed* value saved in the
checkpoint.  Capturing the base rates after that call would restart the cosine
from the already-decayed rate, and compound it once per resume -- an effective
rate decaying as ``scale(N)**segments`` on a job that requeues, with nothing
raising anywhere and only the loss curve to notice.  Capture before, always; the
tests drive a resume and assert the rate at *every* step of the second segment,
because the compounding is invisible in the final value alone.
"""

from __future__ import annotations

import math


def warmup_cosine_scale(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.0,
) -> float:
    """Multiplier in ``[min_lr_scale, 1]`` for a 0-based ``step``.

    Warmup is linear over ``warmup_steps`` and reaches 1.0 at the first
    post-warmup step; cosine then decays to ``min_lr_scale`` at ``total_steps``.
    Steps past the end hold ``min_lr_scale`` rather than turning back up, which a
    naive cosine does if a run overshoots its planned length.
    """

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be below total_steps "
            f"({total_steps}); a run that never leaves warmup has no schedule"
        )
    if not 0.0 <= min_lr_scale <= 1.0:
        raise ValueError(f"min_lr_scale must be in [0,1], got {min_lr_scale}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if step < warmup_steps:
        # (step + 1) so the first step is not exactly zero: a zero rate wastes a
        # forward and a backward, and at warmup_steps=1 it would waste the only
        # warmup step there is.
        return (step + 1) / (warmup_steps + 1)
    if step >= total_steps:
        return min_lr_scale
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine


def capture_base_learning_rates(optimizer) -> list[float]:
    """Each group's rate, read *before* any checkpoint is loaded into it.

    See the module docstring: this must not be called after
    ``optimizer.load_state_dict``.
    """

    return [float(group["lr"]) for group in optimizer.param_groups]


def apply_learning_rate(optimizer, base_learning_rates, scale: float) -> list[float]:
    """Set every group to ``base * scale`` and return what was set.

    Scaling each group's own base is what keeps the per-group ratios the freeze
    presets rely on -- the encoder group runs at a tenth of the decoder's, and a
    schedule that wrote one rate across all groups would silently erase that.
    """

    if len(base_learning_rates) != len(optimizer.param_groups):
        raise ValueError(
            f"{len(base_learning_rates)} base rates for "
            f"{len(optimizer.param_groups)} parameter groups; the optimizer was "
            "rebuilt with a different group layout than the one captured"
        )
    applied = []
    for group, base in zip(optimizer.param_groups, base_learning_rates):
        group["lr"] = base * scale
        applied.append(group["lr"])
    return applied
