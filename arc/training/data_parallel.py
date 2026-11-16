"""Data parallelism for the temporal-tracking trainer, without a DDP wrapper.

``WORLD_SIZE > 1`` trains one process per GPU, each on its own shard of one
sampled stream, and averages the gradients at every accumulation window's close.
``DistributedDataParallel`` cannot do that here: training never calls
``model.forward()`` (``arc.training.runtime.encode_views`` and ``train_step``
drive the model's pieces themselves), and one micro-step runs up to
``1 + anchors * K + 1`` backward calls -- the point term, one per anchor and
refinement pass, then ``backward_through_cut`` -- so the track head's parameters
receive gradient ``anchors * K`` times before the encoder's receive theirs once.
DDP's reducer expects one forward and one backward with every parameter ready
once.  The sync is therefore a plain bucketed all-reduce of ``.grad`` at the
window close, and ``train_temporal_tracking.close_window_across_ranks`` is the
one caller that composes it with the window's own guards.

This module is the collective layer only: reading the launch environment,
joining the NCCL group, the stop-flag consensus, the window totals, the gradient
average, the run-end gather and the replica digest.  It knows nothing about
windows, plans or files.

**Every collective here must be entered by every rank, the same number of times
and in the same order.**  A call only some ranks reach does not fail -- it waits
for the others until the process group's timeout.  Nothing in this module can
enforce that; the trainer's call sites are chosen for it, and each says why it
is reached by every rank.

Deliberately not re-exported from ``arc.training``: like ``runtime``'s
``gradient_norm``, it is device-specific, and the package's public surface is
device-neutral.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from arc.croco.utils.misc import get_world_size

# What every task must carry beyond WORLD_SIZE, and where the sbatch takes each
# from: srun sets SLURM_PROCID and SLURM_LOCALID per task, not RANK/LOCAL_RANK,
# so the export is the sbatch's job and a missing one is named, not guessed.
LAUNCH_VARIABLES = {
    "RANK": "$SLURM_PROCID",
    "LOCAL_RANK": "$SLURM_LOCALID",
    "MASTER_ADDR": "the job's first host",
    "MASTER_PORT": "a free port on that host",
}
# How long a rank may wait inside one collective. The held-out eval, its visual
# dump and the checkpoint write run on rank 0 alone while every other rank waits
# at the next collective, and NCCL's own default (10 minutes) is shorter than
# one held-out eval can take.
DEFAULT_COLLECTIVE_TIMEOUT_MINUTES = 120
# The flat buffer one gradient all-reduce copies through. Bounded rather than
# one full-size flat copy: the committed window sits at ~94% of the device, and
# the trainable set reaches 765M parameters (3 GiB of fp32 gradient) under the
# geometry preset. 64 MiB is noise against that ceiling, and at ~50 calls per
# window the launch latency is noise against a ~19 s step.
GRADIENT_BUCKET_BYTES = 64 * 2**20
# Integer dtypes of each float width, for the bit-pattern digest.
_BIT_PATTERN_DTYPES = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def _decimal(value: str, name: str) -> int:
    """A plain ASCII decimal, refused otherwise -- int() would admit '+1', ' 1 '
    and '1_0', each a launch the sbatch never meant."""

    text = str(value)
    if not (text.isascii() and text.isdigit()):
        raise ValueError(f"{name} must be a non-negative decimal integer, got {value!r}")
    return int(text)


def world_size_from_environment(environ) -> int:
    """``WORLD_SIZE``, or 1 when it is absent -- today's single-process run."""

    value = environ.get("WORLD_SIZE")
    if value is None:
        return 1
    world_size = _decimal(value, "WORLD_SIZE")
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be at least 1, got {value!r}")
    return world_size


@dataclass(frozen=True)
class Launch:
    """This task's place in a ``WORLD_SIZE > 1`` launch."""

    rank: int
    local_rank: int
    world_size: int


def read_launch_environment(environ, *, world_size: int) -> Launch:
    """The per-task launch variables, checked before any GPU is touched.

    ``init_process_group(init_method="env://")`` reads RANK, MASTER_ADDR and
    MASTER_PORT itself, but its refusal would come from inside torch after the
    device was already chosen; LOCAL_RANK is the trainer's own and nothing else
    would check it at all.
    """

    missing = [name for name in LAUNCH_VARIABLES if not environ.get(name)]
    if missing:
        sources = ", ".join(f"{name} ({LAUNCH_VARIABLES[name]})" for name in missing)
        raise ValueError(
            f"WORLD_SIZE {world_size} trains one process per GPU, and every task "
            f"needs {', '.join(missing)} exported; the sbatch sets them from {sources}"
        )
    rank = _decimal(environ["RANK"], "RANK")
    local_rank = _decimal(environ["LOCAL_RANK"], "LOCAL_RANK")
    port = _decimal(environ["MASTER_PORT"], "MASTER_PORT")
    if rank >= world_size:
        raise ValueError(f"RANK {rank} is outside WORLD_SIZE {world_size}")
    if not 1 <= port <= 65535:
        raise ValueError(f"MASTER_PORT must be in [1, 65535], got {port}")
    return Launch(rank=rank, local_rank=local_rank, world_size=world_size)


def join_process_group(launch: Launch, *, timeout_minutes: int) -> None:
    """Bind this task to its GPU, then join the NCCL group.

    The device first: everything after this -- the model's ``.to("cuda")``, the
    scene views, the collectives' own tensors -- lands on the current device, so
    a task that joined before choosing one would put its model on GPU 0
    alongside rank 0's.  Each task must SEE every GPU of its node, because
    LOCAL_RANK indexes them; a launch that binds one GPU per task leaves every
    task a single device 0, which is refused here by name rather than surfacing
    as an invalid-device error.
    """

    if not torch.cuda.is_available():
        raise ValueError(
            f"WORLD_SIZE {launch.world_size} trains one process per GPU over NCCL, "
            "but this task sees no CUDA device"
        )
    visible = torch.cuda.device_count()
    if launch.local_rank >= visible:
        raise ValueError(
            f"LOCAL_RANK {launch.local_rank} names a GPU this task cannot see: it "
            f"sees {visible}. Every task must see every GPU of its node -- launch "
            "with --gpus-per-node equal to --ntasks-per-node and without "
            "--gpus-per-task or --gpu-bind"
        )
    torch.cuda.set_device(launch.local_rank)
    dist.init_process_group(
        "nccl", init_method="env://", timeout=timedelta(minutes=timeout_minutes)
    )


def collective_device() -> torch.device:
    """Where a collective's own small tensors live: NCCL reduces on this rank's
    GPU, gloo (the CPU tests) on the host."""

    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def agree_on_stop(signal_name: str | None) -> str | None:
    """The stop every rank takes: any rank's pending signal, the same on all.

    Slurm may deliver the signal to some tasks and not others, and a rank that
    broke alone would leave the rest waiting at their next collective. The MAX
    of the signal numbers is one name every rank agrees on -- the same reduction
    on the same inputs -- so every rank breaks at the same window top under the
    same ``interrupted_by``.
    """

    code = 0 if signal_name is None else int(signal.Signals[signal_name])
    flag = torch.tensor([code], dtype=torch.int64, device=collective_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    agreed = int(flag.item())
    return None if agreed == 0 else signal.Signals(agreed).name


@dataclass(frozen=True)
class WindowTotals:
    """One window's executed work, summed over every rank."""

    executed_ranks: int
    executed_steps: int
    loss_sum: float


def reduce_window_totals(*, executed_steps: int, loss_sum: float) -> WindowTotals:
    """One SUM over [executed at all, loss sum, executed micro-steps].

    Float64 so a loss sum stays exact enough to report; the two counts are
    small integers, exact in any float. The rank count is what the gradient
    average divides by -- not the world size: a rank that executed nothing in
    the window must not dilute the mean.
    """

    totals = torch.tensor(
        [1.0 if executed_steps else 0.0, float(loss_sum), float(executed_steps)],
        dtype=torch.float64,
        device=collective_device(),
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    ranks, loss, steps = totals.tolist()
    return WindowTotals(
        executed_ranks=int(round(ranks)),
        executed_steps=int(round(steps)),
        loss_sum=float(loss),
    )


def average_gradients(
    parameters,
    *,
    contributed: bool,
    executed_ranks: int,
    bucket_bytes: int = GRADIENT_BUCKET_BYTES,
) -> None:
    """Replace every ``.grad`` with its mean over the ranks that executed.

    A rank that did not contribute zero-fills first, allocating where ``.grad``
    is None: its gradients otherwise still hold the PREVIOUS window's average,
    since only a window's first executed micro-step zeroes them, and summing
    that in would count a stale window twice. Then a SUM, in buckets, divided by
    ``executed_ranks`` -- not ``ReduceOp.AVG``, which divides by the world size
    and so counts the ranks that skipped.

    The buckets are flat copies of at most ``bucket_bytes``, grouped by dtype and
    device; a tensor larger than a bucket, or one whose layout cannot be reduced
    in place, goes through on its own. The parameter order is the model's, the
    same on every rank, so every rank issues the same calls on the same layout.
    """

    parameters = list(parameters)
    if not contributed:
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                parameter.grad.zero_()
    bucket: list[torch.Tensor] = []
    size = 0
    for parameter in parameters:
        grad = parameter.grad
        nbytes = grad.numel() * grad.element_size()
        if bucket and (
            grad.dtype != bucket[0].dtype
            or grad.device != bucket[0].device
            or size + nbytes > bucket_bytes
        ):
            _reduce_bucket(bucket, executed_ranks)
            bucket, size = [], 0
        bucket.append(grad)
        size += nbytes
    if bucket:
        _reduce_bucket(bucket, executed_ranks)


def _reduce_bucket(grads: list[torch.Tensor], executed_ranks: int) -> None:
    if len(grads) == 1 and grads[0].is_contiguous():
        dist.all_reduce(grads[0], op=dist.ReduceOp.SUM)
        grads[0].div_(executed_ranks)
        return
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(executed_ranks)
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset : offset + count].view_as(grad))
        offset += count


def wait_for_every_rank() -> None:
    """Return only once every rank has reached this call.

    An all-reduce on the collective device rather than ``dist.barrier()``, whose
    NCCL form has guessed its device from the rank in some torch versions; this
    runs where every other collective here runs. The ``.item()`` makes the wait
    real under NCCL, whose all-reduce is otherwise only enqueued.
    """

    done = torch.zeros(1, device=collective_device())
    dist.all_reduce(done)
    done.item()


def gather_objects(value) -> list:
    """Every rank's ``value``, in rank order, on every rank."""

    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def parameter_digest(parameters) -> list[int]:
    """One exact integer per tensor: the sum of its elements' bit patterns.

    Exact where a float sum is not -- a single flipped bit anywhere changes the
    tensor's entry -- and cheap: one reduction per tensor on its own device and
    one transfer for the lot, against a GPU-to-host copy of every weight.
    """

    sums = [
        parameter.detach()
        .view(_BIT_PATTERN_DTYPES[parameter.element_size()])
        .sum(dtype=torch.int64)
        for parameter in parameters
    ]
    return [int(value) for value in torch.stack(sums).tolist()] if sums else []


def check_replicas(parameters, *, where: str, extra: dict | None = None) -> None:
    """Refuse replicas whose trainable parameters differ, on every rank alike.

    The gradient average keeps the replicas identical only if every rank applies
    the same update to the same weights; this is the check that they did, and
    the only one there is -- rank 0 alone writes weights, so nothing downstream
    could ever compare them. Every rank compares the same gathered list and so
    raises the same error at the same point. ``extra`` rides along for facts
    that must also agree, like the step a resume starts at.
    """

    local = {"digest": parameter_digest(parameters), **(extra or {})}
    gathered = gather_objects(local)
    reference = gathered[0]
    differing = [rank for rank, entry in enumerate(gathered) if entry != reference]
    if not differing:
        return
    details = []
    for rank in differing:
        entry = gathered[rank]
        tensors = sum(
            1 for mine, theirs in zip(entry["digest"], reference["digest"]) if mine != theirs
        )
        facts = ", ".join(
            f"{key}={entry.get(key)!r} against {reference.get(key)!r}"
            for key in sorted(set(entry) | set(reference))
            if key != "digest" and entry.get(key) != reference.get(key)
        )
        details.append(
            f"rank {rank}: {tensors} of {len(reference['digest'])} trainable tensors "
            "differ" + (f"; {facts}" if facts else "")
        )
    raise RuntimeError(
        f"the replicas differ {where}, against rank 0 -- " + "; ".join(details)
        + ". Every rank must hold the same weights for the averaged gradient to "
        "mean anything, so the run stops before writing anything; a resume whose "
        "ranks read different train_state.pt files does this too"
    )
