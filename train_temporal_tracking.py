#!/usr/bin/env python3
"""Multi-scene temporal-tracking trainer, replaying MVTracker's sample stream.

**Landing 3 of four: everything but the scene source.** The planner, the training
step, the schedule, gradient clipping, checkpoint/resume, signal handling and the
occupancy guard are written and tested. What is *not* here is one function --
``scene_provider(plan) -> DumpedKubricScene`` -- so ``--manifest`` cannot yet train
end to end against real data. That seam is waiting on a cluster measurement which
decides whether scenes arrive from the live MVTracker dataset or through the
dumper, and the step does not depend on the answer: both routes produce a
``DumpedKubricScene``, so the step is written against the scene rather than its
origin.

``--plan_only`` works on its own and is worth running before any GPU is
allocated: it walks a real manifest and reports what every step *would* select, so
a manifest whose rows this trainer cannot replay shows up at submit time instead
of twelve hours in.

  python train_temporal_tracking.py --manifest run/manifest.jsonl --plan_only
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from arc.training.manifest_plan import (
    SKIP_EXCLUDED_DATA_ROOT,
    ManifestPlanError,
    StepPlan,
    plan_manifest,
)
from arc.training.runtime import (
    assert_frozen_gradients_absent,
    assert_trainable_gradients_finite,
    assert_trainable_parameter_set,
    autocast_context,
    build_optimizer,
    confidence_stats,
    gradient_norm,
    move_views_to_cuda,
    tracking_only,
)
from arc.training.sample_manifest import MANIFEST_VERSION, read_manifest
from arc.training.schedule import (
    apply_learning_rate,
    capture_base_learning_rates,
    warmup_cosine_scale,
)
from arc.training.trainer_state import (
    build_trainer_state,
    read_trainer_state,
    restore_rng_state,
    save_atomically,
)


# The committed first-run window: 4 cameras x 12 times at stride 2 = 48
# observations. The budget is a measured memory ceiling -- peak ~= 2.55*N + 9.6
# GiB at HEAD, N = cameras x times, and 48 was measured at 132.2 GiB on an
# H200 -- not a preference, so it is a flag with a default rather than a constant.
# Falling back to 40 gives the 4x10 window at 111.7 GiB and an identical
# embedding contract, since the stride does not move.
DEFAULT_OBSERVATION_BUDGET = 48
DEFAULT_STRIDE = 2
# The time-index embedding's row count. A window may not carry more times than
# the table can index, whatever the budget allows.
DEFAULT_MAX_TIME_INDICES = 32
# Above this share of unreplayable rows the manifest is damaged rather than
# merely untidy, and training on the remainder would be training on a fraction of
# the recorded stream while every other number looked healthy.
DEFAULT_MAX_UNREPLAYABLE_FRACTION = 0.02
# The committed window sits at 94% of an H200's usable memory, measured over two
# steps. A run is thousands, and a CUDA allocator fragments, so a step's peak is
# checked against this fraction of the device rather than trusted -- the point is
# to die with a step index and a number instead of a bare OOM at step 4000.
DEFAULT_MAX_DEVICE_FRACTION = 0.97


# Set by the signal handlers, read at the top of each step. A handler that wrote a
# checkpoint from inside the signal context could land mid-backward; setting a
# flag and checkpointing at a known-safe point cannot.
_STOP_REQUESTED: list[str] = []


def _request_stop(signal_number, _frame) -> None:
    _STOP_REQUESTED.append(signal.Signals(signal_number).name)


def install_signal_handlers() -> None:
    """SIGUSR1/SIGTERM ask the loop to checkpoint and exit 0.

    SLURM sends USR1 at T-300s (``#SBATCH --signal=USR1@300``), so the grace
    window is what this has to fit inside: flag now, write at the top of the next
    step, exit 0 so the requeue is a resume rather than a failure.
    """

    for number in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(number, _request_stop)


def stop_requested() -> str | None:
    return _STOP_REQUESTED[0] if _STOP_REQUESTED else None


@dataclass
class StepOutcome:
    """What one optimizer step produced, for the log and `run_summary.json`."""

    step: int
    seq_name: str
    loss: float
    metric_error_m: float
    sample_count: int
    alignment_scale: float
    alignment_residual_m: float
    learning_rates: list[float]
    gradient_norms: dict
    peak_bytes: int = 0
    confidence: dict | None = None


class SceneCache:
    """One scene, keyed on the whole window it was built for.

    Keying on the scene *name* alone would be wrong: cameras, times and anchors
    all vary per step, and the correspondences are derived from them, so two
    steps on the same scene with different windows need different objects. The
    key is therefore everything ``build_scene`` consumed.

    Size one, and the previous entry is dropped *before* the next load starts —
    with the caller's own reference cleared first, or the loop's local keeps the
    old scene alive across the load and two are resident at the peak.

    **A hit returns the identical object, so a step that mutates a scene in place
    would silently change what the next step trains on.** That is the whole risk
    of caching here, and it is guarded rather than assumed: a fingerprint of the
    mutable per-view state is taken when a scene is stored and re-checked on every
    hit, so a mutating step fails loudly at the next visit instead of quietly
    making a cache hit differ from a reload. Returning a defensive copy instead
    would cost the transfer the cache exists to avoid, and repairing the damage
    silently would be worse than reporting it.
    """

    def __init__(self, loader, size: int = 1):
        if size < 1:
            raise ValueError(f"cache size must be at least 1, got {size}")
        self._loader = loader
        self._size = size
        self._entries: dict[tuple, object] = {}
        self._fingerprints: dict[tuple, tuple] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(plan: StepPlan) -> tuple:
        return (plan.data_root, plan.seq_name, plan.cameras, plan.times)

    @staticmethod
    def fingerprint(scene) -> tuple:
        """The mutable per-view state a step could plausibly rebind or write.

        Deliberately cheap: identities and shapes of the big tensors, values of
        the two tiny index tensors. It catches a rebind (``view[k] = ...``) and a
        write to the indices, which is the class of mutation the step loop can
        actually perform, without touching a pixel.
        """

        state = []
        for view in getattr(scene, "views", []) or []:
            entry = []
            for name in ("img", "time_index", "track_query_idx"):
                value = view.get(name) if hasattr(view, "get") else None
                if value is None:
                    entry.append(None)
                elif name == "img":
                    entry.append((id(value), tuple(value.shape), str(value.device)))
                else:
                    entry.append(tuple(value.reshape(-1).tolist()))
            state.append(tuple(entry))
        return tuple(state)

    def get(self, plan: StepPlan):
        key = self.key(plan)
        cached = self._entries.get(key)
        if cached is not None:
            current = self.fingerprint(cached)
            if current != self._fingerprints[key]:
                raise RuntimeError(
                    f"the cached scene for {plan.seq_name!r} was mutated in place "
                    "since it was stored, so this cache hit would train on "
                    "different content than a reload. A step must not rebind or "
                    "write the scene's view tensors; give it its own shallow "
                    "copies (as the index-reversal probe does) instead."
                )
            self.hits += 1
            return cached
        self.misses += 1
        while len(self._entries) >= self._size:
            # Oldest first: popitem() would take the newest, which at size > 1
            # evicts the entry most likely to be wanted next.
            evicted = next(iter(self._entries))
            self._entries.pop(evicted)
            self._fingerprints.pop(evicted, None)
        scene = self._loader(plan)
        self._entries[key] = scene
        self._fingerprints[key] = self.fingerprint(scene)
        return scene


def train_step(
    *,
    model,
    scene,
    plan: StepPlan,
    optimizer,
    scaler,
    precision: str,
    huber_delta_m: float,
    grad_clip: float,
    learning_rates: list[float],
    step: int,
) -> StepOutcome:
    """One optimizer step over one scene, with every guard the harness runs.

    Reuses the existing machinery unchanged — this adds nothing to
    ``arc/training``'s semantics. There is no within-scene split: `ef8bcff`
    deleted it, so every eligible correspondence is supervised.
    """

    from arc.training import (
        build_anchor_correspondences,
        fit_scene_sim3,
        gather_query_anchor_points,
        sparse_tracking_loss,
    )

    model.train()
    model.head.eval()
    model.cam_dec.eval()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    correspondences = build_anchor_correspondences(scene)
    with autocast_context(precision):
        raw = model(scene.views, force_no_output_conversion=True)
        alignment, alignment_report = fit_scene_sim3(raw, scene)
        anchors = gather_query_anchor_points(raw, scene, correspondences)
        stats = confidence_stats(raw)
        result = sparse_tracking_loss(
            tracking_only(raw),
            scene,
            correspondences,
            alignment,
            anchors,
            huber_delta_m=huber_delta_m,
            collect_diagnostics=False,
        )

    scaler.scale(result.total_loss).backward()
    scaler.unscale_(optimizer)
    assert_trainable_gradients_finite(model)
    norms = {
        "time_embedding": gradient_norm(
            model.backbone.pretrained.time_index_embedding.parameters()
        ),
        "motion_decoder": gradient_norm(model.motion_decoder.parameters()),
        "track_head": gradient_norm(model.track_head.parameters()),
    }
    if norms["time_embedding"] == 0:
        raise RuntimeError("Temporal embedding gradient norm is zero")
    if norms["motion_decoder"] == 0 or norms["track_head"] == 0:
        raise RuntimeError("MotionDecoder or track-head gradient norm is zero")
    # Clip after unscale_ and before step, or the threshold is applied to scaled
    # gradients and means nothing.
    norms["clipped_total"] = float(
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        ).item()
    )
    assert_frozen_gradients_absent(model)
    scaler.step(optimizer)
    scaler.update()

    return StepOutcome(
        step=step,
        seq_name=plan.seq_name,
        loss=float(result.loss.detach().item()),
        metric_error_m=float(result.metric_error.detach().item()),
        sample_count=int(result.sample_count),
        alignment_scale=float(alignment_report["scale"]),
        alignment_residual_m=float(alignment_report["median_residual_metric"]),
        learning_rates=list(learning_rates),
        gradient_norms=norms,
        peak_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        confidence=stats,
    )


def check_device_headroom(peak_bytes: int, *, step: int, max_fraction: float) -> None:
    """Fail with a number and a step index rather than a bare CUDA OOM."""

    if not torch.cuda.is_available() or peak_bytes <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = peak_bytes / total
    if fraction > max_fraction:
        raise RuntimeError(
            f"step {step}: peak {peak_bytes / 2**30:.1f} GiB is {fraction:.1%} of "
            f"the device's {total / 2**30:.1f} GiB, over --max_device_fraction "
            f"{max_fraction:.1%}. The committed window sits near this ceiling by "
            "design; lower --observation_budget, or lower the DPT head's "
            "frames_chunk_size for more recompute and a lower peak."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay an MVTracker sample manifest to train 4RC temporal tracking"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="MVTracker manifest.jsonl to replay, in file order",
    )
    parser.add_argument(
        "--observation_budget",
        type=int,
        default=DEFAULT_OBSERVATION_BUDGET,
        help=(
            "Maximum cameras x times per step. Peak memory depends only on this "
            "product, not on how it splits, so it is the one knob that bounds a "
            "step's footprint (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=(
            "Frames between consecutive selected times. Fixed for the whole run "
            "so embedding row k always means k*stride frames after the anchor; a "
            "run at a different stride trains a different embedding and its "
            "patch is not loadable here (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max_time_indices",
        type=int,
        default=DEFAULT_MAX_TIME_INDICES,
        help="Rows in the time-index embedding (default: %(default)s)",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=2,
        help=(
            "Records with fewer views are skipped. Below two there is no "
            "synchronized cross-view pair, so the run cannot measure the thing "
            "temporal indexing exists for (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--exclude_data_root",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Skip records whose data_root matches, repeatable. A run's "
            "static-pretraining epoch draws from other datasets and is marked by "
            "nothing else; these are replayable rows this trainer declines, not "
            "damage, and they are counted separately"
        ),
    )
    parser.add_argument(
        "--max_unreplayable_fraction",
        type=float,
        default=DEFAULT_MAX_UNREPLAYABLE_FRACTION,
        help=(
            "Abort if more than this share of considered records are "
            "unreplayable. Deliberate exclusions leave both sides of the ratio, "
            "so excluding a large epoch cannot mask real damage "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help=(
            "Report what every step would select and exit. Needs no GPU, no "
            "checkpoint and no scene data. Currently the only supported mode"
        ),
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Plan only the first N records; for eyeballing a long manifest",
    )
    parser.add_argument("--json_out", help="Write the plan report as JSON")

    training = parser.add_argument_group("training")
    training.add_argument("--checkpoint_dir", help="Released 4RC weights to start from")
    training.add_argument("--output_dir", help="Checkpoints, logs and run_summary.json")
    training.add_argument("--num_steps", type=int, default=20000)
    training.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Linear warmup before the cosine decay begins (default: %(default)s)",
    )
    training.add_argument("--lr", type=float, default=1e-5)
    training.add_argument("--embedding_lr", type=float, default=None)
    training.add_argument(
        "--encoder_lr",
        type=float,
        default=None,
        help="Default 0.1 x --lr; low because the frozen geometry heads read these features",
    )
    training.add_argument(
        "--min_lr_scale",
        type=float,
        default=0.0,
        help="Cosine floor as a fraction of the base rate (default: %(default)s)",
    )
    training.add_argument("--grad_clip", type=float, default=1.0)
    training.add_argument("--huber_delta_m", type=float, default=0.05)
    training.add_argument("--freeze_mode", default="temporal_tracking_global_attention")
    training.add_argument("--late_global_blocks", type=int, default=None)
    training.add_argument("--precision", choices=("32", "16-mixed", "bf16-mixed"), default="bf16-mixed")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--save_every", type=int, default=1000)
    training.add_argument(
        "--resume",
        help=(
            "Trainer checkpoint to continue from. Deliberately explicit: "
            "auto-resuming from the lexicographically highest file in an output "
            "directory has already caused problems on the MVTracker side"
        ),
    )
    training.add_argument(
        "--max_device_fraction",
        type=float,
        default=DEFAULT_MAX_DEVICE_FRACTION,
        help=(
            "Fail if a step's peak crosses this fraction of device memory "
            "(default: %(default)s). The committed window runs near it by design"
        ),
    )
    training.add_argument("--scene_cache", type=int, default=1)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.observation_budget < 1:
        raise ValueError("--observation_budget must be positive")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.max_time_indices < 1:
        raise ValueError("--max_time_indices must be positive")
    if args.min_views < 1:
        raise ValueError("--min_views must be at least 1")
    if not 0 <= args.max_unreplayable_fraction <= 1:
        raise ValueError("--max_unreplayable_fraction must be in [0, 1]")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max_records must be positive")
    if args.observation_budget < args.min_views:
        raise ValueError(
            f"--observation_budget {args.observation_budget} cannot seat even one "
            f"time at --min_views {args.min_views}"
        )


def _plan_summary(tally, args) -> dict:
    strides = Counter(plan.stride for plan in tally.planned)
    bounds = Counter(plan.time_bound for plan in tally.planned)
    observations = Counter(plan.observation_count for plan in tally.planned)
    views = Counter(len(plan.cameras) for plan in tally.planned)
    duplicates = [plan.duplicate_track_count for plan in tally.planned]
    tracks = [len(plan.track_indices) for plan in tally.planned]
    transformed = sum(1 for plan in tally.planned if plan.scene_transform)
    return {
        "manifest_version_expected": MANIFEST_VERSION,
        "records_read": len(tally.planned) + len(tally.skipped),
        "planned_steps": len(tally.planned),
        "skipped": tally.skip_counts,
        "unreplayable_fraction": tally.threshold_skip_fraction,
        "considered_for_threshold": tally.considered,
        "distinct_scenes": len({plan.seq_name for plan in tally.planned}),
        "views_per_step": dict(sorted(views.items())),
        "observations_per_step": dict(sorted(observations.items())),
        "time_bound": dict(bounds),
        "stride": dict(strides),
        "tracks_per_step": (
            None
            if not tracks
            else {"min": min(tracks), "max": max(tracks), "mean": sum(tracks) / len(tracks)}
        ),
        # Non-zero is normal: the loader draws from overlapping pools. Reported so
        # a positional selection and a set-based one differ by a number here
        # rather than by supervision quietly going missing.
        "duplicate_track_ids": (
            None
            if not duplicates
            else {"total": sum(duplicates), "max_in_a_step": max(duplicates)}
        ),
        "records_with_scene_transform": transformed,
        "settings": {
            "observation_budget": args.observation_budget,
            "stride": args.stride,
            "max_time_indices": args.max_time_indices,
            "min_views": args.min_views,
            "excluded_data_roots": list(args.exclude_data_root),
        },
    }


def _print_plan(tally, summary, *, limit: int = 20) -> None:
    for plan in tally.planned[:limit]:
        times = list(plan.times)
        shown = times if len(times) <= 6 else times[:5] + ["..."] + times[-1:]
        print(
            f"step={plan.step} scene={plan.seq_name} "
            f"views={list(plan.cameras)} "
            f"window=[{plan.frame_start},{plan.frame_start + plan.seq_len}) "
            f"times={shown} stride={plan.stride} bound={plan.time_bound} "
            f"obs={plan.observation_count} "
            f"tracks={len(plan.track_indices)}"
            f"(+{plan.duplicate_track_count} dup) "
            f"transform={'yes' if plan.scene_transform else 'no'}"
        )
    if len(tally.planned) > limit:
        print(f"... {len(tally.planned) - limit} more planned steps")

    for entry in tally.skipped[:limit]:
        print(f"SKIP step={entry.step} scene={entry.seq_name} {entry.cause}: {entry.detail}")
    if len(tally.skipped) > limit:
        print(f"... {len(tally.skipped) - limit} more skipped records")

    print()
    for key in (
        "records_read",
        "planned_steps",
        "distinct_scenes",
        "views_per_step",
        "observations_per_step",
        "time_bound",
        "tracks_per_step",
        "duplicate_track_ids",
        "records_with_scene_transform",
        "skipped",
        "unreplayable_fraction",
    ):
        print(f"{key}={summary[key]}")


def run_training(
    *,
    model,
    optimizer,
    scaler,
    plans,
    args,
    scene_provider,
    step_fn=train_step,
    output_dir: Path,
) -> dict:
    """The loop: schedule, step, guard, checkpoint, and stop when asked.

    ``scene_provider`` and ``step_fn`` are injected so the loop can be driven end
    to end without a GPU or a scene source -- which is what lets resume, the
    schedule and the cache be tested at all, and is also the seam the real scene
    source plugs into once the cluster probe decides which one it is.
    """

    # Captured BEFORE any optimizer.load_state_dict: that call overwrites each
    # group's lr with the decayed value from the checkpoint, so capturing after
    # would restart the cosine from the decayed rate and compound it once per
    # resume, with nothing raising anywhere.
    base_learning_rates = capture_base_learning_rates(optimizer)
    start_step = 0
    resumed_from = None

    if args.resume:
        from arc.training.checkpoint import load_temporal_tracking_checkpoint

        payload = read_trainer_state(args.resume)
        # The weights, through the existing strict loader rather than a second
        # copy of the same overlay: it checks the key set against the model's own
        # trainable parameters, so a patch from a different freeze mode is refused
        # here instead of silently restoring a subset.
        load_temporal_tracking_checkpoint(model, args.resume)
        optimizer.load_state_dict(payload["optimizer"])
        stored = [float(value) for value in payload["base_learning_rates"]]
        if stored != base_learning_rates:
            print(
                f"WARNING --resume carries base rates {stored} but this "
                f"invocation's flags give {base_learning_rates}; the checkpoint's "
                "are authoritative, so the flag change has NOT taken effect",
                file=sys.stderr,
            )
        base_learning_rates = stored
        start_step = int(payload["step"])
        restore_rng_state(payload["rng"])
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        resumed_from = str(args.resume)
        print(f"resumed from {resumed_from} at step {start_step}")

    cache = SceneCache(scene_provider, size=args.scene_cache)
    history: list[StepOutcome] = []
    scene = None
    interrupted = None
    last_saved_step = None

    for step in range(start_step, args.num_steps):
        signal_name = stop_requested()
        if signal_name is not None:
            interrupted = signal_name
            print(f"{signal_name} received; checkpointing at step {step} and exiting 0")
            break

        scale = warmup_cosine_scale(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=args.num_steps,
            min_lr_scale=args.min_lr_scale,
        )
        learning_rates = apply_learning_rate(optimizer, base_learning_rates, scale)

        plan = plans[step % len(plans)]
        # Drop this loop's own reference before the cache loads the next scene, or
        # the old one stays alive across the load and two are resident at the peak.
        scene = None
        scene = cache.get(plan)

        outcome = step_fn(
            model=model,
            scene=scene,
            plan=plan,
            optimizer=optimizer,
            scaler=scaler,
            precision=args.precision,
            huber_delta_m=args.huber_delta_m,
            grad_clip=args.grad_clip,
            learning_rates=learning_rates,
            step=step,
        )
        check_device_headroom(
            outcome.peak_bytes, step=step, max_fraction=args.max_device_fraction
        )
        history.append(outcome)
        print(
            f"step={step}/{args.num_steps} scene={outcome.seq_name} "
            f"loss={outcome.loss:.8f} metric_error_m={outcome.metric_error_m:.8f} "
            f"lr={learning_rates[0]:.3g} align_scale={outcome.alignment_scale:.6f} "
            f"align_residual_m={outcome.alignment_residual_m:.6f} "
            f"samples={outcome.sample_count} "
            f"peak_gib={outcome.peak_bytes / 2**30:.1f}"
        )

        completed = step + 1
        if args.save_every and completed % args.save_every == 0:
            _write_checkpoint(model, optimizer, scaler, base_learning_rates,
                              step=completed, output_dir=output_dir, args=args)
            last_saved_step = completed

    completed_steps = (step + 1) if history and interrupted is None else (
        history[-1].step + 1 if history else start_step
    )
    # A clean run whose last step lands on a --save_every boundary has already
    # written exactly this state. The rename is atomic and the content identical,
    # so repeating it is harmless -- but it is a multi-GB write to shared storage
    # at the end of every aligned run, for nothing.
    if last_saved_step != completed_steps:
        _write_checkpoint(model, optimizer, scaler, base_learning_rates,
                          step=completed_steps, output_dir=output_dir, args=args)

    return {
        "start_step": start_step,
        "completed_steps": completed_steps,
        "resumed_from": resumed_from,
        "interrupted_by": interrupted,
        "scene_cache": {"hits": cache.hits, "misses": cache.misses},
        "history": history,
    }


def _write_checkpoint(model, optimizer, scaler, base_learning_rates, *, step, output_dir, args) -> Path:
    """The temporal patch plus everything a resume needs, in one atomic file."""

    from arc.training.checkpoint import save_temporal_tracking_checkpoint  # noqa: F401

    state = build_trainer_state(
        step=step,
        optimizer=optimizer,
        base_learning_rates=base_learning_rates,
        scaler=scaler,
        settings={
            "observation_budget": args.observation_budget,
            "stride": args.stride,
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "min_lr_scale": args.min_lr_scale,
            "precision": args.precision,
        },
    )
    payload = {
        "freeze_mode": getattr(model, "freeze", None),
        "state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        **state,
    }
    return save_atomically(payload, Path(output_dir) / "train_state.pt")


def _gate_verdicts(summary: dict) -> dict:
    """Reported, never an exit code — and tri-state, never a silent pass.

    An interrupted run has no verdict on anything, and ``all([])`` is ``True``:
    without the tri-state a run stopped at step 3 of 100,000 reports
    ``gates_passed: true`` and any automation downstream believes it.
    """

    gates = {
        "completed_all_steps": (
            None
            if summary["interrupted_by"]
            else summary["completed_steps"] >= summary["planned_steps"]
        ),
    }
    evaluated = [value for value in gates.values() if value is not None]
    return {
        "gates": gates,
        "gates_evaluated": len(evaluated),
        "gates_passed": None if len(evaluated) != len(gates) else all(evaluated),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")
    records = read_manifest(manifest_path)
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        parser.error(f"{manifest_path} holds no records")

    versions = {record.get("manifest_version") for record in records}
    if versions != {MANIFEST_VERSION}:
        print(
            f"WARNING manifest_version {sorted(versions, key=str)} against the "
            f"vendored schema's {MANIFEST_VERSION}; a key may have changed "
            "meaning. Re-vendor arc/training/sample_manifest.py from mvtracker "
            "rather than reading the difference by hand.",
            file=sys.stderr,
        )

    try:
        tally = plan_manifest(
            records,
            available_cameras=None,
            budget=args.observation_budget,
            stride=args.stride,
            max_time_indices=args.max_time_indices,
            min_views=args.min_views,
            excluded_data_roots=args.exclude_data_root,
        )
    except ManifestPlanError as exc:
        parser.error(str(exc))

    summary = _plan_summary(tally, args)
    _print_plan(tally, summary)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"summary={args.json_out}")

    # stderr is unbuffered and stdout is not, so without this the verdict lands
    # above the report it is a verdict on whenever the two are merged.
    sys.stdout.flush()

    if tally.threshold_skip_fraction > args.max_unreplayable_fraction:
        print(
            f"FAIL {tally.threshold_skip_count}/{tally.considered} considered "
            f"records are unreplayable "
            f"({tally.threshold_skip_fraction:.4%} > "
            f"{args.max_unreplayable_fraction:.4%}). Deliberate exclusions "
            f"({tally.skip_counts.get(SKIP_EXCLUDED_DATA_ROOT, 0)}) are not "
            "counted here.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not tally.planned:
        print("FAIL no record produced a replayable step", file=sys.stderr)
        raise SystemExit(1)

    if args.plan_only:
        print(f"PASS planned {len(tally.planned)} steps from {manifest_path}")
        return

    # Everything above this line is landing 2 and works. Below it, exactly one
    # piece is missing, and it is named rather than approximated: a plan has to
    # become a DumpedKubricScene, and whether that comes from the live MVTracker
    # dataset or through the dumper is what the cluster probe decides. The loop,
    # the step, the schedule, the checkpointing and the guards are all written
    # and tested against an injected provider (see run_training).
    raise NotImplementedError(
        "scene_provider is not bound yet, so --manifest cannot train end to end. "
        "Everything else in the loop is implemented and tested: the training "
        "step, warmup+cosine, gradient clipping, checkpoint/resume, SIGUSR1/"
        "SIGTERM handling, the scene cache and the occupancy guard. What is "
        "missing is the one function mapping a StepPlan to a DumpedKubricScene, "
        "which waits on the merged-env probe (does RC_ENV drive MVTracker's live "
        "dataset at acceptable latency, or does the dumper become the replay?). "
        "Use --plan_only until it lands."
    )


if __name__ == "__main__":
    main()
