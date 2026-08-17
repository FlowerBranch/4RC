#!/usr/bin/env python3
"""Multi-scene temporal-tracking trainer, replaying MVTracker's sample stream.

The planner, the training step, the schedule, gradient clipping, checkpoint/resume,
signal handling, the occupancy guard, the scene source and the held-out eval are
all here, so ``--manifest`` trains end to end.  Scenes arrive through
:class:`arc.training.scene_provider.MVTrackerSceneProvider`, which replays each
record against MVTracker's own loader rather than a reimplementation.

**What "replaying the sample stream" claims, at its honest granularity:** the same
scenes, windows, view sets and step ordering as the run being replayed -- and each
run's own tracks and query points, because the eligible pool is rebuilt from
post-crop visibility and the crop is drawn from an unrecorded global RNG stream.
Measured twice on one scene, two draws shared a fifth of their track ids (jaccard
0.256 and 0.219); of the trajectories they did share, 52.5% also landed on the same
query time, so roughly one supervised query point in five is identical between the
runs.  The comparison that carries the result is the held-out curve at
``--eval_every 500``, not a per-step input diff.

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
from arc.training.scene_provider import SceneProviderError
from arc.training.schedule import (
    apply_learning_rate,
    capture_base_learning_rates,
    warmup_cosine_scale,
)
from arc.training.trainer_state import (
    build_trainer_state,
    capture_rng_state,
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


def scene_skip_cause(error: Exception) -> str:
    """Bucket a scene-load failure, so a tally says *what* is wrong.

    The causes have different remedies -- a missing scene usually means the wrong
    ``data_root``, a small pool means a degenerate scene -- so a single "failed"
    count would report the symptom of a broken run and a healthy one identically.
    """

    text = str(error)
    if "is not in the pool" in text:
        return "scene_absent"
    if "eligible tracks" in text:
        return "pool_too_small"
    if "nothing to supervise" in text:
        return "no_recorded_tracks_present"
    return "other"


def check_scene_skip_rate(
    skips,
    *,
    attempted: int,
    consecutive: int,
    max_fraction: float,
    max_consecutive: int,
    min_attempts: int = 50,
) -> None:
    """Abort when scene loads fail systematically, with the tally.

    Skipping is right for one degenerate scene in thousands and wrong for a
    broken ``--data_root``: without a limit, a wrong root produces a run that
    advances its counter for two days, records no gradients and writes a summary
    that reads as complete. That is the failure this guards, and it is the exact
    shape an inherited ``max_videos=30`` cap took before it was fixed.

    **Two rules, because one does not cover both ends of a run.** A *fraction*
    is meaningless early -- one skip in the first two steps is 50% -- so it only
    applies once ``min_attempts`` steps have been tried. A run of *consecutive*
    failures needs no denominator and is what a broken root actually looks like,
    so it catches the same fault within a few steps instead of fifty.
    """

    total = sum(skips.values())
    causes = dict(sorted(skips.items()))
    if consecutive >= max_consecutive:
        raise RuntimeError(
            f"{consecutive} consecutive steps could not load their scene "
            f"(--max_consecutive_scene_skips {max_consecutive}); causes: {causes}. "
            "A run of failures back to back is a broken data root, not bad luck"
        )
    if attempted < min_attempts or not total:
        return
    fraction = total / attempted
    if fraction <= max_fraction:
        return
    raise RuntimeError(
        f"{total} of {attempted} steps could not load their scene "
        f"({fraction:.1%} > --max_scene_skip_fraction {max_fraction:.1%}); "
        f"causes: {causes}. This is a broken data root or an over-restricted "
        "pool, not a few bad scenes"
    )


def cuda_scene_provider(provider):
    """Wrap a scene provider so loaded scenes arrive on the GPU.

    **The move has to happen inside the loader, and that is forced rather than
    chosen.** :meth:`SceneCache.fingerprint` records ``id(...)`` and the device of
    each view's ``img``, while ``move_views_to_cuda`` *rebinds* ``view["img"]`` to
    a new tensor on a new device. Moving a scene that came *out* of the cache
    therefore trips the mutation guard on the next hit. ``SceneCache.get`` loads
    and only then fingerprints, so a provider that returns an already-moved scene
    is consistent with itself.

    **Only the views move.** ``build_anchor_correspondences`` runs every step and
    takes ``.cpu().numpy()`` of trajectories, visibility, intrinsics, extrinsics
    and a depth slice per anchor; moving the whole scene would turn that into a
    per-step round trip. The loss path already handles the split -- it takes its
    device from the predictions and moves what it needs.
    """

    def load(plan):
        scene = provider(plan)
        if torch.cuda.is_available():
            move_views_to_cuda(scene.views)
        return scene

    return load


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

    correspondences, eligibility = build_anchor_correspondences(scene)
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


def evaluate_held_out(
    *,
    model,
    plans,
    scene_provider,
    precision: str,
    huber_delta_m: float,
    step: int,
    output_dir: Path,
    emit_predictions: bool = True,
) -> dict:
    """Score the held-out scenes without leaving a trace on the training run.

    Two things are restored afterwards, and both matter. The **RNG streams**,
    because an eval that consumed randomness would shift every subsequent
    training draw and make a run with `--eval_every` differ from one without --
    which is exactly what work-order test 5 asserts. And the **module modes**: the
    step loop leaves `model.head` and `model.cam_dec` in `eval()` while the root
    is in `train()`, so a bare `model.train()` here would silently re-enable the
    frozen heads' training behaviour for every step after the first eval.

    Predictions are written in the cluster scorers' schema and **not scored** --
    ``evaluate_3dpt`` lives in the other environment.
    """

    from arc.training import (
        build_anchor_correspondences,
        fit_scene_sim3,
        gather_query_anchor_points,
        sparse_tracking_loss,
    )
    from arc.training.predictions import write_scene_predictions
    from arc.training.runtime import shuffled_index_views

    rng = capture_rng_state()
    modes = {name: module.training for name, module in model.named_modules()}
    directory = Path(output_dir) / "eval" / f"step-{step}"
    per_scene: list[dict] = []

    try:
        model.eval()
        for plan in plans:
            scene = scene_provider(plan)
            correspondences, eligibility = build_anchor_correspondences(scene)
            with torch.no_grad(), autocast_context(precision):
                raw = model(scene.views, force_no_output_conversion=True)
                alignment, alignment_report = fit_scene_sim3(raw, scene)
                anchors = gather_query_anchor_points(raw, scene, correspondences)
                result = sparse_tracking_loss(
                    tracking_only(raw),
                    scene,
                    correspondences,
                    alignment,
                    anchors,
                    huber_delta_m=huber_delta_m,
                )
                entry = {
                    "scene": plan.seq_name,
                    "position_loss": float(result.loss.item()),
                    "metric_error_m": float(result.metric_error.item()),
                    "sample_count": int(result.sample_count),
                    "alignment_scale": float(alignment_report["scale"]),
                    "confidence": confidence_stats(raw),
                }

                # The index-advantage arm: the same model scored with one camera's
                # time indices reversed. None when the window has no cross-camera
                # synchronization to break, and reported as None rather than 0.
                shuffled = shuffled_index_views(scene)
                if shuffled is not None:
                    shuffled_raw = model(shuffled, force_no_output_conversion=True)
                    shuffled_result = sparse_tracking_loss(
                        tracking_only(shuffled_raw),
                        scene,
                        correspondences,
                        alignment,
                        anchors,
                        huber_delta_m=huber_delta_m,
                    )
                    entry["position_loss_shuffled"] = float(shuffled_result.loss.item())
                    del shuffled_raw, shuffled_result
                else:
                    entry["position_loss_shuffled"] = None

                if emit_predictions:
                    arrays = _prediction_arrays(raw, scene, correspondences, alignment, anchors)
                    write_scene_predictions(
                        directory / "pred" / f"{plan.seq_name}.npz", arrays
                    )
            per_scene.append(entry)
            del raw, scene
    finally:
        restore_rng_state(rng)
        for name, module in model.named_modules():
            module.training = modes[name]

    losses = [entry["position_loss"] for entry in per_scene]
    errors = [entry["metric_error_m"] for entry in per_scene]
    shuffled_losses = [
        entry["position_loss_shuffled"]
        for entry in per_scene
        if entry["position_loss_shuffled"] is not None
    ]
    metrics = {
        "step": step,
        "scenes": len(per_scene),
        "position_loss": sum(losses) / len(losses) if losses else None,
        "metric_error_m": sum(errors) / len(errors) if errors else None,
        # None, not 0, when no scene had a synchronized pair to break: a zero here
        # would read as "reversal costs nothing", which is a finding rather than
        # an absence of one.
        "position_loss_shuffled": (
            sum(shuffled_losses) / len(shuffled_losses) if shuffled_losses else None
        ),
        "per_scene": per_scene,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _prediction_arrays(raw, scene, correspondences, alignment, anchors):
    """Assemble one scene's bundle in the scorers' schema.

    **The axis change is the substance here.** This repo's observation axis ``S``
    is camera-major over ``cameras x times``; the scorers' axis is ``T``
    timesteps with the cameras already fused, and their ``gt_vis_any`` is
    visibility reduced with ``any`` over cameras. So each covered timestep's
    cameras are combined before writing.

    The fusion is a **confidence-weighted mean**, mirroring `score_joint.py`
    verbatim (``clip(conf, 1e-6)`` then a weighted average) rather than inventing
    a rule: that is what the existing joint-pass numbers were produced with, so a
    different rule here would make these files incomparable with them.
    """

    from arc.training import gather_at_correspondences, sparse_targets
    from arc.training.predictions import build_prediction_arrays

    positions, visible, _finite, mask = sparse_targets(scene, correspondences)
    metric = float(scene.track_upscaling_factor)
    # From the predictions, matching `sparse_tracking_loss` (`device =
    # tracks.device`). Taking it from `positions` inverts that: targets follow the
    # correspondences, which `build_anchor_correspondences` builds on CPU, so this
    # would gather a CUDA `track_multi` with CPU indices once the views move.
    device = raw["track_multi"].device
    # `sparse_targets` returns on the correspondences' device, which is CPU once
    # only the views have moved. Everything below indexes these with slot indices
    # derived from `device`, so they are co-located here rather than at each use.
    positions = positions.to(device)
    visible = visible.to(device)
    mask = mask.to(device)

    displacement = gather_at_correspondences(raw["track_multi"], correspondences.to(device))
    predicted = (
        alignment.to(device=device, dtype=torch.float32).apply_points(
            torch.as_tensor(anchors, device=device, dtype=torch.float32)[:, None, :]
            + displacement
        )
        * metric
    )
    target = positions * metric

    confidence = raw.get("conf_track_multi")
    weights = (
        gather_at_correspondences(confidence, correspondences.to(device)).clamp_min(1e-6)
        if confidence is not None
        else torch.ones_like(mask, dtype=torch.float32)
    )

    slot_times = scene.slot_times.to(device)
    covered = sorted({int(value) for value in slot_times.tolist()})
    fused, fused_gt, fused_visible = [], [], []
    for original_time in covered:
        slots = (slot_times == original_time).nonzero(as_tuple=True)[0]
        weight = weights[:, slots]
        fused.append(
            (predicted[:, slots] * weight[..., None]).sum(dim=1)
            / weight.sum(dim=1)[..., None]
        )
        # The target does not depend on which camera saw it, so any slot of this
        # instant carries it; taking the first is exact, not an approximation.
        fused_gt.append(target[:, slots[0]])
        fused_visible.append(visible[:, slots].any(dim=1))

    predicted_tn = torch.stack(fused, dim=0)
    target_tn = torch.stack(fused_gt, dim=0)
    visible_tn = torch.stack(fused_visible, dim=0)

    # Column 0 is the index into the covered timesteps, never the original frame.
    position_of_time = {value: index for index, value in enumerate(covered)}
    query_times = [
        position_of_time[int(value)] for value in correspondences.query_times.tolist()
    ]
    query_xyz = scene.trajectories_world.to(device)[
        correspondences.query_times.to(device),
        correspondences.trajectory_indices.to(device),
    ] * metric
    queries = torch.cat(
        [torch.tensor(query_times, device=device, dtype=torch.float32)[:, None], query_xyz],
        dim=1,
    )

    return build_prediction_arrays(
        predicted_positions=predicted_tn,
        ground_truth_positions=target_tn,
        # The scorer wants predicted visibility and inverts this, so what is
        # stored is occlusion.
        occluded=~visible_tn,
        query_points=queries,
        visible_any_camera=visible_tn,
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
            "checkpoint and no scene data"
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
    training.add_argument(
        "--max_scene_skip_fraction",
        type=float,
        default=0.02,
        help=(
            "Abort if more than this share of steps cannot load their scene "
            "(default: %(default)s). Skipping absorbs a few degenerate scenes; "
            "above this the root is wrong and the run would otherwise advance "
            "its counter for days while recording no gradients"
        ),
    )
    training.add_argument(
        "--max_consecutive_scene_skips",
        type=int,
        default=10,
        help=(
            "Abort after this many scene loads fail back to back (default: "
            "%(default)s). Needs no denominator, so it catches a broken root "
            "within a few steps rather than waiting for a rate to be meaningful"
        ),
    )
    training.add_argument(
        "--dataset_name",
        default="kubric-multiview-v3",
        help=(
            "Dataset name parsed for depth source and duster variants only. It "
            "no longer decides which cameras a step sees: the provider loads "
            "every view and indexes the record's own (default: %(default)s)"
        ),
    )
    training.add_argument("--size", type=int, default=512)
    training.add_argument(
        "--min_shared_queries",
        type=int,
        default=64,
        help=(
            "Skip a step whose scene has a smaller eligible track pool than "
            "(default: %(default)s) -- too small to be worth a step. A real "
            "scene's pool is thousands, so this catches a broken scene rather "
            "than trimming a distribution"
        ),
    )
    training.add_argument(
        "--honour_recorded_tracks",
        action="store_true",
        help=(
            "Supervise only the tracks a record names, instead of every eligible "
            "track in the scene. Off by default because it cannot be honoured: "
            "two loads of one scene share about a fifth of their track ids "
            "(jaccard 0.256 and 0.219 over two runs), and the eligible pool -- "
            "itself redrawn each load -- holds 71-81%% of any draw, since both "
            "are rebuilt from post-crop visibility and the crop is never "
            "recorded. Missing ids are counted, not fatal -- unless a record "
            "shares nothing at all with the pool, which skips the step"
        ),
    )

    evaluation = parser.add_argument_group("held-out eval")
    evaluation.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help=(
            "Steps between held-out evals. Pinned to MVTracker's CURVE_EVAL_FREQ "
            "so the two curves overlay step for step (default: %(default)s)"
        ),
    )
    evaluation.add_argument("--val_scenes_file", help="JSON list of held-out scene names")
    evaluation.add_argument(
        "--val_data_root",
        help=(
            "Directory the held-out scenes sit directly under -- the same kind of "
            "resolved path a manifest row's data_root carries, not a split root. "
            "Required whenever --val_scenes_file is given"
        ),
    )
    evaluation.add_argument(
        "--val_cameras",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Cameras for the held-out window; kubric-multiview-v3-views0123 (default: %(default)s)",
    )
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
    # Refused HERE, at parse time, and not where the plans are built: that call
    # sits after `Arc.from_pretrained(...).to("cuda")`, so raising there would
    # burn a full model load on a cluster node before reporting a missing flag --
    # and it would surface as a traceback, since main()'s ValueError handler
    # wraps only this function.
    #
    # Never guessed, either: the manifest's rows name the *training* directory
    # and the held-out set is a different one by construction, so any fallback
    # would silently score training scenes and file them as held-out.
    if args.val_scenes_file and not args.val_data_root:
        raise ValueError(
            "--val_scenes_file needs --val_data_root: the held-out scenes live in "
            "their own directory, and it cannot be derived from the manifest, "
            "whose rows name the training directory"
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
        # The RECORDED draw's size, not what a step supervises: with
        # --honour_recorded_tracks off (the default) every step supervises the
        # scene's whole eligible pool, measured at 5486-5789 entries against a
        # recorded ~2048. Named accordingly so the two are not confused.
        "recorded_tracks_per_step": (
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
        "recorded_tracks_per_step",
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
    val_plans=None,
) -> dict:
    """The loop: schedule, step, guard, checkpoint, and stop when asked.

    ``scene_provider`` and ``step_fn`` are injected so the loop can be driven end
    to end without a GPU or a scene source -- which is what lets resume, the
    schedule and the cache be tested at all, and is also the seam the real scene
    source plugs into: main() builds MVTrackerSceneProvider and wraps it.
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

    # Wrapped here rather than in main() so the wiring is on the path the CPU
    # tests already drive; main() cannot run without a GPU, so a fix installed
    # there would be untested by construction. Both consumers are covered, since
    # evaluate_held_out takes this same callable.
    scene_provider = cuda_scene_provider(scene_provider)
    cache = SceneCache(scene_provider, size=args.scene_cache)

    # Prove the held-out set is reachable BEFORE spending training time on it.
    # Without this a wrong --val_data_root surfaces at the first eval, which at
    # --eval_every 500 is hours into a two-day allocation; the eval's own load is
    # not covered by the step loop's skip policy, so it would end the run. This
    # also exercises the CUDA move against the real provider at step 0.
    # Deliberately NOT retained: this is a reachability check, not a cache, and
    # holding every held-out scene resident would cost GiB at a peak already
    # close to the guard.
    for plan in val_plans or []:
        scene_provider(plan)
    if val_plans:
        print(f"held_out_preflight_ok={len(val_plans)}")
    history: list[StepOutcome] = []
    evaluations: list[dict] = []
    scene = None
    interrupted = None
    last_saved_step = None
    scene_load_skips: Counter = Counter()
    consecutive_skips = 0

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
        try:
            scene = cache.get(plan)
        except SceneProviderError as error:
            # Skip the step, but ADVANCE THE COUNTER. The premise is a curve that
            # overlays MVTracker's step for step: retrying with the next plan would
            # put a different scene at this step and desynchronise the two runs for
            # the rest of the run. Losing one sample in thousands does not.
            scene_load_skips[scene_skip_cause(error)] += 1
            consecutive_skips += 1
            print(f"step={step} scene={plan.seq_name} skipped: {error}")
            check_scene_skip_rate(
                scene_load_skips,
                attempted=step - start_step + 1,
                consecutive=consecutive_skips,
                max_fraction=args.max_scene_skip_fraction,
                max_consecutive=args.max_consecutive_scene_skips,
            )

        # Deliberately NOT `continue`: the eval and checkpoint boundaries below
        # are properties of the step *number*, not of whether this step produced
        # a gradient. Skipping past them would drop the held-out point whenever a
        # scene failed to load exactly on an --eval_every boundary -- a hole in
        # the one curve this trainer exists to produce, with nothing in the
        # output saying why. The eval scores the model against held-out scenes
        # and does not depend on this step's scene at all.
        if scene is not None:
            consecutive_skips = 0
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
        if val_plans and args.eval_every and completed % args.eval_every == 0:
            metrics = evaluate_held_out(
                model=model,
                plans=val_plans,
                scene_provider=scene_provider,
                precision=args.precision,
                huber_delta_m=args.huber_delta_m,
                step=completed,
                output_dir=output_dir,
            )
            evaluations.append(metrics)
            print(
                f"eval step={completed} scenes={metrics['scenes']} "
                f"held_out_loss={metrics['position_loss']} "
                f"held_out_metric_error_m={metrics['metric_error_m']} "
                f"shuffled={metrics['position_loss_shuffled']}"
            )
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
        "scene_load_skips": dict(sorted(scene_load_skips.items())),
        "history": history,
        "evaluations": evaluations,
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

    if not args.checkpoint_dir or not args.output_dir:
        parser.error("--checkpoint_dir and --output_dir are required to train")

    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from arc.models.arc.arc import Arc
    from arc.training.scene_provider import MVTrackerSceneProvider

    install_signal_handlers()

    late_global_blocks = (
        args.late_global_blocks
        if args.freeze_mode == "temporal_tracking_late_global"
        else None
    )
    model = Arc.from_pretrained(
        args.checkpoint_dir, max_time_indices=args.max_time_indices
    ).to("cuda")
    model.set_freeze(args.freeze_mode, late_global_blocks=late_global_blocks)
    report = assert_trainable_parameter_set(
        model,
        freeze_mode=args.freeze_mode,
        max_time_indices=args.max_time_indices,
        late_global_blocks=late_global_blocks,
    )
    print(
        f"trainable={report['tensor_count']} tensors / "
        f"{report['parameter_count']} parameters ({args.freeze_mode})"
    )

    optimizer, learning_rates, _encoder = build_optimizer(
        model,
        lr=args.lr,
        embedding_lr=args.embedding_lr,
        encoder_lr=args.encoder_lr,
    )
    print(f"learning_rates={learning_rates}")
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "16-mixed")

    provider = MVTrackerSceneProvider(
        dataset_name=args.dataset_name,
        size=args.size,
        min_shared_queries=args.min_shared_queries,
        honour_recorded_tracks=args.honour_recorded_tracks,
    )
    val_plans = _val_plans(args)
    if val_plans:
        print(f"held_out_scenes={len(val_plans)} at --eval_every {args.eval_every}")

    output_dir = Path(args.output_dir)
    result = run_training(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        plans=tally.planned,
        args=args,
        scene_provider=provider,
        output_dir=output_dir,
        val_plans=val_plans,
    )

    summary = {
        **_plan_summary(tally, args),
        "start_step": result["start_step"],
        "completed_steps": result["completed_steps"],
        "resumed_from": result["resumed_from"],
        "interrupted_by": result["interrupted_by"],
        "scene_cache": result["scene_cache"],
        "scene_load_skips": result["scene_load_skips"],
        "evaluations": result["evaluations"],
        "trainable_tensor_count": report["tensor_count"],
        "trainable_parameter_count": report["parameter_count"],
        "learning_rates": learning_rates,
        # Zero unless --honour_recorded_tracks: nothing is requested otherwise.
        # Reported rather than raised on, because a recorded draw is not
        # reproducible (V6(c)) and a raise would fire on nearly every record.
        "honour_recorded_tracks": bool(args.honour_recorded_tracks),
        "requested_track_ids": provider.requested_track_ids,
        "missing_track_ids": provider.missing_track_ids,
        **_gate_verdicts(
            {
                "interrupted_by": result["interrupted_by"],
                "completed_steps": result["completed_steps"],
                "planned_steps": len(tally.planned),
            }
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary={output_dir / 'run_summary.json'}")
    # Reported verdicts, never an exit code: the verdict of a multi-scene run is
    # the held-out curve, not a threshold on a single number.
    print(f"gates_passed={summary['gates_passed']}")


def _val_plans(args) -> list[StepPlan]:
    """Held-out scenes as plans, so eval and training share one scene path.

    These carry no ``track_indices``: there is no recorded draw to replay on the
    held-out side, so every eligible track is supervised. The window matches
    training's, which is what makes the two curves the same measurement.
    """

    if not args.val_scenes_file:
        return []
    names = json.loads(Path(args.val_scenes_file).read_text())
    cameras = tuple(args.val_cameras)
    times = tuple(
        range(0, min(args.observation_budget // len(cameras), args.max_time_indices) * args.stride, args.stride)
    )
    return [
        StepPlan(
            step=-1,
            seq_name=str(name),
            data_root=args.val_data_root,
            cameras=cameras,
            times=times,
            frame_start=0,
            seq_len=len(times) * args.stride,
            stride=args.stride,
            time_bound="budget",
            track_indices=(),
            scene_transform=None,
            depth_type="gt",
        )
        for name in names
    ]


if __name__ == "__main__":
    main()
