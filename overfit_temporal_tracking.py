#!/usr/bin/env python3
"""One-scene sparse temporal-tracking overfit for dumped MVTracker data.

This is deliberately a direct PyTorch smoke harness, not a general trainer.
It loads the released reconstruction model, freezes it with the existing
``temporal_tracking`` preset, fits one detached scene Sim(3) per forward, and
optimizes sparse postprocess-equivalent track positions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from arc.training import (
    build_anchor_correspondences,
    compose_tracking_loss,
    fit_scene_sim3,
    gather_query_anchor_points,
    load_dumped_kubric_scene,
    reconstruction_drift_report,
    reconstruction_shift_report,
    save_temporal_tracking_checkpoint,
    sparse_targets,
    sparse_tracking_loss,
    synchronized_consistency_stats,
    temporal_injection_report,
)


# Per freeze mode: (trainable tensor count, trainable parameters excluding the
# time-index embedding, whose row count is a flag). Measured on a meta-device
# Arc. A refactor that silently changes a freeze mask must fail here rather
# than quietly costing GPU weeks.
EXPECTED_TRAINABLE_SETS = {
    "temporal_tracking": (231, 314_551_588),
    "temporal_tracking_global_attention": (483, 711_268_132),
}
TIME_EMBEDDING_DIM = 1536
TIME_EMBEDDING_KEY = "backbone.pretrained.time_index_embedding.weight"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit 4RC sparse temporal tracking on one dumped scene"
    )
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--times", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max_time_indices", type=int, default=32)
    parser.add_argument(
        "--query_anchor",
        nargs="+",
        metavar="CAMERA:TIME",
        help=(
            "Observations that own a dense query field, as CAMERA:TIME pairs, in "
            "priority order; the first is primary and owns the scene Sim(3). "
            "Defaults to the first selected camera at the first selected time. "
            "A query point can only be anchored where it is the front surface at "
            "its own query time, so extra cameras reach queries occluded in the "
            "first and extra times reach queries that do not start at frame 0. "
            "Each query is supervised once, from the anchor it fits best. "
            "Anchoring at a time other than 0 needs the per-frame depth sidecar "
            "(RCMV_DUMP_DEPTH=1)."
        ),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--freeze_mode",
        choices=sorted(EXPECTED_TRAINABLE_SETS),
        default="temporal_tracking",
        help=(
            "Which freeze preset to train under. 'temporal_tracking' trains the "
            "embedding, MotionDecoder and track head; "
            "'temporal_tracking_global_attention' additionally unfreezes the 14 "
            "global-attention encoder blocks, the only place cross-view fusion "
            "can be learned."
        ),
    )
    parser.add_argument(
        "--time_embedding_init",
        choices=("zeros", "orthogonal"),
        default="orthogonal",
        help=(
            "How to re-seed the time-index embedding after loading. 'orthogonal' "
            "writes mutually orthogonal rows scaled to the checkpoint's time "
            "token, so the indices are distinct from step 0; 'zeros' keeps the "
            "constructor state, under which the table cannot grow into a signal "
            "within a short run (AdamW moves each weight by about lr per step)."
        ),
    )
    parser.add_argument(
        "--time_embedding_init_scale",
        type=float,
        default=0.1,
        help=(
            "Row norm of the orthogonal init as a fraction of ||time_token||. "
            "Large values inflate the initial loss by perturbing the pretrained "
            "conditioning, which makes the initial-loss gate trivially passable; "
            "the baseline gate exists for exactly that reason."
        ),
    )
    parser.add_argument(
        "--embedding_lr",
        type=float,
        default=None,
        help="Learning rate for the time-index embedding (default: --lr)",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=None,
        help=(
            "Learning rate for unfrozen encoder blocks (default: 0.1 * --lr). "
            "Only meaningful with --freeze_mode temporal_tracking_global_attention; "
            "kept low because the frozen geometry heads read the drifting features."
        ),
    )
    parser.add_argument(
        "--sync_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the dense synchronized-pair consistency term: same-time "
            "observation slots owe identical displacement fields at every pixel. "
            "0 (the default) skips the term entirely, keeping archived runs "
            "reproducible."
        ),
    )
    parser.add_argument(
        "--min_index_advantage",
        type=float,
        default=0.01,
        help=(
            "Minimum relative margin by which the final loss must beat the same "
            "model scored with one camera's time indices deterministically "
            "shuffled. This is what separates 'the decoder overfitted' from "
            "'the temporal indices were actually exploited'. Skipped when the "
            "window has no cross-camera synchronization to break."
        ),
    )
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("32", "16-mixed", "bf16-mixed"),
        default="bf16-mixed",
    )
    parser.add_argument("--huber_delta_m", type=float, default=0.05)
    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the DUSt3R-style confidence-weighted regression term. "
            "0 (the default) trains positions only, exactly as before the term "
            "existed, so archived runs stay reproducible. Note the two terms have "
            "very different natural scales -- the confidence term carries an "
            "alpha*log(conf) that is order hundreds when confidence is order "
            "hundreds, against a position loss of order 0.03 -- so 1.0 is not "
            "'equal weight'. Start small and read loss_breakdown in run_summary.json."
        ),
    )
    parser.add_argument(
        "--confidence_alpha",
        default="auto",
        help=(
            "Log-confidence regularizer weight. 'auto' (the default) resolves it to "
            "mean(initial confidence) * mean(initial error), which puts the term's "
            "optimum conf*=alpha/err at the released checkpoint's operating point so "
            "confidence is re-ordered without being level-shifted. Downstream "
            "occlusion thresholds confidence absolutely, so the level matters."
        ),
    )
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.01,
        help=(
            "Minimum relative drop in the like-for-like position loss required to "
            "pass. A zero-margin comparison is dominated by reconstruction drift, "
            "so require a real margin."
        ),
    )
    parser.add_argument(
        "--parse_only",
        action="store_true",
        help=(
            "Load and validate the selected MVTracker dump without requiring "
            "a checkpoint or CUDA"
        ),
    )
    parser.add_argument(
        "--eligibility_only",
        action="store_true",
        help=(
            "Build the sparse correspondences for the selected anchors, print the "
            "eligibility split and write eligibility.json, then stop. Needs "
            "neither a checkpoint nor CUDA, so the recovery a given anchor set "
            "buys is measurable without a GPU allocation."
        ),
    )
    return parser


def _parse_query_anchor(value: str) -> tuple[int, int]:
    """Parse one ``CAMERA:TIME`` anchor.

    Rejected explicitly rather than coerced: a silently mis-parsed anchor would
    supervise the wrong observation and still produce plausible numbers.
    """

    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"--query_anchor entries must look like CAMERA:TIME, got {value!r}"
        )
    try:
        camera, time_index = (int(part) for part in parts)
    except ValueError:
        raise ValueError(
            f"--query_anchor entries must be integers CAMERA:TIME, got {value!r}"
        ) from None
    if camera < 0 or time_index < 0:
        raise ValueError(
            f"--query_anchor CAMERA and TIME must be non-negative, got {value!r}"
        )
    return camera, time_index


def _resolve_query_anchors(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    """The anchor list a run will use, defaulted and validated."""

    if not args.query_anchor:
        return ((args.cameras[0], args.times[0]),)
    anchors = tuple(_parse_query_anchor(value) for value in args.query_anchor)
    seen = set()
    for camera, time_index in anchors:
        if camera not in args.cameras:
            raise ValueError(
                f"--query_anchor camera {camera} is not in --cameras "
                f"{list(args.cameras)}"
            )
        if time_index not in args.times:
            raise ValueError(
                f"--query_anchor time {time_index} is not in --times "
                f"{list(args.times)}"
            )
        if (camera, time_index) in seen:
            raise ValueError(
                f"--query_anchor {camera}:{time_index} is listed more than once"
            )
        seen.add((camera, time_index))
    return anchors


def _validate_args(args: argparse.Namespace) -> None:
    if not args.cameras:
        raise ValueError("--cameras must contain at least one camera")
    if len(set(args.cameras)) != len(args.cameras):
        raise ValueError("--cameras must not contain duplicates")
    if any(camera < 0 for camera in args.cameras):
        raise ValueError("--cameras must contain non-negative integers")
    if not args.times:
        raise ValueError("--times must contain at least one frame")
    if len(set(args.times)) != len(args.times):
        raise ValueError("--times must not contain duplicates")
    if any(time < 0 for time in args.times):
        raise ValueError("--times must contain non-negative integers")
    if any(later <= earlier for earlier, later in zip(args.times, args.times[1:])):
        raise ValueError("--times must be strictly increasing")
    if args.max_time_indices <= 0:
        raise ValueError("--max_time_indices must be positive")
    if len(args.times) > args.max_time_indices:
        raise ValueError(
            f"Selected {len(args.times)} semantic times, but "
            f"--max_time_indices={args.max_time_indices}"
        )

    _resolve_query_anchors(args)

    if args.parse_only or args.eligibility_only:
        return
    if not args.checkpoint_dir:
        raise ValueError(
            "--checkpoint_dir is required unless --parse_only or "
            "--eligibility_only is set"
        )
    if not args.output_dir:
        raise ValueError(
            "--output_dir is required unless --parse_only or "
            "--eligibility_only is set"
        )
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(args.huber_delta_m) or args.huber_delta_m <= 0:
        raise ValueError("--huber_delta_m must be finite and positive")
    if not math.isfinite(args.confidence_weight) or args.confidence_weight < 0:
        raise ValueError("--confidence_weight must be finite and non-negative")
    _parse_confidence_alpha(args.confidence_alpha)
    if not math.isfinite(args.min_improvement) or not 0 <= args.min_improvement < 1:
        raise ValueError("--min_improvement must be finite and in [0, 1)")
    if (
        not math.isfinite(args.time_embedding_init_scale)
        or args.time_embedding_init_scale <= 0
    ):
        raise ValueError("--time_embedding_init_scale must be finite and positive")
    if args.embedding_lr is not None and (
        not math.isfinite(args.embedding_lr) or args.embedding_lr <= 0
    ):
        raise ValueError("--embedding_lr must be finite and positive")
    if args.encoder_lr is not None and (
        not math.isfinite(args.encoder_lr) or args.encoder_lr <= 0
    ):
        raise ValueError("--encoder_lr must be finite and positive")
    if not math.isfinite(args.sync_weight) or args.sync_weight < 0:
        raise ValueError("--sync_weight must be finite and non-negative")
    if (
        not math.isfinite(args.min_index_advantage)
        or not 0 <= args.min_index_advantage < 1
    ):
        raise ValueError("--min_index_advantage must be finite and in [0, 1)")
    # An anchor at a time other than 0 needs the per-frame depth sidecar, which
    # only the loaded scene can report on. DumpedKubricScene.surface_depth_map
    # raises there, naming RCMV_DUMP_DEPTH=1, so the guard sits where the fact
    # is known rather than being guessed from the flags.


def _parse_confidence_alpha(value) -> float | None:
    """Return None for 'auto', otherwise the explicit positive float."""

    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"--confidence_alpha must be 'auto' or a float, got {value!r}"
        ) from None
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("--confidence_alpha must be finite and positive")
    return alpha


def _validate_scene_layout(scene) -> None:
    expected_observations = len(scene.cameras) * len(scene.times)
    if scene.num_observations != expected_observations:
        raise RuntimeError(
            f"Expected {expected_observations} camera/time observations, got "
            f"{scene.num_observations}"
        )
    expected_times = tuple(range(len(scene.times))) * len(scene.cameras)
    if scene.time_indices != expected_times:
        raise RuntimeError(
            f"Unexpected camera-major time mapping {scene.time_indices}; "
            f"expected {expected_times}"
        )
    expected_pairs = [
        (camera, original_time)
        for camera in scene.cameras
        for original_time in scene.times
    ]
    actual_pairs = [
        (observation.camera, observation.original_time)
        for observation in scene.observations
    ]
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            f"Unexpected camera/time observation order {actual_pairs}; "
            f"expected {expected_pairs}"
        )


def _print_parse_report(scene) -> None:
    query = scene.observations[scene.query_observation_slot]
    print(f"scene={scene.name}")
    print(f"camera_ids={list(scene.camera_ids)}")
    print(f"view_ids={scene.view_ids.tolist()}")
    print(f"cameras={list(scene.cameras)}")
    print(f"original_times={list(scene.times)}")
    print(f"time_indices={list(scene.time_indices)}")
    print(f"observations={scene.num_observations}")
    print(
        "query_observation="
        f"slot {query.slot}, camera {query.camera_id}, "
        f"original_time {query.original_time}"
    )
    print(
        "query_anchors="
        + " ".join(
            f"{camera}:{time_index}" for camera, time_index in scene.query_anchors
        )
        + f" (slots {list(scene.anchor_observation_slots)})"
    )
    print(
        "time_varying_depth="
        + (
            f"{scene.depth_sidecar_path} {tuple(scene.depth.shape)}"
            if scene.has_time_varying_depth
            else "absent (depth0 only)"
        )
    )
    for observation, view in zip(scene.observations, scene.views):
        print(
            f"slot={observation.slot} camera={observation.camera} "
            f"original_time={observation.original_time} "
            f"time_index={observation.semantic_time_index} "
            f"image_shape={tuple(view['img'].shape)} path={observation.path}"
        )
    print(
        "metadata_shapes="
        f"query_points{tuple(scene.query_points.shape)} "
        f"trajectories{tuple(scene.trajectories_world.shape)} "
        f"visibility{tuple(scene.visibility.shape)} "
        f"intrinsics{tuple(scene.intrinsics.shape)} "
        f"extrinsics{tuple(scene.extrinsics_world_to_camera.shape)} "
        f"depth0{tuple(scene.depth0.shape)}"
    )
    print(f"track_upscaling_factor={scene.track_upscaling_factor}")
    print("PASS mvtracker dump parsing")


def _autocast_context(precision: str):
    if precision == "32":
        return nullcontext()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _gradient_norm(parameters) -> float:
    squared_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        squared_norm += parameter.grad.detach().float().square().sum()
    return float(torch.sqrt(squared_norm).item()) if found else 0.0


def _assert_frozen_gradients_absent(model) -> None:
    offenders = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(
            "Frozen parameters received gradients: " + ", ".join(offenders[:10])
        )


def _assert_trainable_gradients_finite(model) -> None:
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


def _move_views_to_cuda(views: list[dict]) -> None:
    for view in views:
        for key in ("img", "time_index", "track_query_idx"):
            view[key] = view[key].to("cuda", non_blocking=True)


def _shuffled_index_views(scene) -> list[dict] | None:
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


def _measure_temporal_injection(model, views: list[dict], precision: str):
    """Transport of the index signal through the encoder, on the real weights.

    Two backbone-only forwards over the same images — without and with time
    indices — compared tap by tap.  This answers, on the loaded checkpoint,
    the question the synthetic-model experiment answered for the wiring: how
    much the tapped time token (the motion decoder's conditioning) and the
    patch tokens move, and whether the deltas cluster by shared index.
    """

    images, _, time_indices = model._preprocess_input(views)
    if time_indices is None:
        return None
    with torch.no_grad(), _autocast_context(precision):
        baseline_taps, _ = model.backbone(
            images, ref_view_strategy="first", time_indices=None
        )
        indexed_taps, _ = model.backbone(
            images, ref_view_strategy="first", time_indices=time_indices
        )
    return temporal_injection_report(
        baseline_taps,
        indexed_taps,
        time_indices[0].detach().cpu(),
    )


def _build_optimizer(model, args) -> tuple[torch.optim.AdamW, dict[str, float], list]:
    """AdamW over named parameter groups with per-group learning rates.

    The embedding gets its own group so its rate is a free knob, and unfrozen
    encoder blocks get a lower default because the frozen geometry heads read
    the features those blocks produce.  Weight decay stays 0 everywhere: no
    regularizer is part of this proof, and decoupled decay must not drag the
    unsupervised confidence output around.
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
        "decoder": args.lr,
        "embedding": args.lr if args.embedding_lr is None else args.embedding_lr,
        "encoder_blocks": (
            args.lr * 0.1 if args.encoder_lr is None else args.encoder_lr
        ),
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


def _cut_features(feats):
    """Detach the backbone taps into leaves, returning the cut and its pairs.

    This is the encoder/head boundary that lets several anchors be supervised
    without holding several track-head graphs at once.  Each anchor's head pass
    runs on the leaves and backwards immediately, freeing its own graph and
    leaving its gradient on the cut; one
    ``torch.autograd.backward(originals, leaf_grads)`` afterwards pushes the
    summed gradient through the encoder exactly once.  Summing gradients at a
    cut is the chain rule, so this is identical to one combined backward -- not
    an approximation -- while the encoder is neither re-run nor re-differentiated
    per anchor.
    """

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def cut(value):
        if torch.is_tensor(value):
            if not value.requires_grad:
                return value
            leaf = value.detach().requires_grad_(True)
            pairs.append((value, leaf))
            return leaf
        if isinstance(value, (list, tuple)):
            return type(value)(cut(item) for item in value)
        return value

    return cut(feats), pairs


def _backward_through_cut(pairs) -> None:
    """Push the anchors' accumulated gradients back through the encoder once."""

    if not pairs:
        return
    originals = [original for original, _ in pairs]
    gradients = [
        torch.zeros_like(leaf) if leaf.grad is None else leaf.grad
        for _, leaf in pairs
    ]
    torch.autograd.backward(originals, gradients)


def _encode_and_reconstruct(model, views):
    """The per-step work that does not depend on which frame is the query."""

    images, _, time_indices = model._preprocess_input(views)
    feats = model.encode_features(images, time_indices=time_indices)
    return images, feats, model.reconstruct(feats, images)


def _anchor_tracks(model, feats, images, scene, anchor_index):
    """One anchor's dense field, shaped as the Q=1 raw dict the loss expects."""

    slot = scene.anchor_observation_slots[anchor_index]
    track, track_conf = model.track_for_query(feats, images, slot)
    return {
        "track_multi": track[:, None],
        "conf_track_multi": track_conf[:, None],
        "track_query_idx": torch.tensor([slot], device=track.device),
    }


def _anchor_sample_counts(scene, correspondences, anchor_count: int) -> list[int]:
    """Supervised sample count per anchor, from the scene alone.

    These are the weights that make per-anchor backward equal one combined
    ``reduction="mean"``.  They come from ``sparse_targets``, the same masking
    the loss itself applies, so the two cannot drift apart; and because nothing
    in that mask reads a prediction, they are fixed for the whole run.
    """

    counts = []
    for anchor_index in range(anchor_count):
        anchor = correspondences.select_query_slot(anchor_index)
        if anchor.count == 0:
            counts.append(0)
            continue
        _, _, _, mask = sparse_targets(scene, anchor)
        counts.append(int(mask.sum().item()))
    return counts


def _anchor_confidence_counts(
    scene,
    correspondences,
    anchor_count: int,
) -> list[int]:
    """Confidence-term sample count per anchor.

    The confidence term deliberately does *not* mask on visibility -- occluded
    samples are where the error is large, which is the signal it exists to learn
    -- so it reduces over a strictly larger set than the position term and needs
    its own weights.  Reusing the position weights here would make the combined
    objective differ from the stacked-Q one by the ratio between the two masks.

    The remaining predicates in that mask are prediction-dependent
    (``predicted_finite``, ``confidence_finite``), so this is exact whenever
    nothing goes non-finite.  When something does, the run says so loudly:
    the drop is counted by cause in ``confidence_dropped`` and warned about.
    """

    counts = []
    for anchor_index in range(anchor_count):
        anchor = correspondences.select_query_slot(anchor_index)
        if anchor.count == 0:
            counts.append(0)
            continue
        _, _, finite, _ = sparse_targets(scene, anchor)
        counts.append(int(finite.sum().item()))
    return counts


def _weighted_anchor_total(
    result,
    *,
    position_weight: float,
    confidence_weight: float,
    sync_weight: float,
) -> torch.Tensor:
    """One anchor's contribution to the combined objective.

    ``compose_tracking_loss`` is linear in its terms, so backwarding each
    anchor's weighted total in turn accumulates exactly the gradient of the sum
    -- which is the combined loss, given the position and confidence weights are
    that anchor's share of the supervised samples and the sync weight is its
    share of the anchors.

    One scope limit on that equivalence.  The sync share is
    ``1 / active_anchor_count``, not ``1 / anchor_count``, so the objective
    reproduced is a stacked forward over the **active** anchors.  A declared
    anchor that supervises nothing is skipped entirely, and against the
    ``Q = anchor_count`` reference ``_evaluate`` computes, its dense field would
    have contributed a sync term this does not include.  Weighting by the active
    count is the defensible choice -- an anchor with no supervised query still
    has a displacement field, but nothing here has any evidence about it -- but
    it does mean the two numbers differ in that case.

    Terms are inserted in the same order ``sparse_tracking_loss`` uses, because
    ``compose_tracking_loss`` sums in insertion order and float addition is not
    associative: with a single anchor at weight 1.0 this must reproduce that
    function's own ``total_loss`` bit for bit.
    """

    terms = {"position": result.loss}
    weights = {"position": position_weight}
    if result.sync_loss is not None:
        terms["sync"] = result.sync_loss
        weights["sync"] = sync_weight
    if result.confidence_loss is not None:
        terms["confidence"] = result.confidence_loss
        weights["confidence"] = confidence_weight
    total, _ = compose_tracking_loss(terms, weights)
    return total


def _accumulate(total: float | None, value, weight: float) -> float | None:
    """Running weighted sum of a reported scalar across anchors."""

    if value is None:
        return total
    contribution = float(value.detach().item()) * float(weight)
    return contribution if total is None else total + contribution


def _tracking_only(raw_predictions: dict, keep_confidence: bool = False) -> dict:
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


def _breakdown_to_floats(breakdown) -> dict[str, float] | None:
    """`compose_tracking_loss` hands back detached tensors; JSON needs floats."""

    if breakdown is None:
        return None
    return {name: float(value.item()) for name, value in breakdown.items()}


def _warn_about_dropped_confidence_samples(dropped, sample_count) -> None:
    """A silently shrinking confidence set is the failure mode worth shouting about."""

    if not dropped or dropped.get("total", 0) == 0:
        return
    total = dropped["total"]
    print(
        f"WARNING: {total} sparse sample(s) were dropped from the confidence term as "
        "non-finite "
        f"(target={dropped['target_nonfinite']}, "
        f"prediction={dropped['prediction_nonfinite']}, "
        f"confidence={dropped['confidence_nonfinite']}); "
        f"{sample_count} remain. `expp1` is 1+exp(x) and overflows in BF16, so a "
        "large count here means the confidence figures below describe a subset."
    )


def _implied_optimal_confidence(alpha, diagnostics) -> float | None:
    """Where the confidence term wants confidence to sit, given the current error."""

    if alpha is None or not diagnostics:
        return None
    mean_error = diagnostics.get("mean_error")
    if mean_error is None or mean_error <= 0:
        return None
    return float(alpha) / float(mean_error)


def _confidence_gradient_norms(model) -> dict[str, float]:
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


def _exit_criteria_failure(
    *,
    baseline_loss: float,
    initial_loss: float,
    final_loss: float,
    final_shuffled_loss: float | None,
    embedding_change: float,
    min_improvement: float,
    min_index_advantage: float,
    confidence_weight: float = 0.0,
) -> str | None:
    """Return why the run failed its exit criteria, or None if it passed.

    All losses must be like-for-like numbers (baseline alignment, baseline
    anchors).  A bare ``final < initial`` is dominated by reconstruction drift,
    so relative margins are required.

    Three references, each closing a different loophole.  ``initial_loss`` (the
    initialized-embedding start) proves training descended, but a disruptive
    init inflates it, so beating it can mean merely recovering self-inflicted
    damage.  ``baseline_loss`` (the released checkpoint, zero embedding) closes
    that: the run must end better than where the released model started.
    ``final_shuffled_loss`` (the trained model scored with one camera's indices
    shuffled) closes the remaining one: both loss gates are beatable by the
    314M-parameter decoder overfitting the scene with the embedding as dead
    weight, and only degradation under shuffled indices shows the indices were
    exploited.  It is None exactly when the window has no cross-camera
    synchronization to break, which skips that check.

    The gate stays on the *position* loss even when the optimizer is descending a
    total that includes the confidence term.  That is deliberate: the question this
    harness answers is whether supervising sparse tracks moves tracking, and a
    confidence term that buys calibration by giving up track accuracy should read
    as a failure here rather than be averaged away.
    """

    if embedding_change == 0:
        return "Temporal embedding did not change"
    required_loss = initial_loss * (1.0 - min_improvement)
    if not final_loss < required_loss:
        reason = (
            "One-scene overfit did not reduce the like-for-like position loss by "
            f"at least {min_improvement:.4%}: initial={initial_loss:.8f}, "
            f"final={final_loss:.8f}, required<{required_loss:.8f}"
        )
        if confidence_weight > 0:
            # Without this the reader sees a position-loss failure and has no
            # reason to suspect the term they just switched on.
            reason += (
                f". Note --confidence_weight={confidence_weight:.6g} is set, so the "
                "optimizer descended position + weighted confidence while this gate "
                "judges position alone; compare final_loss_breakdown before "
                "concluding that tracking itself regressed"
            )
        return reason
    required_baseline = baseline_loss * (1.0 - min_improvement)
    if not final_loss < required_baseline:
        return (
            "One-scene overfit did not beat the zero-embedding baseline by "
            f"at least {min_improvement:.4%}: baseline={baseline_loss:.8f}, "
            f"final={final_loss:.8f}, required<{required_baseline:.8f}. "
            "Improvement over the initialized start alone can be recovery "
            "from a disruptive init rather than learning"
        )
    if final_shuffled_loss is not None:
        required_shuffled = final_shuffled_loss * (1.0 - min_index_advantage)
        if not final_loss < required_shuffled:
            return (
                "Shuffling one camera's time indices did not hurt the trained "
                f"model by at least {min_index_advantage:.4%}: "
                f"indexed={final_loss:.8f}, shuffled={final_shuffled_loss:.8f}, "
                f"required<{required_shuffled:.8f}. The loss improvement is "
                "therefore attributable to decoder adaptation, not to the "
                "temporal indices"
            )
    return None


def _confidence_stats(raw_predictions) -> dict[str, float] | None:
    """Summarize the track head's confidence channel.

    Nothing supervises it: ``sparse_tracking_loss`` is a position-only Huber, and
    xyz and confidence are split off the *same* final conv, so the channel gets a
    zero gradient while its trunk moves.  ``score_joint.py`` thresholds this
    confidence absolutely to derive occlusion, which feeds OA and AJ, so an
    unsupervised mean shift is not harmless.  Log it so drift is attributable.
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


def _evaluate(
    model,
    scene,
    correspondences,
    precision,
    huber_delta_m,
    initial_alignment,
    initial_query_anchors,
    confidence_weight=0.0,
    confidence_alpha=None,
    *,
    sync_weight=0.0,
    sync_metric_scale,
    shuffled_views,
):
    """Score the trained model two ways, plus the control arms.

    The refit numbers use a Sim(3) and query anchors re-derived from the trained
    model, matching how inference would score it.  The like-for-like numbers
    reuse the *initial* alignment and anchors, so the only thing that differs
    from ``initial_loss`` is the trained parameters.  The time embedding is
    injected at ``alt_start`` and every head-feeding out-layer is downstream of
    it, so a refit moves even though the reconstruction weights are frozen;
    only the like-for-like number isolates the tracking change.

    Also collected: the synchronized-pair consistency stats, the
    reconstruction-vs-ground-truth drift report under the refit alignment (how
    inference would use the geometry), and — when ``shuffled_views`` is not
    None — a like-for-like score of the same trained model with one camera's
    indices shuffled (position-only).
    """

    model.eval()
    with torch.no_grad(), _autocast_context(precision):
        raw = model(scene.views, force_no_output_conversion=True)
        alignment, alignment_report = fit_scene_sim3(raw, scene)
        query_anchors = gather_query_anchor_points(
            raw,
            scene,
            correspondences,
        )
        confidence_stats = _confidence_stats(raw)
        sync_stats = synchronized_consistency_stats(
            raw["track_multi"],
            scene.slot_time_indices,
            metric_scale=sync_metric_scale,
        )
        drift = reconstruction_drift_report(raw, scene, alignment)
        raw = _tracking_only(raw, keep_confidence=confidence_weight > 0)
        refit = sparse_tracking_loss(
            raw,
            scene,
            correspondences,
            alignment,
            query_anchors,
            huber_delta_m=huber_delta_m,
            confidence_weight=confidence_weight,
            confidence_alpha=confidence_alpha,
            sync_weight=sync_weight,
        )
        like_for_like = sparse_tracking_loss(
            raw,
            scene,
            correspondences,
            initial_alignment,
            initial_query_anchors,
            huber_delta_m=huber_delta_m,
            confidence_weight=confidence_weight,
            confidence_alpha=confidence_alpha,
            sync_weight=sync_weight,
        )
        shuffled_loss = None
        shuffled_error = None
        if shuffled_views is not None:
            shuffled_raw = model(shuffled_views, force_no_output_conversion=True)
            shuffled_result = sparse_tracking_loss(
                _tracking_only(shuffled_raw),
                scene,
                correspondences,
                initial_alignment,
                initial_query_anchors,
                huber_delta_m=huber_delta_m,
            )
            shuffled_loss = float(shuffled_result.loss.item())
            shuffled_error = float(shuffled_result.metric_error.item())
            del shuffled_raw, shuffled_result
    return {
        "loss_shuffled": shuffled_loss,
        "metric_error_shuffled_m": shuffled_error,
        "sync_consistency": sync_stats,
        "reconstruction_drift": drift,
        "loss_refit": float(refit.loss.item()),
        "metric_error_refit_m": float(refit.metric_error.item()),
        "loss": float(like_for_like.loss.item()),
        "metric_error_m": float(like_for_like.metric_error.item()),
        "alignment": alignment,
        "alignment_report": alignment_report,
        "confidence": confidence_stats,
        "confidence_loss": (
            None
            if like_for_like.confidence_loss is None
            else float(like_for_like.confidence_loss.item())
        ),
        "confidence_sample_count": like_for_like.confidence_sample_count,
        "confidence_dropped": like_for_like.confidence_dropped,
        "confidence_diagnostics": like_for_like.diagnostics,
        "loss_breakdown": _breakdown_to_floats(like_for_like.loss_breakdown),
    }


def _print_eligibility(report: dict) -> None:
    """The split, with N stated everywhere a count or a share is printed.

    The overfit harness runs the benchmark protocol's 512 queries while the
    training dump draws 2048, so a percentage without its denominator is a
    number that can be read against the wrong N.
    """

    total = report["total_query_count"]
    eligible = report["eligible_query_count"]
    if total == 0:
        raise RuntimeError(
            f"Scene carries no query points at all; nothing to anchor. Report: {report}"
        )
    print(
        f"eligible_queries={eligible}/{total} "
        f"({eligible / total:.1%} of total_query_count={total}); "
        f"supervised_pairs={report['supervised_pair_count']} across "
        f"{report['anchor_count']} anchor(s)"
    )
    rejected = report["rejected"]
    accounted = eligible + sum(rejected.values())
    for reason, count in rejected.items():
        print(f"  rejected.{reason}={count}/{total} ({count / total:.1%})")
    print(f"  accounted={accounted}/{total}")
    if accounted != total:
        raise RuntimeError(
            f"Eligibility split accounts for {accounted} of {total} queries; the "
            "per-reason counts must be exclusive and exhaustive"
        )
    for anchor in report["per_anchor"]:
        print(
            f"  anchor[{anchor['anchor_index']}] camera={anchor['camera']} "
            f"time={anchor['time']} slot={anchor['observation_slot']}: "
            f"eligible={anchor['eligible']}/{anchor['considered']} "
            f"assigned={anchor['assigned']} "
            f"sole_anchor={anchor['sole_anchor']} "
            f"rejected={anchor['rejected']}"
        )
        # The label-quality return on this anchor, in metres. depth_error is how
        # far the anchor pixel's surface sits from the query point; margin is how
        # much better than the runner-up anchor, so it is a difference of depth
        # errors and not a count.
        errors = anchor["assigned_depth_error_m"]
        margin = anchor["contested_depth_error_margin_m"]
        print(
            "    label_quality_m: "
            + (
                "assigned_depth_error median=n/a p95=n/a"
                if errors is None
                else (
                    f"assigned_depth_error median={errors['median']:.4f} "
                    f"p95={errors['p95']:.4f}"
                )
            )
            + f"; contested_assigned={anchor['contested_assigned']}"
            + (
                " (every win uncontested)"
                if margin is None
                else f" margin_over_runner_up median={margin:.4f}"
            )
        )


def _report_eligibility_only(scene, args) -> None:
    """Build correspondences and report the split, without CUDA or a checkpoint."""

    _, eligibility = build_anchor_correspondences(scene)
    print(f"scene={scene.name}")
    print(
        "query_anchors="
        + " ".join(
            f"{camera}:{time_index}" for camera, time_index in scene.query_anchors
        )
    )
    print(
        "time_varying_depth="
        + (
            f"{scene.depth_sidecar_path}"
            if scene.has_time_varying_depth
            else "absent (depth0 only)"
        )
    )
    _print_eligibility(eligibility)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "eligibility.json"
        path.write_text(
            json.dumps(
                {
                    "scene": scene.name,
                    "cameras": list(scene.camera_ids),
                    "times": list(scene.times),
                    "query_anchors": [list(pair) for pair in scene.query_anchors],
                    "time_varying_depth": _time_varying_depth_report(scene),
                    "eligibility": eligibility,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"eligibility={path}")
    print("PASS eligibility report")


def _time_varying_depth_report(scene) -> dict:
    return {
        "present": scene.has_time_varying_depth,
        "path": (
            None
            if scene.depth_sidecar_path is None
            else str(scene.depth_sidecar_path)
        ),
        "shape": (
            None if scene.depth is None else list(scene.depth.shape)
        ),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    scene = load_dumped_kubric_scene(
        args.data_root,
        args.scene,
        cameras=args.cameras,
        times=args.times,
        query_anchors=_resolve_query_anchors(args),
        verbose=True,
    )
    _validate_scene_layout(scene)
    if args.parse_only:
        _print_parse_report(scene)
        return
    if args.eligibility_only:
        _report_eligibility_only(scene, args)
        return
    query_observation = scene.observations[scene.query_observation_slot]
    print(
        "input_layout="
        f"{len(scene.cameras)} cameras x {len(scene.times)} times = "
        f"{scene.num_observations} observations; "
        f"time_indices={list(scene.time_indices)}; "
        f"query=(camera {query_observation.camera_id}, "
        f"time {query_observation.original_time}, "
        f"slot {query_observation.slot})"
    )
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for the released 4RC model")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from arc.models.arc import Arc

    model = Arc.from_pretrained(
        args.checkpoint_dir,
        max_time_indices=args.max_time_indices,
    ).to("cuda")
    model.set_freeze(args.freeze_mode)
    report = model.get_trainable_parameter_report()
    expected_tensors, expected_non_embedding = EXPECTED_TRAINABLE_SETS[
        args.freeze_mode
    ]
    expected_parameter_count = (
        expected_non_embedding + args.max_time_indices * TIME_EMBEDDING_DIM
    )
    if (
        report["tensor_count"] != expected_tensors
        or report["parameter_count"] != expected_parameter_count
    ):
        raise RuntimeError(
            f"Unexpected {args.freeze_mode} parameter set: "
            f"{report['tensor_count']} tensors / "
            f"{report['parameter_count']} parameters; expected "
            f"{expected_tensors} / {expected_parameter_count}"
        )
    print(
        "trainable="
        f"{report['tensor_count']} tensors / {report['parameter_count']} "
        f"parameters ({args.freeze_mode})"
    )

    _move_views_to_cuda(scene.views)

    # ---- Baseline: exact released-checkpoint behaviour. The embedding is
    # still at its constructor zeros here, so this forward -- indexed views
    # included -- is bit-identical to a no-index forward of the base model.
    # Every like-for-like reference (alignment, anchors, confidence alpha)
    # derives from it, and the baseline loss is the number the trained model
    # must beat for "temporal indexing helped" to mean anything.
    model.eval()
    with torch.no_grad(), _autocast_context(args.precision):
        baseline_raw = model(scene.views, force_no_output_conversion=True)
    if baseline_raw["track_multi"].shape[2] != scene.num_observations:
        raise RuntimeError(
            "Baseline output observation axis does not match the selected inputs"
        )
    initial_alignment, initial_alignment_report = fit_scene_sim3(
        baseline_raw,
        scene,
    )
    correspondences, eligibility = build_anchor_correspondences(scene)
    _print_eligibility(eligibility)
    anchor_count = len(scene.anchor_observation_slots)
    per_anchor_correspondences = [
        correspondences.select_query_slot(anchor_index)
        for anchor_index in range(anchor_count)
    ]
    # The query pointmap anchors are gathered once from the scene-level
    # correspondences; these masks keep each anchor's slice of them aligned with
    # the rebased correspondences its head pass is scored against.
    per_anchor_rows = [
        correspondences.anchor_rows(anchor_index)
        for anchor_index in range(anchor_count)
    ]
    anchor_sample_counts = _anchor_sample_counts(
        scene,
        correspondences,
        anchor_count,
    )
    total_anchor_samples = sum(anchor_sample_counts)
    if total_anchor_samples == 0:
        raise RuntimeError(
            "No anchor contributes a supervised sample, so there is nothing to "
            "train on. The split above says why: of "
            f"{eligibility['total_query_count']} queries, "
            f"{eligibility['rejected']}. Anchoring at a time other than 0 needs "
            "the per-frame depth sidecar; anchoring in another camera is what "
            "reaches queries occluded in the first."
        )
    # Each anchor's share of the supervised samples. Weighting by this and
    # backwarding per anchor is exactly one combined reduction="mean"; the
    # counts come from the same masking the loss applies and never move,
    # because nothing in that mask reads a prediction.
    anchor_weights = [
        count / total_anchor_samples for count in anchor_sample_counts
    ]
    active_anchor_count = sum(1 for weight in anchor_weights if weight > 0)
    # The confidence term reduces over a different (larger) set than the
    # position term, so it needs its own shares; see _anchor_confidence_counts.
    anchor_confidence_counts = _anchor_confidence_counts(
        scene,
        correspondences,
        anchor_count,
    )
    total_confidence_samples = sum(anchor_confidence_counts)
    anchor_confidence_weights = [
        count / total_confidence_samples if total_confidence_samples else 0.0
        for count in anchor_confidence_counts
    ]
    print(
        "anchor_sample_counts="
        f"{anchor_sample_counts} (total {total_anchor_samples}); "
        f"active_anchors={active_anchor_count}/{anchor_count}"
    )
    initial_query_anchors = gather_query_anchor_points(
        baseline_raw,
        scene,
        correspondences,
    )
    confidence_enabled = args.confidence_weight > 0
    requested_alpha = _parse_confidence_alpha(args.confidence_alpha)
    sync_metric_scale = (
        float(initial_alignment.scale.item()) * scene.track_upscaling_factor
    )
    baseline_result = sparse_tracking_loss(
        _tracking_only(baseline_raw, keep_confidence=confidence_enabled),
        scene,
        correspondences,
        initial_alignment,
        initial_query_anchors,
        huber_delta_m=args.huber_delta_m,
        confidence_weight=args.confidence_weight,
        confidence_alpha=requested_alpha,
        sync_weight=args.sync_weight,
    )
    baseline_confidence = _confidence_stats(baseline_raw)
    baseline_loss = float(baseline_result.loss.item())
    baseline_error = float(baseline_result.metric_error.item())
    baseline_sync = synchronized_consistency_stats(
        baseline_raw["track_multi"],
        scene.slot_time_indices,
        metric_scale=sync_metric_scale,
    )
    baseline_drift = reconstruction_drift_report(
        baseline_raw,
        scene,
        initial_alignment,
    )
    # Resolve alpha once, here, from the *baseline* forward -- the released
    # checkpoint's operating point, which is what the 'auto' rule promises; the
    # initialized-embedding forward below is already perturbed. Re-resolving
    # every step would chase the confidence the term is itself moving, and the
    # optimum would never settle.
    #
    # Resolving it from an eval-mode forward and then applying it to train-mode
    # steps is safe, and checked rather than assumed: nothing on the path is
    # train/eval sensitive. The DPT head builds no BatchNorm (`bn=False` at
    # dpt_head.py:321, and ResidualConvUnit leaves norm1/norm2 None regardless),
    # MotionDecoder has no BatchNorm or dropout, and the ViT's drop_path_rate is
    # 0.0 and never overridden. train() only turns on activation checkpointing and
    # the one-frame track-head chunk, both value-preserving.
    confidence_alpha = baseline_result.confidence_alpha
    if confidence_enabled:
        print(
            "confidence_term="
            f"weight {args.confidence_weight:.6g}, alpha {confidence_alpha:.6g} "
            f"({'auto' if requested_alpha is None else 'explicit'}), "
            f"{baseline_result.confidence_sample_count} samples "
            f"(position uses {baseline_result.sample_count})"
        )
        _warn_about_dropped_confidence_samples(
            baseline_result.confidence_dropped,
            baseline_result.confidence_sample_count,
        )
    # Kept for the step-0 reconstruction-shift report below.
    baseline_depth = baseline_raw["depth"].detach()
    baseline_pose_enc = baseline_raw["pose_enc"].detach()
    # initial_alignment and initial_query_anchors are deliberately kept: the
    # pass/fail gate re-scores the trained model against this same alignment
    # and these same anchors.
    del baseline_raw, baseline_result
    print(
        "alignment="
        f"{initial_alignment_report['pair_count']} pairs, "
        f"{initial_alignment_report['median_residual_metric']:.6f} m median residual, "
        f"scale={initial_alignment_report['scale']:.6f}"
    )
    print(f"eligible_queries={correspondences.count}")
    print(f"baseline_position_loss={baseline_loss:.8f}")

    # ---- Re-seed the embedding, then measure what the injection does before
    # any training step has run.
    embedding = model.backbone.pretrained.time_index_embedding
    embedding_target_row_norm = None
    if args.time_embedding_init != "zeros":
        if TIME_EMBEDDING_KEY not in model.consumed_legacy_missing_keys:
            raise RuntimeError(
                "Refusing to reinitialize a time-index embedding that was "
                "loaded from the checkpoint rather than zero-filled; pass "
                "--time_embedding_init zeros to keep the loaded table"
            )
        model.backbone.pretrained.reinitialize_time_index_embedding(
            args.time_embedding_init,
            scale=args.time_embedding_init_scale,
            generator=torch.Generator().manual_seed(args.seed),
        )
        embedding_target_row_norm = float(
            args.time_embedding_init_scale
            * model.backbone.pretrained.time_token.detach().float().norm().item()
        )
    # Snapshot AFTER the reinit: the embedding-moved gate compares against
    # this, and a pre-reinit snapshot would count the reinit itself as change.
    initial_embedding = embedding.weight.detach().clone()
    initial_embedding_norm = float(initial_embedding.float().norm().item())

    with torch.no_grad(), _autocast_context(args.precision):
        initial_raw = model(scene.views, force_no_output_conversion=True)
    initial_result = sparse_tracking_loss(
        _tracking_only(initial_raw, keep_confidence=confidence_enabled),
        scene,
        correspondences,
        initial_alignment,
        initial_query_anchors,
        huber_delta_m=args.huber_delta_m,
        confidence_weight=args.confidence_weight,
        confidence_alpha=confidence_alpha,
        sync_weight=args.sync_weight,
    )
    initial_confidence = _confidence_stats(initial_raw)
    initial_loss = float(initial_result.loss.item())
    initial_error = float(initial_result.metric_error.item())
    initial_sync = synchronized_consistency_stats(
        initial_raw["track_multi"],
        scene.slot_time_indices,
        metric_scale=sync_metric_scale,
    )
    initial_confidence_loss = (
        None
        if initial_result.confidence_loss is None
        else float(initial_result.confidence_loss.item())
    )
    initial_confidence_diagnostics = initial_result.diagnostics
    initial_loss_breakdown = _breakdown_to_floats(initial_result.loss_breakdown)
    initial_confidence_dropped = initial_result.confidence_dropped
    injection_report = _measure_temporal_injection(
        model,
        scene.views,
        args.precision,
    )
    reconstruction_shift = reconstruction_shift_report(
        baseline_depth,
        initial_raw["depth"].detach(),
        baseline_pose_enc,
        initial_raw["pose_enc"].detach(),
    )
    del baseline_depth, baseline_pose_enc, initial_raw, initial_result
    print(f"initial_position_loss={initial_loss:.8f} (initialized embedding)")

    # No regularizer is part of this proof: in particular, do not move the
    # unsupervised confidence output through decoupled weight decay.
    optimizer, learning_rates, encoder_parameters = _build_optimizer(model, args)
    print(
        "learning_rates="
        f"decoder {learning_rates['decoder']:.3g}, "
        f"embedding {learning_rates['embedding']:.3g}, "
        f"encoder_blocks {learning_rates['encoder_blocks']}"
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.precision == "16-mixed"
    )
    torch.cuda.reset_peak_memory_stats()

    last_gradient_norms = None
    last_confidence_gradient_norms = None
    for step in range(args.steps):
        # Root/backbone training mode enables DINO activation checkpointing and
        # the memory-safe one-frame track-head chunk. ViT-G has zero drop path.
        model.train()
        model.head.eval()
        model.cam_dec.eval()
        model.motion_decoder.train()
        model.track_head.train()
        optimizer.zero_grad(set_to_none=True)

        # Encode once: the backbone does not depend on which frame is the query,
        # so every anchor shares this forward and the reconstruction built on it.
        with _autocast_context(args.precision):
            images, feats, recon = _encode_and_reconstruct(model, scene.views)
            alignment, alignment_report = fit_scene_sim3(recon, scene)
            query_anchors = gather_query_anchor_points(
                recon,
                scene,
                correspondences,
            )
            if active_anchor_count == 1:
                # One anchor needs no cut: its backward is the only one, so it
                # can run straight through the encoder as it did before. The cut
                # is not free -- it holds an accumulated .grad on every backbone
                # tap for the whole step instead of letting the backward free
                # them as it walks -- and every archived run is single-anchor
                # and was sized against the memory ceiling.
                cut_feats, cut_pairs = feats, []
            else:
                cut_feats, cut_pairs = _cut_features(feats)

        step_confidence = None
        step_loss = None
        step_metric_error = None
        step_confidence_loss = None
        step_sync_loss = None
        for anchor_index, anchor_weight in enumerate(anchor_weights):
            if anchor_weight == 0.0:
                continue
            anchor_correspondences = per_anchor_correspondences[anchor_index]
            with _autocast_context(args.precision):
                raw = _anchor_tracks(model, cut_feats, images, scene, anchor_index)
                if raw["track_multi"].shape[2] != scene.num_observations:
                    raise RuntimeError(
                        "Output observation axis changed during optimization"
                    )
                if step_confidence is None:
                    # First *active* anchor: anchor 0 is skipped when it has no
                    # supervised samples, and the log must not go silent for it.
                    step_confidence = _confidence_stats(raw)
                raw = _tracking_only(raw, keep_confidence=confidence_enabled)
                result = sparse_tracking_loss(
                    raw,
                    scene,
                    anchor_correspondences,
                    alignment,
                    query_anchors[
                        per_anchor_rows[anchor_index].to(query_anchors.device)
                    ],
                    huber_delta_m=args.huber_delta_m,
                    confidence_weight=args.confidence_weight,
                    confidence_alpha=confidence_alpha,
                    sync_weight=args.sync_weight,
                    # Only the initial and final evaluations are reported; a
                    # per-step occlusion report costs a device sync per figure
                    # and is discarded.
                    collect_diagnostics=False,
                )
                anchor_total = _weighted_anchor_total(
                    result,
                    position_weight=anchor_weight,
                    confidence_weight=(
                        args.confidence_weight
                        * anchor_confidence_weights[anchor_index]
                    ),
                    sync_weight=args.sync_weight / active_anchor_count,
                )

            # Backward per anchor, so this anchor's track-head graph is freed
            # before the next one allocates its own. The gradient lands on the
            # cut and is pushed through the encoder once, after the loop.
            scaler.scale(anchor_total).backward()
            step_loss = _accumulate(step_loss, result.loss, anchor_weight)
            step_metric_error = _accumulate(
                step_metric_error,
                result.metric_error,
                anchor_weight,
            )
            step_confidence_loss = _accumulate(
                step_confidence_loss,
                result.confidence_loss,
                anchor_confidence_weights[anchor_index],
            )
            step_sync_loss = _accumulate(
                step_sync_loss,
                result.sync_loss,
                1.0 / active_anchor_count,
            )
            del raw, result, anchor_total

        _backward_through_cut(cut_pairs)
        del cut_feats, cut_pairs, feats, recon, images
        scaler.unscale_(optimizer)
        _assert_trainable_gradients_finite(model)
        embedding_gradient = embedding.weight.grad
        if embedding_gradient is None:
            raise RuntimeError("Temporal embedding received no gradient")
        if not torch.isfinite(embedding_gradient).all():
            raise FloatingPointError("Temporal embedding gradient is NaN or Inf")

        last_gradient_norms = {
            "time_embedding": _gradient_norm(embedding.parameters()),
            "motion_decoder": _gradient_norm(model.motion_decoder.parameters()),
            "track_head": _gradient_norm(model.track_head.parameters()),
            "encoder_blocks": (
                _gradient_norm(encoder_parameters) if encoder_parameters else None
            ),
        }
        if last_gradient_norms["time_embedding"] == 0:
            raise RuntimeError("Temporal embedding gradient norm is zero")
        if (
            last_gradient_norms["motion_decoder"] == 0
            or last_gradient_norms["track_head"] == 0
        ):
            raise RuntimeError(
                "MotionDecoder or track-head gradient norm is zero"
            )
        if encoder_parameters and last_gradient_norms["encoder_blocks"] == 0:
            # An all-zero (rather than None) gradient would slip past the
            # per-parameter checks above; unfrozen blocks that learn nothing
            # are this mode's defining failure, so it is fatal here.
            raise RuntimeError(
                "Encoder blocks are trainable but their gradient norm is zero"
            )
        last_confidence_gradient_norms = (
            _confidence_gradient_norms(model) if confidence_enabled else None
        )
        if (
            last_confidence_gradient_norms is not None
            and last_confidence_gradient_norms[
                "track_head_output_conv_confidence_row"
            ] == 0
        ):
            raise RuntimeError(
                "Confidence term is enabled but the track head's confidence output "
                "row received no gradient"
            )
        _assert_frozen_gradients_absent(model)
        scaler.step(optimizer)
        scaler.update()

        confidence_log = (
            ""
            if step_confidence is None
            else (
                f" conf_mean={step_confidence['mean']:.6g}"
                f" conf_p50={step_confidence['p50']:.6g}"
            )
        )
        if confidence_enabled:
            confidence_log += (
                f" conf_loss={step_confidence_loss:.6g}"
                " grad_conf="
                f"{last_confidence_gradient_norms['track_head_output_conv_confidence_row']:.6g}"
            )
        sync_log = (
            "" if step_sync_loss is None else f" sync_loss={step_sync_loss:.6g}"
        )
        encoder_log = (
            ""
            if last_gradient_norms["encoder_blocks"] is None
            else f" grad_encoder={last_gradient_norms['encoder_blocks']:.6g}"
        )
        anchor_log = (
            "" if active_anchor_count == 1 else f" anchors={active_anchor_count}"
        )
        # The per-step drift watch: the alignment is refit each step anyway,
        # so its scale and residual are free, and a trending scale or residual
        # is the first sign the geometry is being traded for tracking loss.
        print(
            f"step={step + 1}/{args.steps} "
            f"loss={step_loss:.8f} "
            f"metric_error_m={step_metric_error:.8f} "
            f"align_scale={alignment_report['scale']:.6f} "
            f"align_residual_m={alignment_report['median_residual_metric']:.6f} "
            f"grad_time={last_gradient_norms['time_embedding']:.6g} "
            f"grad_motion={last_gradient_norms['motion_decoder']:.6g} "
            f"grad_track={last_gradient_norms['track_head']:.6g}"
            f"{anchor_log}{encoder_log}{sync_log}{confidence_log}"
        )

    evaluation = _evaluate(
        model,
        scene,
        correspondences,
        args.precision,
        args.huber_delta_m,
        initial_alignment,
        initial_query_anchors,
        confidence_weight=args.confidence_weight,
        confidence_alpha=confidence_alpha,
        sync_weight=args.sync_weight,
        sync_metric_scale=sync_metric_scale,
        shuffled_views=_shuffled_index_views(scene),
    )
    # Re-checked after training: the head has moved, so a run can start clean and
    # only then push confidence logits far enough to overflow.
    _warn_about_dropped_confidence_samples(
        evaluation["confidence_dropped"],
        evaluation["confidence_sample_count"],
    )
    final_loss = evaluation["loss"]
    final_error = evaluation["metric_error_m"]
    final_alignment = evaluation["alignment"]
    final_alignment_report = evaluation["alignment_report"]
    embedding_norm = float(embedding.weight.detach().float().norm().item())
    embedding_change = float(
        (embedding.weight.detach() - initial_embedding).float().norm().item()
    )

    # Evaluate the exit criteria without raising: the artifacts below are exactly
    # what a failed run needs for diagnosis, so they are written either way and
    # the process exits non-zero at the end.
    failure_reason = _exit_criteria_failure(
        initial_loss=initial_loss,
        final_loss=final_loss,
        embedding_change=embedding_change,
        min_improvement=args.min_improvement,
        confidence_weight=args.confidence_weight,
        baseline_loss=baseline_loss,
        final_shuffled_loss=evaluation["loss_shuffled"],
        min_index_advantage=args.min_index_advantage,
    )
    peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Re-assert the mode this run actually trained under: the saver keys the
    # patch's parameter set off requires_grad, so re-asserting a narrower mode
    # here would silently drop the trained encoder tensors from the file.
    model.set_freeze(args.freeze_mode)
    checkpoint_path = save_temporal_tracking_checkpoint(
        model,
        output_dir / "temporal_tracking.pt",
    )
    summary = {
        "scene": args.scene,
        "cameras": args.cameras,
        "times": args.times,
        "observation_count": scene.num_observations,
        "time_indices": list(scene.time_indices),
        "max_time_indices": args.max_time_indices,
        "query_observation_slot": scene.query_observation_slot,
        # Primary anchor; unchanged meaning. The full list is query_anchors.
        "query_camera": query_observation.camera_id,
        "query_time": query_observation.original_time,
        # Distinct supervised trajectories, comparable across anchor sets.
        # eligibility.supervised_pair_count is the (query, anchor) row count.
        "eligible_query_count": eligibility["eligible_query_count"],
        "initial_alignment": initial_alignment_report,
        "initial_alignment_scale": float(initial_alignment.scale.item()),
        "initial_alignment_rotation": initial_alignment.rotation.tolist(),
        "initial_alignment_translation": initial_alignment.translation.tolist(),
        "final_alignment": final_alignment_report,
        "final_alignment_scale": float(final_alignment.scale.item()),
        "final_alignment_rotation": final_alignment.rotation.tolist(),
        "final_alignment_translation": final_alignment.translation.tolist(),
        "success": failure_reason is None,
        "failure_reason": failure_reason,
        "min_improvement": args.min_improvement,
        "initial_position_loss": initial_loss,
        # Gated on: same alignment and anchors as initial_position_loss, so the
        # only difference is the trained track head and time embedding.
        "final_position_loss": final_loss,
        # Diagnostic only: alignment and anchors refit from the trained model,
        # which is how inference would score it.
        "final_position_loss_refit": evaluation["loss_refit"],
        "initial_metric_error_m": initial_error,
        "final_metric_error_m": final_error,
        "final_metric_error_refit_m": evaluation["metric_error_refit_m"],
        # Unsupervised unless --confidence_weight is set: see _confidence_stats.
        # Reported so an OA/AJ move can be attributed to confidence drift rather
        # than to tracking.
        "initial_track_confidence": initial_confidence,
        "final_track_confidence": evaluation["confidence"],
        "confidence_weight": args.confidence_weight,
        "confidence_alpha": confidence_alpha,
        "confidence_alpha_mode": (
            None
            if not confidence_enabled
            else ("auto" if requested_alpha is None else "explicit")
        ),
        "initial_confidence_loss": initial_confidence_loss,
        "final_confidence_loss": evaluation["confidence_loss"],
        "confidence_sample_count": evaluation["confidence_sample_count"],
        # Non-finite samples are filtered rather than fatal, so the count is what
        # keeps that from being silent. Attributed by cause; the causes overlap.
        "initial_confidence_dropped": initial_confidence_dropped,
        "final_confidence_dropped": evaluation["confidence_dropped"],
        # Unweighted per-term values. The position and confidence terms have very
        # different natural scales, so this is what says whether the chosen
        # --confidence_weight actually balances them.
        "initial_loss_breakdown": initial_loss_breakdown,
        "final_loss_breakdown": evaluation["loss_breakdown"],
        # conf* = alpha / err is the term's optimum, with err the same per-sample
        # Huber error alpha was calibrated against -- not the L2 metric error, which
        # is a different quantity. Compare against the reported track confidence to
        # see whether the term is holding the pretrained level or dragging it
        # somewhere the downstream absolute threshold will notice.
        "implied_optimal_confidence": _implied_optimal_confidence(
            confidence_alpha,
            evaluation["confidence_diagnostics"],
        ),
        "initial_confidence_diagnostics": initial_confidence_diagnostics,
        "final_confidence_diagnostics": evaluation["confidence_diagnostics"],
        "confidence_gradient_norms": last_confidence_gradient_norms,
        "initial_temporal_embedding_norm": initial_embedding_norm,
        "final_temporal_embedding_norm": embedding_norm,
        "temporal_embedding_change": embedding_change,
        "gradient_norms": last_gradient_norms,
        "trainable_tensor_count": report["tensor_count"],
        "trainable_parameter_count": report["parameter_count"],
        "peak_gpu_memory_bytes": peak_memory_bytes,
        "checkpoint_path": str(checkpoint_path),
        "seed": args.seed,
        "precision": args.precision,
        "steps": args.steps,
        "learning_rate": args.lr,
        # --- Fields below are add-only extensions; everything above keeps its
        # pre-existing meaning (initial_* = step-0 state of *this* run, which
        # since the reinit means the initialized embedding).
        "freeze_mode": args.freeze_mode,
        "time_embedding_init": args.time_embedding_init,
        "time_embedding_init_scale": args.time_embedding_init_scale,
        "time_embedding_target_row_norm": embedding_target_row_norm,
        "learning_rates": learning_rates,
        "sync_weight": args.sync_weight,
        "min_index_advantage": args.min_index_advantage,
        # The released checkpoint scored with the same alignment and anchors as
        # every other like-for-like number; the bar the trained model must beat.
        "baseline_position_loss": baseline_loss,
        "baseline_metric_error_m": baseline_error,
        "baseline_track_confidence": baseline_confidence,
        # The trained model scored with one camera's indices shuffled; None when
        # the window has no cross-camera synchronization to break.
        "final_position_loss_shuffled": evaluation["loss_shuffled"],
        "final_metric_error_shuffled_m": evaluation["metric_error_shuffled_m"],
        # Dense same-instant dP disagreement -- the de-hallucination number.
        "baseline_sync_consistency": baseline_sync,
        "initial_sync_consistency": initial_sync,
        "final_sync_consistency": evaluation["sync_consistency"],
        # Step-0 measurements of what the initialized embedding does to the
        # frozen network: signal transport to the taps, and reconstruction
        # perturbation. Answers on the real checkpoint what the synthetic
        # experiment answered for the wiring.
        "temporal_injection": injection_report,
        "reconstruction_shift": reconstruction_shift,
        # Reconstruction vs. dump ground truth, before and after training.
        "baseline_reconstruction_drift": baseline_drift,
        "final_reconstruction_drift": evaluation["reconstruction_drift"],
        # --- Multi-anchor supervision. Anchors are (camera, time) observations
        # owning a dense query field, in priority order; the first is primary.
        "query_anchors": [list(pair) for pair in scene.query_anchors],
        "query_anchor_slots": list(scene.anchor_observation_slots),
        "anchor_count": anchor_count,
        "active_anchor_count": active_anchor_count,
        # Each anchor's share of the supervised samples: the weights that make
        # per-anchor backward equal one combined mean.
        "anchor_sample_counts": anchor_sample_counts,
        "anchor_weights": anchor_weights,
        # The confidence term does not mask on visibility, so it reduces over a
        # larger set than the position term and carries its own shares.
        "anchor_confidence_sample_counts": anchor_confidence_counts,
        "anchor_confidence_weights": anchor_confidence_weights,
        # The full split, accounting for every query, with both rules stated in
        # it as strings so a summary is self-describing.
        "eligibility": eligibility,
        "view_ids": scene.view_ids.tolist(),
        "time_varying_depth": _time_varying_depth_report(scene),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"baseline_position_loss={baseline_loss:.8f}")
    print(f"initial_position_loss={initial_loss:.8f}")
    print(f"final_position_loss={final_loss:.8f}")
    print(f"final_position_loss_refit={evaluation['loss_refit']:.8f}")
    if evaluation["loss_shuffled"] is not None:
        print(f"final_position_loss_shuffled={evaluation['loss_shuffled']:.8f}")
    print(f"initial_metric_error_m={initial_error:.8f}")
    print(f"final_metric_error_m={final_error:.8f}")
    print(f"final_metric_error_refit_m={evaluation['metric_error_refit_m']:.8f}")
    if evaluation["sync_consistency"] is not None:
        print(
            "sync_consistency_m="
            f"baseline {baseline_sync['mean_m']:.6f}, "
            f"final {evaluation['sync_consistency']['mean_m']:.6f} (mean)"
        )
    print(
        "temporal_embedding="
        f"norm {embedding_norm:.8f}, change {embedding_change:.8f}"
    )
    print(f"frozen_gradients=PASS")
    print(f"peak_gpu_memory_bytes={peak_memory_bytes}")
    print(f"checkpoint={checkpoint_path}")
    print(f"summary={summary_path}")
    if failure_reason is not None:
        print(f"FAIL {args.freeze_mode} one-scene overfit: {failure_reason}")
        raise SystemExit(1)
    print(f"PASS {args.freeze_mode} one-scene overfit")


if __name__ == "__main__":
    main()
