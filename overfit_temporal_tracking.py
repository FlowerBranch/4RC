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
    fit_scene_sim3,
    gather_query_anchor_points,
    load_dumped_kubric_scene,
    save_temporal_tracking_checkpoint,
    sparse_tracking_loss,
)


EXPECTED_TRAINABLE_TENSORS = 231
EXPECTED_NON_TEMPORAL_EMBEDDING_PARAMETERS = 314_551_588
TIME_EMBEDDING_DIM = 1536


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
        "--query_camera",
        type=int,
        help="Selected camera that owns the dense query field (default: first)",
    )
    parser.add_argument(
        "--query_time",
        type=int,
        help="Selected original query frame (default: first selected time)",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    return parser


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

    query_camera = (
        args.cameras[0] if args.query_camera is None else args.query_camera
    )
    query_time = args.times[0] if args.query_time is None else args.query_time
    if query_camera not in args.cameras:
        raise ValueError("--query_camera must be one of --cameras")
    if query_time not in args.times:
        raise ValueError("--query_time must be one of --times")

    if args.parse_only:
        return
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required unless --parse_only is set")
    if not args.output_dir:
        raise ValueError("--output_dir is required unless --parse_only is set")
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
    if query_time != 0:
        raise ValueError(
            "Sparse training requires --query_time 0 because the dump contains "
            "only depth0. Use --parse_only for other windows."
        )


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
    print(f"cameras={list(scene.cameras)}")
    print(f"original_times={list(scene.times)}")
    print(f"time_indices={list(scene.time_indices)}")
    print(f"observations={scene.num_observations}")
    print(
        "query_observation="
        f"slot {query.slot}, camera {query.camera}, "
        f"original_time {query.original_time}"
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
    initial_loss: float,
    final_loss: float,
    embedding_change: float,
    min_improvement: float,
    confidence_weight: float = 0.0,
) -> str | None:
    """Return why the run failed its exit criteria, or None if it passed.

    ``final_loss`` must be the like-for-like number (initial alignment, initial
    anchors).  A bare ``final < initial`` is dominated by reconstruction drift,
    so a relative margin is required.

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
):
    """Score the trained model two ways.

    The refit numbers use a Sim(3) and query anchors re-derived from the trained
    model, matching how inference would score it.  The like-for-like numbers
    reuse the *initial* alignment and anchors, so the only thing that differs
    from ``initial_loss`` is the track head and the time embedding.  The time
    embedding is injected at ``alt_start`` and every head-feeding out-layer is
    downstream of it, so a refit moves even though the reconstruction weights
    are frozen; only the like-for-like number isolates the tracking change.
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
        )
    return {
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
        query_camera=args.query_camera,
        query_time=args.query_time,
        verbose=True,
    )
    _validate_scene_layout(scene)
    if args.parse_only:
        _print_parse_report(scene)
        return
    query_observation = scene.observations[scene.query_observation_slot]
    print(
        "input_layout="
        f"{len(scene.cameras)} cameras x {len(scene.times)} times = "
        f"{scene.num_observations} observations; "
        f"time_indices={list(scene.time_indices)}; "
        f"query=(camera {query_observation.camera}, "
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
    model.set_freeze("temporal_tracking")
    report = model.get_trainable_parameter_report()
    expected_parameter_count = (
        EXPECTED_NON_TEMPORAL_EMBEDDING_PARAMETERS
        + args.max_time_indices * TIME_EMBEDDING_DIM
    )
    if (
        report["tensor_count"] != EXPECTED_TRAINABLE_TENSORS
        or report["parameter_count"] != expected_parameter_count
    ):
        raise RuntimeError(
            "Unexpected temporal_tracking parameter set: "
            f"{report['tensor_count']} tensors / "
            f"{report['parameter_count']} parameters; expected "
            f"{EXPECTED_TRAINABLE_TENSORS} / {expected_parameter_count}"
        )
    print(
        "trainable="
        f"{report['tensor_count']} tensors / {report['parameter_count']} parameters"
    )

    _move_views_to_cuda(scene.views)
    model.eval()
    with torch.no_grad(), _autocast_context(args.precision):
        initial_raw = model(scene.views, force_no_output_conversion=True)
    if initial_raw["track_multi"].shape[2] != scene.num_observations:
        raise RuntimeError(
            "Initial output observation axis does not match the selected inputs"
        )
    initial_alignment, initial_alignment_report = fit_scene_sim3(
        initial_raw,
        scene,
    )
    correspondences = build_anchor_correspondences(scene)
    initial_query_anchors = gather_query_anchor_points(
        initial_raw,
        scene,
        correspondences,
    )
    confidence_enabled = args.confidence_weight > 0
    requested_alpha = _parse_confidence_alpha(args.confidence_alpha)
    initial_result = sparse_tracking_loss(
        _tracking_only(initial_raw, keep_confidence=confidence_enabled),
        scene,
        correspondences,
        initial_alignment,
        initial_query_anchors,
        huber_delta_m=args.huber_delta_m,
        confidence_weight=args.confidence_weight,
        confidence_alpha=requested_alpha,
    )
    initial_confidence = _confidence_stats(initial_raw)
    initial_loss = float(initial_result.loss.item())
    initial_error = float(initial_result.metric_error.item())
    # Resolve alpha once, here, from the untrained model. Re-resolving every step
    # would chase the confidence the term is itself moving, and the optimum would
    # never settle.
    #
    # Resolving it from an eval-mode forward and then applying it to train-mode
    # steps is safe, and checked rather than assumed: nothing on the path is
    # train/eval sensitive. The DPT head builds no BatchNorm (`bn=False` at
    # dpt_head.py:321, and ResidualConvUnit leaves norm1/norm2 None regardless),
    # MotionDecoder has no BatchNorm or dropout, and the ViT's drop_path_rate is
    # 0.0 and never overridden. train() only turns on activation checkpointing and
    # the one-frame track-head chunk, both value-preserving.
    confidence_alpha = initial_result.confidence_alpha
    initial_confidence_loss = (
        None
        if initial_result.confidence_loss is None
        else float(initial_result.confidence_loss.item())
    )
    initial_confidence_diagnostics = initial_result.diagnostics
    initial_loss_breakdown = _breakdown_to_floats(initial_result.loss_breakdown)
    initial_confidence_dropped = initial_result.confidence_dropped
    if confidence_enabled:
        print(
            "confidence_term="
            f"weight {args.confidence_weight:.6g}, alpha {confidence_alpha:.6g} "
            f"({'auto' if requested_alpha is None else 'explicit'}), "
            f"{initial_result.confidence_sample_count} samples "
            f"(position uses {initial_result.sample_count})"
        )
        _warn_about_dropped_confidence_samples(
            initial_confidence_dropped,
            initial_result.confidence_sample_count,
        )
    # initial_query_anchors is deliberately kept: the pass/fail gate re-scores the
    # trained model against this same alignment and these same anchors.
    del initial_raw, initial_result
    print(
        "alignment="
        f"{initial_alignment_report['pair_count']} pairs, "
        f"{initial_alignment_report['median_residual_metric']:.6f} m median residual, "
        f"scale={initial_alignment_report['scale']:.6f}"
    )
    print(f"eligible_queries={correspondences.count}")

    embedding = model.backbone.pretrained.time_index_embedding
    initial_embedding = embedding.weight.detach().clone()
    initial_embedding_norm = float(initial_embedding.float().norm().item())

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    # No regularizer is part of this proof: in particular, do not move the
    # unsupervised confidence output through decoupled weight decay.
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
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

        with _autocast_context(args.precision):
            raw = model(scene.views, force_no_output_conversion=True)
            if raw["track_multi"].shape[2] != scene.num_observations:
                raise RuntimeError(
                    "Output observation axis changed during optimization"
                )
            alignment, alignment_report = fit_scene_sim3(raw, scene)
            query_anchors = gather_query_anchor_points(
                raw,
                scene,
                correspondences,
            )
            step_confidence = _confidence_stats(raw)
            raw = _tracking_only(raw, keep_confidence=confidence_enabled)
            result = sparse_tracking_loss(
                raw,
                scene,
                correspondences,
                alignment,
                query_anchors,
                huber_delta_m=args.huber_delta_m,
                confidence_weight=args.confidence_weight,
                confidence_alpha=confidence_alpha,
                # Only the initial and final evaluations are reported; a per-step
                # occlusion report costs a device sync per figure and is discarded.
                collect_diagnostics=False,
            )

        scaler.scale(result.total_loss).backward()
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
                f" conf_loss={result.confidence_loss.detach().item():.6g}"
                " grad_conf="
                f"{last_confidence_gradient_norms['track_head_output_conv_confidence_row']:.6g}"
            )
        print(
            f"step={step + 1}/{args.steps} "
            f"loss={result.loss.detach().item():.8f} "
            f"metric_error_m={result.metric_error.detach().item():.8f} "
            f"grad_time={last_gradient_norms['time_embedding']:.6g} "
            f"grad_motion={last_gradient_norms['motion_decoder']:.6g} "
            f"grad_track={last_gradient_norms['track_head']:.6g}"
            f"{confidence_log}"
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
    )
    peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.set_freeze("temporal_tracking")
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
        "query_camera": query_observation.camera,
        "query_time": query_observation.original_time,
        "eligible_query_count": correspondences.count,
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
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"initial_position_loss={initial_loss:.8f}")
    print(f"final_position_loss={final_loss:.8f}")
    print(f"final_position_loss_refit={evaluation['loss_refit']:.8f}")
    print(f"initial_metric_error_m={initial_error:.8f}")
    print(f"final_metric_error_m={final_error:.8f}")
    print(f"final_metric_error_refit_m={evaluation['metric_error_refit_m']:.8f}")
    print(
        "temporal_embedding="
        f"norm {embedding_norm:.8f}, change {embedding_change:.8f}"
    )
    print(f"frozen_gradients=PASS")
    print(f"peak_gpu_memory_bytes={peak_memory_bytes}")
    print(f"checkpoint={checkpoint_path}")
    print(f"summary={summary_path}")
    if failure_reason is not None:
        print(f"FAIL temporal_tracking one-scene overfit: {failure_reason}")
        raise SystemExit(1)
    print("PASS temporal_tracking one-scene overfit")


if __name__ == "__main__":
    main()
