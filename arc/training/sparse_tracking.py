"""Detached scene alignment and sparse supervision for raw 4RC tracks.

Raw 4RC tracks are displacements in the predicted reconstruction world.  This
module fits one scene-level Sim(3) from reconstruction geometry and mirrors
inference by adding the detached query pointmap before transforming absolute
positions.  Correspondence selection, pointmap anchors, and alignment are
deliberately detached.

This is the geometry and data-plumbing half.  It assembles predictions, targets and
masks; every scalar it reports comes from ``losses.py`` (differentiable terms) or
``diagnostics.py`` (reporting), so a new loss term can be added there without
touching any of the indexing below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from arc.models.arc.utils.transform import (
    as_homogeneous,
    pose_encoding_to_extri_intri,
    unproject_depth,
)
from arc.training.diagnostics import confidence_occlusion_diagnostics
from arc.training.dumped_kubric import DumpedKubricScene
from arc.training.losses import (
    compose_tracking_loss,
    per_sample_huber_error,
    resolve_confidence_alpha,
    synchronized_consistency_loss,
    synchronized_pair_indices,
    track_confidence_loss,
    track_metric_error,
    track_position_loss,
)
from eval.track.track_eval_util import estimate_sim3


@dataclass(frozen=True)
class DetachedSim3:
    """A fixed transform ``metric_stored = scale * R @ pred + translation``."""

    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor

    def __post_init__(self) -> None:
        scale = torch.as_tensor(self.scale).detach().clone().reshape(())
        rotation = torch.as_tensor(self.rotation).detach().clone()
        translation = torch.as_tensor(self.translation).detach().clone()
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3,3), got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(
                f"translation must have shape (3,), got {translation.shape}"
            )
        if not (
            torch.isfinite(scale)
            and torch.isfinite(rotation).all()
            and torch.isfinite(translation).all()
        ):
            raise ValueError("Sim(3) contains NaN or Inf")
        if scale <= 0:
            raise ValueError(f"Sim(3) scale must be positive, got {scale.item()}")
        # Sim(3) validation is a detached numerical side path. An enclosing
        # mixed-precision context must not autocast the rotation product to
        # BF16 while leaving the comparison identity in float32.
        with torch.autocast(device_type=rotation.device.type, enabled=False):
            identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
            if not torch.allclose(
                rotation @ rotation.mT,
                identity,
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError("Sim(3) rotation is not orthonormal")
            determinant = torch.linalg.det(rotation)
            if not torch.allclose(
                determinant,
                torch.ones_like(determinant),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError(
                    "Sim(3) rotation must be proper (determinant +1), got "
                    f"{determinant.item()}"
                )
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "DetachedSim3":
        return DetachedSim3(
            self.scale.to(device=device, dtype=dtype),
            self.rotation.to(device=device, dtype=dtype),
            self.translation.to(device=device, dtype=dtype),
        )

    def apply_points(self, points: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=points.device.type, enabled=False):
            return self.scale * (points @ self.rotation.mT) + self.translation

    def apply_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=vectors.device.type, enabled=False):
            return self.scale * (vectors @ self.rotation.mT)


@dataclass(frozen=True)
class SparseCorrespondences:
    """Detached dense-grid anchors for sparse metric trajectories."""

    trajectory_indices: torch.Tensor
    query_slots: torch.Tensor
    query_times: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor

    def __post_init__(self) -> None:
        values = {}
        lengths = set()
        for name in (
            "trajectory_indices",
            "query_slots",
            "query_times",
            "rows",
            "columns",
        ):
            value = torch.as_tensor(getattr(self, name)).detach().clone().long()
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional, got {value.shape}")
            values[name] = value
            lengths.add(value.numel())
        if len(lengths) != 1:
            raise ValueError("All sparse correspondence arrays must have equal length")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def count(self) -> int:
        return self.trajectory_indices.numel()

    def to(self, device: torch.device) -> "SparseCorrespondences":
        return SparseCorrespondences(
            trajectory_indices=self.trajectory_indices.to(device),
            query_slots=self.query_slots.to(device),
            query_times=self.query_times.to(device),
            rows=self.rows.to(device),
            columns=self.columns.to(device),
        )

    def anchor_rows(self, index: int) -> torch.Tensor:
        """Boolean mask of the rows belonging to one anchor.

        The companion to :meth:`select_query_slot`: whatever was gathered
        per-row from the scene-level correspondences -- query pointmap anchors,
        above all -- is sliced with this so it stays aligned with the rebased
        correspondences that anchor's head pass is scored against.
        """

        index = int(index)
        if index < 0:
            raise ValueError(f"Query slot index must be non-negative, got {index}")
        return self.query_slots == index

    def select_query_slot(self, index: int) -> "SparseCorrespondences":
        """The rows belonging to one anchor, rebased to a single-query grid.

        Supervising several anchors runs one head pass per anchor so that only
        one query's activations are alive at a time, and each of those passes
        produces ``Q=1``.  This slices out that anchor's rows and renumbers
        ``query_slots`` to 0 so they index the single-query output.

        The renumbering is why the result is for :func:`sparse_tracking_loss`
        only.  It must **not** be handed to
        :func:`gather_query_anchor_points`, which resolves ``query_slots``
        against the scene's anchor list and would therefore read anchor 0's
        pointmaps for anchor *k*'s pixels without complaining.  Gather once from
        the scene-level correspondences and slice the result with the same row
        mask; ``anchor_rows`` returns that mask.
        """

        index = int(index)
        if index < 0:
            raise ValueError(f"Query slot index must be non-negative, got {index}")
        keep = self.anchor_rows(index)
        return SparseCorrespondences(
            trajectory_indices=self.trajectory_indices[keep],
            query_slots=torch.zeros_like(self.query_slots[keep]),
            query_times=self.query_times[keep],
            rows=self.rows[keep],
            columns=self.columns[keep],
        )


@dataclass(frozen=True)
class SparseTrackingLossResult:
    """Position loss, plus whatever else was asked for.

    ``loss`` stays the position-only Huber whatever else is enabled: the overfit
    exit gate compares it against an archived run, so its meaning must not drift.
    ``total_loss`` is what to call ``.backward()`` on.
    """

    loss: torch.Tensor
    metric_error: torch.Tensor
    sample_count: int
    total_loss: torch.Tensor | None = None
    per_sample_error: torch.Tensor | None = None
    target_mask: torch.Tensor | None = None
    confidence_loss: torch.Tensor | None = None
    confidence_mask: torch.Tensor | None = None
    confidence_sample_count: int | None = None
    confidence_dropped: dict | None = None
    confidence_alpha: float | None = None
    sync_loss: torch.Tensor | None = None
    sync_pair_count: int | None = None
    loss_breakdown: dict | None = None
    diagnostics: dict | None = None

    def __post_init__(self) -> None:
        if self.total_loss is None:
            object.__setattr__(self, "total_loss", self.loss)


def _validate_raw_query_mapping(
    raw_predictions: dict,
    scene: DumpedKubricScene,
    correspondences: SparseCorrespondences,
) -> torch.Tensor:
    """Check that a forward's track queries are the anchors it is scored against.

    A run supervising several anchors issues one head pass per anchor, so a
    given forward carries a *subsequence* of the adapter's anchors rather than
    all of them.  What must hold is that every query is a declared anchor and
    that they arrive in the adapter's order -- a reordered or invented query
    would silently score one anchor's field against another's pixels.
    """

    if "track_query_idx" not in raw_predictions:
        raise KeyError("Raw predictions do not contain track_query_idx")
    query_observations = torch.as_tensor(
        raw_predictions["track_query_idx"]
    ).detach().flatten().long().cpu()
    if query_observations.numel() == 0:
        raise ValueError("track_query_idx must contain at least one observation")
    anchors = scene.track_query_observation_slots.cpu().tolist()
    if query_observations.numel() > len(anchors):
        raise ValueError(
            f"Raw track_query_idx has {query_observations.numel()} queries but the "
            f"adapter declared only {len(anchors)} anchors"
        )
    position = 0
    for observation in query_observations.tolist():
        while position < len(anchors) and anchors[position] != observation:
            position += 1
        if position == len(anchors):
            raise ValueError(
                "Raw track_query_idx is not an ordered subsequence of the "
                f"adapter's query observations: got {query_observations.tolist()}, "
                f"anchors are {anchors}"
            )
        position += 1
    if correspondences.query_slots.min().item() < 0:
        raise ValueError("Correspondence query slots must be non-negative")
    if correspondences.query_slots.max().item() >= query_observations.numel():
        raise ValueError(
            "Correspondence query slot exceeds the number of track queries"
        )

    return query_observations


def _metric_pointmap_at_anchor(
    scene: DumpedKubricScene,
    observation_slot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift one anchor observation's ground-truth depth into world points.

    Works at any original time the scene can supply depth for;
    :meth:`DumpedKubricScene.surface_depth_map` is what rejects an off-t0 anchor
    when the per-frame depth sidecar is absent.
    """

    observation = scene.observations[observation_slot]
    camera = observation.camera
    original_time = observation.original_time
    transform = observation.image_transform
    depth = (
        scene.surface_depth_map(camera, original_time)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    intrinsics = (
        scene.intrinsics[camera, original_time]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    world_to_camera = (
        scene.extrinsics_world_to_camera[camera, original_time]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    original_rows, original_columns = transform.output_to_original_indices()
    columns_grid, rows_grid = np.meshgrid(original_columns, original_rows)
    sampled_depth = depth[rows_grid, columns_grid]
    pixels = np.stack(
        (
            columns_grid,
            rows_grid,
            np.ones_like(columns_grid),
        ),
        axis=-1,
    ).astype(np.float64)
    camera_directions = pixels @ np.linalg.inv(intrinsics).T
    camera_points = camera_directions * sampled_depth[..., None]
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    world_points = (camera_points - translation) @ rotation
    valid = (
        (sampled_depth > 1e-6)
        & np.isfinite(sampled_depth)
        & np.isfinite(world_points).all(axis=-1)
    )
    return world_points, valid


def _predicted_pointmaps(raw_predictions: dict) -> torch.Tensor:
    required = {"depth", "pose_enc"}
    missing = required - set(raw_predictions)
    if missing:
        raise KeyError(
            f"Raw predictions are missing alignment fields: {sorted(missing)}"
        )
    depth = raw_predictions["depth"]
    pose_encoding = raw_predictions["pose_enc"]
    if depth.ndim != 4 or depth.shape[0] != 1:
        raise ValueError(
            "The bounded alignment expects depth with shape (1,S,H,W), got "
            f"{tuple(depth.shape)}"
        )
    if pose_encoding.shape[:2] != depth.shape[:2] or pose_encoding.shape[-1] != 9:
        raise ValueError(
            "pose_enc must have shape (1,S,9) matching depth, got "
            f"{tuple(pose_encoding.shape)}"
        )

    # Scene alignment is a detached numerical side path. Keep its geometry in
    # float32 even when the caller is inside mixed-precision autocast; otherwise
    # unprojection can return BF16 and NumPy cannot consume it below.
    with torch.no_grad(), torch.autocast(
        device_type=depth.device.type,
        enabled=False,
    ):
        depth = depth.detach().float()
        pose_encoding = pose_encoding.detach().float()
        height, width = depth.shape[-2:]
        camera_to_world, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding,
            (height, width),
        )
        pointmaps = unproject_depth(
            depth[..., None],
            intrinsics,
            as_homogeneous(camera_to_world),
        )
    return pointmaps.detach()


def fit_scene_sim3(
    raw_predictions: dict,
    scene: DumpedKubricScene,
    *,
    confidence_percentile: float = 80.0,
    robust_iterations: int = 2,
    max_points: int = 50_000,
) -> tuple[DetachedSim3, dict[str, float | int]]:
    """Fit one detached scene transform from the selected time-zero query view."""

    if not 0 <= confidence_percentile <= 100:
        raise ValueError("confidence_percentile must be in [0,100]")
    if robust_iterations < 0:
        raise ValueError("robust_iterations must be non-negative")
    if max_points < 3:
        raise ValueError("max_points must be at least 3")

    observation_slot = scene.query_observation_slot
    pointmaps = _predicted_pointmaps(raw_predictions)
    if pointmaps.shape[1] != scene.num_observations:
        raise ValueError(
            f"Model returned {pointmaps.shape[1]} observations for "
            f"{scene.num_observations} inputs"
        )
    predicted = (
        pointmaps[0, observation_slot].detach().cpu().numpy().astype(np.float64)
    )
    target, target_valid = _metric_pointmap_at_anchor(scene, observation_slot)
    if predicted.shape != target.shape:
        raise ValueError(
            f"Predicted and metric pointmaps disagree: {predicted.shape} vs "
            f"{target.shape}"
        )

    confidence = raw_predictions.get("depth_conf")
    if confidence is None:
        confidence_mask = np.ones(predicted.shape[:2], dtype=bool)
    else:
        confidence = (
            confidence[0, observation_slot]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if confidence.shape != predicted.shape[:2]:
            raise ValueError(
                f"depth_conf grid {confidence.shape} does not match pointmap "
                f"{predicted.shape[:2]}"
            )
        finite_confidence = np.isfinite(confidence)
        if not finite_confidence.any():
            raise ValueError("No finite depth confidence values for Sim(3) alignment")
        threshold = np.percentile(
            confidence[finite_confidence],
            confidence_percentile,
        )
        confidence_mask = finite_confidence & (confidence >= threshold)

    valid = (
        target_valid
        & confidence_mask
        & np.isfinite(predicted).all(axis=-1)
    )
    source = predicted[valid]
    destination = target[valid]
    if source.shape[0] > max_points:
        chosen = np.linspace(0, source.shape[0] - 1, max_points, dtype=np.int64)
        source = source[chosen]
        destination = destination[chosen]
    if source.shape[0] < 3:
        raise ValueError(
            f"Degenerate Sim(3): only {source.shape[0]} valid point pairs"
        )
    centered = source - source.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    tolerance = max(singular_values[0] * 1e-8, 1e-12)
    if np.count_nonzero(singular_values > tolerance) < 2:
        raise ValueError("Degenerate Sim(3): source points are collinear")
    destination_centered = destination - destination.mean(
        axis=0,
        keepdims=True,
    )
    destination_singular_values = np.linalg.svd(
        destination_centered,
        compute_uv=False,
    )
    destination_tolerance = max(
        destination_singular_values[0] * 1e-8,
        1e-12,
    )
    if (
        np.count_nonzero(
            destination_singular_values > destination_tolerance
        )
        < 2
    ):
        raise ValueError("Degenerate Sim(3): target points are collinear")

    scale, rotation, translation = estimate_sim3(
        source,
        destination,
        ransac=False,
    )
    for _ in range(robust_iterations):
        transformed = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(transformed - destination, axis=-1)
        keep = residuals <= np.percentile(residuals, 80.0)
        if keep.sum() < 3:
            raise ValueError("Degenerate Sim(3) after robust residual trimming")
        scale, rotation, translation = estimate_sim3(
            source[keep],
            destination[keep],
            ransac=False,
        )

    sim3 = DetachedSim3(
        scale=torch.tensor(scale, dtype=torch.float32),
        rotation=torch.from_numpy(np.asarray(rotation, dtype=np.float32)),
        translation=torch.from_numpy(np.asarray(translation, dtype=np.float32)),
    )
    residuals = np.linalg.norm(
        sim3.scale.item()
        * (source @ sim3.rotation.detach().cpu().numpy().T)
        + sim3.translation.detach().cpu().numpy()
        - destination,
        axis=-1,
    )
    residual_stored = float(np.median(residuals))
    report = {
        "pair_count": int(source.shape[0]),
        "scale": float(sim3.scale.item()),
        "median_residual_stored": residual_stored,
        "median_residual_metric": (
            residual_stored * scene.track_upscaling_factor
        ),
    }
    return sim3, report


def reconstruction_drift_report(
    raw_predictions: dict,
    scene: DumpedKubricScene,
    alignment: DetachedSim3,
) -> dict:
    """Score the frozen reconstruction against the dump's ground truth.

    The per-step Sim(3) report watches one observation at the top confidence
    quintile, which is nearly blind to where degradation starts.  This instead
    compares, fully detached: predicted depth against the dumped ``depth0`` for
    every camera's time-0 observation (relative error, median and p90), and the
    token camera against the dumped extrinsics for every observation (rotation
    geodesic plus metric camera-centre error), composed through the given
    alignment.  Ground-truth depth and extrinsics live in stored units; the
    scene's track upscaling factor lifts distances to metres.
    """

    depth = raw_predictions.get("depth")
    pose_encoding = raw_predictions.get("pose_enc")
    if depth is None or pose_encoding is None:
        raise KeyError("Raw predictions are missing 'depth' or 'pose_enc'")
    if depth.ndim != 4 or depth.shape[0] != 1:
        raise ValueError(
            f"depth must have shape (1,S,H,W), got {tuple(depth.shape)}"
        )
    if depth.shape[1] != scene.num_observations:
        raise ValueError(
            f"Model returned {depth.shape[1]} observations for "
            f"{scene.num_observations} inputs"
        )

    with torch.no_grad(), torch.autocast(
        device_type=depth.device.type,
        enabled=False,
    ):
        depth = depth.detach().float().cpu()
        scale = float(alignment.scale.item())
        upscaling = float(scene.track_upscaling_factor)

        depth_report: dict[str, dict | None] = {}
        for observation in scene.observations:
            if observation.original_time != 0:
                continue
            transform = observation.image_transform
            original_rows, original_columns = transform.output_to_original_indices()
            columns_grid, rows_grid = np.meshgrid(original_columns, original_rows)
            target = (
                scene.depth0[observation.camera, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)[rows_grid, columns_grid]
            )
            predicted = depth[0, observation.slot].numpy().astype(np.float64) * scale
            valid = (
                np.isfinite(target)
                & (target > 1e-6)
                & np.isfinite(predicted)
            )
            if not valid.any():
                depth_report[str(observation.camera)] = None
                continue
            relative = np.abs(predicted[valid] - target[valid]) / target[valid]
            depth_report[str(observation.camera)] = {
                "median_relative_error": float(np.median(relative)),
                "p90_relative_error": float(np.percentile(relative, 90.0)),
            }

        height, width = depth.shape[-2:]
        camera_to_world, _ = pose_encoding_to_extri_intri(
            pose_encoding.detach().float().cpu(),
            (height, width),
        )
        rotation_align = alignment.rotation.detach().cpu().float()
        rotation_errors = []
        center_errors = []
        for observation in scene.observations:
            rotation_c2w = camera_to_world[0, observation.slot, :3, :3]
            predicted_center = camera_to_world[0, observation.slot, :3, 3]
            ground_truth = (
                scene.extrinsics_world_to_camera[
                    observation.camera, observation.original_time
                ]
                .detach()
                .cpu()
                .float()
            )
            rotation_gt = ground_truth[:3, :3]
            center_gt = -(rotation_gt.mT @ ground_truth[:3, 3])
            # Direction map stored-world -> camera is R_w2c @ R_align^T; its
            # geodesic distance to the ground-truth rotation is the pose error.
            aligned_rotation = rotation_c2w.mT @ rotation_align.mT
            cosine = ((aligned_rotation * rotation_gt).sum() - 1.0) / 2.0
            rotation_errors.append(
                float(torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0))).item())
            )
            stored_center = alignment.apply_points(predicted_center[None, :])[0]
            center_errors.append(
                float(
                    torch.linalg.vector_norm(stored_center - center_gt).item()
                )
                * upscaling
            )

    return {
        "depth": depth_report,
        "pose": {
            "rotation_error_deg": {
                "mean": float(np.mean(rotation_errors)),
                "max": float(np.max(rotation_errors)),
            },
            "camera_center_error_m": {
                "mean": float(np.mean(center_errors)),
                "max": float(np.max(center_errors)),
            },
        },
    }


def gather_query_anchor_points(
    raw_predictions: dict,
    scene: DumpedKubricScene,
    correspondences: SparseCorrespondences,
) -> torch.Tensor:
    """Gather the detached query pointmap term used by ``_postprocess_output``.

    Reads only ``depth`` and ``pose_enc``, so one reconstruction forward serves
    every anchor: the pointmaps cover all S observations, and an anchor is just
    one of them.  ``query_slots`` index the adapter's anchor list, which is what
    that field has always meant, so no ``track_query_idx`` is needed or
    consulted here -- the loss is where a forward's queries are checked against
    the anchors it is scored for.

    Pass the **scene-level** correspondences.  A set rebased by
    :meth:`SparseCorrespondences.select_query_slot` has every ``query_slot`` at
    0 and would silently gather anchor 0's pointmaps; slice this function's
    result with :meth:`SparseCorrespondences.anchor_rows` instead.
    """

    anchor_slots = scene.track_query_observation_slots.cpu()
    if correspondences.query_slots.numel():
        if correspondences.query_slots.min().item() < 0:
            raise ValueError("Correspondence query slots must be non-negative")
        if correspondences.query_slots.max().item() >= anchor_slots.numel():
            raise ValueError(
                "Correspondence query slot exceeds the adapter's query observations"
            )
    pointmaps = _predicted_pointmaps(raw_predictions)
    if pointmaps.shape[1] != scene.num_observations:
        raise ValueError(
            f"Model returned {pointmaps.shape[1]} pointmaps for "
            f"{scene.num_observations} inputs"
        )
    height, width = pointmaps.shape[2:4]
    if (
        correspondences.rows.min().item() < 0
        or correspondences.rows.max().item() >= height
        or correspondences.columns.min().item() < 0
        or correspondences.columns.max().item() >= width
    ):
        raise ValueError("Sparse correspondence is outside the pointmap grid")

    query_slots = correspondences.query_slots.cpu()
    observation_indices = anchor_slots[query_slots].to(pointmaps.device)
    anchors = pointmaps[
        0,
        observation_indices,
        correspondences.rows.to(pointmaps.device),
        correspondences.columns.to(pointmaps.device),
    ]
    return anchors.detach()


# The stages a query passes through at one anchor, in the order they are
# tested. A query is attributed to the first one it fails, so the counts are
# exclusive; the order is also the "furthest reached" order used to roll a
# multi-anchor run up into one split.
ELIGIBILITY_REJECTION_STAGES = (
    "query_time_mismatch",
    "not_visible_in_anchor",
    "projection",
    "anchor_depth_gate",
    "pixel_dedup",
)
_STAGE_QUERY_TIME = 0
_STAGE_NOT_VISIBLE = 1
_STAGE_PROJECTION = 2
_STAGE_DEPTH_GATE = 3
_STAGE_PIXEL_DEDUP = 4

ELIGIBILITY_ASSIGNMENT_RULE = (
    "each query is supervised exactly once, from the anchor whose correspondence "
    "fits best: smallest anchor-depth error, tie-broken by sub-pixel rounding "
    "distance then anchor order. Anchor multiplicity is a property of where the "
    "cameras point rather than of how much a point matters, so one row per query "
    "keeps every query's influence equal; and the best-fitting anchor is the one "
    "whose rounded pixel sits closest to the query's own surface, which is the "
    "label with the least rounding error. Pixel collisions within a single anchor "
    "are resolved first, by the same ordering, so a query that loses one is still "
    "available at its OTHER anchors -- but never again at the one it lost, even if "
    "cross-anchor selection later moves that pixel's winner elsewhere and frees it. "
    "That is lost supervision rather than wrong supervision, and it is counted as "
    "pixel_dedup."
)
ELIGIBILITY_ROLLUP_RULE = (
    "a query eligible at no anchor is attributed to the furthest stage it reached "
    "at any anchor, so the per-reason counts are mutually exclusive and, with "
    "eligible_query_count, sum to total_query_count."
)


def _anchor_candidate(
    point: np.ndarray,
    *,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    transform,
    surface_depth: np.ndarray,
    output_rows_to_original: np.ndarray,
    output_columns_to_original: np.ndarray,
    upscaling_factor: float,
    anchor_depth_tolerance_m: float,
) -> tuple[int | None, tuple[int, int, float, float] | None]:
    """Project one query into one anchor.

    Returns ``(failure_stage, None)`` or ``(None, (row, column, depth_error_m,
    subpixel_distance))``.  The 10 cm depth gate here is the one thing that must
    not move: it is what keeps an anchor pixel on the query's own surface rather
    than on whatever occludes it.
    """

    if not np.isfinite(point).all():
        return _STAGE_PROJECTION, None
    camera_point = world_to_camera @ np.append(point, 1.0)
    if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-6:
        return _STAGE_PROJECTION, None
    projected = intrinsics @ camera_point
    uv_original = projected[:2] / projected[2]
    if not np.isfinite(uv_original).all():
        return _STAGE_PROJECTION, None
    uv_output = transform.original_to_output(uv_original)
    column = int(np.rint(uv_output[0]))
    row = int(np.rint(uv_output[1]))
    if not (
        0 <= column < transform.output_width
        and 0 <= row < transform.output_height
    ):
        return _STAGE_PROJECTION, None

    # Validate the actual dense anchor pixel after resize/crop/rounding,
    # using the same inverse grid map as the scene-level alignment.
    original_column = int(output_columns_to_original[column])
    original_row = int(output_rows_to_original[row])
    depth_at_pixel = float(surface_depth[original_row, original_column])
    depth_error_m = (
        abs(depth_at_pixel - float(camera_point[2])) * upscaling_factor
    )
    if (
        not np.isfinite(depth_at_pixel)
        or depth_at_pixel <= 1e-6
        or depth_error_m > anchor_depth_tolerance_m
    ):
        return _STAGE_DEPTH_GATE, None
    return None, (
        row,
        column,
        depth_error_m,
        float(np.linalg.norm(uv_output - np.array([column, row]))),
    )


def build_anchor_correspondences(
    scene: DumpedKubricScene,
    *,
    anchor_depth_tolerance_m: float = 0.10,
) -> tuple[SparseCorrespondences, dict]:
    """Project depth-verifiable queries into every declared anchor.

    An anchor is one selected ``(camera, time)`` observation owning a dense
    query field.  A query can only be anchored where it is the front surface at
    its own query time, so anchoring in several cameras is what reaches queries
    occluded in the first, and anchoring at several times is what reaches
    queries that do not start at frame 0.  Each query is then supervised exactly
    once, from the anchor it fits best; see :data:`ELIGIBILITY_ASSIGNMENT_RULE`.

    Returns the correspondences and an eligibility report accounting for every
    query, so the recovery a given anchor set buys is measured rather than
    asserted.
    """

    if (
        not np.isfinite(anchor_depth_tolerance_m)
        or anchor_depth_tolerance_m <= 0
    ):
        raise ValueError(
            "anchor_depth_tolerance_m must be finite and positive"
        )

    query_points = scene.query_points.detach().cpu().numpy().astype(np.float64)
    trajectories = (
        scene.trajectories_world.detach().cpu().numpy().astype(np.float64)
    )
    visibility = scene.visibility.detach().cpu().numpy().astype(bool)
    query_times_float = query_points[:, 0]
    if not np.isfinite(query_times_float).all():
        raise ValueError("query_points[:,0] contains NaN or Inf")
    rounded_query_times = np.rint(query_times_float).astype(np.int64)
    if not np.allclose(query_times_float, rounded_query_times, atol=1e-6):
        raise ValueError("query_points[:,0] must contain integer frame indices")
    if np.any(rounded_query_times < 0) or np.any(
        rounded_query_times >= trajectories.shape[0]
    ):
        raise ValueError("query_points contains an out-of-range query time")

    track_indices = np.arange(query_points.shape[0])
    expected_query_xyz = trajectories[rounded_query_times, track_indices]
    if not np.allclose(
        query_points[:, 1:],
        expected_query_xyz,
        atol=1e-5,
        rtol=1e-5,
        equal_nan=False,
    ):
        raise ValueError(
            "query_points XYZ does not match traj3d_world at its query time"
        )

    total_query_count = int(query_points.shape[0])
    slot_cameras = scene.slot_cameras.detach().cpu().numpy()
    slot_times = scene.slot_times.detach().cpu().numpy()
    # Every query that clears an anchor's own visibility test necessarily has a
    # supervised target, because that anchor's observation is itself one of the
    # selected slots this reduces over. Kept as the invariant guard below.
    has_visible_target = visibility[
        slot_cameras[:, None],
        slot_times[:, None],
        track_indices[None, :],
    ].any(axis=0)

    furthest_stage = np.full(total_query_count, -1, dtype=np.int64)
    # trajectory -> [(depth_error_m, subpixel_distance, anchor_index, row, column)]
    # for every anchor that can supervise it. One is chosen per trajectory after
    # all anchors have been walked; see ELIGIBILITY_ASSIGNMENT_RULE.
    candidates: dict[int, list[tuple[float, float, int, int, int]]] = {}
    anchor_times: list[int] = []
    per_anchor_report: list[dict] = []

    for anchor_index, observation_slot in enumerate(scene.anchor_observation_slots):
        observation = scene.observations[observation_slot]
        camera = observation.camera
        anchor_time = observation.original_time
        transform = observation.image_transform
        intrinsics = (
            scene.intrinsics[camera, anchor_time]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        world_to_camera = (
            scene.extrinsics_world_to_camera[camera, anchor_time]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        surface_depth = (
            scene.surface_depth_map(camera, anchor_time)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        output_rows_to_original, output_columns_to_original = (
            transform.output_to_original_indices()
        )

        rejected = dict.fromkeys(ELIGIBILITY_REJECTION_STAGES, 0)
        indices: list[int] = []
        rows: list[int] = []
        columns: list[int] = []
        depth_errors_m: list[float] = []
        subpixel_distances: list[float] = []
        for trajectory_index in track_indices:
            if int(rounded_query_times[trajectory_index]) != anchor_time:
                stage = _STAGE_QUERY_TIME
            elif not visibility[camera, anchor_time, trajectory_index]:
                stage = _STAGE_NOT_VISIBLE
            else:
                stage, candidate = _anchor_candidate(
                    query_points[trajectory_index, 1:],
                    intrinsics=intrinsics,
                    world_to_camera=world_to_camera,
                    transform=transform,
                    surface_depth=surface_depth,
                    output_rows_to_original=output_rows_to_original,
                    output_columns_to_original=output_columns_to_original,
                    upscaling_factor=scene.track_upscaling_factor,
                    anchor_depth_tolerance_m=anchor_depth_tolerance_m,
                )
                if stage is None:
                    row, column, depth_error_m, subpixel = candidate
                    indices.append(int(trajectory_index))
                    rows.append(row)
                    columns.append(column)
                    depth_errors_m.append(depth_error_m)
                    subpixel_distances.append(subpixel)
                    continue
            rejected[ELIGIBILITY_REJECTION_STAGES[stage]] += 1
            furthest_stage[trajectory_index] = max(
                int(furthest_stage[trajectory_index]),
                stage,
            )

        # A dense track-head pixel names exactly one trajectory.  Kubric can
        # provide multiple sparse queries that round to that same pixel, so
        # retain one unambiguous target deterministically.  Anchors have their
        # own grids, so this collision is resolved within an anchor, never
        # across them.
        best_by_pixel: dict[tuple[int, int], tuple[tuple, int]] = {}
        for candidate_index, (row, column) in enumerate(zip(rows, columns)):
            pixel = (int(row), int(column))
            score = (
                float(depth_errors_m[candidate_index]),
                float(subpixel_distances[candidate_index]),
                int(indices[candidate_index]),
            )
            previous = best_by_pixel.get(pixel)
            if previous is None or score < previous[0]:
                best_by_pixel[pixel] = (score, candidate_index)
        kept_set = {candidate_index for _, candidate_index in best_by_pixel.values()}
        kept = sorted(
            kept_set,
            key=lambda candidate_index: int(indices[candidate_index]),
        )
        for candidate_index, trajectory_index in enumerate(indices):
            if candidate_index in kept_set:
                continue
            rejected[ELIGIBILITY_REJECTION_STAGES[_STAGE_PIXEL_DEDUP]] += 1
            furthest_stage[trajectory_index] = max(
                int(furthest_stage[trajectory_index]),
                _STAGE_PIXEL_DEDUP,
            )

        anchor_indices = np.asarray(
            [indices[candidate_index] for candidate_index in kept],
            dtype=np.int64,
        )
        # Not a filter: a query only reaches here by clearing this anchor's own
        # visibility test, and this anchor's observation is one of the selected
        # slots has_visible_target reduces over, so it is necessarily present.
        # Stated rather than mimed as a live rejection path -- a failure here
        # would mean an anchor had fallen outside its own window.
        if anchor_indices.size and not has_visible_target[anchor_indices].all():
            raise RuntimeError(
                f"Anchor {anchor_index} (camera {observation.camera_id}, time "
                f"{anchor_time}) admitted a query with no visible target among "
                "the selected observations, which is impossible while the anchor "
                "is one of them"
            )

        for candidate_index in kept:
            candidates.setdefault(int(indices[candidate_index]), []).append(
                (
                    float(depth_errors_m[candidate_index]),
                    float(subpixel_distances[candidate_index]),
                    anchor_index,
                    int(rows[candidate_index]),
                    int(columns[candidate_index]),
                )
            )

        anchor_times.append(anchor_time)
        per_anchor_report.append(
            {
                "anchor_index": anchor_index,
                "camera": observation.camera_id,
                "view_index": camera,
                "time": anchor_time,
                "observation_slot": observation_slot,
                "considered": total_query_count,
                "eligible": int(anchor_indices.size),
                "assigned": 0,
                "sole_anchor": 0,
                "rejected": rejected,
            }
        )

    # One row per trajectory: the anchor whose correspondence is most
    # trustworthy wins, by the same (depth error, sub-pixel distance) ordering
    # the within-anchor pixel dedup already uses, with anchor order as the
    # deterministic final tiebreak.
    selected: list[tuple[int, int, int, int]] = []
    # Per anchor, the depth error of the labels it actually won, and by how much
    # it beat the runner-up where there was one. Best-fit's whole justification
    # is that the surviving label is the least noisy one; without these the
    # report shows only counts and that claim is invisible short of a training
    # run. Everything here is already computed -- the winner and the runner-up
    # are both in this trajectory's candidate list.
    assigned_depth_errors: list[list[float]] = [[] for _ in per_anchor_report]
    contested_margins: list[list[float]] = [[] for _ in per_anchor_report]
    for trajectory_index, trajectory_candidates in candidates.items():
        ranked = sorted(trajectory_candidates)
        depth_error_m, _, anchor_index, row, column = ranked[0]
        selected.append((anchor_index, trajectory_index, row, column))
        per_anchor_report[anchor_index]["assigned"] += 1
        assigned_depth_errors[anchor_index].append(depth_error_m)
        if len(ranked) == 1:
            per_anchor_report[anchor_index]["sole_anchor"] += 1
        else:
            contested_margins[anchor_index].append(ranked[1][0] - depth_error_m)
    selected.sort()

    for anchor_index, anchor in enumerate(per_anchor_report):
        errors = assigned_depth_errors[anchor_index]
        margins = contested_margins[anchor_index]
        anchor["assigned_depth_error_m"] = (
            None
            if not errors
            else {
                "median": float(np.median(errors)),
                "p95": float(np.percentile(errors, 95.0)),
            }
        )
        # Zero contested wins means this anchor only ever won uncontested, which
        # is a fact about the anchor set rather than missing data -- so the count
        # is reported even when the margin is None.
        anchor["contested_assigned"] = len(margins)
        anchor["contested_depth_error_margin_m"] = (
            None if not margins else float(np.median(margins))
        )

    eligible_indices = np.asarray(
        [trajectory_index for _, trajectory_index, _, _ in selected],
        dtype=np.int64,
    )
    query_slots = np.asarray(
        [anchor_index for anchor_index, _, _, _ in selected],
        dtype=np.int64,
    )
    eligible_query_times = np.asarray(
        [anchor_times[anchor_index] for anchor_index, _, _, _ in selected],
        dtype=np.int64,
    )
    rows = np.asarray([row for _, _, row, _ in selected], dtype=np.int64)
    columns = np.asarray([column for _, _, _, column in selected], dtype=np.int64)

    rolled_up = dict.fromkeys(ELIGIBILITY_REJECTION_STAGES, 0)
    for trajectory_index in range(total_query_count):
        if trajectory_index in candidates:
            continue
        stage = int(furthest_stage[trajectory_index])
        # Unreachable: a query eligible at no anchor was rejected at every one,
        # so its stage is set. Guarded because -1 is a valid Python index that
        # lands on the last stage, which would inflate that bucket with a
        # plausible number instead of failing.
        if stage < 0:
            raise RuntimeError(
                f"Query {trajectory_index} is eligible at no anchor yet was "
                "rejected at none either; the eligibility split cannot account "
                "for it"
            )
        rolled_up[ELIGIBILITY_REJECTION_STAGES[stage]] += 1
    report = {
        "total_query_count": total_query_count,
        "eligible_query_count": len(candidates),
        # Equal to eligible_query_count by construction: one anchor per point.
        "supervised_pair_count": int(eligible_indices.size),
        "anchor_count": len(per_anchor_report),
        "assignment_rule": ELIGIBILITY_ASSIGNMENT_RULE,
        "rollup_rule": ELIGIBILITY_ROLLUP_RULE,
        "rejected": rolled_up,
        "per_anchor": per_anchor_report,
    }

    # An anchor set that reaches nothing is a measurement, not a crash: it is
    # precisely the case worth reporting, so the emptiness is left for the
    # caller to act on. Training refuses to start on it; the eligibility report
    # prints the split that explains why.
    return (
        SparseCorrespondences(
            trajectory_indices=torch.from_numpy(eligible_indices),
            query_slots=torch.from_numpy(query_slots),
            query_times=torch.from_numpy(eligible_query_times),
            rows=torch.from_numpy(rows),
            columns=torch.from_numpy(columns),
        ),
        report,
    )


def sparse_tracking_loss(
    raw_predictions: dict,
    scene: DumpedKubricScene,
    correspondences: SparseCorrespondences,
    alignment: DetachedSim3,
    query_anchor_points: torch.Tensor,
    *,
    huber_delta_m: float = 0.05,
    confidence_weight: float = 0.0,
    confidence_alpha: float | None = None,
    sync_weight: float = 0.0,
    collect_diagnostics: bool = True,
) -> SparseTrackingLossResult:
    """Huber-supervise postprocess-equivalent absolute track positions.

    This assembles predictions, targets and masks; every scalar it reports is
    computed by ``losses.py`` and ``diagnostics.py``.

    ``confidence_weight`` defaults to 0, which skips the confidence term entirely --
    not multiplied by zero, but never built -- so the position-only path is exactly
    what it was before the term existed.

    ``sync_weight`` weights the dense synchronized-pair consistency term
    (:func:`arc.training.losses.synchronized_consistency_loss`): same-time-index
    observation slots owe identical displacement fields, at every pixel and
    regardless of visibility.  Also 0 by default, also never built when off.

    ``confidence_alpha=None`` with a nonzero weight auto-calibrates alpha from this
    call's own sparse statistics.  Resolving it here rather than in the caller is
    what keeps the two inputs commensurate: both the mean confidence and the mean
    error are taken over the same gathered samples.  Callers that train for more
    than one step should resolve it once and pass the value back in, so the target
    does not move underneath the optimizer.

    ``collect_diagnostics=False`` skips the occlusion report.  It costs a device
    sync per reported figure, and a training step throws the report away -- only
    the initial and final evaluations are ever written to ``run_summary.json``.
    """

    if huber_delta_m <= 0 or not np.isfinite(huber_delta_m):
        raise ValueError("huber_delta_m must be finite and positive")
    if confidence_weight < 0 or not np.isfinite(confidence_weight):
        raise ValueError("confidence_weight must be finite and non-negative")
    if sync_weight < 0 or not np.isfinite(sync_weight):
        raise ValueError("sync_weight must be finite and non-negative")
    if correspondences.count == 0:
        raise ValueError("No eligible sparse correspondences")
    if "track_multi" not in raw_predictions:
        raise KeyError("Raw predictions do not contain track_multi")

    tracks = raw_predictions["track_multi"]
    if tracks.ndim != 6 or tracks.shape[0] != 1 or tracks.shape[-1] != 3:
        raise ValueError(
            "track_multi must have shape (1,Q,S,H,W,3), got "
            f"{tuple(tracks.shape)}"
        )
    if tracks.shape[2] != scene.num_observations:
        raise ValueError(
            f"Model returned {tracks.shape[2]} observation slots for "
            f"{scene.num_observations} inputs"
        )

    query_observations = _validate_raw_query_mapping(
        raw_predictions,
        scene,
        correspondences,
    )
    if query_observations.numel() != tracks.shape[1]:
        raise ValueError(
            f"track_query_idx has {query_observations.numel()} entries but "
            f"track_multi has Q={tracks.shape[1]}"
        )
    trajectory_count = scene.trajectories_world.shape[1]
    time_count = scene.trajectories_world.shape[0]
    if correspondences.trajectory_indices.min().item() < 0 or (
        correspondences.trajectory_indices.max().item() >= trajectory_count
    ):
        raise ValueError(
            "Sparse correspondence contains an out-of-range trajectory index"
        )
    if correspondences.query_times.min().item() < 0 or (
        correspondences.query_times.max().item() >= time_count
    ):
        raise ValueError(
            "Sparse correspondence contains an out-of-range query time"
        )

    device = tracks.device
    correspondence = correspondences.to(device)
    if correspondence.query_slots.max().item() >= tracks.shape[1]:
        raise ValueError(
            f"Correspondence query slot exceeds track_multi Q={tracks.shape[1]}"
        )
    height, width = tracks.shape[3:5]
    if (
        correspondence.rows.min().item() < 0
        or correspondence.rows.max().item() >= height
        or correspondence.columns.min().item() < 0
        or correspondence.columns.max().item() >= width
    ):
        raise ValueError("Sparse correspondence is outside the track-head grid")

    predicted_displacement = gather_at_correspondences(tracks, correspondence)
    query_anchor_points = torch.as_tensor(
        query_anchor_points,
        device=device,
        dtype=torch.float32,
    ).detach()
    if query_anchor_points.shape != (correspondences.count, 3):
        raise ValueError(
            "query_anchor_points must have shape "
            f"({correspondences.count},3), got "
            f"{tuple(query_anchor_points.shape)}"
        )
    if not torch.isfinite(query_anchor_points).all():
        raise FloatingPointError("Query pointmap anchor contains NaN or Inf")

    target_positions, target_visible, target_finite, target_mask = sparse_targets(
        scene,
        correspondence,
    )
    if not target_mask.any():
        raise ValueError("No visible finite sparse query-target samples")
    if not torch.isfinite(predicted_displacement[target_mask]).all():
        raise FloatingPointError("Raw track prediction contains NaN or Inf")

    alignment = alignment.to(device=device, dtype=torch.float32)
    metric_factor = float(scene.track_upscaling_factor)
    predicted_metric = (
        alignment.apply_points(
            query_anchor_points[:, None, :] + predicted_displacement
        )
        * metric_factor
    )
    target_metric = target_positions * metric_factor

    loss = track_position_loss(
        predicted_metric,
        target_metric,
        target_mask,
        huber_delta=huber_delta_m,
    )
    metric_error = track_metric_error(predicted_metric, target_metric, target_mask)

    if confidence_weight == 0.0 and sync_weight == 0.0:
        return SparseTrackingLossResult(
            loss=loss,
            metric_error=metric_error,
            sample_count=int(target_mask.sum().item()),
            target_mask=target_mask,
        )

    terms: dict = {"position": loss}
    weights: dict = {"position": 1.0}

    sync_loss = None
    sync_pair_count = None
    if sync_weight > 0.0:
        slot_time_indices = scene.slot_time_indices.reshape(-1)
        sync_loss = synchronized_consistency_loss(
            tracks,
            slot_time_indices.to(device),
            huber_delta=huber_delta_m,
            # The rotation preserves norms, so scale times the metric lift is
            # exactly the factor that puts raw dP differences into metres.
            metric_scale=float(alignment.scale.item()) * metric_factor,
        )
        sync_pair_count = len(synchronized_pair_indices(slot_time_indices)[0])
        terms["sync"] = sync_loss
        weights["sync"] = sync_weight

    confidence_loss = None
    confidence_mask = None
    confidence_sample_count = None
    confidence_dropped = None
    per_sample_error = None
    diagnostics = None
    if confidence_weight > 0.0:
        # The confidence set deliberately drops the visibility mask and keeps only
        # finiteness. Occluded samples are where the error is large, so they are the
        # signal for learning low confidence; masking them out would remove it.
        confidence = gather_at_correspondences(
            _validated_track_confidence(raw_predictions, tracks),
            correspondence,
        )
        predicted_finite = torch.isfinite(predicted_metric).all(dim=-1)
        confidence_finite = torch.isfinite(confidence)
        confidence_mask = target_finite & predicted_finite & confidence_finite
        if not confidence_mask.any():
            raise ValueError("No finite sparse samples for the track confidence loss")
        # Dropping a non-finite sample is the right call -- `expp1` is 1+exp(x), which
        # overflows to inf in BF16 for a large enough logit, and that is a property of
        # the released head rather than a fault in the run. Dropping it *silently* is
        # not: a model quietly losing most of its confidence samples would otherwise
        # look identical to a healthy one. Count it, by cause.
        confidence_dropped = _count_dropped(
            confidence_mask,
            target_finite=target_finite,
            prediction_finite=predicted_finite,
            confidence_finite=confidence_finite,
        )

        per_sample_error = per_sample_huber_error(
            predicted_metric,
            target_metric,
            huber_delta=huber_delta_m,
        )
        if confidence_alpha is None:
            confidence_alpha = resolve_confidence_alpha(
                float(confidence[confidence_mask].detach().float().mean().item()),
                float(per_sample_error[confidence_mask].detach().float().mean().item()),
            )
        confidence_loss = track_confidence_loss(
            confidence,
            per_sample_error,
            confidence_mask,
            alpha=float(confidence_alpha),
        )
        terms["confidence"] = confidence_loss
        weights["confidence"] = confidence_weight
        confidence_sample_count = int(confidence_mask.sum().item())
        if collect_diagnostics:
            diagnostics = confidence_occlusion_diagnostics(
                confidence,
                per_sample_error,
                target_visible,
                confidence_mask,
            )

    total_loss, breakdown = compose_tracking_loss(terms, weights)
    return SparseTrackingLossResult(
        loss=loss,
        metric_error=metric_error,
        sample_count=int(target_mask.sum().item()),
        total_loss=total_loss,
        per_sample_error=per_sample_error,
        target_mask=target_mask,
        confidence_loss=confidence_loss,
        confidence_mask=confidence_mask,
        confidence_sample_count=confidence_sample_count,
        confidence_dropped=confidence_dropped,
        confidence_alpha=(
            None if confidence_alpha is None else float(confidence_alpha)
        ),
        sync_loss=sync_loss,
        sync_pair_count=sync_pair_count,
        loss_breakdown=breakdown,
        diagnostics=diagnostics,
    )


def sparse_targets(
    scene: DumpedKubricScene,
    correspondences: SparseCorrespondences,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ground-truth positions and masks for one set of correspondences.

    Returns ``(positions, visible, finite, mask)``, each ``(M,S)`` except the
    ``(M,S,3)`` positions.  Nothing here reads a prediction, so ``mask.sum()``
    is a property of the scene alone -- which is what lets a run supervising
    several anchors weight them by sample count *before* the first forward, and
    keeps that weight from drifting out of step with the loss's own masking.
    """

    device = correspondences.trajectory_indices.device
    trajectory = scene.trajectories_world.to(device=device, dtype=torch.float32)
    slot_times = scene.slot_times.to(device)
    slot_cameras = scene.slot_cameras.to(device)
    visibility = scene.visibility.to(device)
    trajectory_indices = correspondences.trajectory_indices

    positions = trajectory[
        slot_times[:, None],
        trajectory_indices[None, :],
    ].permute(1, 0, 2)
    visible = visibility[
        slot_cameras[:, None],
        slot_times[:, None],
        trajectory_indices[None, :],
    ].mT
    finite = torch.isfinite(positions).all(dim=-1)
    return positions, visible, finite, visible & finite


def gather_at_correspondences(
    grid: torch.Tensor,
    correspondence: SparseCorrespondences,
) -> torch.Tensor:
    """Read a dense track-head grid at the correspondence pixels.

    Accepts ``(1,Q,S,H,W,C)`` or ``(1,Q,S,H,W)`` and returns ``(M,S,C)`` or
    ``(M,S)``.  Permuting first makes q/y/x adjacent advanced indices, which is what
    keeps the result ordered by correspondence rather than by the broadcast rules
    for separated advanced indices.
    """

    if grid.ndim not in (5, 6) or grid.shape[0] != 1:
        raise ValueError(
            "grid must have shape (1,Q,S,H,W[,C]), got " f"{tuple(grid.shape)}"
        )
    has_channels = grid.ndim == 6
    per_query_grid = (
        grid[0].permute(0, 2, 3, 1, 4) if has_channels else grid[0].permute(0, 2, 3, 1)
    )
    return per_query_grid[
        correspondence.query_slots,
        correspondence.rows,
        correspondence.columns,
    ].float()


def _count_dropped(
    confidence_mask: torch.Tensor,
    *,
    target_finite: torch.Tensor,
    prediction_finite: torch.Tensor,
    confidence_finite: torch.Tensor,
) -> dict:
    """Attribute the confidence set's exclusions to their cause.

    The total alone is derivable from ``eligible_query_count * observation_count``
    minus ``confidence_sample_count``, but nobody performs that subtraction while
    reading a summary, and the total does not say *which* tensor went non-finite.
    The per-cause counts overlap where a sample fails more than one predicate.
    """

    return {
        "total": int((~confidence_mask).sum().item()),
        "target_nonfinite": int((~target_finite).sum().item()),
        "prediction_nonfinite": int((~prediction_finite).sum().item()),
        "confidence_nonfinite": int((~confidence_finite).sum().item()),
    }


def _validated_track_confidence(
    raw_predictions: dict,
    tracks: torch.Tensor,
) -> torch.Tensor:
    if "conf_track_multi" not in raw_predictions:
        raise KeyError(
            "Raw predictions do not contain conf_track_multi, which the track "
            "confidence loss needs"
        )
    confidence = raw_predictions["conf_track_multi"]
    if confidence.shape != tracks.shape[:-1]:
        raise ValueError(
            f"conf_track_multi must have shape {tuple(tracks.shape[:-1])} to match "
            f"track_multi, got {tuple(confidence.shape)}"
        )
    return confidence
