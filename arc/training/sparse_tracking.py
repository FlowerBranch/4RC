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
    if "track_query_idx" not in raw_predictions:
        raise KeyError("Raw predictions do not contain track_query_idx")
    query_observations = torch.as_tensor(
        raw_predictions["track_query_idx"]
    ).detach().flatten().long().cpu()
    if query_observations.numel() == 0:
        raise ValueError("track_query_idx must contain at least one observation")
    if query_observations.numel() != scene.track_query_observation_slots.numel():
        raise ValueError(
            "Raw track_query_idx count does not match the adapter's requested "
            "query observations"
        )
    if not torch.equal(
        query_observations,
        scene.track_query_observation_slots.cpu(),
    ):
        raise ValueError(
            "Raw track_query_idx does not match the adapter's query observations: "
            f"got {query_observations.tolist()}, expected "
            f"{scene.track_query_observation_slots.tolist()}"
        )
    if correspondences.query_slots.min().item() < 0:
        raise ValueError("Correspondence query slots must be non-negative")
    if correspondences.query_slots.max().item() >= query_observations.numel():
        raise ValueError(
            "Correspondence query slot exceeds the number of track queries"
        )
    if (
        correspondences.query_slots.max().item()
        >= scene.track_query_observation_slots.numel()
    ):
        raise ValueError(
            "Correspondence query slot exceeds the adapter's query observations"
        )

    return query_observations


def _metric_pointmap_at_depth0(
    scene: DumpedKubricScene,
    observation_slot: int,
) -> tuple[np.ndarray, np.ndarray]:
    observation = scene.observations[observation_slot]
    if observation.original_time != 0:
        raise ValueError(
            "The current dump can align only an original-time-0 observation"
        )
    camera = observation.camera
    transform = observation.image_transform
    depth = scene.depth0[camera, 0].detach().cpu().numpy().astype(np.float64)
    intrinsics = (
        scene.intrinsics[camera, 0].detach().cpu().numpy().astype(np.float64)
    )
    world_to_camera = (
        scene.extrinsics_world_to_camera[camera, 0]
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
    target, target_valid = _metric_pointmap_at_depth0(scene, observation_slot)
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


def gather_query_anchor_points(
    raw_predictions: dict,
    scene: DumpedKubricScene,
    correspondences: SparseCorrespondences,
) -> torch.Tensor:
    """Gather the detached query pointmap term used by ``_postprocess_output``."""

    query_observations = _validate_raw_query_mapping(
        raw_predictions,
        scene,
        correspondences,
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
    observation_indices = query_observations[query_slots].to(pointmaps.device)
    anchors = pointmaps[
        0,
        observation_indices,
        correspondences.rows.to(pointmaps.device),
        correspondences.columns.to(pointmaps.device),
    ]
    return anchors.detach()


def build_anchor_correspondences(
    scene: DumpedKubricScene,
    *,
    anchor_depth_tolerance_m: float = 0.10,
) -> SparseCorrespondences:
    """Project depth-verifiable queries into the selected time-zero anchor.

    The available depth map rejects rounded pixels whose surface depth differs
    by more than the existing 10 cm query-association gate.  Every selected
    observation remains a supervised target.  Queries colliding at one dense
    pixel are reduced to one deterministic target.  Supervising later query
    anchors requires full ``depth (V,T,1,H,W)`` rather than silently weakening
    this check.
    """

    if (
        not np.isfinite(anchor_depth_tolerance_m)
        or anchor_depth_tolerance_m <= 0
    ):
        raise ValueError(
            "anchor_depth_tolerance_m must be finite and positive"
        )
    query_slot = 0
    observation_slot = scene.query_observation_slot
    observation = scene.observations[observation_slot]
    camera = observation.camera
    query_time = observation.original_time
    if query_time != 0:
        raise ValueError(
            "Sparse depth0 supervision requires the selected query observation "
            "to have original time 0. Parsing and Arc forwarding may use other "
            "times; supervising them requires depth with shape (V,T,1,H,W), "
            "camera-z convention."
        )
    transform = observation.image_transform
    intrinsics = (
        scene.intrinsics[camera, query_time]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    world_to_camera = (
        scene.extrinsics_world_to_camera[camera, query_time]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    output_rows_to_original, output_columns_to_original = (
        transform.output_to_original_indices()
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

    eligible_indices = []
    query_slots = []
    eligible_query_times = []
    rows = []
    columns = []
    depth_errors_m = []
    subpixel_distances = []
    for trajectory_index in track_indices:
        if int(rounded_query_times[trajectory_index]) != query_time:
            continue
        if not visibility[camera, query_time, trajectory_index]:
            continue
        point = query_points[trajectory_index, 1:]
        if not np.isfinite(point).all():
            continue

        camera_point = world_to_camera @ np.append(point, 1.0)
        if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-6:
            continue
        projected = intrinsics @ camera_point
        uv_original = projected[:2] / projected[2]
        if not np.isfinite(uv_original).all():
            continue
        uv_output = transform.original_to_output(uv_original)
        column = int(np.rint(uv_output[0]))
        row = int(np.rint(uv_output[1]))
        if not (
            0 <= column < transform.output_width
            and 0 <= row < transform.output_height
        ):
            continue

        # Validate the actual dense anchor pixel after resize/crop/rounding,
        # using the same inverse grid map as the scene-level alignment.
        original_column = int(output_columns_to_original[column])
        original_row = int(output_rows_to_original[row])
        surface_depth = float(
            scene.depth0[camera, 0, original_row, original_column].item()
        )
        depth_error_m = (
            abs(surface_depth - float(camera_point[2]))
            * scene.track_upscaling_factor
        )
        if (
            not np.isfinite(surface_depth)
            or surface_depth <= 1e-6
            or depth_error_m > anchor_depth_tolerance_m
        ):
            continue

        eligible_indices.append(trajectory_index)
        query_slots.append(query_slot)
        eligible_query_times.append(query_time)
        rows.append(row)
        columns.append(column)
        depth_errors_m.append(depth_error_m)
        subpixel_distances.append(
            float(np.linalg.norm(uv_output - np.array([column, row])))
        )

    eligible_indices = np.asarray(eligible_indices, dtype=np.int64)
    query_slots = np.asarray(query_slots, dtype=np.int64)
    eligible_query_times = np.asarray(eligible_query_times, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    depth_errors_m = np.asarray(depth_errors_m, dtype=np.float64)
    subpixel_distances = np.asarray(subpixel_distances, dtype=np.float64)
    if eligible_indices.size == 0:
        raise ValueError(
            "No eligible sparse queries have a visible, in-bounds, "
            "depth-consistent anchor correspondence"
        )

    # A dense track-head pixel names exactly one trajectory.  Kubric can
    # provide multiple sparse queries that round to that same pixel, so retain
    # one unambiguous target deterministically.
    best_by_pixel = {}
    for candidate, (row, column) in enumerate(zip(rows, columns)):
        pixel = (int(row), int(column))
        score = (
            float(depth_errors_m[candidate]),
            float(subpixel_distances[candidate]),
            int(eligible_indices[candidate]),
        )
        previous = best_by_pixel.get(pixel)
        if previous is None or score < previous[0]:
            best_by_pixel[pixel] = (score, candidate)
    selected = np.asarray(
        sorted(
            (candidate for _, candidate in best_by_pixel.values()),
            key=lambda candidate: int(eligible_indices[candidate]),
        ),
        dtype=np.int64,
    )
    eligible_indices = eligible_indices[selected]
    query_slots = query_slots[selected]
    eligible_query_times = eligible_query_times[selected]
    rows = rows[selected]
    columns = columns[selected]

    selected_visibility = visibility[
        scene.slot_cameras.detach().cpu().numpy()[:, None],
        scene.slot_times.detach().cpu().numpy()[:, None],
        eligible_indices[None, :],
    ]
    has_target = selected_visibility.any(axis=0)
    eligible_indices = eligible_indices[has_target]
    query_slots = query_slots[has_target]
    eligible_query_times = eligible_query_times[has_target]
    rows = rows[has_target]
    columns = columns[has_target]
    if eligible_indices.size == 0:
        raise ValueError("No eligible sparse queries have a visible selected target")

    return SparseCorrespondences(
        trajectory_indices=torch.from_numpy(eligible_indices),
        query_slots=torch.from_numpy(query_slots),
        query_times=torch.from_numpy(eligible_query_times),
        rows=torch.from_numpy(rows),
        columns=torch.from_numpy(columns),
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
    collect_diagnostics: bool = True,
) -> SparseTrackingLossResult:
    """Huber-supervise postprocess-equivalent absolute track positions.

    This assembles predictions, targets and masks; every scalar it reports is
    computed by ``losses.py`` and ``diagnostics.py``.

    ``confidence_weight`` defaults to 0, which skips the confidence term entirely --
    not multiplied by zero, but never built -- so the position-only path is exactly
    what it was before the term existed.

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

    trajectory = scene.trajectories_world.to(device=device, dtype=torch.float32)
    slot_times = scene.slot_times.to(device)
    slot_cameras = scene.slot_cameras.to(device)
    visibility = scene.visibility.to(device)
    trajectory_indices = correspondence.trajectory_indices

    target_positions = trajectory[
        slot_times[:, None],
        trajectory_indices[None, :],
    ].permute(1, 0, 2)
    target_visible = visibility[
        slot_cameras[:, None],
        slot_times[:, None],
        trajectory_indices[None, :],
    ].mT
    target_finite = torch.isfinite(target_positions).all(dim=-1)
    target_mask = target_visible & target_finite
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

    if confidence_weight == 0.0:
        return SparseTrackingLossResult(
            loss=loss,
            metric_error=metric_error,
            sample_count=int(target_mask.sum().item()),
            target_mask=target_mask,
        )

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
    total_loss, breakdown = compose_tracking_loss(
        {"position": loss, "confidence": confidence_loss},
        {"position": 1.0, "confidence": confidence_weight},
    )
    diagnostics = (
        confidence_occlusion_diagnostics(
            confidence,
            per_sample_error,
            target_visible,
            confidence_mask,
        )
        if collect_diagnostics
        else None
    )
    return SparseTrackingLossResult(
        loss=loss,
        metric_error=metric_error,
        sample_count=int(target_mask.sum().item()),
        total_loss=total_loss,
        per_sample_error=per_sample_error,
        target_mask=target_mask,
        confidence_loss=confidence_loss,
        confidence_mask=confidence_mask,
        confidence_sample_count=int(confidence_mask.sum().item()),
        confidence_dropped=confidence_dropped,
        confidence_alpha=float(confidence_alpha),
        loss_breakdown=breakdown,
        diagnostics=diagnostics,
    )


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
