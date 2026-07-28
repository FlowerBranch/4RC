from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import arc.training.sparse_tracking as sparse_module
import overfit_temporal_tracking as overfit_cli
from arc.models.arc.arc import Arc
from arc.training import (
    DetachedSim3,
    SparseCorrespondences,
    build_anchor_correspondences,
    fit_scene_sim3,
    gather_query_anchor_points,
    load_dumped_kubric_scene,
    load_temporal_tracking_checkpoint,
    save_temporal_tracking_checkpoint,
    sparse_tracking_loss,
)
from arc.training.dumped_kubric import compute_image_transform


# Every track lies on this world plane, so depth0 can be rendered analytically
# for any camera pose instead of being hard-coded to a constant.
_PLANE_Z = 5.0


def _yaw_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cos, 0.0, sin],
            [0.0, 1.0, 0.0],
            [-sin, 0.0, cos],
        ],
        dtype=np.float64,
    )


def _pitch_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos, -sin],
            [0.0, sin, cos],
        ],
        dtype=np.float64,
    )


def _world_to_camera(camera: int, rotated_camera: int | None):
    """Return world-to-camera (R, t) with ``X_cam = R @ X_world + t``.

    The default poses are identity-rotation with a pure-x baseline, which makes
    a w2c/c2w flip and an R/R^T transpose numerically invisible. ``rotated_camera``
    opts one camera into a real yaw and a z offset so those mistakes change the
    projected pixels.
    """

    if camera != rotated_camera:
        rotation = np.eye(3, dtype=np.float64)
        centre = np.array([float(camera), 0.0, 0.0])
    else:
        # Yaw and pitch together, plus y and z offsets, so the projected pixels
        # move non-uniformly in both axes and camera-space z stops being constant.
        rotation = _pitch_rotation(-12.0) @ _yaw_rotation(25.0)
        centre = np.array([float(camera), 0.45, -0.6])
    return rotation, -rotation @ centre


def _render_plane_depth(rotation, translation, intrinsics, height, width):
    """Per-pixel camera-space z of the world plane ``z = _PLANE_Z``.

    For an identity camera with no z offset this is exactly the constant
    ``_PLANE_Z``, so the unrotated fixture is unchanged.
    """

    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack(
        (columns, rows, np.ones_like(columns)),
        axis=-1,
    ).astype(np.float64)
    # X_cam = depth * direction, and X_world = R^T (X_cam - t).
    directions = pixels @ np.linalg.inv(intrinsics).T
    normal = rotation[:, 2]  # third row of R^T
    return ((_PLANE_Z + normal @ translation) / (directions @ normal)).astype(
        np.float32
    )


def _write_scene(
    root: Path,
    *,
    scene_name: str = "0000",
    time_count: int = 4,
    view_count: int = 2,
    rotated_camera: int | None = None,
) -> Path:
    scene_path = root / scene_name
    track_count = 3
    height = width = 56

    for camera in range(view_count):
        view_path = scene_path / f"view_{camera}"
        view_path.mkdir(parents=True)
        for time_index in range(time_count):
            pixels = np.full(
                (height, width, 3),
                fill_value=20 * camera + time_index,
                dtype=np.uint8,
            )
            Image.fromarray(pixels).save(view_path / f"{time_index:04d}.png")

    initial_points = np.array(
        [
            [-1.0, -0.5, 5.0],
            [0.0, 0.4, 5.0],
            [1.0, -0.2, 5.0],
        ],
        dtype=np.float32,
    )
    trajectory = np.stack(
        [
            initial_points + np.array([0.1 * time, 0.02 * time, 0.0])
            for time in range(time_count)
        ],
        axis=0,
    ).astype(np.float32)
    query_points = np.concatenate(
        (
            np.zeros((track_count, 1), dtype=np.float32),
            trajectory[0],
        ),
        axis=-1,
    )
    visibility = np.ones(
        (view_count, time_count, track_count),
        dtype=bool,
    )
    intrinsics = np.zeros((view_count, time_count, 3, 3), dtype=np.float32)
    intrinsics[..., 0, 0] = 30.0
    intrinsics[..., 1, 1] = 30.0
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = height / 2
    intrinsics[..., 2, 2] = 1.0
    extrinsics = np.zeros((view_count, time_count, 3, 4), dtype=np.float32)
    depth0 = np.zeros((view_count, 1, height, width), dtype=np.float32)
    for camera in range(view_count):
        rotation, translation = _world_to_camera(camera, rotated_camera)
        extrinsics[camera, :, :3, :3] = rotation.astype(np.float32)
        extrinsics[camera, :, :3, 3] = translation.astype(np.float32)
        # depth0 must agree with the pose: build_anchor_correspondences gates on
        # |depth0 - camera_z| <= 10 cm, so a pose change without a matching
        # depth render rejects every candidate.
        depth0[camera, 0] = _render_plane_depth(
            rotation,
            translation,
            intrinsics[camera, 0].astype(np.float64),
            height,
            width,
        )
    np.savez_compressed(
        scene_path / "meta.npz",
        query_points=query_points,
        traj3d_world=trajectory,
        visibility=visibility,
        intrs=intrinsics,
        extrs=extrinsics,
        depth0=depth0,
        track_upscaling_factor=np.float64(1.0),
    )
    return scene_path


@pytest.fixture
def dumped_scene(tmp_path, monkeypatch):
    _write_scene(tmp_path)

    def fake_load_images(
        paths,
        size,
        square_ok,
        verbose,
        patch_size,
    ):
        result = []
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "true_shape": np.int32(
                        [[transform.output_height, transform.output_width]]
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result

    monkeypatch.setattr(
        "arc.dust3r.utils.image.load_images",
        fake_load_images,
    )
    return load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )


def _identity_alignment() -> DetachedSim3:
    return DetachedSim3(
        scale=torch.tensor(1.0),
        rotation=torch.eye(3),
        translation=torch.zeros(3),
    )


def _perfect_raw_tracks(scene, correspondences):
    height, width = scene.views[0]["img"].shape[-2:]
    tracks = torch.zeros(
        1,
        1,
        scene.num_observations,
        height,
        width,
        3,
    )
    for item in range(correspondences.count):
        trajectory_index = int(correspondences.trajectory_indices[item])
        query_time = int(correspondences.query_times[item])
        row = int(correspondences.rows[item])
        column = int(correspondences.columns[item])
        for slot, original_time in enumerate(scene.slot_times.tolist()):
            tracks[0, 0, slot, row, column] = (
                scene.trajectories_world[original_time, trajectory_index]
                - scene.trajectories_world[query_time, trajectory_index]
            )
    query_anchors = scene.trajectories_world[
        correspondences.query_times,
        correspondences.trajectory_indices,
    ].clone()
    return {
        "track_multi": tracks,
        "track_query_idx": scene.track_query_observation_slots.clone(),
    }, query_anchors


def test_square_512_preprocessing_geometry_is_exact():
    transform = compute_image_transform(512, 512, size=512, patch_size=14)

    assert (
        transform.crop_left,
        transform.crop_top,
        transform.output_width,
        transform.output_height,
    ) == (4, 67, 504, 378)
    mapped = transform.original_to_output(np.array([[256.0, 256.0]]))
    np.testing.assert_allclose(mapped, [[252.0, 189.0]])


def test_production_crop_geometry_is_exact():
    """Cover the geometry the overfit actually runs.

    Every adapter test uses 56x56 at size=56, where crop is (0,0) and scale is 1,
    so output_to_original_indices degenerates to the identity and a crop-offset
    sign flip or a crop_top/crop_left swap is invisible. The real dump is
    384x512 at size=512 -> 378x504 with crop (3,4), asserted nowhere else.
    """

    transform = compute_image_transform(384, 512, size=512, patch_size=14)

    assert (
        transform.crop_top,
        transform.crop_left,
        transform.output_height,
        transform.output_width,
    ) == (3, 4, 378, 504)

    rows, columns = transform.output_to_original_indices()
    np.testing.assert_array_equal(rows, np.arange(378) + 3)
    np.testing.assert_array_equal(columns, np.arange(504) + 4)

    # The forward map must be the inverse of the grid above, not merely close.
    np.testing.assert_allclose(
        transform.original_to_output(np.array([[4.0, 3.0], [507.0, 380.0]])),
        [[0.0, 0.0], [503.0, 377.0]],
    )


def test_adapter_rejects_a_loader_that_reorders_its_output(tmp_path, monkeypatch):
    """Slot s must hold the pixels of paths[s].

    _validate_scene_layout re-derives camera and time from the same slot
    arithmetic the loader used, so it is an identity over the whole input space
    and cannot see a permuted image list.
    """

    _write_scene(tmp_path)

    def reversing_load_images(paths, size, square_ok, verbose, patch_size):
        result = []
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "true_shape": np.int32(
                        [[transform.output_height, transform.output_width]]
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result[::-1]

    monkeypatch.setattr(
        "arc.dust3r.utils.image.load_images",
        reversing_load_images,
    )

    with pytest.raises(RuntimeError, match="out of step"):
        load_dumped_kubric_scene(
            tmp_path,
            "0000",
            cameras=(0, 1),
            times=(0, 1, 2, 3),
            size=56,
        )


def test_adapter_keeps_eight_camera_major_observations(dumped_scene):
    assert dumped_scene.num_observations == 8
    assert dumped_scene.time_indices == (0, 1, 2, 3, 0, 1, 2, 3)
    assert [
        (observation.slot, observation.camera, observation.original_time)
        for observation in dumped_scene.observations
    ] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
        (3, 0, 3),
        (4, 1, 0),
        (5, 1, 1),
        (6, 1, 2),
        (7, 1, 3),
    ]
    assert [
        int(view["time_index"].item()) for view in dumped_scene.views
    ] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert dumped_scene.query_observation_slot == 0
    assert all(
        view["track_query_idx"].tolist() == [0]
        for view in dumped_scene.views
    )


class _ObservationAxisArc(Arc):
    """Exercise Arc's public input plumbing without constructing ViT-G."""

    def __init__(self):
        nn.Module.__init__(self)
        self.max_time_indices = 32
        self.seen_time_indices = None

    def _forward(
        self,
        images,
        track_query_idx,
        inference_track=True,
        time_indices=None,
        **kwargs,
    ):
        self.seen_time_indices = time_indices.detach().clone()
        batch, observations, _, height, width = images.shape
        return {
            "track_multi": torch.zeros(
                batch,
                len(track_query_idx),
                observations,
                height,
                width,
                3,
            ),
            "track_query_idx": torch.tensor(track_query_idx),
        }


def test_arc_public_forward_keeps_all_eight_observations(dumped_scene):
    model = _ObservationAxisArc()

    output = model(dumped_scene.views, force_no_output_conversion=True)

    assert output["track_multi"].shape[:3] == (1, 1, 8)
    assert torch.equal(
        model.seen_time_indices,
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
    )
    assert torch.equal(output["track_query_idx"], torch.tensor([0]))


def test_single_camera_window_runs_through_arc_and_sparse_loss(tmp_path):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(1,),
        times=(0, 2, 3),
        size=56,
    )
    overfit_cli._validate_scene_layout(scene)

    assert scene.num_observations == 3
    assert scene.time_indices == (0, 1, 2)
    assert scene.slot_cameras.tolist() == [1, 1, 1]
    assert scene.slot_times.tolist() == [0, 2, 3]
    output = _ObservationAxisArc()(
        scene.views,
        force_no_output_conversion=True,
    )
    assert output["track_multi"].shape[:3] == (1, 1, 3)

    correspondences = build_anchor_correspondences(scene)
    raw, query_anchors = _perfect_raw_tracks(scene, correspondences)
    result = sparse_tracking_loss(
        raw,
        scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    assert result.sample_count == correspondences.count * 3


def test_nonfirst_query_camera_owns_alignment_correspondence(tmp_path):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_camera=1,
        query_time=0,
        size=56,
    )
    query = scene.observations[scene.query_observation_slot]
    assert (query.slot, query.camera, query.original_time) == (2, 1, 0)
    assert all(view["track_query_idx"].tolist() == [2] for view in scene.views)

    # If correspondence construction accidentally uses camera 0, every
    # candidate will fail its depth-consistency gate.
    scene.depth0[0].fill_(100.0)
    correspondences = build_anchor_correspondences(scene)

    assert correspondences.count > 0
    assert len(set(zip(
        correspondences.rows.tolist(),
        correspondences.columns.tolist(),
    ))) == correspondences.count


def test_selected_camera_order_is_preserved_camera_major(tmp_path):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(1, 0),
        times=(0, 3),
        size=56,
    )
    overfit_cli._validate_scene_layout(scene)

    assert [
        (observation.camera, observation.original_time)
        for observation in scene.observations
    ] == [(1, 0), (1, 3), (0, 0), (0, 3)]
    assert scene.time_indices == (0, 1, 0, 1)
    assert scene.query_observation_slot == 0


def test_adapter_supports_more_than_two_selected_cameras(tmp_path):
    _write_scene(tmp_path, time_count=3, view_count=4)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(3, 1, 0),
        times=(0, 2),
        size=56,
    )
    overfit_cli._validate_scene_layout(scene)

    assert scene.num_observations == 6
    assert [
        (observation.camera, observation.original_time)
        for observation in scene.observations
    ] == [(3, 0), (3, 2), (1, 0), (1, 2), (0, 0), (0, 2)]
    assert scene.time_indices == (0, 1, 0, 1, 0, 1)
    assert scene.slot_cameras.tolist() == [3, 3, 1, 1, 0, 0]


def test_adapter_uses_the_real_image_loader(tmp_path):
    _write_scene(tmp_path)

    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    assert scene.num_observations == 8
    assert all(view["img"].shape == (1, 3, 56, 56) for view in scene.views)
    assert scene.time_indices == (0, 1, 2, 3, 0, 1, 2, 3)


def test_adapter_parses_nonzero_window_but_sparse_supervision_rejects_it(
    tmp_path,
):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0,),
        times=(1, 2, 3),
        size=56,
    )

    assert scene.num_observations == 3
    assert scene.time_indices == (0, 1, 2)
    query = scene.observations[scene.query_observation_slot]
    assert (query.camera, query.original_time) == (0, 1)
    with pytest.raises(ValueError, match="requires.*original time 0"):
        build_anchor_correspondences(scene)


def test_overfit_cli_accepts_dynamic_layouts_and_keeps_depth0_guard():
    parser = overfit_cli.build_arg_parser()
    one_camera = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--checkpoint_dir",
            "checkpoint",
            "--output_dir",
            "output",
            "--cameras",
            "1",
            "--times",
            "0",
            "2",
            "5",
        ]
    )
    overfit_cli._validate_args(one_camera)

    missing_checkpoint = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--cameras",
            "1",
            "--times",
            "0",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="checkpoint_dir"):
        overfit_cli._validate_args(missing_checkpoint)

    parse_only = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--cameras",
            "1",
            "--times",
            "1",
            "2",
            "5",
            "--parse_only",
        ]
    )
    overfit_cli._validate_args(parse_only)
    assert parse_only.checkpoint_dir is None
    assert parse_only.output_dir is None

    training_without_depth = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--checkpoint_dir",
            "checkpoint",
            "--output_dir",
            "output",
            "--cameras",
            "1",
            "--times",
            "1",
            "2",
            "5",
        ]
    )
    with pytest.raises(ValueError, match="query_time 0"):
        overfit_cli._validate_args(training_without_depth)

    too_many_times = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--cameras",
            "1",
            "--times",
            "0",
            "2",
            "5",
            "--max_time_indices",
            "2",
            "--parse_only",
        ]
    )
    with pytest.raises(ValueError, match="Selected 3 semantic times"):
        overfit_cli._validate_args(too_many_times)


def _rotated_two_camera_scene(tmp_path, monkeypatch, *, query_camera=1):
    """A scene whose camera 1 has a real yaw/pitch and a matching depth0 render."""

    _write_scene(tmp_path, rotated_camera=1)

    def fake_load_images(paths, size, square_ok, verbose, patch_size):
        result = []
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "true_shape": np.int32(
                        [[transform.output_height, transform.output_width]]
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result

    monkeypatch.setattr("arc.dust3r.utils.image.load_images", fake_load_images)
    return load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_camera=query_camera,
        query_time=0,
        size=56,
    )


def test_metric_pointmap_lifts_depth0_onto_the_known_world_plane(
    tmp_path,
    monkeypatch,
):
    """Check the world lift against ground truth the fixture knows independently.

    The old Sim(3) test built its source by inverting a known transform applied to
    this same function's output, so the world-lift convention cancelled out and a
    transposed lift or an axis swap passed. Here depth0 renders the plane
    z = _PLANE_Z, so every lifted point must land on that plane -- an invariant
    that R vs R^T and any axis permutation break immediately.
    """

    scene = _rotated_two_camera_scene(tmp_path, monkeypatch)
    world_points, valid = sparse_module._metric_pointmap_at_depth0(
        scene,
        scene.query_observation_slot,
    )

    assert valid.all()
    np.testing.assert_allclose(world_points[..., 2], _PLANE_Z, atol=1e-4)

    # Each anchor pixel must lift back onto its own track, up to the half-pixel
    # rounding in the anchor choice (about 0.18 world units per pixel here).
    correspondences = build_anchor_correspondences(scene)
    for item in range(correspondences.count):
        row = int(correspondences.rows[item])
        column = int(correspondences.columns[item])
        track = int(correspondences.trajectory_indices[item])
        expected = scene.trajectories_world[0, track].numpy()
        assert np.linalg.norm(world_points[row, column] - expected) < 0.15


def test_fit_scene_sim3_reads_the_query_observation_not_slot_zero(
    tmp_path,
    monkeypatch,
):
    """Only the query observation's pointmap may drive the alignment."""

    scene = _rotated_two_camera_scene(tmp_path, monkeypatch)
    query_slot = scene.query_observation_slot
    assert query_slot != 0

    target, _ = sparse_module._metric_pointmap_at_depth0(scene, query_slot)
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 2.25
    translation = np.array([0.3, -0.2, 1.1], dtype=np.float64)

    # Only the query slot carries the correctly pre-transformed source; every
    # other slot holds a decoy that fits a different transform entirely.
    pointmaps = torch.zeros(
        1,
        scene.num_observations,
        target.shape[0],
        target.shape[1],
        3,
    )
    for slot in range(scene.num_observations):
        pointmaps[0, slot] = torch.from_numpy(target).float() * 5.0 + 11.0
    pointmaps[0, query_slot] = torch.from_numpy(
        ((target - translation) @ rotation) / scale
    ).float()

    monkeypatch.setattr(
        sparse_module,
        "_predicted_pointmaps",
        lambda raw: pointmaps,
    )
    raw = {
        "depth_conf": torch.ones(
            1,
            scene.num_observations,
            target.shape[0],
            target.shape[1],
        )
    }

    fitted, report = fit_scene_sim3(raw, scene, confidence_percentile=0)

    assert fitted.scale.item() == pytest.approx(scale, rel=1e-4)
    np.testing.assert_allclose(fitted.rotation.numpy(), rotation, atol=1e-4)
    np.testing.assert_allclose(fitted.translation.numpy(), translation, atol=1e-4)
    assert report["median_residual_metric"] < 1e-5


def test_detached_sim3_rejects_improper_and_non_orthonormal_rotations():
    """These guards exist but no test triggered them."""

    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant"):
        DetachedSim3(
            scale=torch.tensor(1.0),
            rotation=torch.from_numpy(reflection).float(),
            translation=torch.zeros(3),
        )

    with pytest.raises(ValueError, match="orthonormal"):
        DetachedSim3(
            scale=torch.tensor(1.0),
            rotation=torch.tensor(
                [[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            translation=torch.zeros(3),
        )

    with pytest.raises(ValueError, match="scale must be positive"):
        DetachedSim3(
            scale=torch.tensor(-1.0),
            rotation=torch.eye(3),
            translation=torch.zeros(3),
        )


def test_fit_scene_sim3_rejects_collinear_predictions(tmp_path, monkeypatch):
    """The collinearity guard exists but no test reached it."""

    scene = _rotated_two_camera_scene(tmp_path, monkeypatch)
    target, _ = sparse_module._metric_pointmap_at_depth0(
        scene,
        scene.query_observation_slot,
    )
    height, width = target.shape[:2]

    # Every coordinate carries the identical ramp, so the points lie exactly on
    # the (1,1,1) diagonal even in float32. Scaling the axes differently would
    # let float32 rounding lift the cloud off the line and defeat the guard.
    line = torch.zeros(1, scene.num_observations, height, width, 3)
    ramp = torch.linspace(0.0, 1.0, height * width).reshape(height, width)
    for axis in range(3):
        line[0, :, :, :, axis] = ramp

    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: line)
    raw = {"depth_conf": torch.ones(1, scene.num_observations, height, width)}

    with pytest.raises(ValueError, match="collinear"):
        fit_scene_sim3(raw, scene, confidence_percentile=0)


def _project_expected_anchors(scene, camera, rotated_camera):
    """Independent pinhole oracle for the query anchors.

    Derived from the pose and intrinsics directly, not from
    ``build_anchor_correspondences`` or ``ImageTransform``, so a w2c/c2w flip or
    an R/R^T transpose inside the adapter changes one side but not the other.
    """

    rotation, translation = _world_to_camera(camera, rotated_camera)
    intrinsics = scene.intrinsics[camera, 0].numpy().astype(np.float64)
    expected = []
    for point in scene.trajectories_world[0].numpy().astype(np.float64):
        camera_point = rotation @ point + translation
        homogeneous = intrinsics @ camera_point
        u, v = homogeneous[:2] / homogeneous[2]
        expected.append((int(np.rint(v)), int(np.rint(u))))
    return expected


def test_rotated_query_camera_projects_to_independently_derived_pixels(tmp_path):
    """A real rotation makes the world-to-camera convention observable.

    With every camera at identity rotation and a pure-x baseline, inverting the
    extrinsics or transposing R leaves the projected pixels unchanged (or shifted
    uniformly), and the constant-depth plane means the 10 cm depth gate never
    fires. Both mistakes move these pixels.
    """

    _write_scene(tmp_path, rotated_camera=1)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_camera=1,
        query_time=0,
        size=56,
    )
    assert (
        scene.observations[scene.query_observation_slot].slot,
        scene.observations[scene.query_observation_slot].camera,
    ) == (2, 1)

    correspondences = build_anchor_correspondences(scene)

    expected = _project_expected_anchors(scene, camera=1, rotated_camera=1)
    # Hand-derived from R = pitch(-12) @ yaw(25), C = (1, 0.45, -0.6), fx=fy=30,
    # cx=cy=28: distinct in both axes, unlike the identity-camera fixture.
    assert expected == [(30, 31), (34, 36), (30, 42)]

    actual = list(
        zip(correspondences.rows.tolist(), correspondences.columns.tolist())
    )
    assert actual == expected


def test_sparse_loss_is_zero_for_a_nonzero_query_slot(tmp_path):
    """Exercise the query-slot -> observation indirection off its identity.

    ``build_anchor_correspondences`` emits query_slot 0 (an index into the track
    query list) while the anchors live at ``scene.query_observation_slot`` (an
    index into the observation axis). Every other numeric test uses a scene where
    those are both 0, so dropping the indirection is numerically invisible.
    """

    _write_scene(tmp_path, rotated_camera=1)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_camera=1,
        query_time=0,
        size=56,
    )
    assert scene.query_observation_slot == 2

    correspondences = build_anchor_correspondences(scene)
    raw, query_anchors = _perfect_raw_tracks(scene, correspondences)

    result = sparse_tracking_loss(
        raw,
        scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.sample_count == correspondences.count * scene.num_observations
    assert float(result.loss.item()) == pytest.approx(0.0, abs=1e-8)
    assert float(result.metric_error.item()) == pytest.approx(0.0, abs=1e-8)


def test_query_anchor_gather_follows_the_observation_slot(tmp_path):
    """Anchors must be read from the query observation, not from slot 0."""

    _write_scene(tmp_path, rotated_camera=1)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_camera=1,
        query_time=0,
        size=56,
    )
    correspondences = build_anchor_correspondences(scene)
    height, width = scene.views[0]["img"].shape[-2:]

    # Give every observation a distinct constant pointmap so the gather's choice
    # of observation is readable straight off the returned value.
    pointmaps = torch.zeros(1, scene.num_observations, height, width, 3)
    for slot in range(scene.num_observations):
        pointmaps[0, slot] = float(slot + 1)

    import arc.training.sparse_tracking as module

    original = module._predicted_pointmaps
    try:
        module._predicted_pointmaps = lambda raw: pointmaps
        anchors = gather_query_anchor_points(
            {"track_query_idx": scene.track_query_observation_slots},
            scene,
            correspondences,
        )
    finally:
        module._predicted_pointmaps = original

    expected_value = float(scene.query_observation_slot + 1)
    assert torch.allclose(anchors, torch.full_like(anchors, expected_value))


def test_exit_gate_requires_a_real_margin_not_just_any_decrease():
    """A bare `final < initial` is dominated by reconstruction drift.

    The time embedding is injected at alt_start and every head-feeding out-layer
    is downstream of it, so refitting the Sim(3) moves the loss even when the
    track head improved by exactly nothing. Require a margin.
    """

    # Passes: a 5% drop clears the 1% default margin.
    assert overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=0.95,
        embedding_change=0.5,
        min_improvement=0.01,
    ) is None

    # Fails: a 0.1% drop would have passed the old zero-margin comparison.
    reason = overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=0.999,
        embedding_change=0.5,
        min_improvement=0.01,
    )
    assert reason is not None
    assert "like-for-like position loss" in reason

    # Fails: loss went up.
    assert overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=1.5,
        embedding_change=0.5,
        min_improvement=0.01,
    ) is not None

    # A frozen embedding fails regardless of the loss.
    reason = overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=0.1,
        embedding_change=0.0,
        min_improvement=0.01,
    )
    assert reason == "Temporal embedding did not change"


def test_min_improvement_is_validated():
    parser = overfit_cli.build_arg_parser()
    base = [
        "--data_root", "root", "--scene", "0000",
        "--checkpoint_dir", "ckpt", "--output_dir", "out",
    ]
    args = parser.parse_args(base)
    assert args.min_improvement == 0.01
    overfit_cli._validate_args(args)

    for bad in ("-0.1", "1.0", "nan"):
        rejected = parser.parse_args(base + ["--min_improvement", bad])
        with pytest.raises(ValueError, match="--min_improvement"):
            overfit_cli._validate_args(rejected)


def test_evaluate_scores_like_for_like_against_the_initial_alignment(monkeypatch):
    """The gated number must reuse the initial alignment and anchors.

    Otherwise `initial_loss` and `final_loss` are measured under two different
    transforms and their difference is not attributable to tracking.
    """

    initial_alignment = _identity_alignment()
    initial_anchors = torch.full((3, 3), 7.0)
    refit_alignment = _identity_alignment()
    refit_anchors = torch.full((3, 3), -1.0)

    calls = []

    class _RecordedResult:
        def __init__(self, value):
            self.loss = torch.tensor(value)
            self.metric_error = torch.tensor(value * 2)

    def fake_loss(raw, scene, correspondences, alignment, anchors, **kwargs):
        calls.append((alignment, anchors))
        return _RecordedResult(0.25 if len(calls) == 1 else 0.75)

    monkeypatch.setattr(overfit_cli, "sparse_tracking_loss", fake_loss)
    monkeypatch.setattr(
        overfit_cli,
        "fit_scene_sim3",
        lambda raw, scene: (refit_alignment, {"pair_count": 1}),
    )
    monkeypatch.setattr(
        overfit_cli,
        "gather_query_anchor_points",
        lambda raw, scene, correspondences: refit_anchors,
    )
    monkeypatch.setattr(
        overfit_cli,
        "_tracking_only",
        lambda raw: raw,
    )

    class _StubModel:
        def eval(self):
            return self

        def __call__(self, views, **kwargs):
            return {"conf_track_multi": torch.full((1, 1, 2, 2, 2), 3.0)}

    evaluation = overfit_cli._evaluate(
        _StubModel(),
        SimpleNamespace(views=[]),
        object(),
        "32",
        0.05,
        initial_alignment,
        initial_anchors,
    )

    assert len(calls) == 2
    # First call is the refit (diagnostic), second is the gated like-for-like.
    assert calls[0][0] is refit_alignment and calls[0][1] is refit_anchors
    assert calls[1][0] is initial_alignment and calls[1][1] is initial_anchors
    assert evaluation["loss_refit"] == pytest.approx(0.25)
    assert evaluation["loss"] == pytest.approx(0.75)
    assert evaluation["confidence"]["mean"] == pytest.approx(3.0)


def test_parse_only_main_needs_neither_cuda_nor_checkpoint(
    tmp_path,
    monkeypatch,
    capsys,
):
    _write_scene(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overfit_temporal_tracking.py",
            "--data_root",
            str(tmp_path),
            "--scene",
            "0000",
            "--cameras",
            "1",
            "--times",
            "1",
            "2",
            "3",
            "--parse_only",
        ],
    )

    def unexpected_cuda_check():
        raise AssertionError("parse-only mode must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)
    monkeypatch.setattr(
        Arc,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail(
            "parse-only mode must not load an Arc checkpoint"
        ),
    )

    overfit_cli.main()

    output = capsys.readouterr().out
    assert "observations=3" in output
    assert "time_indices=[0, 1, 2]" in output
    assert "query_observation=slot 0, camera 1, original_time 1" in output
    assert "PASS mvtracker dump parsing" in output


def test_nonconsecutive_frames_keep_local_semantic_time_indices(
    tmp_path,
    monkeypatch,
):
    _write_scene(tmp_path, time_count=7)

    def fake_load_images(paths, size, square_ok, verbose, patch_size):
        result = []
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result

    monkeypatch.setattr(
        "arc.dust3r.utils.image.load_images",
        fake_load_images,
    )
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 2, 4, 6),
        size=56,
    )

    assert scene.time_indices == (0, 1, 2, 3, 0, 1, 2, 3)
    assert scene.slot_time_indices.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert scene.slot_times.tolist() == [0, 2, 4, 6, 0, 2, 4, 6]
    assert [
        observation.original_time for observation in scene.observations
    ] == [0, 2, 4, 6, 0, 2, 4, 6]


def test_known_sim3_is_recovered_and_detached(dumped_scene, monkeypatch):
    target, _ = sparse_module._metric_pointmap_at_depth0(dumped_scene, 0)
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 2.25
    translation = np.array([0.3, -0.2, 1.1], dtype=np.float64)
    source = ((target - translation) @ rotation) / scale
    pointmaps = torch.zeros(
        1,
        dumped_scene.num_observations,
        target.shape[0],
        target.shape[1],
        3,
        requires_grad=True,
    )
    pointmaps_data = pointmaps.detach().clone()
    pointmaps_data[0, 0] = torch.from_numpy(source).float()
    pointmaps_data.requires_grad_(True)
    monkeypatch.setattr(
        sparse_module,
        "_predicted_pointmaps",
        lambda raw: pointmaps_data,
    )
    raw = {
        "depth_conf": torch.ones(
            1,
            dumped_scene.num_observations,
            target.shape[0],
            target.shape[1],
        )
    }

    fitted, report = fit_scene_sim3(
        raw,
        dumped_scene,
        confidence_percentile=0,
    )

    assert fitted.scale.item() == pytest.approx(scale, rel=1e-4)
    np.testing.assert_allclose(fitted.rotation.numpy(), rotation, atol=1e-4)
    np.testing.assert_allclose(
        fitted.translation.numpy(),
        translation,
        atol=1e-4,
    )
    assert report["median_residual_metric"] < 1e-5
    assert not fitted.scale.requires_grad
    assert not fitted.rotation.requires_grad
    assert not fitted.translation.requires_grad


def test_predicted_pointmaps_stay_float32_inside_bfloat16_autocast(monkeypatch):
    depth = torch.ones(1, 2, 4, 5, dtype=torch.bfloat16)
    pose_encoding = torch.ones(1, 2, 9, dtype=torch.bfloat16)

    def fake_pose_conversion(converted_pose, image_shape):
        assert converted_pose.dtype == torch.float32
        assert image_shape == (4, 5)
        camera_to_world = torch.eye(4).expand(1, 2, 4, 4).clone()
        intrinsics = torch.eye(3).expand(1, 2, 3, 3).clone()
        return camera_to_world, intrinsics

    def fake_unproject(converted_depth, intrinsics, camera_to_world):
        assert converted_depth.dtype == torch.float32
        # Matmul is deliberately autocast-sensitive. The alignment helper must
        # disable the caller's BF16 autocast before reaching this operation.
        marker = torch.ones(1, 1, dtype=torch.float32)
        marker = marker @ marker
        return marker * torch.ones(1, 2, 4, 5, 3, dtype=torch.float32)

    monkeypatch.setattr(
        sparse_module,
        "pose_encoding_to_extri_intri",
        fake_pose_conversion,
    )
    monkeypatch.setattr(sparse_module, "as_homogeneous", lambda value: value)
    monkeypatch.setattr(sparse_module, "unproject_depth", fake_unproject)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pointmaps = sparse_module._predicted_pointmaps(
            {"depth": depth, "pose_enc": pose_encoding}
        )

    assert pointmaps.dtype == torch.float32
    assert pointmaps.numpy().dtype == np.float32


def test_detached_sim3_stays_float32_inside_bfloat16_autocast():
    points = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    vectors = torch.tensor(
        [[0.25, -0.5, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        alignment = DetachedSim3(
            scale=torch.tensor(2.0, dtype=torch.float32),
            rotation=torch.eye(3, dtype=torch.float32),
            translation=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        ).to(device=torch.device("cpu"), dtype=torch.float32)
        transformed_points = alignment.apply_points(points)
        transformed_vectors = alignment.apply_vectors(vectors)
        loss = transformed_points.sum() + transformed_vectors.sum()

    assert transformed_points.dtype == torch.float32
    assert transformed_vectors.dtype == torch.float32
    loss.backward()
    assert points.grad is not None
    assert vectors.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(vectors.grad).all()


def test_correspondence_is_direct_projection_and_detached(dumped_scene):
    correspondences = build_anchor_correspondences(dumped_scene)

    assert correspondences.count == 3
    assert correspondences.trajectory_indices.tolist() == [0, 1, 2]
    assert correspondences.query_slots.tolist() == [0, 0, 0]
    assert correspondences.query_times.tolist() == [0, 0, 0]
    assert not correspondences.rows.requires_grad
    assert not correspondences.columns.requires_grad
    assert correspondences.rows.tolist() == [25, 30, 27]
    assert correspondences.columns.tolist() == [22, 28, 34]
    assert len(set(zip(
        correspondences.rows.tolist(),
        correspondences.columns.tolist(),
    ))) == 3


def test_query_pointmap_anchor_is_gathered_and_detached(
    dumped_scene,
    monkeypatch,
):
    correspondences = build_anchor_correspondences(dumped_scene)
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    pointmaps = torch.arange(
        dumped_scene.num_observations * height * width * 3,
        dtype=torch.float32,
    ).reshape(1, dumped_scene.num_observations, height, width, 3)
    pointmaps.requires_grad_(True)
    monkeypatch.setattr(
        sparse_module,
        "_predicted_pointmaps",
        lambda raw: pointmaps,
    )
    raw = {
        "track_query_idx": dumped_scene.track_query_observation_slots,
    }

    anchors = gather_query_anchor_points(
        raw,
        dumped_scene,
        correspondences,
    )

    expected = pointmaps[
        0,
        0,
        correspondences.rows,
        correspondences.columns,
    ]
    torch.testing.assert_close(anchors, expected)
    assert not anchors.requires_grad
    with pytest.raises(ValueError, match="does not match"):
        gather_query_anchor_points(
            {"track_query_idx": torch.tensor([1])},
            dumped_scene,
            correspondences,
        )


def test_perfect_sparse_tracks_have_numerical_zero_loss(dumped_scene):
    correspondences = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    assert result.metric_error.item() == pytest.approx(0.0, abs=1e-8)
    assert result.sample_count == correspondences.count * 8


def test_known_perturbation_increases_loss(dumped_scene):
    correspondences = build_anchor_correspondences(dumped_scene)
    perfect, query_anchors = _perfect_raw_tracks(
        dumped_scene,
        correspondences,
    )
    baseline = sparse_tracking_loss(
        perfect,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )
    perturbed_tracks = perfect["track_multi"].clone()
    perturbed_tracks[
        0,
        0,
        7,
        correspondences.rows[0],
        correspondences.columns[0],
        0,
    ] += 0.2
    perturbed = sparse_tracking_loss(
        {
            "track_multi": perturbed_tracks,
            "track_query_idx": perfect["track_query_idx"],
        },
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert perturbed.loss.item() > baseline.loss.item()
    assert perturbed.metric_error.item() > baseline.metric_error.item()


def test_invisible_target_does_not_contribute(dumped_scene):
    correspondences = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    trajectory_index = int(correspondences.trajectory_indices[0])
    row = int(correspondences.rows[0])
    column = int(correspondences.columns[0])
    # Slot 7 is camera 1 / time 3.
    dumped_scene.visibility[1, 3, trajectory_index] = False
    raw["track_multi"][0, 0, 7, row, column, 0] += 100.0

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    assert result.metric_error.item() == pytest.approx(0.0, abs=1e-8)
    assert result.sample_count == correspondences.count * 8 - 1


def test_loss_preserves_eight_observation_axis(dumped_scene):
    correspondences = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    with pytest.raises(ValueError, match="observation slots"):
        sparse_tracking_loss(
            {
                "track_multi": raw["track_multi"][:, :, :7],
                "track_query_idx": raw["track_query_idx"],
            },
            dumped_scene,
            correspondences,
            _identity_alignment(),
            query_anchors,
        )


def test_nonidentity_sim3_absolute_position_loss_is_zero(dumped_scene):
    correspondences = build_anchor_correspondences(dumped_scene)
    angle = torch.tensor(0.35)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    alignment = DetachedSim3(
        scale=torch.tensor(1.7),
        rotation=rotation,
        translation=torch.tensor([0.4, -0.8, 1.2]),
    )
    dumped_scene.track_upscaling_factor = 2.5
    query_anchors = torch.tensor(
        [
            [-0.4, 0.2, 1.1],
            [0.3, -0.7, 2.0],
            [1.2, 0.1, -0.5],
        ],
        requires_grad=True,
    )
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    tracks = torch.zeros(
        1,
        1,
        dumped_scene.num_observations,
        height,
        width,
        3,
    )
    for item, trajectory_index in enumerate(
        correspondences.trajectory_indices.tolist()
    ):
        row = int(correspondences.rows[item])
        column = int(correspondences.columns[item])
        for slot, original_time in enumerate(dumped_scene.slot_times.tolist()):
            target = dumped_scene.trajectories_world[
                original_time,
                trajectory_index,
            ]
            target_in_predicted_world = (
                (target - alignment.translation) @ alignment.rotation
            ) / alignment.scale
            tracks[0, 0, slot, row, column] = (
                target_in_predicted_world - query_anchors.detach()[item]
            )
    tracks.requires_grad_(True)
    raw = {
        "track_multi": tracks,
        "track_query_idx": dumped_scene.track_query_observation_slots,
    }

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        alignment,
        query_anchors,
    )
    result.loss.backward()

    assert result.loss.item() == pytest.approx(0.0, abs=2e-7)
    assert result.metric_error.item() == pytest.approx(0.0, abs=2e-6)
    assert query_anchors.grad is None
    assert tracks.grad is not None


def test_depth_inconsistent_rounded_anchor_is_rejected(dumped_scene):
    all_correspondences = build_anchor_correspondences(dumped_scene)
    row = int(all_correspondences.rows[0])
    column = int(all_correspondences.columns[0])
    transform = dumped_scene.observations[0].image_transform
    original_row = int(np.rint((row + transform.crop_top) / transform.scale_y))
    original_column = int(
        np.rint((column + transform.crop_left) / transform.scale_x)
    )
    dumped_scene.depth0[0, 0, original_row, original_column] = 8.0

    filtered = build_anchor_correspondences(dumped_scene)

    assert filtered.trajectory_indices.tolist() == [1, 2]


def test_duplicate_dense_anchor_keeps_the_best_depth_match(dumped_scene):
    farther = torch.tensor([-1.0, -0.5, 5.05])
    nearer = torch.tensor([-1.0, -0.5, 5.0])
    dumped_scene.query_points[0, 1:] = farther
    dumped_scene.trajectories_world[0, 0] = farther
    dumped_scene.query_points[1, 1:] = nearer
    dumped_scene.trajectories_world[0, 1] = nearer

    correspondences = build_anchor_correspondences(dumped_scene)

    assert correspondences.trajectory_indices.tolist() == [1, 2]
    pixels = list(zip(
        correspondences.rows.tolist(),
        correspondences.columns.tolist(),
    ))
    assert len(pixels) == len(set(pixels))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query_slots", -1, "query slots must be non-negative"),
        ("trajectory_indices", -1, "trajectory index"),
        ("query_times", -1, "query time"),
    ],
)
def test_sparse_loss_rejects_wrapping_negative_indices(
    dumped_scene,
    field,
    value,
    message,
):
    correspondences = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    values = {
        name: getattr(correspondences, name).clone()
        for name in (
            "trajectory_indices",
            "query_slots",
            "query_times",
            "rows",
            "columns",
        )
    }
    values[field][0] = value
    invalid = SparseCorrespondences(**values)

    with pytest.raises(ValueError, match=message):
        sparse_tracking_loss(
            raw,
            dumped_scene,
            invalid,
            _identity_alignment(),
            query_anchors,
        )


class _TinyPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_index_embedding = nn.Embedding(4, 2)
        self.frozen_backbone_weight = nn.Parameter(torch.ones(1))


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = _TinyPretrained()


def _tiny_arc():
    model = Arc.__new__(Arc)
    nn.Module.__init__(model)
    model.backbone = _TinyBackbone()
    model.head = nn.Linear(1, 1)
    model.cam_dec = nn.Linear(1, 1)
    model.motion_decoder = nn.Linear(1, 1)
    model.track_head = nn.Linear(1, 1)
    model.set_freeze("temporal_tracking")
    return model


def test_only_temporal_tracking_parameters_receive_gradients(dumped_scene):
    model = _tiny_arc()
    correspondences = build_anchor_correspondences(dumped_scene)
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    value = (
        model.backbone.pretrained.time_index_embedding.weight.sum()
        + model.backbone.pretrained.frozen_backbone_weight.sum()
        + model.head.weight.sum()
        + model.head.bias.sum()
        + model.cam_dec.weight.sum()
        + model.cam_dec.bias.sum()
        + model.motion_decoder.weight.sum()
        + model.motion_decoder.bias.sum()
        + model.track_head.weight.sum()
        + model.track_head.bias.sum()
    )
    tracks = value.expand(
        1,
        1,
        dumped_scene.num_observations,
        height,
        width,
        3,
    )
    query_anchors = dumped_scene.trajectories_world[
        correspondences.query_times,
        correspondences.trajectory_indices,
    ]
    result = sparse_tracking_loss(
        {
            "track_multi": tracks,
            "track_query_idx": dumped_scene.track_query_observation_slots,
        },
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )
    result.loss.backward()

    expected_prefixes = (
        "backbone.pretrained.time_index_embedding.",
        "motion_decoder.",
        "track_head.",
    )
    for name, parameter in model.named_parameters():
        should_train = name.startswith(expected_prefixes)
        assert parameter.requires_grad is should_train
        assert (parameter.grad is not None) is should_train


def test_save_reload_preserves_temporal_embedding(tmp_path):
    source = _tiny_arc()
    with torch.no_grad():
        source.backbone.pretrained.time_index_embedding.weight.fill_(3.25)
        source.motion_decoder.weight.fill_(1.5)
        source.track_head.bias.fill_(-2.0)
    checkpoint = save_temporal_tracking_checkpoint(
        source,
        tmp_path / "temporal_tracking.pt",
    )
    target = _tiny_arc()

    load_temporal_tracking_checkpoint(target, checkpoint)

    torch.testing.assert_close(
        target.backbone.pretrained.time_index_embedding.weight,
        source.backbone.pretrained.time_index_embedding.weight,
    )
    torch.testing.assert_close(
        target.motion_decoder.weight,
        source.motion_decoder.weight,
    )
    torch.testing.assert_close(
        target.track_head.bias,
        source.track_head.bias,
    )
