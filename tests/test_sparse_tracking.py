from __future__ import annotations

import sys
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


def _write_scene(
    root: Path,
    *,
    scene_name: str = "0000",
    time_count: int = 4,
    view_count: int = 2,
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
    extrinsics[..., :3, :3] = np.eye(3, dtype=np.float32)
    # Camera c has centre (+c,0,0), hence world-to-camera translation -c.
    for camera in range(view_count):
        extrinsics[camera, :, 0, 3] = -float(camera)
    depth0 = np.full(
        (view_count, 1, height, width),
        fill_value=5.0,
        dtype=np.float32,
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
