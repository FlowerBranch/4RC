"""The held-out eval's visual dump: what it holds, what it reads, what it leaves alone.

``--eval_visual_scenes`` writes what the plain arm saw of a held-out scene --
the aligned reconstruction, both camera sets, the model's own anchors and the
scored tracks -- for ``arc.viz.viser_multiview_eval``.  Every array is pinned
here to the function the rest of the eval already trusts for it, so the viewer
draws what the readouts measure; and the flags are pinned to change nothing
else: no byte of any other output, no RNG draw, no checkpoint or resume key.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import arc.training.sparse_tracking as sparse_module
import train_temporal_tracking as train_cli
from arc.models.arc.utils.transform import pose_encoding_to_extri_intri
from arc.training import (
    OPTIONAL_VISUAL_KEYS,
    VISUAL_KEYS,
    VISUAL_METADATA_KEYS,
    VISUAL_TRACK_KEYS,
    aligned_query_anchors,
    build_anchor_correspondences,
    build_visual_arrays,
    capture_rng_state,
    fit_scene_sim3,
    gather_query_anchor_points,
    plan_record,
    query_anchor_errors,
    read_scene_predictions,
    read_visual_dump,
    visual_geometry,
    write_visual_dump,
)
from scene_fixtures import fixture_scene
from test_manifest_plan import _record
from test_sparse_tracking import (
    _assert_same_rng_state,
    _ground_truth_raw_reconstruction,
    _skewed_alignment,
    _steered_to,
)
from test_trainer_loop import (
    _EVAL_ALPHA,
    _FOUR_ANCHOR_SPEC,
    _FOUR_CAMERA_WINDOW,
    _FakeArc,
    _GroundTruthPoseArc,
    _cpu_eval_scene,
    _loop_args,
    _plans,
    _step_scene,
    _validator_args,
)
from test_track_refinement import _RefiningFakeArc

_IDENTITY = {"checkpoint_dir": "released/4rc", "commit": "0123abcd"}
_STEP = 6
# The fixture's factor is 1.0, which would hide a missing lift to metres.
_UPSCALING = 2.5
# The steered four-camera window: rows on anchors 3:0 and 1:0 only, out of
# camera order, so a dump that files rows by position or sorts them fails.
_STEERED = _steered_to({0: 3, 1: 3, 2: 1})


def _evaluate(output_dir, scenes, model, *, query_anchors=("0:0",), **kwargs):
    """evaluate_held_out over ``scenes`` (name -> scene) at step 6."""

    return train_cli.evaluate_held_out(
        model=model,
        plans=[
            plan_record(_record(seq_name=name), budget=48, stride=2) for name in scenes
        ],
        scene_provider=lambda plan: scenes[plan.seq_name],
        precision="32",
        huber_delta_m=0.05,
        step=_STEP,
        output_dir=output_dir,
        query_anchors=list(query_anchors),
        confidence_alpha=_EVAL_ALPHA,
        **kwargs,
    )


def _dumping(*names):
    return {"visual_scenes": frozenset(names), "run_identity": dict(_IDENTITY)}


def _dumped(output_dir, scene="0000"):
    return read_visual_dump(
        Path(output_dir) / "eval" / f"step-{_STEP}" / "visual" / f"{scene}.npz"
    )


def _bundle(output_dir, scene="0000"):
    return read_scene_predictions(
        Path(output_dir) / "eval" / f"step-{_STEP}" / "pred" / f"{scene}.npz"
    )


def _files(directory) -> dict[str, bytes]:
    """Every file under ``directory``, relative path -> bytes."""

    root = Path(directory)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _scaled_scene(monkeypatch, **scene_kwargs):
    """The scaled-gauge window, its metric factor lifted off 1.0."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled", **scene_kwargs)
    # The same views list, so _GroundTruthPoseArc still finds its own scene.
    return dataclasses.replace(scene, track_upscaling_factor=_UPSCALING)


# ------------------------------------------------------------------ contents ---


@pytest.mark.parametrize("merge", (False, True), ids=("per_slot", "merged"))
def test_the_dump_holds_every_documented_key_at_its_shape_and_dtype(
    tmp_path, monkeypatch, merge
):
    """The schema as the plan documents it, written out as the expectation.

    Stride 3 on the 56-pixel grid, so the strided axes are 19 long and a
    stride read as 4 or a grid read as H // stride both fail. The fake emits
    no depth confidence, so ``depth_conf`` is absent rather than filled.
    """

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56, views_per_time=2 if merge else 1)

    metrics = _evaluate(
        tmp_path,
        {"0000": scene},
        model,
        merge_synchronized_slots=merge,
        visual_stride=3,
        **_dumping("0000"),
    )
    arrays, metadata = _dumped(tmp_path)

    slots, grid, rows, times, tracks = 8, 19, 3, 4, 3
    expected = {
        "slot_camera_id": (np.int64, (slots,)),
        "slot_original_time": (np.int64, (slots,)),
        "slot_time_index": (np.int64, (slots,)),
        "image_size": (np.int64, (2,)),
        "pixel_rows": (np.int64, (grid,)),
        "pixel_columns": (np.int64, (grid,)),
        "pred_points_m": (np.float32, (slots, grid, grid, 3)),
        "rgb": (np.uint8, (slots, grid, grid, 3)),
        "gt_points_m": (np.float32, (slots, grid, grid, 3)),
        "gt_points_valid": (np.bool_, (slots, grid, grid)),
        "pred_camera_rotation": (np.float32, (slots, 3, 3)),
        "pred_camera_center_m": (np.float32, (slots, 3)),
        "pred_intrinsics": (np.float32, (slots, 3, 3)),
        "gt_camera_rotation": (np.float32, (slots, 3, 3)),
        "gt_camera_center_m": (np.float32, (slots, 3)),
        "gt_intrinsics": (np.float32, (slots, 3, 3)),
        "anchor_pred_m": (np.float32, (rows, 3)),
        "anchor_true_m": (np.float32, (rows, 3)),
        "track_pred": (np.float32, (times, tracks, 3)),
        "track_gt": (np.float32, (times, tracks, 3)),
        "track_gt_vis_any": (np.bool_, (times, tracks)),
        "track_query_points": (np.float32, (tracks, 4)),
        "track_conf": (np.float32, (times, tracks)),
    }
    assert set(arrays) == set(expected) | {"anchor_key", "metadata"}
    assert set(VISUAL_KEYS) - set(arrays) == set(OPTIONAL_VISUAL_KEYS) == {"depth_conf"}
    for key, (dtype, shape) in expected.items():
        assert arrays[key].dtype == np.dtype(dtype), key
        assert arrays[key].shape == shape, key
    assert arrays["anchor_key"].dtype.kind == "U"
    assert arrays["anchor_key"].shape == (rows,)
    np.testing.assert_array_equal(arrays["image_size"], [56, 56])
    np.testing.assert_array_equal(arrays["pixel_rows"], np.arange(0, 56, 3))
    np.testing.assert_array_equal(arrays["pixel_columns"], np.arange(0, 56, 3))

    entry = metrics["per_scene"][0]
    assert metadata == {
        "scene": "0000",
        "step": _STEP,
        "output_dir": str(tmp_path),
        **_IDENTITY,
        "merge_synchronized_slots": merge,
        "refine_iters": 1,
        "depth_input": False,
        "camera_input": False,
        "oracle_query_anchor": False,
        "ground_truth_query_anchor": False,
        "query_anchors": ["0:0"],
        "base_ratio": entry["drift"]["base_ratio"],
        "anchor_error_m": entry["anchor_error_m"],
    }
    assert list(metadata) == list(VISUAL_METADATA_KEYS)


def test_the_dumped_clouds_are_the_aligned_pointmaps_and_the_ground_truth(
    tmp_path, monkeypatch
):
    """Each cloud against the function the eval already reads it through.

    The predicted cloud is the planted gauge carried home by the fitted Sim(3)
    and lifted to metres -- read through sparse_tracking's own name, so the
    planted pointmaps are what the dump saw. The ground truth is
    _metric_pointmap_at_anchor at the same pixels. The fixture's frames are
    filled with 20 * camera + time, so the RGB is known exactly.
    """

    scene = _scaled_scene(monkeypatch)
    model = _FakeArc(scene.num_observations, 56, 56)

    _evaluate(tmp_path, {"0000": scene}, model, **_dumping("0000"))
    arrays, _ = _dumped(tmp_path)

    alignment, _ = fit_scene_sim3({}, scene)
    planted = sparse_module._predicted_pointmaps({})
    np.testing.assert_allclose(
        arrays["pred_points_m"],
        (alignment.apply_points(planted)[0, :, ::4, ::4] * _UPSCALING).numpy(),
        rtol=1e-6,
        atol=1e-5,
    )
    for observation in scene.observations:
        slot = observation.slot
        assert arrays["slot_camera_id"][slot] == observation.camera_id
        assert arrays["slot_original_time"][slot] == observation.original_time
        assert arrays["slot_time_index"][slot] == observation.semantic_time_index
        world_points, valid = sparse_module._metric_pointmap_at_anchor(scene, slot)
        np.testing.assert_array_equal(
            arrays["gt_points_m"][slot],
            (world_points[::4, ::4] * _UPSCALING).astype(np.float32),
        )
        np.testing.assert_array_equal(arrays["gt_points_valid"][slot], valid[::4, ::4])
        assert np.all(
            arrays["rgb"][slot] == 20 * observation.camera + observation.original_time
        )
    # Not vacuous: the gauge is not the identity, and the lift is not 1.
    assert not np.allclose(arrays["pred_points_m"], planted[0, :, ::4, ::4].numpy())
    assert arrays["gt_points_valid"].any()


def test_the_dumped_cameras_compose_as_the_drift_report_does(tmp_path, monkeypatch):
    """Both camera sets, and the readouts the dump's own centres reproduce.

    The predicted camera is the token camera through the drift report's
    composition, slot by slot in its float32; the ground-truth camera inverts
    the extrinsics, its intrinsics mapped onto the model grid. The dump's
    centres then give back the drift report's camera_center_error_m and every
    base_ratio, so what the viewer draws is the baseline the readout judged.
    """

    scene = _scaled_scene(monkeypatch, **_FOUR_CAMERA_WINDOW)
    model = _GroundTruthPoseArc([scene], [{1: 0.9, 2: 1.25, 3: 0.6}])

    metrics = _evaluate(
        tmp_path,
        {"0000": scene},
        model,
        query_anchors=_FOUR_ANCHOR_SPEC,
        **_dumping("0000"),
    )
    arrays, metadata = _dumped(tmp_path)

    raw = model.reconstructions[id(scene.views)]
    alignment, _ = fit_scene_sim3({}, scene)
    camera_to_world, intrinsics = pose_encoding_to_extri_intri(
        raw["pose_enc"].float(), (56, 56)
    )
    for observation in scene.observations:
        slot = observation.slot
        centre = alignment.apply_points(camera_to_world[0, slot, :3, 3][None, :])[0]
        np.testing.assert_allclose(
            arrays["pred_camera_center_m"][slot],
            (centre * _UPSCALING).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            arrays["pred_camera_rotation"][slot],
            (alignment.rotation @ camera_to_world[0, slot, :3, :3]).numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            arrays["pred_intrinsics"][slot], intrinsics[0, slot].numpy(), rtol=1e-6
        )
        world_to_camera = scene.extrinsics_world_to_camera[
            observation.camera, observation.original_time
        ].double()
        rotation, translation = world_to_camera[:3, :3], world_to_camera[:3, 3]
        np.testing.assert_allclose(
            arrays["gt_camera_center_m"][slot],
            (-(rotation.T @ translation) * _UPSCALING).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            arrays["gt_camera_rotation"][slot], rotation.T.numpy(), atol=1e-7
        )
        np.testing.assert_allclose(
            arrays["gt_intrinsics"][slot],
            observation.image_transform.intrinsics_to_output(
                scene.intrinsics[observation.camera, observation.original_time].numpy()
            ),
            rtol=1e-6,
        )

    drift = metrics["per_scene"][0]["drift"]
    predicted = arrays["pred_camera_center_m"].astype(np.float64)
    truth = arrays["gt_camera_center_m"].astype(np.float64)
    distances = np.linalg.norm(predicted - truth, axis=1)
    assert distances.mean() == pytest.approx(
        drift["pose"]["camera_center_error_m"]["mean"], rel=1e-5
    )
    assert distances.max() == pytest.approx(
        drift["pose"]["camera_center_error_m"]["max"], rel=1e-5
    )
    reference = scene.query_observation_slot
    ratios = {}
    for (camera, time), slot in zip(scene.query_anchors, scene.anchor_observation_slots):
        if slot == reference:
            continue
        ratios[f"{camera}:{time}"] = np.linalg.norm(
            predicted[slot] - predicted[reference]
        ) / np.linalg.norm(truth[slot] - truth[reference])
    assert drift["base_ratio"]["0:0"] is None
    assert ratios == {
        key: pytest.approx(value, rel=1e-5)
        for key, value in drift["base_ratio"].items()
        if value is not None
    }
    assert len(ratios) == 3
    assert metadata["base_ratio"] == drift["base_ratio"]


def test_each_dumped_camera_projects_its_own_cloud_onto_the_model_grid():
    """The frusta the viewer draws are the cameras of the clouds it draws.

    Independent of how either is composed: a camera dumped as camera-to-world
    rotation, centre and model-grid intrinsics must project the cloud dumped
    beside it back onto the pixels that cloud was read at. The predicted pair
    lands exactly on the model grid -- one Sim(3) carries both, and a
    similarity survives the perspective division. The ground-truth pair lands
    exactly on the source pixel the nearest-pixel gather read for each model
    pixel, carried onto the model grid -- up to half a source pixel off the
    model pixel inside the frame, more where the gather clips at its border,
    so that is the expectation rather than a tolerance. At a non-identity transform
    (size 504 on the fixture's 56-pixel frames: scale 9 and a crop), through a
    gauge whose scale, rotation and translation are all non-trivial, at a
    metric factor that is not 1, with camera 1 yawed and pitched:
    original-grid intrinsics, a dropped translation, a rotation composed in
    the wrong order or stored transposed, or a lift applied to one side only
    each misses by pixels. The eval fixtures cannot see the first or the
    rotations: their transform is the identity and their cameras unrotated.
    """

    scene = dataclasses.replace(
        fixture_scene(cameras=(0, 1), times=(0, 1), size=504, rotated_camera=1),
        track_upscaling_factor=_UPSCALING,
    )
    transform = scene.observations[0].image_transform
    assert (transform.output_height, transform.output_width) == (378, 504)
    assert transform.scale_x == 9.0 and transform.crop_top > 0
    raw = _ground_truth_raw_reconstruction(scene)
    correspondences, _ = build_anchor_correspondences(scene)
    assert correspondences.count > 0
    model_anchors, _ = gather_query_anchor_points(raw, scene, correspondences)

    geometry = visual_geometry(
        raw, scene, correspondences, _skewed_alignment(), model_anchors, stride=5
    )

    columns, rows = np.meshgrid(geometry["pixel_columns"], geometry["pixel_rows"])
    source_rows, source_columns = transform.output_to_original_indices()
    expected = {
        "pred": np.stack([columns, rows], axis=-1).astype(np.float64),
        "gt": transform.original_to_output(
            np.stack([source_columns[columns], source_rows[rows]], axis=-1)
        ),
    }
    # Not vacuous: the gather is off the model pixel by more than the
    # predicted tolerance almost everywhere.
    assert np.abs(expected["gt"] - expected["pred"]).max() > 4.0
    for side, pixels in expected.items():
        for slot in range(scene.num_observations):
            points = geometry[f"{side}_points_m"][slot].astype(np.float64)
            rotation = geometry[f"{side}_camera_rotation"][slot].astype(np.float64)
            centre = geometry[f"{side}_camera_center_m"][slot].astype(np.float64)
            intrinsics = geometry[f"{side}_intrinsics"][slot].astype(np.float64)
            # Camera coordinates R^T (p - c), one row per point.
            projected = ((points - centre) @ rotation) @ intrinsics.T
            projected = projected[..., :2] / projected[..., 2:]
            keep = (
                np.ones(rows.shape, dtype=bool)
                if side == "pred"
                else geometry["gt_points_valid"][slot]
            )
            assert keep.sum() > 100
            np.testing.assert_allclose(projected[keep], pixels[keep], atol=1e-2)


def test_the_dumped_rgb_inverts_the_loaders_normalisation_over_the_full_range():
    """The colour is the input image's own, every value 0..255 exactly.

    The eval fixtures' frames hold values up to 23, where a scale off by half
    a count still rounds back; a ramp through the loader's own ImgNorm covers
    the whole range at every stride pixel.
    """

    from PIL import Image

    from arc.dust3r.utils.image import ImgNorm

    scene = fixture_scene(cameras=(0, 1), times=(0, 1), size=56)
    height, width = scene.views[0]["img"].shape[-2:]
    ramps = []
    for slot, view in enumerate(scene.views):
        ramp = ((np.arange(height * width * 3) + 61 * slot) % 256).astype(np.uint8)
        ramp = ramp.reshape(height, width, 3)
        view["img"] = ImgNorm(Image.fromarray(ramp))[None]
        ramps.append(ramp)
    raw = _ground_truth_raw_reconstruction(scene)
    correspondences, _ = build_anchor_correspondences(scene)
    model_anchors, _ = gather_query_anchor_points(raw, scene, correspondences)

    geometry = visual_geometry(
        raw, scene, correspondences, _skewed_alignment(), model_anchors, stride=3
    )

    for slot, ramp in enumerate(ramps):
        np.testing.assert_array_equal(geometry["rgb"][slot], ramp[::3, ::3])
    assert geometry["rgb"].min() == 0 and geometry["rgb"].max() == 255


@pytest.mark.parametrize(
    "diagnostic",
    ({}, {"oracle_query_anchor": True}, {"ground_truth_query_anchor": True}),
    ids=("model", "oracle", "ground_truth"),
)
def test_the_dumped_anchors_are_what_the_anchor_readout_measures(
    tmp_path, monkeypatch, diagnostic
):
    """The model's own anchors, by construction, whichever anchor was scored.

    The expectation is the same under all three arms: aligned_query_anchors on
    the model's own gather, which is query_anchor_errors' composition, so the
    stored rows are its output exactly and their distances are the readout's.
    At the live four-anchor spec with rows steered onto two anchors out of
    camera order, so a key filed by position or sorted fails.
    """

    scene = _scaled_scene(monkeypatch, invisible=_STEERED, **_FOUR_CAMERA_WINDOW)
    model = _GroundTruthPoseArc([scene], [{1: 0.9, 2: 1.25, 3: 0.6}])

    metrics = _evaluate(
        tmp_path,
        {"0000": scene},
        model,
        query_anchors=_FOUR_ANCHOR_SPEC,
        **diagnostic,
        **_dumping("0000"),
    )
    arrays, metadata = _dumped(tmp_path)

    correspondences, _ = build_anchor_correspondences(scene)
    alignment, _ = fit_scene_sim3({}, scene)
    model_anchors, frame = gather_query_anchor_points({}, scene, correspondences)
    assert frame == "model"
    aligned, truth = aligned_query_anchors(model_anchors, scene, correspondences, alignment)
    np.testing.assert_array_equal(
        arrays["anchor_pred_m"], (aligned * _UPSCALING).numpy().astype(np.float32)
    )
    np.testing.assert_array_equal(
        arrays["anchor_true_m"], (truth * _UPSCALING).numpy().astype(np.float32)
    )
    keys = [f"{camera}:{time}" for camera, time in scene.query_anchors]
    assert arrays["anchor_key"].tolist() == [
        keys[slot] for slot in correspondences.query_slots.tolist()
    ]
    readout = query_anchor_errors(model_anchors, scene, correspondences, alignment)
    distances = np.linalg.norm(
        arrays["anchor_pred_m"].astype(np.float64) - arrays["anchor_true_m"], axis=1
    )
    for key in keys:
        np.testing.assert_allclose(
            distances[arrays["anchor_key"] == key], readout[key], rtol=1e-5, atol=1e-5
        )
    assert [readout[key].size for key in _FOUR_ANCHOR_SPEC] == [0, 2, 1, 0]
    assert metadata["anchor_error_m"] == metrics["per_scene"][0]["anchor_error_m"]
    assert metadata["oracle_query_anchor"] is bool(diagnostic.get("oracle_query_anchor"))
    assert metadata["ground_truth_query_anchor"] is bool(
        diagnostic.get("ground_truth_query_anchor")
    )


@pytest.mark.parametrize("merge", (False, True), ids=("per_slot", "merged"))
@pytest.mark.parametrize(
    "diagnostic", ({}, {"oracle_query_anchor": True}), ids=("model", "oracle")
)
def test_the_dumped_tracks_are_the_bundle_that_was_scored(
    tmp_path, monkeypatch, merge, diagnostic
):
    """Every track array is the bundle's own, under either head and anchor."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56, views_per_time=2 if merge else 1)

    _evaluate(
        tmp_path,
        {"0000": scene},
        model,
        merge_synchronized_slots=merge,
        **diagnostic,
        **_dumping("0000"),
    )
    arrays, _ = _dumped(tmp_path)
    bundle = _bundle(tmp_path)

    for key in VISUAL_TRACK_KEYS:
        np.testing.assert_array_equal(arrays[f"track_{key}"], bundle[key])


def test_the_tracks_follow_the_scored_anchor_while_the_anchors_stay_the_models(
    tmp_path, monkeypatch
):
    """The oracle moves what was scored, and the dump moves with it; the
    anchors the dump draws are the model's under both, as the readout's are."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56)

    dumps = {}
    for name, diagnostic in (("model", {}), ("oracle", {"oracle_query_anchor": True})):
        _evaluate(tmp_path / name, {"0000": scene}, model, **diagnostic, **_dumping("0000"))
        dumps[name], _ = _dumped(tmp_path / name)

    assert not np.array_equal(dumps["model"]["track_pred"], dumps["oracle"]["track_pred"])
    for key in VISUAL_KEYS:
        if key in ("track_pred", "metadata") or key not in dumps["model"]:
            continue
        np.testing.assert_array_equal(dumps["model"][key], dumps["oracle"][key], key)


@pytest.mark.parametrize("merge", (False, True), ids=("per_slot", "merged"))
@pytest.mark.parametrize("refine_iters", (1, 4))
def test_every_head_configuration_dumps_at_the_live_anchor_count(
    tmp_path, monkeypatch, merge, refine_iters
):
    """Per-slot and merged, K=1 and the live K=4, four anchors.

    _RefiningFakeArc emits a distinct field per pass and a valid pose, so the
    tracks dumped at K=4 are the final pass the bundle scored and the
    predicted cameras are finite.
    """

    scene = _cpu_eval_scene(
        monkeypatch, gauge="scaled", invisible=_STEERED, **_FOUR_CAMERA_WINDOW
    )
    model = _RefiningFakeArc(
        scene.num_observations, 56, 56, views_per_time=4 if merge else 1
    )

    _evaluate(
        tmp_path,
        {"0000": scene},
        model,
        query_anchors=_FOUR_ANCHOR_SPEC,
        merge_synchronized_slots=merge,
        refine_iters=refine_iters,
        **_dumping("0000"),
    )
    arrays, metadata = _dumped(tmp_path)
    bundle = _bundle(tmp_path)

    for key in VISUAL_TRACK_KEYS:
        np.testing.assert_array_equal(arrays[f"track_{key}"], bundle[key])
    assert metadata["refine_iters"] == refine_iters
    assert metadata["merge_synchronized_slots"] is merge
    assert arrays["slot_camera_id"].tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert np.isfinite(arrays["pred_camera_rotation"]).all()
    assert np.isfinite(arrays["pred_camera_center_m"]).all()
    assert sorted(set(arrays["anchor_key"].tolist())) == ["1:0", "3:0"]


def test_refinement_moves_the_dumped_tracks_and_nothing_else(tmp_path, monkeypatch):
    """K=4 dumps its final pass, not pass 0: the tracks differ from K=1's
    while the reconstruction, cameras and anchors -- which no pass touches --
    are identical."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _RefiningFakeArc(scene.num_observations, 56, 56)

    dumps = {}
    for refine_iters in (1, 4):
        _evaluate(
            tmp_path / str(refine_iters),
            {"0000": scene},
            model,
            refine_iters=refine_iters,
            **_dumping("0000"),
        )
        dumps[refine_iters], _ = _dumped(tmp_path / str(refine_iters))

    assert not np.array_equal(dumps[1]["track_pred"], dumps[4]["track_pred"])
    for key in VISUAL_KEYS:
        if key.startswith("track_") or key == "metadata" or key not in dumps[1]:
            continue
        np.testing.assert_array_equal(dumps[1][key], dumps[4][key], key)


def test_the_metadata_records_the_geometry_inputs_the_forward_received(
    tmp_path, monkeypatch
):
    scene = _cpu_eval_scene(
        monkeypatch, gauge="scaled", input_depth_max=24.0, input_camera_vectors=True
    )
    assert "depth_map" in scene.views[0] and "camera_vector" in scene.views[0]
    model = _FakeArc(scene.num_observations, 56, 56)

    _evaluate(tmp_path, {"0000": scene}, model, **_dumping("0000"))
    _, metadata = _dumped(tmp_path)

    assert metadata["depth_input"] is True
    assert metadata["camera_input"] is True


def test_the_dump_needs_no_bundle_and_adds_no_entry_key(tmp_path, monkeypatch):
    """With emit_predictions off the tracks are still the bundle's arrays,
    built the same way; no pred/ appears and the entry gains no key."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56)

    _evaluate(tmp_path / "bundled", {"0000": scene}, model, **_dumping("0000"))
    _evaluate(tmp_path / "off", {"0000": scene}, model, emit_predictions=False)
    metrics = _evaluate(
        tmp_path / "on", {"0000": scene}, model, emit_predictions=False, **_dumping("0000")
    )

    step = Path("eval") / f"step-{_STEP}"
    assert not (tmp_path / "on" / step / "pred").exists()
    assert "predicted_occluded_fraction" not in metrics["per_scene"][0]
    assert (tmp_path / "on" / step / "metrics.json").read_bytes() == (
        tmp_path / "off" / step / "metrics.json"
    ).read_bytes()
    unbundled, _ = _dumped(tmp_path / "on")
    bundled, _ = _dumped(tmp_path / "bundled")
    for key in VISUAL_TRACK_KEYS:
        np.testing.assert_array_equal(unbundled[f"track_{key}"], bundled[f"track_{key}"])


# ------------------------------------------------------------ off by default ---


def test_an_absent_or_empty_scene_set_writes_and_changes_nothing(tmp_path, monkeypatch):
    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56)

    absent = _evaluate(tmp_path / "absent", {"0000": scene}, model)
    empty = _evaluate(
        tmp_path / "empty", {"0000": scene}, model, visual_scenes=frozenset()
    )

    assert _files(tmp_path / "absent") == _files(tmp_path / "empty")
    assert not any("visual" in name for name in _files(tmp_path / "empty"))
    # Bytes, not parsed dicts: _FakeArc's zero quaternion leaves NaN drift
    # rotations, and NaN != NaN.
    assert json.dumps(absent) == json.dumps(empty)


@pytest.mark.parametrize("merge", (False, True), ids=("per_slot", "merged"))
def test_the_dump_adds_its_file_and_changes_no_other_byte(tmp_path, monkeypatch, merge):
    """Two scored scenes, one dumped: the only new file is that scene's dump,
    and metrics.json, both bundles and the returned metrics are byte for byte
    the flag-off run's."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56, views_per_time=2 if merge else 1)
    scenes = {"0000": scene, "0001": scene}

    off = _evaluate(tmp_path / "off", scenes, model, merge_synchronized_slots=merge)
    on = _evaluate(
        tmp_path / "on", scenes, model, merge_synchronized_slots=merge, **_dumping("0001")
    )

    off_files, on_files = _files(tmp_path / "off"), _files(tmp_path / "on")
    assert set(on_files) - set(off_files) == {f"eval/step-{_STEP}/visual/0001.npz"}
    assert {name: data for name, data in on_files.items() if "/visual/" not in name} == (
        off_files
    )
    assert json.dumps(on) == json.dumps(off)


def test_the_dump_draws_no_randomness(tmp_path, monkeypatch):
    """No draw at all, rather than one the eval's RNG restore would hide:
    nothing restores between the two snapshots here."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56)
    _evaluate(tmp_path / "bundle", {"0000": scene}, model)
    tracks = _bundle(tmp_path / "bundle")
    correspondences, _ = build_anchor_correspondences(scene)
    alignment, _ = fit_scene_sim3({}, scene)
    model_anchors, _ = gather_query_anchor_points({}, scene, correspondences)
    raw = _ground_truth_raw_reconstruction(scene)
    metadata = dict.fromkeys(VISUAL_METADATA_KEYS)
    before = capture_rng_state()

    geometry = visual_geometry(
        raw, scene, correspondences, alignment, model_anchors, stride=4
    )
    path = write_visual_dump(
        tmp_path / "dump.npz",
        build_visual_arrays(geometry=geometry, tracks=tracks, metadata=metadata),
    )
    read_visual_dump(path)

    _assert_same_rng_state(capture_rng_state(), before)


def test_the_eval_draws_nothing_with_the_dump_on(tmp_path, monkeypatch):
    """The eval-level check, made able to see a draw: with the eval's own
    restore disabled, a dumping eval leaves every stream where seeding put it."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56)
    monkeypatch.setattr(train_cli, "restore_rng_state", lambda state: None)
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    before = capture_rng_state()

    _evaluate(tmp_path, {"0000": scene}, model, **_dumping("0000"))

    assert _dumped(tmp_path)[0]["pred_points_m"].size
    _assert_same_rng_state(capture_rng_state(), before)


# ------------------------------------------------------------- the interface ---


def test_the_visual_parameters_are_keyword_only_and_off_by_default():
    parameters = inspect.signature(train_cli.evaluate_held_out).parameters
    for name, default in (
        ("visual_scenes", frozenset()),
        ("visual_stride", 4),
        ("run_identity", None),
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameters[name].default == default, name


def test_a_dump_without_a_run_identity_is_refused(tmp_path, monkeypatch):
    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56)

    with pytest.raises(ValueError, match="visual_scenes needs run_identity"):
        _evaluate(tmp_path, {"0000": scene}, model, visual_scenes=frozenset({"0000"}))
    assert not (tmp_path / "eval").exists()


def test_the_visual_flags_parse_to_off_and_take_names_or_all():
    parser = train_cli.build_arg_parser()
    base = ["--manifest", "m.jsonl"]

    defaults = parser.parse_args(base)
    assert defaults.eval_visual_scenes == []
    assert defaults.eval_visual_stride == 4
    assert parser.parse_args(base + ["--eval_visual_scenes"]).eval_visual_scenes == []
    assert parser.parse_args(base + ["--eval_visual_scenes", "all"]).eval_visual_scenes == [
        "all"
    ]
    named = parser.parse_args(
        base + ["--eval_visual_scenes", "0001", "0042", "--eval_visual_stride", "2"]
    )
    assert named.eval_visual_scenes == ["0001", "0042"]
    assert named.eval_visual_stride == 2


_HELD_OUT = {"val_scenes_file": "val.json", "val_data_root": "/held/out/dir"}


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"eval_visual_scenes": ["0000"], "eval_every": 500}, "needs --val_scenes_file"),
        (
            {"eval_visual_scenes": ["0000"], "eval_every": 0, **_HELD_OUT},
            "needs a nonzero --eval_every",
        ),
        (
            {"eval_visual_scenes": ["all", "0000"], "eval_every": 500, **_HELD_OUT},
            "or 'all' alone",
        ),
        (
            {"eval_visual_scenes": ["0000", "0000"], "eval_every": 500, **_HELD_OUT},
            re.escape("names ['0000'] more than once"),
        ),
        ({"eval_visual_stride": 0}, "must be at least 1"),
    ),
    ids=("no_held_out_set", "eval_off", "all_with_names", "duplicate", "stride"),
)
def test_the_visual_flags_are_refused_where_they_cannot_write(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        train_cli._validate_args(_validator_args(tmp_path, **overrides))


def test_the_visual_flags_validate_beside_a_held_out_eval(tmp_path):
    for scenes in (["all"], ["0000", "0042"], []):
        train_cli._validate_args(
            _validator_args(
                tmp_path, eval_visual_scenes=scenes, eval_every=500, **_HELD_OUT
            )
        )


def test_the_visual_flags_are_eval_only_and_never_resumed(tmp_path):
    """Stored nowhere in the checkpoint and compared by no tier, like the
    anchor diagnostics: a segment may switch the dump on to look at a
    checkpoint and off again to train."""

    stored = train_cli._checkpoint_settings(
        _loop_args(tmp_path, eval_visual_scenes=["all"], eval_visual_stride=2)
    )
    assert stored == train_cli._checkpoint_settings(_loop_args(tmp_path))
    for key in ("eval_visual_scenes", "eval_visual_stride"):
        assert key not in stored
        assert key not in train_cli._RESUME_SETTINGS_REFUSED
        assert key not in train_cli._RESUME_SETTINGS_WARNED
        assert key not in train_cli._RESUME_SETTINGS_ABSENT_DEFAULTS
    train_cli.check_resume_settings(
        stored, _loop_args(tmp_path, eval_visual_scenes=["0000"], eval_visual_stride=8)
    )


# ------------------------------------------------------------------ the loop ---


def _no_op_step(*, step, plan, **_):
    return train_cli.StepOutcome(
        step=step,
        seq_name=plan.seq_name,
        loss=0.0,
        metric_error_m=0.0,
        sample_count=1,
        alignment_scale=1.0,
        alignment_residual_m=0.0,
        learning_rates=[1e-3],
        gradient_norms={},
    )


def _run(output_dir, scenes, default_scene, **overrides):
    """Four no-op steps with a real eval at 2 and 4 over ``scenes``.

    Seeded, so two runs evaluate the same weights: the no-op step never moves
    them, and _FakeArc's are drawn at construction.
    """

    torch.manual_seed(0)
    model = _FakeArc(default_scene.num_observations, 56, 56)
    return train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(output_dir, num_steps=4, eval_every=2, **overrides),
        scene_provider=lambda plan: scenes.get(plan.seq_name, default_scene),
        step_fn=_no_op_step,
        output_dir=output_dir,
        val_plans=[
            plan_record(_record(seq_name=name), budget=48, stride=2) for name in scenes
        ],
    )


def test_the_loop_dumps_the_named_scene_at_every_eval_and_reads_the_commit_once(
    tmp_path, monkeypatch
):
    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    reads = []
    monkeypatch.setattr(
        train_cli, "_fork_commit", lambda repo: reads.append(repo) or "c0ffee"
    )

    _run(
        tmp_path,
        {"0000": scene, "0001": scene},
        scene,
        eval_visual_scenes=["0000"],
        checkpoint_dir="released/4rc",
    )

    assert reads == [Path(train_cli.__file__).resolve().parent]
    for step in (2, 4):
        directory = tmp_path / "eval" / f"step-{step}" / "visual"
        assert sorted(path.name for path in directory.iterdir()) == ["0000.npz"]
        _, metadata = read_visual_dump(directory / "0000.npz")
        assert metadata["step"] == step
        assert metadata["commit"] == "c0ffee"
        assert metadata["checkpoint_dir"] == "released/4rc"
        assert metadata["output_dir"] == str(tmp_path)


def test_all_dumps_every_scored_held_out_scene(tmp_path, monkeypatch):
    good = _cpu_eval_scene(monkeypatch)
    bad = _step_scene(monkeypatch, query_anchors=((0, 2),))
    monkeypatch.setattr(train_cli, "_fork_commit", lambda repo: "c0ffee")

    _run(
        tmp_path,
        {"good": good, "bad": bad},
        good,
        eval_visual_scenes=["all"],
        checkpoint_dir="released/4rc",
    )

    for step in (2, 4):
        directory = tmp_path / "eval" / f"step-{step}" / "visual"
        assert sorted(path.name for path in directory.iterdir()) == ["good.npz"]


@pytest.mark.parametrize(
    "named, message",
    ((["nope"], "which are not held-out scenes"), (["bad"], "which no anchor can")),
    ids=("unknown", "unsupervisable"),
)
def test_a_named_scene_that_cannot_be_dumped_is_refused_before_step_zero(
    tmp_path, monkeypatch, named, message
):
    good = _cpu_eval_scene(monkeypatch)
    bad = _step_scene(monkeypatch, query_anchors=((0, 2),))
    reads = []
    monkeypatch.setattr(train_cli, "_fork_commit", lambda repo: reads.append(repo))

    with pytest.raises(ValueError, match=message):
        _run(
            tmp_path,
            {"good": good, "bad": bad},
            good,
            eval_visual_scenes=named,
            checkpoint_dir="released/4rc",
        )
    # Refused at the preflight, before the history is opened or any step runs.
    assert not (tmp_path / "history.jsonl").exists()
    assert not (tmp_path / "eval").exists()
    assert reads == []


def test_the_loop_with_the_dump_on_changes_no_other_byte(tmp_path, monkeypatch):
    """history.jsonl, every metrics.json and every bundle are byte for byte the
    flag-off run's; the commit is never read when nothing is dumped."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    reads = []
    monkeypatch.setattr(
        train_cli, "_fork_commit", lambda repo: reads.append(repo) or "c0ffee"
    )

    off = _run(tmp_path / "off", {"0000": scene}, scene)
    assert reads == []
    on = _run(
        tmp_path / "on",
        {"0000": scene},
        scene,
        eval_visual_scenes=["0000"],
        checkpoint_dir="released/4rc",
    )

    def outputs(directory):
        return {
            name: data
            for name, data in _files(directory).items()
            if name == "history.jsonl" or name.startswith("eval/")
        }

    off_files, on_files = outputs(tmp_path / "off"), outputs(tmp_path / "on")
    assert set(on_files) - set(off_files) == {
        "eval/step-2/visual/0000.npz",
        "eval/step-4/visual/0000.npz",
    }
    assert {name: data for name, data in on_files.items() if "/visual/" not in name} == (
        off_files
    )
    assert json.dumps(on["evaluations"]) == json.dumps(off["evaluations"])


def test_the_fork_commit_is_heads_hash_or_none(tmp_path, monkeypatch):
    """HEAD of a checkout, None outside one or without git: never an error.

    A throwaway repository, so the test holds wherever the suite runs; signing
    and hooks are off so no global git setting can refuse the commit.
    """

    repo = tmp_path / "checkout"
    repo.mkdir()
    git = [
        "git", "-C", str(repo),
        "-c", "user.name=test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false",
    ]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "--no-verify", "-m", "pinned"],
        check=True,
    )
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert re.fullmatch(r"[0-9a-f]{40}", head)
    assert train_cli._fork_commit(repo) == head
    plain = tmp_path / "plain"
    plain.mkdir()
    assert train_cli._fork_commit(plain) is None

    def no_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(train_cli.subprocess, "run", no_git)
    assert train_cli._fork_commit(repo) is None


# ------------------------------------------------------------------- the file ---


def test_the_reader_refuses_a_file_that_breaks_the_schema(tmp_path, monkeypatch):
    """A dump the eval wrote reads back whole; every way of breaking it is
    refused by name rather than handed to the viewer."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled")
    model = _FakeArc(scene.num_observations, 56, 56)
    _evaluate(tmp_path / "out", {"0000": scene}, model, **_dumping("0000"))
    arrays, metadata = _dumped(tmp_path / "out")
    assert set(metadata) == set(VISUAL_METADATA_KEYS)

    def refused(name, broken, message):
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, **broken)
        with pytest.raises(ValueError, match=message):
            read_visual_dump(path)

    refused("missing", {k: v for k, v in arrays.items() if k != "rgb"}, r"missing \['rgb'\]")
    refused("extra", {**arrays, "stray": np.zeros(1)}, r"unexpected \['stray'\]")
    refused(
        "dtype",
        {**arrays, "pred_points_m": arrays["pred_points_m"].astype(np.float64)},
        "pred_points_m must be float32",
    )
    refused(
        "slots",
        {**arrays, "gt_camera_center_m": arrays["gt_camera_center_m"][:-1]},
        "but S is 8",
    )
    refused(
        "pixels",
        {**arrays, "pixel_rows": arrays["pixel_rows"][::-1].copy()},
        "strictly increasing",
    )
    refused(
        "times",
        {**arrays, "slot_time_index": arrays["slot_time_index"] + 4},
        "must index the 4 covered timesteps",
    )
    refused(
        "metadata",
        {**arrays, "metadata": np.array(json.dumps({"scene": "0000"}))},
        "metadata key mismatch",
    )
    with pytest.raises(ValueError, match="track_gt axis 1 is 2, but N is 3"):
        build_visual_arrays(
            geometry={
                key: value
                for key, value in arrays.items()
                if not key.startswith("track_") and key != "metadata"
            },
            tracks={
                "pred": arrays["track_pred"],
                "gt": arrays["track_gt"][:, :-1],
                "gt_vis_any": arrays["track_gt_vis_any"],
                "query_points": arrays["track_query_points"],
                "conf": arrays["track_conf"],
            },
            metadata=metadata,
        )
