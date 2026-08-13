"""A window must be the same window whichever source produced it.

``build_scene`` owns every derivation that turns arrays plus frames into a
``DumpedKubricScene`` -- camera-id resolution, the camera-major slot arithmetic,
the anchor slots, the loader cross-checks.  ``load_dumped_kubric_scene`` and
``scene_from_datapoint`` are two thin front-ends over it, and the point of the
split is that they *cannot* disagree about what a window is.

That is a claim about two code paths, so it is tested by running both and
comparing the objects, not by reading the core and being satisfied.  The dump is
written to disk by the same fixture the rest of the suite uses; the live sample
is a duck-typed stand-in carrying exactly the attributes an MVTracker
``Datapoint`` carries, because MVTracker is not importable here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arc.training import (
    build_scene,
    load_dumped_kubric_scene,
    scene_from_datapoint,
)
from test_sparse_tracking import _write_scene


def _datapoint_from_dump(scene_path: Path, *, view_ids=None):
    """A Datapoint-shaped object holding exactly what the dump on disk holds.

    Built from the dump's own arrays rather than re-derived, so a difference
    between the two scenes can only come from the two front-ends -- which is the
    thing under test. ``video`` is read back from the written PNGs for the same
    reason: it makes the comparison cover the frame path, not just the metadata.
    """

    from PIL import Image

    meta = np.load(scene_path / "meta.npz", allow_pickle=False)
    depth = np.load(scene_path / "depth_full.npz", allow_pickle=False)["depth"]
    view_count, time_count = depth.shape[0], depth.shape[1]

    frames = np.zeros((view_count, time_count, 3, *depth.shape[-2:]), dtype=np.uint8)
    for camera in range(view_count):
        for time_index in range(time_count):
            with Image.open(scene_path / f"view_{camera}" / f"{time_index:04d}.png") as image:
                frames[camera, time_index] = np.asarray(image).transpose(2, 0, 1)

    ids = meta["view_ids"].tolist() if "view_ids" in meta.files else list(range(view_count))
    return SimpleNamespace(
        seq_name=scene_path.name,
        video=torch.from_numpy(frames),
        videodepth=torch.from_numpy(depth.astype(np.float32)),
        query_points_3d=torch.from_numpy(meta["query_points"]),
        trajectory_3d=torch.from_numpy(meta["traj3d_world"]),
        visibility=torch.from_numpy(meta["visibility"]),
        intrs=torch.from_numpy(meta["intrs"]),
        extrs=torch.from_numpy(meta["extrs"]),
        track_upscaling_factor=float(meta["track_upscaling_factor"]),
        sample_views=ids if view_ids is None else view_ids,
    )


def _assert_scenes_match(dumped, live):
    """Every field a consumer reads, compared field by field.

    Deliberately not ``==`` on the dataclass: it holds tensors, so equality would
    return a tensor and ``assert`` would take its truthiness. Naming the fields is
    also what makes a future field fail here until someone decides how the live
    front-end should populate it.
    """

    assert dumped.name == live.name
    assert dumped.cameras == live.cameras
    assert dumped.camera_ids == live.camera_ids
    assert dumped.times == live.times
    assert dumped.query_anchors == live.query_anchors
    assert dumped.query_observation_slot == live.query_observation_slot
    assert dumped.num_observations == live.num_observations
    assert dumped.time_indices == live.time_indices
    assert dumped.anchor_observation_slots == live.anchor_observation_slots
    assert dumped.track_upscaling_factor == pytest.approx(live.track_upscaling_factor)

    for name in (
        "view_ids",
        "slot_cameras",
        "slot_times",
        "slot_time_indices",
        "track_query_observation_slots",
        "query_points",
        "trajectories_world",
        "visibility",
        "intrinsics",
        "extrinsics_world_to_camera",
        "depth0",
    ):
        torch.testing.assert_close(
            getattr(dumped, name),
            getattr(live, name),
            rtol=0,
            atol=0,
            msg=f"{name} differs between the dumped and live scenes",
        )

    for slot, (a, b) in enumerate(zip(dumped.observations, live.observations)):
        assert (a.slot, a.camera, a.camera_id, a.original_time, a.semantic_time_index) == (
            b.slot,
            b.camera,
            b.camera_id,
            b.original_time,
            b.semantic_time_index,
        ), f"observation {slot} differs"
        assert a.image_transform == b.image_transform

    # PNG is lossless, so the dump round trip has to be the identity. A tolerance
    # here would let a real colour-space or transpose fault through.
    assert len(dumped.views) == len(live.views)
    for slot, (a, b) in enumerate(zip(dumped.views, live.views)):
        torch.testing.assert_close(
            a["img"], b["img"], rtol=0, atol=0, msg=f"frame {slot} differs"
        )
        torch.testing.assert_close(a["time_index"], b["time_index"], rtol=0, atol=0)
        torch.testing.assert_close(
            a["track_query_idx"], b["track_query_idx"], rtol=0, atol=0
        )


def test_a_live_sample_and_its_dump_build_the_same_scene(tmp_path):
    """The check the whole adapter split rests on.

    If these two ever diverge, every number the trainer produces from live
    samples is incomparable with every number the overfit harness produced from
    dumps -- silently, because both paths return a well-formed scene.
    """

    scene_path = _write_scene(tmp_path, time_count=4, view_count=2, depth_sidecar=True)

    dumped = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )
    live = scene_from_datapoint(
        _datapoint_from_dump(scene_path),
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    _assert_scenes_match(dumped, live)


def test_the_two_front_ends_agree_on_a_non_identity_view_id_map(tmp_path):
    """Camera ids and view positions differ exactly when ``view_ids`` is not the identity.

    That is the case where a front-end could plausibly index one array by the id
    and another by the position, so it is the case worth running both paths
    through rather than the ascending-complete one where the bug is invisible.
    """

    scene_path = _write_scene(
        tmp_path,
        time_count=3,
        view_count=2,
        depth_sidecar=True,
        view_ids=[5, 9],
    )

    dumped = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(9, 5),
        times=(0, 1, 2),
        query_anchors=((9, 0),),
        size=56,
    )
    live = scene_from_datapoint(
        _datapoint_from_dump(scene_path),
        cameras=(9, 5),
        times=(0, 1, 2),
        query_anchors=((9, 0),),
        size=56,
    )

    assert dumped.camera_ids == (9, 5)
    assert dumped.cameras == (1, 0)
    _assert_scenes_match(dumped, live)


def test_a_live_scene_carries_per_frame_depth_without_a_sidecar(tmp_path):
    """A live sample always has ``videodepth``, so any-timestep anchoring works.

    The dump needs ``RCMV_DUMP_DEPTH=1`` for this and is otherwise limited to
    original time 0; the live path has no such mode, which is a real advantage of
    training off the dataset rather than off a dump.
    """

    scene_path = _write_scene(tmp_path, time_count=4, view_count=2, depth_sidecar=True)

    live = scene_from_datapoint(
        _datapoint_from_dump(scene_path),
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 2),),
        size=56,
    )

    assert live.has_time_varying_depth
    assert live.depth_sidecar_path is None
    # The claim that matters: an anchor away from t=0 resolves instead of raising.
    assert live.surface_depth_map(0, 2).shape == live.depth0[0, 0].shape


def test_build_scene_rejects_a_frame_source_that_yields_the_grid_out_of_order(tmp_path):
    """The check that the loader-index cross-check cannot stand in for.

    ``preprocess_images`` numbers its outputs by receipt order (``idx=len(imgs)``),
    so ``view["idx"] == slot`` holds for *any* source ordering -- it catches a
    preprocessor that reordered its own output, never a source that fed the grid
    in the wrong order. That was safe while the only source was ``_iter_frames``,
    which generates the grid itself; it stopped being safe when the core began
    accepting an arbitrary frame source. Every array is indexed by the slot
    arithmetic, so an unchecked reordering yields a scene where each slot carries
    the right metadata and the wrong picture, and nothing downstream can tell.
    """

    scene_path = _write_scene(tmp_path, time_count=3, view_count=2, depth_sidecar=True)
    sample = _datapoint_from_dump(scene_path)
    meta = np.load(scene_path / "meta.npz", allow_pickle=False)
    depth = np.load(scene_path / "depth_full.npz", allow_pickle=False)["depth"]

    from PIL import Image

    def frames_in(pairs):
        for camera, time_index in pairs:
            array = np.asarray(sample.video[camera, time_index]).transpose(1, 2, 0)
            yield (
                camera,
                time_index,
                f"view_{camera}/{time_index:04d}",
                Image.fromarray(array.astype(np.uint8)),
            )

    common = dict(
        name="0000",
        query_points=meta["query_points"],
        trajectories=meta["traj3d_world"],
        visibility=meta["visibility"],
        intrinsics=meta["intrs"],
        extrinsics=meta["extrs"],
        depth0=depth[:, 0],
        depth=depth,
        track_upscaling_factor=float(meta["track_upscaling_factor"]),
        cameras=(0, 1),
        times=(0, 1, 2),
        size=56,
    )
    grid = [(camera, time_index) for camera in (0, 1) for time_index in (0, 1, 2)]

    # Camera-major is accepted, and is the order both shipped front-ends use.
    ordered = build_scene(open_frames=lambda c, t: frames_in(grid), **common)
    assert ordered.num_observations == 6

    # Time-major carries the same six frames, one shared processed shape and the
    # same count, so only the declared (camera, time) can distinguish it.
    time_major = [(camera, time_index) for time_index in (0, 1, 2) for camera in (0, 1)]
    with pytest.raises(RuntimeError, match="out of step at slot 1"):
        build_scene(open_frames=lambda c, t: frames_in(time_major), **common)

    with pytest.raises(RuntimeError, match="out of step"):
        build_scene(open_frames=lambda c, t: frames_in(grid[::-1]), **common)

    # A source that stops early in the *right* order clears every per-frame check,
    # so the count is the only thing left that can see it. Without it the scene
    # would come back short and well-formed, and `S` would silently disagree with
    # the camera/time grid the caller asked for.
    with pytest.raises(RuntimeError, match="yielded 5 frames"):
        build_scene(open_frames=lambda c, t: frames_in(grid[:-1]), **common)
