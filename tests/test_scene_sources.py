"""A live sample must make the same window its arrays make directly.

``build_scene`` owns every derivation that turns arrays plus frames into a
``DumpedKubricScene`` -- camera-id resolution, the camera-major slot arithmetic,
the anchor slots, the loader cross-checks.  ``scene_from_datapoint`` is the thin
live front-end over it and the trainer's only scene path, so what it hands the
core -- frames out of the video tensor, camera ids out of ``sample_views``,
``depth0`` sliced from ``videodepth`` -- must not change what a window *is*.

That is a claim about two routes into one core, so it is tested by running both
and comparing the objects, not by reading the front-end and being satisfied.  The
direct route is the suite's in-memory fixture; the live sample is a duck-typed
stand-in carrying exactly the attributes an MVTracker ``Datapoint`` carries,
because MVTracker is not importable here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arc.training import build_scene, scene_from_datapoint
from scene_fixtures import fixture_datapoint, fixture_scene, scene_arrays


def _assert_scenes_match(built, live):
    """Every field a consumer reads, compared field by field.

    Deliberately not ``==`` on the dataclass: it holds tensors, so equality would
    return a tensor and ``assert`` would take its truthiness. Naming the fields is
    also what makes a future field fail here until someone decides how the live
    front-end should populate it.
    """

    assert built.name == live.name
    assert built.cameras == live.cameras
    assert built.camera_ids == live.camera_ids
    assert built.times == live.times
    assert built.query_anchors == live.query_anchors
    assert built.query_observation_slot == live.query_observation_slot
    assert built.num_observations == live.num_observations
    assert built.time_indices == live.time_indices
    assert built.anchor_observation_slots == live.anchor_observation_slots
    assert built.track_upscaling_factor == pytest.approx(live.track_upscaling_factor)

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
        "depth",
    ):
        torch.testing.assert_close(
            getattr(built, name),
            getattr(live, name),
            rtol=0,
            atol=0,
            msg=f"{name} differs between the built and live scenes",
        )

    for slot, (a, b) in enumerate(zip(built.observations, live.observations)):
        assert (a.slot, a.camera, a.camera_id, a.original_time, a.semantic_time_index) == (
            b.slot,
            b.camera,
            b.camera_id,
            b.original_time,
            b.semantic_time_index,
        ), f"observation {slot} differs"
        assert a.image_transform == b.image_transform

    # Both routes decode the same PNG bytes, so the frames have to be identical.
    # A tolerance here would let a real colour-space or transpose fault through.
    assert len(built.views) == len(live.views)
    for slot, (a, b) in enumerate(zip(built.views, live.views)):
        torch.testing.assert_close(
            a["img"], b["img"], rtol=0, atol=0, msg=f"frame {slot} differs"
        )
        torch.testing.assert_close(a["time_index"], b["time_index"], rtol=0, atol=0)
        torch.testing.assert_close(
            a["track_query_idx"], b["track_query_idx"], rtol=0, atol=0
        )


def test_a_live_sample_builds_the_same_scene_as_its_arrays():
    """The check the live front-end rests on.

    If the two ever diverge, the trainer's scenes are not the windows the rest
    of the suite tests -- silently, because both routes return a well-formed
    scene.
    """

    built = fixture_scene(cameras=(0, 1), times=(0, 1, 2, 3), size=56)
    live = scene_from_datapoint(
        fixture_datapoint(),
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    _assert_scenes_match(built, live)


def test_a_live_sample_agrees_with_its_arrays_on_a_non_identity_view_id_map():
    """Camera ids and view positions differ exactly when ``view_ids`` is not the identity.

    That is the case where a front-end could plausibly index one array by the id
    and another by the position, so it is the case worth running both routes
    through rather than the ascending-complete one where the bug is invisible.
    """

    world = dict(time_count=3, view_count=2, view_ids=[5, 9])
    window = dict(cameras=(9, 5), times=(0, 1, 2), query_anchors=((9, 0),), size=56)
    built = fixture_scene(**world, **window)
    live = scene_from_datapoint(fixture_datapoint(**world), **window)

    assert built.camera_ids == (9, 5)
    assert built.cameras == (1, 0)
    _assert_scenes_match(built, live)


def test_a_live_scene_reads_per_frame_depth_away_from_time_zero():
    """A live sample always has ``videodepth``, so an anchor may sit at any time.

    Frame 2 is given its own depth here, so the read below can only pass if it
    comes from that frame and not from frame 0.
    """

    sample = fixture_datapoint(time_count=4, view_count=2)
    sample.videodepth[0, 2] += 0.25

    live = scene_from_datapoint(
        sample,
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 2),),
        size=56,
    )

    assert torch.equal(live.surface_depth_map(0, 2), sample.videodepth[0, 2, 0])
    assert not torch.equal(live.surface_depth_map(0, 2), live.surface_depth_map(0, 0))


def test_build_scene_rejects_a_frame_source_that_yields_the_grid_out_of_order():
    """The check that the loader-index cross-check cannot stand in for.

    ``preprocess_images`` numbers its outputs by receipt order (``idx=len(imgs)``),
    so ``view["idx"] == slot`` holds for *any* source ordering -- it catches a
    preprocessor that reordered its own output, never a source that fed the grid
    in the wrong order. The core accepts an arbitrary frame source, and every
    array is indexed by the slot arithmetic, so an unchecked reordering yields a
    scene where each slot carries the right metadata and the wrong picture, and
    nothing downstream can tell.
    """

    sample = fixture_datapoint(time_count=3, view_count=2)

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
        scene_arrays(time_count=3, view_count=2),
        cameras=(0, 1),
        times=(0, 1, 2),
        size=56,
    )
    del common["open_frames"]
    grid = [(camera, time_index) for camera in (0, 1) for time_index in (0, 1, 2)]

    # Camera-major is accepted, and is the order the live front-end uses.
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
