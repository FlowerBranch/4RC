"""Pins for the extra training pool: dump-format clips mixed into the sampled stream.

Grouped by the claim each pins.  The clip provider builds from a clip exactly
the scene its arrays make through ``build_scene``, and exactly the scene
``MVTrackerSceneProvider`` builds from the same world; every malformed clip is
one skipped scene, never a dead run.  The index lists a clip pool by name and
says so.  The two-pool draw takes each pool's scenes from its own epoch shuffle
at the requested share, and without an extra pool consumes the stream exactly
as 4b76464 did.  Records, logs and guards name the pool only in a mixed run.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import sys
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import train_temporal_tracking as train_cli
from arc.training.dump_clip import CLIP_POOL_KIND, ClipFormatError, clip_view_ids, open_clip
from arc.training.dumped_kubric import check_clip
from arc.training.manifest_plan import StepPlan
from arc.training.scene_provider import (
    DumpClipSceneProvider,
    MVTrackerSceneProvider,
    SceneProviderByRoot,
    SceneProviderError,
)
from arc.training.scene_sampler import (
    DEFAULT_VIEW_COUNT_WEIGHTS,
    EXTRA_POOL,
    POOL_INDEX_NAME,
    SCENE_POOL,
    ExtraPool,
    PoolIndex,
    PoolScene,
    ScenePoolError,
    build_clip_pool_index,
    build_pool_index,
    draw_scenes,
    parse_view_count_weights,
    plan_pool,
    read_clip_pool_index,
    read_pool_index,
    sampler_identity,
    write_pool_index,
)
from scene_fixtures import dump_clip_members, fixture_datapoint, fixture_scene, write_dump_clip
from test_scene_sampler import write_pool
from test_trainer_loop import _loop_args, _toy_model

DEFAULT_WEIGHTS = parse_view_count_weights(DEFAULT_VIEW_COUNT_WEIGHTS)
PLAN_KWARGS = dict(
    available_cameras=None,
    budget=48,
    stride=2,
    max_time_indices=32,
    min_views=2,
    excluded_data_roots=(),
)


# ----------------------------------------------------------------- fixtures ---


def _plan(data_root, *, seq_name="0000", cameras=(2, 0), times=(0, 2, 4), real_len=1, transform=None):
    """A planned step on a 6-frame clip: ``cameras`` are view positions, as the sampler draws them."""

    return StepPlan(
        step=0,
        seq_name=seq_name,
        data_root=str(data_root),
        cameras=tuple(cameras),
        times=tuple(times),
        frame_start=0,
        seq_len=6,
        stride=2,
        time_bound="window",
        track_indices=(),
        scene_transform=transform,
        depth_type="gt",
        real_len=real_len,
    )


def _clip(pool, name="0000", *, as_zip=False, mutate=None, **world):
    """One fixture clip: V=3, T=6, 56x56, camera 1 rotated, optionally corrupted by ``mutate``."""

    world = {"time_count": 6, "view_count": 3, "rotated_camera": 1, **world}
    members = dump_clip_members(fixture_datapoint(**world), name=name)
    if mutate is not None:
        mutate(members)
    return write_dump_clip(pool, name, members, as_zip=as_zip)


def _clip_provider(**overrides):
    settings = dict(size=56, min_shared_queries=1)
    settings.update(overrides)
    return DumpClipSceneProvider(**settings)


def _index(view_counts: dict[str, int], *, integer=True) -> PoolIndex:
    names = sorted(view_counts, key=int) if integer else sorted(view_counts)
    return PoolIndex(
        scenes=tuple(PoolScene(name=name, view_count=view_counts[name]) for name in names),
        content_hash="in-memory",
    )


KUBRIC = _index({"0000": 10, "0001": 6, "0002": 3, "0003": 4})
CLIPS = _index({"a__0": 8, "a__1": 6, "b__0": 8}, integer=False)


def _extra(weight=0.3, index=CLIPS):
    return ExtraPool(index=index, data_root="/clips", weight=weight)


def _assert_same_scene(actual, expected):
    """Every array, slot, anchor and pixel of two scenes; ``Observation.path`` is a label."""

    assert actual.name == expected.name
    for field in ("cameras", "camera_ids", "times", "query_anchors", "query_observation_slot"):
        assert getattr(actual, field) == getattr(expected, field), field
    for field in (
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
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field
    assert actual.track_upscaling_factor == expected.track_upscaling_factor
    assert [replace(o, path=None) for o in actual.observations] == [
        replace(o, path=None) for o in expected.observations
    ]
    assert len(actual.views) == len(expected.views)
    for mine, theirs in zip(actual.views, expected.views):
        assert set(mine) == set(theirs)
        for key in mine:
            if isinstance(mine[key], torch.Tensor):
                assert torch.equal(mine[key], theirs[key]), key
            elif isinstance(mine[key], np.ndarray):
                assert np.array_equal(mine[key], theirs[key]), key
            else:
                assert mine[key] == theirs[key], key


def _install_similarity_transform(monkeypatch):
    """A real similarity in place of MVTracker's ``transform_scene`` (not importable here).

    X' = R (s X): depth scales by s, a camera keeps its pose relative to the moved
    world, and the factor is the provider's own business.  Both providers go
    through the same stub, so a comparison under it pins their wiring.
    """

    import types

    def transform_scene(*, transformation_scale, transformation_rotation, depth, extrs, query_points, traj3d_world):
        scale = float(transformation_scale)
        rotation = transformation_rotation.double()
        world_to_camera = extrs.double()
        moved_extrs = torch.cat(
            [world_to_camera[..., :3, :3] @ rotation.T, scale * world_to_camera[..., :3, 3:]], dim=-1
        )
        moved_queries = torch.cat(
            [query_points[:, :1].double(), (scale * query_points[:, 1:].double()) @ rotation.T], dim=1
        )
        moved_tracks = (scale * traj3d_world.double()) @ rotation.T
        return (
            depth * scale,
            moved_extrs.to(extrs.dtype),
            moved_queries.to(query_points.dtype),
            moved_tracks.to(traj3d_world.dtype),
            None,
        )

    utils = types.ModuleType("mvtracker.datasets.utils")
    utils.transform_scene = transform_scene
    datasets = types.ModuleType("mvtracker.datasets")
    datasets.utils = utils
    root = types.ModuleType("mvtracker")
    root.datasets = datasets
    monkeypatch.setitem(sys.modules, "mvtracker", root)
    monkeypatch.setitem(sys.modules, "mvtracker.datasets", datasets)
    monkeypatch.setitem(sys.modules, "mvtracker.datasets.utils", utils)


def _png(array) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# ----------------------------------------------------------- the provider ---


@pytest.mark.parametrize("as_zip", [False, True], ids=["directory", "zip"])
def test_a_clip_round_trips_to_the_scene_its_arrays_build(tmp_path, as_zip):
    """Positions (2, 0) on a clip whose converter kept cameras 0, 2 and 5 are
    camera ids (5, 0): the scene is the one build_scene makes from the same
    world at those ids, frames, anchors and all."""

    _clip(tmp_path / "clips", as_zip=as_zip, view_ids=[0, 2, 5])
    provider = _clip_provider(query_anchor_slots=((0, 0), (1, 1)))
    scene = provider(_plan(tmp_path / "clips"))

    assert scene.camera_ids == (5, 0) and scene.cameras == (2, 0)
    expected = fixture_scene(
        cameras=(5, 0),
        times=(0, 2, 4),
        size=56,
        query_anchors=((5, 0), (0, 2)),
        time_count=6,
        view_count=3,
        rotated_camera=1,
        view_ids=[0, 2, 5],
    )
    _assert_same_scene(scene, expected)


@pytest.mark.parametrize(
    "transform",
    [None, {"scale": 1.4, "rot_x_deg": 7.0, "rot_y_deg": -11.0}],
    ids=["untransformed", "scaled_and_rotated"],
)
def test_a_kubric_world_written_as_a_clip_is_the_scene_mvtracker_provider_builds(
    tmp_path, monkeypatch, transform
):
    """The same world, the same plan, both providers: identical scenes -- the
    transformed arrays AND the factor, which apply_recorded_scene_transform
    divides in place on the namespace and the clip provider must read back."""

    _install_similarity_transform(monkeypatch)
    _clip(tmp_path / "clips", rotated_camera=2)

    class _Pool:
        seq_names = ["0000"]

        def __getitem__(self, index):
            sample = fixture_datapoint(time_count=6, view_count=3, rotated_camera=2)
            sample.sample_track_indices = [0, 1, 2]
            return sample, True

    slots = ((0, 0), (1, 0))
    kubric = MVTrackerSceneProvider(size=56, min_shared_queries=1, query_anchor_slots=slots)
    kubric._datasets["/kubric"] = _Pool()
    plan = _plan("/kubric", transform=transform)

    live = kubric(plan)
    clip = _clip_provider(query_anchor_slots=slots)(replace(plan, data_root=str(tmp_path / "clips")))

    _assert_same_scene(clip, live)
    if transform is not None:
        assert clip.track_upscaling_factor == pytest.approx(1 / 1.4)
        assert not torch.equal(clip.depth, _clip_provider()(_plan(tmp_path / "clips")).depth)


def test_label_depth_beyond_the_cap_is_zeroed_before_the_transform(tmp_path, monkeypatch):
    """30 m reads 0 (strictly beyond 24), 24 m and 20 m survive -- and survive the
    1.5x scale to 36 m and 30 m, which only a cap applied FIRST allows."""

    _install_similarity_transform(monkeypatch)

    def plant(members):
        depth = members["depth"]
        depth[..., 0:4, :] = 30.0
        depth[..., 4:6, :] = 24.0
        depth[..., 6:8, :] = 20.0
        members["meta"]["depth0"] = depth[:, 0].copy()

    _clip(tmp_path / "clips", mutate=plant)
    scene = _clip_provider(max_depth=24.0)(_plan(tmp_path / "clips", transform={"scale": 1.5}))

    depth = scene.depth.numpy()
    assert (depth[..., 0:4, :] == 0).all()
    assert np.allclose(depth[..., 4:6, :], 36.0)
    assert np.allclose(depth[..., 6:8, :], 30.0)
    assert torch.equal(scene.depth0, scene.depth[:, 0])


def _drop(key):
    return lambda members: members["meta"].pop(key)


def _set_meta(key, value):
    return lambda members: members["meta"].__setitem__(key, value)


def _edit_meta(key, edit):
    def mutate(members):
        array = members["meta"][key].copy()
        edit(array)
        members["meta"][key] = array

    return mutate


def _edit_depth(edit):
    def mutate(members):
        edit(members["depth"])

    return mutate


def _set_frame(member, payload):
    return lambda members: members["frames"].__setitem__(member, payload)


MALFORMED = [
    ("missing key", _drop("view_ids"), "lacks \\['view_ids'\\]"),
    ("seq_name", _set_meta("seq_name", np.array("another")), "not a string equal"),
    ("intrinsics shape", lambda m: m["meta"].__setitem__("intrs", m["meta"]["intrs"][:, :5]), "intrs must have shape"),
    ("non-finite camera", _edit_meta("extrs", lambda a: a.__setitem__((1, 2, 0, 3), np.nan)), "extrs is not finite"),
    ("integer intrinsics", lambda m: m["meta"].__setitem__("intrs", m["meta"]["intrs"].astype(np.int64)), "intrs must be floating"),
    ("visibility dtype", lambda m: m["meta"].__setitem__("visibility", m["meta"]["visibility"].astype(np.uint8)), "visibility must be bool"),
    ("query off its track", _edit_meta("query_points", lambda a: a.__setitem__((1, 2), a[1, 2] + 0.5)), "does not match traj3d_world"),
    ("query time out of range", _edit_meta("query_points", lambda a: a.__setitem__((0, 0), 6.0)), "out-of-range query time"),
    ("repeated view id", _set_meta("view_ids", np.array([0, 1, 1])), "distinct"),
    ("negative view id", _set_meta("view_ids", np.array([0, -1, 2])), "non-negative"),
    ("view ids count", _set_meta("view_ids", np.array([0, 1])), "names 2 views"),
    ("float view ids", _set_meta("view_ids", np.array([0.0, 1.0, 2.0])), "must be integers"),
    ("zero factor", _set_meta("track_upscaling_factor", np.float64(0.0)), "finite and positive"),
    ("depth key", lambda m: m.__setitem__("depth", {"other": m["depth"]}), "lacks 'depth'"),
    ("depth shape", lambda m: m.__setitem__("depth", m["depth"][:, :5]), "covers 3 views x 5 frames"),
    ("depth0 differs", _edit_meta("depth0", lambda a: a.__iadd__(1.0)), "differs from depth0"),
    ("negative depth", _edit_depth(lambda d: d.__setitem__((0, 3, 0, 0, 0), -1.0)), "finite and non-negative"),
    ("nan depth", _edit_depth(lambda d: d.__setitem__((0, 3, 0, 0, 0), np.nan)), "finite and non-negative"),
    ("missing frame", lambda m: m["frames"].pop("view_1/0003.png"), "1 missing"),
    ("a seventh frame", _set_frame("view_0/0006.png", b"x"), "1 unexpected"),
    ("frame size", _set_frame("view_2/0004.png", _png(np.zeros((40, 56, 3), np.uint8))), "depth grid"),
    ("frame mode", _set_frame("view_0/0002.png", _png(np.zeros((56, 56), np.uint8))), "mode L, not RGB"),
    ("frame bytes", _set_frame("view_0/0000.png", b"not a png"), "frames.zip/view_0/0000.png"),
    ("no depth file", lambda m: m.__setitem__("depth", None), "missing \\['depth_full.npz'\\]"),
    ("no frames file", lambda m: m.__setitem__("frames", None), "missing \\['frames.zip'\\]"),
    ("meta bytes", lambda m: m.__setitem__("meta", b"not an npz"), "meta.npz"),
]


@pytest.mark.parametrize("as_zip", [False, True], ids=["directory", "zip"])
@pytest.mark.parametrize("case, mutate, fragment", MALFORMED, ids=[case for case, _, _ in MALFORMED])
def test_every_malformed_clip_is_one_skipped_scene(tmp_path, case, mutate, fragment, as_zip):
    _clip(tmp_path / "clips", mutate=mutate, as_zip=as_zip)
    with pytest.raises(SceneProviderError, match=fragment) as excinfo:
        _clip_provider()(_plan(tmp_path / "clips"))
    message = str(excinfo.value)
    assert "clip '0000'" in message and "is malformed" in message
    assert train_cli.scene_skip_cause(excinfo.value) == "other"


def test_a_wrapped_clip_zip_is_refused_naming_the_extra_directory(tmp_path):
    members = dump_clip_members(fixture_datapoint(time_count=6, view_count=3), name="0000")
    unpacked = write_dump_clip(tmp_path / "staging", "0000", members)
    (tmp_path / "clips").mkdir()
    with zipfile.ZipFile(tmp_path / "clips" / "0000.zip", "w", zipfile.ZIP_STORED) as archive:
        for member in ("meta.npz", "depth_full.npz", "frames.zip"):
            archive.write(unpacked / member, f"0000/{member}")
    with pytest.raises(SceneProviderError, match="nested under '0000/'"):
        _clip_provider()(_plan(tmp_path / "clips"))


@pytest.mark.parametrize(
    "plan_overrides, provider_overrides, fragment, cause",
    [
        ({"seq_name": "0001"}, {}, "is not in the pool", "scene_absent"),
        ({"real_len": 2}, {}, "holds 1 clips but the plan recorded real_len=2", "other"),
        ({"cameras": (3, 0)}, {}, r"draws view position\(s\) \[3\]", "other"),
        ({"times": (0, 2, 6)}, {}, r"reaches frame\(s\) \[6\]", "other"),
        ({}, {"min_shared_queries": 4}, "only 3 eligible tracks", "pool_too_small"),
    ],
    ids=["absent", "real_len", "stale_view_count", "short_clip", "track_pool"],
)
def test_a_plan_the_clip_cannot_serve_is_one_skipped_scene(
    tmp_path, plan_overrides, provider_overrides, fragment, cause
):
    _clip(tmp_path / "clips")
    with pytest.raises(SceneProviderError, match=fragment) as excinfo:
        _clip_provider(**provider_overrides)(replace(_plan(tmp_path / "clips"), **plan_overrides))
    assert train_cli.scene_skip_cause(excinfo.value) == cause


def test_a_misfit_anchor_spec_still_ends_the_run_before_any_read(tmp_path):
    """A ValueError, not a skip -- raised before the pool is even listed."""

    with pytest.raises(ValueError, match="does not fit") as excinfo:
        _clip_provider(query_anchor_slots=((2, 0),))(_plan(tmp_path / "nowhere", cameras=(1, 0)))
    assert not isinstance(excinfo.value, SceneProviderError)

    _clip(tmp_path / "clips")
    adaptive = _clip_provider(query_anchor_slots=((0, 0), (2, 0)), adaptive_query_anchors=True)
    assert adaptive(_plan(tmp_path / "clips", cameras=(1, 0))).query_anchors == ((1, 0),)
    with pytest.raises(ValueError, match="negative"):
        _clip_provider(query_anchor_slots=((-1, 0),))


def test_geometry_inputs_ride_the_clip_path_with_the_cap_as_the_threshold(tmp_path):
    _clip(tmp_path / "clips")
    scene = _clip_provider(input_depth=True, input_camera_vectors=True)(_plan(tmp_path / "clips"))
    assert all("depth_map" in view and "camera_vector" in view for view in scene.views)
    assert all("depth_map" not in view for view in _clip_provider()(_plan(tmp_path / "clips")).views)


def test_the_dispatcher_routes_by_root_and_refuses_an_unknown_one():
    seen = []
    dispatch = SceneProviderByRoot(
        {"/kubric": lambda plan: seen.append(("kubric", plan.seq_name)), "/clips": lambda plan: seen.append(("clips", plan.seq_name))}
    )
    dispatch(_plan("/clips", seq_name="a"))
    dispatch(_plan("/kubric", seq_name="b"))
    assert seen == [("clips", "a"), ("kubric", "b")]
    with pytest.raises(RuntimeError, match="no scene provider serves data_root '/elsewhere'") as excinfo:
        dispatch(_plan("/elsewhere"))
    assert not isinstance(excinfo.value, SceneProviderError)


def test_the_clip_layout_reads_both_forms_and_names_what_is_missing(tmp_path):
    for as_zip, name in ((False, "dir"), (True, "zipped")):
        path = _clip(tmp_path, name, as_zip=as_zip, view_ids=[4, 1, 9])
        assert clip_view_ids(path).tolist() == [4, 1, 9]
        with open_clip(path) as clip:
            assert len(clip.frames().namelist()) == 18
    with pytest.raises(ClipFormatError, match="neither a directory nor"):
        with open_clip(tmp_path / "absent.zip"):
            pass


# ------------------------------------------------------------------ the index ---


def test_a_clip_pool_is_listed_by_name_and_says_so(tmp_path):
    pool = tmp_path / "clips"
    _clip(pool, "b__0002")
    _clip(pool, "a__0001", as_zip=True, view_count=2, view_ids=[3, 7])
    _clip(pool, "c")
    (pool / "_cache").mkdir()
    (pool / ".tmp-d.zip").write_bytes(b"partial")
    (pool / "notes.txt").write_text("not a clip")

    document = build_clip_pool_index(pool, weights=DEFAULT_WEIGHTS)
    assert document["pool_kind"] == CLIP_POOL_KIND
    assert [scene["name"] for scene in document["scenes"]] == ["a__0001", "b__0002", "c"]
    assert [scene["view_count"] for scene in document["scenes"]] == [2, 3, 3]
    assert list(document) == ["pool_index_version", "pool_kind", "scene_count", "content_hash", "scenes"]

    write_pool_index(pool, document)
    index = read_clip_pool_index(pool)
    assert [scene.name for scene in index.scenes] == ["a__0001", "b__0002", "c"]
    with pytest.raises(ScenePoolError, match="indexes a pool of dump-format clips, but it was given as --scene_pool"):
        read_pool_index(pool)

    kubric = write_pool(tmp_path / "kubric", {"0001": 3, "0002": 4})
    kubric_document, _ = build_pool_index(kubric)
    assert "pool_kind" not in kubric_document
    write_pool_index(kubric, kubric_document)
    with pytest.raises(ScenePoolError, match="indexes an MV-Kubric scene pool, but it was given as --extra_pool"):
        read_clip_pool_index(kubric)


def test_the_clip_listing_refuses_both_forms_and_an_edited_order(tmp_path):
    pool = tmp_path / "clips"
    _clip(pool, "x")
    _clip(pool, "x", as_zip=True)
    with pytest.raises(ScenePoolError, match="clip 'x' exists both as a directory and as an archive"):
        build_clip_pool_index(pool, weights=DEFAULT_WEIGHTS)

    pool = tmp_path / "ordered"
    _clip(pool, "a")
    _clip(pool, "b")
    document = build_clip_pool_index(pool, weights=DEFAULT_WEIGHTS)
    document["scenes"].reverse()
    from arc.training.scene_sampler import _scenes_hash

    document["content_hash"] = _scenes_hash(document["scenes"])
    write_pool_index(pool, document)
    with pytest.raises(ScenePoolError, match="not in name order") as excinfo:
        read_clip_pool_index(pool)
    assert "--extra_pool" in str(excinfo.value) and "--build_pool_index" in str(excinfo.value)


def test_an_undrawable_or_unreadable_clip_is_refused_at_index_time_by_name(tmp_path):
    pool = tmp_path / "clips"
    _clip(pool, "wide")
    _clip(pool, "mono", view_count=1)
    with pytest.raises(ScenePoolError, match=r"seat none of the listed view counts \[2, 3, 4, 5, 6\]: mono \(1 views\)"):
        build_clip_pool_index(pool, weights=DEFAULT_WEIGHTS)
    assert build_clip_pool_index(pool, weights={1: 1.0})["scene_count"] == 2

    _clip(pool, "broken", mutate=_drop("view_ids"))
    with pytest.raises(ScenePoolError, match="1 clip\\(s\\) under .* cannot be read: .*broken.*lacks view_ids"):
        build_clip_pool_index(pool, weights={1: 1.0})


def test_an_undrawable_clip_is_refused_at_plan_time_against_the_runs_weights(tmp_path, monkeypatch, capsys):
    """Indexed under '1:1', planned under the default: the draw names the clip
    up front, whatever --num_steps the plan pass asks for."""

    kubric = write_pool(tmp_path / "kubric", {"0001": 3, "0002": 4})
    clips = tmp_path / "clips"
    _clip(clips, "wide")
    _clip(clips, "mono", view_count=1)
    _cli(monkeypatch, "--scene_pool", str(kubric), "--build_pool_index")
    _cli(monkeypatch, "--extra_pool", str(clips), "--build_pool_index", "--view_count_weights", "1:1")
    for steps in ("1", "40"):
        with pytest.raises(SystemExit) as excinfo:
            _cli(
                monkeypatch, "--scene_pool", str(kubric), "--extra_pool", str(clips),
                "--extra_pool_weight", "0.5", "--plan_only", "--num_steps", steps, "--warmup_steps", "0",
            )
        assert excinfo.value.code == 2
        assert "seat none of the listed view counts [2, 3, 4, 5, 6]: mono (1 views)" in capsys.readouterr().err


def test_check_clips_passes_a_conforming_pool_and_names_every_offender(tmp_path, monkeypatch, capsys):
    good = tmp_path / "good"
    _clip(good, "a", time_count=24)
    _clip(good, "b", as_zip=True, time_count=24)
    assert check_clip(good / "a", name="a", min_frames=24) == []
    _cli(monkeypatch, "--extra_pool", str(good), "--build_pool_index", "--check_clips")
    assert "check_clips=2 clips pass" in capsys.readouterr().out
    assert (good / POOL_INDEX_NAME).is_file()

    bad = tmp_path / "bad"
    _clip(bad, "short")
    _clip(bad, "holed", time_count=24, mutate=lambda m: m["frames"].pop("view_2/0017.png"))
    _clip(bad, "flat", time_count=24, mutate=lambda m: m.__setitem__("depth", m["depth"][:, :, :, :50]))
    _clip(bad, "fine", time_count=24)
    with pytest.raises(SystemExit) as excinfo:
        _cli(monkeypatch, "--extra_pool", str(bad), "--build_pool_index", "--check_clips")
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "3 problem(s) among the 4 clips" in err and "no index was written" in err
    assert "short: 6 frames, fewer than the 24-frame window" in err
    assert "holed" in err and "1 missing" in err
    assert "flat" in err and "expected floating (3, 24, 1, 56, 56)" in err
    assert not (bad / POOL_INDEX_NAME).exists()


def test_building_both_indexes_writes_neither_when_one_is_refused(tmp_path, monkeypatch, capsys):
    kubric = write_pool(tmp_path / "kubric", {"0001": 3, "0002": 4})
    clips = tmp_path / "clips"
    _clip(clips, "mono", view_count=1)
    with pytest.raises(SystemExit):
        _cli(monkeypatch, "--scene_pool", str(kubric), "--extra_pool", str(clips), "--build_pool_index")
    assert not (kubric / POOL_INDEX_NAME).exists() and not (clips / POOL_INDEX_NAME).exists()

    _clip(clips, "wide")
    (clips / "mono").rename(tmp_path / "mono")
    _cli(monkeypatch, "--scene_pool", str(kubric), "--extra_pool", str(clips), "--build_pool_index")
    out = capsys.readouterr().out
    assert f"pool_index={kubric / POOL_INDEX_NAME} scenes=2" in out
    assert f"pool_index={clips / POOL_INDEX_NAME} scenes=1" in out
    assert read_pool_index(kubric).scene_count == 2 and read_clip_pool_index(clips).scene_count == 1


# ------------------------------------------------------------------------ cli ---


def _cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["train_temporal_tracking.py", *argv])
    train_cli.main()


def _parse(*argv):
    args = train_cli.build_arg_parser().parse_args(list(argv))
    train_cli._validate_args(args)
    return args


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--manifest", "m.jsonl", "--extra_pool", "/c", "--extra_pool_weight", "0.3"], "--extra_pool needs --scene_pool"),
        (["--manifest", "m.jsonl", "--extra_pool_weight", "0.3"], "--extra_pool_weight needs --scene_pool"),
        (["--manifest", "m.jsonl", "--extra_pool", "/c", "--build_pool_index"], "--extra_pool needs --scene_pool"),
        (["--sampler_from_manifest", "m.jsonl", "--extra_pool", "/c"], "takes neither"),
        (["--scene_pool", "/p", "--extra_pool_weight", "0.3"], "--extra_pool_weight needs --extra_pool"),
        (["--scene_pool", "/p", "--extra_pool", "/c"], "--extra_pool needs --extra_pool_weight"),
        (["--scene_pool", "/p", "--extra_pool", "/c", "--extra_pool_weight", "0"], r"must be in \(0, 1\]"),
        (["--scene_pool", "/p", "--extra_pool", "/c", "--extra_pool_weight", "1.5"], r"must be in \(0, 1\]"),
        (["--scene_pool", "/p", "--extra_pool", "/c", "--extra_pool_weight", "-0.2"], r"must be in \(0, 1\]"),
        (["--scene_pool", "/p", "--extra_pool", "/c", "--extra_pool_weight", "nan"], r"must be in \(0, 1\]"),
        (["--scene_pool", "/p", "--extra_pool", "/p/", "--extra_pool_weight", "0.3"], "is the --scene_pool directory"),
        (["--scene_pool", "/p", "--extra_pool", "/c", "--extra_pool_weight", "0.3", "--exclude_data_root", "/c"], "names --extra_pool itself"),
        (["--scene_pool", "/p", "--check_clips"], "--check_clips needs --extra_pool --build_pool_index"),
        (["--scene_pool", "/p", "--build_pool_index", "--check_clips"], "--check_clips needs"),
        (["--extra_pool", "/c", "--check_clips"], "exactly one of --manifest and --scene_pool"),
        (["--extra_pool", "/c"], "exactly one of --manifest and --scene_pool"),
        (["--build_pool_index"], "exactly one of --manifest and --scene_pool"),
    ],
)
def test_the_extra_pool_flags_are_refused_at_parse_time(argv, message):
    with pytest.raises(ValueError, match=message):
        _parse(*argv)


def test_the_extra_pool_flags_resolve_and_index_alone():
    args = _parse("--scene_pool", "/p", "--extra_pool", "/c", "--extra_pool_weight", "1")
    assert args.extra_pool_weight == 1.0 and isinstance(args.extra_pool_weight, float)
    args = _parse("--extra_pool", "/c", "--build_pool_index", "--check_clips", "--min_views", "7")
    assert args.scene_pool is None and args.view_count_weights is None
    assert _parse("--scene_pool", "/p").extra_pool is None


# ------------------------------------------------------------------- the draw ---


def test_the_single_pool_stream_is_4b76464s_by_value():
    """Captured at 4b76464 before the two-pool draw existed: a 1-view scene that
    seats no count, last in one of the first permutations, over four epochs.
    The pool draw happens only with an extra pool, so this stream must not move."""

    index = _index({"0000": 10, "0001": 6, "0002": 1, "0003": 4, "0004": 3})
    draws = list(itertools.islice(draw_scenes(index, seed=2, weights=DEFAULT_WEIGHTS, transform_probability=0.5, min_views=2), 14))
    assert [
        (
            d.epoch,
            d.seq_name,
            d.views,
            None if d.scene_transform is None else tuple(d.scene_transform[k] for k in ("scale", "rot_x_deg", "rot_y_deg")),
        )
        for d in draws
    ] == [
        (0, "0004", (0, 1, 2), (1.2334896764454646, -5.089955369883777, -8.860540978864725)),
        (0, "0001", (5, 1, 3, 0), None),
        (0, "0003", (3, 0, 2), None),
        (0, "0000", (3, 1, 5, 8), (0.9412202583854757, -4.505211449901022, -0.9663754625309924)),
        (1, "0000", (8, 7, 3, 4), None),
        (1, "0004", (1, 2), None),
        (1, "0001", (5, 4, 1, 2, 0, 3), None),
        (1, "0003", (0, 1), None),
        (2, "0004", (1, 2), (0.981821326922996, 0.12001318137457773, -5.2937604739590345)),
        (2, "0003", (2, 1, 0), (0.9885721583174109, -14.609479899163482, 8.922148170977806)),
        (2, "0001", (1, 3, 5, 0, 4), (1.4810374829123774, -4.473842960950785, 1.5293371597325773)),
        (2, "0000", (6, 9), (0.9193339934386395, 4.31525802444585, 4.844439801463242)),
        (3, "0000", (5, 2, 4), (1.0938148229147746, -7.086644121599407, 12.41746431495396)),
        (3, "0003", (1, 0, 2, 3), None),
    ]
    assert {d.pool for d in draws} == {SCENE_POOL}

    full = tuple(range(0, 24, 2))
    tally = plan_pool(index, data_root="/pool", num_steps=14, seed=2, weights=DEFAULT_WEIGHTS, transform_probability=0.5, **PLAN_KWARGS)
    assert [(p.seq_name, p.cameras, p.times) for p in tally.planned] == [
        ("0004", (0, 1, 2), full), ("0001", (5, 1, 3, 0), full), ("0003", (3, 0, 2), full),
        ("0000", (3, 1, 5, 8), full), ("0000", (8, 7, 3, 4), full), ("0004", (1, 2), full),
        ("0001", (5, 4, 1, 2, 0, 3), full[:8]), ("0003", (0, 1), full), ("0004", (1, 2), full),
        ("0003", (2, 1, 0), full), ("0001", (1, 3, 5, 0, 4), full[:9]), ("0000", (6, 9), full),
        ("0000", (5, 2, 4), full), ("0003", (1, 0, 2, 3), full),
    ]
    assert (tally.epochs_drawn, tally.draws, tally.admissible_scenes) == (4, 14, 4)
    assert all(p.real_len == 5 and p.data_root == "/pool" for p in tally.planned)


def test_the_mix_ratio_and_each_pools_own_epochs():
    draws = list(itertools.islice(
        draw_scenes(KUBRIC, seed=11, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2, extra=_extra(0.3)),
        10_000,
    ))
    share = sum(d.pool == EXTRA_POOL for d in draws) / len(draws)
    assert abs(share - 0.3) < 4 * math.sqrt(0.3 * 0.7 / len(draws))

    for label, index in ((SCENE_POOL, KUBRIC), (EXTRA_POOL, CLIPS)):
        names = [d.seq_name for d in draws if d.pool == label]
        epochs = [d.epoch for d in draws if d.pool == label]
        size = index.scene_count
        complete = len(names) // size
        assert complete > 100
        for epoch in range(complete):
            assert sorted(names[epoch * size:(epoch + 1) * size]) == sorted(s.name for s in index.scenes)
            assert set(epochs[epoch * size:(epoch + 1) * size]) == {epoch}
        assert all(d.views and len(set(d.views)) == len(d.views) and max(d.views) < d.view_count for d in draws)


def test_the_mixed_stream_for_a_seed_is_pinned_by_value():
    """Decided by this change: Bernoulli first, then the chosen pool's next scene
    (a new permutation of it when its epoch is spent), then count, views, transform."""

    draws = list(itertools.islice(
        draw_scenes(KUBRIC, seed=3, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2, extra=_extra(0.5)),
        10,
    ))
    assert [(d.pool, d.epoch, d.seq_name, d.views) for d in draws] == MIXED_STREAM
    assert draws[0].scene_transform["scale"] == pytest.approx(MIXED_FIRST_SCALE, abs=1e-12)


# Generated by this change's draw and cross-checked against a hand replay of the
# documented order on a bare RandomState(3). Both pools' epochs are visible: the
# extra pool's three clips run through epochs 0 and 1 while Kubric is in epoch 0.
MIXED_STREAM = [
    ("scene_pool", 0, "0003", (0, 3, 2, 1)),
    ("extra_pool", 0, "a__1", (3, 1, 0)),
    ("scene_pool", 0, "0002", (2, 1)),
    ("extra_pool", 0, "a__0", (3, 7, 2, 1, 5, 0)),
    ("extra_pool", 0, "b__0", (6, 1, 7)),
    ("extra_pool", 1, "b__0", (5, 0, 3)),
    ("extra_pool", 1, "a__0", (2, 4, 6, 3, 7, 0)),
    ("extra_pool", 1, "a__1", (1, 2, 3, 5, 4, 0)),
    ("extra_pool", 2, "a__1", (4, 1, 5, 0, 3, 2)),
    ("scene_pool", 0, "0000", (2, 1, 9, 3, 7, 6)),
]
MIXED_FIRST_SCALE = 1.1085668905554455


def test_the_draws_own_refusals_are_up_front():
    with pytest.raises(ScenePoolError, match=r"weight 0.0 is not in \(0, 1\]"):
        next(draw_scenes(KUBRIC, seed=1, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2, extra=_extra(0.0)))
    with pytest.raises(ScenePoolError, match=r"weight 1.5 is not in \(0, 1\]"):
        next(draw_scenes(KUBRIC, seed=1, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2, extra=_extra(1.5)))
    narrow = _extra(index=_index({"z": 1, "a": 8}, integer=False))
    with pytest.raises(ScenePoolError, match=r"1 clip\(s\) of the extra pool seat none .*: z \(1 views\)"):
        next(draw_scenes(KUBRIC, seed=1, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2, extra=narrow))
    # Weight 1 never draws the Kubric pool, and still checks it seats something.
    only = list(itertools.islice(draw_scenes(KUBRIC, seed=1, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2, extra=_extra(1.0)), 30))
    assert {d.pool for d in only} == {EXTRA_POOL}
    with pytest.raises(ScenePoolError, match="no scene in the pool seats"):
        next(draw_scenes(_index({"0000": 1}), seed=1, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2, extra=_extra(1.0)))


def test_each_pool_draws_its_own_root_real_len_and_epoch_guard():
    tally = plan_pool(KUBRIC, data_root="/kubric", num_steps=60, seed=5, weights=DEFAULT_WEIGHTS, transform_probability=1.0, extra=_extra(0.4), **PLAN_KWARGS)
    roots = Counter(p.data_root for p in tally.planned)
    assert set(roots) == {"/kubric", "/clips"}
    assert all(p.real_len == (3 if p.data_root == "/clips" else 4) for p in tally.planned)
    pools = tally.pools
    assert pools[SCENE_POOL].planned_steps == roots["/kubric"] and pools[EXTRA_POOL].planned_steps == roots["/clips"]
    assert pools[SCENE_POOL].draws + pools[EXTRA_POOL].draws == tally.draws == 60
    assert tally.epochs_drawn == pools[SCENE_POOL].epochs_drawn
    assert sum(pools[EXTRA_POOL].views_per_step.values()) == roots["/clips"]

    with pytest.raises(ScenePoolError, match="the extra pool's epoch 0 drew every admissible scene and planned none of them"):
        plan_pool(KUBRIC, data_root="/kubric", num_steps=60, seed=5, weights=DEFAULT_WEIGHTS, transform_probability=1.0,
                  extra=_extra(0.4), **{**PLAN_KWARGS, "excluded_data_roots": ("/clips",)})


# --------------------------------------------------------------- the records ---


def test_the_identity_gains_the_extra_pool_only_with_one():
    plain = sampler_identity(pool="/kubric", index=KUBRIC, sampler_seed=3, view_count_weights="2:0.1", transform_probability=1.0)
    assert list(plain) == ["pool", "index_hash", "sampler_seed", "view_count_weights", "transform_probability"]
    mixed = sampler_identity(pool="/kubric", index=KUBRIC, sampler_seed=3, view_count_weights="2:0.1", transform_probability=1.0, extra=_extra(0.25))
    assert mixed == {**plain, "extra_pool": "/clips", "extra_pool_index_hash": "in-memory", "extra_pool_weight": 0.25}
    assert isinstance(mixed["extra_pool_weight"], float)


def test_a_resume_refuses_a_changed_or_added_or_removed_extra_pool(tmp_path):
    plain = sampler_identity(pool="/kubric", index=KUBRIC, sampler_seed=3, view_count_weights="2:0.1", transform_probability=1.0)
    mixed = {**plain, "extra_pool": "/clips", "extra_pool_index_hash": "h", "extra_pool_weight": 0.25}
    stored = train_cli._checkpoint_settings(_loop_args(tmp_path, scene_sampler=mixed))
    train_cli.check_resume_settings(stored, _loop_args(tmp_path, scene_sampler=dict(mixed)))
    for changed in (
        {**mixed, "extra_pool": "/elsewhere"},
        {**mixed, "extra_pool_index_hash": "rebuilt"},
        {**mixed, "extra_pool_weight": 0.3},
        plain,
    ):
        with pytest.raises(RuntimeError, match="scene_sampler"):
            train_cli.check_resume_settings(stored, _loop_args(tmp_path, scene_sampler=changed))
    single = train_cli._checkpoint_settings(_loop_args(tmp_path, scene_sampler=plain))
    train_cli.check_resume_settings(single, _loop_args(tmp_path, scene_sampler=dict(plain)))
    with pytest.raises(RuntimeError, match="scene_sampler"):
        train_cli.check_resume_settings(single, _loop_args(tmp_path, scene_sampler=mixed))


def test_a_single_pool_plan_keeps_4b76464s_scene_source(tmp_path, monkeypatch, capsys):
    """Without an extra pool, plan.json's scene_source has exactly 4b76464's keys
    and no line names a pool."""

    monkeypatch.chdir(tmp_path)
    write_pool(Path("pool"), {"0010": 4, "0002": 10, "0003": 3, "0001": 2, "0007": 0}, loose={"0003"}, zero_view={"0007"})
    _cli(monkeypatch, "--scene_pool", "pool", "--build_pool_index")
    _cli(
        monkeypatch, "--scene_pool", "pool", "--plan_only", "--num_steps", "12", "--warmup_steps", "2",
        "--sampler_seed", "5", "--transform_probability", "0.5", "--json_out", "plan.json",
    )
    source = json.loads(Path("plan.json").read_text())["scene_source"]
    assert set(source) == {
        "kind", "pool", "index_hash", "sampler_seed", "view_count_weights",
        "transform_probability", "scene_count", "admissible_scenes", "epochs_drawn",
    }
    assert "pool=" not in capsys.readouterr().out.replace("--scene_pool", "")


def test_a_mixed_plan_reports_both_pools(tmp_path, monkeypatch, capsys):
    kubric = write_pool(tmp_path / "kubric", {"0000": 10, "0001": 6})
    clips = tmp_path / "clips"
    _clip(clips, "0001")
    _clip(clips, "b__7", as_zip=True, view_count=4)
    _cli(monkeypatch, "--scene_pool", str(kubric), "--extra_pool", str(clips), "--build_pool_index")
    capsys.readouterr()
    out_path = tmp_path / "plan.json"
    _cli(
        monkeypatch, "--scene_pool", str(kubric), "--extra_pool", str(clips), "--extra_pool_weight", "0.5",
        "--plan_only", "--num_steps", "40", "--warmup_steps", "2", "--sampler_seed", "4", "--json_out", str(out_path),
    )
    printed = capsys.readouterr().out
    summary = json.loads(out_path.read_text())
    source = summary["scene_source"]
    assert source["extra_pool"] == str(clips) and source["extra_pool_weight"] == 0.5
    assert source["extra_pool_index_hash"] == read_clip_pool_index(clips).content_hash
    assert set(source["pools"]) == {SCENE_POOL, EXTRA_POOL}
    extra = source["pools"][EXTRA_POOL]
    assert extra["pool"] == str(clips) and extra["scene_count"] == 2
    assert extra["planned_steps"] + source["pools"][SCENE_POOL]["planned_steps"] == 40
    assert 0 < extra["draws"] < 40 and extra["epochs_drawn"] >= 1
    assert sum(extra["views_per_step"].values()) == extra["planned_steps"]
    assert set(int(k) for k in extra["views_per_step"]) <= {2, 3, 4}
    # Clip "0001" shares its name with a Kubric scene: counted apart.
    assert summary["distinct_scenes"] == 4
    assert "pool=extra_pool" in printed and "pool=scene_pool" in printed
    assert f"+ extra pool {clips} at weight 0.5 ({extra['planned_steps']} of 40 steps" in printed


def test_swapped_pools_are_refused_when_the_plan_is_read(tmp_path, monkeypatch, capsys):
    kubric = write_pool(tmp_path / "kubric", {"0000": 10})
    clips = tmp_path / "clips"
    _clip(clips, "a")
    _cli(monkeypatch, "--scene_pool", str(kubric), "--extra_pool", str(clips), "--build_pool_index")
    with pytest.raises(SystemExit) as excinfo:
        _cli(monkeypatch, "--scene_pool", str(clips), "--extra_pool", str(kubric), "--extra_pool_weight", "0.5", "--plan_only")
    assert excinfo.value.code == 2
    assert "indexes a pool of dump-format clips, but it was given as --scene_pool" in capsys.readouterr().err


# ------------------------------------------------------------------- the loop ---


def _mixed_plans(count, *, every):
    """Kubric plans with an extra-pool plan at every ``every``-th step."""

    kubric = _plan("/kubric", seq_name="0000", real_len=None)
    clip = _plan("/clips", seq_name="0000", real_len=None)
    return [replace(clip if step % every == every - 1 else kubric, step=step) for step in range(count)]


def _outcome(step, plan, learning_rates):
    return train_cli.StepOutcome(
        step=step, seq_name=plan.seq_name, loss=0.5, metric_error_m=0.0, sample_count=1,
        alignment_scale=1.0, alignment_residual_m=0.0, learning_rates=list(learning_rates), gradient_norms={},
    )


def _drive(tmp_path, args, plans, *, provider, step_fn=None):
    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])

    def default_step(*, model, scene, plan, optimizer, scaler, learning_rates, step, **_):
        loss = model(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return _outcome(step, plan, learning_rates)

    train_cli._STOP_REQUESTED.clear()
    return train_cli.run_training(
        model=model, optimizer=optimizer, scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=plans, args=args, scene_provider=provider, step_fn=step_fn or default_step, output_dir=tmp_path,
    )


@pytest.mark.parametrize("mixed", [False, True], ids=["single_source", "mixed"])
def test_rows_lines_and_records_name_the_pool_only_in_a_mixed_run(tmp_path, capsys, mixed):
    plans = _mixed_plans(8, every=4)

    def provider(plan):
        if plan.step == 7:
            raise SceneProviderError("scene '0000' is not in the pool at '/clips' (0 clips)")
        return SimpleNamespace(name=plan.seq_name)

    def step_fn(*, plan, step, learning_rates, **_):
        if step == 3:
            raise train_cli.UnsupervisableSceneError("nothing", scene="0000", eligibility={"total_query_count": 0})
        return _outcome(step, plan, learning_rates)

    args = _loop_args(tmp_path, num_steps=8, scene_pool="/kubric", extra_pool="/clips" if mixed else None)
    result = _drive(tmp_path, args, plans, provider=provider, step_fn=step_fn)
    printed = capsys.readouterr().out
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]

    if mixed:
        assert [(row["step"], row["pool"]) for row in rows] == [(s, "scene_pool") for s in (0, 1, 2, 4, 5, 6)]
        assert "step=0/8 scene=0000 pool=scene_pool loss=" in printed
        assert "step=3 scene=0000 pool=extra_pool skipped: nothing" in printed
        assert "step=7 scene=0000 pool=extra_pool skipped: scene '0000' is not in the pool" in printed
        assert result["unsupervisable_steps"][0]["pool"] == "extra_pool"
        assert result["scene_load_skips_by_pool"] == {"extra_pool": {"scene_absent": 1}, "scene_pool": {}}
    else:
        assert all("pool" not in row for row in rows)
        assert "pool=" not in printed
        assert "pool" not in result["unsupervisable_steps"][0]
        assert "scene_load_skips_by_pool" not in result
    assert result["scene_load_skips"] == {"scene_absent": 1}


def test_a_dead_extra_pool_on_a_small_share_aborts_by_its_own_rule(tmp_path):
    """One step in sixty from a pool whose every clip fails: 1.7% of the run, under
    the global 2%, never two failures back to back -- the run would train on Kubric
    alone. The per-pool consecutive rule stops it at that pool's third failure."""

    plans = _mixed_plans(400, every=60)

    def provider(plan):
        if plan.data_root == "/clips":
            raise SceneProviderError(f"clip {plan.seq_name!r} at /clips/{plan.seq_name} is malformed: meta.npz lacks ['view_ids']")
        return SimpleNamespace(name=plan.seq_name)

    settings = dict(num_steps=400, scene_pool="/kubric", max_consecutive_scene_skips=3)
    single = _drive(tmp_path / "single", _loop_args(tmp_path, **settings), plans, provider=provider)
    assert single["completed_steps"] == 400 and single["scene_load_skips"] == {"other": 6}

    with pytest.raises(RuntimeError, match="extra_pool: 3 consecutive steps could not load their scene"):
        _drive(tmp_path / "mixed", _loop_args(tmp_path, extra_pool="/clips", **settings), plans, provider=provider)


def test_a_pool_that_fails_intermittently_resets_its_own_streak(tmp_path):
    """Every other extra-pool clip fails: never two in a row from that pool, so
    its consecutive rule must not fire -- each load from the pool ends its streak."""

    plans = _mixed_plans(120, every=4)
    extra_steps = [plan.step for plan in plans if plan.data_root == "/clips"]
    failing = set(extra_steps[::2])

    def provider(plan):
        if plan.step in failing:
            raise SceneProviderError(f"clip {plan.seq_name!r} is malformed: meta.npz lacks ['view_ids']")
        return SimpleNamespace(name=plan.seq_name)

    args = _loop_args(
        tmp_path, num_steps=120, scene_pool="/kubric", extra_pool="/clips",
        max_consecutive_scene_skips=2, max_scene_skip_fraction=1.0,
    )
    result = _drive(tmp_path, args, plans, provider=provider)
    assert result["completed_steps"] == 120
    assert result["scene_load_skips_by_pool"]["extra_pool"] == {"other": len(failing)}


def test_a_manifest_still_counts_distinct_scenes_by_name(tmp_path, monkeypatch):
    """Two datasets' rows sharing a scene name stay one name in a manifest's
    plan.json, as they always were; only a mixed pool run counts (pool, name)."""

    from test_manifest_plan import _record, _write_manifest

    records = [
        _record(step=0, seq_name="0001", data_root="/a"),
        _record(step=1, seq_name="0001", data_root="/b"),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    out = tmp_path / "plan.json"
    _cli(monkeypatch, "--manifest", str(manifest), "--plan_only", "--json_out", str(out))
    assert json.loads(out.read_text())["distinct_scenes"] == 1


def test_main_trains_through_the_dispatcher_and_scores_through_the_kubric_provider():
    """main() cannot run without a GPU, so its wiring is read: the training loop
    gets the by-root dispatcher, while the held-out set and the track-id summary
    keep the Kubric provider -- a strict dispatcher would refuse --val_data_root."""

    import ast

    tree = ast.parse(Path(train_cli.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            calls.setdefault(name, []).append(node)
    (run,) = calls["run_training"]
    scene_provider = next(k.value for k in run.keywords if k.arg == "scene_provider")
    assert isinstance(scene_provider, ast.Name) and scene_provider.id == "training_provider"
    (held_out,) = calls["_held_out_set"]
    assert isinstance(held_out.args[1], ast.Name) and held_out.args[1].id == "provider"
    (dispatch,) = calls["SceneProviderByRoot"]
    keys = [ast.unparse(key) for key in dispatch.args[0].keys]
    assert keys == ["str(args.scene_pool)", "str(args.extra_pool)"]
    assignments = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "training_provider" for t in node.targets)
    ]
    assert {ast.unparse(node.value) for node in assignments} == {
        "provider",
        ast.unparse(dispatch),
    }
