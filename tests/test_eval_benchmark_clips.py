"""--eval_benchmark_clips: a held-out benchmark read from the clip set 4I4 wrote, and its parity check.

The set is a directory of dump-format clips -- the layout the extra pool reads --
with benchmark.json at its root, read by this repo's clip reader with nothing
imported from MVTracker. The window takes each clip's views by position and its
view_ids name the cameras; every clip is scored, in name order, on every query it
holds. benchmark_clip_parity.py builds each scene both ways, through the loader
benchmark and through the clip set, and compares them. The clips are the
in-memory fixture world; the loader benchmarks are test_eval_benchmark.py's and
test_eval_benchmark_dexycb.py's MVTracker stubs, whose own samples are exported
here as prepare_clips.py exports MVTracker's.
"""

from __future__ import annotations

import copy
import io
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import arc.training.scene_provider as scene_provider_module
import benchmark_clip_parity as parity
import train_temporal_tracking as train_cli
from arc.training.scene_provider import BenchmarkClipSceneProvider, SceneProviderError
from scene_fixtures import dump_clip_members, fixture_datapoint, fixture_scene, write_dump_clip
from test_eval_benchmark import _CACHE, _bundle, _evaluate, _plant_pointmaps
from test_eval_benchmark import _NAME as _KUBRIC_NAME
from test_eval_benchmark import _args as _kubric_args
from test_eval_benchmark import benchmark  # noqa: F401 -- the Kubric stub fixture
from test_eval_benchmark_dexycb import _NAME as _DEXYCB_NAME
from test_eval_benchmark_dexycb import _benchmark_lines, _dexycb_datapoint, _write_tree
from test_eval_benchmark_dexycb import dexycb  # noqa: F401 -- the DexYCB stub fixture
from test_extra_pool import _assert_same_scene
from test_manifest_plan import _record, _write_manifest
from test_trainer_loop import _FakeArc, _loop_args, _plan_only, _plans, _validator_args

_PROTOCOL = {"source": "mvtracker", "dataset": _KUBRIC_NAME, "setting": "kubric-multiview"}


def _clip(directory, name, *, mutate=None, as_zip=False, **world):
    """One clip of the fixture world in 4I4's layout: two views of four frames
    unless ``world`` says otherwise, ``mutate`` editing its parts first."""

    world = {"view_count": 2, "time_count": 4, **world}
    members = dump_clip_members(fixture_datapoint(scene_name=name, **world), name=name)
    if mutate is not None:
        mutate(members)
    return write_dump_clip(directory, name, members, as_zip=as_zip)


def _args(tmp_path, clips, **overrides):
    """The held-out window on two cameras: four frames at stride 2, the
    fixture's 56-pixel frames untransformed."""

    return _loop_args(
        tmp_path,
        **{
            "eval_every": 2,
            "size": 56,
            "val_cameras": [0, 1],
            "val_seq_len": 4,
            "stride": 2,
            "eval_benchmark_clips": str(clips),
            **overrides,
        },
    )


def _open(args):
    return train_cli._open_benchmark(args)


def _png(pixels) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _exported(sample, *, frames, view_ids):
    """``sample`` as prepare_clips.py's mvtracker export writes it: its first
    ``frames`` frames, uint8 frames, depth as the loader returned it, and the
    camera ids the loader path numbers its views with."""

    return SimpleNamespace(
        video=sample.video[:, :frames].to(torch.uint8),
        videodepth=sample.videodepth[:, :frames],
        query_points_3d=sample.query_points_3d,
        trajectory_3d=sample.trajectory_3d[:frames],
        visibility=sample.visibility[:, :frames],
        intrs=sample.intrs[:, :frames],
        extrs=sample.extrs[:, :frames],
        track_upscaling_factor=sample.track_upscaling_factor,
        sample_views=list(view_ids),
    )


def _write_set(directory, samples, *, frames, view_ids, protocol=None, mutate=None):
    """A clip set: every ``(name, sample)`` exported as a clip directory, and
    ``protocol`` as benchmark.json; ``mutate(name, members)`` edits a clip first."""

    directory = Path(directory)
    for name, sample in samples:
        members = dump_clip_members(
            _exported(sample, frames=frames, view_ids=view_ids), name=name
        )
        if mutate is not None:
            mutate(name, members)
        write_dump_clip(directory, name, members)
    if protocol is not None:
        (directory / "benchmark.json").write_text(json.dumps(protocol))
    return directory


def _kubric_set(tmp_path, state, *, names=None, mutate=None):
    """The Kubric stub's own samples, exported as its clip set."""

    loader = _open(_kubric_args(tmp_path, state))
    samples = [
        (name, loader.dataset[index][0])
        for index, name in enumerate(loader.scene_names)
        if names is None or name in names
    ]
    return _write_set(
        tmp_path / "clips",
        samples,
        frames=4,
        view_ids=[0, 1],
        protocol=_PROTOCOL,
        mutate=mutate,
    )


# ------------------------------------------------------------ the parse ---


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "eval_benchmark_dataset": "kubric-multiview-v3-views0123-cached",
                "eval_benchmark_root": "/bench",
            },
            "--eval_benchmark_clips cannot be combined with "
            "--eval_benchmark_dataset or --eval_benchmark_root",
        ),
        ({"eval_benchmark_dataset": "kubric-multiview-v3-views0123-cached"}, "cannot be combined"),
        ({"eval_benchmark_root": "/bench"}, "cannot be combined"),
        ({"eval_every": 0}, "--eval_benchmark_clips needs a nonzero --eval_every"),
        # Given, even empty: refused as a clip set would be.
        (
            {"eval_benchmark_clips": "", "eval_every": 0},
            "--eval_benchmark_clips needs a nonzero --eval_every",
        ),
    ],
    ids=["both_loader_flags", "dataset", "root", "eval_off", "empty_eval_off"],
)
def test_the_clip_flag_is_refused_at_parse_time(tmp_path, overrides, message):
    args = _validator_args(
        tmp_path, **{"eval_every": 500, "eval_benchmark_clips": "/clips", **overrides}
    )
    with pytest.raises(ValueError, match=message):
        train_cli._validate_args(args)


def test_the_clip_set_is_a_held_out_set_of_its_own(tmp_path):
    """Every eval-only flag validates beside the clip set alone, and the
    window's seating is still checked strictly."""

    base = dict(eval_every=500, eval_benchmark_clips="/clips")
    args = _validator_args(
        tmp_path,
        **base,
        eval_query_anchors=["all"],
        eval_query_anchor_source="gt_depth",
        eval_visual_scenes=["a"],
    )
    train_cli._validate_args(args)
    assert train_cli._has_held_out(args)
    assert len(train_cli._eval_anchor_spec(args)) == 47
    for alias in ("oracle_query_anchor", "ground_truth_query_anchor"):
        train_cli._validate_args(_validator_args(tmp_path, **base, **{alias: True}))
    with pytest.raises(ValueError, match="time slot 12 does not fit the held-out window"):
        train_cli._validate_args(_validator_args(tmp_path, **base, query_anchors=["0:12"]))

    # Given at all, even empty: an unset submit variable is a held-out set that
    # opening refuses, never a silent run without one.
    empty = _validator_args(tmp_path, eval_every=500, eval_benchmark_clips="")
    train_cli._validate_args(empty)
    assert train_cli._has_held_out(empty)
    with pytest.raises(ValueError, match="--eval_benchmark_clips '' is not a directory"):
        _open(_args(tmp_path, ""))
    assert not train_cli._has_held_out(_validator_args(tmp_path, eval_every=500))


def test_the_clip_flag_is_stored_nowhere_a_resume_reads(tmp_path):
    args = _validator_args(tmp_path, eval_every=500, eval_benchmark_clips="/clips")
    train_cli._validate_args(args)

    settings = train_cli._checkpoint_settings(args)
    tiers = train_cli._RESUME_SETTINGS_REFUSED + train_cli._RESUME_SETTINGS_WARNED
    assert "eval_benchmark_clips" not in settings
    assert "eval_benchmark_clips" not in tiers
    plain = _validator_args(tmp_path, eval_every=500)
    train_cli._validate_args(plain)
    assert settings == train_cli._checkpoint_settings(plain)


def test_the_plan_summary_names_the_clip_set_eval_without_opening_it(tmp_path, monkeypatch):
    """--plan_only plans a clip-set run as a held-out one, and opens nothing:
    the set does not exist."""

    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(step=0)])
    out = tmp_path / "plan.json"
    _plan_only(
        monkeypatch,
        manifest,
        "--eval_benchmark_clips",
        str(tmp_path / "absent"),
        "--json_out",
        str(out),
    )

    train_cli.main()

    settings = json.loads(out.read_text())["settings"]
    assert settings["realized_val_query_anchors"] == ["0:0"]
    assert settings["eval_query_anchor_source"] == "predicted"


# ------------------------------------------------------------ the opening ---


def _not_a_directory(root):
    return root / "absent"


def _empty_value(root):
    return ""


def _no_clip(root):
    directory = root / "set"
    directory.mkdir()
    (directory / "benchmark.json").write_text(json.dumps(_PROTOCOL))
    # A clip prepare_clips.py was still writing: dot-prefixed, so not listed.
    (directory / ".tmp-a").mkdir()
    return directory


def _malformed_clip(root):
    directory = root / "set"
    _clip(directory, "a")
    _clip(directory, "b", mutate=lambda members: members["frames"].pop("view_1/0003.png"))
    return directory


def _three_view_clip(root):
    directory = root / "set"
    _clip(directory, "a")
    _clip(directory, "b", view_count=3)
    return directory


def _short_clip(root):
    directory = root / "set"
    _clip(directory, "a", time_count=3)
    return directory


def _protocol(text):
    def make(root):
        directory = root / "set"
        _clip(directory, "a")
        (directory / "benchmark.json").write_text(text)
        return directory

    return make


OPENING = [
    (_not_a_directory, r"--eval_benchmark_clips '.*absent' is not a directory"),
    (_empty_value, r"--eval_benchmark_clips '' is not a directory"),
    (_no_clip, r"--eval_benchmark_clips '.*set' cannot be listed: .*holds no clips"),
    (
        _malformed_clip,
        r"1 of 2 clip\(s\) fail the clip format check at the held-out window's "
        r"--val_seq_len 4: .*set/b: frames\.zip must hold exactly .*1 missing",
    ),
    (
        _three_view_clip,
        r"1 of 2 clip\(s\) do not hold the held-out window's 2 cameras "
        r"\(len\(--val_cameras\).*\): b \(3 views\)",
    ),
    (_short_clip, r"set/a: 3 frames, fewer than the 4-frame window"),
    (_protocol("[1, 2]"), r"benchmark\.json holds a list, not the JSON object"),
    (_protocol("{not json"), r"benchmark\.json cannot be read as JSON: JSONDecodeError"),
]
OPENING_IDS = [
    "not_a_directory",
    "empty_value",
    "no_clip",
    "malformed_clip",
    "view_count",
    "short_clip",
    "protocol_not_an_object",
    "protocol_not_json",
]


@pytest.mark.parametrize("make, message", OPENING, ids=OPENING_IDS)
def test_what_opening_the_clip_set_reveals_is_refused(tmp_path, make, message):
    with pytest.raises(ValueError, match=message):
        _open(_args(tmp_path, make(tmp_path)))


@pytest.mark.parametrize("make, message", OPENING, ids=OPENING_IDS)
def test_every_opening_refusal_reaches_the_command_line_before_the_model_loads(
    tmp_path, monkeypatch, capsys, make, message
):
    clips = make(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(step=0)])
    monkeypatch.setattr(
        train_cli.Arc,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("the model loaded before the refusal"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_temporal_tracking.py",
            "--manifest",
            str(manifest),
            "--checkpoint_dir",
            str(tmp_path / "checkpoint"),
            "--output_dir",
            str(tmp_path / "out"),
            "--eval_benchmark_clips",
            str(clips),
            "--val_cameras",
            "0",
            "1",
            "--val_seq_len",
            "4",
            "--size",
            "56",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    assert re.search(message, capsys.readouterr().err)


def test_clips_are_listed_by_name_as_directories_or_zips(tmp_path):
    clips = tmp_path / "clips"
    _clip(clips, "b")
    _clip(clips, "a", as_zip=True)
    # Skipped, as list_pool skips it for --extra_pool.
    _clip(clips, "_staged")
    args = _args(tmp_path, clips)
    provider = _open(args)

    assert provider.scene_names == ["a", "b"]
    assert provider.cameras == (0, 1) and provider.data_root == str(clips)
    held_out = train_cli._held_out_set(args, object(), provider)
    assert held_out.scene_provider(held_out.plans[0]).name == "a"


def test_the_identity_records_the_set_and_the_rule_that_wrote_it(tmp_path):
    protocol = {"source": "panoptic", "views": [1, 7, 14, 20], "setting": "panoptic-multiview"}
    clips = tmp_path / "panoptic-views1_7_14_20"
    _clip(clips, "a")
    _clip(clips, "b")
    (clips / "benchmark.json").write_text(json.dumps(protocol))

    for given in (str(clips), str(clips) + os.sep):
        identity = _open(_args(tmp_path, given, kubric_max_depth=30.0)).identity()
        assert identity == {
            "clips": given,
            "name": "panoptic-views1_7_14_20",
            "protocol": protocol,
            "scene_count": 2,
            "max_depth": 30.0,
        }
    (clips / "benchmark.json").unlink()
    assert _open(_args(tmp_path, clips)).identity()["protocol"] is None


def test_the_clip_set_replaces_the_scene_list_and_plans_every_clip(tmp_path):
    clips = tmp_path / "clips"
    _clip(clips, "a")
    _clip(clips, "b")
    args = _args(tmp_path, clips, val_scenes_file="never-read.json", val_data_root="/held")
    held_out = train_cli._held_out_set(args, object(), _open(args))

    assert [plan.seq_name for plan in held_out.plans] == ["a", "b"]
    for plan in held_out.plans:
        assert plan.cameras == (0, 1) and plan.times == (0, 2)
        assert plan.data_root == str(clips)
        assert plan.track_indices == () and plan.scene_transform is None
        assert plan.real_len is None and plan.step == -1
    assert held_out.benchmark == {
        **held_out.scene_provider.identity(),
        "window": {"stride": 2, "covered_times": [0, 2]},
    }


# ------------------------------------------------------------ the scenes ---


def test_views_are_positions_named_by_each_clips_view_ids(tmp_path):
    """Panoptic's cameras 1 7 14 20 at positions 0..3; --val_cameras' values
    are not compared, only its count."""

    clips = tmp_path / "clips"
    _clip(clips, "a", view_count=4, view_ids=[1, 7, 14, 20], rotated_camera=2)
    args = _args(tmp_path, clips, val_cameras=[5, 6, 7, 8], query_anchors=["0:0", "1:1"])
    held_out = train_cli._held_out_set(args, object(), _open(args))

    assert held_out.plans[0].cameras == (0, 1, 2, 3)
    scene = held_out.scene_provider(held_out.plans[0])
    assert scene.cameras == (0, 1, 2, 3)
    assert scene.camera_ids == (1, 7, 14, 20) and scene.view_ids.tolist() == [1, 7, 14, 20]
    assert scene.query_anchors == ((1, 0), (7, 2))
    _assert_same_scene(
        scene,
        fixture_scene(
            cameras=(1, 7, 14, 20),
            times=(0, 2),
            size=56,
            query_anchors=((1, 0), (7, 2)),
            scene_name="a",
            view_count=4,
            time_count=4,
            view_ids=[1, 7, 14, 20],
            rotated_camera=2,
        ),
    )


def test_the_anchors_are_resolved_strictly(tmp_path):
    """The window's slots were checked at parse time, so one that does not fit
    is a configuration error, never a dropped anchor."""

    clips = tmp_path / "clips"
    _clip(clips, "a")
    args = _args(tmp_path, clips)
    plan = train_cli._held_out_set(args, object(), _open(args)).plans[0]
    provider = BenchmarkClipSceneProvider(
        clips=clips,
        view_count=2,
        seq_len=4,
        max_depth=24.0,
        query_anchor_slots=((0, 0), (2, 0)),
        size=56,
    )

    with pytest.raises(ValueError, match="does not fit") as error:
        provider(plan)
    assert not isinstance(error.value, SceneProviderError)


def test_a_clip_longer_than_the_window_is_scored_on_its_first_frames(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    _clip(clips, "a", time_count=6, query_times=(0, 2, 5))
    args = _args(tmp_path, clips)
    held_out = train_cli._held_out_set(args, object(), _open(args))

    scene = held_out.scene_provider(held_out.plans[0])
    assert scene.times == (0, 2)
    assert scene.trajectories_world.shape[0] == 6 and scene.depth.shape[1] == 6
    _plant_pointmaps(monkeypatch, scene)
    metrics, _ = _evaluate(tmp_path / "eval", held_out)

    bundle = _bundle(tmp_path / "eval", "a")
    assert bundle["gt"].shape[0] == 2
    # The frame-2 track falls back at its own covered index; the frame-5 one,
    # past the window, at the last.
    assert bundle["query_points"][:, 0].tolist() == [0.0, 1.0, 1.0]
    assert metrics["eval_benchmark"]["per_scene"][0]["unanchored_by_start_time"] == {
        "2": 1,
        "5": 1,
    }


def test_label_depth_is_capped_strictly_and_reported_once_per_scene_on_rank_zero(
    tmp_path, capsys, monkeypatch
):
    def plant(members):
        depth = members["depth"]
        # 24 is kept (strict >); 24.5 and a 30 outside the window are zeroed,
        # and the count is over every frame the clip holds -- out of its valid
        # pixels, which a pixel already invalid (0) is not.
        depth[0, 0, 0, 10, 10] = 24.0
        depth[0, 0, 0, 10, 11] = 24.5
        depth[1, 3, 0, 20, 30] = 30.0
        depth[1, 1, 0, 0, 0] = 0.0
        members["meta"]["depth0"] = depth[:, 0].copy()

    clips = tmp_path / "clips"
    _clip(clips, "a", mutate=plant)
    _clip(clips, "b")
    args = _args(tmp_path, clips, depth_input=True)
    held_out = train_cli._held_out_set(args, object(), _open(args))
    capsys.readouterr()

    scene = held_out.scene_provider(held_out.plans[0])

    assert scene.depth[0, 0, 0, 10, 10] == 24.0
    assert scene.depth[0, 0, 0, 10, 11] == 0.0 and scene.depth[1, 3, 0, 20, 30] == 0.0
    # The encoder's input reads the zeroed pixel invalid and the kept one at
    # the cap: one number for both depths.
    depth_input = scene.views[0]["depth_map"][0]
    assert depth_input[:, 10, 11].tolist() == [0.0, 0.0]
    assert depth_input[:, 10, 10].tolist() == [1.0, 1.0]
    # One line per scene, out of the valid pixels: 2 views x 4 frames x 56 x 56,
    # less the one planted invalid.
    assert _benchmark_lines(capsys.readouterr().out) == [
        "benchmark scene=a frames=4 label_depth_zeroed=2/25087 (0.0080%) beyond 24"
    ]
    held_out.scene_provider(held_out.plans[0])
    held_out.scene_provider(held_out.plans[1])
    assert _benchmark_lines(capsys.readouterr().out) == [
        "benchmark scene=b frames=4 label_depth_zeroed=0/25088 (0.0000%) beyond 24"
    ]
    # Every rank loads every scene at the preflight; only rank 0 says so.
    monkeypatch.setattr(scene_provider_module, "is_main_process", lambda: False)
    other_rank = _open(args)
    assert torch.equal(other_rank(held_out.plans[0]).depth, scene.depth)
    assert _benchmark_lines(capsys.readouterr().out) == []


def test_the_geometry_inputs_reach_the_views(tmp_path):
    clips = tmp_path / "clips"
    _clip(clips, "a")
    for flags, keys in (
        ({}, set()),
        ({"depth_input": True}, {"depth_map"}),
        ({"camera_input": True}, {"camera_vector"}),
        ({"depth_input": True, "camera_input": True}, {"depth_map", "camera_vector"}),
    ):
        args = _args(tmp_path, clips, **flags)
        held_out = train_cli._held_out_set(args, object(), _open(args))
        scene = held_out.scene_provider(held_out.plans[0])
        for view in scene.views:
            assert {"depth_map", "camera_vector"} & set(view) == keys, flags


def _corrupt_frame(members):
    members["frames"]["view_0/0002.png"] = b"not a png"


def test_a_clip_that_fails_to_read_raises_naming_it(tmp_path):
    """check_clip decodes no frame, so a broken one passes the opening and is
    refused at the first load."""

    clips = tmp_path / "clips"
    _clip(clips, "a", mutate=_corrupt_frame)
    args = _args(tmp_path, clips)
    held_out = train_cli._held_out_set(args, object(), _open(args))

    with pytest.raises(
        SceneProviderError, match=r"clip 'a' at .* is malformed: .*frames\.zip/view_0/0002\.png"
    ):
        held_out.scene_provider(held_out.plans[0])
    with pytest.raises(SceneProviderError, match="is not in the benchmark clip set"):
        held_out.scene_provider(replace(held_out.plans[0], seq_name="z"))


def test_the_preflight_ends_the_run_on_a_clip_that_fails_to_read(tmp_path):
    """A benchmark keeps every scene: nothing on the held-out path absorbs the
    failure, so the run ends before its first step."""

    clips = tmp_path / "clips"
    _clip(clips, "a")
    _clip(clips, "b", mutate=_corrupt_frame)
    args = _args(tmp_path, clips, num_steps=2)
    held_out = train_cli._held_out_set(args, object(), _open(args))
    scene = held_out.scene_provider(held_out.plans[0])
    steps = []

    torch.manual_seed(0)
    model = _FakeArc(scene.num_observations, *scene.views[0]["img"].shape[-2:])
    with pytest.raises(SceneProviderError, match="clip 'b'"):
        train_cli.run_training(
            model=model,
            optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=_plans(2),
            args=args,
            scene_provider=lambda plan: steps.append(plan) or scene,
            output_dir=tmp_path / "run",
            held_out=held_out,
        )
    assert steps == []


def test_the_clip_path_imports_nothing_from_mvtracker(tmp_path, monkeypatch):
    for name in [name for name in sys.modules if name.split(".")[0] == "mvtracker"]:
        monkeypatch.delitem(sys.modules, name)
    # Any import of it now raises.
    monkeypatch.setitem(sys.modules, "mvtracker", None)
    clips = tmp_path / "clips"
    _clip(clips, "a", query_times=(0, 1, 2))
    _clip(clips, "b")
    args = _args(tmp_path, clips)
    held_out = train_cli._held_out_set(args, object(), _open(args))
    _plant_pointmaps(monkeypatch, held_out.scene_provider(held_out.plans[0]))

    metrics, _ = _evaluate(tmp_path / "eval", held_out)

    assert metrics["scenes"] == 2
    with pytest.raises(ImportError):
        import mvtracker.datasets.utils  # noqa: F401


# ------------------------------------------------------------ the bundle ---


def test_every_query_is_written_in_clip_order(tmp_path, monkeypatch):
    """Clip a starts its tracks at frames 0, 1 and 2 on a window covering 0 and
    2: the time-0 track is scored, and the other two are the fallback rows --
    the query point held still, confidence 0, occluded throughout."""

    clips = tmp_path / "clips"
    _clip(clips, "b", query_times=(0, 0, 3))
    _clip(clips, "a", query_times=(0, 1, 2), invisible=((1, 2, 1),))
    args = _args(tmp_path, clips)
    held_out = train_cli._held_out_set(args, object(), _open(args))
    scene = held_out.scene_provider(held_out.plans[0])
    _plant_pointmaps(monkeypatch, scene)

    metrics, _ = _evaluate(tmp_path / "eval", held_out)

    bundle = _bundle(tmp_path / "eval", "a")
    assert bundle["query_points"].shape == (3, 4) and bundle["pred"].shape == (2, 3, 3)
    trajectories = scene.trajectories_world.numpy() * scene.track_upscaling_factor
    for track, covered_index in ((1, 0), (2, 1)):
        point = trajectories[int(scene.query_points[track, 0]), track]
        assert np.array_equal(bundle["query_points"][track], [covered_index, *point])
        assert np.array_equal(bundle["pred"][:, track], np.stack([point, point]))
        assert np.array_equal(bundle["conf"][:, track], [0.0, 0.0])
        assert bundle["occ"][:, track].all()
        assert np.array_equal(bundle["gt"][:, track], trajectories[[0, 2], track])
    assert bundle["conf"][:, 0].min() > 0.0
    assert _bundle(tmp_path / "eval", "b")["query_points"][:, 0].tolist() == [0.0, 0.0, 1.0]
    assert [entry["scene"] for entry in metrics["eval_benchmark"]["per_scene"]] == ["a", "b"]
    assert metrics["eval_benchmark"]["per_scene"][0]["unanchored_by_start_time"] == {
        "1": 1,
        "2": 1,
    }


@pytest.mark.parametrize("extra", [False, True], ids=["seated", "all"])
def test_the_clip_sets_eval_is_the_loader_benchmarks(tmp_path, monkeypatch, benchmark, extra):
    """The Kubric stub's own samples, scored through its loader and through
    their exported clip set: every bundle array and every figure the same, the
    benchmark's identity aside -- with the extra anchors' forward B too."""

    clips = _kubric_set(tmp_path, benchmark)
    runs = {
        "loader": _kubric_args(tmp_path, benchmark),
        "clips": _args(tmp_path, clips),
    }
    held_outs = {
        side: train_cli._held_out_set(args, object(), _open(args)) for side, args in runs.items()
    }
    _plant_pointmaps(
        monkeypatch, held_outs["loader"].scene_provider(held_outs["loader"].plans[0])
    )
    spec = (
        train_cli._eval_anchor_spec(_args(tmp_path, clips, eval_query_anchors=["all"]))
        if extra
        else None
    )

    metrics = {
        side: _evaluate(tmp_path / side, held_out, spec)[0]
        for side, held_out in held_outs.items()
    }

    for name in ("1", "2"):
        loaded = _bundle(tmp_path / "loader", name)
        read = _bundle(tmp_path / "clips", name)
        assert loaded.keys() == read.keys()
        for key in loaded:
            assert loaded[key].dtype == read[key].dtype, key
            assert np.array_equal(loaded[key], read[key], equal_nan=loaded[key].dtype.kind == "f"), key

    def without_identity(record):
        kept = dict(record)
        kept["eval_benchmark"] = {
            key: value
            for key, value in record["eval_benchmark"].items()
            if key in ("scene_count", "max_depth", "window", "per_scene")
        }
        return kept

    assert json.dumps(without_identity(metrics["clips"])) == json.dumps(
        without_identity(metrics["loader"])
    )
    assert metrics["clips"]["eval_benchmark"]["protocol"] == _PROTOCOL
    assert metrics["clips"]["eval_benchmark"]["name"] == "clips"


def test_extra_anchors_on_a_clip_set_are_its_cameras_not_its_positions(tmp_path, monkeypatch):
    """--eval_query_anchors resolves its slots on the scene's camera ids: on
    positions, slot 1:0 would be camera 1 again -- the seated anchor -- and
    every other slot a camera the window does not hold."""

    clips = tmp_path / "clips"
    _clip(clips, "a", view_count=4, view_ids=[1, 7, 14, 20], query_times=(0, 2, 2))
    args = _args(tmp_path, clips, val_cameras=[0, 1, 2, 3], eval_query_anchors=["all"])
    held_out = train_cli._held_out_set(args, object(), _open(args))
    _plant_pointmaps(monkeypatch, held_out.scene_provider(held_out.plans[0]))
    spec = train_cli._eval_anchor_spec(args)
    assert spec == ["1:0", "2:0", "3:0", "0:1", "1:1", "2:1", "3:1"]

    metrics, _ = _evaluate(tmp_path / "eval", held_out, spec)

    per_anchor = metrics["per_scene"][0]["eval_anchors"]["per_anchor"]
    assert list(per_anchor) == ["7:0", "14:0", "20:0", "1:2", "7:2", "14:2", "20:2"]
    assert sum(entry["assigned"] for entry in per_anchor.values()) == 2
    assert metrics["eval_benchmark"]["per_scene"][0]["admitted_by_start_time"] == {
        "0": 1,
        "2": 2,
    }


# ------------------------------------------------------------ the parity ---


def _parity(*argv):
    return parity.main([str(item) for item in argv])


def _kubric_parity(state, clips, *extra):
    return _parity(
        "--eval_benchmark_dataset",
        _KUBRIC_NAME,
        "--eval_benchmark_root",
        state["root"],
        "--eval_benchmark_clips",
        clips,
        "--val_cameras",
        "0",
        "1",
        "--val_seq_len",
        "4",
        "--size",
        "56",
        *extra,
    )


def _scene_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("parity scene=")]


def test_the_parity_flags_are_the_trainers_own():
    trainer = train_cli.build_arg_parser()._option_string_actions
    ours = parity.build_parser()._option_string_actions
    for flag in parity.WINDOW_FLAGS:
        mine, theirs = ours[flag], trainer[flag]
        assert type(mine) is type(theirs), flag
        assert (mine.dest, mine.default, mine.type, mine.nargs, mine.const) == (
            theirs.dest,
            theirs.default,
            theirs.type,
            theirs.nargs,
            theirs.const,
        ), flag


def test_the_comparator_reads_the_fields_no_clip_edit_reaches_alone(monkeypatch):
    """depth0, the observations and the correspondences follow from fields
    compared before them -- build_scene holds depth0 to depth[:, 0], and the
    rest are derived -- so no edit to a clip changes one of them alone. Each
    is pinned on two identical scenes with that one field made to differ."""

    def scene():
        return fixture_scene(
            cameras=(0, 1), times=(0, 2), size=56, time_count=4, query_times=(0, 1, 2)
        )

    expected, actual = scene(), scene()
    assert parity.first_difference(expected, actual) is None

    depth0 = actual.depth0.clone()
    depth0[1, 0, 3, 4] += 0.5
    assert parity.first_difference(expected, replace(actual, depth0=depth0)) == (
        "depth0",
        "max |diff| 0.5, 1 of 6272 elements differ",
    )

    observations = list(actual.observations)
    transform = observations[2].image_transform
    observations[2] = replace(
        observations[2], image_transform=replace(transform, crop_top=transform.crop_top + 1)
    )
    field, _ = parity.first_difference(expected, replace(actual, observations=tuple(observations)))
    assert field == "observations[2].image_transform"

    build = parity.build_anchor_correspondences

    def tampered(*, correspondences=None, eligibility=None):
        def tamper(target):
            built, record = build(target)
            if target is actual:
                if correspondences is not None:
                    built = correspondences(built)
                if eligibility is not None:
                    record = eligibility(copy.deepcopy(record))
            return built, record

        return tamper

    monkeypatch.setattr(
        parity,
        "build_anchor_correspondences",
        tampered(correspondences=lambda built: replace(built, query_slots=built.query_slots + 1)),
    )
    assert parity.first_difference(expected, actual)[0] == "correspondences.query_slots"

    def bump(record):
        record["per_anchor"][0]["eligible"] += 1
        return record

    monkeypatch.setattr(parity, "build_anchor_correspondences", tampered(eligibility=bump))
    field, detail = parity.first_difference(expected, actual)
    assert field == "eligibility.per_anchor[0].eligible" and "(|diff| 1)" in detail


def test_parity_passes_a_set_exported_from_the_loaders_own_samples(tmp_path, capsys, benchmark):
    clips = _kubric_set(tmp_path, benchmark)

    assert _kubric_parity(benchmark, clips, "--depth_input", "--camera_input") == 0

    out = capsys.readouterr().out
    assert _scene_lines(out) == [
        "parity scene=1 PASS frames=4/4",
        "parity scene=2 PASS frames=4/4",
    ]
    assert out.splitlines()[-1] == "parity: 2 of 2 scenes identical, 0 failed"
    assert "parity window: 2 cameras x times [0, 2] (bound window), anchors ['0:0']" in out
    assert json.dumps(_PROTOCOL) in out


def _edit_meta(key, edit, *, scene="1"):
    def mutate(name, members):
        if name == scene:
            array = np.array(members["meta"][key], copy=True)
            edit(array)
            members["meta"][key] = array

    return mutate


def _edit_depth(edit):
    def mutate(name, members):
        if name == "1":
            edit(members["depth"])

    return mutate


def _repaint(member):
    def mutate(name, members):
        if name == "1":
            pixels = np.asarray(Image.open(io.BytesIO(members["frames"][member]))).copy()
            pixels[7, 9, 0] += 1
            members["frames"][member] = _png(pixels)

    return mutate


def _flip(array, index):
    array[index] = not array[index]


CHANGES = [
    ("visibility", _edit_meta("visibility", lambda a: _flip(a, (0, 2, 0)))),
    ("extrinsics_world_to_camera", _edit_meta("extrs", lambda a: a.__setitem__((1, 0, 0, 3), a[1, 0, 0, 3] + 0.25))),
    ("views[3].img", _repaint("view_1/0002.png")),
    ("depth", _edit_depth(lambda d: d.__setitem__((0, 2, 0, 5, 5), d[0, 2, 0, 5, 5] + 0.5))),
    ("track_upscaling_factor", _edit_meta("track_upscaling_factor", lambda a: a.__setitem__((), 2.5))),
    # Scene 1's track 1 starts at frame 1, outside the window, where its
    # fallback row reads the trajectory: nudged within check_clip's tolerance.
    ("trajectories_world", _edit_meta("traj3d_world", lambda a: a.__setitem__((1, 1, 0), a[1, 1, 0] + 5e-6))),
]


@pytest.mark.parametrize("field, mutate", CHANGES, ids=[field for field, _ in CHANGES])
def test_a_one_element_change_fails_naming_the_field(tmp_path, capsys, benchmark, field, mutate):
    clips = _kubric_set(tmp_path, benchmark, mutate=mutate)

    assert _kubric_parity(benchmark, clips) == 1

    out = capsys.readouterr().out
    lines = _scene_lines(out)
    assert lines[0].startswith(f"parity scene=1 FAIL {field}: "), lines[0]
    assert "|diff| " in lines[0] and lines[0].endswith("(frames=4/4)"), lines[0]
    assert lines[1] == "parity scene=2 PASS frames=4/4"
    assert out.splitlines()[-1] == "parity: 1 of 2 scenes identical, 1 failed"


def test_a_change_the_eval_never_reads_still_passes(tmp_path, capsys, benchmark):
    """Visibility at frame 1, between the window's times 0 and 2."""

    clips = _kubric_set(
        tmp_path, benchmark, mutate=_edit_meta("visibility", lambda a: _flip(a, (0, 1, 0)))
    )

    assert _kubric_parity(benchmark, clips) == 0
    assert _scene_lines(capsys.readouterr().out) == [
        "parity scene=1 PASS frames=4/4",
        "parity scene=2 PASS frames=4/4",
    ]


def test_parity_compares_a_longer_loader_clip_on_what_the_eval_reads(tmp_path, capsys, dexycb):
    """DexYCB's 25-frame scenes against their 24-frame export, here 5 against 4,
    with depth beyond the cap in both: each side zeroes it on its own."""

    shutil.rmtree(dexycb["data_root"])
    _write_tree(dexycb["data_root"], {"s01": 4, "s02": 5})

    def sample(name):
        loaded = _dexycb_datapoint(name, time_count=5 if name == "s02" else 4)
        loaded.videodepth[0, 0, 0, 10, 10] = 24.0
        loaded.videodepth[0, 2, 0, 10, 11] = 30.0
        return loaded

    dexycb["sample"] = sample
    clips = _write_set(
        tmp_path / "clips",
        [(name, sample(name)) for name in ("s01", "s02")],
        frames=4,
        view_ids=[0, 1, 2, 3],
    )

    status = _parity(
        "--eval_benchmark_dataset",
        _DEXYCB_NAME,
        "--eval_benchmark_root",
        dexycb["root"],
        "--eval_benchmark_clips",
        clips,
        "--val_cameras",
        "0",
        "1",
        "2",
        "3",
        "--val_seq_len",
        "4",
        "--size",
        "56",
    )

    assert status == 0
    assert _scene_lines(capsys.readouterr().out) == [
        "parity scene=s01 PASS frames=4/4",
        "parity scene=s02 PASS frames=5/4",
    ]


def test_parity_fails_a_scene_on_one_side_only_or_one_that_does_not_load(
    tmp_path, capsys, benchmark
):
    loader = _open(_kubric_args(tmp_path, benchmark))
    stray = loader.dataset[0][0]
    clips = _kubric_set(tmp_path, benchmark, names={"2"})
    _write_set(clips, [("9", stray)], frames=4, view_ids=[0, 1])

    assert _kubric_parity(benchmark, clips) == 1
    out = capsys.readouterr().out
    assert _scene_lines(out) == [
        "parity scene=1 FAIL scene: in the loader's set, not in the clip set",
        "parity scene=9 FAIL scene: in the clip set, not in the loader's",
        "parity scene=2 PASS frames=4/4",
    ]
    assert out.splitlines()[-1] == "parity: 1 of 3 scenes identical, 2 failed"

    shutil.rmtree(clips)
    clips = _kubric_set(tmp_path, benchmark)
    benchmark["gotit"] = False
    assert _kubric_parity(benchmark, clips) == 1
    assert _scene_lines(capsys.readouterr().out) == [
        "parity scene=1 FAIL load (loader): SceneProviderError: benchmark scene '1' failed to load",
        "parity scene=2 FAIL load (loader): SceneProviderError: benchmark scene '2' failed to load",
    ]


def test_parity_reports_another_order_and_matches_by_name(tmp_path, capsys, benchmark):
    benchmark["names"][:] = ["2", "10", "1"]
    benchmark["query_times"]["10"] = (0, 0, 0)
    cache = benchmark["root"] / "kubric-multiview" / "test" / "10" / "cache"
    cache.mkdir(parents=True)
    (cache / f"{_CACHE}.npz").write_bytes(b"cached")
    clips = _kubric_set(tmp_path, benchmark)

    assert _kubric_parity(benchmark, clips) == 0

    out = capsys.readouterr().out
    assert (
        "parity order: the clip set lists the shared scenes in another order "
        "(scene:loader->clips) 2:0->2 10:1->1 1:2->0"
    ) in out
    assert _scene_lines(out) == [
        "parity scene=2 PASS frames=4/4",
        "parity scene=10 PASS frames=4/4",
        "parity scene=1 PASS frames=4/4",
    ]


def test_an_opening_refusal_reaches_the_parity_command_line(tmp_path, capsys, benchmark):
    with pytest.raises(SystemExit) as exit_info:
        _kubric_parity(benchmark, tmp_path / "absent")
    assert exit_info.value.code == 2
    assert "--eval_benchmark_clips" in capsys.readouterr().err
