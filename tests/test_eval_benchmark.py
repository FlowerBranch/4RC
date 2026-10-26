"""--eval_benchmark_dataset: an MVTracker benchmark's test scenes as the held-out set.

Opened by the benchmark's own protocol -- ``from_name(name, dataset_root=root,
subset="test", just_return_kwargs=True)`` then the constructor, with only the
label depth capped at --kubric_max_depth -- and scored on every one of its
queries, in dataset order: the rows an anchor admits, and for every other
query its point held still at confidence 0. MVTracker is stubbed through
``sys.modules``, as test_scene_provider.py does; the scenes are the in-memory
fixture world.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import train_temporal_tracking as train_cli
from arc.training import read_scene_predictions
from arc.training.scene_provider import SceneProviderError, benchmark_name_refusal
from scene_fixtures import fixture_datapoint
from test_eval_query_anchors import _SlotRecordingArc, _without_new_keys
from test_manifest_plan import _record, _write_manifest
from test_trainer_loop import _EVAL_ALPHA, _FakeArc, _loop_args, _plan_only, _plans

_NAME = "kubric-multiview-v3-views01-cached"
_CACHE = "cachedtracks--stub"


@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    """A two-scene stub of MVTracker's Kubric loader, recording every call.

    ``state["kwargs"]`` is what ``from_name`` returns beside the joined
    data_root, ``state["query_times"]`` each scene's query frames, and
    ``state["archived"]`` the scenes stored as archives. Every scene's track
    cache exists where the loader would look for it, unless a test removes it.
    """

    root = tmp_path / "benchmark"
    state = {
        "root": root,
        "names": ["1", "2"],
        "kwargs": {
            "seq_len": 4,
            "traj_per_sample": 512,
            "seed": 72,
            "max_videos": 30,
            "views_to_return": [0, 1],
            "num_views": -1,
            "use_cached_tracks": True,
        },
        "query_times": {"1": (0, 1, 2), "2": (0, 0, 3)},
        # Track 1 of scene 1 is hidden from camera 1 at frame 2, so a fallback
        # row's gt_vis_any reduces with any, not all.
        "invisible": {"1": ((1, 2, 1),)},
        # Scene 1 is stored at a factor of 2, so every row is lifted to metres.
        "factor": {"1": 2.0},
        "archived": set(),
        "from_name": [],
        "built": [],
        "loads": [],
        "gotit": True,
    }

    class _Dataset:
        cache_name = _CACHE

        def __init__(self, **kwargs):
            state["built"].append(dict(kwargs))
            self.data_root = kwargs["data_root"]
            self.seq_names = list(state["names"])
            self.seq_paths = [
                os.path.join(self.data_root, name + (".zip" if name in state["archived"] else ""))
                for name in self.seq_names
            ]

        @staticmethod
        def from_name(
            dataset_name,
            dataset_root,
            training_args=None,
            fabric=None,
            just_return_kwargs=False,
            subset="test",
        ):
            state["from_name"].append(
                {
                    "dataset_name": dataset_name,
                    "dataset_root": dataset_root,
                    "training_args": training_args,
                    "fabric": fabric,
                    "just_return_kwargs": just_return_kwargs,
                    "subset": subset,
                }
            )
            return {
                "data_root": os.path.join(dataset_root, "kubric-multiview", subset),
                **state["kwargs"],
            }

        def __getitem__(self, index):
            state["loads"].append(index)
            name = self.seq_names[index]
            sample = fixture_datapoint(
                scene_name=name,
                time_count=4,
                view_count=2,
                query_times=state["query_times"][name],
                invisible=state["invisible"].get(name, ()),
            )
            sample.track_upscaling_factor = state["factor"].get(name, 1.0)
            return sample, state["gotit"]

    loader = types.ModuleType("mvtracker.datasets.kubric_multiview_dataset")
    loader.KubricMultiViewDataset = _Dataset
    archive = types.ModuleType("mvtracker.datasets.scene_archive")
    archive.ARCHIVE_SUFFIX = ".zip"
    datasets = types.ModuleType("mvtracker.datasets")
    datasets.kubric_multiview_dataset = loader
    datasets.scene_archive = archive
    package = types.ModuleType("mvtracker")
    package.datasets = datasets
    for name, module in (
        ("mvtracker", package),
        ("mvtracker.datasets", datasets),
        ("mvtracker.datasets.kubric_multiview_dataset", loader),
        ("mvtracker.datasets.scene_archive", archive),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    test_root = root / "kubric-multiview" / "test"
    for name in state["names"]:
        cache = test_root / name / "cache"
        cache.mkdir(parents=True)
        (cache / f"{_CACHE}.npz").write_bytes(b"cached")
    return state


def _args(tmp_path, state, **overrides):
    """The held-out window on the stub's two cameras: four frames at stride 2."""

    return _loop_args(
        tmp_path,
        **{
            "eval_every": 2,
            # The fixture's 56-pixel frames, untransformed.
            "size": 56,
            "val_cameras": [0, 1],
            "val_seq_len": 4,
            "stride": 2,
            "eval_benchmark_dataset": _NAME,
            "eval_benchmark_root": str(state["root"]),
            **overrides,
        },
    )


def _provider(args):
    return train_cli._open_benchmark(args)


def _plant_pointmaps(monkeypatch, scene):
    """Identity alignment for the eval, as _cpu_eval_scene plants it."""

    import arc.training.sparse_tracking as sparse_module

    target, _ = sparse_module._metric_pointmap_at_anchor(scene, scene.query_observation_slot)
    pointmaps = torch.from_numpy(target).float().expand(
        1, scene.num_observations, *target.shape
    ).contiguous()
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)


def _evaluate(directory, held_out, eval_query_anchors=None, model=None, **kwargs):
    if model is None:
        scene = held_out.scene_provider(held_out.plans[0])
        torch.manual_seed(0)
        model = _SlotRecordingArc(scene.num_observations, *scene.views[0]["img"].shape[-2:])
    metrics = train_cli.evaluate_held_out(
        model=model,
        plans=held_out.plans,
        scene_provider=held_out.scene_provider,
        precision="32",
        huber_delta_m=0.05,
        step=3,
        output_dir=directory,
        query_anchors=["0:0"],
        confidence_alpha=_EVAL_ALPHA,
        query_anchor_source="predicted",
        eval_query_anchors=eval_query_anchors,
        benchmark=held_out.benchmark,
        **kwargs,
    )
    return metrics, model


def _bundle(directory, name):
    loaded = np.load(Path(directory) / "eval" / "step-3" / "pred" / f"{name}.npz")
    return {key: loaded[key] for key in loaded.files}


# ------------------------------------------------------------ the opening ---


def test_the_benchmark_is_opened_by_its_own_protocol_plus_the_depth_cap(tmp_path, benchmark):
    args = _args(tmp_path, benchmark, kubric_max_depth=24.0)
    provider = _provider(args)

    assert benchmark["from_name"] == [
        {
            "dataset_name": _NAME,
            "dataset_root": str(benchmark["root"]),
            "training_args": None,
            "fabric": None,
            "just_return_kwargs": True,
            "subset": "test",
        }
    ]
    # Exactly from_name's kwargs, with one key set on top: no replay override.
    returned = {
        "data_root": os.path.join(str(benchmark["root"]), "kubric-multiview", "test"),
        **benchmark["kwargs"],
    }
    assert benchmark["built"] == [{**returned, "max_depth": 24.0}]
    assert provider.identity() == {
        "dataset": _NAME,
        "root": str(benchmark["root"]),
        "subset": "test",
        "data_root": returned["data_root"],
        "scene_count": 2,
        "max_depth": 24.0,
    }
    # Opening loads nothing.
    assert benchmark["loads"] == []


def test_the_benchmark_replaces_the_scene_list_and_plans_every_scene(tmp_path, benchmark):
    args = _args(tmp_path, benchmark, val_scenes_file="never-read.json", val_data_root="/held")
    held_out = train_cli._held_out_set(args, object(), _provider(args))

    assert [plan.seq_name for plan in held_out.plans] == ["1", "2"]
    for plan in held_out.plans:
        assert plan.cameras == (0, 1)
        assert plan.times == (0, 2)
        assert plan.data_root.endswith(os.path.join("kubric-multiview", "test"))
        assert plan.track_indices == () and plan.scene_transform is None
        assert plan.real_len is None and plan.step == -1
    assert held_out.benchmark == {
        **held_out.scene_provider.identity(),
        "window": {"stride": 2, "covered_times": [0, 2]},
    }
    # Without the benchmark, the scene list on the training provider, as today.
    scenes = tmp_path / "val.json"
    scenes.write_text(json.dumps(["0000"]))
    provider = object()
    plain = _loop_args(tmp_path, val_scenes_file=str(scenes), val_data_root="/held")
    listed = train_cli._held_out_set(plain, provider, None)
    assert listed.scene_provider is provider and listed.benchmark is None
    assert listed.plans == train_cli._val_plans(plain)
    assert train_cli._held_out_set(
        _loop_args(tmp_path, val_scenes_file=None), provider, None
    ) is None


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda state: state["kwargs"].update(views_to_return=None), "fixes no camera set"),
        (
            lambda state: state["kwargs"].update(views_to_return=[1, 0]),
            r"returns cameras \[1, 0\], but the held-out window is --val_cameras \[0, 1\]",
        ),
        (lambda state: state["kwargs"].update(seq_len=24), "loads 24-frame clips"),
        (lambda state: state["kwargs"].update(seq_len=2), "loads 2-frame clips"),
        (
            lambda state: (
                state["root"] / "kubric-multiview" / "test" / "2" / "cache" / f"{_CACHE}.npz"
            ).unlink(),
            "1 of 2 scenes .* have no cached track file",
        ),
        (lambda state: state["names"].clear(), "holds no scenes"),
    ],
    ids=["no_views", "other_views", "longer_clip", "shorter_clip", "missing_cache", "empty"],
)
def test_what_opening_the_set_reveals_is_refused(tmp_path, benchmark, change, message):
    change(benchmark)
    with pytest.raises(ValueError, match=message):
        _provider(_args(tmp_path, benchmark))


def test_an_archived_scene_keeps_its_cache_where_the_loader_looks(tmp_path, benchmark):
    """kubric_multiview_dataset.py:925-931: an archive's cache sits under
    <data_root>/_trackcache/<scene>/, not beside the archive."""

    benchmark["archived"].add("2")
    with pytest.raises(ValueError, match="no cached track file"):
        _provider(_args(tmp_path, benchmark))

    cache = benchmark["root"] / "kubric-multiview" / "test" / "_trackcache" / "2"
    cache.mkdir(parents=True)
    (cache / f"{_CACHE}.npz").write_bytes(b"cached")
    assert _provider(_args(tmp_path, benchmark)).scene_names == ["1", "2"]


def test_a_benchmark_is_opened_before_the_model_loads(tmp_path, benchmark, monkeypatch, capsys):
    # The parser's own window: four cameras over a 24-frame clip, which the
    # stub's two cameras do not fill.
    benchmark["kwargs"]["seq_len"] = 24
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
            "--eval_benchmark_dataset",
            _NAME,
            "--eval_benchmark_root",
            str(benchmark["root"]),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    assert (
        "returns cameras [0, 1], but the held-out window is --val_cameras [0, 1, 2, 3]"
        in capsys.readouterr().err
    )


# ------------------------------------------------------------ the parse ---


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"eval_benchmark_root": None}, "--eval_benchmark_dataset needs --eval_benchmark_root"),
        (
            {"eval_benchmark_dataset": None},
            "--eval_benchmark_root needs --eval_benchmark_dataset",
        ),
        (
            {"eval_benchmark_dataset": "panoptic-multiview-views1_7_14_20"},
            "is not a supported benchmark",
        ),
        (
            {"eval_benchmark_dataset": "kubric-multiview-v3-views0123"},
            "reads no cached track set",
        ),
        ({"eval_every": 0}, "--eval_benchmark_dataset needs a nonzero --eval_every"),
    ],
    ids=["no_root", "no_name", "other_family", "not_cached", "eval_off"],
)
def test_the_benchmark_flags_are_refused_at_parse_time(tmp_path, overrides, message):
    args = _loop_args(
        tmp_path,
        **{
            "max_unreplayable_fraction": 0.02,
            "max_records": None,
            "val_scenes_file": None,
            "val_data_root": None,
            "embedding_lr": None,
            "encoder_lr": None,
            "confidence_alpha": "auto",
            "eval_every": 500,
            "eval_benchmark_dataset": "kubric-multiview-v3-views0123-cached",
            "eval_benchmark_root": "/bench",
            **overrides,
        },
    )
    with pytest.raises(ValueError, match=message):
        train_cli._validate_args(args)


def test_the_name_alone_decides_the_parse_time_refusal():
    assert benchmark_name_refusal("kubric-multiview-v3-views0123-cached") is None
    assert benchmark_name_refusal("kubric-multiview-v3-duster0123-cached") is None
    assert "not a supported benchmark" in benchmark_name_refusal("dex-ycb-multiview")
    assert "reads no cached track set" in benchmark_name_refusal("kubric-multiview-v3")


def test_the_benchmark_is_a_held_out_set_of_its_own(tmp_path):
    """Every eval-only flag validates beside the benchmark alone, and the
    held-out window's seating is still checked: strict, a slot the window
    cannot seat is refused without --val_scenes_file too."""

    base = dict(
        max_unreplayable_fraction=0.02,
        max_records=None,
        val_scenes_file=None,
        val_data_root=None,
        embedding_lr=None,
        encoder_lr=None,
        confidence_alpha="auto",
        eval_every=500,
        eval_benchmark_dataset="kubric-multiview-v3-views0123-cached",
        eval_benchmark_root="/bench",
    )
    args = _loop_args(
        tmp_path,
        **base,
        eval_query_anchors=["all"],
        eval_query_anchor_source="gt_depth",
        eval_visual_scenes=["1"],
    )
    train_cli._validate_args(args)
    assert train_cli._has_held_out(args)
    assert len(train_cli._eval_anchor_spec(args)) == 47
    # The two aliases of the eval source, one at a time: they gate on the same
    # held-out set.
    for alias in ("oracle_query_anchor", "ground_truth_query_anchor"):
        train_cli._validate_args(_loop_args(tmp_path, **base, **{alias: True}))

    with pytest.raises(ValueError, match="time slot 12 does not fit the held-out window"):
        train_cli._validate_args(_loop_args(tmp_path, **base, query_anchors=["0:12"]))


def test_the_plan_summary_names_the_benchmark_eval(tmp_path, monkeypatch):
    """A benchmark-only run has a held-out set, so plan.json records the
    window's seated anchors and the eval's anchor source, as it does for a
    scene list."""

    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(step=0)])
    out = tmp_path / "plan.json"
    _plan_only(
        monkeypatch,
        manifest,
        "--eval_benchmark_dataset",
        "kubric-multiview-v3-views0123-cached",
        "--eval_benchmark_root",
        "/bench",
        "--json_out",
        str(out),
    )

    train_cli.main()

    settings = json.loads(out.read_text())["settings"]
    assert settings["realized_val_query_anchors"] == ["0:0"]
    assert settings["eval_query_anchor_source"] == "predicted"


def test_a_name_refusal_reaches_the_command_line_before_any_load(tmp_path, monkeypatch, capsys):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(step=0)])
    _plan_only(
        monkeypatch,
        manifest,
        "--eval_benchmark_dataset",
        "kubric-multiview-v3-views0123",
        "--eval_benchmark_root",
        "/bench",
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    assert "reads no cached track set" in capsys.readouterr().err


# ------------------------------------------------------------ the scenes ---


def test_the_benchmark_is_anchored_at_the_slots_the_window_seats(tmp_path, benchmark):
    """Under a ceiling the spec is wider than the window; the benchmark's
    scenes are anchored at what the window seats, as the scene list's are."""

    args = _args(
        tmp_path,
        benchmark,
        query_anchors=["0:0", "1:0", "2:0", "3:0"],
        adaptive_query_anchors=True,
    )
    provider = _provider(args)
    assert provider.query_anchor_slots == train_cli._val_anchor_slots(args) == ((0, 0), (1, 0))
    held_out = train_cli._held_out_set(args, object(), provider)
    assert held_out.scene_provider(held_out.plans[0]).query_anchors == ((0, 0), (1, 0))


def test_a_benchmark_scene_is_the_sample_as_loaded(tmp_path, benchmark):
    """No world transform and no column selection: the scene's queries are the
    dataset's, all of them, in order, anchored at the seated slots."""

    args = _args(
        tmp_path, benchmark, query_anchors=["1:0", "0:1"], depth_input=True, camera_input=True
    )
    held_out = train_cli._held_out_set(args, object(), _provider(args))
    sample = fixture_datapoint(
        scene_name="2", time_count=4, view_count=2, query_times=(0, 0, 3)
    )

    scene = held_out.scene_provider(held_out.plans[1])

    assert benchmark["loads"] == [1]
    assert scene.name == "2"
    assert torch.equal(scene.query_points, sample.query_points_3d)
    assert torch.equal(scene.trajectories_world, sample.trajectory_3d)
    assert torch.equal(scene.visibility, sample.visibility)
    assert scene.query_anchors == ((1, 0), (0, 2))
    assert scene.times == (0, 2) and scene.camera_ids == (0, 1)
    assert scene.track_upscaling_factor == 1.0
    # Both geometry inputs the run trained on reach the encoder, the depth one
    # normalised at the run's cap, as on the held-out path...
    assert "depth_map" in scene.views[0] and "camera_vector" in scene.views[0]
    # ...and neither without its flag.
    plain_args = _args(tmp_path, benchmark, query_anchors=["1:0", "0:1"])
    plain = train_cli._held_out_set(plain_args, object(), _provider(plain_args))
    plain_scene = plain.scene_provider(plain.plans[1])
    assert "depth_map" not in plain_scene.views[0]
    assert "camera_vector" not in plain_scene.views[0]

    with pytest.raises(SceneProviderError, match="not in the benchmark"):
        held_out.scene_provider(held_out.plans[0].__class__(**{**vars(held_out.plans[0]), "seq_name": "9"}))
    benchmark["gotit"] = False
    with pytest.raises(SceneProviderError, match="failed to load"):
        held_out.scene_provider(held_out.plans[0])


# ------------------------------------------------------------ the bundle ---


def test_every_query_is_written_in_dataset_order(tmp_path, monkeypatch, benchmark):
    """Scene 1 starts its tracks at frames 0, 1 and 2 on a window covering 0
    and 2: the time-0 track is scored, the odd-frame one can only fall back,
    and the frame-2 one falls back without extra anchors and is scored with
    them. Scene 2's frame-3 track starts past the window's last time."""

    args = _args(tmp_path, benchmark)
    held_out = train_cli._held_out_set(args, object(), _provider(args))
    scene = held_out.scene_provider(held_out.plans[0])
    _plant_pointmaps(monkeypatch, scene)
    spec = train_cli._eval_anchor_spec(_args(tmp_path, benchmark, eval_query_anchors=["all"]))
    assert spec == ["1:0", "0:1", "1:1"]

    plain, _ = _evaluate(tmp_path / "plain", train_cli.HeldOut(held_out.plans, held_out.scene_provider, None))
    off, _ = _evaluate(tmp_path / "off", held_out)
    on, on_model = _evaluate(tmp_path / "on", held_out, spec)
    rows = {name: _bundle(tmp_path / name, "1") for name in ("plain", "off", "on")}

    # One row per query, in dataset order.
    for name in ("off", "on"):
        assert rows[name]["query_points"].shape == (3, 4)
        assert rows[name]["pred"].shape == (2, 3, 3)
    # The scored rows are the non-benchmark eval's rows.
    for key, value in rows["plain"].items():
        if key == "query_points":
            assert np.array_equal(rows["off"][key][:1], value), key
            assert np.array_equal(rows["on"][key][:1], value), key
        elif value.ndim >= 2:
            assert np.array_equal(rows["off"][key][:, :1], value), key
            assert np.array_equal(rows["on"][key][:, :1], value), key
        else:
            assert np.array_equal(rows["off"][key], value, equal_nan=True), key

    # The fallback rows: the query point held still, confidence 0, occluded,
    # with the ground truth a scored row would carry.
    # In metres: scene 1 is stored at a factor of 2.
    assert scene.track_upscaling_factor == 2.0
    trajectories = scene.trajectories_world.numpy() * scene.track_upscaling_factor
    visibility = scene.visibility.numpy()
    for name, fallback, index in (("off", (1, 2), (0, 1)), ("on", (1,), (0,))):
        bundle = rows[name]
        for track, covered_index in zip(fallback, index):
            start = int(scene.query_points[track, 0])
            point = trajectories[start, track]
            assert np.array_equal(bundle["query_points"][track], [covered_index, *point])
            assert np.array_equal(bundle["pred"][:, track], np.stack([point, point]))
            assert np.array_equal(bundle["conf"][:, track], [0.0, 0.0])
            assert bundle["occ"][:, track].all()
            assert np.array_equal(bundle["gt"][:, track], trajectories[[0, 2], track])
            assert np.array_equal(
                bundle["gt_vis_any"][:, track], visibility[:, [0, 2], track].any(axis=0)
            )
    # The hidden camera does not hide the point: camera 0 still sees it.
    assert not visibility[:, 2, 1].all() and rows["off"]["gt_vis_any"][1, 1]
    # With the extra anchors the frame-2 track is scored at its own time,
    # through forward B, whose only anchor is 0:1 -- camera 0 at frame 2.
    assert rows["on"]["query_points"][2, 0] == 1.0
    assert rows["on"]["conf"][:, 2].min() > 1.0
    assert not rows["on"]["occ"][:, 2].any()
    assert on_model.forward_slots[2] == [1]

    # Counts by original start frame: the odd frame shows by name.
    assert off["eval_benchmark"]["per_scene"][0] == {
        "scene": "1",
        "query_count": 3,
        "skipped": False,
        "admitted_by_start_time": {"0": 1},
        "unanchored_by_start_time": {"1": 1, "2": 1},
        "unanchored_rejected": {
            "query_time_mismatch": 2,
            "not_visible_in_anchor": 0,
            "projection": 0,
            "anchor_depth_gate": 0,
            "pixel_dedup": 0,
        },
    }
    assert on["eval_benchmark"]["per_scene"][0]["admitted_by_start_time"] == {"0": 1, "2": 1}
    assert on["eval_benchmark"]["per_scene"][0]["unanchored_by_start_time"] == {"1": 1}
    assert on["eval_benchmark"]["per_scene"][0]["unanchored_rejected"]["query_time_mismatch"] == 1
    # Scene 2: its frame-3 track lies past the window and lands on its last index.
    second = _bundle(tmp_path / "off", "2")
    assert second["query_points"][:, 0].tolist() == [0.0, 0.0, 1.0]
    assert off["eval_benchmark"]["per_scene"][1]["unanchored_by_start_time"] == {"3": 1}
    # The record: the set's identity and window, then every scene in order.
    assert off["eval_benchmark"] == {
        **held_out.benchmark,
        "per_scene": off["eval_benchmark"]["per_scene"],
    }
    assert [entry["scene"] for entry in off["eval_benchmark"]["per_scene"]] == ["1", "2"]
    for entry in off["eval_benchmark"]["per_scene"]:
        admitted = sum(entry["admitted_by_start_time"].values())
        unanchored = sum(entry["unanchored_by_start_time"].values())
        assert admitted + unanchored == entry["query_count"]
        assert unanchored == sum(entry["unanchored_rejected"].values())
    # Every figure but the bundle layout is the non-benchmark eval's.
    assert json.dumps(_without_new_keys(off)) == json.dumps(_without_new_keys(plain))
    assert plain["eval_benchmark"] is None


def test_a_scene_no_seated_anchor_reaches_still_writes_every_query(tmp_path, monkeypatch, benchmark):
    benchmark["query_times"]["1"] = (1, 2, 3)
    args = _args(tmp_path, benchmark)
    held_out = train_cli._held_out_set(args, object(), _provider(args))
    _plant_pointmaps(monkeypatch, held_out.scene_provider(held_out.plans[1]))

    metrics, model = _evaluate(tmp_path, held_out)

    assert [entry["scene"] for entry in metrics["skipped_scenes"]] == ["1"]
    assert metrics["scenes"] == 1
    bundle = read_scene_predictions(Path(tmp_path) / "eval" / "step-3" / "pred" / "1.npz")
    assert bundle["query_points"][:, 0].tolist() == [0.0, 1.0, 1.0]
    assert np.array_equal(bundle["conf"], np.zeros((2, 3), dtype=np.float32))
    assert bundle["occ"].all()
    assert metrics["eval_benchmark"]["per_scene"][0] == {
        "scene": "1",
        "query_count": 3,
        "skipped": True,
        "admitted_by_start_time": {},
        "unanchored_by_start_time": {"1": 1, "2": 1, "3": 1},
        "unanchored_rejected": metrics["skipped_scenes"][0]["eligibility"]["rejected"],
    }
    # Only the scored scene ran a forward: two, the plain arm and the shuffled.
    assert len(model.forward_slots) == 2


def test_the_scene_line_counts_every_benchmark_query(tmp_path, monkeypatch, benchmark, capsys):
    args = _args(tmp_path, benchmark)
    held_out = train_cli._held_out_set(args, object(), _provider(args))
    _plant_pointmaps(monkeypatch, held_out.scene_provider(held_out.plans[0]))

    _evaluate(tmp_path, held_out)

    lines = [line for line in capsys.readouterr().out.splitlines() if "anchors=" in line]
    assert [line.split(" seconds=")[0] for line in lines] == [
        "eval step=3 scene=1 anchors=1+0/0 rows=1+0/3",
        "eval step=3 scene=2 anchors=1+0/0 rows=2+0/3",
    ]


def test_the_benchmark_keeps_the_datasets_own_scene_order(tmp_path, benchmark):
    benchmark["names"][:] = ["2", "10", "1"]
    benchmark["query_times"]["10"] = (0, 0, 0)
    cache = benchmark["root"] / "kubric-multiview" / "test" / "10" / "cache"
    cache.mkdir(parents=True)
    (cache / f"{_CACHE}.npz").write_bytes(b"cached")

    args = _args(tmp_path, benchmark)
    held_out = train_cli._held_out_set(args, object(), _provider(args))

    assert [plan.seq_name for plan in held_out.plans] == ["2", "10", "1"]


def test_counts_are_keyed_in_frame_order():
    assert list(train_cli._count_by_frame([3, 1, 3, 0, 10])) == ["0", "1", "3", "10"]
    assert train_cli._count_by_frame([3, 1, 3, 0, 10]) == {"0": 1, "1": 1, "3": 2, "10": 1}


def test_the_fallback_index_is_the_last_covered_time_at_or_before_the_start():
    from scene_fixtures import fixture_scene

    scene = fixture_scene(
        cameras=(0, 1), times=(0, 2), size=56, time_count=4, query_times=(0, 1, 3)
    )
    columns = train_cli._benchmark_columns(scene, None, torch.zeros(0, dtype=torch.long))
    assert columns["query_points"][:, 0].tolist() == [0.0, 0.0, 1.0]
    covered_start = fixture_scene(
        cameras=(0, 1), times=(0, 1, 2, 3), size=56, time_count=4, query_times=(0, 1, 3)
    )
    columns = train_cli._benchmark_columns(covered_start, None, torch.zeros(0, dtype=torch.long))
    assert columns["query_points"][:, 0].tolist() == [0.0, 1.0, 3.0]


# ------------------------------------------------------------ the loop ---


def test_the_benchmark_moves_no_training_byte(tmp_path, monkeypatch, benchmark):
    """The real step, scored on the scene list or on the benchmark: the
    training artifacts are byte for byte the same either way."""

    args = _args(tmp_path, benchmark)
    benchmark_held_out = train_cli._held_out_set(args, object(), _provider(args))
    scene = benchmark_held_out.scene_provider(benchmark_held_out.plans[0])
    _plant_pointmaps(monkeypatch, scene)
    listed = train_cli.HeldOut(
        plans=[benchmark_held_out.plans[0]], scene_provider=lambda _plan: scene, benchmark=None
    )

    def run(directory, held_out, **overrides):
        torch.manual_seed(0)
        model = _FakeArc(scene.num_observations, *scene.views[0]["img"].shape[-2:])
        return train_cli.run_training(
            model=model,
            optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=_plans(4),
            args=_args(directory, benchmark, num_steps=4, save_every=2, **overrides),
            scene_provider=lambda _plan: scene,
            output_dir=directory,
            held_out=held_out,
        )

    listed_run = run(tmp_path / "listed", listed, eval_benchmark_dataset=None, eval_benchmark_root=None)
    benchmark["loads"].clear()
    benchmark_run = run(tmp_path / "benchmark", benchmark_held_out)

    # The preflight and both evals load through the benchmark's own provider,
    # every scene in order; the steps never touch it.
    assert benchmark["loads"] == [0, 1, 0, 1, 0, 1]
    for name in ("history.jsonl", "train_state.pt"):
        assert (tmp_path / "benchmark" / name).read_bytes() == (
            tmp_path / "listed" / name
        ).read_bytes()
    assert [entry["step"] for entry in benchmark_run["evaluations"]] == [2, 4]
    assert benchmark_run["evaluations"][0]["eval_benchmark"]["scene_count"] == 2
    assert listed_run["evaluations"][0]["eval_benchmark"] is None
