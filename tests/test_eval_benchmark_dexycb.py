"""--eval_benchmark_dataset dex-ycb-multiview-views<digits>-cached: DexYCB as the held-out set.

Opened by its own protocol -- ``from_name(name, dataset_root=root)``, which
builds the dataset itself -- with what the provider checks read off the dataset
and the data: the cameras, each scene's frame count, the track cache. Every
loaded sample is completed in place: label depth capped at --kubric_max_depth
in the loader's stored units (metres x6), and ``sample_views`` stamped from
``views_to_return``. MVTracker is stubbed through ``sys.modules`` as in
test_eval_benchmark.py, over a synthetic tree in the loader's layout; the
scenes are the fixture world stored at a factor of 1/6, as DexYCB's are.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import arc.training.scene_provider as scene_provider_module
import train_temporal_tracking as train_cli
from arc.training.scene_provider import SceneProviderError, benchmark_name_refusal
from arc.training.visual_dump import read_visual_dump
from scene_fixtures import fixture_datapoint
from test_eval_benchmark import _bundle, _evaluate, _plant_pointmaps, _provider
from test_manifest_plan import _record, _write_manifest
from test_trainer_loop import _loop_args, _plan_only

_NAME = "dex-ycb-multiview-views0123-cached"
_SIXTH = 1 / 6
# The stored-unit gauges: stored length per model length, so a scene stored at
# metres x6 reads them six times larger. Every other float in the metrics is
# metres, a count or a ratio.
_GAUGES = ("alignment_scale", "baseline_scale")
_IDENTITY = {"checkpoint_dir": "released/4rc", "commit": "0123abcd"}


def _cache_name(views=(0, 1, 2, 3)) -> str:
    """The loader's cache file name at from_name's settings (seed 72, 384 tracks)."""

    return f"cachedtracks--seed72-views{'_'.join(map(str, views))}-n384--v1"


def _write_tree(data_root: Path, frames: dict, *, view_folders=4, views=(0, 1, 2, 3)):
    """One scene directory per ``frames`` entry (scene -> frame count), in the
    loader's layout. The opener only lists and stats, so every file is empty
    but the cache, which the stub's load tells apart from a redrawn one."""

    for scene, count in frames.items():
        scene_dir = data_root / scene
        for view in range(view_folders):
            view_dir = scene_dir / f"view_{view:02d}"
            for kind, suffix in (("rgb", "jpg"), ("depth", "png")):
                (view_dir / kind).mkdir(parents=True)
                for time_index in range(count):
                    (view_dir / kind / f"{time_index:06d}.{suffix}").write_bytes(b"")
            (view_dir / "intrinsics_extrinsics.npz").write_bytes(b"")
        (scene_dir / "tracks_3d.npz").write_bytes(b"")
        (scene_dir / "cache").mkdir()
        (scene_dir / "cache" / f"{_cache_name(views)}.npz").write_bytes(b"cached")


def _listing(root: Path) -> list:
    """Every path under ``root`` with its size, so a written or grown file shows."""

    return sorted(
        (path.relative_to(root).as_posix(), path.stat().st_size if path.is_file() else None)
        for path in root.rglob("*")
    )


def _dexycb_datapoint(name, *, time_count=4):
    """The fixture world shaped as DexYCB's Datapoint: four views of float
    video, no camera ids, a factor of 1/6, and a track that starts at frame 2."""

    sample = fixture_datapoint(
        scene_name=name, view_count=4, time_count=time_count, query_times=(0, 0, 2)
    )
    sample.video = sample.video.float()
    sample.sample_views = None
    sample.track_upscaling_factor = _SIXTH
    return sample


@pytest.fixture
def dexycb(monkeypatch, tmp_path):
    """A stub of MVTracker's DexYCB loader over a two-scene tree, recording every call.

    ``state["sample"]`` makes each scene's sample (a fresh one per load, as the
    loader's is), ``state["from_name"]`` and ``state["loads"]`` record the calls.
    """

    root = tmp_path / "benchmark"
    state = {
        "root": root,
        "data_root": root / "dex-ycb-multiview",
        "from_name": [],
        "loads": [],
        "gotit": True,
        "sample": _dexycb_datapoint,
    }

    class _Dataset:
        """DexYCBMultiViewDataset as far as the opener and the provider reach it."""

        def __init__(self, *, data_root, views_to_return, use_cached_tracks):
            self.data_root = data_root
            self.views_to_return = views_to_return
            self.traj_per_sample = 384
            self.seed = 72
            self.remove_hand = False
            self.use_cached_tracks = use_cached_tracks
            self.cache_name = self._cache_key()
            self.seq_names = self._get_sequence_names(10)

        @staticmethod
        def from_name(dataset_name, dataset_root):
            state["from_name"].append(
                {"dataset_name": dataset_name, "dataset_root": dataset_root}
            )
            views = re.search(r"-views(\d+)", dataset_name)
            return _Dataset(
                data_root=os.path.join(dataset_root, "dex-ycb-multiview"),
                views_to_return=list(map(int, views.group(1))),
                use_cached_tracks="-cached" in dataset_name,
            )

        # dexycb_multiview_dataset.py:139-185 (MVTracker 4002795), less its prints.
        def _get_sequence_names(self, max_videos):
            names = sorted(
                name
                for name in os.listdir(self.data_root)
                if os.path.isdir(os.path.join(self.data_root, name))
                and not name.startswith(".")
                and not name.startswith("_")
            )
            valid = [
                name
                for name in names
                if any(
                    os.path.isdir(os.path.join(self.data_root, name, entry))
                    and entry.startswith("view_")
                    for entry in os.listdir(os.path.join(self.data_root, name))
                )
            ]
            return valid[:max_videos]

        def _cache_key(self):
            name = f"cachedtracks--seed{self.seed}"
            name += f"-views{'_'.join(map(str, self.views_to_return))}"
            name += f"-n{self.traj_per_sample}"
            if self.remove_hand:
                name += "-removehand"
            return name + "--v1"

        def __getitem__(self, index):
            # The loader's own side effects (:385-387, :464-475): the cache
            # directory is made unconditionally, and a missing file is written.
            state["loads"].append(index)
            cache_root = os.path.join(self.data_root, self.seq_names[index], "cache")
            os.makedirs(cache_root, exist_ok=True)
            cache_file = os.path.join(cache_root, f"{self.cache_name}.npz")
            if self.use_cached_tracks and not os.path.isfile(cache_file):
                Path(cache_file).write_bytes(b"redrawn")
            return state["sample"](self.seq_names[index]), state["gotit"]

    loader = types.ModuleType("mvtracker.datasets.dexycb_multiview_dataset")
    loader.DexYCBMultiViewDataset = _Dataset
    datasets = types.ModuleType("mvtracker.datasets")
    datasets.dexycb_multiview_dataset = loader
    package = types.ModuleType("mvtracker")
    package.datasets = datasets
    for name, module in (
        ("mvtracker", package),
        ("mvtracker.datasets", datasets),
        ("mvtracker.datasets.dexycb_multiview_dataset", loader),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    _write_tree(state["data_root"], {"s01": 4, "s02": 4})
    return state


def _args(tmp_path, state, **overrides):
    """The held-out window on the stub's four cameras: four frames at stride 2."""

    return _loop_args(
        tmp_path,
        **{
            "eval_every": 2,
            # The fixture's 56-pixel frames, untransformed.
            "size": 56,
            "val_cameras": [0, 1, 2, 3],
            "val_seq_len": 4,
            "stride": 2,
            "eval_benchmark_dataset": _NAME,
            "eval_benchmark_root": str(state["root"]),
            **overrides,
        },
    )


def _benchmark_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("benchmark scene=")]


# ------------------------------------------------------------ the name ---


@pytest.mark.parametrize(
    "name, message",
    [
        (_NAME, None),
        ("dex-ycb-multiview-views4567-cached", None),
        ("dex-ycb-multiview-views01-cached", None),
        ("dex-ycb-multiview-views0123", "reads no cached track set"),
        ("dex-ycb-multiview", "reads no cached track set"),
        ("dex-ycb-multiview-duster0123-views0123-cached", "-duster0123 swaps the sensor depth"),
        (
            "dex-ycb-multiview-duster0123cleaned-views0123-cached",
            "-duster0123cleaned swaps the sensor depth",
        ),
        ("dex-ycb-multiview-vggt0123-views0123-cached", "-vggt0123 swaps the sensor depth"),
        ("dex-ycb-multiview-duster0123-views0123", "-duster0123 swaps the sensor depth"),
        (
            "dex-ycb-multiview-views0123-novelviews45-cached",
            "-novelviews45 loads cameras beyond the returned ones",
        ),
        ("dex-ycb-multiview-views0123-removehand-cached", "-removehand drops the hand's tracks"),
        ("dex-ycb-multiview-views0123-single-cached", "-single is MVTracker's one-query"),
        ("dex-ycb-multiview-views0123-2dpt-cached", "-2dpt scores 2-D point tracks"),
        ("dex-ycb-multiview-cached", "names no cameras"),
        ("dex-ycb-multiview-views0011-cached", r"names a camera twice \(views0011\)"),
        ("dex-ycb-multiview-views0123-cached-extra", "reads no cached track set"),
        ("dex-ycb-multiviewx-views0123-cached", "is not of the form"),
        ("dexycb-multiview-views0123-cached", "is not a supported benchmark"),
    ],
)
def test_a_dexycb_name_is_refused_by_what_it_changes(monkeypatch, name, message):
    # The parse-time half imports nothing: an import of MVTracker would raise.
    monkeypatch.setitem(sys.modules, "mvtracker", None)
    refusal = benchmark_name_refusal(name)
    if message is None:
        assert refusal is None
    else:
        assert re.search(message, refusal), refusal
        assert name in refusal


def test_a_dexycb_name_refusal_reaches_the_command_line_before_any_load(
    tmp_path, monkeypatch, capsys
):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(step=0)])
    _plan_only(
        monkeypatch,
        manifest,
        "--eval_benchmark_dataset",
        "dex-ycb-multiview-views0123-removehand-cached",
        "--eval_benchmark_root",
        "/bench",
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    assert "-removehand drops the hand's tracks" in capsys.readouterr().err


# ------------------------------------------------------------ the opening ---


def test_dexycb_is_opened_by_its_own_protocol(tmp_path, dexycb, capsys):
    before = _listing(dexycb["root"])
    args = _args(tmp_path, dexycb)
    provider = _provider(args)

    assert dexycb["from_name"] == [
        {"dataset_name": _NAME, "dataset_root": str(dexycb["root"])}
    ]
    data_root = os.path.join(str(dexycb["root"]), "dex-ycb-multiview")
    # What the provider checks, read off the dataset and the data: no seq_len,
    # the loader has none.
    assert provider.kwargs == {
        "data_root": data_root,
        "views_to_return": [0, 1, 2, 3],
        "scene_frames": {"s01": 4, "s02": 4},
    }
    assert provider.identity() == {
        "dataset": _NAME,
        "root": str(dexycb["root"]),
        "subset": "test",
        "data_root": data_root,
        "scene_count": 2,
        "max_depth": 24.0,
    }
    assert provider.scene_names == ["s01", "s02"]
    held_out = train_cli._held_out_set(args, object(), provider)
    for plan in held_out.plans:
        assert plan.cameras == (0, 1, 2, 3) and plan.times == (0, 2)
        assert plan.data_root == data_root and plan.seq_len == 4
    assert held_out.benchmark == {
        **provider.identity(),
        "window": {"stride": 2, "covered_times": [0, 2]},
    }
    # Opening loads nothing, writes nothing and says nothing.
    assert dexycb["loads"] == []
    assert _listing(dexycb["root"]) == before
    assert _benchmark_lines(capsys.readouterr().out) == []


def test_dexycb_opens_only_the_scenes_the_loader_lists(tmp_path, dexycb):
    """The loader keeps the first 10 sorted directories holding a view folder;
    the opener checks those and nothing else -- an eleventh scene without a
    cache is never loaded, so it is not refused."""

    data_root = dexycb["data_root"]
    shutil.rmtree(data_root)
    _write_tree(data_root, {f"s{index:02d}": 4 for index in range(1, 12)})
    (data_root / "s11" / "cache" / f"{_cache_name()}.npz").unlink()
    (data_root / "s00").mkdir()
    (data_root / "_x" / "view_00").mkdir(parents=True)
    (data_root / "notes.txt").write_text("not a scene")

    provider = _provider(_args(tmp_path, dexycb))

    assert provider.scene_names == [f"s{index:02d}" for index in range(1, 11)]
    assert list(provider.kwargs["scene_frames"]) == provider.scene_names


def _remove(*parts):
    def change(state):
        path = state["data_root"].joinpath(*parts)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    return change


def _drop_frames(*kinds):
    def change(state):
        for kind in kinds:
            frames = sorted((state["data_root"] / "s01" / "view_02" / kind).iterdir())
            frames[-1].unlink()

    return change


@pytest.mark.parametrize(
    "change, overrides, message",
    [
        (
            lambda state: shutil.rmtree(state["data_root"]),
            {},
            "has no dataset directory at .*dex-ycb-multiview",
        ),
        (_remove("s02", "view_03"), {}, "scene 's02' .* has 3 view folders, but the set returns view 3"),
        (_remove("s01", "view_01", "rgb"), {}, "scene 's01' .* has no rgb folder"),
        (_remove("s01", "view_01", "depth"), {}, "scene 's01' .* has no depth folder"),
        (
            _drop_frames("depth"),
            {},
            r"scene 's01' .* holds different frame counts .*view_02/rgb 4, view_02/depth 3",
        ),
        (
            _drop_frames("rgb", "depth"),
            {},
            r"scene 's01' .* holds different frame counts .*view_01/depth 4, view_02/rgb 3",
        ),
        (
            _remove("s02", "cache", f"{_cache_name()}.npz"),
            {},
            "1 of 2 scenes .* have no cached track file",
        ),
        (_remove("s02", "cache"), {}, "1 of 2 scenes .* have no cached track file"),
        (
            lambda state: None,
            {"val_seq_len": 5},
            r"has 2 of 2 scenes shorter than the held-out window's --val_seq_len 5 "
            r"\(first: 's01', 4 frames\)",
        ),
        (
            lambda state: None,
            {"val_cameras": [0, 1, 2]},
            r"returns cameras \[0, 1, 2, 3\], but the held-out window is --val_cameras \[0, 1, 2\]",
        ),
        (
            lambda state: [shutil.rmtree(state["data_root"] / name) for name in ("s01", "s02")],
            {},
            "holds no scenes",
        ),
    ],
    ids=[
        "no_dataset_dir",
        "view_folder_missing",
        "rgb_folder_missing",
        "depth_folder_missing",
        "rgb_depth_disagree",
        "views_disagree",
        "cache_file_missing",
        "cache_dir_missing",
        "scene_shorter_than_window",
        "other_cameras",
        "empty",
    ],
)
def test_what_opening_dexycb_reveals_is_refused_before_any_load(
    tmp_path, dexycb, change, overrides, message
):
    change(dexycb)
    before = _listing(dexycb["root"])

    with pytest.raises(ValueError, match=message):
        _provider(_args(tmp_path, dexycb, **overrides))

    assert dexycb["loads"] == []
    assert _listing(dexycb["root"]) == before


def test_a_load_would_have_written_the_missing_cache(dexycb):
    """The control for the no-write assertions above: the stub's load, like the
    loader's, makes the cache directory and writes the file it did not find."""

    shutil.rmtree(dexycb["data_root"] / "s02" / "cache")
    dataset = sys.modules[
        "mvtracker.datasets.dexycb_multiview_dataset"
    ].DexYCBMultiViewDataset.from_name(_NAME, str(dexycb["root"]))

    dataset[1]

    written = dexycb["data_root"] / "s02" / "cache" / f"{_cache_name()}.npz"
    assert written.read_bytes() == b"redrawn"


def test_dexycb_opener_names_what_failed_to_import(tmp_path, dexycb, monkeypatch):
    """The DexYCB module imports pandas and matplotlib at module level; when that
    fails the refusal carries the import error, which names the package."""

    broken = types.ModuleType("mvtracker.datasets.dexycb_multiview_dataset")

    def missing(name):
        raise ModuleNotFoundError("No module named 'pandas'", name="pandas")

    broken.__getattr__ = missing
    monkeypatch.setitem(sys.modules, "mvtracker.datasets.dexycb_multiview_dataset", broken)

    with pytest.raises(SceneProviderError, match="No module named 'pandas'") as error:
        _provider(_args(tmp_path, dexycb))
    assert "pandas and matplotlib at module level" in str(error.value)
    assert dexycb["from_name"] == []


def test_dexycb_scenes_longer_than_the_window_are_scored_on_their_first_frames(
    tmp_path, dexycb
):
    """The set mixes 24- and 25-frame scenes: every scene must hold the window,
    and a longer one is windowed on its first frames."""

    shutil.rmtree(dexycb["data_root"])
    _write_tree(dexycb["data_root"], {"s01": 4, "s02": 5})
    dexycb["sample"] = lambda name: _dexycb_datapoint(
        name, time_count=5 if name == "s02" else 4
    )
    args = _args(tmp_path, dexycb)
    provider = _provider(args)
    assert provider.kwargs["scene_frames"] == {"s01": 4, "s02": 5}

    held_out = train_cli._held_out_set(args, object(), provider)
    longer = held_out.scene_provider(held_out.plans[1])
    assert longer.depth.shape[1] == 5 and longer.times == (0, 2)

    with pytest.raises(
        ValueError,
        match=r"has 1 of 2 scenes shorter than .* --val_seq_len 5 \(first: 's01', 4 frames\)",
    ):
        _provider(_args(tmp_path, dexycb, val_seq_len=5))


def test_a_dexycb_name_refusal_on_a_missing_cache_reaches_main_before_the_model(
    tmp_path, dexycb, monkeypatch, capsys
):
    (dexycb["data_root"] / "s01" / "cache" / f"{_cache_name()}.npz").unlink()
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
            str(dexycb["root"]),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    assert "1 of 2 scenes" in capsys.readouterr().err
    assert dexycb["loads"] == []


# ------------------------------------------------------------ the scenes ---


def test_a_dexycb_scene_is_the_sample_as_loaded(tmp_path, dexycb, capsys):
    args = _args(tmp_path, dexycb, depth_input=True, camera_input=True)
    held_out = train_cli._held_out_set(args, object(), _provider(args))
    sample = _dexycb_datapoint("s02")

    scene = held_out.scene_provider(held_out.plans[1])

    assert dexycb["loads"] == [1]
    assert scene.name == "s02"
    assert torch.equal(scene.query_points, sample.query_points_3d)
    assert torch.equal(scene.trajectories_world, sample.trajectory_3d)
    assert torch.equal(scene.visibility, sample.visibility)
    assert torch.equal(scene.depth, sample.videodepth)
    assert scene.track_upscaling_factor == _SIXTH
    assert scene.camera_ids == (0, 1, 2, 3) and scene.view_ids.tolist() == [0, 1, 2, 3]
    assert scene.times == (0, 2) and scene.query_anchors == ((0, 0),)
    assert "depth_map" in scene.views[0] and "camera_vector" in scene.views[0]
    plain_args = _args(tmp_path, dexycb)
    plain = train_cli._held_out_set(plain_args, object(), _provider(plain_args))
    plain_scene = plain.scene_provider(plain.plans[1])
    assert "depth_map" not in plain_scene.views[0]
    assert "camera_vector" not in plain_scene.views[0]

    capsys.readouterr()
    dexycb["gotit"] = False
    with pytest.raises(SceneProviderError, match="failed to load"):
        held_out.scene_provider(held_out.plans[0])
    # A failed load is not completed, and not reported.
    assert _benchmark_lines(capsys.readouterr().out) == []


def test_dexycb_label_depth_is_capped_in_stored_units(tmp_path, dexycb, capsys, monkeypatch):
    def planted(name):
        sample = _dexycb_datapoint(name)
        # Stored units are metres x6: 24 is 4 m, kept (strict >); 24.5 is
        # 4.08 m, which a cap read in metres would keep.
        sample.videodepth[0, 0, 0, 10, 10] = 24.0
        sample.videodepth[0, 0, 0, 10, 11] = 24.5
        sample.videodepth[1, 3, 0, 20, 30] = 30.0
        return sample

    dexycb["sample"] = planted
    args = _args(tmp_path, dexycb, depth_input=True)
    held_out = train_cli._held_out_set(args, object(), _provider(args))

    scene = held_out.scene_provider(held_out.plans[0])

    expected = planted("s01").videodepth.clone()
    expected[0, 0, 0, 10, 11] = 0.0
    expected[1, 3, 0, 20, 30] = 0.0
    assert torch.equal(scene.depth, expected)
    # The encoder's input reads the capped pixel invalid and the kept one at
    # the cap: one number for both depths, as on the held-out path.
    depth_input = scene.views[0]["depth_map"][0]
    assert depth_input[:, 10, 11].tolist() == [0.0, 0.0]
    assert depth_input[:, 10, 10].tolist() == [1.0, 1.0]
    # One line per scene, out of the valid pixels (4 views x 4 frames x 56 x 56).
    assert _benchmark_lines(capsys.readouterr().out) == [
        "benchmark scene=s01 frames=4 label_depth_zeroed=2/50176 (0.0040%) beyond 24"
    ]
    held_out.scene_provider(held_out.plans[0])
    held_out.scene_provider(held_out.plans[1])
    assert _benchmark_lines(capsys.readouterr().out) == [
        "benchmark scene=s02 frames=4 label_depth_zeroed=2/50176 (0.0040%) beyond 24"
    ]
    # Every rank loads every scene at the preflight; only rank 0 says so.
    monkeypatch.setattr(scene_provider_module, "is_main_process", lambda: False)
    other_rank = train_cli._held_out_set(args, object(), _provider(args))
    capped = other_rank.scene_provider(other_rank.plans[0])
    assert torch.equal(capped.depth, expected)
    assert _benchmark_lines(capsys.readouterr().out) == []


@pytest.mark.parametrize(
    "name, cameras",
    [
        ("dex-ycb-multiview-views4567-cached", [4, 5, 6, 7]),
        ("dex-ycb-multiview-views1032-cached", [1, 0, 3, 2]),
    ],
)
def test_a_dexycb_scene_names_the_cameras_it_loaded(tmp_path, dexycb, name, cameras):
    """The video axis is stacked in views_to_return order, so each position's
    camera is that list's entry, not its index."""

    shutil.rmtree(dexycb["data_root"])
    _write_tree(dexycb["data_root"], {"s01": 4}, view_folders=8, views=cameras)
    args = _args(tmp_path, dexycb, eval_benchmark_dataset=name, val_cameras=cameras)
    held_out = train_cli._held_out_set(args, object(), _provider(args))

    scene = held_out.scene_provider(held_out.plans[0])

    assert scene.camera_ids == tuple(cameras)
    assert scene.view_ids.tolist() == cameras
    assert scene.query_anchors == ((cameras[0], 0),)
    # Slot 0 holds position 0 of the video -- the first camera returned.
    assert scene.observations[0].camera == 0
    assert scene.observations[0].camera_id == cameras[0]


# ------------------------------------------------------------ the units ---


def _rescaled(sample, scale):
    """The same world with every stored length times ``scale`` and the factor
    divided by it, so every distance in metres is unchanged."""

    out = copy.deepcopy(sample)
    out.videodepth = sample.videodepth * scale
    out.trajectory_3d = sample.trajectory_3d * scale
    out.query_points_3d = torch.cat(
        [sample.query_points_3d[:, :1], sample.query_points_3d[:, 1:] * scale], dim=1
    )
    out.extrs = sample.extrs.clone()
    out.extrs[..., :3, 3] *= scale
    out.track_upscaling_factor = sample.track_upscaling_factor / scale
    return out


def _anchor_pixel(sample, track, *, view=0, time=0):
    """Where ``track`` projects in one view: the pixel the anchor gate reads."""

    world_to_camera = sample.extrs[view, time].double()
    camera_point = (
        world_to_camera[:, :3] @ sample.trajectory_3d[time, track].double()
        + world_to_camera[:, 3]
    )
    u, v, w = (sample.intrs[view, time].double() @ camera_point).tolist()
    return int(np.rint(v / w)), int(np.rint(u / w))


def _gated_world(name):
    """The DexYCB-shaped world, stored at metres x6, with the depth at two of
    camera 0's anchor pixels moved off the surface: track 0 by 0.05 m (inside
    the 10 cm gate only when the factor is applied once) and track 1 by 0.3 m
    (outside it unless the factor is applied twice)."""

    world = _dexycb_datapoint(name)
    for track, metres in ((0, 0.05), (1, 0.3)):
        row, column = _anchor_pixel(world, track)
        world.videodepth[0, 0, 0, row, column] += metres / _SIXTH
    return world


def _assert_same_metres(metres, stored, path="metrics"):
    """Every figure equal between the two worlds, but the stored-unit gauges,
    which read exactly six times larger in the world stored at metres x6."""

    if isinstance(metres, dict):
        assert list(stored) == list(metres), path
        for key, value in metres.items():
            if key in _GAUGES and value is not None:
                assert stored[key] == pytest.approx(6 * value, rel=1e-6), f"{path}.{key}"
            else:
                _assert_same_metres(value, stored[key], f"{path}.{key}")
    elif isinstance(metres, (list, tuple)):
        assert len(stored) == len(metres), path
        for index, (left, right) in enumerate(zip(metres, stored)):
            _assert_same_metres(left, right, f"{path}[{index}]")
    elif isinstance(metres, float):
        assert type(stored) is float, path
        assert stored == pytest.approx(metres, rel=1e-5, abs=1e-6, nan_ok=True), path
    else:
        assert type(stored) is type(metres) and stored == metres, path


_REJECTED_NOTHING = {
    "query_time_mismatch": 0,
    "not_visible_in_anchor": 0,
    "projection": 0,
    "anchor_depth_gate": 0,
    "pixel_dedup": 0,
}


@pytest.mark.parametrize("source", ["predicted", "query_position", "gt_depth"])
def test_a_dexycb_scene_is_scored_in_metres(tmp_path, monkeypatch, dexycb, source):
    """The pin of the units audit: the world stored at metres x6 with a factor
    of 1/6 is scored exactly as the same world stored in metres at 1. Both
    planted with the same model-unit pointmaps, so the stored world's Sim(3)
    reads a scale of 6 and every figure in metres must agree: a factor
    dropped anywhere reads six times larger, one applied twice six times
    smaller, and at the anchor gate either one moves a planted query across
    10 cm."""

    stored = _gated_world("s01")
    metres = _rescaled(stored, _SIXTH)
    assert metres.track_upscaling_factor == 1.0
    args = _args(tmp_path, dexycb)
    opened = train_cli._held_out_set(args, object(), _provider(args))
    held_out = train_cli.HeldOut(opened.plans[:1], opened.scene_provider, opened.benchmark)
    spec = train_cli._eval_anchor_spec(_args(tmp_path, dexycb, eval_query_anchors=["all"]))
    assert spec == ["1:0", "2:0", "3:0", "0:1", "1:1", "2:1", "3:1"]
    dexycb["sample"] = lambda name: copy.deepcopy(metres)
    _plant_pointmaps(monkeypatch, held_out.scene_provider(held_out.plans[0]))

    runs = {}
    for world_name, world in (("metres", metres), ("stored", stored)):
        dexycb["sample"] = lambda name, world=world: copy.deepcopy(world)
        for anchors_name, anchors in (("seated", None), ("all", spec)):
            directory = tmp_path / f"{world_name}-{anchors_name}"
            metrics, _ = _evaluate(
                directory,
                held_out,
                anchors,
                eval_query_anchor_source=source,
                visual_scenes=frozenset({"s01"}),
                run_identity=dict(_IDENTITY),
            )
            runs[world_name, anchors_name] = (
                metrics,
                _bundle(directory, "s01"),
                read_visual_dump(directory / "eval" / "step-3" / "visual" / "s01.npz"),
            )

    for anchors_name in ("seated", "all"):
        metrics_m, bundle_m, (visual_m, meta_m) = runs["metres", anchors_name]
        metrics_s, bundle_s, (visual_s, meta_s) = runs["stored", anchors_name]
        # Every figure the eval reports, metres and counts alike.
        _assert_same_metres(metrics_m, metrics_s)
        # Not vacuous: real errors, a fitted baseline, the anchor readout.
        entry_m, entry_s = metrics_m["per_scene"][0], metrics_s["per_scene"][0]
        assert metrics_m["metric_error_m"] > 0.1
        assert entry_m["velocity_consistency"] is not None
        assert entry_m["drift"]["pose"]["camera_center_error_m"]["mean"] > 0.1
        assert entry_m["drift"]["pose"]["baseline_scale"] is not None
        assert entry_m["alignment_scale"] == pytest.approx(1.0, rel=1e-6)
        assert entry_s["alignment_scale"] == pytest.approx(6.0, rel=1e-6)
        assert entry_m["anchor_error_m"]["0:0"]["count"] >= 1

        # The bundles, as the scorer reads them at a factor of 1.
        assert list(bundle_s) == list(bundle_m)
        for key in ("pred", "gt", "query_points"):
            np.testing.assert_allclose(bundle_s[key], bundle_m[key], rtol=1e-5, atol=1e-6)
        for key in ("conf", "occ", "gt_vis_any", "tau", "confidence_alpha", "tau_distance_m"):
            assert np.array_equal(bundle_s[key], bundle_m[key]), key
        # The stored world's rows are its stored values times 1/6.
        np.testing.assert_allclose(
            bundle_s["gt"], stored.trajectory_3d[[0, 2]].numpy() * _SIXTH, rtol=1e-6
        )
        np.testing.assert_allclose(
            bundle_s["query_points"][:, 1:],
            stored.query_points_3d[:, 1:].numpy() * _SIXTH,
            rtol=1e-6,
        )

        # The visual dump, in the same metres.
        assert sorted(visual_s) == sorted(visual_m)
        for key, value in visual_m.items():
            if key == "metadata":
                continue
            if value.dtype.kind == "f":
                np.testing.assert_allclose(visual_s[key], value, rtol=1e-5, atol=1e-6, err_msg=key)
            else:
                assert np.array_equal(visual_s[key], value), key
        assert meta_s.pop("output_dir") != meta_m.pop("output_dir")
        _assert_same_metres(meta_m, meta_s, "metadata")
        np.testing.assert_allclose(
            visual_s["gt_camera_center_m"][:, 0],
            visual_s["slot_camera_id"] * _SIXTH,
            rtol=1e-6,
        )
        np.testing.assert_allclose(visual_s["track_gt"], bundle_s["gt"])

    # The gate, at the seated anchor: track 0 sits 5 cm off the surface and is
    # admitted, track 1 sits 30 cm off and is not, track 2 starts at frame 2.
    for world_name in ("metres", "stored"):
        seated = runs[world_name, "seated"][0]["eval_benchmark"]["per_scene"][0]
        assert seated["admitted_by_start_time"] == {"0": 1}
        assert seated["unanchored_by_start_time"] == {"0": 1, "2": 1}
        assert seated["unanchored_rejected"] == {
            **_REJECTED_NOTHING,
            "query_time_mismatch": 1,
            "anchor_depth_gate": 1,
        }
        # Camera 1 sees track 1 on its surface, and frame 2 seats track 2.
        everything = runs[world_name, "all"][0]["eval_benchmark"]["per_scene"][0]
        assert everything["admitted_by_start_time"] == {"0": 2, "2": 1}
        # The two unscored rows, held still at their query point in metres.
        bundle = runs[world_name, "seated"][1]
        world = metres if world_name == "metres" else stored
        for track in (1, 2):
            start = int(world.query_points_3d[track, 0])
            point = world.trajectory_3d[start, track].numpy() * world.track_upscaling_factor
            np.testing.assert_allclose(bundle["pred"][:, track], np.stack([point, point]), rtol=1e-6)
            assert np.array_equal(bundle["conf"][:, track], [0.0, 0.0])
            assert bundle["occ"][:, track].all()
