"""A trained train_state.pt through inference.py and the official monocular
eval, with the weights, freeze, unroll count and head geometry it trained
under. The eval is not an exact replay of training: it attaches no
time_index, so the patch's time-index table is never read, as the parser
test at the end pins.

load_temporal_patch_model is the one loader both scripts share: header,
construction kwargs, set_freeze, strict load, and the two forward keywords no
weight can carry -- refine_iters, and merge_synchronized_slots, which adds no
parameters and lives only in the trainer's stored settings. The helper tests
run it on tiny real-Arc subclasses against patches the trainer's own writer
produced, so a key path the writer and the reader disagree on fails here. The
script tests execute both main()s with their heavy hops stubbed at the modules
main() imports them from: inference.py up to its inference() call, and
eval/track/launch.py through the real eval_tapvid3d and get_inference_output
of both eval modules, down to the model call itself, where a recorder sees
the keywords a dropped hop would silently lose.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import arc.dust3r.inference_multiview as inference_multiview
import arc.dust3r.utils.image as image_utils
import arc.training.checkpoint as patch_checkpoint
import inference as inference_cli
from arc.models.arc.arc import Arc
from arc.training.checkpoint import (
    load_temporal_patch_model,
    load_temporal_tracking_checkpoint,
    read_temporal_patch_metadata,
)
from arc.training.trainer_state import (
    build_trainer_state,
    capture_rng_state,
    restore_rng_state,
)
# Importing the launcher appends the repo root and eval/track to sys.path for
# the rest of the session (launch.py:8-9). Harmless here: the root is already
# on it via pytest.ini, and eval/track is how launch.main() resolves its
# bare-name track_eval_dense / track_eval imports, the modules stubbed below.
from eval.track import launch as track_launch
from test_time_indexing import _GlobalAttnTinyArc, _TinyHubArc, _trainer_patch


# Not the tiny fixtures' 4 rows, so a construction kwarg the helper dropped
# builds a 4-row table and the strict load fails on its shape.
_ROWS = 6
_CPU = torch.device("cpu")
_EVAL_MODULES = {"kubric": "track_eval_dense", "pstudio_mini": "track_eval"}
_PATCH_FORWARD = {"refine_iters": 4, "merge_synchronized_slots": True}


def _with_table(model, rows):
    """Size ``model``'s time table to ``rows``, as Arc's constructor sizes it
    from max_time_indices; the tiny fixtures fix it at 4 and refuse the
    keyword."""

    model.max_time_indices = rows
    model.backbone.pretrained.time_index_embedding = nn.Embedding(rows, 2)
    nn.init.zeros_(model.backbone.pretrained.time_index_embedding.weight)


class _TableTinyArc(_TinyHubArc):
    def __init__(self, freeze="none", max_time_indices=4):
        super().__init__(freeze="none")
        _with_table(self, max_time_indices)
        self.set_freeze(freeze)


class _TableLateGlobalArc(_GlobalAttnTinyArc):
    def __init__(self, freeze="none", late_global_blocks=None, max_time_indices=4):
        super().__init__(freeze="none")
        _with_table(self, max_time_indices)
        self.set_freeze(freeze, late_global_blocks=late_global_blocks)


class _Reached(Exception):
    """Stops a main() at the hop a test records. Not a RuntimeError: the
    eval's OOM handler catches those (torch.cuda.OutOfMemoryError is one)."""


class _RecordingTinyArc(_TableTinyArc):
    """The tiny patched model with its forward replaced by the eval's
    recorder: set_freeze and the strict load run for real, and the forward
    records the keywords and image size the eval handed it, raises the
    queued failures one call at a time, then stops the eval."""

    def __init__(self, freeze="none", max_time_indices=4):
        super().__init__(freeze=freeze, max_time_indices=max_time_indices)
        self.calls = []
        self.failures = []

    def forward(self, views, **kwargs):
        self.calls.append((dict(kwargs), tuple(views[0]["img"].shape[-2:])))
        if self.failures:
            raise self.failures.pop(0)
        raise _Reached


def _fill_distinct(model):
    """A different constant in every trainable tensor, so a tensor loaded
    under the wrong name, or never loaded, fails by value."""

    with torch.no_grad():
        trainable = [p for p in model.parameters() if p.requires_grad]
        for index, parameter in enumerate(trainable):
            parameter.fill_(0.25 * (index + 1))


def _trainable(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _record_late_global_blocks(patch, blocks):
    """Add the one field the trainer's writer never records (a known
    pre-existing bug: no trainer-written late-global payload loads) to an
    otherwise real payload."""

    payload = torch.load(patch, map_location="cpu", weights_only=True)
    payload["late_global_blocks"] = blocks
    torch.save(payload, patch)


def _stub_from_pretrained(monkeypatch, build):
    """Arc.from_pretrained builds ``build(**kwargs)`` and records each call."""

    calls = []

    def from_pretrained(checkpoint_dir, **kwargs):
        calls.append((checkpoint_dir, kwargs))
        return build(**kwargs)

    monkeypatch.setattr(Arc, "from_pretrained", staticmethod(from_pretrained))
    return calls


def _never_built(checkpoint_dir, **kwargs):
    raise AssertionError("a refused patch must not cost the base-model load")


def _merged_k4_patch(directory):
    trained = _TableTinyArc(max_time_indices=_ROWS)
    trained.set_freeze("temporal_tracking", refine=True)
    _fill_distinct(trained)
    return _trainer_patch(trained, directory, **_PATCH_FORWARD)


def _depth_patch(directory):
    trained = _TinyHubArc(freeze="none")
    trained.set_freeze("temporal_tracking", depth_input=True)
    return _trainer_patch(trained, directory)


# ------------------------------------------------------------------------------
# the shared loader
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("refine_iters", [1, 4], ids=["K1", "K4"])
@pytest.mark.parametrize("merge", [False, True], ids=["per-slot", "merged"])
@pytest.mark.parametrize("freeze", ["temporal_tracking", "temporal_tracking_late_global"])
def test_the_helper_runs_a_patch_as_trained(tmp_path, monkeypatch, freeze, merge, refine_iters):
    """Metadata, construction kwargs, set_freeze, strict load, forward
    keywords: the whole pipeline, on a patch the trainer's own writer wrote.
    Every trainable tensor must come back under its own name (the fills are
    distinct), the base model must be built once at the patch's table size,
    and the forward keywords must be the patch's, exact types included -- a
    merge read from anywhere but the stored settings, or a count the refiner
    was frozen against, fails one of the eight cells."""

    late = freeze == "temporal_tracking_late_global"
    build = _TableLateGlobalArc if late else _TableTinyArc
    blocks = {"late_global_blocks": 1} if late else {}
    trained = build(max_time_indices=_ROWS)
    trained.set_freeze(freeze, refine=refine_iters > 1, **blocks)
    _fill_distinct(trained)
    patch = _trainer_patch(
        trained,
        tmp_path / "patch",
        merge_synchronized_slots=merge,
        refine_iters=refine_iters,
    )
    if late:
        _record_late_global_blocks(patch, 1)
    calls = _stub_from_pretrained(monkeypatch, build)

    model, forward_kwargs = load_temporal_patch_model("hub/4rc", patch, device=_CPU)

    assert calls == [("hub/4rc", {"max_time_indices": _ROWS})]
    assert model.freeze == freeze
    assert model.late_global_blocks == (1 if late else None)
    assert model.motion_decoder.refiner.read_proj.weight.requires_grad is (refine_iters > 1)
    expected, loaded = _trainable(trained), _trainable(model)
    assert loaded.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(loaded[name], value), name
    assert forward_kwargs == {"refine_iters": refine_iters, "merge_synchronized_slots": merge}
    assert type(forward_kwargs["refine_iters"]) is int
    assert type(forward_kwargs["merge_synchronized_slots"]) is bool


@pytest.mark.parametrize(
    "depth_input, camera_input",
    [(True, False), (False, True), (True, True)],
    ids=["depth", "camera", "both"],
)
def test_the_helper_refuses_a_geometry_patch_before_building(
    tmp_path, monkeypatch, depth_input, camera_input
):
    """Both callers feed frames with no depth map and no camera vector, and
    the encoder skips a missing injection silently, so such a patch would load
    cleanly and run with its trained injection never used. Refused by name,
    before the base model is paid for."""

    trained = _TinyHubArc(freeze="none")
    trained.set_freeze("temporal_tracking", depth_input=depth_input, camera_input=camera_input)
    patch = _trainer_patch(trained, tmp_path / "patch")
    monkeypatch.setattr(Arc, "from_pretrained", staticmethod(_never_built))

    with pytest.raises(ValueError) as refused:
        load_temporal_patch_model("hub/4rc", patch, device=_CPU)

    message = str(refused.value)
    assert ("--depth_input" in message) is depth_input
    assert ("--camera_input" in message) is camera_input


# The six settings the trainer stored at aba9e22, before the merge flag
# (48ecd0c) and the top-level refine_iters (66f4835) existed.
_ABA9E22_SETTINGS = {
    "observation_budget": 48,
    "stride": 2,
    "num_steps": 8,
    "warmup_steps": 2,
    "min_lr_scale": 0.0,
    "precision": "bf16-mixed",
}


def _historical_patch(tmp_path, model, shape):
    """``model``'s trainable tensors in an older payload shape: the trainer's
    from aba9e22 (build_trainer_state is unchanged since), the same from
    48ecd0c with the merge stored, or the retired one-scene saver's, which
    wrote no settings and no trainer state."""

    state_dict = _trainable(model)
    if shape == "retired-saver":
        payload = {
            "freeze_mode": model.freeze,
            "late_global_blocks": None,
            "state_dict": state_dict,
        }
    else:
        settings = dict(_ABA9E22_SETTINGS)
        if shape == "48ecd0c":
            settings["merge_synchronized_slots"] = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        payload = {
            "freeze_mode": model.freeze,
            "state_dict": state_dict,
            **build_trainer_state(
                step=0, optimizer=optimizer, base_learning_rates=[1e-3], settings=settings
            ),
        }
    path = tmp_path / f"{shape}.pt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize(
    "shape, merged",
    [("aba9e22", False), ("48ecd0c", True), ("retired-saver", False)],
)
def test_every_payload_shape_since_aba9e22_reads_its_head_geometry(
    tmp_path, monkeypatch, shape, merged
):
    """A payload without the key -- or without settings at all -- predates
    the flag and trained the per-slot head, the resume path's own absence
    rule; one that stores it is read from the settings. None of these carries
    a top-level refine_iters, so each also reads one iteration, and the helper
    loads every one."""

    trained = _TableTinyArc(freeze="temporal_tracking", max_time_indices=_ROWS)
    _fill_distinct(trained)
    patch = _historical_patch(tmp_path, trained, shape)
    _stub_from_pretrained(monkeypatch, _TableTinyArc)

    metadata = read_temporal_patch_metadata(patch)
    model, forward_kwargs = load_temporal_patch_model("hub/4rc", patch, device=_CPU)

    assert metadata["merge_synchronized_slots"] is merged
    assert metadata["refine_iters"] == 1
    assert forward_kwargs == {"refine_iters": 1, "merge_synchronized_slots": merged}
    for name, value in _trainable(trained).items():
        assert torch.equal(dict(model.named_parameters())[name], value), name


def _settings_patch(tmp_path, settings):
    trained = _TinyHubArc(freeze="temporal_tracking")
    path = tmp_path / "patch.pt"
    torch.save(
        {"freeze_mode": "temporal_tracking", "state_dict": _trainable(trained), "settings": settings},
        path,
    )
    return path


@pytest.mark.parametrize(
    "settings, match",
    [
        ([], "settings must be a dict or absent"),
        ({"merge_synchronized_slots": 1}, "must be a bool or absent"),
        ({"merge_synchronized_slots": "true"}, "must be a bool or absent"),
        ({"merge_synchronized_slots": None}, "must be a bool or absent"),
    ],
    ids=["settings-list", "merge-int", "merge-str", "merge-none"],
)
def test_a_malformed_merge_setting_is_refused_by_both_readers(tmp_path, settings, match):
    """The trainer only ever stores a bool, so anything else is a hand-edited
    or foreign payload, and coercing it would guess -- bool("false") is True.
    A stored None is not an absence either: the key is present, so resume
    compares it, and None matches neither setting. The loader parses the same
    payload, so the trainer's --resume refuses it too."""

    patch = _settings_patch(tmp_path, settings)

    with pytest.raises(RuntimeError, match=match):
        read_temporal_patch_metadata(patch)
    with pytest.raises(RuntimeError, match=match):
        load_temporal_tracking_checkpoint(_TinyHubArc(freeze="temporal_tracking"), patch)


def test_a_stored_settings_of_none_reads_as_absent(tmp_path):
    """Resume reads ``payload.get("settings") or {}``, so a stored None is no
    settings at all to it, and to the patch reader."""

    patch = _settings_patch(tmp_path, None)

    assert read_temporal_patch_metadata(patch)["merge_synchronized_slots"] is False


# ------------------------------------------------------------------------------
# inference.py
# ------------------------------------------------------------------------------


class _Loaded(nn.Module):
    """What a stubbed load hands main(): a module with a time table size."""

    def __init__(self, max_time_indices):
        super().__init__()
        self.max_time_indices = max_time_indices


# The stubbed patch's table: larger than Arc.MAX_TIME_INDICES, so an index
# between the two tells a check against the patch's table from one against
# the released default.
_PATCH_ROWS = 40


@pytest.fixture
def inference_main(monkeypatch, tmp_path):
    """Run inference.main() up to its inference() call, every heavy hop
    stubbed where main() imports it from, and return what reached each hop."""

    seen = {"from_pretrained": [], "helper": [], "inference": []}

    def from_pretrained(checkpoint_dir, **kwargs):
        seen["from_pretrained"].append((checkpoint_dir, kwargs))
        return _Loaded(Arc.MAX_TIME_INDICES)

    def helper(checkpoint_dir, path, *, device):
        seen["helper"].append((checkpoint_dir, path, device))
        return _Loaded(_PATCH_ROWS), dict(_PATCH_FORWARD)

    def run_inference(imgs, model, device, **kwargs):
        seen["inference"].append(kwargs)
        raise _Reached

    monkeypatch.setattr(
        inference_cli, "collect_images", lambda path: (["a.png", "b.png"], False)
    )
    monkeypatch.setattr(
        image_utils,
        "load_images",
        lambda paths, **kwargs: [{"img": torch.zeros(1, 3, 4, 4)} for _ in paths],
    )
    monkeypatch.setattr(inference_multiview, "inference", run_inference)
    monkeypatch.setattr(patch_checkpoint, "load_temporal_patch_model", helper)
    monkeypatch.setattr(Arc, "from_pretrained", staticmethod(from_pretrained))

    def run(*argv):
        monkeypatch.setattr(
            sys,
            "argv",
            ["inference.py", "--input", "frames", "--save", str(tmp_path / "out.npz"), *argv],
        )
        inference_cli.main()

    return run, seen


_INFERENCE_CALL = {
    "dtype": "bf16-mixed",
    "verbose": True,
    "profiling": True,
    "use_center_as_anchor": False,
}


def test_inference_runs_a_patch_with_the_forward_keywords_it_trained_under(inference_main):
    run, seen = inference_main

    with pytest.raises(_Reached):
        run("--temporal_patch", "patch.pt")

    assert seen["helper"] == [("Luo-Yihang/4RC", "patch.pt", _CPU)]
    assert seen["from_pretrained"] == []
    assert seen["inference"] == [{**_INFERENCE_CALL, **_PATCH_FORWARD}]


def test_inference_without_a_patch_runs_the_released_weights_as_today(inference_main):
    """No forward keyword at all: inference()'s defaults, which are
    Arc.forward's own, reach the model -- today's call."""

    run, seen = inference_main

    with pytest.raises(_Reached):
        run()

    assert seen["helper"] == []
    assert seen["from_pretrained"] == [("Luo-Yihang/4RC", {})]
    assert seen["inference"] == [_INFERENCE_CALL]


def test_without_a_patch_a_bad_time_index_fails_before_the_base_load(inference_main):
    run, seen = inference_main

    with pytest.raises(SystemExit):
        run("--time_indices", "0", str(Arc.MAX_TIME_INDICES))

    assert seen["from_pretrained"] == []


def test_with_a_patch_the_time_indices_are_bounded_by_the_patch_table_after_the_load(
    inference_main,
):
    """No pre-load check for a patched run: the table size lives in
    train_state.pt, and a third read of it (it carries the optimizer state,
    several GB) is not worth a fail-fast on an error path. The post-load
    check binds instead, against the patch's own table: an index past the
    released default but inside the patch's runs, and one past the patch's
    table is refused once the helper has run."""

    run, seen = inference_main

    with pytest.raises(_Reached):
        run("--temporal_patch", "patch.pt", "--time_indices", "0", str(_PATCH_ROWS - 1))
    assert len(seen["inference"]) == 1

    with pytest.raises(SystemExit):
        run("--temporal_patch", "patch.pt", "--time_indices", "0", str(_PATCH_ROWS))
    assert len(seen["helper"]) == 2
    assert len(seen["inference"]) == 1


def test_inference_refuses_a_geometry_patch(inference_main, monkeypatch, tmp_path):
    run, seen = inference_main
    monkeypatch.setattr(patch_checkpoint, "load_temporal_patch_model", load_temporal_patch_model)

    with pytest.raises(ValueError, match="--depth_input"):
        run("--temporal_patch", str(_depth_patch(tmp_path / "patch")))

    assert seen["from_pretrained"] == []


# ------------------------------------------------------------------------------
# eval/track/launch.py
# ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_rng():
    """launch.main() seeds python, numpy and torch (launch.py:37-39); hand
    the suite its generators back."""

    state = capture_rng_state()
    yield
    restore_rng_state(state)


def _npz_data(track_npz, num_frames=None):
    """load_npz_data's eight fields for one clip: two 40x30 frames and one
    visible point, enough for both eval modules to reach the model call."""

    frames = [Image.new("RGB", (40, 30)) for _ in range(2)]
    return (
        frames,
        np.zeros((2, 1, 3), dtype=np.float32),
        np.full((2, 1, 2), 5.0, dtype=np.float32),
        np.array([20.0, 20.0, 20.0, 15.0], dtype=np.float32),
        np.zeros((2, 1, 3), dtype=np.float32),
        np.ones((2, 1), dtype=bool),
        "clip",
        np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    )


@pytest.fixture
def launch_eval(monkeypatch, tmp_path):
    """Run launch.main() on one stubbed clip through the real eval_tapvid3d
    and get_inference_output, and return every base-model build: its
    checkpoint, its kwargs, and the recording model the eval called."""

    built = []
    failures = []

    def from_pretrained(checkpoint_dir, **kwargs):
        model = _RecordingTinyArc(**kwargs)
        model.failures = list(failures)
        built.append((checkpoint_dir, kwargs, model))
        return model

    monkeypatch.setattr(Arc, "from_pretrained", staticmethod(from_pretrained))

    def run(data_type, *argv, fail_first=None):
        failures[:] = [] if fail_first is None else [fail_first]
        module = importlib.import_module(_EVAL_MODULES[data_type])
        monkeypatch.setattr(module, "load_npz_data", _npz_data)
        clips = tmp_path / "data" / data_type
        clips.mkdir(parents=True, exist_ok=True)
        (clips / "clip.npz").touch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "launch.py",
                "--output_dir", str(tmp_path / "out"),
                "--data_root", str(tmp_path / "data"),
                "--data_type", data_type,
                *argv,
            ],
        )
        track_launch.main()

    return run, built


@pytest.mark.parametrize("data_type", ["kubric", "pstudio_mini"], ids=["dense", "sparse"])
def test_launch_runs_the_released_weights_exactly_as_today(launch_eval, data_type):
    run, built = launch_eval

    with pytest.raises(_Reached):
        run(data_type)

    [(checkpoint_dir, kwargs, model)] = built
    assert (checkpoint_dir, kwargs) == ("Luo-Yihang/4RC", {})
    assert model.freeze == "none"
    assert [call for call, _ in model.calls] == [{"inference_track": True}]


@pytest.mark.parametrize("data_type", ["kubric", "pstudio_mini"], ids=["dense", "sparse"])
def test_launch_runs_a_patch_with_the_forward_keywords_it_trained_under(
    launch_eval, data_type, tmp_path
):
    """The real helper on a real-writer merged K=4 patch, with
    --checkpoint_dir left at its empty default: the base model must come from
    the resolved Luo-Yihang/4RC, at the patch's table size, frozen and loaded,
    and the eval's own model call must carry the patch's keywords."""

    run, built = launch_eval

    with pytest.raises(_Reached):
        run(data_type, "--temporal_patch", str(_merged_k4_patch(tmp_path / "patch")))

    [(checkpoint_dir, kwargs, model)] = built
    assert (checkpoint_dir, kwargs) == ("Luo-Yihang/4RC", {"max_time_indices": _ROWS})
    assert model.freeze == "temporal_tracking"
    assert [call for call, _ in model.calls] == [{"inference_track": True, **_PATCH_FORWARD}]


@pytest.mark.parametrize(
    "failure",
    [lambda: RuntimeError("CUDA out of memory"), torch.cuda.OutOfMemoryError],
    ids=["message", "oom-type"],
)
@pytest.mark.parametrize("data_type", ["kubric", "pstudio_mini"], ids=["dense", "sparse"])
def test_the_oom_retry_runs_the_patch_with_the_same_keywords(
    launch_eval, data_type, failure, tmp_path
):
    """Both arms of the eval's OOM handler -- an error that says so, and the
    OutOfMemoryError type with no message -- retry at size 336, and the retry
    is a second model call that must carry the same keywords."""

    run, built = launch_eval

    with pytest.raises(_Reached):
        run(
            data_type,
            "--temporal_patch",
            str(_merged_k4_patch(tmp_path / "patch")),
            fail_first=failure(),
        )

    [(_, _, model)] = built
    patched = {"inference_track": True, **_PATCH_FORWARD}
    assert model.calls == [(patched, (378, 504)), (patched, (252, 336))]


def test_launch_refuses_a_geometry_patch_before_building(launch_eval, tmp_path):
    run, built = launch_eval

    with pytest.raises(ValueError, match="--depth_input"):
        run("kubric", "--temporal_patch", str(_depth_patch(tmp_path / "patch")))

    assert built == []


def test_the_launch_parser_takes_a_patch_and_says_the_time_table_is_not_read():
    """The help the gate's reader sees says the eval attaches no time_index,
    so the patch's time-index table is never read and the gate is not an
    exact replay of training."""

    parser = track_launch.get_args_parser()

    assert parser.parse_args(["--output_dir", "out"]).temporal_patch is None
    patched = parser.parse_args(["--output_dir", "out", "--temporal_patch", "p.pt"])
    assert patched.temporal_patch == "p.pt"
    [action] = [action for action in parser._actions if action.dest == "temporal_patch"]
    assert "attaches no time_index" in action.help
    assert "never read" in action.help
