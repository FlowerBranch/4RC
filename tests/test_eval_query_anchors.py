"""--eval_query_anchors: eval-only anchors behind the training spec's seated ones.

An anchor admits only the queries that start at its own time, so a held-out
eval anchored at time slot 0 rejects every later-starting query as
``query_time_mismatch``. The extra anchors reach those queries without moving
anything the seated anchors produce: a priority tier keeps every seated row
(``build_anchor_correspondences(priority_anchor_count=...)``), an extra anchor
that receives no query runs no head pass, and the kept ones run a forward of
their own, so forward A -- today's forward -- is untouched.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

import train_temporal_tracking as train_cli
from arc.training import (
    ELIGIBILITY_ASSIGNMENT_RULE,
    ELIGIBILITY_PRIORITY_RULE,
    VISUAL_TRACK_KEYS,
    build_anchor_correspondences,
    capture_rng_state,
    plan_record,
    read_visual_dump,
    with_query_anchors,
)
from scene_fixtures import fixture_scene
from test_manifest_plan import _record, _write_manifest
from test_sparse_tracking import _assert_same_rng_state
from test_trainer_loop import (
    _EVAL_ALPHA,
    _FOUR_ANCHOR_SPEC,
    _FOUR_CAMERA_WINDOW,
    _FakeArc,
    _GroundTruthPoseArc,
    _cpu_eval_scene,
    _loop_args,
    _plan_only,
    _plans,
    _validator_args,
)
from test_track_refinement import _RefiningFakeArc

_NEW_KEYS = ("eval_query_anchors", "eval_benchmark")


class _SlotRecordingArc(_RefiningFakeArc):
    """The refinement fake, recording the anchor slots each forward is handed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_slots = []
        self.conversions = []

    def forward(self, views, **kwargs):
        self.forward_slots.append(torch.as_tensor(views[0]["track_query_idx"]).tolist())
        # The real Arc converts its output to per-view dicts unless told not to,
        # and every consumer here reads the raw dict.
        self.conversions.append(kwargs.get("force_no_output_conversion"))
        return super().forward(views, **kwargs)


def _window_plan(scene, name="0000"):
    """A held-out plan on the scene's own window: the extra slots resolve against it."""

    return dataclasses.replace(
        plan_record(_record(seq_name=name), budget=48, stride=2),
        cameras=scene.camera_ids,
        times=scene.times,
    )


def _all_spec(tmp_path, scene, query_anchors=("0:0",)):
    """What --eval_query_anchors all resolves to on the scene's window, by the
    trainer's own resolver: every camera at stride 1 over the scene's times."""

    args = _loop_args(
        tmp_path,
        val_scenes_file="val.json",
        val_cameras=list(scene.camera_ids),
        val_seq_len=len(scene.times),
        stride=1,
        observation_budget=len(scene.camera_ids) * len(scene.times),
        query_anchors=list(query_anchors),
        eval_query_anchors=["all"],
    )
    return train_cli._eval_anchor_spec(args)


def _evaluate(
    directory,
    scene,
    eval_query_anchors,
    *,
    merge=False,
    refine_iters=1,
    source="predicted",
    model=None,
    query_anchors=("0:0",),
    **kwargs,
):
    """One eval of ``scene`` on its own window; seeded, so two calls score the
    same weights. ``kwargs`` go to evaluate_held_out."""

    if model is None:
        height, width = scene.views[0]["img"].shape[-2:]
        torch.manual_seed(0)
        model = _SlotRecordingArc(
            scene.num_observations,
            height,
            width,
            views_per_time=len(scene.cameras) if merge else 1,
        )
    metrics = train_cli.evaluate_held_out(
        model=model,
        plans=[_window_plan(scene)],
        scene_provider=lambda _plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=5,
        output_dir=directory,
        query_anchors=list(query_anchors),
        confidence_alpha=_EVAL_ALPHA,
        query_anchor_source="predicted",
        merge_synchronized_slots=merge,
        eval_query_anchor_source=source,
        refine_iters=refine_iters,
        eval_query_anchors=eval_query_anchors,
        **kwargs,
    )
    return metrics, model


def _bundle(directory, name="0000"):
    loaded = np.load(Path(directory) / "eval" / "step-5" / "pred" / f"{name}.npz")
    return {key: loaded[key] for key in loaded.files}


def _without_new_keys(metrics):
    kept = {key: value for key, value in metrics.items() if key not in _NEW_KEYS}
    kept["per_scene"] = [
        {key: value for key, value in entry.items() if key != "eval_anchors"}
        for entry in metrics["per_scene"]
    ]
    return kept


def _slot_of(scene, key):
    camera, time = (int(part) for part in key.split(":"))
    return with_query_anchors(scene, ((camera, time),)).query_observation_slot


# ------------------------------------------------------- the priority tier ---


def _camera_one_fits_better(monkeypatch):
    """Every candidate camera 1 offers fits 1 cm better than it is, so without a
    tier an anchor in camera 1 takes every query it shares with camera 0."""

    import arc.training.sparse_tracking as sparse_module

    original = sparse_module._anchor_candidate

    def candidate(point, *, world_to_camera, **kwargs):
        stage, found = original(point, world_to_camera=world_to_camera, **kwargs)
        # Camera 1 sits at x = 1 on the fixture's pure-x baseline.
        if found is not None and world_to_camera[0, 3] == -1.0:
            row, column, depth_error_m, subpixel = found
            found = (row, column, depth_error_m - 0.01, subpixel)
        return stage, found

    monkeypatch.setattr(sparse_module, "_anchor_candidate", candidate)


def test_the_tier_keeps_every_row_a_call_on_the_tier_alone_makes(monkeypatch):
    """Rows, per-anchor counts and contested margins of the tier are exactly a
    tier-only call's, and only the queries the tier leaves go behind it."""

    _camera_one_fits_better(monkeypatch)
    seated = fixture_scene(
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
        query_times=(0, 0, 2),
        query_anchors=((0, 0),),
    )
    extended = with_query_anchors(seated, ((0, 0), (1, 0), (0, 2), (1, 2)))

    alone, alone_report = build_anchor_correspondences(seated)
    tiered, tiered_report = build_anchor_correspondences(
        extended, priority_anchor_count=1
    )
    pooled, _ = build_anchor_correspondences(extended)

    # Not vacuous: one pool hands camera 1 the time-0 queries.
    assert sorted(pooled.query_slots.tolist()) == [1, 1, 3]
    # The tier's rows are the tier-alone call's, a prefix of the tiered set.
    count = alone.count
    assert count == 2
    assert tiered.query_slots[:count].tolist() == [0, 0]
    assert (tiered.query_slots[count:] >= 1).all()
    for name in ("trajectory_indices", "query_slots", "query_times", "rows", "columns"):
        assert torch.equal(getattr(tiered, name)[:count], getattr(alone, name)), name
    assert tiered_report["per_anchor"][0] == alone_report["per_anchor"][0]
    # 1:0 passes its gates for both time-0 queries and wins neither.
    assert tiered_report["per_anchor"][1]["eligible"] == 2
    assert tiered_report["per_anchor"][1]["assigned"] == 0
    # Behind the tier: only the query it left, by the best fit there, which
    # camera 1 still offers.
    assert tiered.trajectory_indices[count:].tolist() == [2]
    assert tiered.query_slots[count:].tolist() == [3]
    rolled = tiered_report["eligible_query_count"] + sum(tiered_report["rejected"].values())
    assert rolled == tiered_report["total_query_count"] == 3


def test_the_tier_is_validated_and_named_in_the_rule():
    scene = fixture_scene(
        cameras=(0, 1), times=(0, 1, 2, 3), size=56, query_anchors=((0, 0), (1, 0))
    )
    for bad in (0, 3, -1, True, 1.0):
        with pytest.raises(ValueError, match="priority_anchor_count"):
            build_anchor_correspondences(scene, priority_anchor_count=bad)

    plain, plain_report = build_anchor_correspondences(scene)
    assert plain_report["assignment_rule"] == ELIGIBILITY_ASSIGNMENT_RULE
    _, tiered_report = build_anchor_correspondences(scene, priority_anchor_count=1)
    assert tiered_report["assignment_rule"] == (
        ELIGIBILITY_ASSIGNMENT_RULE + " " + ELIGIBILITY_PRIORITY_RULE.format(count=1)
    )
    # A tier holding every anchor is one pool under another name.
    whole, _ = build_anchor_correspondences(scene, priority_anchor_count=2)
    for name in ("trajectory_indices", "query_slots", "query_times", "rows", "columns"):
        assert torch.equal(getattr(whole, name), getattr(plain, name)), name


# ------------------------------------------------- re-anchoring a window ---


def test_with_query_anchors_matches_a_window_built_with_those_anchors():
    anchors = ((11, 2), (10, 0), (11, 0))
    window = dict(cameras=(10, 11), times=(0, 1, 2, 3), size=56, view_ids=(10, 11))
    built = fixture_scene(**window, query_anchors=anchors)
    source = fixture_scene(**window)
    before = [view["track_query_idx"].clone() for view in source.views]

    moved = with_query_anchors(source, anchors)

    assert moved.query_anchors == built.query_anchors == anchors
    assert moved.query_observation_slot == built.query_observation_slot == 6
    assert torch.equal(
        moved.track_query_observation_slots, built.track_query_observation_slots
    )
    for view, built_view in zip(moved.views, built.views):
        assert torch.equal(view["track_query_idx"], built_view["track_query_idx"])
    # One tensor per view, as build_scene's clone gives.
    assert len({id(view["track_query_idx"]) for view in moved.views}) == len(moved.views)
    # The source is untouched, and shares its pixels rather than copying them.
    assert source.query_anchors == ((10, 0),)
    assert all(
        torch.equal(view["track_query_idx"], old)
        for view, old in zip(source.views, before)
    )
    assert all(new is not old for new, old in zip(moved.views, source.views))
    assert all(new["img"] is old["img"] for new, old in zip(moved.views, source.views))

    for bad, message in (
        (((12, 0),), "camera 12 is not in selected cameras"),
        (((10, 4),), "time 4 is not in selected times"),
        (((10, 0), (10, 0)), "must not contain duplicates"),
        ((), "at least one"),
    ):
        with pytest.raises(ValueError, match=message):
            with_query_anchors(source, bad)


# ------------------------------------------------------------- the flag ---


def _held_out_args(tmp_path, **overrides):
    return _validator_args(
        tmp_path, **{"val_scenes_file": "val.json", "val_data_root": "/held", **overrides}
    )


def test_all_resolves_to_every_unseated_window_pair_time_major(tmp_path):
    args = _held_out_args(tmp_path, eval_query_anchors=["all"])
    train_cli._validate_args(args)

    expected = tuple(
        (view_slot, time_slot)
        for time_slot in range(12)
        for view_slot in range(4)
        if (view_slot, time_slot) != (0, 0)
    )
    assert train_cli._eval_anchor_slots(args) == expected
    assert train_cli._eval_anchor_spec(args)[:5] == ["1:0", "2:0", "3:0", "0:1", "1:1"]
    assert len(train_cli._eval_anchor_spec(args)) == 47


def test_all_sits_behind_the_slots_the_window_seats_under_a_ceiling(tmp_path):
    """Adaptive seats 0:0..3:0 of a six-camera spec on four cameras; the rest of
    the window follows them, and the unseated 4:0 and 5:0 never appear."""

    args = _held_out_args(
        tmp_path,
        query_anchors=["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"],
        adaptive_query_anchors=True,
        eval_query_anchors=["all"],
    )
    train_cli._validate_args(args)

    slots = train_cli._eval_anchor_slots(args)
    assert slots[:4] == ((0, 1), (1, 1), (2, 1), (3, 1))
    assert len(slots) == 44
    assert all(time_slot > 0 for _, time_slot in slots)


def test_explicit_slots_are_extras_in_the_order_given_and_canonical(tmp_path):
    args = _held_out_args(tmp_path, eval_query_anchors=["01:2", "0:11", "3:0"])
    train_cli._validate_args(args)
    train_cli._validate_args(args)  # idempotent: the flag is resolved, never rewritten

    assert train_cli._eval_anchor_slots(args) == ((1, 2), (0, 11), (3, 0))
    assert train_cli._eval_anchor_spec(args) == ["1:2", "0:11", "3:0"]
    assert train_cli._eval_anchor_spec(_held_out_args(tmp_path)) is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"val_scenes_file": None, "val_data_root": None, "eval_query_anchors": ["all"]},
            r"--eval_query_anchors needs a held-out set \(--val_scenes_file or "
            r"--eval_benchmark_dataset\)",
        ),
        ({"eval_query_anchors": ["all", "0:1"]}, "or 'all' alone"),
        ({"eval_query_anchors": ["0"]}, "^--eval_query_anchors entries must look like"),
        ({"eval_query_anchors": ["0:1", "00:1"]}, "--eval_query_anchors 0:1 is listed more than once"),
        ({"eval_query_anchors": ["4:1"]}, "view slot 4 does not fit the held-out window's 4 cameras"),
        ({"eval_query_anchors": ["0:12"]}, "time slot 12 does not fit the held-out window's 12 times"),
        ({"eval_query_anchors": ["1:3", "0:0"]}, "0:0 is already a seated training anchor"),
    ],
    ids=["no_held_out_set", "all_with_slots", "malformed", "duplicate", "view", "time", "seated"],
)
def test_the_eval_anchor_flag_is_refused_at_parse_time(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        train_cli._validate_args(_held_out_args(tmp_path, **overrides))


def test_the_slots_are_refused_before_the_model_loads(tmp_path, monkeypatch, capsys):
    """Through the real parser and main(), which exits before anything loads."""

    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(step=0)])
    _plan_only(
        monkeypatch,
        manifest,
        "--val_scenes_file",
        "val.json",
        "--val_data_root",
        "/held",
        "--eval_query_anchors",
        "0:12",
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    assert "--eval_query_anchors time slot 12 does not fit" in capsys.readouterr().err


def test_the_flag_is_stored_nowhere_a_resume_reads(tmp_path):
    args = _held_out_args(
        tmp_path,
        eval_every=500,
        eval_query_anchors=["all"],
        eval_benchmark_dataset="kubric-multiview-v3-views0123-cached",
        eval_benchmark_root="/bench",
    )
    train_cli._validate_args(args)

    settings = train_cli._checkpoint_settings(args)
    tiers = train_cli._RESUME_SETTINGS_REFUSED + train_cli._RESUME_SETTINGS_WARNED
    for name in ("eval_query_anchors", "eval_benchmark_dataset", "eval_benchmark_root"):
        assert name not in settings
        assert name not in tiers
    plain = _held_out_args(tmp_path)
    train_cli._validate_args(plain)
    assert settings == train_cli._checkpoint_settings(plain)


def test_the_plan_summary_is_byte_identical_with_the_flag(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(step=0)])
    held_out = ["--val_scenes_file", "val.json", "--val_data_root", "/held"]

    written = {}
    for name, extra in (("off", []), ("on", ["--eval_query_anchors", "all"])):
        out = tmp_path / f"{name}.json"
        _plan_only(monkeypatch, manifest, *held_out, *extra, "--json_out", str(out))
        train_cli.main()
        written[name] = out.read_bytes()

    assert written["on"] == written["off"]


# ------------------------------------------------------------- the eval ---


@pytest.mark.parametrize("source", ["predicted", "query_position", "gt_depth"])
@pytest.mark.parametrize("refine_iters", [1, 2])
@pytest.mark.parametrize("merge", [False, True], ids=["per_slot", "merged"])
def test_later_starting_queries_are_scored_at_their_own_time(
    tmp_path, monkeypatch, merge, refine_iters, source
):
    """Off, the two tracks starting at time 2 are refused at the time-0 anchor;
    on, they are admitted at their own time, and every row and figure of the
    time-0 track is the flag-off one, bit for bit."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))
    spec = _all_spec(tmp_path, scene)
    assert spec == ["1:0", "0:1", "1:1", "0:2", "1:2", "0:3", "1:3"]

    off, off_model = _evaluate(
        tmp_path / "off", scene, None, merge=merge, refine_iters=refine_iters, source=source
    )
    on, on_model = _evaluate(
        tmp_path / "on", scene, spec, merge=merge, refine_iters=refine_iters, source=source
    )
    off_bundle, on_bundle = _bundle(tmp_path / "off"), _bundle(tmp_path / "on")

    # Off: the seated anchor refuses the later two at its own time.
    _, eligibility = build_anchor_correspondences(scene)
    assert eligibility["rejected"]["query_time_mismatch"] == 2
    assert off_bundle["query_points"][:, 0].tolist() == [0.0]
    # On: admitted at time 2, whose covered index is 2, at their own points.
    assert on_bundle["query_points"][:, 0].tolist() == [0.0, 2.0, 2.0]
    assert np.array_equal(
        on_bundle["query_points"][1:, 1:], scene.query_points[1:, 1:].numpy()
    )
    # The time-0 row, key by key, is the flag-off row; the new rows follow it.
    assert set(on_bundle) == set(off_bundle)
    for key, value in off_bundle.items():
        if key == "query_points":
            assert np.array_equal(on_bundle[key][:1], value), key
        elif value.ndim == 2 or value.ndim == 3:
            assert np.array_equal(on_bundle[key][:, :1], value, equal_nan=True), key
        else:
            assert np.array_equal(on_bundle[key], value, equal_nan=True), key
    # Every figure the flag-off eval writes is unchanged, order included.
    assert json.dumps(_without_new_keys(on)) == json.dumps(_without_new_keys(off))
    assert off["eval_query_anchors"] is None and off["eval_benchmark"] is None
    assert on["eval_query_anchors"] == ["0:0", *spec]
    assert on["eval_benchmark"] is None
    assert "eval_anchors" not in off["per_scene"][0]
    assert list(on["per_scene"][0])[-1] == "eval_anchors"

    # Which extra anchors ran: only 0:2 wins a query (1:2 ties it and loses on
    # anchor order; 1:0 is eligible for the time-0 track the tier holds).
    record = on["per_scene"][0]["eval_anchors"]
    assert list(record["per_anchor"]) == spec
    assert record["per_anchor"]["0:2"] == {"eligible": 2, "assigned": 2, "head_pass": True}
    assert record["per_anchor"]["1:2"] == {"eligible": 2, "assigned": 0, "head_pass": False}
    assert record["per_anchor"]["1:0"] == {"eligible": 1, "assigned": 0, "head_pass": False}
    for key in ("0:1", "1:1", "0:3", "1:3"):
        assert record["per_anchor"][key] == {"eligible": 0, "assigned": 0, "head_pass": False}
    assert record["eligibility"] == {
        "total_query_count": 3,
        "eligible_query_count": 3,
        "rejected": {stage: 0 for stage in eligibility["rejected"]},
    }
    assert list(record["anchor_error_m"]) == list(record["base_ratio"]) == ["0:2"]
    assert record["anchor_error_m"]["0:2"]["count"] == 2

    # Forward A and the shuffled arm see the seated slot alone, exactly as
    # without the flag; forward B the one kept extra anchor, and no dropped one.
    seated = [scene.query_observation_slot]
    assert off_model.forward_slots == [seated, seated]
    assert on_model.forward_slots == [seated, seated, [_slot_of(scene, "0:2")]]
    assert on_model.conversions == [True, True, True]


def test_an_explicit_extra_slot_is_scored_alone(tmp_path, monkeypatch):
    """One explicit extra slot, the first anchor behind the seated one: its rows
    follow the time-0 row, forward B runs it alone, and nothing else is seated."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))

    off, _ = _evaluate(tmp_path / "off", scene, None)
    on, model = _evaluate(tmp_path / "on", scene, ["1:2"])

    record = on["per_scene"][0]["eval_anchors"]
    assert record["per_anchor"] == {"1:2": {"eligible": 2, "assigned": 2, "head_pass": True}}
    assert on["eval_query_anchors"] == ["0:0", "1:2"]
    assert model.forward_slots[2] == [_slot_of(scene, "1:2")]
    bundle = _bundle(tmp_path / "on")
    assert bundle["query_points"][:, 0].tolist() == [0.0, 2.0, 2.0]
    assert json.dumps(_without_new_keys(on)) == json.dumps(_without_new_keys(off))


def test_an_extra_anchor_never_takes_a_query_a_seated_one_holds(tmp_path, monkeypatch):
    """Camera 1 fits every time-0 track better than camera 0 does, and one pool
    would move all three to 1:0. The seated 0:0 holds them: 1:0 wins nothing,
    runs no head pass, and the bundle is the flag-off one byte for byte."""

    _camera_one_fits_better(monkeypatch)
    scene = _cpu_eval_scene(monkeypatch)

    off, _ = _evaluate(tmp_path / "off", scene, None)
    on, model = _evaluate(tmp_path / "on", scene, ["1:0"])

    assert on["per_scene"][0]["eval_anchors"]["per_anchor"] == {
        "1:0": {"eligible": 3, "assigned": 0, "head_pass": False}
    }
    assert len(model.forward_slots) == 2
    pred = Path("eval") / "step-5" / "pred" / "0000.npz"
    assert (tmp_path / "on" / pred).read_bytes() == (tmp_path / "off" / pred).read_bytes()
    assert json.dumps(_without_new_keys(on)) == json.dumps(_without_new_keys(off))


@pytest.mark.parametrize("source", ["query_position", "gt_depth"])
def test_the_extra_anchors_measure_the_models_own_anchor_whatever_the_source(
    tmp_path, monkeypatch, source
):
    """As the seated anchors' readout does: under a ground-truth source the
    scored anchor is ground truth, so the model's own is gathered again. The
    planted gauge is scaled, so a world-frame anchor read as the model's would
    be carried through the Sim(3) a second time and miss by metres."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled", query_times=(0, 2, 2))
    spec = _all_spec(tmp_path, scene)

    predicted, _ = _evaluate(tmp_path / "predicted", scene, spec)
    other, _ = _evaluate(tmp_path / source, scene, spec, source=source)

    record = predicted["per_scene"][0]["eval_anchors"]["anchor_error_m"]
    assert 0.0 < record["0:2"]["median"] < 0.2
    assert other["per_scene"][0]["eval_anchors"]["anchor_error_m"] == record


class _ResidencyArc(_SlotRecordingArc):
    """Records, as each forward starts, which earlier forwards' fields are alive."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields = []
        self.alive_at_forward = []

    def forward(self, views, **kwargs):
        self.alive_at_forward.append([field() is not None for field in self.fields])
        output = super().forward(views, **kwargs)
        self.fields.append(weakref.ref(output["track_multi"]))
        return output


def test_forward_a_fields_are_gone_before_forward_b_allocates(tmp_path, monkeypatch):
    """Forward A's fields are scored before the extra anchors run, so forward B
    allocates its own without the seated anchors' beside them; the shuffled arm
    still runs beside forward A's, as it always has."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))
    height, width = scene.views[0]["img"].shape[-2:]
    model = _ResidencyArc(scene.num_observations, height, width)

    _evaluate(tmp_path, scene, _all_spec(tmp_path, scene), model=model)

    assert model.alive_at_forward == [[], [True], [False, False]]


class _OverflowingArc(_SlotRecordingArc):
    """Confidence +inf, as `expp1` overflows in BF16, for one query slot."""

    def __init__(self, *args, overflowing_slot, **kwargs):
        super().__init__(*args, **kwargs)
        self.overflowing_slot = overflowing_slot

    def track_for_query(self, feats, images, query_idx, **kwargs):
        track, confidence = super().track_for_query(feats, images, query_idx, **kwargs)
        if query_idx == self.overflowing_slot:
            confidence = torch.full_like(confidence, float("inf"))
        return track, confidence


def test_the_seated_figures_count_the_seated_rows_alone(tmp_path, monkeypatch):
    """An extra anchor whose confidence overflows writes its rows as they came;
    confidence_nonfinite and the occluded fraction, figures the flag-off eval
    writes, stay on the seated rows."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))
    height, width = scene.views[0]["img"].shape[-2:]
    slot = _slot_of(scene, "0:2")

    def model():
        torch.manual_seed(0)
        return _OverflowingArc(
            scene.num_observations, height, width, overflowing_slot=slot
        )

    off, _ = _evaluate(tmp_path / "off", scene, None, model=model())
    on, _ = _evaluate(tmp_path / "on", scene, _all_spec(tmp_path, scene), model=model())

    assert np.isinf(_bundle(tmp_path / "on")["conf"][:, 1:]).all()
    assert on["confidence_nonfinite"] == off["confidence_nonfinite"] == 0
    assert json.dumps(_without_new_keys(on)) == json.dumps(_without_new_keys(off))


def test_the_visual_dump_shows_the_rows_the_bundle_holds(tmp_path, monkeypatch):
    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))

    _evaluate(
        tmp_path,
        scene,
        _all_spec(tmp_path, scene),
        visual_scenes=frozenset({"0000"}),
        run_identity={"checkpoint_dir": "released/4rc", "commit": "c0ffee"},
    )

    arrays, _ = read_visual_dump(tmp_path / "eval" / "step-5" / "visual" / "0000.npz")
    bundle = _bundle(tmp_path)
    assert bundle["query_points"].shape[0] == 3
    for key in VISUAL_TRACK_KEYS:
        assert np.array_equal(arrays[f"track_{key}"], bundle[key], equal_nan=True), key


def test_all_on_a_window_the_spec_fills_is_on_with_nothing_extra(tmp_path, monkeypatch):
    """None and [] differ: the flag off, against an `all` that found every pair
    of the window seated. On, the spec is recorded and every scene says so."""

    args = _held_out_args(tmp_path, val_cameras=[0], val_seq_len=1, eval_query_anchors=["all"])
    train_cli._validate_args(args)
    assert train_cli._eval_anchor_spec(args) == []

    scene = _cpu_eval_scene(monkeypatch)
    off, off_model = _evaluate(tmp_path / "off", scene, None)
    on, on_model = _evaluate(tmp_path / "on", scene, [])

    assert off["eval_query_anchors"] is None
    assert on["eval_query_anchors"] == ["0:0"]
    assert on["per_scene"][0]["eval_anchors"]["per_anchor"] == {}
    assert on_model.forward_slots == off_model.forward_slots
    assert json.dumps(_without_new_keys(on)) == json.dumps(_without_new_keys(off))


def test_each_kept_extra_anchor_scores_its_own_rows(tmp_path, monkeypatch):
    """Two kept extra anchors, at frames 2 and 3: each row is scored on its own
    anchor's field -- the fake's confidence names the query slot it came from
    -- and filed under its own key."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 3))

    on, model = _evaluate(tmp_path, scene, _all_spec(tmp_path, scene))

    record = on["per_scene"][0]["eval_anchors"]
    kept = [key for key, anchor in record["per_anchor"].items() if anchor["head_pass"]]
    assert kept == ["0:2", "0:3"]
    assert model.forward_slots[2] == [_slot_of(scene, "0:2"), _slot_of(scene, "0:3")]
    assert record["anchor_error_m"]["0:2"]["count"] == 1
    assert record["anchor_error_m"]["0:3"]["count"] == 1
    bundle = _bundle(tmp_path)
    assert bundle["query_points"][:, 0].tolist() == [0.0, 2.0, 3.0]
    # The fake's confidence is 1 + 100 * (slot + pixel fraction): each row
    # carries its own anchor's band.
    for row, key in ((1, "0:2"), (2, "0:3")):
        band = 1.0 + 100.0 * _slot_of(scene, key)
        assert band <= bundle["conf"][:, row].min() and bundle["conf"][:, row].max() < band + 100.0


def _plant_own_pointmaps(monkeypatch, scene):
    """Every slot's own ground-truth pointmap, where _cpu_eval_scene plants the
    reference's at every slot: an anchor read at the wrong observation lands
    on another camera's surface point."""

    import arc.training.sparse_tracking as sparse_module

    pointmaps = torch.stack(
        [
            torch.from_numpy(sparse_module._metric_pointmap_at_anchor(scene, slot)[0]).float()
            for slot in range(scene.num_observations)
        ]
    )[None]
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)


def test_an_extra_anchor_is_read_at_its_own_observation(tmp_path, monkeypatch):
    """Camera 1 takes the later tracks; with each slot holding its own surface,
    their model anchor is the true point up to pixel rounding -- read at the
    primary's camera-0 observation instead, it would sit a baseline away."""

    scene = _cpu_eval_scene(
        monkeypatch, query_times=(0, 2, 2), invisible=((0, 2, 1), (0, 2, 2))
    )
    _plant_own_pointmaps(monkeypatch, scene)

    on, _ = _evaluate(tmp_path, scene, _all_spec(tmp_path, scene))

    errors = on["per_scene"][0]["eval_anchors"]["anchor_error_m"]
    assert list(errors) == ["1:2"]
    assert errors["1:2"]["count"] == 2
    assert errors["1:2"]["p90"] < 0.1


def test_a_world_frame_anchor_is_composed_once(tmp_path, monkeypatch):
    """Under query_position every anchor is the query point in the world frame
    and only the displacement is mapped, so every row -- seated or extra -- is
    its query point plus one mapped displacement, the fake's being uniform.
    The gauge is scaled, so a second trip through the Sim(3) would show."""

    scene = _cpu_eval_scene(monkeypatch, gauge="scaled", query_times=(0, 2, 2))

    _evaluate(tmp_path, scene, _all_spec(tmp_path, scene), source="query_position")

    bundle = _bundle(tmp_path)
    offsets = bundle["pred"] - bundle["query_points"][None, :, 1:]
    assert np.allclose(offsets, offsets[:, :1], atol=1e-5)
    assert np.abs(offsets).max() > 0.1


def test_an_extra_anchor_that_admits_nothing_runs_no_head_pass(tmp_path, monkeypatch):
    """Every track starts at 0, so every extra anchor is dropped: the eval runs
    the flag-off forwards alone and writes the flag-off bundle, byte for byte,
    while the record keeps every dropped anchor's counts."""

    scene = _cpu_eval_scene(monkeypatch)
    spec = _all_spec(tmp_path, scene)

    off, off_model = _evaluate(tmp_path / "off", scene, None)
    on, on_model = _evaluate(tmp_path / "on", scene, spec)

    assert on_model.forward_slots == off_model.forward_slots
    assert len(on_model.forward_slots) == 2
    pred = Path("eval") / "step-5" / "pred" / "0000.npz"
    assert (tmp_path / "on" / pred).read_bytes() == (tmp_path / "off" / pred).read_bytes()
    record = on["per_scene"][0]["eval_anchors"]
    assert not any(anchor["head_pass"] for anchor in record["per_anchor"].values())
    assert record["per_anchor"]["1:0"] == {"eligible": 3, "assigned": 0, "head_pass": False}
    assert record["anchor_error_m"] == {} and record["base_ratio"] == {}
    assert json.dumps(_without_new_keys(on)) == json.dumps(_without_new_keys(off))


def test_a_scene_the_seated_anchors_cannot_reach_is_skipped_as_without_the_flag(
    tmp_path, monkeypatch, capsys
):
    """The skip stays the seated anchors' call: a scene they cannot reach is
    skipped exactly as without the flag, extra anchors or not, and no forward
    runs at all."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(2, 2, 2))
    spec = _all_spec(tmp_path, scene)

    off, off_model = _evaluate(tmp_path / "off", scene, None)
    capsys.readouterr()
    on, on_model = _evaluate(tmp_path / "on", scene, spec)

    (line,) = [line for line in capsys.readouterr().out.splitlines() if "anchors=" in line]
    assert line.startswith("eval step=5 scene=0000 anchors=1+0/7 rows=0+0 seconds=")
    assert off_model.forward_slots == on_model.forward_slots == []
    assert on["skipped_scenes"] == off["skipped_scenes"]
    assert on["scenes"] == 0
    assert not (tmp_path / "on" / "eval" / "step-5" / "pred").exists()
    assert json.dumps(_without_new_keys(on)) == json.dumps(_without_new_keys(off))


def test_the_extra_anchors_read_out_against_the_same_primary(tmp_path, monkeypatch):
    """A kept extra anchor's base_ratio and anchor error are measured exactly as
    a seated anchor's are -- off forward A's reconstruction and the primary --
    so, the cameras being static, camera 1 at time 2 reads what camera 1 at
    time 0 reads. The fake moves forward B's cameras three times as far, so a
    readout taken off forward B would not."""

    scene = _cpu_eval_scene(
        monkeypatch,
        gauge="scaled",
        query_times=(0, 2, 2),
        # Hidden from camera 0 at time 2, so camera 1 -- next in spec order --
        # takes both later tracks.
        invisible=((0, 2, 1), (0, 2, 2)),
        **_FOUR_CAMERA_WINDOW,
    )
    spec = _all_spec(tmp_path, scene, query_anchors=_FOUR_ANCHOR_SPEC)
    factors = [{1: 0.9, 2: 1.25, 3: 0.6}]

    off, _ = _evaluate(
        tmp_path / "off",
        scene,
        None,
        model=_GroundTruthPoseArc([scene], factors),
        query_anchors=_FOUR_ANCHOR_SPEC,
    )
    on, _ = _evaluate(
        tmp_path / "on",
        scene,
        spec,
        model=_GroundTruthPoseArc([scene], factors),
        query_anchors=_FOUR_ANCHOR_SPEC,
    )

    record = on["per_scene"][0]["eval_anchors"]
    assert [key for key, anchor in record["per_anchor"].items() if anchor["head_pass"]] == [
        "1:2"
    ]
    # base_ratio composes through the fitted scale on purpose (the planted
    # gauge's 2.25), so camera 1's factor 0.9 reads as 0.9 * 2.25; forward B's
    # cameras, three times as far, would read three times that.
    scale = off["per_scene"][0]["alignment_scale"]
    assert record["base_ratio"]["1:2"] == pytest.approx(0.9 * scale, rel=1e-5)
    assert record["base_ratio"]["1:2"] == off["per_scene"][0]["drift"]["base_ratio"]["1:0"]
    assert record["anchor_error_m"]["1:2"]["count"] == 2
    # The pooled readouts stay on the seated anchors.
    for key in train_cli._readout_aggregates([], []):
        assert json.dumps(on[key]) == json.dumps(off[key]), key


def test_the_eval_restores_rng_with_extra_anchors(tmp_path, monkeypatch):
    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))
    spec = _all_spec(tmp_path, scene)
    height, width = scene.views[0]["img"].shape[-2:]
    model = _SlotRecordingArc(scene.num_observations, height, width)
    model.head.eval()
    modes = {name: module.training for name, module in model.named_modules()}
    torch.manual_seed(7)
    before = capture_rng_state()

    _evaluate(tmp_path, scene, spec, model=model)

    assert len(model.forward_slots) == 3
    _assert_same_rng_state(capture_rng_state(), before)
    assert {name: module.training for name, module in model.named_modules()} == modes


def test_the_scene_line_is_printed_only_with_the_flag(tmp_path, monkeypatch, capsys):
    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))
    spec = _all_spec(tmp_path, scene)

    _evaluate(tmp_path / "off", scene, None)
    assert "anchors=" not in capsys.readouterr().out
    _evaluate(tmp_path / "on", scene, spec)
    (line,) = [
        line for line in capsys.readouterr().out.splitlines() if "anchors=" in line
    ]
    assert line.startswith("eval step=5 scene=0000 anchors=1+1/7 rows=1+2 seconds=")


# ------------------------------------------------------------- the loop ---


def _loop(directory, scene, **overrides):
    """Four real CPU train_steps with a held-out eval at 2 and 4 and a save at 2."""

    torch.manual_seed(0)
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)
    plan = _window_plan(scene)
    return train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(
            directory,
            num_steps=4,
            eval_every=2,
            save_every=2,
            val_scenes_file="val.json",
            val_cameras=list(scene.camera_ids),
            val_seq_len=len(scene.times),
            stride=1,
            observation_budget=len(scene.camera_ids) * len(scene.times),
            **overrides,
        ),
        scene_provider=lambda _plan: scene,
        output_dir=directory,
        held_out=train_cli.HeldOut(
            plans=[plan], scene_provider=lambda _plan: scene, benchmark=None
        ),
    )


def test_the_flag_moves_no_training_byte(tmp_path, monkeypatch):
    """The real step, with and without the extra anchors: history.jsonl and
    train_state.pt are byte for byte the same, and so are the RNG streams the
    run leaves behind; the evals differ by the extra rows and keys alone."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))

    off = _loop(tmp_path / "off", scene)
    off_rng = capture_rng_state()
    on = _loop(tmp_path / "on", scene, eval_query_anchors=["all"])
    on_rng = capture_rng_state()

    for name in ("history.jsonl", "train_state.pt"):
        assert (tmp_path / "on" / name).read_bytes() == (tmp_path / "off" / name).read_bytes()
    _assert_same_rng_state(on_rng, off_rng)
    assert [entry["step"] for entry in on["evaluations"]] == [2, 4]
    for on_eval, off_eval in zip(on["evaluations"], off["evaluations"]):
        assert json.dumps(_without_new_keys(on_eval)) == json.dumps(_without_new_keys(off_eval))
        assert on_eval["eval_query_anchors"][0] == "0:0"
        assert len(on_eval["eval_query_anchors"]) == 8


def test_the_held_out_provider_is_moved_to_the_device_like_the_training_one(
    tmp_path, monkeypatch
):
    """Both loaders go through cuda_scene_provider: the preflight and every
    eval load the held-out scenes through the wrapped held-out provider."""

    scene = _cpu_eval_scene(monkeypatch, query_times=(0, 2, 2))
    original = train_cli.cuda_scene_provider
    through = []

    def recording(provider):
        moved = original(provider)

        def load(plan):
            through.append((provider, plan.seq_name))
            return moved(plan)

        return load

    monkeypatch.setattr(train_cli, "cuda_scene_provider", recording)

    def held(_plan):
        return scene

    torch.manual_seed(0)
    model = _FakeArc(scene.num_observations, *scene.views[0]["img"].shape[-2:])
    train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=4, eval_every=4, val_scenes_file="val.json"),
        scene_provider=lambda _plan: scene,
        output_dir=tmp_path,
        held_out=train_cli.HeldOut(
            plans=[_window_plan(scene, "held")], scene_provider=held, benchmark=None
        ),
    )

    assert [name for provider, name in through if provider is held] == ["held", "held"]


def test_main_scores_the_benchmark_it_opened():
    """Inspected, as main() cannot run past the model load here: the held-out
    set main hands run_training is built from the provider _open_benchmark
    returned."""

    tree = ast.parse(Path(train_cli.__file__).read_text())
    (main,) = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_held_out_set"
    ]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == ["args", "provider", "benchmark_provider"]
    opened = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "benchmark_provider" for target in node.targets)
    ]
    assert [ast.unparse(node.value) for node in opened] == ["_open_benchmark(args)"]


def test_the_trainer_wires_the_resolved_extra_anchors_into_the_eval():
    """Inspected, as the other eval wiring pins are: the eval call passes the
    resolver's value, bound once from args, and the held-out set's benchmark."""

    tree = ast.parse(Path(train_cli.__file__).read_text())

    def is_name(node, name):
        return isinstance(node, ast.Name) and node.id == name

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and is_name(node.func, "evaluate_held_out")
    ]
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert is_name(keywords["eval_query_anchors"], "eval_query_anchors")
    assert is_name(keywords["benchmark"], "benchmark")
    assert is_name(keywords["scene_provider"], "val_scene_provider")
    bound = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in ("eval_query_anchors", "benchmark", "val_scene_provider")
    }
    assert is_name(bound["eval_query_anchors"].func, "_eval_anchor_spec")
    assert [is_name(arg, "args") for arg in bound["eval_query_anchors"].args] == [True]
    assert ast.unparse(bound["benchmark"]) == (
        "held_out.benchmark if held_out is not None else None"
    )
    assert "held_out.scene_provider" in ast.unparse(bound["val_scene_provider"])
