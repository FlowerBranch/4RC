"""Pins for the manifest replay's decision layer.

The planner is the whole of landing 2: it turns one MVTracker manifest record
into a concrete window selection, and it is the part that can be settled without
a GPU, without scene data and without knowing yet whether scenes will arrive from
the live dataset or through the dumper.

Three of these tests exist because the schema names a hazard that a reasonable
implementation walks straight into -- repeated ``track_indices``, null-provenance
rows, and the difference between a row that cannot be replayed and one this
trainer merely declines. Those are the ones worth reading.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

import train_temporal_tracking as train_cli
from arc.training.manifest_plan import (
    BOUND_BUDGET,
    BOUND_EMBEDDING_ROWS,
    BOUND_WINDOW,
    SKIP_EXCLUDED_DATA_ROOT,
    SKIP_TOO_FEW_VIEWS,
    SKIP_UNREPLAYABLE,
    ManifestPlanError,
    StepPlan,
    plan_manifest,
    plan_record,
    require_cameras,
    select_times,
)
from arc.training.sample_manifest import (
    MANIFEST_RECORD_KEYS,
    MANIFEST_VERSION,
    encode_manifest_line,
    read_manifest,
)


def _record(**overrides):
    """A complete, valid record. Every key present, as the schema requires."""

    record = {
        "manifest_version": MANIFEST_VERSION,
        "step": 0,
        "rank": 0,
        "batch_index": 0,
        "batch_size": 1,
        "data_root": "/data/splits/curve/kubric-multiview/train",
        "seq_name": "0731",
        "sample_index": 0,
        "real_len": 100,
        "dataset_seed": 7,
        "sample_seed": 7,
        "views": [0, 1, 2, 3],
        "frame_start": 0,
        "seq_len": 24,
        "track_indices": [3, 1, 4, 1, 5],
        "traj_per_sample": 5,
        "traj_per_sample_configured": 2048,
        "depth_type": "gt",
        "augmented": False,
        "scene_transform": {"scale": 1.0, "rot_x_deg": 0.0, "rot_y_deg": 0.0},
    }
    record.update(overrides)
    return record


def _write_manifest(path: Path, records) -> Path:
    path.write_text("".join(encode_manifest_line(r) + "\n" for r in records))
    return path


# ------------------------------------------------------------------- replay ---


def test_replay_is_faithful_over_a_hand_written_manifest(tmp_path):
    """Work-order test 1: five records in, five steps out, in that order.

    The manifest is the authority on what a step consumed -- ``sample_index``
    plus the dataset config provably does not reproduce a draw, because the
    cropping augmentation feeds off the global numpy stream. So a replay that
    reordered, dropped or re-derived anything here would be training on a
    different stream while every curve still looked comparable.
    """

    records = [
        _record(step=0, seq_name="0731", views=[0, 1, 2, 3], frame_start=0, seq_len=24),
        _record(step=1, seq_name="0004", views=[2, 3], frame_start=0, seq_len=24),
        _record(step=2, seq_name="0731", views=[0, 1], frame_start=4, seq_len=20),
        _record(step=3, seq_name="1200", views=[1, 2, 3], frame_start=0, seq_len=12),
        _record(step=4, seq_name="0004", views=[0, 3], frame_start=2, seq_len=22),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)

    tally = plan_manifest(read_manifest(manifest), budget=40, stride=2)

    assert [plan.step for plan in tally.planned] == [0, 1, 2, 3, 4]
    assert [plan.seq_name for plan in tally.planned] == [
        "0731",
        "0004",
        "0731",
        "1200",
        "0004",
    ]
    assert [plan.cameras for plan in tally.planned] == [
        (0, 1, 2, 3),
        (2, 3),
        (0, 1),
        (1, 2, 3),
        (0, 3),
    ]
    assert [(plan.frame_start, plan.seq_len) for plan in tally.planned] == [
        (0, 24),
        (0, 24),
        (4, 20),
        (0, 12),
        (2, 22),
    ]
    assert not tally.skipped


def test_camera_order_is_preserved_rather_than_sorted(tmp_path):
    """``views`` is the order along the view axis, not a set.

    Sorting them would silently transpose the model's view axis relative to the
    arrays the manifest describes.
    """

    plan = plan_record(_record(views=[3, 1, 0, 2]), budget=40, stride=2)

    assert plan.cameras == (3, 1, 0, 2)


# ------------------------------------------------------------------ cameras ---


def test_a_camera_the_scene_does_not_have_fails_loudly_naming_both():
    """Work-order test 2, checked against ids rather than a loader exception.

    ``load_dumped_kubric_scene`` raises ``ValueError`` for non-increasing times,
    a bad anchor and a bad upscaling factor too, so a wrapper that caught it and
    reported "views the dump cannot serve" would mislabel three unrelated faults
    as a camera problem.
    """

    with pytest.raises(ManifestPlanError, match=r"0731.*camera\(s\) \[7\].*\[0, 1, 2, 3\]"):
        require_cameras("0731", [0, 1, 2, 3], [0, 7])


def test_the_camera_check_runs_through_plan_record_when_scene_facts_are_supplied():
    """The check has to be reachable from the planner, not only callable."""

    available = {"0731": [0, 1, 2, 3]}

    plan = plan_record(_record(views=[0, 2]), available_cameras=available, budget=40)
    assert plan.cameras == (0, 2)

    with pytest.raises(ManifestPlanError, match="camera"):
        plan_record(_record(views=[0, 9]), available_cameras=available, budget=40)

    # A callable is accepted too, which is how landing 3 will hand it a scene source.
    plan = plan_record(
        _record(views=[0, 2]),
        available_cameras=lambda name: available[name],
        budget=40,
    )
    assert plan.cameras == (0, 2)


def test_a_single_view_record_is_skipped_rather_than_clamped():
    """One camera has no synchronized pair, so there is nothing to learn from.

    Clamping T instead would produce a step that trains happily and cannot
    measure the only thing the run exists to measure.
    """

    outcome = plan_record(_record(views=[1]), budget=40, stride=2)

    assert outcome.cause == SKIP_TOO_FEW_VIEWS
    assert "synchronized cross-view pair" in outcome.detail


# -------------------------------------------------------------------- times ---


def test_time_selection_stays_in_window_and_is_strictly_increasing():
    """Work-order test 3, over every window the committed settings can meet."""

    for seq_len in range(1, 40):
        for view_count in (2, 3, 4):
            times, bound = select_times(
                frame_start=5,
                seq_len=seq_len,
                view_count=view_count,
                budget=40,
                stride=2,
                max_time_indices=32,
            )

            assert times, "selection must never be empty"
            assert list(times) == sorted(set(times)), "strictly increasing, no repeats"
            assert times[0] == 5
            assert times[-1] < 5 + seq_len, "must stay inside the manifest's window"
            assert bound in {BOUND_BUDGET, BOUND_EMBEDDING_ROWS, BOUND_WINDOW}


def test_the_committed_window_is_four_cameras_by_twelve_times_at_stride_two():
    """The first run's window, pinned so a default drift is visible here.

    48 observations, measured at 132.2 GiB on an H200 — 94% of the card, which is
    why the number is pinned rather than left to a default someone can nudge.
    """

    times, bound = select_times(
        frame_start=0,
        seq_len=24,
        view_count=4,
        budget=train_cli.DEFAULT_OBSERVATION_BUDGET,
        stride=train_cli.DEFAULT_STRIDE,
        max_time_indices=32,
    )

    assert train_cli.DEFAULT_OBSERVATION_BUDGET == 48
    assert times == (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    assert bound == BOUND_BUDGET
    assert len(times) * 4 == 48
    # T=12 is inside the embedding table, which is the other ceiling on a window.
    assert len(times) <= 32


def test_the_forty_observation_fallback_keeps_the_embedding_contract():
    """`--observation_budget 40` is the documented way down from 94% occupancy.

    It must change only how many times a window carries, never what a row means:
    the stride is unchanged, so row k is still k*stride frames after the anchor
    and a patch stays loadable across the change.
    """

    fallback, _ = select_times(
        frame_start=0, seq_len=24, view_count=4, budget=40, stride=2, max_time_indices=32
    )
    committed, _ = select_times(
        frame_start=0, seq_len=24, view_count=4, budget=48, stride=2, max_time_indices=32
    )

    assert fallback == committed[: len(fallback)]
    assert len(fallback) * 4 == 40


def test_which_bound_decided_the_time_count_is_reported():
    """Three bounds with different fixes: a flag, a model constant, the manifest.

    Reporting only ``T`` would leave an operator unable to tell "raise the
    budget" from "this clip is short" from "the embedding table is full".
    """

    _, budget_bound = select_times(
        frame_start=0, seq_len=100, view_count=4, budget=40, stride=1, max_time_indices=32
    )
    assert budget_bound == BOUND_BUDGET

    _, rows_bound = select_times(
        frame_start=0, seq_len=100, view_count=1, budget=1000, stride=1, max_time_indices=32
    )
    assert rows_bound == BOUND_EMBEDDING_ROWS

    _, window_bound = select_times(
        frame_start=0, seq_len=6, view_count=2, budget=1000, stride=1, max_time_indices=32
    )
    assert window_bound == BOUND_WINDOW


def test_a_stride_that_overruns_the_window_still_stays_inside_it():
    """The window bound is computed from the stride, not from the frame count."""

    times, bound = select_times(
        frame_start=10, seq_len=5, view_count=2, budget=40, stride=3, max_time_indices=32
    )

    assert times == (10, 13)
    assert bound == BOUND_WINDOW


# ------------------------------------------------------------------- tracks ---


def test_repeated_track_indices_survive_positionally():
    """The hazard the schema calls out by name, and the reason not to use a set.

    The loader draws independently from a "dynamic" and a "very dynamic" pool,
    the second a subset of the first, so an id can legitimately appear twice --
    as a second, independent column with its own query timestep. A ``set`` or a
    dict keyed by id drops that column, and nothing downstream can tell: the
    tensors are simply narrower than the manifest says.
    """

    record = _record(track_indices=[3, 1, 4, 1, 5, 3, 3], traj_per_sample=7)

    plan = plan_record(record, budget=40, stride=2)

    assert plan.track_indices == (3, 1, 4, 1, 5, 3, 3)
    assert len(plan.track_indices) == record["traj_per_sample"]
    assert len(plan.track_indices) != len(set(plan.track_indices))
    # Reported as a number so a set-based regression is visible in the artifacts.
    assert plan.duplicate_track_count == 3


# -------------------------------------------------------------- row policy ---


def test_a_null_data_root_row_is_unreplayable_and_counts_toward_the_threshold():
    """No provenance stamp means the loader observations are null with it."""

    outcome = plan_record(
        _record(
            data_root=None,
            sample_index=None,
            views=None,
            frame_start=None,
            track_indices=None,
        ),
        budget=40,
    )

    assert outcome.cause == SKIP_UNREPLAYABLE
    assert outcome.counts_toward_threshold


def test_an_excluded_data_root_is_a_deliberate_choice_not_damage():
    """Static-pretraining rows are replayable rows this trainer declines.

    They are hundreds of rows at the head of a run and are marked by nothing but
    their ``data_root``. Counting them with genuine corruption would trip any
    threshold small enough to catch corruption at all -- so they are excluded
    from both sides of the ratio, which also stops a large exclusion from diluting
    a real failure into passing.
    """

    static = "/data/static-pretrain"
    records = [_record(step=i, data_root=static, dataset_seed=None) for i in range(300)]
    records += [_record(step=300 + i) for i in range(10)]
    records += [_record(step=400, data_root=None)]

    tally = plan_manifest(records, budget=40, excluded_data_roots=[static])

    assert tally.skip_counts[SKIP_EXCLUDED_DATA_ROOT] == 300
    assert tally.skip_counts[SKIP_UNREPLAYABLE] == 1
    assert len(tally.planned) == 10
    # 1 unreplayable against 11 considered, not against 311.
    assert tally.considered == 11
    assert tally.threshold_skip_fraction == pytest.approx(1 / 11)


# ------------------------------------------------------------ vendored copy ---


def test_the_vendored_manifest_module_is_byte_identical_to_its_recorded_hash():
    """"We vendored it" is a comment; this makes it a check.

    The upstream module is the single definition of the format and is stdlib-only
    so it can be copied rather than reimplemented. A local edit is how a reader
    and a writer drift apart with no test noticing, so the header records the
    hash of the body and this recomputes it.
    """

    source = Path(train_cli.__file__).parent / "arc" / "training" / "sample_manifest.py"
    text = source.read_text()
    header, _, body = text.partition('"""Per-step sample manifest')
    body = '"""Per-step sample manifest' + body

    recorded = re.findall(r"^#\s*sha256\s*:\s*([0-9a-f]{64})\s*$", header, re.MULTILINE)
    assert recorded, "the vendoring header must record the upstream hash"
    assert hashlib.sha256(body.encode()).hexdigest() == recorded[0]


def test_the_vendored_module_round_trips_every_schema_key(tmp_path):
    """A record must survive encode/read with its key set intact."""

    record = _record()
    assert set(record) == set(MANIFEST_RECORD_KEYS)

    manifest = _write_manifest(tmp_path / "m.jsonl", [record])
    (read_back,) = read_manifest(manifest)

    assert read_back == record


def test_a_malformed_manifest_line_is_rejected_by_line_number(tmp_path):
    """The writer flushes per record, so a truncation lands between lines.

    A malformed line is therefore real corruption rather than a killed job, and
    papering over it would replay a manifest that is not the one written.
    """

    path = tmp_path / "m.jsonl"
    path.write_text(encode_manifest_line(_record()) + "\n" + "{not json\n")

    with pytest.raises(ValueError, match=r":2: malformed manifest line"):
        read_manifest(path)


# ----------------------------------------------------------------------- cli ---


def test_plan_only_reports_a_manifest_without_cuda_or_a_checkpoint(tmp_path, monkeypatch, capsys):
    """The mode's whole point: a submit-time answer, before a GPU is allocated."""

    records = [_record(step=i, seq_name=f"{i:04d}") for i in range(3)]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    out = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_temporal_tracking.py",
            "--manifest",
            str(manifest),
            "--plan_only",
            "--json_out",
            str(out),
        ],
    )

    train_cli.main()

    printed = capsys.readouterr().out
    assert "PASS planned 3 steps" in printed
    summary = json.loads(out.read_text())
    assert summary["planned_steps"] == 3
    assert summary["distinct_scenes"] == 3
    assert summary["observations_per_step"] == {"48": 3}
    assert summary["time_bound"] == {BOUND_BUDGET: 3}
    assert summary["duplicate_track_ids"]["total"] == 3
    assert summary["records_with_scene_transform"] == 3


def test_the_trainer_names_the_one_missing_piece_rather_than_planning_quietly(
    tmp_path, monkeypatch
):
    """Landing 3 has the loop but not the scene source; it must say which.

    An entry point that accepted the training flags and quietly planned instead
    would be the worst version of a partial landing — and a generic "not
    implemented" would send someone auditing the loop, which is written. The
    message names `scene_provider` and why it is unbound.
    """

    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record()])
    monkeypatch.setattr(
        sys, "argv", ["train_temporal_tracking.py", "--manifest", str(manifest)]
    )

    with pytest.raises(NotImplementedError, match="scene_provider is not bound yet"):
        train_cli.main()


def test_an_all_unreplayable_manifest_exits_non_zero(tmp_path, monkeypatch):
    """Work-order test 6's shape: a manifest this trainer cannot use must fail.

    Training on the handful of rows that happened to parse, while reporting a
    healthy run, is the failure mode the threshold exists to prevent.
    """

    records = [_record(step=i, data_root=None) for i in range(5)]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_temporal_tracking.py", "--manifest", str(manifest), "--plan_only"],
    )

    with pytest.raises(SystemExit) as excinfo:
        train_cli.main()

    assert excinfo.value.code == 1
