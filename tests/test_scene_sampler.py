"""Pins for the scene-pool sampler: the manifest-free record source.

The sampler's one promise is that a sampled run is "a different record source"
and nothing else: every record it builds is a manifest-schema record, every
planned step went through ``plan_record`` with the arguments ``main()`` hands
the manifest replay, and the stream is a pure function of the pool index, the
seed, the weights, the transform probability and the planner's own flags -- so
a longer run is a prefix-extension and a resume is a slice.  The tests here are
grouped by which of those claims they pin.
"""

from __future__ import annotations

import itertools
import json
import signal
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_temporal_tracking as train_cli
from arc.training import scene_sampler
from arc.training.manifest_plan import (
    SKIP_TOO_FEW_VIEWS,
    SKIP_UNREPLAYABLE,
    StepPlan,
    plan_manifest,
    plan_record,
)
from arc.training.sample_manifest import MANIFEST_RECORD_KEYS, MANIFEST_VERSION
from arc.training.scene_provider import (
    apply_recorded_scene_transform,
    rotation_from_degrees,
)
from arc.training.scene_sampler import (
    DEFAULT_VIEW_COUNT_WEIGHTS,
    LOADER_CLIP_FRAMES,
    POOL_INDEX_NAME,
    ROTATION_RANGE_DEG,
    SCALE_RANGE,
    WINDOW_FIELDS,
    PoolIndex,
    PoolScene,
    SampledPlanTally,
    ScenePoolError,
    admissible_counts,
    build_pool_index,
    build_record,
    check_view_count_weights,
    compare_sampler_against_manifest,
    draw_scenes,
    format_view_count_weights,
    parse_view_count_weights,
    plan_pool,
    read_pool_index,
    sampler_identity,
    write_pool_index,
)
from arc.training.trainer_state import read_trainer_state
import test_scene_provider
from test_manifest_plan import _record, _write_manifest
from test_trainer_loop import _loop_args, _toy_model, _validator_args

# test_scene_provider's stand-in for MVTracker's transform_scene, registered
# here under its own name: pytest takes a fixture's name from the attribute it
# finds it under, so this is the same fixture, not a copy.
transform_scene_calls = test_scene_provider.stub_transform_scene

DEFAULT_WEIGHTS = parse_view_count_weights(DEFAULT_VIEW_COUNT_WEIGHTS)
MIX = "1:0.20,2:0.10,3:0.10,4:0.25,5:0.10,6:0.25"
# The planner arguments main() hands plan_manifest and plan_pool alike.
PLAN_KWARGS = dict(
    available_cameras=None,
    budget=48,
    stride=2,
    max_time_indices=32,
    min_views=2,
    excluded_data_roots=(),
)


# ----------------------------------------------------------------- fixtures ---


def _index(view_counts: dict[str, int]) -> PoolIndex:
    """An in-memory index in the loader's order, for the draw tests."""

    scenes = tuple(
        PoolScene(name=name, view_count=count)
        for name, count in sorted(view_counts.items(), key=lambda item: int(item[0]))
    )
    return PoolIndex(scenes=scenes, content_hash="in-memory")


def write_pool(root: Path, view_counts: dict[str, int], *, loose=(), zero_view=()) -> Path:
    """A packed pool: ``<name>.zip`` (ZIP_STORED) per scene, or a loose directory.

    A scene is only ever opened for its ``view_*`` folders here, so each view
    carries one member. ``zero_view`` names scenes that exist but hold no view
    folder, which the loader leaves out of the pool.
    """

    root.mkdir(parents=True, exist_ok=True)
    for name, count in view_counts.items():
        members = [f"view_{index}/metadata.json" for index in range(count)]
        if name in zero_view:
            members = ["cameras.npz"]
        if name in loose:
            for member in members:
                path = root / name / member
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"{}")
        else:
            with zipfile.ZipFile(
                root / f"{name}.zip", "w", compression=zipfile.ZIP_STORED
            ) as archive:
                for member in members:
                    archive.writestr(member, b"{}")
    return root


def _sampled(num_steps, *, seed=3, index=None, weights=None, p=1.0, **overrides):
    kwargs = dict(
        data_root="/pool",
        num_steps=num_steps,
        seed=seed,
        weights=DEFAULT_WEIGHTS if weights is None else weights,
        transform_probability=p,
        **PLAN_KWARGS,
    )
    kwargs.update(overrides)
    return plan_pool(
        _index({"0000": 10, "0001": 6, "0002": 3, "0003": 4}) if index is None else index,
        **kwargs,
    )


def _identity(**overrides) -> dict:
    identity = {
        "pool": "/pool",
        "index_hash": "abc123",
        "sampler_seed": 3,
        "view_count_weights": format_view_count_weights(DEFAULT_WEIGHTS),
        "transform_probability": 1.0,
    }
    identity.update(overrides)
    return identity


def _draws(count, **kwargs):
    settings = dict(seed=3, weights=DEFAULT_WEIGHTS, transform_probability=1.0, min_views=2)
    settings.update(kwargs)
    index = settings.pop("index", _index({"0000": 10}))
    return list(itertools.islice(draw_scenes(index, **settings), count))


# -------------------------------------------------------------- the draw ---


def test_the_view_count_histogram_follows_the_default_weights():
    """10k draws on a 10-view pool: every count at its renormalised weight.

    The default is MVTracker's draw without its single-view mass, so the five
    counts renormalise to .125/.125/.3125/.125/.3125; 0.02 is ~4.6 sigma at
    n=10k, so a shifted or unnormalised draw fails and a correct one cannot.
    """

    draws = _draws(10_000)
    counts = Counter(len(draw.views) for draw in draws)
    total = sum(DEFAULT_WEIGHTS.values())
    for count, weight in DEFAULT_WEIGHTS.items():
        assert abs(counts[count] / 10_000 - weight / total) < 0.02, (count, counts)
    assert 1 not in counts
    assert set(counts) == set(DEFAULT_WEIGHTS)


def test_the_single_view_control_and_the_monocular_mix():
    """'1:1.0' is all ones; the six-entry mix carries the loader's 0.20."""

    control = _draws(500, weights=parse_view_count_weights("1:1.0"), min_views=1)
    assert all(len(draw.views) == 1 for draw in control)

    mix = _draws(10_000, weights=parse_view_count_weights(MIX), min_views=1)
    counts = Counter(len(draw.views) for draw in mix)
    assert abs(counts[1] / 10_000 - 0.20) < 0.02, counts
    assert abs(counts[4] / 10_000 - 0.25) < 0.02, counts


def test_a_scene_never_draws_more_views_than_it_has_and_renormalises_the_rest():
    """A 3-view scene under the default draws 2 or 3, at 0.10:0.10 -> 1:1."""

    draws = _draws(10_000, index=_index({"0000": 3, "0001": 10}))
    narrow = Counter(len(draw.views) for draw in draws if draw.seq_name == "0000")
    assert set(narrow) == {2, 3}, narrow
    assert abs(narrow[2] / sum(narrow.values()) - 0.5) < 0.03, narrow
    wide = Counter(len(draw.views) for draw in draws if draw.seq_name == "0001")
    assert set(wide) == set(DEFAULT_WEIGHTS)
    assert admissible_counts(3, DEFAULT_WEIGHTS) == ([2, 3], [0.5, 0.5])
    assert admissible_counts(1, DEFAULT_WEIGHTS) == ([], [])


def test_views_are_distinct_in_range_and_in_drawn_order():
    """The loader keeps choice()'s order, so the view axis is NOT sorted."""

    draws = _draws(2_000)
    for draw in draws:
        assert len(set(draw.views)) == len(draw.views)
        assert all(0 <= view < 10 for view in draw.views)
    assert any(list(draw.views) != sorted(draw.views) for draw in draws)


def test_every_admissible_scene_appears_once_per_epoch_before_any_repeats():
    """Seven scenes, one of them single-view: six per epoch, each exactly once."""

    index = _index({f"{i:04d}": 10 for i in range(6)} | {"0006": 1})
    admissible = {f"{i:04d}" for i in range(6)}

    twelve = _sampled(12, index=index)
    names = [plan.seq_name for plan in twelve.planned]
    assert set(names[:6]) == admissible
    assert set(names[6:]) == admissible
    assert "0006" not in names
    assert twelve.epochs_drawn == 2
    assert twelve.scene_count == 7
    assert twelve.admissible_scenes == 6
    assert twelve.draws == 12

    assert _sampled(6, index=index).epochs_drawn == 1
    assert _sampled(7, index=index).epochs_drawn == 2


def test_a_pool_no_listed_count_fits_is_refused_not_looped():
    with pytest.raises(ScenePoolError, match="no scene in the pool seats"):
        _sampled(4, index=_index({"0000": 1, "0001": 1}))


def test_a_listed_count_below_min_views_is_refused_at_the_draw_too():
    """The parse-time rule, re-checked at the seam a direct caller reaches."""

    with pytest.raises(ScenePoolError, match="below min_views 2"):
        _draws(1, weights=parse_view_count_weights(MIX), min_views=2)


# ------------------------------------------------------------- transforms ---


def test_the_transform_fraction_is_the_probability_and_the_dict_is_the_schemas():
    for probability, expected in ((1.0, 1.0), (0.0, 0.0), (0.3, 0.3)):
        draws = _draws(10_000, transform_probability=probability)
        fraction = sum(draw.scene_transform is not None for draw in draws) / 10_000
        assert abs(fraction - expected) < 0.02, (probability, fraction)
    for draw in _draws(500):
        transform = draw.scene_transform
        assert set(transform) == {"scale", "rot_x_deg", "rot_y_deg"}
        assert SCALE_RANGE[0] <= transform["scale"] <= SCALE_RANGE[1]
        assert ROTATION_RANGE_DEG[0] <= transform["rot_x_deg"] <= ROTATION_RANGE_DEG[1]
        assert ROTATION_RANGE_DEG[0] <= transform["rot_y_deg"] <= ROTATION_RANGE_DEG[1]
        assert all(isinstance(value, float) for value in transform.values())


def test_the_probability_only_masks_the_transform_and_leaves_the_draw_alone():
    """Two runs differing only in p see the same scenes, views and values."""

    always = _draws(300, transform_probability=1.0)
    sometimes = _draws(300, transform_probability=0.5)
    assert [(d.seq_name, d.views) for d in always] == [(d.seq_name, d.views) for d in sometimes]
    for full, masked in zip(always, sometimes):
        assert masked.scene_transform in (None, full.scene_transform)
    assert any(d.scene_transform is None for d in sometimes)
    assert any(d.scene_transform is not None for d in sometimes)


def test_a_sampled_transform_reaches_transform_scene_unchanged(transform_scene_calls):
    """The dict the sampler emits is the dict the provider applies, as is."""

    (draw,) = _draws(1)
    sample = SimpleNamespace(
        videodepth="depth-in",
        extrs="extrs-in",
        query_points_3d="query-in",
        trajectory_3d="traj-in",
        track_upscaling_factor=1.0,
    )
    apply_recorded_scene_transform(sample, draw.scene_transform)
    (call,) = transform_scene_calls
    assert call["transformation_scale"] == draw.scene_transform["scale"]
    assert torch.equal(
        call["transformation_rotation"],
        rotation_from_degrees(
            draw.scene_transform["rot_x_deg"], draw.scene_transform["rot_y_deg"]
        ),
    )
    assert sample.track_upscaling_factor == 1.0 / draw.scene_transform["scale"]


# ----------------------------------------------------------- determinism ---


def test_the_same_settings_give_the_same_plan_and_a_longer_run_the_same_prefix():
    short = _sampled(50).planned
    assert short == _sampled(50).planned
    assert _sampled(200).planned[:50] == short
    assert _sampled(50, seed=4).planned != short
    assert _sampled(50, p=0.5).planned != short


def test_every_sampled_plan_is_plan_records_verdict_on_its_own_record():
    """The sampler decides nothing the planner does not; it only builds records.

    Rebuilding each plan's record and planning it again with main()'s arguments
    must give the identical plan, and the sampled window is the loader's clip.
    """

    tally = _sampled(40)
    assert len(tally.planned) == 40
    assert tally.skipped == []
    assert tally.skip_counts == {}
    assert tally.draws == 40
    for index, plan in enumerate(tally.planned):
        assert plan.step == index
        assert plan.frame_start == 0
        assert plan.seq_len == LOADER_CLIP_FRAMES == 24
        assert plan.real_len == 4
        assert plan.track_indices == ()
        assert plan.depth_type == "gt"
        assert plan.data_root == "/pool"
        record = build_record(
            step=plan.step,
            data_root=plan.data_root,
            seq_name=plan.seq_name,
            views=plan.cameras,
            frame_start=plan.frame_start,
            scene_transform=plan.scene_transform,
            dataset_seed=3,
            real_len=plan.real_len,
        )
        assert plan_record(record, **PLAN_KWARGS) == plan
    # Times are the planner's own budget arithmetic at the drawn view count.
    assert all(len(plan.times) == min(48 // len(plan.cameras), 12) for plan in tally.planned)


def test_a_record_carries_every_schema_key_and_nothing_else():
    record = build_record(
        step=0, data_root="/pool", seq_name="0000", views=(2, 0),
        frame_start=0, scene_transform=None,
    )
    assert set(record) == set(MANIFEST_RECORD_KEYS)
    assert record["manifest_version"] == MANIFEST_VERSION
    assert record["seq_len"] == LOADER_CLIP_FRAMES
    assert record["views"] == [2, 0]
    assert record["track_indices"] == []
    assert record["real_len"] is None and record["sample_index"] is None


def test_the_sampled_clip_is_the_held_out_clip():
    """One loader fact, imported by the trainer rather than written twice."""

    assert train_cli.DEFAULT_VAL_SEQ_LEN == LOADER_CLIP_FRAMES


def test_a_resume_at_step_k_continues_on_plans_k_onward(tmp_path):
    """The prefix property, driven through the loop: a run interrupted at step
    4 and resumed on a freshly sampled plan list trains the same (scene,
    cameras, transform) at steps 4-7 as the uninterrupted run."""

    seen_whole: list[tuple] = []
    seen_first: list[tuple] = []
    seen_second: list[tuple] = []

    def run(output_dir, args, seen, *, stop_at=None):
        model = _toy_model()
        optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])

        def step_fn(*, model, scene, plan, optimizer, scaler, learning_rates, step, **_):
            loss = model(torch.ones(1, 3)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            seen.append((step, plan.seq_name, plan.cameras, plan.scene_transform))
            if stop_at is not None and step == stop_at:
                train_cli._request_stop(signal.SIGUSR1, None)
            return train_cli.StepOutcome(
                step=step, seq_name=plan.seq_name, loss=float(loss.item()),
                metric_error_m=0.0, sample_count=1, alignment_scale=1.0,
                alignment_residual_m=0.0, learning_rates=list(learning_rates),
                gradient_norms={},
            )

        return train_cli.run_training(
            model=model, optimizer=optimizer,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=_sampled(8).planned, args=args,
            scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
            step_fn=step_fn, output_dir=output_dir,
        )

    train_cli._STOP_REQUESTED.clear()
    run(tmp_path / "whole", _loop_args(tmp_path, num_steps=8, scene_sampler=_identity()), seen_whole)
    run(tmp_path / "a", _loop_args(tmp_path, num_steps=8, scene_sampler=_identity()), seen_first, stop_at=3)
    train_cli._STOP_REQUESTED.clear()
    checkpoint = tmp_path / "a" / "train_state.pt"
    assert checkpoint.is_file() and len(seen_first) == 4

    run(
        tmp_path / "b",
        _loop_args(tmp_path, num_steps=8, resume=str(checkpoint), scene_sampler=_identity()),
        seen_second,
    )
    assert [entry[0] for entry in seen_second] == [4, 5, 6, 7]
    assert seen_second == seen_whole[4:]
    assert seen_first == seen_whole[:4]

    # The same checkpoint under another sampler seed is a different stream.
    with pytest.raises(RuntimeError, match="scene_sampler"):
        run(
            tmp_path / "c",
            _loop_args(
                tmp_path, num_steps=8, resume=str(checkpoint),
                scene_sampler=_identity(sampler_seed=4),
            ),
            [],
        )


# --------------------------------------------------- like-for-like check ---


def test_the_record_builder_reproduces_a_manifests_planned_rows_field_for_field():
    records = [
        _record(step=0, seq_name="0731", views=[3, 1, 0, 2]),
        _record(step=1, seq_name="0004", views=[2, 3], scene_transform=None),
        _record(step=2, seq_name="1200", views=[1, 2, 3, 4, 5, 0]),
    ]
    result = compare_sampler_against_manifest(records, **PLAN_KWARGS)
    assert result == {"compared": 3, "mismatched": [], "inexpressible": {}}

    reference = plan_manifest(records, **PLAN_KWARGS).planned
    for record, plan in zip(records, reference):
        rebuilt = plan_record(
            build_record(
                step=record["step"], data_root=record["data_root"],
                seq_name=record["seq_name"], views=record["views"],
                frame_start=record["frame_start"],
                scene_transform=record["scene_transform"],
            ),
            **PLAN_KWARGS,
        )
        assert isinstance(rebuilt, StepPlan)
        for name in WINDOW_FIELDS:
            assert getattr(rebuilt, name) == getattr(plan, name), name


def test_a_row_the_sampler_cannot_express_is_reported_by_field_or_by_cause():
    records = [
        _record(step=0),
        _record(step=1, seq_len=20),
        _record(step=2, data_root=None),
        _record(step=3, views=[1]),
    ]
    result = compare_sampler_against_manifest(records, **PLAN_KWARGS)
    assert result["compared"] == 2
    assert result["inexpressible"] == {SKIP_UNREPLAYABLE: 1, SKIP_TOO_FEW_VIEWS: 1}
    ((step, seq_name, differences),) = result["mismatched"]
    assert (step, seq_name) == (1, "0731")
    assert differences["seq_len"] == (20, 24)
    assert "times" in differences


def test_sampler_from_manifest_passes_and_fails_on_the_command_line(
    tmp_path, monkeypatch, capsys
):
    good = _write_manifest(
        tmp_path / "good.jsonl", [_record(step=i, seq_name=f"{i:04d}") for i in range(3)]
    )
    monkeypatch.setattr(
        sys, "argv", ["train_temporal_tracking.py", "--sampler_from_manifest", str(good)]
    )
    train_cli.main()
    printed = capsys.readouterr().out
    assert "PASS" in printed and "all 3 planned rows" in printed

    bad = _write_manifest(tmp_path / "bad.jsonl", [_record(step=0), _record(step=1, seq_len=20)])
    monkeypatch.setattr(
        sys, "argv", ["train_temporal_tracking.py", "--sampler_from_manifest", str(bad)]
    )
    with pytest.raises(SystemExit) as excinfo:
        train_cli.main()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "MISMATCH step=1" in captured.out and "seq_len" in captured.out
    assert "FAIL" in captured.err


# ----------------------------------------------------------- cache key ---


def test_the_cache_key_separates_plans_differing_only_in_the_transform():
    base = plan_record(_record(), **PLAN_KWARGS)
    same = plan_record(_record(), **PLAN_KWARGS)
    other = plan_record(
        _record(scene_transform={"scale": 1.2, "rot_x_deg": 0.0, "rot_y_deg": 0.0}),
        **PLAN_KWARGS,
    )
    none = plan_record(_record(scene_transform=None), **PLAN_KWARGS)
    key = train_cli.SceneCache.key
    assert key(base) == key(same)
    assert key(base) != key(other)
    assert key(base) != key(none)
    assert key(none) == key(plan_record(_record(scene_transform={}), **PLAN_KWARGS))

    loads = []
    cache = train_cli.SceneCache(lambda plan: loads.append(plan) or SimpleNamespace(views=[]), size=1)
    cache.get(base)
    cache.get(other)
    assert len(loads) == 2, "a changed transform must reload, not hit"


# ---------------------------------------------------------------- resume ---


def test_the_sampler_identity_is_a_refused_resume_key_with_a_none_default(tmp_path):
    assert "scene_sampler" in train_cli._RESUME_SETTINGS_REFUSED
    assert "scene_sampler" not in train_cli._RESUME_SETTINGS_WARNED
    assert train_cli._RESUME_SETTINGS_ABSENT_DEFAULTS["scene_sampler"] is None

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path, scene_sampler=_identity()))
    assert stored["scene_sampler"] == _identity()
    train_cli.check_resume_settings(stored, _loop_args(tmp_path, scene_sampler=_identity()))
    for change in (
        {"sampler_seed": 4},
        {"view_count_weights": "1:1.0"},
        {"transform_probability": 0.5},
        {"index_hash": "other"},
        {"pool": "/elsewhere"},
    ):
        with pytest.raises(RuntimeError, match="scene_sampler"):
            train_cli.check_resume_settings(
                stored, _loop_args(tmp_path, scene_sampler=_identity(**change))
            )
    # Pool against manifest, both ways.
    with pytest.raises(RuntimeError, match="scene_sampler"):
        train_cli.check_resume_settings(stored, _loop_args(tmp_path))
    with pytest.raises(RuntimeError, match="scene_sampler"):
        train_cli.check_resume_settings(
            train_cli._checkpoint_settings(_loop_args(tmp_path)),
            _loop_args(tmp_path, scene_sampler=_identity()),
        )

    # A checkpoint from before the key resumes in manifest mode and is refused
    # under a pool, by the absent-default rule.
    del stored["scene_sampler"]
    train_cli.check_resume_settings(stored, _loop_args(tmp_path))
    with pytest.raises(RuntimeError, match="predates 'scene_sampler'"):
        train_cli.check_resume_settings(stored, _loop_args(tmp_path, scene_sampler=_identity()))


def test_the_identity_round_trips_through_the_checkpoint_file(tmp_path):
    from test_trainer_loop import _run

    train_cli._STOP_REQUESTED.clear()
    _run(tmp_path, _loop_args(tmp_path, num_steps=4, scene_sampler=_identity()), [])
    settings = read_trainer_state(tmp_path / "train_state.pt")["settings"]
    assert settings["scene_sampler"] == _identity()
    assert isinstance(settings["scene_sampler"], dict)


def test_sampler_identity_is_plain_types_from_the_index():
    index = _index({"0000": 10})
    identity = sampler_identity(
        pool=Path("/pool"), index=index, sampler_seed=3,
        view_count_weights="2:0.1", transform_probability=1,
    )
    assert identity == {
        "pool": "/pool", "index_hash": "in-memory", "sampler_seed": 3,
        "view_count_weights": "2:0.1", "transform_probability": 1.0,
    }
    assert isinstance(identity["transform_probability"], float)


# ------------------------------------------------------------ the index ---


def test_build_pool_index_lists_what_the_loader_lists(tmp_path, capsys):
    pool = write_pool(
        tmp_path / "pool",
        {"0010": 4, "0002": 10, "0003": 3, "0001": 2, "0007": 0},
        loose={"0003"},
        zero_view={"0007"},
    )
    (pool / "_trackcache").mkdir()
    (pool / ".hidden.zip").write_bytes(b"not a scene")
    (pool / "notes.txt").write_text("ignored: neither an archive nor a directory")

    document, zero_view = build_pool_index(pool)
    assert [scene["name"] for scene in document["scenes"]] == ["0001", "0002", "0003", "0010"]
    assert [scene["view_count"] for scene in document["scenes"]] == [2, 10, 3, 4]
    assert document["scene_count"] == 4
    assert document["pool_index_version"] == 1
    assert zero_view == ["0007"]
    assert "skipped" not in json.dumps(document)

    path = write_pool_index(pool, document)
    assert path == pool / POOL_INDEX_NAME
    index = read_pool_index(pool)
    assert [scene.name for scene in index.scenes] == ["0001", "0002", "0003", "0010"]
    assert index.content_hash == document["content_hash"]
    assert len(index.content_hash) == 64
    # The hash is over the scene list and nothing else: a rebuild agrees, a
    # changed view count does not.
    assert build_pool_index(pool)[0]["content_hash"] == document["content_hash"]
    other, _ = build_pool_index(write_pool(tmp_path / "other", {"0001": 2, "0002": 9, "0003": 3, "0010": 4}))
    assert other["content_hash"] != document["content_hash"]


def test_build_pool_index_refuses_what_the_loader_would_refuse(tmp_path):
    with pytest.raises(ScenePoolError, match="not integers"):
        build_pool_index(write_pool(tmp_path / "named", {"scene_a": 3}))
    pool = write_pool(tmp_path / "both", {"0001": 3})
    write_pool(pool, {"0001": 3}, loose={"0001"})
    with pytest.raises(ScenePoolError, match="both as a directory and as an archive"):
        build_pool_index(pool)
    empty = tmp_path / "empty_pool"
    empty.mkdir()
    with pytest.raises(ScenePoolError, match="holds no scenes"):
        build_pool_index(empty)
    with pytest.raises(ScenePoolError, match="not a directory"):
        build_pool_index(tmp_path / "missing")


def test_read_pool_index_refuses_a_missing_edited_or_foreign_index(tmp_path):
    pool = write_pool(tmp_path / "pool", {"0001": 3, "0002": 4})
    with pytest.raises(ScenePoolError, match="--build_pool_index") as excinfo:
        read_pool_index(pool)
    assert str(pool) in str(excinfo.value) and "does not exist" in str(excinfo.value)

    document, _ = build_pool_index(pool)
    path = write_pool_index(pool, document)

    edited = json.loads(path.read_text())
    edited["scenes"][0]["view_count"] = 9
    path.write_text(json.dumps(edited))
    with pytest.raises(ScenePoolError, match="content_hash does not match"):
        read_pool_index(pool)

    foreign = dict(document, pool_index_version=2)
    path.write_text(json.dumps(foreign))
    with pytest.raises(ScenePoolError, match="pool_index_version 2"):
        read_pool_index(pool)

    path.write_text("{not json")
    with pytest.raises(ScenePoolError, match="not readable JSON"):
        read_pool_index(pool)

    unsorted = dict(document, scenes=list(reversed(document["scenes"])))
    unsorted["content_hash"] = scene_sampler._scenes_hash(unsorted["scenes"])
    path.write_text(json.dumps(unsorted))
    with pytest.raises(ScenePoolError, match="loader's order"):
        read_pool_index(pool)


# ------------------------------------------------------------------- cli ---


def _parse(*argv):
    args = train_cli.build_arg_parser().parse_args(list(argv))
    train_cli._validate_args(args)
    return args


def test_the_pool_flags_resolve_their_defaults_and_canonical_forms():
    args = _parse("--scene_pool", "/pool", "--seed", "9")
    assert args.sampler_seed == 9
    assert args.transform_probability == 1.0
    assert args.view_count_weights == "2:0.1,3:0.1,4:0.25,5:0.1,6:0.25"
    assert args.scene_sampler is None
    assert args.num_steps == 20000

    args = _parse(
        "--scene_pool", "/pool", "--sampler_seed", "4", "--min_views", "1",
        "--view_count_weights", "1:0.20,2:0.10,3:0.10,4:0.25,5:0.10,6:0.25",
        "--transform_probability", "0.5",
    )
    assert args.sampler_seed == 4
    assert args.view_count_weights == "1:0.2,2:0.1,3:0.1,4:0.25,5:0.1,6:0.25"
    assert args.transform_probability == 0.5

    manifest = _parse("--manifest", "m.jsonl")
    assert manifest.scene_sampler is None
    assert manifest.sampler_seed is None and manifest.view_count_weights is None


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "exactly one of --manifest and --scene_pool"),
        (["--manifest", "m.jsonl", "--scene_pool", "/pool"], "exactly one of"),
        (["--sampler_from_manifest", "m.jsonl", "--manifest", "m.jsonl"], "takes neither"),
        (["--sampler_from_manifest", "m.jsonl", "--scene_pool", "/pool"], "takes neither"),
        (["--manifest", "m.jsonl", "--build_pool_index"], "--build_pool_index needs --scene_pool"),
        (["--manifest", "m.jsonl", "--sampler_seed", "3"], "--sampler_seed needs --scene_pool"),
        (["--manifest", "m.jsonl", "--view_count_weights", "2:1"], "--view_count_weights needs --scene_pool"),
        (["--manifest", "m.jsonl", "--transform_probability", "0.5"], "--transform_probability needs --scene_pool"),
        (["--scene_pool", "/pool", "--max_records", "3"], "--max_records needs --manifest"),
        (["--scene_pool", "/pool", "--exclude_data_root", "/pool"], "names --scene_pool itself"),
        (["--scene_pool", "/pool", "--transform_probability", "1.5"], "--transform_probability must be in"),
        (["--scene_pool", "/pool", "--transform_probability", "nan"], "--transform_probability must be in"),
        (["--scene_pool", "/pool", "--view_count_weights", "2:0.5,2:0.5"], "twice"),
        (["--scene_pool", "/pool", "--view_count_weights", "0:1"], "at least 1"),
        (["--scene_pool", "/pool", "--view_count_weights", "2:-1"], "finite and positive"),
        (["--scene_pool", "/pool", "--view_count_weights", "2:0"], "finite and positive"),
        (["--scene_pool", "/pool", "--view_count_weights", "2:abc"], "not a number"),
        (["--scene_pool", "/pool", "--view_count_weights", "2"], "not COUNT:WEIGHT"),
        (["--scene_pool", "/pool", "--view_count_weights", "2.5:1"], "not an integer"),
        (["--scene_pool", "/pool", "--view_count_weights", ""], "is empty"),
        (["--scene_pool", "/pool", "--view_count_weights", "2:1,"], "empty entry"),
        (["--scene_pool", "/pool", "--view_count_weights", MIX], r"\[1\] below --min_views 2"),
        (["--scene_pool", "/pool", "--view_count_weights", "7:1", "--observation_budget", "6"], r"\[7\] above --observation_budget 6"),
    ],
)
def test_the_pool_flags_are_refused_at_parse_time(argv, message):
    with pytest.raises(ValueError, match=message):
        _parse(*argv)


def test_the_weights_check_names_both_flags():
    with pytest.raises(ValueError, match="--view_count_weights.*--min_views 2") as excinfo:
        check_view_count_weights(parse_view_count_weights(MIX), min_views=2, budget=48)
    assert "[1]" in str(excinfo.value)
    check_view_count_weights(parse_view_count_weights(MIX), min_views=1, budget=48)
    assert format_view_count_weights({3: 0.10, 2: 0.10}) == "2:0.1,3:0.1"


def test_the_validator_fixture_still_validates_in_manifest_mode(tmp_path):
    args = _validator_args(tmp_path)
    train_cli._validate_args(args)
    assert args.scene_sampler is None


def test_a_manifest_plan_gains_exactly_the_scene_source_key(tmp_path, monkeypatch):
    """Manifest mode is unchanged: the pinned key sets of plan.json at 607d232,
    plus ``scene_source``, and the settings block untouched."""

    records = [_record(step=i, seq_name=f"{i:04d}") for i in range(3)]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    out = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys, "argv",
        ["train_temporal_tracking.py", "--manifest", str(manifest), "--plan_only", "--json_out", str(out)],
    )
    train_cli.main()
    summary = json.loads(out.read_text())

    before = {
        "considered_for_threshold", "distinct_scenes", "distinct_time_indices_per_step",
        "duplicate_track_ids", "manifest_version_expected", "observations_per_step",
        "planned_steps", "planned_steps_without_a_velocity_pair", "recorded_tracks_per_step",
        "records_read", "records_with_scene_transform", "settings", "skipped", "stride",
        "time_bound", "unreplayable_fraction", "views_per_step",
    }
    assert set(summary) == before | {"scene_source"}
    assert set(summary["settings"]) == {
        "adaptive_query_anchors", "camera_input", "confidence_alpha", "confidence_weight",
        "depth_input", "encoder_local_checkpointing", "eval_query_anchor_source",
        "excluded_data_roots", "grad_accum", "ground_truth_query_anchor", "max_time_indices",
        "merge_synchronized_slots", "min_views", "num_optimizer_steps", "observation_budget",
        "oracle_query_anchor", "point_weight", "query_anchor_source", "query_anchors",
        "realized_val_query_anchors", "refine_gamma", "refine_iters", "resolved_confidence_alpha",
        "stride", "sync_weight", "time_embedding_init", "time_embedding_init_scale",
        "velocity_weight", "warmup_optimizer_steps",
    }
    assert summary["scene_source"] == {"kind": "manifest", "manifest": str(manifest)}
    assert summary["planned_steps"] == 3


def test_the_pool_cli_end_to_end_reads_only_the_index(tmp_path, monkeypatch, capsys):
    """Build the index, then delete every scene: --plan_only must still pass,
    and its plan.json must say the run was sampled and from what."""

    pool = write_pool(
        tmp_path / "pool", {"0010": 4, "0002": 10, "0003": 3, "0001": 2, "0007": 0},
        loose={"0003"}, zero_view={"0007"},
    )
    monkeypatch.setattr(
        sys, "argv", ["train_temporal_tracking.py", "--scene_pool", str(pool), "--build_pool_index"]
    )
    train_cli.main()
    captured = capsys.readouterr()
    assert f"pool_index={pool / POOL_INDEX_NAME} scenes=4 content_hash=" in captured.out
    assert "WARNING scene '0007' has no views" in captured.err
    index = read_pool_index(pool)

    for entry in list(pool.iterdir()):
        if entry.name != POOL_INDEX_NAME:
            if entry.is_dir():
                for member in sorted(entry.rglob("*"), reverse=True):
                    member.unlink() if member.is_file() else member.rmdir()
                entry.rmdir()
            else:
                entry.unlink()
    assert [entry.name for entry in pool.iterdir()] == [POOL_INDEX_NAME]

    out = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_temporal_tracking.py", "--scene_pool", str(pool), "--plan_only",
            "--num_steps", "12", "--warmup_steps", "2", "--sampler_seed", "5",
            "--json_out", str(out),
        ],
    )
    train_cli.main()
    printed = capsys.readouterr().out
    # Four scenes of 2, 10, 3 and 4 views: the 2-view scene seats count 2, so
    # every scene is admissible and twelve steps are exactly three epochs.
    assert f"PASS planned 12 steps from pool {pool} (seed 5, 3 epoch(s) over 4 of 4 scenes)" in printed
    assert "scene_source=" in printed
    summary = json.loads(out.read_text())
    assert summary["planned_steps"] == 12
    assert summary["records_read"] == 12
    assert summary["skipped"] == {}
    assert summary["distinct_scenes"] == 4
    assert summary["records_with_scene_transform"] == 12
    assert set(int(k) for k in summary["views_per_step"]) <= {2, 3, 4, 5, 6}
    assert summary["scene_source"] == {
        "kind": "pool",
        "pool": str(pool),
        "index_hash": index.content_hash,
        "sampler_seed": 5,
        "view_count_weights": "2:0.1,3:0.1,4:0.25,5:0.1,6:0.25",
        "transform_probability": 1.0,
        "scene_count": 4,
        "admissible_scenes": 4,
        "epochs_drawn": 3,
    }
    assert summary["settings"]["min_views"] == 2

    # The same pool, same seed, longer: the same prefix (the plan.json of a
    # 20-step pass describes the first 12 steps of a 12-step run).
    tally = plan_pool(
        index, data_root=str(pool), num_steps=20, seed=5, weights=DEFAULT_WEIGHTS,
        transform_probability=1.0, **PLAN_KWARGS,
    )
    assert len(tally.planned) == 20
    assert Counter(len(plan.cameras) for plan in tally.planned[:12]) == Counter(
        {int(k): v for k, v in summary["views_per_step"].items()}
    )


def test_a_pool_without_an_index_is_refused_naming_the_build_command(tmp_path, monkeypatch, capsys):
    pool = write_pool(tmp_path / "pool", {"0001": 3})
    monkeypatch.setattr(
        sys, "argv", ["train_temporal_tracking.py", "--scene_pool", str(pool), "--plan_only"]
    )
    with pytest.raises(SystemExit) as excinfo:
        train_cli.main()
    assert excinfo.value.code == 2
    assert f"--scene_pool {pool} --build_pool_index" in capsys.readouterr().err


def test_a_pool_whose_scenes_seat_no_listed_count_is_refused_on_the_command_line(
    tmp_path, monkeypatch, capsys
):
    pool = write_pool(tmp_path / "pool", {"0001": 1, "0002": 1})
    write_pool_index(pool, build_pool_index(pool)[0])
    monkeypatch.setattr(
        sys, "argv", ["train_temporal_tracking.py", "--scene_pool", str(pool), "--plan_only"]
    )
    with pytest.raises(SystemExit) as excinfo:
        train_cli.main()
    assert excinfo.value.code == 2
    assert "no scene in the pool seats" in capsys.readouterr().err


def test_the_tally_subclass_keeps_the_planner_tally_contract():
    tally = SampledPlanTally()
    assert tally.planned == [] and tally.skipped == []
    assert tally.epochs_drawn == 0 and tally.draws == 0
    assert tally.considered == 0 and tally.threshold_skip_fraction == 0.0


# ------------------------------------------------ pinned by value (review) ---
#
# An independent review mutated the sampler (seed + 1, the transform's three
# values drawn before the gate, scale before the rotations, the epoch shuffle
# replaced by index order) and the suite above passed every time: it only ever
# compared the draw with itself. The stream is a contract across resume
# segments, so it is pinned by value here. Changing any of these numbers means
# every sampled stream changes, and every checkpoint's identity stays equal --
# so a change here is a decision, not a refactor.


def test_the_stream_for_a_seed_is_pinned_by_value():
    draws = _draws(8, index=_index({"0000": 10, "0001": 6, "0002": 3, "0003": 4}))
    assert [(draw.epoch, draw.seq_name, draw.views) for draw in draws] == [
        (0, "0003", (2, 1, 3, 0)),
        (0, "0001", (3, 2)),
        (0, "0000", (8, 6, 7, 4, 9, 0)),
        (0, "0002", (1, 2, 0)),
        (1, "0002", (2, 0, 1)),
        (1, "0001", (5, 1, 4, 0)),
        (1, "0003", (3, 2, 1, 0)),
        (1, "0000", (1, 5, 6, 7)),
    ]
    first = draws[0].scene_transform
    assert first["scale"] == pytest.approx(0.9450700146967308, abs=1e-12)
    assert first["rot_x_deg"] == pytest.approx(11.888792668003141, abs=1e-12)
    assert first["rot_y_deg"] == pytest.approx(-11.232440686084912, abs=1e-12)
    second = draws[1].scene_transform
    assert second["scale"] == pytest.approx(1.1448451475322543, abs=1e-12)
    assert second["rot_x_deg"] == pytest.approx(-8.532689747552435, abs=1e-12)
    assert second["rot_y_deg"] == pytest.approx(-4.382847843411215, abs=1e-12)
    # Epochs are shuffled: neither epoch is the index's order, and the two
    # orders differ from each other.
    assert [d.seq_name for d in draws[:4]] != ["0000", "0001", "0002", "0003"]
    assert [d.seq_name for d in draws[4:]] != ["0000", "0001", "0002", "0003"]
    assert [d.seq_name for d in draws[:4]] != [d.seq_name for d in draws[4:]]


def test_non_default_weights_and_probability_reach_the_plan_and_the_identity(
    tmp_path, monkeypatch, capsys
):
    """main() must hand the sampler the flags it recorded, not the defaults."""

    pool = write_pool(tmp_path / "pool", {"0000": 10, "0001": 6, "0002": 3, "0003": 4})
    write_pool_index(pool, build_pool_index(pool)[0])
    out = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_temporal_tracking.py", "--scene_pool", str(pool), "--plan_only",
            "--num_steps", "40", "--warmup_steps", "2", "--sampler_seed", "5",
            "--view_count_weights", "2:1,3:1", "--transform_probability", "0.5",
            "--json_out", str(out),
        ],
    )
    train_cli.main()
    assert "PASS planned 40 steps" in capsys.readouterr().out
    summary = json.loads(out.read_text())
    assert set(int(k) for k in summary["views_per_step"]) == {2, 3}
    assert 0 < summary["records_with_scene_transform"] < 40
    assert summary["scene_source"]["view_count_weights"] == "2:1.0,3:1.0"
    assert summary["scene_source"]["transform_probability"] == 0.5
    assert summary["scene_source"]["sampler_seed"] == 5
    identity = sampler_identity(
        pool=pool, index=read_pool_index(pool), sampler_seed=5,
        view_count_weights="2:1.0,3:1.0", transform_probability=0.5,
    )
    assert identity["transform_probability"] == 0.5
    assert {k: summary["scene_source"][k] for k in identity} == {**identity, "pool": str(pool)}


def test_an_epoch_that_plans_nothing_raises_instead_of_looping():
    with pytest.raises(ScenePoolError, match="planned none of them"):
        _sampled(2, excluded_data_roots=("/pool",))


def test_the_index_orders_scenes_numerically_not_lexically(tmp_path):
    pool = write_pool(tmp_path / "wide", {"1000": 3, "999": 4})
    document, _ = build_pool_index(pool)
    assert [scene["name"] for scene in document["scenes"]] == ["999", "1000"]
    path = write_pool_index(pool, document)
    assert [scene.name for scene in read_pool_index(pool).scenes] == ["999", "1000"]

    lexical = dict(document, scenes=sorted(document["scenes"], key=lambda s: s["name"]))
    assert [scene["name"] for scene in lexical["scenes"]] == ["1000", "999"]
    lexical["content_hash"] = scene_sampler._scenes_hash(lexical["scenes"])
    path.write_text(json.dumps(lexical))
    with pytest.raises(ScenePoolError, match="loader's order"):
        read_pool_index(pool)

    edited = dict(document, scene_count=3)
    path.write_text(json.dumps(edited))
    with pytest.raises(ScenePoolError, match="scene_count 3 disagrees"):
        read_pool_index(pool)


def test_the_diagnostic_feeds_the_rows_own_frame_start_and_compares_the_window():
    assert {"scene_transform", "frame_start", "cameras", "times", "seq_len", "step"} <= set(
        WINDOW_FIELDS
    )
    records = [_record(step=0, frame_start=4), _record(step=1, frame_start=0)]
    result = compare_sampler_against_manifest(records, **PLAN_KWARGS)
    assert result["mismatched"] == [] and result["compared"] == 2
    reference = plan_manifest(records, **PLAN_KWARGS).planned
    assert reference[0].frame_start == 4 and reference[0].times[0] == 4


def test_a_count_equal_to_the_budget_is_admissible_and_the_key_ignores_dict_order():
    check_view_count_weights({6: 1.0}, min_views=2, budget=6)
    with pytest.raises(ValueError, match=r"\[7\] above --observation_budget 6"):
        check_view_count_weights({7: 1.0}, min_views=2, budget=6)

    forward = plan_record(
        _record(scene_transform={"scale": 1.2, "rot_x_deg": 3.0, "rot_y_deg": -2.0}),
        **PLAN_KWARGS,
    )
    reordered = plan_record(
        _record(scene_transform={"rot_y_deg": -2.0, "rot_x_deg": 3.0, "scale": 1.2}),
        **PLAN_KWARGS,
    )
    assert train_cli.SceneCache.key(forward) == train_cli.SceneCache.key(reordered)


def test_build_pool_index_is_not_gated_by_the_sampling_checks():
    """An index is what a --min_views the weights cannot meet needs first."""

    args = _parse("--scene_pool", "/pool", "--build_pool_index", "--min_views", "7")
    assert args.build_pool_index and args.view_count_weights is None
    with pytest.raises(ValueError, match="below --min_views 7"):
        _parse("--scene_pool", "/pool", "--min_views", "7")


@pytest.mark.parametrize("seed", ["-1", "4294967296"])
def test_a_sampler_seed_outside_random_states_range_is_refused_by_name(seed):
    with pytest.raises(ValueError, match=f"--sampler_seed {seed} is outside"):
        _parse("--scene_pool", "/pool", "--sampler_seed", seed)
    with pytest.raises(ValueError, match="--sampler_seed"):
        _parse("--scene_pool", "/pool", "--seed", seed)
