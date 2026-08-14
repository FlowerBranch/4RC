"""Pins for the training loop: schedule, resume, cache, signals, guards.

The loop is driven end to end here with an injected step function and scene
provider. That is not a testing convenience bolted on: the scene source is
genuinely unbound until a cluster measurement decides it, and the same seam is
what the real provider plugs into. Injecting it means the planner, the schedule,
the cache, the checkpointing and the resume are all under test on CPU, which is
where every regression in this file was found.

The resume test is the one worth reading. A schedule that captures its base rates
*after* `optimizer.load_state_dict` restarts the cosine from the already-decayed
value and compounds it once per requeue — nothing raises, and the only symptom is
a loss curve that flattens on a long job. It is asserted at every step of the
resumed segment, because the final value alone hides it.
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import train_temporal_tracking as train_cli
from arc.training.schedule import (
    apply_learning_rate,
    capture_base_learning_rates,
    warmup_cosine_scale,
)
from arc.training.trainer_state import (
    build_trainer_state,
    read_trainer_state,
    restore_rng_state,
    save_atomically,
)
from test_manifest_plan import _record
from arc.training.manifest_plan import plan_record


# ----------------------------------------------------------------- schedule ---


def test_warmup_rises_then_cosine_decays_without_a_discontinuity():
    """The boundary is where an off-by-one shows, so it is checked directly."""

    values = [
        warmup_cosine_scale(step, warmup_steps=5, total_steps=100, min_lr_scale=0.0)
        for step in range(100)
    ]

    assert values[0] == pytest.approx(1 / 6)
    assert all(b > a for a, b in zip(values[:5], values[1:6])), "warmup must rise"
    assert values[5] == pytest.approx(1.0), "cosine starts at the full rate"
    assert all(b <= a for a, b in zip(values[5:], values[6:])), "then only decays"
    assert values[-1] < 0.01


def test_the_schedule_never_returns_zero_during_warmup():
    """A zero first step wastes a forward and a backward.

    At `warmup_steps=1` it would waste the only warmup step there is.
    """

    assert warmup_cosine_scale(0, warmup_steps=1, total_steps=10) > 0
    assert warmup_cosine_scale(0, warmup_steps=500, total_steps=20000) > 0


def test_steps_past_the_end_hold_the_floor_rather_than_turning_back_up():
    """A naive cosine rises again past total_steps; an overrun must not retrain hot."""

    beyond = warmup_cosine_scale(150, warmup_steps=5, total_steps=100, min_lr_scale=0.1)

    assert beyond == pytest.approx(0.1)


def test_apply_learning_rate_preserves_the_per_group_ratios():
    """The encoder group runs at a tenth of the decoder's, by design.

    A schedule that wrote one rate across all groups would erase the ratio the
    freeze presets depend on, silently and with no error.
    """

    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "lr": 1e-5}, {"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-6}]
    )
    base = capture_base_learning_rates(optimizer)

    applied = apply_learning_rate(optimizer, base, 0.25)

    assert applied == [pytest.approx(2.5e-6), pytest.approx(2.5e-7)]
    assert applied[0] / applied[1] == pytest.approx(10.0)


# -------------------------------------------------------------- trainer state ---


def test_the_resume_payload_survives_weights_only_and_stays_a_temporal_patch():
    """Measured hazard: `weights_only=True` rejects every numpy object.

    `np.random.get_state()` returns a uint32 ndarray, and one ndarray anywhere in
    the payload does not merely lose that field — it makes the whole file
    unreadable by `load_temporal_tracking_checkpoint`, i.e. it stops being a
    temporal patch at all. The decomposition to plain types is what prevents that,
    and this asserts the patch reader still parses the superset.
    """

    from arc.training.checkpoint import read_temporal_patch_metadata

    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-5}])
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()

    state = build_trainer_state(
        step=17,
        optimizer=optimizer,
        base_learning_rates=[1e-5],
        settings={"stride": 2},
    )
    payload = {
        "freeze_mode": "temporal_tracking",
        "state_dict": {"weight": model.weight.detach().cpu()},
        **state,
    }

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = save_atomically(payload, Path(directory) / "train_state.pt")
        restored = read_trainer_state(path)
        assert restored["step"] == 17
        # The reason the decomposition exists: this call is weights_only=True.
        assert read_temporal_patch_metadata(path)["freeze_mode"] == "temporal_tracking"


def test_rng_state_round_trips_across_all_four_streams():
    """A resume that restarts the data order is not a resume."""

    import random

    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = build_trainer_state(
        step=0,
        optimizer=torch.optim.AdamW([{"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-5}]),
        base_learning_rates=[1e-5],
    )

    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng_state(state["rng"])
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_a_bare_temporal_patch_is_refused_as_a_resume_source():
    """It has no optimizer and no step, so resuming from it restarts training.

    Silently, while reporting that it continued — which is the failure this
    message exists to prevent.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "patch.pt"
        torch.save({"freeze_mode": "temporal_tracking", "state_dict": {}}, path)

        with pytest.raises(RuntimeError, match="carries no trainer_state_version"):
            read_trainer_state(path)


# ---------------------------------------------------------------- the loop ---


@dataclass
class _Recorded:
    step: int
    scene_id: int
    lr: float


def _loop_args(tmp_path, **overrides):
    args = SimpleNamespace(
        observation_budget=48,
        stride=2,
        num_steps=8,
        warmup_steps=2,
        min_lr_scale=0.0,
        grad_clip=1.0,
        huber_delta_m=0.05,
        precision="32",
        save_every=0,
        resume=None,
        max_device_fraction=0.97,
        scene_cache=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _plans(count):
    return [
        plan_record(_record(step=i, seq_name=f"{i % 3:04d}"), budget=48, stride=2)
        for i in range(count)
    ]


def _toy_model():
    model = nn.Linear(3, 2)
    model.freeze = "temporal_tracking"
    return model


def _recording_step(recorded):
    def step_fn(*, model, scene, plan, optimizer, scaler, learning_rates, step, **_):
        # A real gradient so the optimizer state actually advances; without one,
        # resume equality would hold trivially.
        loss = model(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        recorded.append(_Recorded(step=step, scene_id=id(scene), lr=learning_rates[0]))
        return train_cli.StepOutcome(
            step=step,
            seq_name=plan.seq_name,
            loss=float(loss.item()),
            metric_error_m=0.0,
            sample_count=1,
            alignment_scale=1.0,
            alignment_residual_m=0.0,
            learning_rates=list(learning_rates),
            gradient_norms={},
        )

    return step_fn


def _run(tmp_path, args, recorded, plans=None):
    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, train_cli.run_training(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        plans=plans or _plans(4),
        args=args,
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=_recording_step(recorded),
        output_dir=tmp_path,
    )


def test_resume_reproduces_an_uninterrupted_run_at_every_step(tmp_path):
    """Work-order test 4, asserted per step rather than only at the end.

    The bug it exists for compounds: capturing `base_lrs` after
    `optimizer.load_state_dict` restarts the cosine from the decayed rate, so the
    effective rate decays as scale(N)**segments across requeues. On a short run
    the final weights barely differ, which is exactly why the final value is not
    what gets compared here.
    """

    train_cli._STOP_REQUESTED.clear()
    torch.manual_seed(0)
    whole: list[_Recorded] = []
    reference_model, _ = _run(tmp_path / "whole", _loop_args(tmp_path, num_steps=8), whole)

    # Interrupt an *8-step* run after 4 rather than running a 4-step one: the
    # cosine denominator is total_steps, so a shorter run is a different schedule
    # and would not be the same first half at all.
    torch.manual_seed(0)
    first: list[_Recorded] = []
    base_step = _recording_step(first)

    def interrupted_step(**kwargs):
        outcome = base_step(**kwargs)
        if kwargs["step"] == 3:
            train_cli._request_stop(signal.SIGUSR1, None)
        return outcome

    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])
    train_cli.run_training(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=8),
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=interrupted_step,
        output_dir=tmp_path / "b",
    )
    train_cli._STOP_REQUESTED.clear()
    checkpoint = tmp_path / "b" / "train_state.pt"
    assert checkpoint.is_file()
    assert len(first) == 4, "the interrupt must land after exactly four steps"

    second: list[_Recorded] = []
    resumed_model, _ = _run(
        tmp_path / "c",
        _loop_args(tmp_path, num_steps=8, resume=str(checkpoint)),
        second,
    )

    assert [r.step for r in first + second] == list(range(8))
    # The LR at every resumed step must match the uninterrupted run's.
    for whole_step, resumed_step in zip(whole[4:], second):
        assert resumed_step.step == whole_step.step
        assert resumed_step.lr == pytest.approx(whole_step.lr), (
            f"step {resumed_step.step}: resumed lr {resumed_step.lr} != "
            f"uninterrupted {whole_step.lr}; the cosine restarted"
        )
    for name, parameter in resumed_model.named_parameters():
        torch.testing.assert_close(
            parameter, dict(reference_model.named_parameters())[name],
            rtol=1e-6, atol=1e-6, msg=f"{name} diverged across the resume",
        )


def _stub_scene(name, *, times=(0, 2)):
    """A scene carrying the view keys the cache fingerprints."""

    return SimpleNamespace(
        name=name,
        views=[
            {
                "img": torch.zeros(1, 3, 4, 4),
                "time_index": torch.tensor([index]),
                "track_query_idx": torch.tensor([0]),
            }
            for index in range(len(times))
        ],
    )


def test_a_cache_hit_returns_the_same_scene_object_and_reloads_are_counted(tmp_path):
    """The efficiency half: a repeat window must not reload."""

    loads: list[tuple] = []

    def loader(plan):
        loads.append(train_cli.SceneCache.key(plan))
        return _stub_scene(plan.seq_name)

    cache = train_cli.SceneCache(loader, size=1)
    same = plan_record(_record(seq_name="0001"), budget=48, stride=2)
    other = plan_record(_record(seq_name="0002"), budget=48, stride=2)

    first = cache.get(same)
    again = cache.get(same)
    assert first is again, "a repeat window must not reload"
    assert cache.hits == 1 and cache.misses == 1
    assert len(loads) == 1

    cache.get(other)
    assert len(loads) == 2
    # And coming back evicts, because the cache holds one window.
    cache.get(same)
    assert len(loads) == 3


def test_a_step_that_mutates_a_cached_scene_is_caught_at_the_next_visit(tmp_path):
    """Work-order test 7's real content: a cache hit must not differ from a reload.

    Returning the identical object is what makes the cache worth having and is
    also exactly how an in-place mutation reaches the next step — so the two
    properties are only compatible if mutation is prevented or detected. A
    defensive copy per hit would cost the transfer the cache exists to avoid, and
    silently repairing the damage would hide it, so the fingerprint detects it and
    the run fails loudly at the next visit.

    The `_move_views_to_cuda` framing the work order used is vacuous here: on CPU
    that call is either skipped or a no-op, so it can never demonstrate the
    hazard. Rebinding a view tensor, which is what it does on a GPU, can.
    """

    plan = plan_record(_record(seq_name="0001"), budget=48, stride=2)
    cache = train_cli.SceneCache(lambda p: _stub_scene(p.seq_name), size=1)

    scene = cache.get(plan)
    assert cache.get(plan) is scene, "unmutated repeats stay cheap"

    # Exactly what move_views_to_cuda does on a real device: rebind the tensor.
    scene.views[0]["img"] = torch.ones(1, 3, 4, 4)

    with pytest.raises(RuntimeError, match="was mutated in place"):
        cache.get(plan)


def test_a_run_ending_on_a_save_boundary_does_not_write_the_same_step_twice(
    tmp_path, monkeypatch
):
    """The final write is a multi-GB write to shared storage; it must earn itself.

    Content-identical through an atomic replace, so repeating it is harmless —
    but it happens at the end of every run whose length is a multiple of
    --save_every, which is the common case for a scheduled job.
    """

    written: list[int] = []
    real = train_cli._write_checkpoint

    def counting(*args, **kwargs):
        written.append(kwargs["step"])
        return real(*args, **kwargs)

    monkeypatch.setattr(train_cli, "_write_checkpoint", counting)

    model = _toy_model()
    train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=8, save_every=4),
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=_recording_step([]),
        output_dir=tmp_path,
    )

    assert written == [4, 8], f"expected one write per boundary, got {written}"


def test_a_run_ending_off_a_save_boundary_still_checkpoints_its_last_step(
    tmp_path, monkeypatch
):
    """The other half: skipping the duplicate must not skip a needed write."""

    written: list[int] = []
    real = train_cli._write_checkpoint
    monkeypatch.setattr(
        train_cli,
        "_write_checkpoint",
        lambda *a, **k: (written.append(k["step"]), real(*a, **k))[1],
    )

    model = _toy_model()
    train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=7, save_every=4),
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=_recording_step([]),
        output_dir=tmp_path,
    )

    assert written == [4, 7]


def test_the_fingerprint_notices_a_written_index_not_only_a_rebind(tmp_path):
    """An in-place write leaves the object identity alone.

    `view["time_index"][0] = ...` rebinds nothing, so an identity-only check would
    miss it — and the time index is precisely what decides which embedding row an
    observation reads.
    """

    plan = plan_record(_record(seq_name="0001"), budget=48, stride=2)
    cache = train_cli.SceneCache(lambda p: _stub_scene(p.seq_name), size=1)

    scene = cache.get(plan)
    scene.views[1]["time_index"][0] = 99

    with pytest.raises(RuntimeError, match="was mutated in place"):
        cache.get(plan)


def test_the_cache_key_separates_two_windows_on_one_scene(tmp_path):
    """Scene name alone is not a key.

    Cameras and times vary per step and the correspondences derive from them, so
    two windows on one scene are different objects. Keying on the name would hand
    the second step the first's window with the right scene name attached.
    """

    wide = plan_record(_record(views=[0, 1, 2, 3]), budget=48, stride=2)
    narrow = plan_record(_record(views=[0, 1]), budget=48, stride=2)

    assert wide.seq_name == narrow.seq_name
    assert train_cli.SceneCache.key(wide) != train_cli.SceneCache.key(narrow)


# ----------------------------------------------------------------- signals ---


def test_a_stop_signal_checkpoints_and_leaves_the_loop_early(tmp_path):
    """SLURM sends USR1 at T-300s; the run must land a resumable file and exit 0.

    The handler only sets a flag — writing a multi-GB checkpoint from inside a
    signal context could land mid-backward.
    """

    train_cli._STOP_REQUESTED.clear()
    recorded: list[_Recorded] = []

    def stopping_step(**kwargs):
        outcome = _recording_step(recorded)(**kwargs)
        if kwargs["step"] == 2:
            train_cli._request_stop(signal.SIGUSR1, None)
        return outcome

    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])
    result = train_cli.run_training(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=50),
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=stopping_step,
        output_dir=tmp_path,
    )
    train_cli._STOP_REQUESTED.clear()

    assert result["interrupted_by"] == "SIGUSR1"
    assert result["completed_steps"] == 3, "the step in flight completes, then it stops"
    assert (tmp_path / "train_state.pt").is_file()


def test_an_interrupted_run_reports_no_verdict_rather_than_a_pass():
    """`all([])` is True, so a tri-state is the only honest answer here.

    Without it a run stopped at step 3 of 100,000 writes `gates_passed: true` and
    downstream automation believes it.
    """

    interrupted = train_cli._gate_verdicts(
        {"interrupted_by": "SIGUSR1", "completed_steps": 3, "planned_steps": 100}
    )
    finished = train_cli._gate_verdicts(
        {"interrupted_by": None, "completed_steps": 100, "planned_steps": 100}
    )

    assert interrupted["gates_passed"] is None
    assert interrupted["gates_evaluated"] == 0
    assert finished["gates_passed"] is True


# ------------------------------------------------------------ device guard ---


def test_the_headroom_guard_is_inert_without_cuda_and_names_the_step_with_it(monkeypatch):
    """At 94% occupancy the guard is the difference between a number and an OOM.

    On this CPU-only box it must not fire at all; the message is checked by
    faking a device so the assertion is not merely "it did nothing".
    """

    train_cli.check_device_headroom(10**12, step=7, max_fraction=0.5)  # no CUDA: inert

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=100 * 2**30),
    )

    train_cli.check_device_headroom(50 * 2**30, step=7, max_fraction=0.9)
    with pytest.raises(RuntimeError, match=r"step 7: peak .* over --max_device_fraction"):
        train_cli.check_device_headroom(95 * 2**30, step=7, max_fraction=0.9)
