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

import dataclasses
import json
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
from arc.training.predictions import PREDICTION_KEYS
from arc.training.scene_provider import SceneProviderError
from arc.training.manifest_plan import StepPlan
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
        # Present so a val_plans-carrying run reaches the eval instead of dying on
        # a missing attribute; 0 keeps every existing test's behaviour unchanged.
        eval_every=0,
        max_scene_skip_fraction=0.02,
        max_consecutive_scene_skips=10,
        # _checkpoint_settings reads these at every checkpoint write and resume.
        min_views=2,
        max_time_indices=32,
        exclude_data_root=[],
        kubric_max_depth=24.0,
        # The canonical anchor spec: _checkpoint_settings stores it and the
        # eval call site records it. The default is the single-anchor spec
        # every existing test's behaviour assumes.
        query_anchors=["0:0"],
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


def test_a_resume_that_changes_the_replayed_stream_is_refused(tmp_path):
    """A changed budget or stride re-plans a different stream under a continued
    step counter; the loop must refuse before restoring any state."""

    train_cli._STOP_REQUESTED.clear()
    _run(tmp_path / "a", _loop_args(tmp_path, num_steps=4), [])
    checkpoint = tmp_path / "a" / "train_state.pt"
    assert checkpoint.is_file()

    with pytest.raises(RuntimeError, match="stride") as excinfo:
        _run(
            tmp_path / "b",
            _loop_args(tmp_path, num_steps=4, resume=str(checkpoint), stride=4),
            [],
        )
    assert "stride=2" in str(excinfo.value) and "stride=4" in str(excinfo.value)


def test_a_resume_that_extends_the_run_warns_and_continues(tmp_path, capsys):
    """Raising num_steps is how a finished run is extended; it must proceed,
    but say so, because the cosine's denominator changes with it."""

    train_cli._STOP_REQUESTED.clear()
    _run(tmp_path / "a", _loop_args(tmp_path, num_steps=4), [])
    checkpoint = tmp_path / "a" / "train_state.pt"

    second: list[_Recorded] = []
    _run(
        tmp_path / "b",
        _loop_args(tmp_path, num_steps=6, resume=str(checkpoint)),
        second,
    )

    error_output = capsys.readouterr().err
    assert "num_steps=4" in error_output and "num_steps=6" in error_output
    assert [r.step for r in second] == [4, 5]


def test_a_checkpoint_without_stored_settings_is_refused(tmp_path):
    """A refused-tier key absent from the checkpoint refuses the resume.

    The tolerance this replaces let a pre-field checkpoint resume under a
    different anchor spec -- or budget, or stride -- unchecked. The only
    checkpoints predating the newest key are disposable short smokes, so
    refusing loses nothing worth resuming.
    """

    with pytest.raises(RuntimeError, match="carries no stored"):
        train_cli.check_resume_settings({}, _loop_args(tmp_path))

    # A partial dict is refused too, naming the specific missing key.
    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    del stored["query_anchors"]
    with pytest.raises(RuntimeError, match="query_anchors"):
        train_cli.check_resume_settings(stored, _loop_args(tmp_path))

    # Warned-tier keys stay tolerant when absent: they only reshape the
    # remaining schedule.
    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    del stored["num_steps"]
    train_cli.check_resume_settings(stored, _loop_args(tmp_path))


def test_the_settings_refusal_names_both_values(tmp_path):
    stored = train_cli._checkpoint_settings(_loop_args(tmp_path, precision="bf16-mixed"))
    with pytest.raises(RuntimeError) as excinfo:
        train_cli.check_resume_settings(stored, _loop_args(tmp_path, precision="32"))
    assert "bf16-mixed" in str(excinfo.value)
    assert "'32'" in str(excinfo.value)


def test_the_refusal_covers_the_mapped_and_coerced_settings(tmp_path):
    """excluded_data_roots is stored under the plan summary's name rather than
    the args attribute, and kubric_max_depth as a float; both are as
    stream-defining as stride, so both must still be compared."""

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path, exclude_data_root=["/old"]))
    with pytest.raises(RuntimeError) as excinfo:
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, exclude_data_root=["/new"])
        )
    assert "/old" in str(excinfo.value) and "/new" in str(excinfo.value)

    # 1000 is what MVTracker's loader defaults to when built without training
    # args -- exactly the silent drift the flag exists to prevent.
    stored = train_cli._checkpoint_settings(_loop_args(tmp_path, kubric_max_depth=1000.0))
    with pytest.raises(RuntimeError, match="kubric_max_depth"):
        train_cli.check_resume_settings(stored, _loop_args(tmp_path))


def test_a_resume_that_changes_the_anchor_spec_is_refused(tmp_path):
    """The spec is stream-defining: it decides which observations each step
    supervises and how the loss reduces over them, so a changed -- or reordered
    -- spec under a continued step counter is a different run. Order matters:
    the first anchor owns the Sim(3) and assignment tiebreaks on anchor index."""

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, query_anchors=["0:0", "1:0"])
        )
    assert "['0:0']" in str(excinfo.value)
    assert "['0:0', '1:0']" in str(excinfo.value)

    reordered = train_cli._checkpoint_settings(
        _loop_args(tmp_path, query_anchors=["1:0", "0:0"])
    )
    with pytest.raises(RuntimeError, match="query_anchors"):
        train_cli.check_resume_settings(
            reordered, _loop_args(tmp_path, query_anchors=["0:0", "1:0"])
        )

    # The matching spec continues.
    train_cli.check_resume_settings(
        train_cli._checkpoint_settings(_loop_args(tmp_path, query_anchors=["0:0", "1:0"])),
        _loop_args(tmp_path, query_anchors=["0:0", "1:0"]),
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


# ------------------------------------------------------------- the real step ---


class _FakeArc(nn.Module):
    """The submodule surface ``train_step`` reaches for, and nothing else.

    Every other loop test injects ``step_fn``, so the real ``train_step`` body was
    never executed -- which is how it shipped calling
    ``build_anchor_correspondences`` without unpacking its ``(correspondences,
    eligibility)`` tuple. A stub that produces a differentiable ``track_multi``
    from real parameters is enough to run the body end to end on CPU, gradient
    guards included.

    The step drives the decomposed surface (``_preprocess_input`` /
    ``encode_features`` / ``reconstruct`` / ``track_for_query``) while the eval
    drives ``forward``, so ``forward`` is composed from the pieces exactly the
    way ``Arc._forward`` composes them -- otherwise the two paths could drift
    apart inside this fake while both kept passing.
    """

    def __init__(self, observations, height, width):
        super().__init__()
        self.observations, self.height, self.width = observations, height, width
        self.head = nn.Linear(1, 1)
        self.cam_dec = nn.Linear(1, 1)
        self.motion_decoder = nn.Linear(1, 1)
        self.track_head = nn.Linear(1, 1)
        embedding = nn.Module()
        embedding.time_index_embedding = nn.Embedding(8, 2)
        pretrained = nn.Module()
        pretrained.pretrained = embedding
        self.backbone = pretrained
        # The reconstruction and camera heads are frozen under every temporal
        # preset. Leaving them trainable here would trip
        # assert_trainable_gradients_finite on parameters the real model never
        # asks for a gradient on -- the guard is right, the fake was wrong.
        self.head.requires_grad_(False)
        self.cam_dec.requires_grad_(False)

    def _preprocess_input(self, views):
        images = torch.zeros(1, self.observations, 3, self.height, self.width)
        # The real preprocessor reads the anchor slot list off the views; the
        # fake mirrors that so a multi-anchor scene reaches the Q loop.
        return images, views[0]["track_query_idx"], None

    def encode_features(self, images, ref_view_strategy="first", time_indices=None):
        # Every trainable parameter must receive a gradient or train_step's own
        # guards fire -- which is part of what is being tested, so the "taps"
        # touch biases as well as weights.
        scale = (
            self.motion_decoder.weight.sum()
            + self.motion_decoder.bias.sum()
            + self.track_head.weight.sum()
            + self.track_head.bias.sum()
            + self.backbone.pretrained.time_index_embedding.weight.sum()
        )
        return [scale]

    def reconstruct(self, feats, images):
        return {
            "depth": torch.ones(1, self.observations, self.height, self.width),
            "pose_enc": torch.zeros(1, self.observations, 9),
        }

    def track_for_query(self, feats, images, query_idx):
        # (track, confidence) tuple, shaped (1,S,H,W,3) / (1,S,H,W) exactly as
        # the real head returns them -- anchor_tracks adds the Q=1 axis.
        track = (
            torch.ones(1, self.observations, self.height, self.width, 3) * feats[0]
        )
        confidence = torch.ones(1, self.observations, self.height, self.width)
        return track, confidence

    def forward(self, views, force_no_output_conversion=False):
        images, track_query_idx, time_indices = self._preprocess_input(views)
        feats = self.encode_features(images, time_indices=time_indices)
        output = self.reconstruct(feats, images)
        query_slots = [
            int(value)
            for value in torch.as_tensor(track_query_idx).flatten().tolist()
        ]
        tracks, confidences = zip(
            *(self.track_for_query(feats, images, slot) for slot in query_slots)
        )
        output["track_multi"] = torch.stack(tracks, dim=1)
        output["conf_track_multi"] = torch.stack(confidences, dim=1)
        output["track_query_idx"] = torch.tensor(query_slots, dtype=torch.long)
        return output


def test_the_real_train_step_runs_end_to_end_on_cpu(tmp_path, monkeypatch):
    """Covers the body every other loop test injects past.

    Not a numerical check -- it asserts the step completes, produces a finite
    loss and reports the guards it ran. That is enough to catch the class of
    defect that actually occurred: an API on the `arc/training` side changing
    shape under a call site nothing executes.
    """

    import arc.training.sparse_tracking as sparse_module
    from arc.training import load_dumped_kubric_scene
    from test_sparse_tracking import _write_scene

    _write_scene(tmp_path, time_count=4, view_count=2, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path, "0000", cameras=(0, 1), times=(0, 1, 2, 3), size=56
    )

    # Identity alignment, the same lever the sparse-tracking tests use: hand
    # fit_scene_sim3 the scene's own metric pointmap so the fit is exact and the
    # step's arithmetic is what is under test, not the geometry fixture.
    target, _ = sparse_module._metric_pointmap_at_anchor(
        scene, scene.query_observation_slot
    )
    pointmaps = torch.from_numpy(target).float().expand(
        1, scene.num_observations, *target.shape
    ).contiguous()
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)

    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)
    optimizer = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": 1e-3}]
    )
    plan = plan_record(_record(seq_name="0000"), budget=48, stride=2)

    outcome = train_cli.train_step(
        model=model,
        scene=scene,
        plan=plan,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        precision="32",
        huber_delta_m=0.05,
        grad_clip=1.0,
        learning_rates=[1e-3],
        step=0,
    )

    assert outcome.step == 0
    assert outcome.seq_name == "0000"
    assert np.isfinite(outcome.loss)
    assert outcome.sample_count > 0
    # The guards ran and found gradient on every trainable group.
    for group in ("time_embedding", "motion_decoder", "track_head"):
        assert outcome.gradient_norms[group] > 0, group
    assert "clipped_total" in outcome.gradient_norms
    # The step reports its eligibility split for the run-level aggregation.
    assert outcome.eligibility["total_query_count"] > 0
    assert (
        outcome.eligibility["eligible_query_count"]
        + sum(outcome.eligibility["rejected"].values())
        == outcome.eligibility["total_query_count"]
    )


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
        {"interrupted_by": "SIGUSR1", "completed_steps": 3, "target_steps": 100}
    )
    finished = train_cli._gate_verdicts(
        {"interrupted_by": None, "completed_steps": 100, "target_steps": 100}
    )

    assert interrupted["gates_passed"] is None
    assert interrupted["gates_evaluated"] == 0
    assert finished["gates_passed"] is True


def test_the_gate_targets_num_steps_not_the_manifests_planned_count():
    """--num_steps below the planned count is the normal case (20k against a
    ~50k-record manifest): the loop targets range(start_step, num_steps), so a
    run that finished every step it was asked for must not report a failed gate.
    Regression: the gate compared completed_steps against planned_steps."""

    finished_short = train_cli._gate_verdicts(
        {"interrupted_by": None, "completed_steps": 20000, "target_steps": 20000}
    )
    unfinished = train_cli._gate_verdicts(
        {"interrupted_by": None, "completed_steps": 19999, "target_steps": 20000}
    )

    assert finished_short["gates_passed"] is True
    assert unfinished["gates_passed"] is False


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


# ------------------------------------------------------ the eval, end to end ---


def _cpu_eval_scene(tmp_path, monkeypatch):
    """A real two-camera window plus identity alignment, ready for the eval.

    Two cameras is required rather than tidy: at one camera
    ``shuffled_index_views`` returns ``None`` early and the index-advantage arm
    never runs, which is where one of the two bugs this test exists for lives.
    """

    import arc.training.sparse_tracking as sparse_module
    from arc.training import load_dumped_kubric_scene
    from test_sparse_tracking import _write_scene

    _write_scene(tmp_path, time_count=4, view_count=2, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path, "0000", cameras=(0, 1), times=(0, 1, 2, 3), size=56
    )
    target, _ = sparse_module._metric_pointmap_at_anchor(
        scene, scene.query_observation_slot
    )
    pointmaps = torch.from_numpy(target).float().expand(
        1, scene.num_observations, *target.shape
    ).contiguous()
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)
    return scene


def test_the_real_held_out_eval_runs_end_to_end_on_cpu(tmp_path, monkeypatch):
    """The test that would have caught both eval-fatal bugs.

    ``evaluate_held_out`` had no test at all, and shipped calling an unimported
    ``capture_rng_state`` and passing two arguments to a one-argument
    ``shuffled_index_views``. Neither is subtle; both survived a 339-test suite
    because nothing ever executed the function. `main()` is `.to("cuda")`
    unconditionally, so no CPU test can reach it from above -- this calls the
    function directly, the way the train_step test does.
    """

    scene = _cpu_eval_scene(tmp_path, monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)
    plan = plan_record(_record(seq_name="0000"), budget=48, stride=2)

    metrics = train_cli.evaluate_held_out(
        model=model,
        plans=[plan],
        scene_provider=lambda _plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=7,
        output_dir=tmp_path / "out",
        query_anchors=["0:0"],
    )

    assert metrics["step"] == 7 and metrics["scenes"] == 1
    # Not None at two cameras: a None here would mean the index-advantage arm was
    # skipped, which is how the wrong-arity call could hide.
    assert metrics["position_loss_shuffled"] is not None
    assert metrics["per_scene"][0]["scene"] == "0000"

    directory = tmp_path / "out" / "eval" / "step-7"
    written = json.loads((directory / "metrics.json").read_text())
    assert written["scenes"] == 1
    # The spec the eval scored under, so a curve is never read against the
    # wrong supervision scheme.
    assert written["query_anchors"] == ["0:0"]
    loaded = np.load(directory / "pred" / "0000.npz")
    assert set(loaded.files) == set(PREDICTION_KEYS)


def test_the_eval_restores_rng_and_module_modes(tmp_path, monkeypatch):
    """Work-order test 5's mechanism, at the level it actually operates.

    The eval must leave no trace: it consumes randomness and calls
    ``model.eval()``, and the step loop deliberately leaves ``head``/``cam_dec``
    in eval() while the root trains. Restoring the root alone would silently
    re-enable the frozen heads' training behaviour for every step after the
    first eval.
    """

    scene = _cpu_eval_scene(tmp_path, monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)
    model.train()
    model.head.eval()
    model.cam_dec.eval()
    before = {name: module.training for name, module in model.named_modules()}
    torch.manual_seed(1234)
    state = torch.random.get_rng_state()

    train_cli.evaluate_held_out(
        model=model,
        plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
        scene_provider=lambda _plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=1,
        output_dir=tmp_path / "out",
        query_anchors=["0:0"],
        emit_predictions=False,
    )

    assert {name: module.training for name, module in model.named_modules()} == before
    assert torch.equal(torch.random.get_rng_state(), state)


# ---------------------------------------------------------- the device seam ---


def test_the_cuda_wrapper_moves_the_views_when_cuda_is_present(monkeypatch):
    """F3 was "a function is never called", so this asserts a call, not a device.

    Asserting tensor placement would be vacuous on a CPU box, where cpu == cpu
    passes whether or not the move happened.
    """

    moved = []
    monkeypatch.setattr(train_cli, "move_views_to_cuda", moved.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    scene = SimpleNamespace(views=[{"img": torch.zeros(1)}])

    loaded = train_cli.cuda_scene_provider(lambda plan: scene)(object())

    assert loaded is scene, "the cache stores and fingerprints whatever this returns"
    assert len(moved) == 1 and moved[0] is scene.views


def test_the_cuda_wrapper_is_a_no_op_without_cuda(monkeypatch):
    """Which is what keeps every CPU test working once the wrap is wired in."""

    moved = []
    monkeypatch.setattr(train_cli, "move_views_to_cuda", moved.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    scene = SimpleNamespace(views=[{"img": torch.zeros(1)}])

    assert train_cli.cuda_scene_provider(lambda plan: scene)(object()) is scene
    assert moved == []


def test_run_training_wraps_its_provider(tmp_path, monkeypatch):
    """The wiring, which the two tests above cannot see.

    If ``run_training`` stopped wrapping, both of them would still pass -- the
    move is skipped on CPU either way. Faking availability is what makes the
    branch reachable here, so this pins the call site rather than arguing about
    it.
    """

    moved = []
    monkeypatch.setattr(train_cli, "move_views_to_cuda", moved.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])
    # A distinct views list per scene, so the count below cannot be satisfied by
    # one scene being moved repeatedly.
    train_cli.run_training(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=4),
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name, views=[]),
        step_fn=_recording_step([]),
        output_dir=tmp_path,
    )

    assert len(moved) == 4, "one move per scene load, through the loop's own cache"


# ------------------------------------------------- scene-load failure policy ---


def _failing_provider(bad_names):
    def load(plan):
        if plan.seq_name in bad_names:
            raise SceneProviderError(
                f"scene {plan.seq_name!r} is not in the pool at '/root' (0 scenes)"
            )
        return SimpleNamespace(name=plan.seq_name, views=[])

    return load


def test_one_unloadable_scene_skips_its_step_without_ending_the_run(tmp_path):
    """Over 4956 scenes a single bad one must not end a two-day allocation.

    And the step counter still ADVANCES: the whole premise is a curve that
    overlays MVTracker's step for step, so putting the next plan's scene at this
    step would desynchronise the two runs permanently. Losing one sample does not.
    """

    recorded = []
    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])
    plans = _plans(4)

    result = train_cli.run_training(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=plans,
        args=_loop_args(tmp_path, num_steps=4),
        scene_provider=_failing_provider({plans[1].seq_name}),
        step_fn=_recording_step(recorded),
        output_dir=tmp_path,
    )

    assert result["completed_steps"] == 4, "the run reaches its full step count"
    assert result["scene_load_skips"] == {"scene_absent": 1}
    # The surviving steps keep their ORIGINAL numbers -- 1 is missing, not
    # backfilled by what would otherwise have been step 2.
    assert [entry.step for entry in recorded] == [0, 2, 3]


def test_a_wholly_unloadable_pool_aborts_with_the_tally(tmp_path):
    """Skipping must not become "silently do nothing for two days".

    This is the shape a wrong --data_root takes, and the shape an inherited
    max_videos=30 cap took: every step advancing, no gradients, a summary that
    reads as a completed run.
    """

    plans = _plans(4)
    model = _toy_model()
    with pytest.raises(RuntimeError, match=r"consecutive steps could not load"):
        train_cli.run_training(
            model=model,
            optimizer=torch.optim.AdamW(
                [{"params": list(model.parameters()), "lr": 1e-3}]
            ),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=plans,
            args=_loop_args(tmp_path, num_steps=4, max_consecutive_scene_skips=3),
            scene_provider=_failing_provider({plan.seq_name for plan in plans}),
            step_fn=_recording_step([]),
            output_dir=tmp_path,
        )


def test_the_skip_causes_are_distinguished(tmp_path):
    """A single "failed" count reports a broken root and a bad scene alike."""

    absent = SceneProviderError("scene '0001' is not in the pool at '/root' (0 scenes)")
    small = SceneProviderError("scene '0002': only 3 eligible tracks, below ...")
    empty = SceneProviderError("scene '0003': ... leaves nothing to supervise")

    assert train_cli.scene_skip_cause(absent) == "scene_absent"
    assert train_cli.scene_skip_cause(small) == "pool_too_small"
    assert train_cli.scene_skip_cause(empty) == "no_recorded_tracks_present"
    assert train_cli.scene_skip_cause(SceneProviderError("unrecognised")) == "other"


def test_val_scenes_without_a_val_root_is_refused_at_parse_time(tmp_path, monkeypatch, capsys):
    """Refused through the CLI, not just by the function that would have raised.

    Two things this pins that a direct call on the helper cannot. It must be a
    clean ``parser.error`` -- exit 2 with a message -- rather than an unhandled
    traceback, which is what it was when the check lived where the plans are
    built, outside main()'s ValueError handler. And it must fire BEFORE
    ``Arc.from_pretrained(...).to("cuda")``, or a cluster job burns a full model
    load to be told a flag is missing. Reaching that load would need a manifest,
    a checkpoint dir and a GPU, none of which exist here -- so the SystemExit
    arriving at all is the proof that nothing downstream ran.
    """

    scenes = tmp_path / "val.json"
    scenes.write_text(json.dumps(["0000"]))
    monkeypatch.setattr(
        sys, "argv",
        ["train_temporal_tracking.py", "--manifest", str(tmp_path / "m.jsonl"),
         "--val_scenes_file", str(scenes)],
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    assert "--val_scenes_file needs --val_data_root" in capsys.readouterr().err


def test_the_val_flag_pair_is_checked_by_the_argument_validator(tmp_path):
    """And the unit underneath, so the message itself is pinned."""

    args = _loop_args(
        tmp_path, val_scenes_file="val.json", val_data_root=None,
        min_views=2, max_time_indices=32, max_unreplayable_fraction=0.02,
        max_records=None, val_cameras=[0, 1, 2, 3],
    )

    with pytest.raises(ValueError, match="--val_scenes_file needs --val_data_root"):
        train_cli._validate_args(args)

    args.val_data_root = "/held/out/dir"
    train_cli._validate_args(args)


def test_the_depth_clip_defaults_to_the_paired_runs_value_and_is_a_float():
    """The flag exists so the mirrored value is visible, not buried in a dict.

    24 is not this project's number -- it is configs/train.yaml's
    datasets.train.kubric_max_depth in the mvtracker checkout. A run that has to
    match a different paired run needs to say so on the command line, and a run
    that matched should be able to prove it from its own artifacts.
    """

    args = train_cli.build_arg_parser().parse_args(["--manifest", "m.jsonl"])
    assert args.kubric_max_depth == 24.0
    assert isinstance(args.kubric_max_depth, float)

    override = train_cli.build_arg_parser().parse_args(
        ["--manifest", "m.jsonl", "--kubric_max_depth", "12.5"]
    )
    assert override.kubric_max_depth == 12.5


def test_the_held_out_set_is_proved_reachable_before_the_first_step(tmp_path):
    """A bad held-out root must die at step 0, not at the first eval.

    evaluate_held_out loads through its own call, outside the step loop's skip
    policy, so an unreachable val scene ends the run -- at --eval_every 500 that
    is hours into a two-day allocation. The preflight also exercises the CUDA
    move against the real provider before any training time is spent.
    """

    loaded = []

    def provider(plan):
        loaded.append(plan.seq_name)
        if plan.seq_name == "val-bad":
            raise SceneProviderError(f"scene {plan.seq_name!r} is not in the pool")
        return SimpleNamespace(name=plan.seq_name, views=[])

    val = [
        StepPlan(
            step=-1, seq_name=name, data_root="/val", cameras=(0, 1), times=(0, 2),
            frame_start=0, seq_len=4, stride=2, time_bound="budget",
            track_indices=(), scene_transform=None, depth_type=None,
            real_len=None,
        )
        for name in ("val-bad",)
    ]
    model = _toy_model()

    with pytest.raises(SceneProviderError, match="not in the pool"):
        train_cli.run_training(
            model=model,
            optimizer=torch.optim.AdamW(
                [{"params": list(model.parameters()), "lr": 1e-3}]
            ),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=_plans(4),
            args=_loop_args(tmp_path, num_steps=4),
            scene_provider=provider,
            step_fn=_recording_step([]),
            output_dir=tmp_path,
            val_plans=val,
        )

    # Before any training step ran, which is the whole point.
    assert loaded == ["val-bad"]


def test_a_scene_failing_on_an_eval_boundary_still_produces_its_curve_point(
    tmp_path, monkeypatch
):
    """The hole S4 could have punched in the thing it was protecting.

    Skipping a step must not skip that step's *boundaries*. The whole deliverable
    is a held-out curve that overlays MVTracker's every --eval_every steps, so a
    scene failing to load exactly on a boundary would drop that point silently --
    a gap in the one output that carries the result, with nothing saying why. The
    eval scores the model against held-out scenes and does not depend on the
    training scene that failed, so there is no reason to lose it.

    The earlier skip tests assert step numbers and would all pass with the
    boundary skipped; this puts the failure on the boundary deliberately, and
    drives the REAL evaluate_held_out rather than a stub, so the point it
    produces is a real one.
    """

    scene = _cpu_eval_scene(tmp_path, monkeypatch)
    plans = _plans(4)
    failing = plans[1].seq_name

    def provider(plan):
        if plan.seq_name == failing:
            raise SceneProviderError(f"scene {plan.seq_name!r} is not in the pool")
        return scene

    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)
    result = train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": 1e-3}]
        ),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=plans,
        # The distinctive spec pins that the loop hands the eval THE RUN'S
        # spec rather than a constant: the eval records it verbatim, and the
        # scene's own anchors are the provider's business, so the mismatch
        # with this test's single-anchor scenes is deliberate.
        args=_loop_args(tmp_path, num_steps=4, eval_every=2, query_anchors=["3:1"]),
        scene_provider=provider,
        # The model here is a _FakeArc, for the real eval; the recording step_fn
        # expects the toy model, so this test supplies its own no-op step. What
        # is under test is which boundaries fire, not what a step computes.
        step_fn=lambda *, step, plan, **_: train_cli.StepOutcome(
            step=step, seq_name=plan.seq_name, loss=0.0, metric_error_m=0.0,
            sample_count=1, alignment_scale=1.0, alignment_residual_m=0.0,
            learning_rates=[1e-3], gradient_norms={},
        ),
        output_dir=tmp_path,
        val_plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
    )

    # Step 1 failed to load, and completed == 2 is an eval boundary.
    assert result["scene_load_skips"] == {"scene_absent": 1}
    assert [entry["step"] for entry in result["evaluations"]] == [2, 4], (
        "the boundary at 2 must survive its own step's skip"
    )
    written = json.loads(
        (tmp_path / "eval" / "step-2" / "metrics.json").read_text()
    )
    assert written["query_anchors"] == ["3:1"], (
        "the loop must record the run's own spec, not a constant"
    )


# ----------------------------------------- multi-anchor query supervision ---


def _step_scene(tmp_path, monkeypatch, *, query_anchors=None, **scene_kwargs):
    """A real dumped-fixture scene with identity alignment, for the real step."""

    import arc.training.sparse_tracking as sparse_module
    from arc.training import load_dumped_kubric_scene
    from test_sparse_tracking import _write_scene

    _write_scene(
        tmp_path, time_count=4, view_count=2, depth_sidecar=True, **scene_kwargs
    )
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
        query_anchors=query_anchors,
    )
    target, _ = sparse_module._metric_pointmap_at_anchor(
        scene, scene.query_observation_slot
    )
    pointmaps = torch.from_numpy(target).float().expand(
        1, scene.num_observations, *target.shape
    ).contiguous()
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)
    return scene


def test_the_restructured_step_matches_the_combined_forward_at_one_anchor(
    tmp_path, monkeypatch
):
    """The default-spec invariant, pinned: same loss, same post-step weights.

    The per-anchor step decomposes the forward into encode -> reconstruct ->
    track_for_query and skips the cut at one active anchor, which is claimed to
    be the exact pre-multi-anchor graph. This drives the real ``train_step``
    against an in-test replica of the old combined pipeline on an identically
    initialized copy, and requires equality -- not closeness -- of the loss and
    of every parameter after the optimizer step.
    """

    import copy

    from arc.training import (
        build_anchor_correspondences,
        fit_scene_sim3,
        gather_query_anchor_points,
        sparse_tracking_loss,
    )
    from arc.training.runtime import tracking_only

    scene = _step_scene(tmp_path, monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    stepped = _FakeArc(scene.num_observations, height, width)
    reference = copy.deepcopy(stepped)

    outcome = train_cli.train_step(
        model=stepped,
        scene=scene,
        plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
        optimizer=torch.optim.AdamW(
            [{"params": list(stepped.parameters()), "lr": 1e-3}]
        ),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        precision="32",
        huber_delta_m=0.05,
        grad_clip=1.0,
        learning_rates=[1e-3],
        step=0,
    )

    # The combined pipeline exactly as the step ran it before multi-anchor.
    reference.train()
    reference.head.eval()
    reference.cam_dec.eval()
    reference_optimizer = torch.optim.AdamW(
        [{"params": list(reference.parameters()), "lr": 1e-3}]
    )
    reference_optimizer.zero_grad(set_to_none=True)
    correspondences, _ = build_anchor_correspondences(scene)
    raw = reference(scene.views, force_no_output_conversion=True)
    alignment, _ = fit_scene_sim3(raw, scene)
    anchors = gather_query_anchor_points(raw, scene, correspondences)
    result = sparse_tracking_loss(
        tracking_only(raw),
        scene,
        correspondences,
        alignment,
        anchors,
        huber_delta_m=0.05,
        collect_diagnostics=False,
    )
    result.total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in reference.parameters() if p.requires_grad], 1.0
    )
    reference_optimizer.step()

    assert outcome.loss == float(result.loss.item())
    assert outcome.sample_count == int(result.sample_count)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in stepped.named_parameters():
        assert torch.equal(parameter, reference_parameters[name]), name


def test_a_two_anchor_step_runs_end_to_end_on_the_dumped_fixture(
    tmp_path, monkeypatch
):
    """The per-anchor path, on a scene where the second anchor earns its keep.

    Track 2 is occluded in camera 0 at time 0, so only the camera-1 anchor can
    supervise it -- the recovery the second anchor exists for. The cut engages
    (two active anchors), every gradient guard still fires, and the outcome's
    counts stay consistent with the per-anchor split.
    """

    import copy

    from arc.training import (
        build_anchor_correspondences,
        fit_scene_sim3,
        gather_query_anchor_points,
        sparse_tracking_loss,
    )
    from arc.training.runtime import (
        anchor_sample_counts,
        anchor_tracks,
        encode_and_reconstruct,
        tracking_only,
    )

    scene = _step_scene(
        tmp_path,
        monkeypatch,
        query_anchors=((0, 0), (1, 0)),
        invisible=((0, 0, 2),),
    )
    assert scene.query_anchors == ((0, 0), (1, 0))
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)

    # The expected weighted loss, from the pre-step weights: each anchor's mean
    # times its share of the supervised samples, accumulated in anchor order --
    # exactly what makes per-anchor backward equal one combined mean. Computed
    # on an identical copy BEFORE train_step moves the parameters, so the
    # assertion below is exact, and it fails if the step ever weights anchors
    # by anything other than their sample counts.
    reference = copy.deepcopy(model)
    correspondences, _ = build_anchor_correspondences(scene)
    counts = anchor_sample_counts(scene, correspondences, 2)
    images, feats, recon = encode_and_reconstruct(reference, scene.views)
    reference_alignment, _ = fit_scene_sim3(recon, scene)
    anchor_points = gather_query_anchor_points(recon, scene, correspondences)
    expected_loss = None
    for anchor_index in range(2):
        raw = anchor_tracks(reference, feats, images, scene, anchor_index)
        result = sparse_tracking_loss(
            tracking_only(raw),
            scene,
            correspondences.select_query_slot(anchor_index),
            reference_alignment,
            anchor_points[correspondences.anchor_rows(anchor_index)],
            huber_delta_m=0.05,
            collect_diagnostics=False,
        )
        contribution = float(result.loss.item()) * (counts[anchor_index] / sum(counts))
        expected_loss = (
            contribution if expected_loss is None else expected_loss + contribution
        )

    outcome = train_cli.train_step(
        model=model,
        scene=scene,
        plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
        optimizer=torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": 1e-3}]
        ),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        precision="32",
        huber_delta_m=0.05,
        grad_clip=1.0,
        learning_rates=[1e-3],
        step=0,
    )

    _, eligibility = build_anchor_correspondences(scene)
    assert all(count > 0 for count in counts), (
        "both anchors must supervise something for this test to test the cut"
    )
    assert np.isfinite(outcome.loss)
    assert outcome.loss == expected_loss, (
        "the step's loss must be the sample-count-weighted sum of the "
        "per-anchor means, in anchor order"
    )
    assert outcome.sample_count == sum(counts)
    for group in ("time_embedding", "motion_decoder", "track_head"):
        assert outcome.gradient_norms[group] > 0, group
    assert eligibility["per_anchor"][1]["assigned"] >= 1, (
        "the occluded track must be recovered by the second anchor"
    )
    assert (
        outcome.eligibility["eligible_query_count"]
        + sum(outcome.eligibility["rejected"].values())
        == outcome.eligibility["total_query_count"]
    )


def test_the_eval_runs_at_two_anchors_end_to_end(tmp_path, monkeypatch):
    """The Q=2 eval path, which a real multi-anchor run hits at every boundary.

    The eval keeps the combined forward (no_grad needs no cut), so a two-anchor
    run is the first time anywhere that Q=2 flows through the model call, the
    combined-correspondence loss, the shuffled-index arm and the prediction
    writer. evaluate_held_out has already shipped two eval-fatal bugs precisely
    because nothing executed it; this keeps the multi-anchor variant from
    repeating that history.
    """

    scene = _step_scene(
        tmp_path,
        monkeypatch,
        query_anchors=((0, 0), (1, 0)),
        invisible=((0, 0, 2),),
    )
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)

    metrics = train_cli.evaluate_held_out(
        model=model,
        plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
        scene_provider=lambda _plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=3,
        output_dir=tmp_path / "out",
        query_anchors=["0:0", "1:0"],
    )

    assert metrics["scenes"] == 1
    assert metrics["position_loss_shuffled"] is not None
    entry = metrics["per_scene"][0]
    # confidence_stats at Q=2 switches to the per-anchor form: pooled mean,
    # None quantiles, one summary per anchor -- documented in runtime.
    assert len(entry["confidence"]["per_anchor"]) == 2
    assert entry["confidence"]["p50"] is None

    directory = tmp_path / "out" / "eval" / "step-3"
    written = json.loads((directory / "metrics.json").read_text())
    assert written["query_anchors"] == ["0:0", "1:0"]
    loaded = np.load(directory / "pred" / "0000.npz")
    assert set(loaded.files) == set(PREDICTION_KEYS)


def test_a_zero_supervision_scene_fails_the_step_loudly(tmp_path, monkeypatch):
    """An anchor set that reaches nothing must raise, citing the split.

    Anchoring only at time 2 while every fixture query starts at time 0 rejects
    everything at query_time_mismatch. This was fatal before multi-anchor too --
    deep inside the loss, after the forward; now it raises before any GPU work,
    and it must NOT be a SceneProviderError, which the loop's skip policy would
    absorb as one bad scene.
    """

    scene = _step_scene(tmp_path, monkeypatch, query_anchors=((0, 2),))
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)

    with pytest.raises(RuntimeError, match="No anchor contributes") as excinfo:
        train_cli.train_step(
            model=model,
            scene=scene,
            plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
            optimizer=torch.optim.AdamW(
                [{"params": list(model.parameters()), "lr": 1e-3}]
            ),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            precision="32",
            huber_delta_m=0.05,
            grad_clip=1.0,
            learning_rates=[1e-3],
            step=0,
        )
    assert "query_time_mismatch" in str(excinfo.value)
    assert not isinstance(excinfo.value, SceneProviderError)


def test_a_malformed_anchor_pair_is_refused_at_parse_time():
    # int() alone would admit "+1:0", " 1 : 2 ", unicode digits and "1_0:0"
    # (slot TEN) -- each either misleading or a slot the user never typed, so
    # the parse takes plain ASCII decimals only.
    for bad in (
        ["0"],
        ["0:0:0"],
        ["a:0"],
        ["-1:0"],
        ["0:-1"],
        ["+1:0"],
        ["1_0:0"],
        [" 1 : 2 "],
        [],
    ):
        with pytest.raises(ValueError):
            train_cli.parse_query_anchor_slots(bad)
    # "00:0" parses to the same slot as "0:0": a duplicate, not a new anchor.
    with pytest.raises(ValueError, match="more than once"):
        train_cli.parse_query_anchor_slots(["0:0", "00:0"])
    # Order is preserved, never sorted: the first anchor owns the Sim(3).
    assert train_cli.parse_query_anchor_slots(["1:0", "0:0"]) == ((1, 0), (0, 0))


def _validator_args(tmp_path, **overrides):
    """_loop_args plus the flags only _validate_args reads."""

    merged = {
        "max_unreplayable_fraction": 0.02,
        "max_records": None,
        "val_scenes_file": None,
        "val_data_root": None,
        "val_cameras": [0, 1, 2, 3],
    }
    merged.update(overrides)
    return _loop_args(tmp_path, **merged)


def test_an_anchor_view_slot_min_views_cannot_guarantee_is_refused_at_parse_time(
    tmp_path,
):
    args = _validator_args(tmp_path, query_anchors=["0:0", "2:0"], min_views=2)
    with pytest.raises(ValueError, match="--min_views >= 3"):
        train_cli._validate_args(args)

    # The same spec is fine once min_views guarantees the slot on every step.
    args = _validator_args(tmp_path, query_anchors=["0:0", "2:0"], min_views=3)
    train_cli._validate_args(args)


def test_the_anchor_spec_is_canonicalized_format_only_and_order_preserving(tmp_path):
    args = _validator_args(tmp_path, query_anchors=["01:02", "1:0", "0:0"])
    train_cli._validate_args(args)
    assert args.query_anchors == ["1:2", "1:0", "0:0"], (
        "zero-padding normalized, order untouched"
    )


def test_an_anchor_the_val_window_cannot_seat_is_refused_at_parse_time(tmp_path):
    args = _validator_args(
        tmp_path,
        query_anchors=["1:0"],
        val_scenes_file="val.json",
        val_data_root="/held",
        val_cameras=[0],
        min_views=2,
    )
    with pytest.raises(ValueError, match="held-out window"):
        train_cli._validate_args(args)

    # 48 observations over 4 val cameras seat 12 times; slot 12 does not exist.
    args = _validator_args(
        tmp_path,
        query_anchors=["0:12"],
        val_scenes_file="val.json",
        val_data_root="/held",
    )
    with pytest.raises(ValueError, match="held-out window"):
        train_cli._validate_args(args)


def test_an_anchor_time_slot_a_planned_step_cannot_seat_is_refused_under_plan_only(
    tmp_path, monkeypatch, capsys
):
    """Caught at submit time, GPU-free, naming the step that cannot seat it.

    A step's window length is budget // views, capped by the record's own
    window -- so a time slot can fit every step but one, and that one would
    otherwise fail hours into an allocation.
    """

    from test_manifest_plan import _write_manifest

    records = [
        _record(step=0, seq_name="0000"),
        # seq_len 2 at stride 2 seats exactly one time, so time slot 1 cannot.
        _record(step=1, seq_name="0001", seq_len=2),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_temporal_tracking.py",
            "--manifest",
            str(manifest),
            "--plan_only",
            "--query_anchors",
            "0:0",
            "0:1",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    error_output = capsys.readouterr().err
    assert "0:1" in error_output and "'0001'" in error_output


def test_the_default_anchor_spec_plans_exactly_as_before(tmp_path, monkeypatch, capsys):
    """--plan_only with defaults passes and records the single-anchor spec."""

    from test_manifest_plan import _write_manifest

    records = [_record(step=index, seq_name=f"{index:04d}") for index in range(2)]
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

    assert "PASS planned 2 steps" in capsys.readouterr().out
    assert json.loads(out.read_text())["settings"]["query_anchors"] == ["0:0"]


def test_the_anchor_spec_is_recorded_in_the_checkpoint_settings(tmp_path):
    train_cli._STOP_REQUESTED.clear()
    _run(tmp_path / "a", _loop_args(tmp_path, num_steps=4), [])
    payload = read_trainer_state(tmp_path / "a" / "train_state.pt")
    assert payload["settings"]["query_anchors"] == ["0:0"]
    assert isinstance(payload["settings"]["query_anchors"], list)

    # And the wiring end to end: resuming under a different spec is refused
    # before any state is restored.
    with pytest.raises(RuntimeError, match="query_anchors"):
        _run(
            tmp_path / "b",
            _loop_args(
                tmp_path,
                num_steps=4,
                resume=str(tmp_path / "a" / "train_state.pt"),
                query_anchors=["0:0", "1:0"],
            ),
            [],
        )


def test_eligibility_counts_sum_over_executed_steps_into_the_totals(tmp_path):
    """The run-level split the 2-anchor question is answered from.

    Summed over executed steps only: a skipped scene load contributes nothing,
    so the totals stay consistent with steps_counted as their denominator.
    """

    plans = _plans(4)
    failing = plans[1].seq_name

    def provider(plan):
        if plan.seq_name == failing:
            raise SceneProviderError(f"scene {plan.seq_name!r} is not in the pool")
        return SimpleNamespace(name=plan.seq_name, views=[])

    def step_fn(*, step, plan, **_):
        return train_cli.StepOutcome(
            step=step,
            seq_name=plan.seq_name,
            loss=0.0,
            metric_error_m=0.0,
            sample_count=1,
            alignment_scale=1.0,
            alignment_residual_m=0.0,
            learning_rates=[1e-3],
            gradient_norms={},
            eligibility={
                "total_query_count": 10,
                "eligible_query_count": 6,
                "rejected": {"query_time_mismatch": 3, "not_visible_in_anchor": 1},
            },
        )

    model = _toy_model()
    result = train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": 1e-3}]
        ),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=plans,
        args=_loop_args(tmp_path, num_steps=4),
        scene_provider=provider,
        step_fn=step_fn,
        output_dir=tmp_path,
    )

    totals = result["eligibility_totals"]
    assert totals["steps_counted"] == 3, "the skipped step contributes nothing"
    assert totals["total_query_count"] == 30
    assert totals["eligible_query_count"] == 18
    assert totals["rejected"]["query_time_mismatch"] == 9
    assert totals["rejected"]["not_visible_in_anchor"] == 3
    # Every stage is present with an explicit zero, so the summary's split is
    # readable without knowing the stage list by heart.
    assert set(totals["rejected"]) == set(train_cli.ELIGIBILITY_REJECTION_STAGES)
    assert totals["rejected"]["projection"] == 0


def test_steps_without_eligibility_reports_leave_the_totals_at_zero(tmp_path):
    """Injected step functions that report no split must not fake one."""

    train_cli._STOP_REQUESTED.clear()
    _, result = _run(tmp_path, _loop_args(tmp_path, num_steps=4), [])

    assert result["eligibility_totals"] == {
        "steps_counted": 0,
        "total_query_count": 0,
        "eligible_query_count": 0,
        "rejected": dict.fromkeys(train_cli.ELIGIBILITY_REJECTION_STAGES, 0),
    }


# ------------------------------- an unsupervisable held-out scene ---


def test_an_unsupervisable_held_out_scene_is_skipped_and_recorded(
    tmp_path, monkeypatch
):
    """The eval's disposition is the opposite of train_step's, deliberately.

    Anchoring only at time 2 while every fixture query starts at time 0 rejects
    everything at query_time_mismatch, leaving an empty correspondence set. That
    set used to reach `gather_query_anchor_points`, whose `rows.min()` is
    unguarded -- so the scene died with a torch reduction error naming neither
    the scene nor the reason, and the exception escaped `run_training`, skipping
    the post-loop checkpoint and the summary. One bad held-out scene must not end
    a multi-day run, so it is skipped and recorded instead.
    """

    scene = _step_scene(tmp_path, monkeypatch, query_anchors=((0, 2),))
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)

    metrics = train_cli.evaluate_held_out(
        model=model,
        plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
        scene_provider=lambda _plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=5,
        output_dir=tmp_path / "out",
        query_anchors=["0:2"],
    )

    # Scored zero scenes, and said so rather than raising.
    assert metrics["scenes"] == 0
    assert metrics["per_scene"] == []
    # None, not 0.0: an average over nothing is an absence, not a measurement.
    assert metrics["position_loss"] is None
    assert metrics["metric_error_m"] is None

    skipped = metrics["skipped_scenes"]
    assert [entry["scene"] for entry in skipped] == ["0000"]
    assert skipped[0]["reason"] == "no_eligible_correspondences"
    # The split has to travel with the skip, or a reader cannot tell an anchor
    # spec that reaches nothing from a held-out set that is simply missing.
    split = skipped[0]["eligibility"]
    assert split["eligible_query_count"] == 0
    assert split["rejected"]["query_time_mismatch"] == split["total_query_count"] > 0

    # And it reaches the file, which is the only thing a later reader has.
    written = json.loads(
        (tmp_path / "out" / "eval" / "step-5" / "metrics.json").read_text()
    )
    assert written["skipped_scenes"] == skipped
    assert written["scenes"] == 0
    # No prediction bundle for a scene that was never scored.
    assert not (tmp_path / "out" / "eval" / "step-5" / "pred").exists()


def test_one_unsupervisable_held_out_scene_leaves_the_others_scored(
    tmp_path, monkeypatch
):
    """The skip is per scene: the rest of the held-out set still makes a point."""

    good = _cpu_eval_scene(tmp_path / "good", monkeypatch)
    bad = _step_scene(tmp_path / "bad", monkeypatch, query_anchors=((0, 2),))
    height, width = good.views[0]["img"].shape[-2:]
    model = _FakeArc(good.num_observations, height, width)

    metrics = train_cli.evaluate_held_out(
        model=model,
        plans=[
            plan_record(_record(seq_name="bad"), budget=48, stride=2),
            plan_record(_record(seq_name="good"), budget=48, stride=2),
        ],
        scene_provider=lambda plan: bad if plan.seq_name == "bad" else good,
        precision="32",
        huber_delta_m=0.05,
        step=1,
        output_dir=tmp_path / "out",
        query_anchors=["0:0"],
        emit_predictions=False,
    )

    assert [entry["scene"] for entry in metrics["skipped_scenes"]] == ["bad"]
    assert [entry["scene"] for entry in metrics["per_scene"]] == ["good"]
    # "scenes" counts the SCORED ones, so the curve point is not silently an
    # average over a set half of which contributed nothing.
    assert metrics["scenes"] == 1
    assert metrics["position_loss"] is not None


def test_an_unsupervisable_val_scene_does_not_end_the_run(tmp_path, monkeypatch):
    """End to end: the run reaches its full step count and still checkpoints.

    Two held-out scenes rather than one, and not for tidiness: at a one-scene
    held-out set "one bad scene" and "the whole set is unsupervisable" are the
    same event, and the preflight refuses the latter. Two is also the real shape
    -- the policy exists so the good scenes keep producing a curve.
    """

    train_cli._STOP_REQUESTED.clear()
    good = _cpu_eval_scene(tmp_path / "good", monkeypatch)
    bad = _step_scene(tmp_path / "bad", monkeypatch, query_anchors=((0, 2),))
    height, width = good.views[0]["img"].shape[-2:]
    model = _FakeArc(good.num_observations, height, width)

    result = train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": 1e-3}]
        ),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=4, eval_every=2),
        scene_provider=lambda plan: bad if plan.seq_name == "val-bad" else good,
        step_fn=lambda *, step, plan, **_: train_cli.StepOutcome(
            step=step, seq_name=plan.seq_name, loss=0.0, metric_error_m=0.0,
            sample_count=1, alignment_scale=1.0, alignment_residual_m=0.0,
            learning_rates=[1e-3], gradient_norms={},
        ),
        output_dir=tmp_path,
        val_plans=[
            plan_record(_record(seq_name=name), budget=48, stride=2)
            for name in ("val-bad", "val-good")
        ],
    )

    assert result["completed_steps"] == 4, "the run reaches its full step count"
    assert [entry["step"] for entry in result["evaluations"]] == [2, 4]
    first = result["evaluations"][0]
    assert [entry["scene"] for entry in first["skipped_scenes"]] == ["val-bad"]
    assert first["scenes"] == 1, "the good scene still makes the curve point"
    assert first["position_loss"] is not None
    # The post-loop checkpoint is what the escaping exception used to skip.
    assert (tmp_path / "train_state.pt").is_file()


def test_a_wholly_unsupervisable_held_out_set_dies_at_step_zero(
    tmp_path, monkeypatch
):
    """Per scene the skip is right; for the whole set it would produce no curve.

    A held-out set none of whose scenes can be supervised would otherwise spend
    the entire allocation writing `scenes: 0` and a null loss at every boundary
    -- a run that reads as complete and measured nothing, which is exactly what
    the preflight exists to prevent.
    """

    bad = _step_scene(tmp_path / "bad", monkeypatch, query_anchors=((0, 2),))
    stepped = []

    with pytest.raises(RuntimeError, match="held-out scenes can be supervised"):
        train_cli.run_training(
            model=_toy_model(),
            optimizer=torch.optim.AdamW(
                [{"params": list(_toy_model().parameters()), "lr": 1e-3}]
            ),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=_plans(4),
            args=_loop_args(tmp_path, num_steps=4, query_anchors=["0:2"]),
            scene_provider=lambda _plan: bad,
            step_fn=lambda **kwargs: stepped.append(kwargs),
            output_dir=tmp_path,
            val_plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
        )

    assert stepped == [], "before any training time was spent, which is the point"


# --------------------------------------- the per-step history file ---


def _history(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_every_step_is_flushed_to_the_history_file(tmp_path, monkeypatch):
    """The curve has to reach a file, and run_summary.json is not that file.

    gradient_norms (clipped_total included) and the training-stream confidence
    stats were computed every step and discarded: the summary literal has no
    history key, and `json.dumps` on a StepOutcome raises TypeError, which is
    itself proof the key was never written. The step print carries neither.
    """

    train_cli._STOP_REQUESTED.clear()
    scene = _step_scene(tmp_path / "scene", monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    model = _FakeArc(scene.num_observations, height, width)

    result = train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": 1e-3}]
        ),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(3),
        args=_loop_args(tmp_path, num_steps=3),
        scene_provider=lambda _plan: scene,
        step_fn=train_cli.train_step,
        output_dir=tmp_path,
    )

    records = _history(tmp_path)
    assert [record["step"] for record in records] == [0, 1, 2]
    # One record per StepOutcome field, under the dataclass's own names, so the
    # file and the in-memory history cannot drift into two vocabularies.
    assert set(records[0]) == {
        field.name for field in dataclasses.fields(train_cli.StepOutcome)
    }
    # The two things that were being computed and thrown away.
    assert "clipped_total" in records[0]["gradient_norms"]
    assert records[0]["confidence"] is not None
    assert records[0]["eligibility"]["total_query_count"] > 0
    # Same steps as the returned history, which main() still reads for the peak.
    assert [outcome.step for outcome in result["history"]] == [0, 1, 2]


def test_the_history_survives_a_run_that_never_returns(tmp_path):
    """The reason it is not a summary key.

    run_summary.json is written only after run_training returns, so a segment
    killed by the wall clock -- the normal end of 4 of the run's 5 segments --
    writes none at all. A per-step flush leaves the steps that did happen.
    """

    train_cli._STOP_REQUESTED.clear()

    def dying_step(*, step, plan, **_):
        if step == 2:
            raise RuntimeError("the wall clock, near enough")
        return train_cli.StepOutcome(
            step=step, seq_name=plan.seq_name, loss=float(step), metric_error_m=0.0,
            sample_count=1, alignment_scale=1.0, alignment_residual_m=0.0,
            learning_rates=[1e-3], gradient_norms={},
        )

    model = _toy_model()
    with pytest.raises(RuntimeError, match="wall clock"):
        train_cli.run_training(
            model=model,
            optimizer=torch.optim.AdamW(
                [{"params": list(model.parameters()), "lr": 1e-3}]
            ),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=_plans(4),
            args=_loop_args(tmp_path, num_steps=4),
            scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
            step_fn=dying_step,
            output_dir=tmp_path,
        )

    assert not (tmp_path / "run_summary.json").exists()
    assert [record["step"] for record in _history(tmp_path)] == [0, 1]


def test_a_resume_rewrites_the_history_from_its_restart_step(tmp_path):
    """A killed segment's uncheckpointed steps must not be recorded twice.

    The gradients of the steps after the last checkpoint were rolled back, so the
    resumed segment recomputes them. Appending would leave two contradictory
    records for the same step number and a curve that doubles back on itself.
    """

    train_cli._STOP_REQUESTED.clear()
    path = tmp_path / "history.jsonl"
    path.write_text(
        "".join(
            json.dumps({"step": step, "seq_name": "old", "loss": 9.0}) + "\n"
            for step in range(4)
        )
    )

    train_cli.open_step_history(tmp_path, start_step=2)

    records = _history(tmp_path)
    assert [record["step"] for record in records] == [0, 1]
    assert all(record["seq_name"] == "old" for record in records), (
        "the checkpointed prefix is kept verbatim"
    )
    # A fresh run, by contrast, keeps whatever is there and appends -- there is
    # nothing to contradict.
    train_cli.open_step_history(tmp_path, start_step=0)
    assert [record["step"] for record in _history(tmp_path)] == [0, 1]
