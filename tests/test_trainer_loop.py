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

import ast
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
from arc.training.scene_provider import MVTrackerSceneProvider, SceneProviderError
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
        # The spec's other half, stored alongside it: off is the per-run
        # contract every existing test assumes.
        adaptive_query_anchors=False,
        # The held-out window's cameras, the parser's default. Stored and
        # refused on resume like the spec, and run_training reads it to work out
        # which slots that window seats; every loop test needs it present.
        val_cameras=[0, 1, 2, 3],
        # Where the time-index table started. _checkpoint_settings stores both
        # and check_resume_settings refuses a change, so every loop test needs
        # them present; these are the parser's defaults.
        time_embedding_init="orthogonal",
        time_embedding_init_scale=0.3,
        # Read by seed_time_index_embedding, which draws the orthogonal rows
        # from its own generator rather than the global stream.
        seed=0,
        # Which objective the run descends. Both weights at 0 is the position-only
        # contract every existing test assumes; _checkpoint_settings stores all
        # three and check_resume_settings refuses a change, so every loop test
        # needs them present. `confidence_alpha` is post-_validate_args here --
        # a float or None, never the flag's "auto" string.
        confidence_weight=0.0,
        sync_weight=0.0,
        confidence_alpha=None,
        # The run's frozen alpha, which run_training pins after the first step
        # and _checkpoint_settings carries. None until something resolves it.
        resolved_confidence_alpha=None,
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


def test_a_resume_that_changes_the_held_out_cameras_is_refused(tmp_path):
    """--val_cameras stopped being only the window's shape.

    Under --adaptive_query_anchors the window drops every slot it cannot seat,
    so this count IS the eval's anchor count: a second segment run under fewer
    cameras would keep extending one held-out curve while measuring something
    narrower, with the step counter continuing and nothing saying so. Order is
    refused too, because the first surviving slot owns the eval's Sim(3).
    """

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    assert stored["val_cameras"] == [0, 1, 2, 3]

    with pytest.raises(RuntimeError, match="val_cameras") as excinfo:
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, val_cameras=[0, 1])
        )
    assert "[0, 1, 2, 3]" in str(excinfo.value) and "[0, 1]" in str(excinfo.value)

    with pytest.raises(RuntimeError, match="val_cameras"):
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, val_cameras=[3, 2, 1, 0])
        )

    # The same window continues.
    train_cli.check_resume_settings(stored, _loop_args(tmp_path))


# ------------------------------------------------- time-index embedding seed ---
#
# The trainer trained a zero-filled table for a full run: the constructor
# zero-fills it, the released checkpoint carries no key to overwrite it, and the
# lookup is additive, so every index contributed nothing and the index-reversal
# readout measured only backward nondeterminism.  main() is .to("cuda")-gated, so
# these drive seed_time_index_embedding directly -- that seam existing is the
# point, since an inline block in main() is what the CPU suite could not reach.


def _seedable_model(*, zero_filled=True, max_time_indices=7):
    """A duck-typed stand-in for a freshly built Arc, on CPU.

    seed_time_index_embedding reads only ``consumed_legacy_missing_keys`` and
    ``backbone.pretrained``, so the real encoder is needed but the rest of Arc is
    not -- and neither is a checkpoint or a GPU.  ``zero_filled`` picks which
    provenance the loader reported: True is the released checkpoint (the key was
    missing and got zeros), False is a checkpoint that already carried a table.
    """

    from test_time_indexing import _configured_time_transformer

    return SimpleNamespace(
        backbone=SimpleNamespace(
            pretrained=_configured_time_transformer(max_time_indices)
        ),
        consumed_legacy_missing_keys=(
            frozenset({train_cli.TIME_EMBEDDING_KEY}) if zero_filled else frozenset()
        ),
    )


def test_the_default_flags_seed_the_table_orthogonally_at_scale(tmp_path):
    """An unflagged run must leave step 0 with distinguishable indices.

    This is the regression: with the table at zeros the whole temporal mechanism
    is a no-op that nothing downstream reports, because the gradient guard sees a
    live gradient on a zero row just the same.
    """

    model = _seedable_model()
    encoder = model.backbone.pretrained
    assert torch.count_nonzero(encoder.time_index_embedding.weight) == 0

    assert train_cli.seed_time_index_embedding(model, _loop_args(tmp_path)) is True

    weight = encoder.time_index_embedding.weight
    assert torch.count_nonzero(weight) > 0
    # Calibrated to this checkpoint's own time token, not an absolute constant.
    expected_row_norm = 0.3 * encoder.time_token.detach().float().norm()
    torch.testing.assert_close(
        weight.norm(dim=1),
        expected_row_norm.expand(7),
        atol=1e-5,
        rtol=1e-5,
    )
    gram = weight @ weight.T
    off_diagonal = gram - torch.diag(torch.diagonal(gram))
    assert off_diagonal.abs().max() < 1e-4
    assert weight.requires_grad


def test_the_seed_is_reproducible_from_the_run_seed_alone(tmp_path):
    """Two invocations at one --seed must place the same rows.

    And the global stream must be untouched: the loop's per-step determinism and
    the RNG capture/restore across a resume both assume this call consumes
    nothing from it.
    """

    # Built first: constructing the encoder draws from the global stream, and
    # the claim under test is about the seeding call alone.
    first = _seedable_model()
    torch.manual_seed(1234)
    undisturbed = torch.rand(4)

    torch.manual_seed(1234)
    train_cli.seed_time_index_embedding(first, _loop_args(tmp_path, seed=7))
    assert torch.equal(torch.rand(4), undisturbed)

    second = _seedable_model()
    with torch.no_grad():
        second.backbone.pretrained.time_token.copy_(
            first.backbone.pretrained.time_token
        )
    train_cli.seed_time_index_embedding(second, _loop_args(tmp_path, seed=7))
    assert torch.equal(
        second.backbone.pretrained.time_index_embedding.weight.detach(),
        first.backbone.pretrained.time_index_embedding.weight.detach(),
    )

    third = _seedable_model()
    with torch.no_grad():
        third.backbone.pretrained.time_token.copy_(first.backbone.pretrained.time_token)
    train_cli.seed_time_index_embedding(third, _loop_args(tmp_path, seed=8))
    assert not torch.equal(
        third.backbone.pretrained.time_index_embedding.weight.detach(),
        first.backbone.pretrained.time_index_embedding.weight.detach(),
    )


def test_the_zeros_init_keeps_the_constructor_state(tmp_path):
    """The escape hatch has to actually opt out, not merely rename the init."""

    model = _seedable_model()
    assert train_cli.seed_time_index_embedding(
        model, _loop_args(tmp_path, time_embedding_init="zeros")
    ) is False
    assert torch.count_nonzero(
        model.backbone.pretrained.time_index_embedding.weight
    ) == 0


def test_seeding_a_checkpoint_loaded_table_is_refused(tmp_path):
    """Overwriting a trained table would discard a finetune without saying so.

    The provenance comes from the loader, not from a flag: a table absent from
    the checkpoint is recorded in consumed_legacy_missing_keys, and its absence
    from that set means the weights came from the file.
    """

    model = _seedable_model(zero_filled=False)
    with pytest.raises(RuntimeError, match="loaded from the checkpoint"):
        train_cli.seed_time_index_embedding(model, _loop_args(tmp_path))

    # Named in the message, because it is the only way to proceed.
    train_cli.seed_time_index_embedding(
        model, _loop_args(tmp_path, time_embedding_init="zeros")
    )


def test_a_resume_is_never_re_seeded(tmp_path):
    """--resume restores a trained table; re-seeding it would discard the run.

    Not merely harmless-because-overwritten: run_training refuses a mismatched
    resume *before* restoring any state, on the promise that the model is still
    exactly as built, so a seed applied here would survive a refusal it was
    supposed to be rolled back by.
    """

    model = _seedable_model()
    assert train_cli.seed_time_index_embedding(
        model, _loop_args(tmp_path, resume=str(tmp_path / "train_state.pt"))
    ) is False
    assert torch.count_nonzero(
        model.backbone.pretrained.time_index_embedding.weight
    ) == 0

    # The resume short-circuits before the provenance guard, so a resume whose
    # table came from a checkpoint does not trip the loaded-table refusal either.
    loaded = _seedable_model(zero_filled=False)
    assert train_cli.seed_time_index_embedding(
        loaded, _loop_args(tmp_path, resume=str(tmp_path / "train_state.pt"))
    ) is False


def test_a_resume_that_changes_the_time_embedding_init_is_refused(tmp_path):
    """The only thing tying a restored table to the run that trained it.

    seed_time_index_embedding deliberately does not re-seed on a resume, so
    nothing else compares the init the stream started from against the one this
    invocation declares.
    """

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    assert stored["time_embedding_init"] == "orthogonal"
    assert stored["time_embedding_init_scale"] == 0.3

    with pytest.raises(RuntimeError) as excinfo:
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, time_embedding_init="zeros")
        )
    assert "'orthogonal'" in str(excinfo.value)
    assert "'zeros'" in str(excinfo.value)

    with pytest.raises(RuntimeError, match="time_embedding_init_scale"):
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, time_embedding_init_scale=0.2)
        )

    # Unchanged flags continue, which is what a requeue-resume runs with.
    train_cli.check_resume_settings(stored, _loop_args(tmp_path))

    # And the refused tier still rejects a checkpoint that predates the key.
    partial = train_cli._checkpoint_settings(_loop_args(tmp_path))
    del partial["time_embedding_init"]
    with pytest.raises(RuntimeError, match="time_embedding_init"):
        train_cli.check_resume_settings(partial, _loop_args(tmp_path))


def test_main_seeds_the_table_between_the_freeze_and_the_optimizer():
    """Every test above passes on a main() that never calls the seeder.

    Which is exactly what shipped: reinitialize_time_index_embedding existed and
    was tested, and its only caller was the overfit harness, so the multi-scene
    trainer ran a full job on a zero table. Inspected rather than executed
    because main() cannot run without a GPU and a checkpoint -- the same reason
    nothing covered the gap in the first place.
    """

    main_def = next(
        node
        for node in ast.walk(ast.parse(Path(train_cli.__file__).read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    ordered = ("set_freeze", "seed_time_index_embedding", "build_optimizer")
    found = {name: [] for name in ordered}
    for node in ast.walk(main_def):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in found:
            found[name].append(node.lineno)

    for name in ordered:
        assert len(found[name]) == 1, f"expected exactly one {name} call in main()"

    # The order is the contract: after set_freeze so the table is the one the
    # freeze mode will train, and before build_optimizer, which selects the
    # embedding into its own --embedding_lr group.
    assert [found[name][0] for name in ordered] == sorted(
        found[name][0] for name in ordered
    )


def test_the_init_flags_default_to_the_swept_band_and_reach_the_artifacts():
    """The default is deliberately 0.3, not the overfit harness's 0.1."""

    args = train_cli.build_arg_parser().parse_args(["--manifest", "m.jsonl"])
    assert args.time_embedding_init == "orthogonal"
    assert args.time_embedding_init_scale == 0.3
    assert isinstance(args.time_embedding_init_scale, float)

    override = train_cli.build_arg_parser().parse_args(
        ["--manifest", "m.jsonl", "--time_embedding_init", "zeros"]
    )
    assert override.time_embedding_init == "zeros"


def test_the_init_is_recorded_in_the_plan_summary_settings(tmp_path):
    """run_summary.json's settings come from _plan_summary, not the checkpoint.

    Without this an archived summary cannot say whether its readout was
    structural -- which is exactly the question job 19852617 left open.
    """

    args = _validator_args(tmp_path, manifest="m.jsonl")
    tally = SimpleNamespace(planned=[], skipped=[], skip_counts={}, considered=0,
                            threshold_skip_fraction=0.0)
    settings = train_cli._plan_summary(tally, args)["settings"]

    assert settings["time_embedding_init"] == "orthogonal"
    assert settings["time_embedding_init_scale"] == 0.3


def test_the_validator_rejects_an_unusable_init_scale_or_rate(tmp_path):
    """Parse time, before a cluster node burns a full model load on the flag."""

    for bad in (0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="time_embedding_init_scale"):
            train_cli._validate_args(
                _validator_args(tmp_path, time_embedding_init_scale=bad)
            )

    # --embedding_lr 0 reproduces the same structurally zero readout by a
    # different route: the table stays at its init and nothing raises.
    for bad in (0, -1e-5, float("nan")):
        with pytest.raises(ValueError, match="embedding_lr"):
            train_cli._validate_args(_validator_args(tmp_path, embedding_lr=bad))
        with pytest.raises(ValueError, match="encoder_lr"):
            train_cli._validate_args(_validator_args(tmp_path, encoder_lr=bad))

    # None is the parser default and means "follow --lr"; it must stay allowed.
    train_cli._validate_args(
        _validator_args(tmp_path, embedding_lr=None, encoder_lr=None)
    )
    train_cli._validate_args(
        _validator_args(tmp_path, embedding_lr=2e-5, encoder_lr=3e-6)
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
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
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

    args = _validator_args(tmp_path, val_scenes_file="val.json", val_data_root=None)

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
        # with this test's single-anchor scenes is deliberate. 3:1 fits the
        # held-out window whole (4 cameras x 12 times), so what seated equals
        # what was asked for and this assertion is the same as before the
        # window learned to drop.
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


def test_the_eval_records_the_anchors_the_held_out_window_actually_seated(
    tmp_path, monkeypatch
):
    """Reason (b), discharged where it lives.

    Letting the held-out window drop a slot is only safe because the curve stops
    being labelled with supervision the eval never applied. A 6-slot spec against
    a 4-camera window must land in metrics.json as the four that seated -- and
    the same run under a spec the window holds whole must record exactly what it
    records today, which is the other half of the same guarantee.
    """

    scene = _cpu_eval_scene(tmp_path, monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    val = [plan_record(_record(seq_name="0000"), budget=48, stride=2)]
    wide = ["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"]

    def run(directory, **overrides):
        model = _FakeArc(scene.num_observations, height, width)
        train_cli.run_training(
            model=model,
            optimizer=torch.optim.AdamW(
                [{"params": list(model.parameters()), "lr": 1e-3}]
            ),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            plans=_plans(4),
            args=_loop_args(directory, num_steps=4, eval_every=4, **overrides),
            scene_provider=lambda _plan: scene,
            step_fn=lambda *, step, plan, **_: train_cli.StepOutcome(
                step=step, seq_name=plan.seq_name, loss=0.0, metric_error_m=0.0,
                sample_count=1, alignment_scale=1.0, alignment_residual_m=0.0,
                learning_rates=[1e-3], gradient_norms={},
            ),
            output_dir=directory,
            val_plans=val,
        )
        return json.loads(
            (directory / "eval" / "step-4" / "metrics.json").read_text()
        )

    written = run(
        tmp_path / "wide", query_anchors=wide, adaptive_query_anchors=True
    )
    assert written["query_anchors"] == ["0:0", "1:0", "2:0", "3:0"], (
        "the four the 4-camera window seated, not the six that were asked for"
    )

    # A spec the window holds whole records identically under both modes, which
    # is what keeps every non-adaptive run's artifacts where they were.
    fits = ["0:0", "1:0"]
    for name, adaptive in (("strict", False), ("ceiling", True)):
        written = run(
            tmp_path / name, query_anchors=fits, adaptive_query_anchors=adaptive
        )
        assert written["query_anchors"] == fits, name


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
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
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


def test_a_step_at_fewer_anchors_reduces_over_its_own_samples(
    tmp_path, monkeypatch
):
    """What --adaptive_query_anchors makes routine: a step whose realized anchor
    count is below the spec's, reducing correctly anyway.

    Here the drop is by eligibility rather than by seating -- every fixture query
    starts at time 0, so the time-2 anchor is rejected wholesale at
    query_time_mismatch -- but the arithmetic under test is the same either way,
    because both reach train_step as a weight of zero. The weights sum to 1 over
    whatever survives, so the reported loss is the mean over *this* step's
    samples with no 1/anchor_count anywhere; a step supervising fewer anchors
    must not report a proportionally smaller loss.
    """

    from arc.training import build_anchor_correspondences
    from arc.training.runtime import anchor_sample_counts

    scene = _step_scene(tmp_path, monkeypatch, query_anchors=((0, 0), (0, 2)))
    height, width = scene.views[0]["img"].shape[-2:]
    correspondences, _ = build_anchor_correspondences(scene)
    counts = anchor_sample_counts(scene, correspondences, 2)
    assert counts[0] > 0 and counts[1] == 0, "the second anchor must reach nothing"

    outcome = train_cli.train_step(
        model=_FakeArc(scene.num_observations, height, width),
        scene=scene,
        plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
        optimizer=torch.optim.AdamW(
            [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3}]
        ),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        precision="32",
        huber_delta_m=0.05,
        grad_clip=1.0,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
        learning_rates=[1e-3],
        step=0,
    )

    # Seated two, supervised one, and the record says which: length is what the
    # window seated, the zero entry is the anchor that reached nothing.
    assert outcome.anchor_sample_counts == counts
    assert len(counts) == 2 and counts[1] == 0
    # The denominator is this step's own supervised samples, not the spec's
    # anchor count: the empty anchor contributes to neither sum.
    assert outcome.sample_count == counts[0]
    assert np.isfinite(outcome.loss) and outcome.loss > 0


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
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
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
    # Reported by the real step, not just by the injected ones the histogram
    # tests drive: both anchors seated and both supervised something.
    assert len(outcome.anchor_sample_counts) == 2 and all(
        outcome.anchor_sample_counts
    )
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
            confidence_weight=0.0,
            confidence_alpha=None,
            sync_weight=0.0,
            learning_rates=[1e-3],
            step=0,
        )
    assert "query_time_mismatch" in str(excinfo.value)
    assert not isinstance(excinfo.value, SceneProviderError)


# ------------------------------------------- the confidence and sync terms ---


def _weighted_step(tmp_path, monkeypatch, *, scene=None, **weights):
    """One real train_step at the given weights, plus what it was handed.

    Records every ``weighted_anchor_total`` and ``tracking_only`` call, which is
    where the per-anchor shares are decided and where the retained fields are.
    """

    scene = scene or _step_scene(
        tmp_path, monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),)
    )
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    model = _FakeArc(scene.num_observations, height, width)

    totals: list[dict] = []
    kept: list[bool] = []
    real_total = train_cli.weighted_anchor_total
    real_tracking_only = train_cli.tracking_only

    def recording_total(result, **kwargs):
        totals.append(dict(kwargs))
        return real_total(result, **kwargs)

    def recording_tracking_only(raw, keep_confidence=False):
        kept.append(keep_confidence)
        return real_tracking_only(raw, keep_confidence=keep_confidence)

    monkeypatch.setattr(train_cli, "weighted_anchor_total", recording_total)
    monkeypatch.setattr(train_cli, "tracking_only", recording_tracking_only)

    outcome = train_cli.train_step(
        model=model,
        scene=scene,
        plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        precision="32",
        huber_delta_m=0.05,
        grad_clip=1.0,
        learning_rates=[1e-3],
        step=0,
        **weights,
    )
    return outcome, totals, kept, scene


def test_both_weights_zero_leaves_the_step_exactly_position_only(
    tmp_path, monkeypatch
):
    """The zeros control, which every archived comparison is against.

    `compose_tracking_loss` omits a zero-weight term rather than multiplying by
    it, so this must not merely be close: the anchor totals must carry the
    position weight alone, nothing may retain the confidence field, and no
    breakdown may be claimed for a step that has only one term to break down.
    """

    outcome, totals, kept, _ = _weighted_step(
        tmp_path,
        monkeypatch,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
    )

    assert outcome.loss_breakdown is None
    assert outcome.confidence_alpha is None
    # None, not a zeroed dict: a position-only step never looked, which is a
    # different finding from a confidence step that looked and found nothing.
    assert outcome.confidence_dropped is None
    assert kept == [False, False], "conf_track_multi must not be retained"
    for call in totals:
        assert call["confidence_weight"] == 0.0
        assert call["sync_weight"] == 0.0
    # The position shares still sum to 1 over the step's own anchors.
    assert sum(call["position_weight"] for call in totals) == pytest.approx(1.0)


def test_the_confidence_term_keeps_its_field_and_splits_by_its_own_mask(
    tmp_path, monkeypatch
):
    """The share is the CONFIDENCE sample share, not the position one.

    The confidence term deliberately does not mask on visibility, so with an
    invisible sample in the fixture the two masks differ and the two share
    vectors differ with them. Reusing the position shares would be wrong by
    exactly the ratio between the masks, and would pass a weaker assertion that
    only checked the shares sum to the weight.
    """

    from arc.training import build_anchor_correspondences
    from arc.training.runtime import anchor_confidence_counts, anchor_sample_counts

    outcome, totals, kept, scene = _weighted_step(
        tmp_path,
        monkeypatch,
        confidence_weight=0.25,
        confidence_alpha=3.0,
        sync_weight=0.0,
    )

    # Load-bearing: runtime.tracking_only drops conf_track_multi by default and
    # the term needs it. Missing this is a KeyError, not a wrong number.
    assert kept == [True, True]

    correspondences, _ = build_anchor_correspondences(scene)
    position_counts = anchor_sample_counts(scene, correspondences, 2)
    confidence_counts = anchor_confidence_counts(scene, correspondences, 2)
    assert confidence_counts != position_counts, "the fixture must separate the masks"

    expected = [0.25 * count / sum(confidence_counts) for count in confidence_counts]
    assert [call["confidence_weight"] for call in totals] == pytest.approx(expected)
    assert sum(call["confidence_weight"] for call in totals) == pytest.approx(0.25)
    # And the position side is untouched by any of it.
    assert [call["position_weight"] for call in totals] == pytest.approx(
        [count / sum(position_counts) for count in position_counts]
    )

    assert set(outcome.loss_breakdown) == {"position", "confidence"}
    # The breakdown is unweighted, so the position entry IS the reported loss.
    assert outcome.loss_breakdown["position"] == pytest.approx(outcome.loss)
    # Summed over the anchors by the real step, and zero on a clean fixture: the
    # shares above are exact precisely while this stays zero.
    assert outcome.confidence_dropped == {
        "total": 0,
        "target_nonfinite": 0,
        "prediction_nonfinite": 0,
        "confidence_nonfinite": 0,
    }


def test_the_sync_share_is_one_over_the_anchors_the_step_actually_runs(
    tmp_path, monkeypatch
):
    """What makes the term safe under --adaptive_query_anchors.

    Every anchor's sync_loss is a mean over an identical element count -- P, H
    and W come from the window, never from which anchor -- so the stacked-Q mean
    is the plain mean of the per-anchor means and the share is a flat 1/N. The
    property that matters across a stream whose N varies is that the shares sum
    to the weight at every N, which is asserted at both 1 and 2 supervising
    anchors. N is len(active_anchors), not the seated count: dividing by what
    the window seated would undershoot the weight when an anchor seats and
    supervises nothing.
    """

    _, two_anchor, _, _ = _weighted_step(
        tmp_path / "two",
        monkeypatch,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.5,
    )
    assert [call["sync_weight"] for call in two_anchor] == pytest.approx([0.25, 0.25])
    assert sum(call["sync_weight"] for call in two_anchor) == pytest.approx(0.5)

    single = _step_scene(tmp_path / "one", monkeypatch)
    _, one_anchor, _, _ = _weighted_step(
        tmp_path / "one",
        monkeypatch,
        scene=single,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.5,
    )
    assert [call["sync_weight"] for call in one_anchor] == pytest.approx([0.5])
    # Same weight reaches the objective at either anchor count: that is the
    # whole claim, and it is what the trainer's old comment doubted.
    assert sum(call["sync_weight"] for call in one_anchor) == pytest.approx(0.5)


def test_the_undivided_sync_weight_is_what_reaches_the_loss(tmp_path, monkeypatch):
    """The gate and the share are different numbers and must not be conflated.

    ``sparse_tracking_loss``'s ``sync_weight`` only decides whether the term is
    BUILT; the total it composes is discarded by the multi-anchor path. Passing
    the divided share there instead would still build the term, so no other
    assertion in this file would notice -- until an anchor count of zero-ish
    size made the share underflow the > 0.0 gate and the term silently vanished.
    """

    import arc.training as training_package

    seen: list[float] = []
    # train_step does `from arc.training import sparse_tracking_loss` inside the
    # function, so the package namespace is the only interception point; patching
    # arc.training.sparse_tracking would rebind a name nothing looks up again.
    real_loss = training_package.sparse_tracking_loss

    def recording(*args, **kwargs):
        seen.append(kwargs["sync_weight"])
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(training_package, "sparse_tracking_loss", recording)
    _weighted_step(
        tmp_path,
        monkeypatch,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.5,
    )

    assert seen == [0.5, 0.5], "the loss gets the undivided weight, not the share"


def test_the_reported_loss_does_not_move_when_the_extra_terms_are_enabled(
    tmp_path, monkeypatch
):
    """`SparseTrackingLossResult.loss` stays the position-only Huber.

    The run's curve and every archived comparison are that number, so it must
    mean the same thing at every weight setting. Only the *total* being
    descended may change.
    """

    baseline, base_totals, _, _ = _weighted_step(
        tmp_path / "off",
        monkeypatch,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
    )
    enabled, _, _, _ = _weighted_step(
        tmp_path / "on",
        monkeypatch,
        confidence_weight=0.25,
        confidence_alpha=3.0,
        sync_weight=0.5,
    )

    assert enabled.loss == baseline.loss
    assert enabled.sample_count == baseline.sample_count
    assert set(enabled.loss_breakdown) == {"position", "sync", "confidence"}
    assert enabled.loss_breakdown["position"] == pytest.approx(baseline.loss)
    assert base_totals[0]["position_weight"] == pytest.approx(
        base_totals[0]["position_weight"]
    )


def test_auto_alpha_is_resolved_once_and_then_reused(tmp_path, monkeypatch):
    """A moving target is not one the optimizer can descend.

    ``sparse_tracking_loss`` re-resolves alpha on every call it is handed None,
    from that call's own confidence and error means -- so without pinning, each
    anchor of each step would optimize toward a different conf*. The step must
    resolve once and hand the resolved value to every later anchor.
    """

    import arc.training as training_package

    seen: list[float | None] = []
    real_loss = training_package.sparse_tracking_loss

    def recording(*args, **kwargs):
        seen.append(kwargs["confidence_alpha"])
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(training_package, "sparse_tracking_loss", recording)
    outcome, _, _, _ = _weighted_step(
        tmp_path,
        monkeypatch,
        confidence_weight=0.25,
        confidence_alpha=None,
        sync_weight=0.0,
    )

    assert seen[0] is None, "the first anchor is what resolves it"
    assert seen[1] is not None and seen[1] == pytest.approx(outcome.confidence_alpha)
    # Reported so the history file can be read without the checkpoint.
    assert outcome.confidence_alpha > 0


def test_an_explicit_alpha_is_never_re_resolved(tmp_path, monkeypatch):
    import arc.training as training_package

    seen: list[float | None] = []
    real_loss = training_package.sparse_tracking_loss

    def recording(*args, **kwargs):
        seen.append(kwargs["confidence_alpha"])
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(training_package, "sparse_tracking_loss", recording)
    outcome, _, _, _ = _weighted_step(
        tmp_path,
        monkeypatch,
        confidence_weight=0.25,
        confidence_alpha=7.5,
        sync_weight=0.0,
    )

    assert seen == [7.5, 7.5]
    assert outcome.confidence_alpha == pytest.approx(7.5)


def test_the_loop_pins_the_first_steps_alpha_onto_the_run(tmp_path):
    """Across steps, not only across anchors: the loop owns the pin.

    Driven through an injected step function, because what is under test is the
    loop's bookkeeping rather than the loss -- the real step needs a GPU-shaped
    fixture per step and would say nothing more about this.
    """

    train_cli._STOP_REQUESTED.clear()
    handed: list[float | None] = []

    def resolving_step(**kwargs):
        handed.append(kwargs["confidence_alpha"])
        loss = kwargs["model"](torch.ones(1, 3)).sum()
        loss.backward()
        kwargs["optimizer"].step()
        kwargs["optimizer"].zero_grad(set_to_none=True)
        return train_cli.StepOutcome(
            step=kwargs["step"],
            seq_name=kwargs["plan"].seq_name,
            loss=float(loss.item()),
            metric_error_m=0.0,
            sample_count=1,
            alignment_scale=1.0,
            alignment_residual_m=0.0,
            learning_rates=list(kwargs["learning_rates"]),
            gradient_norms={},
            # What a real step returns once its first anchor has resolved it.
            confidence_alpha=4.25,
        )

    args = _loop_args(tmp_path, num_steps=3, confidence_weight=0.25)
    model = _toy_model()
    train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=args,
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=resolving_step,
        output_dir=tmp_path,
    )

    assert handed == [None, 4.25, 4.25], "resolved once, then reused"
    # And it reaches the checkpoint, which is how it survives a requeue.
    assert train_cli._checkpoint_settings(args)["resolved_confidence_alpha"] == 4.25


def test_a_resume_restores_the_pinned_alpha_rather_than_re_resolving_it(tmp_path):
    stored = train_cli._checkpoint_settings(
        _loop_args(tmp_path, confidence_weight=0.25, resolved_confidence_alpha=4.25)
    )
    assert stored["resolved_confidence_alpha"] == 4.25
    # Deliberately in neither tier: an 'auto' run's request is None while its
    # checkpoint carries a float, so comparing them would refuse every
    # legitimate resume of exactly the runs that need the pin.
    assert "resolved_confidence_alpha" not in train_cli._RESUME_SETTINGS_REFUSED
    assert "resolved_confidence_alpha" not in train_cli._RESUME_SETTINGS_WARNED
    train_cli.check_resume_settings(
        stored, _loop_args(tmp_path, confidence_weight=0.25)
    )


def test_a_resume_that_changes_a_loss_weight_is_refused(tmp_path):
    """A segment that changes the objective and keeps counting steps reports one
    curve over two of them -- and the reported `loss` stays the position-only
    Huber throughout, so nothing in the history would show the switch."""

    for key, before, after in (
        ("confidence_weight", 0.0, 0.25),
        ("sync_weight", 0.0, 0.5),
        ("confidence_alpha", None, 3.0),
    ):
        stored = train_cli._checkpoint_settings(_loop_args(tmp_path, **{key: before}))
        with pytest.raises(RuntimeError, match=key):
            train_cli.check_resume_settings(
                stored, _loop_args(tmp_path, **{key: after})
            )
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, **{key: before})
        )


def test_a_checkpoint_predating_the_loss_flags_still_resumes(tmp_path):
    """The live-run case: a multi-day run resuming across eight-plus segments.

    A checkpoint written before --confidence_weight and --sync_weight existed
    could not have set them -- no other value was reachable -- so its run was
    necessarily position-only. Refusing it strands the run, and the tier's
    original premise ("only disposable smokes lack the newest key") stopped
    being true the moment real runs started resuming.
    """

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    # Exactly what a pre-change checkpoint carries: every other refused key
    # present, these three simply not yet invented.
    for key in ("confidence_weight", "sync_weight", "confidence_alpha"):
        del stored[key]

    # Position-only on both sides: the stream is the same one, so it continues.
    train_cli.check_resume_settings(stored, _loop_args(tmp_path))


def test_resuming_a_pre_flag_checkpoint_with_a_term_switched_on_is_refused(tmp_path):
    """The fallback resolves the gap, it does not tolerate it.

    Reading an absent key as 0 is only sound because 0 is what it must have
    been. That value is then compared like any stored one, so turning a term on
    against a position-only checkpoint is refused exactly as changing a stored
    weight is -- otherwise the fallback would have opened the hole the tier
    exists to close.
    """

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    for key in ("confidence_weight", "sync_weight", "confidence_alpha"):
        del stored[key]

    for key, value in (
        ("confidence_weight", 0.25),
        ("sync_weight", 0.5),
        ("confidence_alpha", 3.0),
    ):
        with pytest.raises(RuntimeError, match="predates"):
            train_cli.check_resume_settings(
                stored, _loop_args(tmp_path, **{key: value})
            )


def test_an_absent_key_whose_meaning_is_ambiguous_is_still_refused(tmp_path):
    """The fallback is a named list, not a blanket tolerance.

    val_cameras and the time-embedding pair each had a reachable non-default
    value before they were recorded, so nothing can say which one an old
    checkpoint used. Tolerating those would let a run resume under a different
    held-out window, or a differently conditioned encoder, with nothing raising.
    """

    for key in ("val_cameras", "time_embedding_init", "query_anchors"):
        assert key not in train_cli._RESUME_SETTINGS_ABSENT_DEFAULTS
        stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
        del stored[key]
        with pytest.raises(RuntimeError, match="does not resolve to a single value"):
            train_cli.check_resume_settings(stored, _loop_args(tmp_path))


def test_every_absent_default_is_the_parsers_own_default(tmp_path):
    """The mapping claims what a pre-flag run necessarily was, so it has to agree
    with what the flag defaults to -- a drift between the two would silently
    refuse, or silently admit, the wrong resumes."""

    parser = train_cli.build_arg_parser()
    defaults = parser.parse_args(["--manifest", "m.jsonl"])
    train_cli._validate_args(defaults)

    for key, implied in train_cli._RESUME_SETTINGS_ABSENT_DEFAULTS.items():
        assert key in train_cli._RESUME_SETTINGS_REFUSED, key
        assert getattr(defaults, key) == implied, key


def test_the_loss_weights_are_recorded_in_the_plan_summary_settings(tmp_path):
    tally = SimpleNamespace(
        planned=[],
        skipped=[],
        skip_counts={},
        considered=0,
        threshold_skip_fraction=0.0,
    )
    args = _validator_args(
        tmp_path,
        manifest="m.jsonl",
        confidence_weight=0.25,
        sync_weight=0.5,
        confidence_alpha=3.0,
        resolved_confidence_alpha=3.0,
    )

    settings = train_cli._plan_summary(tally, args)["settings"]

    assert settings["confidence_weight"] == 0.25
    assert settings["sync_weight"] == 0.5
    assert settings["confidence_alpha"] == 3.0
    assert settings["resolved_confidence_alpha"] == 3.0


def test_unusable_loss_weights_are_refused_at_parse_time(tmp_path):
    for bad in (-1e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="--confidence_weight"):
            train_cli._validate_args(_validator_args(tmp_path, confidence_weight=bad))
        with pytest.raises(ValueError, match="--sync_weight"):
            train_cli._validate_args(_validator_args(tmp_path, sync_weight=bad))
    # Zero is the default and must stay admissible: it is the zeros control.
    train_cli._validate_args(
        _validator_args(tmp_path, confidence_weight=0.0, sync_weight=0.0)
    )


def test_the_confidence_alpha_flag_parses_auto_and_refuses_the_rest(tmp_path):
    args = _validator_args(tmp_path, confidence_alpha="auto")
    train_cli._validate_args(args)
    assert args.confidence_alpha is None, "'auto' means resolve it later"
    assert args.resolved_confidence_alpha is None

    args = _validator_args(tmp_path, confidence_alpha="2.5")
    train_cli._validate_args(args)
    assert args.confidence_alpha == 2.5
    # An explicit value is pinned from the start; nothing resolves over it.
    assert args.resolved_confidence_alpha == 2.5

    for bad in ("", "none", "AUTOMATIC", "0", "-1", "nan", "inf"):
        with pytest.raises(ValueError, match="--confidence_alpha"):
            train_cli._validate_args(_validator_args(tmp_path, confidence_alpha=bad))


def test_the_sync_term_needs_a_window_that_can_hold_a_synchronized_pair(tmp_path):
    """--min_views 1 admits a single-camera window, which has no slot pair
    sharing a time index; synchronized_differences raises rather than returning
    an empty mean, and that would kill the run mid-step as something the loop's
    scene-skip policy cannot absorb."""

    with pytest.raises(ValueError, match="--sync_weight needs --min_views >= 2"):
        train_cli._validate_args(
            _validator_args(tmp_path, sync_weight=0.5, min_views=1)
        )
    # Off, the same window is nobody's problem.
    train_cli._validate_args(_validator_args(tmp_path, sync_weight=0.0, min_views=1))


def test_dropped_confidence_samples_are_warned_once_and_totalled(tmp_path, capsys):
    """The one signal that the per-anchor shares stopped matching the mask.

    ``anchor_confidence_counts`` cannot see prediction- or confidence-finiteness,
    so it is exact only while nothing goes non-finite -- and `expp1` is 1+exp(x),
    which overflows in BF16. Warned once rather than per step: at 20k steps a
    per-step warning buries the first occurrence, which is the one worth reading.
    """

    train_cli._STOP_REQUESTED.clear()

    def dropping_step(**kw):
        loss = kw["model"](torch.ones(1, 3)).sum()
        loss.backward()
        kw["optimizer"].step()
        kw["optimizer"].zero_grad(set_to_none=True)
        return train_cli.StepOutcome(
            step=kw["step"],
            seq_name=kw["plan"].seq_name,
            loss=float(loss.item()),
            metric_error_m=0.0,
            sample_count=1,
            alignment_scale=1.0,
            alignment_residual_m=0.0,
            learning_rates=list(kw["learning_rates"]),
            gradient_norms={},
            confidence_dropped={
                "total": 3,
                "target_nonfinite": 1,
                "prediction_nonfinite": 1,
                "confidence_nonfinite": 2,
            },
        )

    model = _toy_model()
    result = train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=4, confidence_weight=0.25),
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=dropping_step,
        output_dir=tmp_path,
    )

    errors = capsys.readouterr().err
    assert errors.count("dropped from the confidence term") == 1, "warned once"
    assert "step=0" in errors, "the FIRST occurrence is the one that gets named"
    # Every step still counts toward the run total, warning or not.
    assert result["confidence_dropped_totals"] == {
        "total": 12,
        "target_nonfinite": 4,
        "prediction_nonfinite": 4,
        "confidence_nonfinite": 8,
    }
    # And each step's own counts survive in the history file.
    lines = [
        json.loads(line)
        for line in (tmp_path / "history.jsonl").read_text().strip().splitlines()
    ]
    assert all(entry["confidence_dropped"]["total"] == 3 for entry in lines)


def test_a_run_that_drops_nothing_reports_an_empty_total_not_a_warning(
    tmp_path, capsys
):
    """A position-only run never looked, which is not the same finding as a
    confidence run that looked and found nothing wrong."""

    train_cli._STOP_REQUESTED.clear()
    recorded: list[_Recorded] = []
    _, result = _run(tmp_path, _loop_args(tmp_path, num_steps=4), recorded)

    assert result["confidence_dropped_totals"] == {}
    assert "dropped from the confidence term" not in capsys.readouterr().err


def test_the_eval_is_position_only_whatever_the_training_flags_say(
    tmp_path, monkeypatch
):
    """The held-out curve must stay comparable to the zeros control and to every
    other arm, so the eval never builds the extra terms -- and the shuffled arm
    is a second full pass whose dense terms would be built and thrown away."""

    import arc.training as training_package

    seen: list[dict] = []
    real_loss = training_package.sparse_tracking_loss

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real_loss(*args, **kwargs)

    scene = _cpu_eval_scene(tmp_path, monkeypatch)
    monkeypatch.setattr(training_package, "sparse_tracking_loss", recording)

    height, width = scene.views[0]["img"].shape[-2:]
    train_cli.evaluate_held_out(
        model=_FakeArc(scene.num_observations, height, width),
        plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
        scene_provider=lambda plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=0,
        output_dir=tmp_path,
        query_anchors=["0:0"],
    )

    assert seen, "the eval must have scored something"
    for call in seen:
        assert "confidence_weight" not in call
        assert "sync_weight" not in call
        assert "confidence_alpha" not in call


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
        # Optional rates: None means "follow --lr", which is what the parser
        # defaults to and the only value that skips the finite/positive check.
        "embedding_lr": None,
        "encoder_lr": None,
        # _validate_args parses this one in place, so unlike _loop_args it starts
        # as the parser's raw string.
        "confidence_alpha": "auto",
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


def test_adaptive_admits_a_view_slot_min_views_cannot_guarantee(tmp_path):
    """This check is precisely what the flag exists to lift.

    --min_views is a per-run floor, so it can only answer "does every step seat
    this slot?" -- the question adaptive stops asking. Raising min_views to 4
    instead is the alternative, and it deletes 17,624 of the manifest's 69,344
    replayable rows. The plan-time check takes over, on the real windows.
    """

    args = _validator_args(
        tmp_path,
        query_anchors=["0:0", "3:0"],
        min_views=2,
        adaptive_query_anchors=True,
    )
    train_cli._validate_args(args)
    assert args.query_anchors == ["0:0", "3:0"], "canonicalized, order preserved"

    # And it is only lifted by the flag.
    with pytest.raises(ValueError, match="--min_views >= 4"):
        train_cli._validate_args(
            _validator_args(tmp_path, query_anchors=["0:0", "3:0"], min_views=2)
        )


def test_adaptive_seats_what_the_held_out_window_holds_and_drops_the_rest(tmp_path):
    """The case this whole change exists for, and the run job 19857965 died on.

    Each step's camera count decides its anchor count, so the spec is as wide as
    the manifest's widest step -- 6 views -- while the held-out window is a fixed
    4 cameras. Under a ceiling an unseatable val slot is expected rather than a
    typo, and the guard that used to refuse it rested on eval/*/metrics.json
    recording the full spec; the eval now records what seated, so the reason is
    gone.

    The provider call is the substance here, not a garnish. resolve_query_anchors
    is the single anchor-resolution path every plan goes through, training and
    held-out alike, and asserting it against the flag-only helper is what proves
    the two cannot drift -- which is the only thing making the parse-time verdict
    trustworthy.
    """

    scenes = tmp_path / "val.json"
    scenes.write_text(json.dumps(["0000"]))
    args = _validator_args(
        tmp_path,
        query_anchors=["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"],
        min_views=2,
        adaptive_query_anchors=True,
        val_scenes_file=str(scenes),
        val_data_root="/held",
    )

    train_cli._validate_args(args)
    assert args.query_anchors == ["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"], (
        "the spec is recorded whole; only what the window seats is narrowed"
    )
    assert train_cli._val_anchor_slots(args) == ((0, 0), (1, 0), (2, 0), (3, 0))

    # The WINDOW is untouched -- 4 cameras x 12 times, exactly as before. This
    # change moves which anchors resolve inside it, never the window itself.
    plan = train_cli._val_plans(args)[0]
    assert len(plan.cameras) == 4 and len(plan.times) == 12
    provider = MVTrackerSceneProvider(
        query_anchor_slots=train_cli.parse_query_anchor_slots(args.query_anchors),
        adaptive_query_anchors=True,
    )
    assert provider.resolve_query_anchors(plan) == ((0, 0), (1, 0), (2, 0), (3, 0)), (
        "the provider seats the same four the parse-time helper promised"
    )

    # The same drop rule on the time axis: 48 observations over 4 val cameras
    # seat 12 times, so slot 12 does not exist and adaptive drops it too.
    args = _validator_args(
        tmp_path,
        query_anchors=["0:0", "0:12"],
        adaptive_query_anchors=True,
        val_scenes_file=str(scenes),
        val_data_root="/held",
    )
    train_cli._validate_args(args)
    assert train_cli._val_anchor_slots(args) == ((0, 0),)


def test_a_held_out_window_that_seats_no_anchor_at_all_is_refused(tmp_path):
    """Adaptive drops what the window cannot hold, but not down to nothing.

    An eval that seats no anchor supervises nothing and writes a null loss at
    every boundary, producing no curve -- the held-out mirror of the
    training-side rule that at least one slot must seat on every planned step.
    """

    args = _validator_args(
        tmp_path,
        query_anchors=["5:0"],
        min_views=6,
        adaptive_query_anchors=True,
        val_scenes_file="val.json",
        val_data_root="/held",
    )
    with pytest.raises(ValueError, match="the eval would supervise nothing"):
        train_cli._validate_args(args)


def test_the_held_out_seating_check_stays_strict_without_the_flag(tmp_path):
    """Job 19857965's exact refusal, still fatal under a contract.

    Without --adaptive_query_anchors the spec is a promise about one fixed
    window, so a slot that window cannot seat is a typo and nothing else --
    --min_views is a per-step floor on the TRAINING stream and cannot speak for
    the held-out one.
    """

    args = _validator_args(
        tmp_path,
        query_anchors=["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"],
        # High enough that the --min_views check passes, so the val check is what
        # fires rather than being masked by the earlier one.
        min_views=6,
        val_scenes_file="val.json",
        val_data_root="/held",
    )
    with pytest.raises(
        ValueError, match=r"view slot 4 does not fit the held-out window's 4 cameras"
    ):
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


def _plan_only(monkeypatch, manifest, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_temporal_tracking.py",
            "--manifest",
            str(manifest),
            "--plan_only",
            *extra,
        ],
    )


def test_adaptive_admits_a_time_slot_only_some_steps_can_seat(
    tmp_path, monkeypatch, capsys
):
    """The same manifest the strict check refuses, and the point of the flag.

    One short window vetoes the slot for every other step under the contract;
    under the ceiling it costs that one step its second anchor and nothing else.
    """

    from test_manifest_plan import _write_manifest

    records = [
        _record(step=0, seq_name="0000"),
        # seq_len 2 at stride 2 seats exactly one time, so time slot 1 cannot.
        _record(step=1, seq_name="0001", seq_len=2),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    _plan_only(
        monkeypatch,
        manifest,
        "--adaptive_query_anchors",
        "--query_anchors",
        "0:0",
        "0:1",
    )

    train_cli.main()

    assert "PASS planned 2 steps" in capsys.readouterr().out


def test_a_slot_no_planned_step_can_seat_is_refused_under_adaptive(
    tmp_path, monkeypatch, capsys
):
    """Adaptive drops what a step cannot seat, so a slot no step can seat would
    otherwise be silently inert for the whole run. Still the typo guard the
    strict check was, still GPU-free at submit time."""

    from test_manifest_plan import _write_manifest

    records = [_record(step=index, seq_name=f"{index:04d}") for index in range(2)]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    _plan_only(
        monkeypatch,
        manifest,
        "--adaptive_query_anchors",
        "--query_anchors",
        "0:0",
        "1:0",
        # Every planned step seats 4 views x 12 times (budget 48 // 4), so this
        # one is beyond the widest window the manifest can produce.
        "0:20",
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    error_output = capsys.readouterr().err
    # Names the slot that seats nowhere, and only that one: 1:0 seats fine.
    assert "0:20" in error_output and "1:0" not in error_output


def test_a_spec_with_no_always_seatable_slot_is_refused_under_adaptive(
    tmp_path, monkeypatch, capsys
):
    """Every slot seating somewhere is not enough on its own.

    With 1:0 dropped on the 1-view step and 0:1 dropped on the short-window
    step, each slot seats somewhere and one step seats neither -- which resolves
    to an empty anchor tuple and dies inside build_scene, mid-run. 0:0 always
    seats, and the message says so.
    """

    from test_manifest_plan import _write_manifest

    records = [
        _record(step=0, seq_name="0000"),
        # One view and one time: seats neither 1:0 nor 0:1.
        _record(step=1, seq_name="0001", views=[0], seq_len=2),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    _plan_only(
        monkeypatch,
        manifest,
        "--adaptive_query_anchors",
        "--min_views",
        "1",
        "--query_anchors",
        "1:0",
        "0:1",
    )

    with pytest.raises(SystemExit) as exit_info:
        train_cli.main()

    assert exit_info.value.code == 2
    error_output = capsys.readouterr().err
    assert "no --query_anchors slot seats on all 2 planned steps" in error_output
    assert "0:0" in error_output

    # Adding 0:0 is the fix the message names, and it is enough.
    _plan_only(
        monkeypatch,
        manifest,
        "--adaptive_query_anchors",
        "--min_views",
        "1",
        "--query_anchors",
        "1:0",
        "0:1",
        "0:0",
    )
    train_cli.main()
    assert "PASS planned 2 steps" in capsys.readouterr().out


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
    settings = json.loads(out.read_text())["settings"]
    assert settings["query_anchors"] == ["0:0"]
    # No held-out set, so there is no window to seat anything -- distinct from a
    # window that seats nothing, which _validate_args refuses outright.
    assert settings["realized_val_query_anchors"] is None


def test_a_wide_adaptive_spec_plans_against_a_narrower_held_out_window(
    tmp_path, monkeypatch, capsys
):
    """The submit-time check the cluster run was blocked on, end to end.

    The manifest's steps carry 2-6 views and the held-out window a fixed 4
    cameras, so the spec is deliberately wider than that window. It must plan
    GPU-free, and the summary must say both things: the six that were asked for,
    and the four the eval will actually seat.
    """

    from test_manifest_plan import _write_manifest

    records = [
        _record(step=0, seq_name="0000", views=[0, 1, 2, 3, 4, 5]),
        _record(step=1, seq_name="0001", views=[0, 1]),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    scenes = tmp_path / "val.json"
    scenes.write_text(json.dumps(["9000", "9001"]))
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
            "--adaptive_query_anchors",
            "--query_anchors",
            "0:0",
            "1:0",
            "2:0",
            "3:0",
            "4:0",
            "5:0",
            "--val_scenes_file",
            str(scenes),
            "--val_data_root",
            "/held",
        ],
    )

    train_cli.main()

    assert "PASS planned 2 steps" in capsys.readouterr().out
    settings = json.loads(out.read_text())["settings"]
    assert settings["query_anchors"] == ["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"]
    assert settings["realized_val_query_anchors"] == ["0:0", "1:0", "2:0", "3:0"]


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


def test_flipping_the_adaptive_flag_on_resume_is_refused(tmp_path):
    """Stream-defining on its own: the same slot strings supervise a different
    set of observations depending on whether a step may drop from them, so a
    "resumed" run would train a different stream under a continuing counter."""

    train_cli._STOP_REQUESTED.clear()
    _run(tmp_path / "a", _loop_args(tmp_path / "a", num_steps=4), [])
    payload = read_trainer_state(tmp_path / "a" / "train_state.pt")
    assert payload["settings"]["adaptive_query_anchors"] is False

    with pytest.raises(RuntimeError, match="adaptive_query_anchors"):
        _run(
            tmp_path / "b",
            _loop_args(
                tmp_path / "b",
                num_steps=4,
                resume=str(tmp_path / "a" / "train_state.pt"),
                adaptive_query_anchors=True,
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
    # Same rule for the anchor histogram, and the same reason: a step that
    # reports no counts must not be filed as a step that realized zero anchors.
    assert result["realized_anchor_counts"] == {
        "steps_counted": 0,
        "seated": {},
    }


def test_the_realized_anchor_histogram_records_what_each_step_seated(tmp_path):
    """Under --adaptive_query_anchors the spec is a ceiling, so it stops being
    an answer to "how many anchors did this run realize?" -- this histogram is.

    Seated only. Step 2 seats four anchors and one supervises nothing, and it
    still files under 4: what a step did with them is per step, and
    history.jsonl already records it as anchor_sample_counts.
    """

    train_cli._STOP_REQUESTED.clear()
    counts = {0: [5, 5, 5, 5], 1: [5, 5], 2: [5, 5, 0, 5], 3: [5, 5]}

    def step_fn(*, plan, learning_rates, step, **_):
        return train_cli.StepOutcome(
            step=step,
            seq_name=plan.seq_name,
            loss=0.0,
            metric_error_m=0.0,
            sample_count=1,
            alignment_scale=1.0,
            alignment_residual_m=0.0,
            learning_rates=list(learning_rates),
            gradient_norms={},
            anchor_sample_counts=counts[step],
        )

    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])
    result = train_cli.run_training(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(tmp_path, num_steps=4, adaptive_query_anchors=True),
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=step_fn,
        output_dir=tmp_path,
    )

    assert result["realized_anchor_counts"] == {
        "steps_counted": 4,
        "seated": {2: 2, 4: 2},
    }
    # Sorted by count, so the histogram reads as a distribution rather than in
    # whatever order the steps happened to arrive.
    assert list(result["realized_anchor_counts"]["seated"]) == [2, 4]


def test_a_skipped_scene_load_is_absent_from_the_anchor_histogram(tmp_path):
    """steps_counted is the denominator, so it must count executed steps only."""

    train_cli._STOP_REQUESTED.clear()
    plans = _plans(4)
    failing = plans[1].seq_name

    def provider(plan):
        if plan.seq_name == failing:
            raise SceneProviderError("scene 'x' is not in the pool at '/p' (0 scenes)")
        return SimpleNamespace(name=plan.seq_name)

    def step_fn(*, plan, learning_rates, step, **_):
        return train_cli.StepOutcome(
            step=step,
            seq_name=plan.seq_name,
            loss=0.0,
            metric_error_m=0.0,
            sample_count=1,
            alignment_scale=1.0,
            alignment_residual_m=0.0,
            learning_rates=list(learning_rates),
            gradient_norms={},
            anchor_sample_counts=[7, 7, 7],
        )

    model = _toy_model()
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}])
    result = train_cli.run_training(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=plans,
        args=_loop_args(
            tmp_path,
            num_steps=4,
            adaptive_query_anchors=True,
            max_scene_skip_fraction=1.0,
        ),
        scene_provider=provider,
        step_fn=step_fn,
        output_dir=tmp_path,
    )

    assert result["realized_anchor_counts"]["steps_counted"] == 3
    assert result["realized_anchor_counts"]["seated"] == {3: 3}


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
    # file and the in-memory history cannot drift into two vocabularies -- plus
    # the two columns append_step_history derives for resuming runs that predate
    # anchor_sample_counts. Delete both with that write; see its docstring.
    assert set(records[0]) == {
        field.name for field in dataclasses.fields(train_cli.StepOutcome)
    } | {"anchor_count", "active_anchor_count"}
    assert records[0]["anchor_count"] == len(records[0]["anchor_sample_counts"])
    assert records[0]["active_anchor_count"] == sum(
        1 for count in records[0]["anchor_sample_counts"] if count > 0
    )
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
