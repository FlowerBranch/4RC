"""Pins for data-parallel training: WORLD_SIZE > 1 without a DDP wrapper.

The two-rank tests run a REAL gloo group of two spawned processes, each driving
``run_training`` through its injected step and window seams -- the same code the
NCCL launch runs, on CPU. A collective that only one rank reaches does not
fail; it waits, so every group here carries a short timeout and a missed
collective surfaces as a failed test instead of a hung suite.

What they pin, in the order the trainer meets it: the stop is agreed at every
window top; every window closes on every rank -- a rank that executed nothing
included -- with the gradient averaged over the ranks that executed and its own
stale average kept out; a window no rank executed steps nowhere; the replicas
stay bit-identical through skips and accumulation, and the replica check
refuses ranks that diverged or resumed apart; the checkpoint step and
completed_steps come from the record of every rank, not rank 0's history; and
rank 0 alone writes, evaluates and prints the run's lines while the totals
combine across ranks.
"""

from __future__ import annotations

import ast
import datetime
import io
import json
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import train_temporal_tracking as train_cli
from arc.training import data_parallel
from arc.training.data_parallel import (
    Launch,
    join_process_group,
    read_launch_environment,
    world_size_from_environment,
)
from arc.training.scene_provider import SceneProviderError
from arc.training.scene_sampler import ScenePoolError, plan_pool, shard_plans
from arc.training.trainer_state import read_trainer_state
from test_extra_pool import DEFAULT_WEIGHTS, KUBRIC, PLAN_KWARGS, _clip, _cli, _extra
from test_scene_sampler import write_pool
from test_trainer_loop import _FakeArc, _loop_args, _plans, _validator_args

WORLD = 2
# Long enough for the slowest spawned rank to reach the next collective, short
# enough that a collective only one rank reaches fails within a test's time.
COLLECTIVE_TIMEOUT = datetime.timedelta(seconds=30)


# ----------------------------------------------------------------- fixtures ---


def _fake_arc():
    """The real finish_window's submodule surface, built alike on every rank
    from the same seed -- as main() builds the model from --seed."""

    torch.manual_seed(0)
    model = _FakeArc(1, 1, 1)
    model.freeze = "temporal_tracking"
    return model


def _trainable(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _adamw(model):
    return torch.optim.AdamW([{"params": _trainable(model), "lr": 1e-3}])


def _scaler():
    return torch.amp.GradScaler("cuda", enabled=False)


def _coefficients(plan, model):
    """A known gradient per plan and per tensor: distinct across both, nonzero
    on every trainable group so finish_window's zero-norm guards hold."""

    return [
        torch.full_like(parameter, 0.01 * (plan.step + 1) * (index + 1))
        for index, parameter in enumerate(_trainable(model))
    ]


def _accumulating_step(
    *, model, plan, optimizer, scaler, learning_rates, step, accum_steps,
    window_start, window_end, grad_clip, **_,
):
    """train_step's window contract with a known gradient: zero at the window's
    first executed micro-step, backward the loss over accum_steps, and close
    the window only when told to -- which above one rank is never."""

    if window_start:
        optimizer.zero_grad(set_to_none=True)
    loss = sum(
        (parameter * coefficient).sum()
        for parameter, coefficient in zip(_trainable(model), _coefficients(plan, model))
    )
    (loss / accum_steps).backward()
    norms = (
        train_cli.finish_window(
            model=model, optimizer=optimizer, scaler=scaler, grad_clip=grad_clip
        )
        if window_end
        else None
    )
    return train_cli.StepOutcome(
        step=step,
        seq_name=plan.seq_name,
        loss=float(loss.detach()),
        metric_error_m=0.0,
        sample_count=1,
        alignment_scale=1.0,
        alignment_residual_m=0.0,
        learning_rates=list(learning_rates),
        gradient_norms=norms,
    )


def _provider(lost=()):
    """Scenes for every plan but the global draw indices in ``lost``."""

    def load(plan):
        if plan.step in lost:
            raise SceneProviderError(
                f"scene {plan.seq_name!r} is not in the pool at '/root' (0 scenes)"
            )
        return SimpleNamespace(name=plan.seq_name)

    return load


def _shard(rank, num_steps):
    """This rank's share of a two-rank draw over the loop tests' plans."""

    return shard_plans(
        _plans(WORLD * num_steps), rank=rank, world_size=WORLD, num_steps=num_steps
    )


def _state(model, optimizer):
    """Parameters and optimizer state, for a bitwise comparison across ranks."""

    return {
        "parameters": {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        },
        "optimizer": {
            index: {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in state.items()
            }
            for index, state in optimizer.state_dict()["state"].items()
        },
    }


def _assert_same_state(first, second):
    assert first["parameters"].keys() == second["parameters"].keys()
    for name, value in first["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name
    assert first["optimizer"].keys() == second["optimizer"].keys()
    for index, state in first["optimizer"].items():
        for key, value in state.items():
            other = second["optimizer"][index][key]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other


def _train(rank, directory, *, args, lost=(), step_fn=_accumulating_step, **kwargs):
    """run_training on this rank's shard, its printed output captured."""

    model = _fake_arc()
    optimizer = _adamw(model)
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = train_cli.run_training(
            model=model,
            optimizer=optimizer,
            scaler=_scaler(),
            plans=_shard(rank, args.num_steps),
            args=args,
            scene_provider=_provider(lost),
            step_fn=step_fn,
            output_dir=directory / "out",
            **kwargs,
        )
    return model, optimizer, result, stdout.getvalue(), stderr.getvalue()


# ------------------------------------------------------ the two-rank harness ---


def _spawn(tmp_path, scenario, **options):
    """Run ``scenario`` on both ranks of a real gloo group; return each rank's result."""

    mp.spawn(
        _rank_main,
        args=(WORLD, str(tmp_path), scenario, options),
        nprocs=WORLD,
        join=True,
    )
    return [
        torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in range(WORLD)
    ]


def _rank_main(rank, world, directory, scenario, options):
    directory = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory / 'store'}",
        rank=rank,
        world_size=world,
        timeout=COLLECTIVE_TIMEOUT,
    )
    try:
        outcome = _SCENARIOS[scenario](rank, directory, **options)
        torch.save(outcome, directory / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _recording_finish(entries):
    """The real finish_window, recording the gradient it is handed -- the
    cross-rank average, before the clip."""

    def finish(*, model, optimizer, scaler, grad_clip):
        entries.append([parameter.grad.detach().clone() for parameter in _trainable(model)])
        return train_cli.finish_window(
            model=model, optimizer=optimizer, scaler=scaler, grad_clip=grad_clip
        )

    return finish


def _scenario_average(rank, directory):
    """Four windows at accum 1. Plans: rank 0 trains draws 0,2,4,6 and rank 1
    draws 1,3,5,7. Rank 1 loses draw 3 (window 1); both lose 4 and 5
    (window 2)."""

    entries = []
    model, optimizer, result, stdout, _ = _train(
        rank, directory,
        args=_loop_args(directory, num_steps=4, world_size=WORLD),
        lost={3, 4, 5},
        finish_window_fn=_recording_finish(entries),
    )
    return {
        "entries": entries,
        "state": _state(model, optimizer),
        "completed": result["completed_steps"],
        "stdout": stdout,
    }


def _scenario_average_accumulated(rank, directory):
    """Two windows at accum 2: rank 0 loses draw 2, the second micro-step of
    window 0, so that window averages three micro-steps over two ranks."""

    entries = []
    _train(
        rank, directory,
        args=_loop_args(directory, num_steps=4, grad_accum=2, world_size=WORLD),
        lost={2},
        finish_window_fn=_recording_finish(entries),
    )
    return {"entries": entries}


def _scenario_wait(rank, directory):
    """Rank 0 writes late; rank 1 must not pass the wait before it has."""

    marker = directory / "rank0-wrote"
    if rank == 0:
        time.sleep(1.0)
        marker.write_text("done")
    data_parallel.wait_for_every_rank()
    return {"seen": marker.exists()}


def _scenario_two_ranks_at_accum_one(rank, directory):
    model, _, _, _, _ = _train(
        rank, directory,
        args=_loop_args(directory, num_steps=3, warmup_steps=1, world_size=WORLD),
        finish_window_fn=train_cli.finish_window,
    )
    return {"parameters": {name: p.detach().clone() for name, p in model.named_parameters()}}


def _scenario_replicas(rank, directory):
    """Accum 2 over 8 steps per rank, every boundary saved: rank 0 loses draw 2
    (half of window 0) and finds draw 12 unsupervisable; rank 1 loses draws 5
    and 7 -- the whole of window 1 -- so its zero fill runs at accum 2."""

    def step_fn(**kwargs):
        if kwargs["plan"].step == 12:
            raise train_cli.UnsupervisableSceneError(
                "nothing", scene=kwargs["plan"].seq_name, eligibility={"total_query_count": 0}
            )
        return _accumulating_step(**kwargs)

    model, optimizer, result, stdout, _ = _train(
        rank, directory,
        args=_loop_args(directory, num_steps=8, grad_accum=2, save_every=2, world_size=WORLD),
        lost={2, 5, 7},
        step_fn=step_fn,
    )
    return {
        "state": _state(model, optimizer),
        "replica_checks": result["replica_checks"],
        "completed": result["completed_steps"],
        "executed": [outcome.step for outcome in result["history"]],
        "losses": {outcome.step: outcome.loss for outcome in result["history"]},
        "stdout": stdout,
    }


def _scenario_stop(rank, directory, signals):
    """The ranks named in ``signals`` receive their signal during window [2,3]
    at accum 2."""

    def step_fn(**kwargs):
        outcome = _accumulating_step(**kwargs)
        if rank in signals and kwargs["step"] == 2:
            train_cli._request_stop(getattr(signal, signals[rank]), None)
        return outcome

    _, _, result, stdout, _ = _train(
        rank, directory,
        args=_loop_args(directory, num_steps=8, grad_accum=2, world_size=WORLD),
        step_fn=step_fn,
    )
    return {
        "received": list(train_cli._STOP_REQUESTED),
        "interrupted": result["interrupted_by"],
        "executed": [outcome.step for outcome in result["history"]],
        "completed": result["completed_steps"],
        "stdout": stdout,
    }


def _scenario_interrupt_after_a_window_rank_zero_skipped(rank, directory):
    """Accum 1: rank 0 loses draw 6 (its loop step 3), which rank 1's draw 7
    trains alone; rank 1 asks to stop at that step."""

    def step_fn(**kwargs):
        outcome = _accumulating_step(**kwargs)
        if rank == 1 and kwargs["step"] == 3:
            train_cli._request_stop(signal.SIGTERM, None)
        return outcome

    _, _, result, _, _ = _train(
        rank, directory,
        args=_loop_args(directory, num_steps=8, world_size=WORLD),
        lost={6},
        step_fn=step_fn,
    )
    return {
        "interrupted": result["interrupted_by"],
        "completed": result["completed_steps"],
        "executed": [outcome.step for outcome in result["history"]],
    }


def _scenario_idle_owner(rank, directory):
    """Rank 0 loses every one of its draws (0, 2, 4); rank 1 trains all three."""

    model, optimizer, result, stdout, _ = _train(
        rank, directory,
        args=_loop_args(directory, num_steps=3, world_size=WORLD),
        lost={0, 2, 4},
    )
    return {
        "completed": result["completed_steps"],
        "state": _state(model, optimizer),
        "stdout": stdout,
    }


def _scenario_ownership(rank, directory):
    """Accum 1 over 4 steps, an eval every 2: rank 0 loses draw 2 (its loop
    step 1) and rank 1 finds draw 5 unsupervisable (its loop step 2)."""

    import arc.training as training_package

    evaluated, written, appended = [], [], []

    def fake_eval(*, step, **_):
        evaluated.append(step)
        return {
            "step": step, "scenes": 1, "skipped_scenes": [], "position_loss": 0.5,
            "metric_error_m": 0.1, "position_loss_shuffled": None, "confidence_tau": None,
        }

    # run_training imports the correspondence builder from the package inside
    # its preflight, and resolves the eval and both writers as module globals.
    training_package.build_anchor_correspondences = lambda scene: (
        SimpleNamespace(count=1),
        {"total_query_count": 1, "eligible_query_count": 1, "rejected": {}},
    )
    train_cli.evaluate_held_out = fake_eval
    real_write, real_append = train_cli._write_checkpoint, train_cli.append_step_history
    train_cli._write_checkpoint = lambda *a, **k: (written.append(k["step"]), real_write(*a, **k))[1]
    train_cli.append_step_history = lambda path, outcome, **k: (
        appended.append(outcome.step),
        real_append(path, outcome, **k),
    )[1]

    def step_fn(**kwargs):
        if kwargs["plan"].step == 5:
            raise train_cli.UnsupervisableSceneError(
                "nothing to supervise", scene=kwargs["plan"].seq_name,
                eligibility={"total_query_count": 0},
            )
        outcome = _accumulating_step(**kwargs)
        if kwargs["plan"].step == 7:
            # Rank 1's last step drops two confidence samples: its own warning.
            outcome.confidence_dropped = {
                "total": 2, "target_nonfinite": 0, "prediction_nonfinite": 0,
                "confidence_nonfinite": 2,
            }
        return outcome

    # Boundaries every 3 steps: the step-3 write is a boundary's, the step-4
    # write the tail's, so each writer is seen on its own.
    _, _, result, stdout, stderr = _train(
        rank, directory,
        args=_loop_args(directory, num_steps=4, eval_every=2, save_every=3, world_size=WORLD),
        lost={2},
        step_fn=step_fn,
        held_out=train_cli.HeldOut(
            plans=[_plans(1)[0]],
            scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
            benchmark=None,
        ),
    )
    return {
        "stdout": stdout,
        "stderr": stderr,
        "evaluated": evaluated,
        "written": written,
        "appended": appended,
        "losses": {outcome.step: outcome.loss for outcome in result["history"]},
        "result": {
            key: result[key]
            for key in (
                "scene_load_skips", "unsupervisable_steps", "scene_cache",
                "eligibility_totals", "velocity_term_totals", "realized_anchor_counts",
                "confidence_dropped_totals", "peak_gpu_memory_bytes", "completed_steps",
                "replica_checks",
            )
        },
    }


def _scenario_diverged(rank, directory):
    """Rank 1 nudges one weight during step 1; the step-2 checkpoint compares."""

    def step_fn(**kwargs):
        if rank == 1 and kwargs["step"] == 1:
            with torch.no_grad():
                kwargs["model"].track_head.bias.add_(1e-3)
        return _accumulating_step(**kwargs)

    try:
        _train(
            rank, directory,
            args=_loop_args(directory, num_steps=4, save_every=2, world_size=WORLD),
            step_fn=step_fn,
        )
    except RuntimeError as error:
        return {"error": str(error)}
    return {"error": None}


def _scenario_resumed_apart(rank, directory, checkpoints):
    try:
        _train(
            rank, directory,
            args=_loop_args(directory, num_steps=8, world_size=WORLD, resume=checkpoints[rank]),
        )
    except RuntimeError as error:
        return {"error": str(error)}
    return {"error": None}


_SCENARIOS = {
    "average": _scenario_average,
    "average_accumulated": _scenario_average_accumulated,
    "wait": _scenario_wait,
    "two_ranks_at_accum_one": _scenario_two_ranks_at_accum_one,
    "replicas": _scenario_replicas,
    "stop": _scenario_stop,
    "interrupt_after_a_window_rank_zero_skipped": _scenario_interrupt_after_a_window_rank_zero_skipped,
    "idle_owner": _scenario_idle_owner,
    "ownership": _scenario_ownership,
    "diverged": _scenario_diverged,
    "resumed_apart": _scenario_resumed_apart,
}


# ------------------------------------------------------------- the launch ---


def test_world_size_comes_from_the_environment_and_a_malformed_one_is_refused():
    assert world_size_from_environment({}) == 1
    assert world_size_from_environment({"WORLD_SIZE": "1"}) == 1
    assert world_size_from_environment({"WORLD_SIZE": "4"}) == 4
    for value in ("0", "-1", "two", "1.5", " 2", "+2", "1_0", ""):
        with pytest.raises(ValueError, match="WORLD_SIZE"):
            world_size_from_environment({"WORLD_SIZE": value})


def test_the_launch_variables_are_named_when_missing_and_range_checked():
    full = {"RANK": "1", "LOCAL_RANK": "1", "MASTER_ADDR": "node7", "MASTER_PORT": "29500"}
    assert read_launch_environment(full, world_size=2) == Launch(rank=1, local_rank=1, world_size=2)

    partial = {"RANK": "1", "MASTER_ADDR": "node7"}
    with pytest.raises(ValueError) as excinfo:
        read_launch_environment(partial, world_size=2)
    message = str(excinfo.value)
    assert "needs LOCAL_RANK, MASTER_PORT exported" in message
    assert "LOCAL_RANK ($SLURM_LOCALID)" in message and "RANK ($SLURM_PROCID)" not in message

    for overrides, fragment in (
        ({"RANK": "2"}, "RANK 2 is outside WORLD_SIZE 2"),
        ({"LOCAL_RANK": "-1"}, "LOCAL_RANK must be a non-negative decimal"),
        ({"MASTER_PORT": "0"}, r"MASTER_PORT must be in \[1, 65535\]"),
        ({"MASTER_PORT": "70000"}, r"MASTER_PORT must be in \[1, 65535\]"),
        ({"MASTER_PORT": "http"}, "MASTER_PORT must be a non-negative decimal"),
    ):
        with pytest.raises(ValueError, match=fragment):
            read_launch_environment({**full, **overrides}, world_size=2)


def test_joining_binds_the_gpu_before_the_group_and_refuses_what_cannot_bind(monkeypatch):
    """The device first, or the model's .to("cuda") lands on GPU 0 next to rank
    0's; and a launch that binds one GPU per task is refused by name."""

    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.append(("set_device", index)))
    monkeypatch.setattr(
        data_parallel.dist, "init_process_group",
        lambda backend, **kwargs: calls.append(("init_process_group", backend, kwargs)),
    )
    launch = Launch(rank=1, local_rank=1, world_size=2)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="sees no CUDA device"):
        join_process_group(launch, timeout_minutes=7)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="--gpus-per-node equal to --ntasks-per-node"):
        join_process_group(launch, timeout_minutes=7)
    assert calls == [], "a refused task must not have touched a device or the group"

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    join_process_group(launch, timeout_minutes=7)
    assert calls == [
        ("set_device", 1),
        (
            "init_process_group",
            "nccl",
            {"init_method": "env://", "timeout": datetime.timedelta(minutes=7)},
        ),
    ]


def _pool_args(tmp_path, **overrides):
    return _validator_args(tmp_path, manifest=None, scene_pool=str(tmp_path / "pool"), **overrides)


def test_two_ranks_refuse_a_manifest_16_mixed_and_a_fresh_auto_alpha(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="--manifest cannot train under WORLD_SIZE 2"):
        train_cli._validate_args(_validator_args(tmp_path))
    with pytest.raises(ValueError, match="--precision 16-mixed cannot train under WORLD_SIZE 2"):
        train_cli._validate_args(_pool_args(tmp_path, precision="16-mixed"))
    with pytest.raises(ValueError, match="--confidence_alpha auto cannot start a run under WORLD_SIZE 2"):
        train_cli._validate_args(_pool_args(tmp_path, confidence_weight=0.1))

    # What passes: the default flags -- auto at weight 0 resolves nothing, so
    # every rank agrees on None -- an explicit alpha, a resume (which restores
    # the pinned value from the one file every rank reads), and bf16 or 32.
    for overrides in (
        {},
        {"confidence_weight": 0.1, "confidence_alpha": "0.5"},
        {"confidence_weight": 0.1, "resume": "train_state.pt"},
        {"precision": "32"},
    ):
        args = _pool_args(tmp_path, **overrides)
        train_cli._validate_args(args)
        assert args.world_size == 2

    # One process refuses none of them.
    monkeypatch.delenv("WORLD_SIZE")
    for args in (
        _validator_args(tmp_path),
        _pool_args(tmp_path, precision="16-mixed"),
        _pool_args(tmp_path, confidence_weight=0.1),
    ):
        train_cli._validate_args(args)
        assert args.world_size == 1


def test_a_malformed_world_size_and_the_timeout_bound_are_refused_at_parse_time(tmp_path, monkeypatch):
    defaults = train_cli.build_arg_parser().parse_args(["--manifest", "m.jsonl"])
    assert defaults.collective_timeout_minutes == 120
    with pytest.raises(ValueError, match="--collective_timeout_minutes must be at least 1"):
        train_cli._validate_args(_validator_args(tmp_path, collective_timeout_minutes=0))
    monkeypatch.setenv("WORLD_SIZE", "0")
    with pytest.raises(ValueError, match="WORLD_SIZE must be at least 1"):
        train_cli._validate_args(_validator_args(tmp_path))


def test_world_size_is_a_refused_resume_key_whose_absence_means_one(tmp_path):
    stored = train_cli._checkpoint_settings(_loop_args(tmp_path, world_size=2))
    with pytest.raises(RuntimeError, match="carries world_size=2"):
        train_cli.check_resume_settings(stored, _loop_args(tmp_path))
    train_cli.check_resume_settings(stored, _loop_args(tmp_path, world_size=2))

    # A checkpoint from before the key existed was written by one process.
    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    del stored["world_size"]
    train_cli.check_resume_settings(stored, _loop_args(tmp_path))
    with pytest.raises(RuntimeError, match="predates 'world_size'"):
        train_cli.check_resume_settings(stored, _loop_args(tmp_path, world_size=2))


# ---------------------------------------------------------------- the plans ---


def _draw(num_steps, *, extra=None):
    return plan_pool(
        KUBRIC, data_root="/kubric", num_steps=num_steps, seed=7, weights=DEFAULT_WEIGHTS,
        transform_probability=1.0, extra=extra, **PLAN_KWARGS,
    ).planned


@pytest.mark.parametrize("mixed", [False, True], ids=["single_pool", "extra_pool"])
def test_each_rank_trains_every_nth_plan_of_one_draw(mixed):
    world, num_steps = 3, 40
    extra = _extra(weight=0.4) if mixed else None
    draw = _draw(world * num_steps, extra=extra)
    shards = [
        shard_plans(draw, rank=rank, world_size=world, num_steps=num_steps)
        for rank in range(world)
    ]
    for rank, shard in enumerate(shards):
        assert len(shard) == num_steps
        # The plan keeps its global draw index.
        assert [plan.step for plan in shard] == [k * world + rank for k in range(num_steps)]
        if mixed:
            # The pool draw rides in the stream, so every shard mixes both.
            assert {plan.data_root for plan in shard} == {"/kubric", "/clips"}
    together = sorted((plan for shard in shards for plan in shard), key=lambda plan: plan.step)
    assert together == draw

    # A run extended through --num_steps keeps every rank's earlier steps.
    longer = _draw(world * (num_steps + 10), extra=extra)
    for rank in range(world):
        extended = shard_plans(longer, rank=rank, world_size=world, num_steps=num_steps + 10)
        assert extended[:num_steps] == shards[rank]


def test_a_draw_of_the_wrong_length_or_a_rank_outside_the_world_is_refused():
    draw = _draw(8)
    with pytest.raises(ScenePoolError, match="exactly 8 plans, but the draw holds 7"):
        shard_plans(draw[:-1], rank=0, world_size=2, num_steps=4)
    with pytest.raises(ScenePoolError, match="rank 2 is outside a world of 2"):
        shard_plans(draw, rank=2, world_size=2, num_steps=4)


@pytest.mark.parametrize("mixed", [False, True], ids=["single_pool", "extra_pool"])
def test_plan_only_plans_the_n_fold_draw_and_records_it_only_above_one_rank(
    tmp_path, monkeypatch, capsys, mixed
):
    kubric = write_pool(tmp_path / "kubric", {"0000": 10, "0001": 6, "0002": 4})
    flags = ["--scene_pool", str(kubric)]
    if mixed:
        clips = tmp_path / "clips"
        _clip(clips, "a__0")
        flags += ["--extra_pool", str(clips)]
    _cli(monkeypatch, *flags, "--build_pool_index")
    if mixed:
        flags += ["--extra_pool_weight", "0.4"]

    asked = []
    real_plan_pool = train_cli.plan_pool
    monkeypatch.setattr(
        train_cli, "plan_pool",
        lambda *a, **k: (asked.append(k["num_steps"]), real_plan_pool(*a, **k))[1],
    )

    def plan(name):
        out = tmp_path / name
        _cli(
            monkeypatch, *flags, "--plan_only", "--num_steps", "6", "--warmup_steps", "2",
            "--grad_accum", "2", "--json_out", str(out),
        )
        return json.loads(out.read_text())

    one = plan("one.json")
    monkeypatch.setenv("WORLD_SIZE", "3")
    three = plan("three.json")
    printed = capsys.readouterr().out

    assert asked == [6, 18], "one process draws num_steps itself; three draw three times it"
    assert one["planned_steps"] == 6 and three["planned_steps"] == 18
    assert "world_size" not in one["settings"]
    assert "scenes_per_optimizer_step" not in one["settings"]
    assert three["settings"]["world_size"] == 3
    assert three["settings"]["scenes_per_optimizer_step"] == 6
    # Nothing else in plan.json moves with the world size but the draw it plans.
    assert set(three) == set(one)
    assert set(three["settings"]) == set(one["settings"]) | {"world_size", "scenes_per_optimizer_step"}
    assert "PASS planned 18 steps" in printed


def test_main_joins_before_the_model_moves_and_trains_its_shard_of_one_draw():
    """main() cannot run without a GPU, so its wiring is read, as the other main
    pins read it: the group is joined before the model moves to the device, the
    draw is WORLD_SIZE-fold, and run_training trains the shard taken after it."""

    tree = ast.parse(Path(train_cli.__file__).read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            calls.setdefault(name, []).append(node)

    def guard(call):
        """The test of the innermost `if` whose body holds ``call``."""

        enclosing = [
            node for node in ast.walk(main)
            if isinstance(node, ast.If)
            and any(child is call for statement in node.body for child in ast.walk(statement))
        ]
        return ast.unparse(min(enclosing, key=lambda node: node.end_lineno - node.lineno).test)

    def spelled(source):
        return ast.unparse(ast.parse(source, mode="eval").body)

    (join,) = calls["join_process_group"]
    assert guard(join) == spelled("args.world_size > 1 and not args.plan_only")
    to_device = [
        node for node in calls["to"]
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "cuda"
    ]
    assert to_device and all(join.lineno < node.lineno for node in to_device)
    (draw,) = calls["plan_pool"]
    steps = next(keyword.value for keyword in draw.keywords if keyword.arg == "num_steps")
    assert ast.unparse(steps) == "args.num_steps * args.world_size"
    (shard,) = calls["shard_plans"]
    assert guard(shard) == spelled("args.world_size > 1")
    assert shard.lineno > draw.lineno
    assert ast.unparse(shard.args[0]) == "tally.planned"
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in shard.keywords
    } == {"rank": "get_rank()", "world_size": "args.world_size", "num_steps": "args.num_steps"}
    (run,) = calls["run_training"]
    plans = next(keyword.value for keyword in run.keywords if keyword.arg == "plans")
    assert isinstance(plans, ast.Name) and plans.id == "plans"

    # Who owns the files, read off the group: decided after the join, so a
    # rank's place is the group's rank and not every task's default of 0.
    (owner,) = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "owner" for t in node.targets)
    ]
    assert ast.unparse(owner.value) == "is_main_process()" and owner.lineno > join.lineno
    writes = calls["write_text"]
    (plan_json,) = [node for node in writes if "args.json_out" in ast.unparse(node.func)]
    assert guard(plan_json) == spelled("args.json_out and owner")
    (summary_json,) = [node for node in writes if "run_summary.json" in ast.unparse(node.func)]
    # Every other rank leaves before run_summary.json, through the last wait;
    # rank 0 enters the same wait only after its own writes.
    (leave,) = [
        node for node in ast.walk(main)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not owner"
    ]
    assert isinstance(leave.body[-1], ast.Return) and leave.end_lineno < summary_json.lineno
    waits = sorted(calls["wait_for_every_rank"], key=lambda node: node.lineno)
    destroys = sorted(calls["destroy_process_group"], key=lambda node: node.lineno)
    assert len(waits) == len(destroys) == 2
    assert leave.lineno < waits[0].lineno < destroys[0].lineno < leave.end_lineno
    assert summary_json.lineno < waits[1].lineno < destroys[1].lineno
    assert guard(waits[1]) == spelled("args.world_size > 1")


# --------------------------------------------------------- one rank, untouched ---


def test_one_rank_never_reaches_the_collective_layer(tmp_path, monkeypatch, capsys):
    """The single-process run is today's code path: with every collective, and
    every data-parallel entry the trainer imports, made to raise, a run with
    skips, accumulation, boundaries and a partial window still completes -- and
    prints and records exactly what it did before the ranks existed."""

    def refuse(*args, **kwargs):
        raise AssertionError("a single-process run reached the collective layer")

    for name in ("all_reduce", "all_gather_object", "init_process_group", "barrier", "broadcast"):
        monkeypatch.setattr(dist, name, refuse)
    for name in (
        "agree_on_stop", "close_window_across_ranks", "check_replicas", "gather_objects",
        "reduce_window_totals", "average_gradients", "join_process_group",
    ):
        monkeypatch.setattr(train_cli, name, refuse)

    def step_fn(**kwargs):
        if kwargs["plan"].step == 5:
            raise train_cli.UnsupervisableSceneError(
                "nothing", scene=kwargs["plan"].seq_name, eligibility={"total_query_count": 0}
            )
        return _accumulating_step(**kwargs)

    model = _fake_arc()
    result = train_cli.run_training(
        model=model,
        optimizer=_adamw(model),
        scaler=_scaler(),
        plans=_plans(7),
        args=_loop_args(tmp_path, num_steps=7, grad_accum=2, save_every=2),
        scene_provider=_provider(lost={3}),
        step_fn=step_fn,
        output_dir=tmp_path,
    )
    assert result["completed_steps"] == 7
    assert "replica_checks" not in result
    assert result["peak_gpu_memory_bytes"] == 0
    assert [set(record) for record in result["unsupervisable_steps"]] == [
        {"step", "scene", "reason", "eligibility"}
    ]
    printed = capsys.readouterr().out
    assert "step=3 scene=0000 skipped:" in printed and "step=5 scene=0002 skipped:" in printed
    assert "rank" not in printed and "world_loss" not in printed


def test_an_executing_rank_checks_its_own_gradients_before_any_collective(monkeypatch):
    """The zero fill gives every parameter a gradient, so a disconnected one on a
    rank that executed must be refused on that rank's OWN gradients, before the
    totals -- after them nothing could tell it from a skipped rank's zeros."""

    def refuse(**kwargs):
        raise AssertionError("reached the collective layer")

    monkeypatch.setattr(train_cli, "reduce_window_totals", refuse)
    model = _fake_arc()
    for parameter in _trainable(model):
        parameter.grad = torch.ones_like(parameter)
    model.track_head.bias.grad = None
    close = dict(
        model=model, optimizer=_adamw(model), scaler=_scaler(), grad_clip=1.0,
        finish_window_fn=train_cli.finish_window,
    )
    with pytest.raises(RuntimeError, match=r"missing gradients: \['track_head.bias'\]"):
        train_cli.close_window_across_ranks(executed_steps=1, loss_sum=0.5, **close)
    # A rank that executed nothing has no gradient of its own to judge.
    with pytest.raises(AssertionError, match="reached the collective layer"):
        train_cli.close_window_across_ranks(executed_steps=0, loss_sum=0.0, **close)


def test_a_world_size_the_process_group_does_not_hold_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(train_cli, "get_world_size", lambda: 1)
    model = _fake_arc()
    with pytest.raises(RuntimeError, match="WORLD_SIZE is 2 but the process group holds 1 rank"):
        train_cli.run_training(
            model=model,
            optimizer=_adamw(model),
            scaler=_scaler(),
            plans=_plans(2),
            args=_loop_args(tmp_path, num_steps=2, world_size=2),
            scene_provider=_provider(),
            step_fn=_accumulating_step,
            output_dir=tmp_path,
        )


# ------------------------------------------------------------ the helpers ---


def test_the_average_runs_in_bounded_buckets_and_a_skipped_rank_contributes_zeros(monkeypatch):
    reduced = []

    def fake_all_reduce(tensor, op=None):
        # A second rank holding the same values: the SUM doubles them.
        reduced.append(tensor.detach().clone())
        tensor.mul_(2)

    monkeypatch.setattr(data_parallel.dist, "all_reduce", fake_all_reduce)
    shapes = [(3,), (40,), (2, 2), (5,), (1,), (30,)]
    parameters = [nn.Parameter(torch.randn(shape)) for shape in shapes]
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    expected = [parameter.grad.clone() for parameter in parameters]

    data_parallel.average_gradients(parameters, contributed=True, executed_ranks=2, bucket_bytes=64)

    for parameter, value in zip(parameters, expected):
        assert torch.equal(parameter.grad, value), "2g / 2 is g exactly"
    sizes = [tensor.numel() * tensor.element_size() for tensor in reduced]
    single = {parameter.numel() * 4 for parameter in parameters}
    total = sum(parameter.numel() * 4 for parameter in parameters)
    assert sum(sizes) == total, "every gradient element is reduced exactly once"
    assert all(size <= 64 or size in single for size in sizes), (
        f"a bucket above the bound must be a lone tensor, got {sizes}"
    )
    assert len(reduced) < len(parameters), "not one call per tensor"
    assert max(sizes) < total, "not one full-size flat copy"

    # A rank that executed nothing: its stale average must not enter the sum,
    # and a parameter it never computed a gradient for gets one.
    reduced.clear()
    stale = nn.Parameter(torch.randn(4))
    stale.grad = torch.full((4,), 7.0)
    untouched = nn.Parameter(torch.randn(3))
    data_parallel.average_gradients(
        [stale, untouched], contributed=False, executed_ranks=1, bucket_bytes=64
    )
    assert all(not tensor.any() for tensor in reduced)
    assert torch.equal(stale.grad, torch.zeros(4))
    assert torch.equal(untouched.grad, torch.zeros(3))


def test_the_digest_moves_with_one_flipped_bit():
    parameters = [nn.Parameter(torch.randn(5)), nn.Parameter(torch.randn(2, 3))]
    before = data_parallel.parameter_digest(parameters)
    parameters[1].data.view(torch.int32)[0, 1] ^= 1
    after = data_parallel.parameter_digest(parameters)
    assert before[0] == after[0] and before[1] != after[1]


def _rank_totals(**overrides):
    totals = {
        "attempted": 4,
        "last_executed": 3,
        "scene_load_skips": {"scene_absent": 1},
        "pool_scene_skips": {"scene_pool": {}, "extra_pool": {"other": 2}},
        "unsupervisable_steps": [{"step": 1, "rank": 0}],
        "eligibility_totals": {
            "steps_counted": 2, "total_query_count": 10, "eligible_query_count": 8,
            "rejected": {"invisible": 2, "out_of_window": 0},
        },
        "anchor_count_steps": 2,
        "seated_anchor_counts": {1: 2},
        "confidence_dropped_totals": {"total": 0, "target_nonfinite": 0},
        "velocity_steps_supervised": 1,
        "velocity_steps_without_a_pair": 1,
        "scene_cache": {"hits": 1, "misses": 3},
        "peak_bytes": 100,
    }
    totals.update(overrides)
    return totals


def test_the_run_totals_combine_by_sum_concatenation_and_maximum():
    first = _rank_totals()
    second = _rank_totals(
        last_executed=4,
        scene_load_skips={"other": 1, "scene_absent": 2},
        pool_scene_skips={"scene_pool": {"other": 1}, "extra_pool": {}},
        unsupervisable_steps=[{"step": 2, "rank": 1}],
        seated_anchor_counts={1: 1, 2: 1},
        confidence_dropped_totals={"total": 3, "target_nonfinite": 3},
        peak_bytes=90,
    )
    combined = train_cli._combine_run_totals([first, second])
    assert combined["attempted"] == 8
    assert combined["last_executed"] == 4
    assert combined["scene_load_skips"] == {"scene_absent": 3, "other": 1}
    assert combined["pool_scene_skips"] == {
        "scene_pool": {"other": 1}, "extra_pool": {"other": 2},
    }
    assert combined["unsupervisable_steps"] == [{"step": 1, "rank": 0}, {"step": 2, "rank": 1}]
    assert combined["eligibility_totals"] == {
        "steps_counted": 4, "total_query_count": 20, "eligible_query_count": 16,
        "rejected": {"invisible": 4, "out_of_window": 0},
    }
    assert combined["seated_anchor_counts"] == {1: 3, 2: 1}
    assert combined["confidence_dropped_totals"] == {"total": 3, "target_nonfinite": 3}
    assert combined["velocity_steps_supervised"] == 2
    assert combined["scene_cache"] == {"hits": 2, "misses": 6}
    assert combined["peak_bytes"] == 100
    assert train_cli._combine_run_totals([_rank_totals(last_executed=None)] * 2)[
        "last_executed"
    ] is None


def test_one_ranks_totals_come_back_as_they_went_in():
    """Zeros kept, key order kept: the single-process run_summary.json is the one
    it always was, which Counter addition (it drops zeros) would not give."""

    entry = _rank_totals()
    combined = train_cli._combine_run_totals([entry])
    assert combined == entry
    assert list(combined["eligibility_totals"]) == list(entry["eligibility_totals"])
    assert list(combined["eligibility_totals"]["rejected"]) == ["invisible", "out_of_window"]
    assert json.dumps(combined["eligibility_totals"]) == json.dumps(entry["eligibility_totals"])


# ---------------------------------------------------------------- two ranks ---


def test_the_average_is_the_mean_over_the_ranks_that_executed(tmp_path):
    """Window 0: both ranks. Window 1: rank 0 alone -- rank 1 lost its scene and
    still holds window 0's average, which must stay out. Window 2: no rank --
    no step anywhere, so finish_window runs three times, not four."""

    ranks = _spawn(tmp_path, "average")
    model = _fake_arc()
    plans = _plans(8)

    def coefficients(step):
        return _coefficients(plans[step], model)

    def mean(first, second):
        return [(a + b) / 2 for a, b in zip(coefficients(first), coefficients(second))]

    expected = [mean(0, 1), coefficients(2), mean(6, 7)]
    for rank in ranks:
        assert len(rank["entries"]) == 3
        for window, (entry, want) in enumerate(zip(rank["entries"], expected)):
            for got, value in zip(entry, want):
                assert torch.equal(got, value), f"window {window}"
        assert rank["completed"] == 4
    _assert_same_state(ranks[0]["state"], ranks[1]["state"])
    adam_steps = {float(state["step"]) for state in ranks[0]["state"]["optimizer"].values()}
    assert adam_steps == {3.0}, "the window no rank executed must not step"
    assert "step=2/4 ranks=0/2" in ranks[0]["stdout"]


def test_the_average_divides_by_the_ranks_not_the_micro_steps_at_accum_two(tmp_path):
    """At accum 1 the executing ranks and the executed micro-steps are the same
    count, so only accumulation tells the divisor apart: each rank's window
    gradient already carries its own 1/accum, and the average then divides by
    the RANKS that executed -- a partial window on one rank still counts once.
    Compared exactly, because AdamW would hide a uniformly scaled gradient."""

    ranks = _spawn(tmp_path, "average_accumulated")
    model = _fake_arc()
    plans = _plans(8)

    def accumulated(*steps):
        """One rank's window gradient: each micro-step's c/2, summed in order."""

        total = None
        for step in steps:
            halves = [coefficient / 2 for coefficient in _coefficients(plans[step], model)]
            total = halves if total is None else [a + b for a, b in zip(total, halves)]
        return total

    def mean(first, second):
        return [(a + b) / 2 for a, b in zip(first, second)]

    # Rank 0 trains draws 0,2 | 4,6 and lost 2; rank 1 trains 1,3 | 5,7.
    expected = [
        mean(accumulated(0), accumulated(1, 3)),
        mean(accumulated(4, 6), accumulated(5, 7)),
    ]
    for rank in ranks:
        assert len(rank["entries"]) == 2
        for window, (entry, want) in enumerate(zip(rank["entries"], expected)):
            for got, value in zip(entry, want):
                assert torch.equal(got, value), f"window {window}"


def test_the_last_wait_holds_every_rank_until_all_have_reached_it(tmp_path):
    """What main() leaves through: the other ranks wait there until rank 0 has
    finished its writes, rather than exiting under them."""

    ranks = _spawn(tmp_path, "wait")
    assert [rank["seen"] for rank in ranks] == [True, True]


def test_two_ranks_match_one_rank_accumulating_the_same_draws(tmp_path):
    """The semantics, not just agreement between ranks: two ranks at accum 1
    take the same steps as one rank averaging the same two draws per window at
    accum 2, under the same schedule (3 optimizer steps, 1 of warmup)."""

    ranks = _spawn(tmp_path, "two_ranks_at_accum_one")
    reference = _fake_arc()
    train_cli.run_training(
        model=reference,
        optimizer=_adamw(reference),
        scaler=_scaler(),
        plans=_plans(6),
        args=_loop_args(tmp_path / "one", num_steps=6, warmup_steps=2, grad_accum=2),
        scene_provider=_provider(),
        step_fn=_accumulating_step,
        output_dir=tmp_path / "one",
    )
    for rank in ranks:
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(rank["parameters"][name], parameter.detach(), rtol=0, atol=0)


def test_replicas_and_optimizer_state_stay_identical_through_skips_and_accumulation(tmp_path):
    ranks = _spawn(tmp_path, "replicas")
    _assert_same_state(ranks[0]["state"], ranks[1]["state"])
    for rank in ranks:
        # Before the first step, at the checkpoints after 2, 4, 6 and 8, and at
        # the end.
        assert rank["replica_checks"] == 6
        assert rank["completed"] == 8
    assert ranks[0]["executed"] == [0, 2, 3, 4, 5, 7]
    assert ranks[1]["executed"] == [0, 1, 4, 5, 6, 7]
    # Window [0,1] ran three micro-steps on two ranks, so its mean is over the
    # three -- summed as the reduction sums, rank 0's then rank 1's -- and rank
    # 0, which lost its closing micro-step, carries it on the short line.
    own, other = ranks[0]["losses"], ranks[1]["losses"]
    window = (own[0] + (other[0] + other[1])) / 3
    assert f"step=1/8 world_loss={window:.8f} ranks=2/2" in ranks[0]["stdout"]


@pytest.mark.parametrize(
    "signals, agreed",
    [
        ({1: "SIGUSR1"}, "SIGUSR1"),
        # The usual case: Slurm signals every task alike.
        ({0: "SIGUSR1", 1: "SIGUSR1"}, "SIGUSR1"),
        ({0: "SIGUSR1", 1: "SIGTERM"}, "SIGTERM"),
    ],
    ids=["one_rank", "every_rank_alike", "ranks_differ"],
)
def test_a_stop_on_any_rank_stops_every_rank_at_the_same_window_top(tmp_path, signals, agreed):
    ranks = _spawn(tmp_path, "stop", signals=signals)
    assert [rank["received"] for rank in ranks] == [
        [signals[index]] if index in signals else [] for index in range(WORLD)
    ]
    for rank in ranks:
        # One name every rank agrees on, whichever ranks the signals reached.
        assert rank["interrupted"] == agreed
        # The flag landed inside window [2,3]: both ranks close it, then break.
        assert rank["executed"] == [0, 1, 2, 3]
        assert rank["completed"] == 4
    assert read_trainer_state(tmp_path / "out" / "train_state.pt")["step"] == 4
    assert f"{agreed} received; checkpointing at step 4 and exiting 0" in ranks[0]["stdout"]
    assert "received" not in ranks[1]["stdout"]


def test_the_interrupted_checkpoint_follows_the_last_window_that_stepped_on_any_rank(tmp_path):
    """Rank 0's own history ends at step 2, but window 3 stepped on rank 1's
    draw: recording 3 would make the resume replay a window that already moved
    the weights."""

    ranks = _spawn(tmp_path, "interrupt_after_a_window_rank_zero_skipped")
    assert ranks[0]["executed"] == [0, 1, 2]
    for rank in ranks:
        assert rank["interrupted"] == "SIGTERM"
        assert rank["completed"] == 4
    assert read_trainer_state(tmp_path / "out" / "train_state.pt")["step"] == 4


def test_a_clean_end_counts_windows_only_other_ranks_executed(tmp_path):
    ranks = _spawn(tmp_path, "idle_owner")
    for rank in ranks:
        assert rank["completed"] == 3
    _assert_same_state(ranks[0]["state"], ranks[1]["state"])
    assert not torch.equal(
        ranks[0]["state"]["parameters"]["track_head.bias"],
        _fake_arc().track_head.bias.detach(),
    ), "rank 0 executed nothing, yet its weights must move with the average"
    assert read_trainer_state(tmp_path / "out" / "train_state.pt")["step"] == 3
    # Rank 0 executed nothing and rank 1 writes no rows: no row exists at all.
    assert not (tmp_path / "out" / "history.jsonl").exists()
    assert ranks[0]["stdout"].count("ranks=1/2") == 3


def test_rank_zero_alone_writes_evaluates_and_prints_and_the_totals_combine(tmp_path):
    ranks = _spawn(tmp_path, "ownership")
    owner, other = ranks

    assert owner["evaluated"] == [2, 4] and other["evaluated"] == []
    # The step-3 boundary, then the tail at 4: both rank 0's, neither rank 1's.
    assert owner["written"] == [3, 4] and other["written"] == []
    assert owner["appended"] == [0, 2, 3] and other["appended"] == []
    rows = [
        json.loads(row)
        for row in (tmp_path / "out" / "history.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [0, 2, 3]
    # Every row closes a window at accum 1, so each carries the averaged
    # gradient's norms, as a single-process closing row carries its own.
    assert all(row["gradient_norms"] and "clipped_total" in row["gradient_norms"] for row in rows)

    printed = owner["stdout"]
    window_zero = (owner["losses"][0] + other["losses"][0]) / 2
    assert f" world_loss={window_zero:.8f} ranks=2/2" in printed
    # Rank 0 lost step 1's scene: its skip line, then the window's short line.
    assert "step=1 scene=0002 rank=0 skipped: scene '0002' is not in the pool" in printed
    assert f"step=1/4 world_loss={other['losses'][1]:.8f} ranks=1/2" in printed
    # Rank 1 found step 2 unsupervisable: rank 0's own line carries ranks=1/2.
    assert f"world_loss={owner['losses'][2]:.8f} ranks=1/2" in printed
    assert "eval step=2" in printed and "eval step=4" in printed
    assert "unsupervisable_steps=1/8" in printed

    assert "held_out_preflight_ok=1/1" in printed

    quiet = other["stdout"]
    assert "step=2 scene=0002 rank=1 skipped: nothing to supervise" in quiet
    assert "world_loss" not in quiet and "loss=" not in quiet
    assert "eval step" not in quiet and "unsupervisable_steps=" not in quiet
    assert "held_out_preflight" not in quiet
    # A rank's own warning stays on that rank, and names it.
    assert "WARNING step=3 rank=1: 2 sparse sample(s)" in other["stderr"]
    assert "WARNING" not in owner["stderr"]

    for rank in ranks:
        result = rank["result"]
        assert result["scene_load_skips"] == {"scene_absent": 1}
        assert result["unsupervisable_steps"] == [
            {
                "step": 2, "scene": "0002", "reason": "no_supervised_sample",
                "eligibility": {"total_query_count": 0}, "rank": 1,
            }
        ]
        assert result["scene_cache"] == {"hits": 0, "misses": 8}
        assert result["confidence_dropped_totals"] == {
            "confidence_nonfinite": 2, "prediction_nonfinite": 0, "target_nonfinite": 0,
            "total": 2,
        }
        assert result["completed_steps"] == 4
        # Before the first step, at the step-3 boundary, and at the end.
        assert result["replica_checks"] == 3
        assert result == owner["result"], "every rank returns the run's totals"


def test_replicas_that_diverged_are_refused_on_every_rank_before_any_write(tmp_path):
    ranks = _spawn(tmp_path, "diverged")
    error = ranks[0]["error"]
    assert error is not None and ranks[1]["error"] == error
    assert "the replicas differ at the step 2 checkpoint" in error
    assert "rank 1: 1 of 5 trainable tensors differ" in error
    assert not (tmp_path / "out" / "train_state.pt").exists()


def test_ranks_that_resumed_different_states_are_refused_before_the_first_step(tmp_path):
    checkpoints = []
    for step, nudge in ((2, 0.0), (4, 1e-3)):
        model = _fake_arc()
        with torch.no_grad():
            model.track_head.bias.add_(nudge)
        path = train_cli._write_checkpoint(
            model, _adamw(model), _scaler(), [1e-3], step=step,
            output_dir=tmp_path / f"saved-{step}", args=_loop_args(tmp_path, world_size=WORLD),
        )
        checkpoints.append(str(path))

    ranks = _spawn(tmp_path, "resumed_apart", checkpoints=checkpoints)
    error = ranks[0]["error"]
    assert error is not None and ranks[1]["error"] == error
    assert "the replicas differ before the first step" in error
    assert "rank 1: 1 of 5 trainable tensors differ; start_step=4 against 2" in error
