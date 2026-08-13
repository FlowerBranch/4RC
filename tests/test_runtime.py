"""Pins for the helpers shared by every bounded training entry point.

Two things are being protected here. The extraction of these helpers out of
``overfit_temporal_tracking.py`` must not have changed the harness -- there is no
GPU, no checkpoint and no dump in this environment, so the harness cannot be run
end to end and "the tests still pass" is not the same claim. And the helpers must
work on CPU, because every test that drives a training step in this suite runs
there; ``gradient_norm`` used to name CUDA outright, which made those tests
impossible to write at all.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import overfit_temporal_tracking as overfit_cli
from arc.models.arc.arc import Arc
from arc.training import runtime


# The commit the extraction was taken from. Pinned rather than derived so a
# rebase fails loudly here instead of silently comparing against the wrong tree.
EXTRACTION_BASELINE = "95b521a"

# What the extraction was allowed to do to the harness, declared up front so the
# test below checks the plan rather than describing whatever happened.
MOVED_TO_RUNTIME = {
    "_assert_frozen_gradients_absent",
    "_assert_trainable_gradients_finite",
    "_autocast_context",
    "_confidence_gradient_norms",
    "_confidence_stats",
    "_expected_trainable_set",
    "_gradient_norm",
    "_move_views_to_cuda",
    "_shuffled_index_views",
    "_tracking_only",
}
REWRITTEN_AS_WRAPPERS = {"_build_optimizer"}


def _top_level_sources(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    return {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _baseline_harness_source() -> str:
    result = subprocess.run(
        ["git", "show", f"{EXTRACTION_BASELINE}:overfit_temporal_tracking.py"],
        capture_output=True,
        text=True,
        cwd=Path(overfit_cli.__file__).parent,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot read {EXTRACTION_BASELINE} from git: {result.stderr.strip()}")
    return result.stdout


# ------------------------------------------------------------------ extraction ---


def test_the_extraction_left_every_surviving_harness_function_byte_identical():
    """The invariant the whole extraction rests on, checked instead of argued.

    All 286 tests would still pass if the move had quietly reordered a step-loop
    guard, dropped a ``del``, or changed a printed format -- none of the guards
    in ``main`` has a test of its own. Comparing source segments catches exactly
    that class of change, and it runs without a GPU, which the real before/after
    ``run_summary.json`` comparison cannot.
    """

    before = _top_level_sources(_baseline_harness_source())
    after = _top_level_sources(Path(overfit_cli.__file__).read_text())

    assert set(before) - set(after) == MOVED_TO_RUNTIME
    assert set(after) - set(before) == set()

    changed = {
        name
        for name in set(before) & set(after)
        if before[name] != after[name]
    }
    assert changed == REWRITTEN_AS_WRAPPERS
    # main carries the step loop, every guard and the run_summary literal, so it
    # is the one that matters most; assert it by name rather than by set algebra.
    assert before["main"] == after["main"]


def test_the_harness_still_exposes_the_moved_helpers_as_module_globals():
    """Six tests monkeypatch these onto the harness module by name.

    The alias-import form is what keeps that working after the move. A plain
    ``from arc.training import runtime`` plus ``runtime.tracking_only(...)`` call
    sites would leave those patches setting an attribute nothing reads, and the
    tests would pass while testing nothing.
    """

    for name in MOVED_TO_RUNTIME:
        assert hasattr(overfit_cli, name), name

    assert overfit_cli._tracking_only is runtime.tracking_only
    assert overfit_cli._confidence_stats is runtime.confidence_stats
    assert overfit_cli._autocast_context is runtime.autocast_context
    assert overfit_cli.EXPECTED_TRAINABLE_SETS is runtime.EXPECTED_TRAINABLE_SETS


# --------------------------------------------------------------- gradient norm ---


def _parameter_with_grad(value, device="cpu"):
    parameter = nn.Parameter(torch.zeros(len(value), device=device))
    parameter.grad = torch.tensor(value, dtype=torch.float32, device=device)
    return parameter


def test_gradient_norm_runs_on_cpu_and_follows_the_gradient_device():
    """The regression that made every CPU step test impossible to write.

    The pre-extraction body opened with ``torch.zeros((), device="cuda", ...)``,
    which raises ``AssertionError: Torch not compiled with CUDA enabled`` on a
    CPU-only build -- before looking at a single gradient. Any test that drives a
    training step reaches this, so the guard it protects was untestable here.
    """

    parameters = [_parameter_with_grad([3.0, 4.0]), _parameter_with_grad([12.0])]

    assert runtime.gradient_norm(parameters) == pytest.approx(13.0)


def test_gradient_norm_matches_a_zero_seeded_accumulation_exactly():
    """Dropping the zero seed must not move an archived number.

    ``0 + x`` is exact in float32 and the summation order is unchanged, so the
    device-following accumulator is bit-identical to the original, not merely
    close. Asserted with ``==`` rather than a tolerance, because a tolerance
    would hide precisely the drift this is here to exclude.
    """

    generator = torch.Generator().manual_seed(7)
    parameters = [
        _parameter_with_grad(torch.randn(17, generator=generator).tolist())
        for _ in range(5)
    ]

    seeded = torch.zeros((), dtype=torch.float32)
    for parameter in parameters:
        seeded += parameter.grad.detach().float().square().sum()

    assert runtime.gradient_norm(parameters) == float(torch.sqrt(seeded).item())


def test_gradient_norm_is_zero_when_nothing_carries_a_gradient():
    """Distinguishes "no gradients yet" from "gradients that sum to zero"."""

    parameter = nn.Parameter(torch.zeros(3))

    assert runtime.gradient_norm([parameter]) == 0.0


# ------------------------------------------------------------- build_optimizer ---


def _meta_arc(freeze="none", max_time_indices=32, late_global_blocks=None):
    """A full Arc with no allocated storage, as test_time_indexing builds one."""

    original_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            model = Arc(max_time_indices=max_time_indices)
    finally:
        torch.linspace = original_linspace
    if freeze != "none":
        model.set_freeze(freeze, late_global_blocks=late_global_blocks)
    return model


@pytest.mark.parametrize(
    "freeze_mode, late_global_blocks",
    [
        ("temporal_tracking", None),
        ("temporal_tracking_global_attention", None),
        ("temporal_tracking_late_global", runtime.DEFAULT_LATE_GLOBAL_BLOCKS),
    ],
)
def test_build_optimizer_covers_every_trainable_parameter_in_every_preset(
    freeze_mode,
    late_global_blocks,
):
    """The claim that let the extraction leave this function alone.

    ``build_optimizer`` selects groups by ``requires_grad`` and the embedding's
    name, never by the freeze mode, so a new preset needs no change here. That is
    an argument until a preset it was not written against is driven through it --
    ``temporal_tracking_late_global`` postdates the function, so it is the one
    that makes this a check rather than a restatement.
    """

    model = _meta_arc(freeze_mode, late_global_blocks=late_global_blocks)

    optimizer, learning_rates, encoder_parameters = runtime.build_optimizer(
        model,
        lr=1e-5,
        embedding_lr=None,
        encoder_lr=None,
    )

    grouped = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert grouped == trainable
    expected_tensors, _ = runtime.expected_trainable_set(freeze_mode, late_global_blocks)
    assert sum(1 for _ in model.parameters() if _.requires_grad) == expected_tensors
    # The narrow preset unfreezes no encoder block, so its rate is reported as
    # None rather than as a rate that governs nothing.
    if freeze_mode == "temporal_tracking":
        assert encoder_parameters == []
        assert learning_rates["encoder_blocks"] is None
    else:
        assert encoder_parameters
        assert learning_rates["encoder_blocks"] == pytest.approx(1e-6)


def test_build_optimizer_takes_scalars_so_a_second_parser_can_call_it():
    """The reason it stopped taking a Namespace.

    A second driver has its own flag names. Binding the builder to this driver's
    would make the next one either rename its flags to match or copy the builder,
    and the copy is what the extraction exists to prevent.
    """

    from inspect import signature

    parameters = signature(runtime.build_optimizer).parameters
    assert "args" not in parameters
    assert {"lr", "embedding_lr", "encoder_lr"} <= set(parameters)
    assert parameters["lr"].kind is parameters["lr"].KEYWORD_ONLY
