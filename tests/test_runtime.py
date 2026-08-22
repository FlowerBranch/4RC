"""Pins for the helpers shared by every bounded training entry point.

What is being protected: the helpers must work on CPU, because every test that
drives a training step in this suite runs there; ``gradient_norm`` used to name
CUDA outright, which made those tests impossible to write at all. And they must
stay reachable as module globals on the harness, because six tests monkeypatch
them there by name.

A source-equivalence pin on ``overfit_temporal_tracking.py`` used to live here
too, comparing every top-level function against a pinned commit. Removed: it
guarded the one-scene harness, which `4I4/docs/execution.md` records as exhausted
as an instrument, while ``train_temporal_tracking.py`` -- the program that
produces every current result -- was never pinned at all. It fired three times,
each on a deliberate change, each resolved by moving the baseline, and caught
nothing; meanwhile each extraction into ``arc/training/runtime.py`` shrank the
surface it still covered. If a mechanical check comes back, it belongs on the
trainer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import overfit_temporal_tracking as overfit_cli
from arc.models.arc.arc import Arc
from arc.training import runtime


# The helpers the extractions moved out of ``overfit_temporal_tracking.py``. The
# harness must still expose every one of them as a module global; see below.
# The second group is the per-anchor supervision mechanism, moved when the
# multi-scene trainer became its second consumer.
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
    "_accumulate",
    "_anchor_confidence_counts",
    "_anchor_sample_counts",
    "_anchor_tracks",
    "_backward_through_cut",
    "_cut_features",
    "_encode_and_reconstruct",
    "_weighted_anchor_total",
}


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
    assert overfit_cli._cut_features is runtime.cut_features
    assert overfit_cli._weighted_anchor_total is runtime.weighted_anchor_total
    # Moved when the trainer became its second consumer: the confidence term's
    # per-anchor shares are the trainer's too, and a copy would let the two
    # drivers weight the same objective differently.
    assert overfit_cli._anchor_confidence_counts is runtime.anchor_confidence_counts


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
