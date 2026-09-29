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

from types import SimpleNamespace

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
    "_anchor_velocity_counts",
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
    # Same reason, one term later: the velocity term's shares are a third
    # reduction over the same targets, and a copy in either driver would let
    # them weight one objective two ways.
    assert overfit_cli._anchor_velocity_counts is runtime.anchor_velocity_counts


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


def test_injection_trainable_sets_match_the_meta_arc_modules():
    """The per-injection constants are arithmetic from embed_dim and patch
    size; this proves the arithmetic against the real modules, the way the
    late-global per-block entry is proven against a meta-device Arc."""

    pretrained = _meta_arc().backbone.pretrained
    assert runtime.DEPTH_EMBED_TRAINABLE == (
        2,
        sum(parameter.numel() for parameter in pretrained.depth_patch_embed.parameters()),
    )
    assert runtime.CAMERA_PROJ_TRAINABLE == (
        2,
        sum(parameter.numel() for parameter in pretrained.camera_proj.parameters()),
    )


@pytest.mark.parametrize(
    "depth_input, camera_input",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_assert_trainable_parameter_set_prices_each_injection_arm(
    depth_input, camera_input
):
    """Matching freeze and assert flags pass; a mismatch in either direction
    raises. This is the startup guard that catches a driver wiring only one
    of its two call sites, and the pin for fork (b)'s one silent case: a
    module that is fed but frozen trains nothing and no gradient guard can
    see it, so the count mismatch here is what stops the run."""

    model = _meta_arc()
    model.set_freeze(
        "temporal_tracking", depth_input=depth_input, camera_input=camera_input
    )
    report = runtime.assert_trainable_parameter_set(
        model,
        freeze_mode="temporal_tracking",
        max_time_indices=32,
        depth_input=depth_input,
        camera_input=camera_input,
    )
    assert report["parameter_count"] == (
        314_600_740
        + runtime.DEPTH_EMBED_TRAINABLE[1] * int(depth_input)
        + runtime.CAMERA_PROJ_TRAINABLE[1] * int(camera_input)
    )

    with pytest.raises(RuntimeError, match="parameter set"):
        runtime.assert_trainable_parameter_set(
            model,
            freeze_mode="temporal_tracking",
            max_time_indices=32,
            depth_input=not depth_input,
            camera_input=camera_input,
        )
    with pytest.raises(RuntimeError, match="parameter set"):
        runtime.assert_trainable_parameter_set(
            model,
            freeze_mode="temporal_tracking",
            max_time_indices=32,
            depth_input=depth_input,
            camera_input=not camera_input,
        )


@pytest.mark.parametrize(
    "depth_input, camera_input",
    [(True, False), (False, True), (True, True)],
)
def test_build_optimizer_places_injections_in_the_decoder_group(
    depth_input, camera_input
):
    """Fresh zero-init capacity belongs at the full decoder rate, and under
    the narrow preset the encoder group must stay empty with its reported
    rate None -- the lazy encoder-group placement would silently change both
    (landmine 3)."""

    model = _meta_arc()
    model.set_freeze(
        "temporal_tracking", depth_input=depth_input, camera_input=camera_input
    )

    optimizer, learning_rates, encoder_parameters = runtime.build_optimizer(
        model, lr=1e-3
    )

    assert encoder_parameters == []
    assert learning_rates["encoder_blocks"] is None
    decoder_group = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    pretrained = model.backbone.pretrained
    for enabled, module in (
        (depth_input, pretrained.depth_patch_embed),
        (camera_input, pretrained.camera_proj),
    ):
        if not enabled:
            continue
        for parameter in module.parameters():
            assert id(parameter) in decoder_group


def test_encode_and_reconstruct_threads_geometry_to_encode_features():
    """Pins the runtime forwarding hop independently of zero init: recording
    is weight-free, so a dropped kwarg fails here even though every output
    would be bit-identical."""

    depth_sentinel = torch.zeros(1, 2, 2, 4, 4)
    camera_sentinel = torch.zeros(1, 2, 11)
    recorded = {}

    class _Recorder:
        def _preprocess_input(self, views):
            return (
                torch.zeros(1, 2, 3, 4, 4),
                [0],
                None,
                depth_sentinel,
                camera_sentinel,
            )

        def encode_features(self, images, **kwargs):
            recorded.update(kwargs)
            return [images]

        def reconstruct(self, feats, images):
            return {}

    runtime.encode_and_reconstruct(_Recorder(), [])

    assert recorded["depth_maps"] is depth_sentinel
    assert recorded["camera_vectors"] is camera_sentinel


# ------------------------------------------------------------- the refiner ---


def test_refiner_trainable_set_matches_the_meta_arc_module():
    """REFINER_TRAINABLE is arithmetic from the refiner's widths, which
    runtime imports from motiondecoder rather than restating; this proves the
    arithmetic against the real module's parameter count and the module's
    shapes against those widths -- the way the injection constants are proven
    against the meta-device Arc. A drift prices the arm wrong, and the startup
    guard then refuses every --refine_iters run -- or, worse, accepts one whose
    refiner is frozen."""

    refiner = _meta_arc().motion_decoder.refiner
    assert runtime.REFINER_TRAINABLE == (
        8,
        sum(parameter.numel() for parameter in refiner.parameters()),
    )
    assert runtime.REFINER_TRAINABLE[1] == 1_398_016
    assert refiner.field_embed.in_channels == runtime.REFINER_FIELD_CHANNELS
    assert refiner.field_embed.kernel_size == (runtime.ENCODER_PATCH_SIZE,) * 2
    assert refiner.query_proj.out_features == runtime.REFINER_CORRELATION_DIM
    assert refiner.key_proj.out_features == runtime.REFINER_CORRELATION_DIM
    assert refiner.read_proj.in_features == runtime.REFINER_NEIGHBOURS * 4
    assert refiner.read_proj.out_features == runtime.ENCODER_EMBED_DIM


@pytest.mark.parametrize("refine", [False, True])
def test_assert_trainable_parameter_set_prices_the_refine_arm(refine):
    """The refiner's startup guard, in both directions: a driver that unfroze
    the refiner but priced it as off fails here, and so does one that priced
    a K=4 arm as on while set_freeze left it frozen -- the fed-but-frozen
    case no gradient guard can see: a refiner fed K iterations but frozen is
    zero-init and therefore bit-identical to K=1. At refine=False the rows
    are today's exactly, which is the flag-off half of the pin."""

    model = _meta_arc()
    model.set_freeze("temporal_tracking", refine=refine)
    report = runtime.assert_trainable_parameter_set(
        model, freeze_mode="temporal_tracking", max_time_indices=32, refine=refine
    )
    assert report["tensor_count"] == 231 + runtime.REFINER_TRAINABLE[0] * int(refine)
    assert report["parameter_count"] == (
        314_600_740 + runtime.REFINER_TRAINABLE[1] * int(refine)
    )

    with pytest.raises(RuntimeError, match="parameter set"):
        runtime.assert_trainable_parameter_set(
            model, freeze_mode="temporal_tracking", max_time_indices=32, refine=not refine
        )


@pytest.mark.parametrize("refine", [False, True])
def test_build_optimizer_places_the_refiner_by_its_flag(refine):
    """No change to build_optimizer was needed, which is a claim until driven:
    the refiner sits inside motion_decoder, so it lands in the decoder group
    at the full rate when trainable (fresh capacity, like the injections) and
    is excluded by requires_grad when not. The narrow preset's encoder group
    must stay empty either way, and this is what keeps that true if the
    grouping is ever rewritten."""

    model = _meta_arc()
    model.set_freeze("temporal_tracking", refine=refine)

    optimizer, learning_rates, encoder_parameters = runtime.build_optimizer(
        model, lr=1e-3
    )

    assert encoder_parameters == []
    assert learning_rates["encoder_blocks"] is None
    grouped = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    decoder_group = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    refiner_parameters = list(model.motion_decoder.refiner.parameters())
    assert len(refiner_parameters) == 8
    for parameter in refiner_parameters:
        assert parameter.requires_grad is refine
        assert (id(parameter) in decoder_group) is refine
        assert (id(parameter) in grouped) is refine


def test_anchor_tracks_passes_refinement_only_when_set():
    """Recording is weight-free, so a dropped or always-passed keyword fails
    here even though every field would look right. The three-positional fake
    is the surface test_sparse_tracking's fakes bind; it must still be reached
    with refinement None. The keyword fake must receive the very object, with
    views_per_time and merge beside it."""

    scene = SimpleNamespace(anchor_observation_slots=[3, 5])
    feats, images = object(), object()
    track = torch.zeros(1, 4, 2, 3, 3)
    confidence = torch.zeros(1, 4, 2, 3)

    class _Positional:
        def track_for_query(self, feats, images, query_idx):
            return track, confidence

    raw = runtime.anchor_tracks(_Positional(), feats, images, scene, 1)
    assert raw["track_multi"].shape == (1, 1, 4, 2, 3, 3)
    assert raw["track_query_idx"].tolist() == [5]

    seen = {}

    class _Keyword:
        def track_for_query(
            self, feats, images, query_idx, *, views_per_time=1, merge=False, refinement=None
        ):
            seen.update(
                query_idx=query_idx,
                views_per_time=views_per_time,
                merge=merge,
                refinement=refinement,
            )
            return track, confidence

    sentinel = object()
    runtime.anchor_tracks(
        _Keyword(), feats, images, scene, 0,
        views_per_time=2, merge=True, refinement=sentinel,
    )
    assert seen == {"query_idx": 3, "views_per_time": 2, "merge": True, "refinement": sentinel}


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
