"""Pins for training against a chosen query anchor: --query_anchor_source.

A track head trained against its own predicted query anchor compensates that
anchor, so substituting a ground-truth one at eval time measures how much the
head depends on it rather than what correct geometry buys. Training against
the reading is what measures that. ``train_step`` therefore gathers its anchor
from ``query_anchor_source`` and composes it in the frame the gather returns,
exactly as the held-out eval already did for its two ground-truth readings;
the eval scores on the training source unless ``--eval_query_anchor_source``
(or one of its two aliases) says otherwise.

Every source test runs on the SCALED two-anchor window. On the identity gauge
the fitted Sim(3) is the identity to ~1e-10, so a world-frame anchor composed
as if it were in the model's gauge is bit-identical to the right composition,
and at the reference camera the ground-truth depth anchor equals the predicted
one. The scaled gauge (2.25, a 25-degree rotation and a translation) parts
both; the camera-1 anchor reads the reference camera's planted pointmap at its
own pixel, so its prediction is wrong by about a metre and the two sources
part there too. The tests compare the recorded tensors, not only the losses.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import arc.training as training_package
import arc.training.sparse_tracking as sparse_module
import train_temporal_tracking as train_cli
from arc.models.arc.heads.motiondecoder import patch_centre_points
from arc.models.arc.utils.transform import predicted_pointmaps
from arc.training import (
    QUERY_ANCHOR_SOURCES,
    build_anchor_correspondences,
    fit_scene_sim3,
    gather_query_anchor_points,
    refinement_iteration_weights,
)
from arc.training.manifest_plan import plan_record
from scene_fixtures import fixture_scene
from test_manifest_plan import _record
from test_sparse_tracking import _steered_to
from test_trainer_loop import (
    _EVAL_ALPHA,
    _FOUR_CAMERA_WINDOW,
    _FakeArc,
    _cpu_eval_scene,
    _loop_args,
    _plans,
    _validator_args,
)
from test_track_refinement import (
    _RecordingScaler,
    _RefiningFakeArc,
    _gradients,
    _hand_passes,
    _refine_step,
)

_GROUND_TRUTH_SOURCES = ("query_position", "gt_depth")
_HELD_OUT = ("--val_scenes_file", "val.json", "--val_data_root", "/held/out")


def _two_anchor_scene(monkeypatch, **scene_kwargs):
    """The scaled window with anchors at cameras 0 and 1, time 0.

    Slots 0 and 4; one occluded track gives the correspondences query_slots
    [0, 0, 1], so the anchors' row counts differ and a slice taken with the
    wrong anchor's mask changes shape.
    """

    return _cpu_eval_scene(
        monkeypatch,
        gauge="scaled",
        query_anchors=((0, 0), (1, 0)),
        invisible=((0, 0, 2),),
        **scene_kwargs,
    )


def _truth(scene, correspondences, source):
    """Each row's ground-truth anchor, computed here rather than by the gather."""

    if source == "query_position":
        return scene.trajectories_world[
            correspondences.query_times, correspondences.trajectory_indices
        ].to(torch.float32)
    slots = scene.track_query_observation_slots[correspondences.query_slots]
    rows = []
    for item in range(correspondences.count):
        world_points, _ = sparse_module._metric_pointmap_at_anchor(
            scene, int(slots[item])
        )
        rows.append(
            torch.from_numpy(
                world_points[
                    int(correspondences.rows[item]), int(correspondences.columns[item])
                ]
            ).to(torch.float32)
        )
    return torch.stack(rows)


def _record_losses(monkeypatch):
    """Every sparse_tracking_loss call, with the anchors and frame it was handed.

    Both train_step and evaluate_held_out import the loss from the package at
    call time, so the package namespace is the one interception point.
    """

    calls = []
    real = training_package.sparse_tracking_loss

    def recording(raw, scene, correspondences, alignment, anchors, **kwargs):
        anchors_tensor = torch.as_tensor(anchors)
        calls.append(
            SimpleNamespace(
                correspondences=correspondences,
                anchors=anchors_tensor.detach().clone(),
                requires_grad=anchors_tensor.requires_grad,
                frame=kwargs["query_anchor_frame"],
            )
        )
        return real(raw, scene, correspondences, alignment, anchors, **kwargs)

    monkeypatch.setattr(training_package, "sparse_tracking_loss", recording)
    return calls


def _record_gathers(monkeypatch):
    """Every gather_query_anchor_points call: its correspondences and keywords."""

    calls = []
    real = training_package.gather_query_anchor_points

    def recording(raw, scene, correspondences, **kwargs):
        calls.append((correspondences, dict(kwargs)))
        return real(raw, scene, correspondences, **kwargs)

    monkeypatch.setattr(training_package, "gather_query_anchor_points", recording)
    return calls


def _assert_anchor_major(calls, correspondences, anchor_indices, refine_iters):
    """The step scores each active anchor's passes in turn, on its own rows."""

    assert len(calls) == len(anchor_indices) * refine_iters
    for position, call in enumerate(calls):
        index = anchor_indices[position // refine_iters]
        rebased = correspondences.select_query_slot(index)
        assert torch.equal(
            call.correspondences.trajectory_indices, rebased.trajectory_indices
        ), position
        assert torch.equal(call.correspondences.rows, rebased.rows), position


# ------------------------------------------------- the default is today's ---


@pytest.mark.parametrize("refine_iters", (1, 2))
@pytest.mark.parametrize("merge", (False, True), ids=("per_slot", "merged"))
def test_the_predicted_source_is_todays_step_bit_for_bit(monkeypatch, merge, refine_iters):
    """``source="predicted"`` spelled out and the default are one step: loss,
    per-pass losses, metric error and every gradient equal exactly, and every
    pass composes the model's own anchor in its own gauge. Against HEAD itself
    this was checked by a snapshot diff; a committed test cannot name HEAD."""

    scene = _two_anchor_scene(monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    default = _RefiningFakeArc(
        scene.num_observations, height, width, views_per_time=2 if merge else 1
    )
    explicit = copy.deepcopy(default)
    calls = _record_losses(monkeypatch)

    first = _refine_step(
        default, scene, refine_iters=refine_iters, refine_gamma=0.5,
        merge_synchronized_slots=merge,
    )
    default_calls = list(calls)
    second = _refine_step(
        explicit, scene, refine_iters=refine_iters, refine_gamma=0.5,
        merge_synchronized_slots=merge, query_anchor_source="predicted",
    )

    assert first.loss == second.loss
    assert first.metric_error_m == second.metric_error_m
    assert first.iteration_losses == second.iteration_losses
    gradients, explicit_gradients = _gradients(default), _gradients(explicit)
    assert set(gradients) == set(explicit_gradients)
    for name in gradients:
        assert torch.equal(gradients[name], explicit_gradients[name]), name

    correspondences, _ = build_anchor_correspondences(scene)
    predicted, frame = gather_query_anchor_points({}, scene, correspondences)
    assert frame == "model"
    _assert_anchor_major(default_calls, correspondences, (0, 1), refine_iters)
    for position, call in enumerate(default_calls):
        index = position // refine_iters
        assert call.frame == "model"
        assert torch.equal(call.anchors, predicted[correspondences.anchor_rows(index)])


# --------------------------------------------- every pass, at the live K ---


@pytest.mark.parametrize("merge", (False, True), ids=("per_slot", "merged"))
@pytest.mark.parametrize("source", _GROUND_TRUTH_SOURCES)
def test_every_pass_of_the_live_shape_composes_on_the_ground_truth_anchor(
    monkeypatch, source, merge
):
    """--refine_iters 4 --refine_gamma 0.5 (powers of two, so every scale is
    bit-exact) at two anchors, with distinct per-pass fields.

    One gather, from the scene-level correspondences, spelling the source.
    Eight loss calls, anchor-major, every one handed a detached WORLD-frame
    anchor equal to the ground truth computed here independently, sliced to
    that anchor's own rows -- so a source dropped on any pass, a pass composed
    in the model's gauge, or the anchors' slices swapped fails. The eight
    backwarded scalars and the per-pass losses equal K=1 steps at each pass's
    field under the same source, exactly. The refiner sees exactly what it
    sees at the predicted source -- the model's own cloud, the anchor slot's
    patch centres and its own carried field -- so the source moves only what
    the loss composes on; and the per-pass losses differ from that predicted
    run's, because the camera-1 anchor's prediction is wrong.
    """

    scene = _two_anchor_scene(monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    refine_iters = 4
    torch.manual_seed(0)
    model = _RefiningFakeArc(
        scene.num_observations, height, width, views_per_time=2 if merge else 1
    )
    pristine = copy.deepcopy(model)
    control = copy.deepcopy(model)
    gathers = _record_gathers(monkeypatch)
    calls = _record_losses(monkeypatch)

    scaler = _RecordingScaler()
    outcome = _refine_step(
        model, scene, refine_iters=refine_iters, refine_gamma=0.5, scaler=scaler,
        merge_synchronized_slots=merge, query_anchor_source=source,
    )
    step_gathers, step_calls = list(gathers), list(calls)

    correspondences, _ = build_anchor_correspondences(scene)
    assert correspondences.query_slots.tolist() == [0, 0, 1]
    assert len(step_gathers) == 1
    gathered_on, keywords = step_gathers[0]
    assert keywords == {"source": source}
    assert torch.equal(gathered_on.query_slots, correspondences.query_slots)

    truth = _truth(scene, correspondences, source)
    alignment, _ = fit_scene_sim3({}, scene)
    predicted, _ = gather_query_anchor_points({}, scene, correspondences)
    assert not torch.allclose(alignment.apply_points(predicted), truth, atol=1e-2)
    _assert_anchor_major(step_calls, correspondences, (0, 1), refine_iters)
    for position, call in enumerate(step_calls):
        index = position // refine_iters
        assert call.frame == "world", position
        assert call.requires_grad is False
        assert torch.equal(call.anchors, truth[correspondences.anchor_rows(index)]), position

    per_anchor = [
        model.applied_offsets[index * refine_iters : (index + 1) * refine_iters]
        for index in range(2)
    ]
    assert per_anchor[0] == per_anchor[1]
    offsets = per_anchor[0]
    assert offsets[0] == 0.0 and len(set(offsets)) == refine_iters
    references = _hand_passes(
        pristine, scene, offsets,
        merge_synchronized_slots=merge, query_anchor_source=source,
    )
    weights = refinement_iteration_weights(refine_iters, 0.5)
    assert scaler.scaled == [
        weights[iteration] * references[iteration][2][index]
        for index in range(2)
        for iteration in range(refine_iters)
    ]
    assert outcome.iteration_losses == [
        reference_outcome.loss for _, reference_outcome, _, _ in references
    ]

    control_outcome = _refine_step(
        control, scene, refine_iters=refine_iters, refine_gamma=0.5,
        merge_synchronized_slots=merge,
    )
    assert outcome.iteration_losses != control_outcome.iteration_losses
    expected_cloud = patch_centre_points(
        predicted_pointmaps(pristine.reconstruct(None, None)), pristine.PATCH_SIZE
    )
    assert len(model.seen_previous_fields) == len(control.seen_previous_fields) == 8
    for position, (seen, seen_control) in enumerate(
        zip(model.seen_previous_fields, control.seen_previous_fields)
    ):
        (carry, requires_grad, anchor, cloud) = seen
        (control_carry, control_requires_grad, control_anchor, control_cloud) = seen_control
        assert torch.equal(carry, control_carry), position
        assert requires_grad is control_requires_grad is False
        assert torch.equal(cloud, control_cloud) and torch.equal(cloud, expected_cloud)
        index = position // refine_iters
        assert torch.equal(anchor, control_anchor)
        assert torch.equal(anchor, expected_cloud[:, scene.anchor_observation_slots[index]])


@pytest.mark.parametrize("refine_iters", (1, 2))
@pytest.mark.parametrize("merge", (False, True), ids=("per_slot", "merged"))
@pytest.mark.parametrize("source", _GROUND_TRUTH_SOURCES)
def test_the_step_trains_against_the_anchors_the_eval_scores_on(
    tmp_path, monkeypatch, source, merge, refine_iters
):
    """For one scene, the anchors every pass of the step composes on are the
    eval's own, sliced to each anchor's rows: the eval at the same source
    gathers once from the scene-level correspondences and hands its main arm
    exactly those points, so a model is scored on the anchor it trained
    against. Every pass of both is in the world frame."""

    scene = _two_anchor_scene(monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    views_per_time = 2 if merge else 1
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width, views_per_time=views_per_time)
    evaluated = copy.deepcopy(model)
    calls = _record_losses(monkeypatch)

    train_cli.evaluate_held_out(
        model=evaluated,
        plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
        scene_provider=lambda _plan: scene,
        precision="32",
        huber_delta_m=0.05,
        step=1,
        output_dir=tmp_path,
        query_anchors=["0:0", "1:0"],
        confidence_alpha=_EVAL_ALPHA,
        query_anchor_source=source,
        eval_query_anchor_source=source,
        merge_synchronized_slots=merge,
        refine_iters=refine_iters,
    )
    assert all(call.frame == "world" for call in calls)
    scored = calls[0].anchors
    calls.clear()

    _refine_step(
        model, scene, refine_iters=refine_iters, refine_gamma=0.5,
        merge_synchronized_slots=merge, query_anchor_source=source,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    assert torch.equal(scored, _truth(scene, correspondences, source))
    _assert_anchor_major(calls, correspondences, (0, 1), refine_iters)
    for position, call in enumerate(calls):
        index = position // refine_iters
        assert call.frame == "world", position
        assert torch.equal(call.anchors, scored[correspondences.anchor_rows(index)]), position


# -------------------------------------------- the loss moves where it should ---


def _scaled_preimage(points: np.ndarray) -> np.ndarray:
    """The model-gauge points the scaled gauge maps onto ``points``: the same
    similarity _cpu_eval_scene plants."""

    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ((points - np.array([0.3, -0.2, 1.1])) @ rotation) / 2.25


def _exactly_predicted_scene(monkeypatch, source):
    """The two-anchor window with every slot's prediction exact for ``source``.

    Each slot plants the preimage of its OWN ground-truth unprojection, so the
    predicted anchor IS ground-truth depth at the anchor pixel once aligned;
    for ``"query_position"`` each correspondence pixel is overwritten with the
    preimage of its tracked point's true position, so the aligned prediction
    is that point instead.
    """

    scene = fixture_scene(
        time_count=4,
        view_count=2,
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
        query_anchors=((0, 0), (1, 0)),
        invisible=((0, 0, 2),),
    )
    planted = np.stack(
        [
            _scaled_preimage(sparse_module._metric_pointmap_at_anchor(scene, slot)[0])
            for slot in range(scene.num_observations)
        ]
    )
    if source == "query_position":
        correspondences, _ = build_anchor_correspondences(scene)
        slots = scene.track_query_observation_slots[correspondences.query_slots]
        truth = _truth(scene, correspondences, "query_position").double().numpy()
        for item in range(correspondences.count):
            planted[
                int(slots[item]),
                int(correspondences.rows[item]),
                int(correspondences.columns[item]),
            ] = _scaled_preimage(truth[item])
    pointmaps = torch.from_numpy(planted).float()[None].contiguous()
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)
    return scene


def _one_step(scene, source):
    height, width = scene.views[0]["img"].shape[-2:]
    torch.manual_seed(0)
    model = _FakeArc(scene.num_observations, height, width)
    outcome = _refine_step(
        model, scene, refine_iters=1, refine_gamma=0.8, query_anchor_source=source
    )
    return outcome, _gradients(model)


@pytest.mark.parametrize("source", _GROUND_TRUTH_SOURCES)
def test_the_source_moves_the_loss_only_where_the_prediction_is_wrong(monkeypatch, source):
    """Exact predictions train the same model whichever anchor is named; a
    wrong one does not. With every slot's prediction exact for the source, the
    loss, the metric error and every gradient agree with the predicted step to
    fp32 rounding -- the model-gauge and world-frame compositions are equal
    algebraically, not bit for bit. On the two-anchor window, where the
    camera-1 anchor reads the reference camera's pointmap, they part."""

    exact = _exactly_predicted_scene(monkeypatch, source)
    named, named_gradients = _one_step(exact, source)
    own, own_gradients = _one_step(exact, "predicted")
    assert named.loss == pytest.approx(own.loss, rel=1e-4, abs=1e-7)
    assert named.metric_error_m == pytest.approx(own.metric_error_m, rel=1e-4, abs=1e-7)
    for name in own_gradients:
        torch.testing.assert_close(
            named_gradients[name], own_gradients[name], rtol=1e-4, atol=1e-7
        )

    wrong = _two_anchor_scene(monkeypatch)
    named, named_gradients = _one_step(wrong, source)
    own, own_gradients = _one_step(wrong, "predicted")
    assert abs(named.loss - own.loss) > 1e-2 * own.loss
    assert abs(named.metric_error_m - own.metric_error_m) > 1e-2 * own.metric_error_m
    assert any(
        not torch.allclose(named_gradients[name], own_gradients[name], rtol=1e-3)
        for name in own_gradients
    )


@pytest.mark.parametrize("inputs", (False, True), ids=("rgb", "depth_and_camera_inputs"))
@pytest.mark.parametrize("source", _GROUND_TRUTH_SOURCES)
def test_silent_anchors_and_geometry_inputs_leave_the_anchor_slices_right(
    monkeypatch, source, inputs
):
    """The live four-anchor spec, two of whose anchors seat and win no row:
    the steered window puts rows on anchors 1 and 2 only, so the step runs
    those two, and each is handed ITS OWN rows' ground truth -- an anchor's
    index, not its place among the active ones, picks the slice. With the
    depth and camera inputs attached to the views as well, the source path is
    unchanged: it reads neither."""

    extra = {"input_depth_max": 24.0, "input_camera_vectors": True} if inputs else {}
    scene = _cpu_eval_scene(
        monkeypatch,
        gauge="scaled",
        invisible=_steered_to({0: 3, 1: 3, 2: 1}),
        **_FOUR_CAMERA_WINDOW,
        **extra,
    )
    if inputs:
        assert {"depth_map", "camera_vector"} <= set(scene.views[0])
    height, width = scene.views[0]["img"].shape[-2:]
    refine_iters = 2
    torch.manual_seed(0)
    model = _RefiningFakeArc(scene.num_observations, height, width)
    calls = _record_losses(monkeypatch)

    outcome = _refine_step(
        model, scene, refine_iters=refine_iters, refine_gamma=0.5,
        query_anchor_source=source,
    )

    correspondences, _ = build_anchor_correspondences(scene)
    assert sorted(set(correspondences.query_slots.tolist())) == [1, 2]
    assert outcome.anchor_sample_counts[0] == outcome.anchor_sample_counts[3] == 0
    truth = _truth(scene, correspondences, source)
    _assert_anchor_major(calls, correspondences, (1, 2), refine_iters)
    for position, call in enumerate(calls):
        index = (1, 2)[position // refine_iters]
        assert call.frame == "world", position
        assert torch.equal(call.anchors, truth[correspondences.anchor_rows(index)]), position


# --------------------------------------------------- the run's two sources ---


def _no_op_step(handed):
    def step_fn(*, step, plan, **kwargs):
        handed.append(kwargs["query_anchor_source"])
        return train_cli.StepOutcome(
            step=step, seq_name=plan.seq_name, loss=0.0, metric_error_m=0.0,
            sample_count=1, alignment_scale=1.0, alignment_residual_m=0.0,
            learning_rates=[1e-3], gradient_norms={},
        )

    return step_fn


@pytest.mark.parametrize(
    "flags, scored",
    (
        ({"query_anchor_source": "gt_depth"}, "gt_depth"),
        ({"query_anchor_source": "gt_depth", "oracle_query_anchor": True}, "query_position"),
        (
            {"query_anchor_source": "query_position", "eval_query_anchor_source": "predicted"},
            "predicted",
        ),
    ),
    ids=("follows_training", "alias_overrides", "canonical_overrides"),
)
def test_run_training_threads_the_training_source_and_the_resolved_eval_source(
    tmp_path, monkeypatch, flags, scored
):
    """The loop hands every step the run's training source and the held-out
    eval the resolved one -- the training source unless an eval flag
    overrides it -- and the written metrics name both, with the two booleans
    derived from the eval's reading alone."""

    scene = _two_anchor_scene(monkeypatch)
    height, width = scene.views[0]["img"].shape[-2:]
    gathers = _record_gathers(monkeypatch)
    handed: list[str] = []
    model = _FakeArc(scene.num_observations, height, width)

    train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=_loop_args(
            tmp_path, num_steps=4, eval_every=4, query_anchors=["0:0", "1:0"], **flags
        ),
        scene_provider=lambda _plan: scene,
        step_fn=_no_op_step(handed),
        output_dir=tmp_path,
        held_out=train_cli.HeldOut(
            plans=[plan_record(_record(seq_name="0000"), budget=48, stride=2)],
            scene_provider=lambda _plan: scene,
            benchmark=None,
        ),
    )

    assert handed == [flags["query_anchor_source"]] * 4
    # The eval's scoring gather, plus -- under a ground-truth reading -- the
    # readouts' bare re-gather of the model's own anchor.
    assert [keywords for _, keywords in gathers] == [{"source": scored}] + (
        [] if scored == "predicted" else [{}]
    )
    written = json.loads((tmp_path / "eval" / "step-4" / "metrics.json").read_text())
    assert written["query_anchor_source"] == flags["query_anchor_source"]
    assert written["eval_query_anchor_source"] == scored
    assert written["oracle_query_anchor"] is (scored == "query_position")
    assert written["ground_truth_query_anchor"] is (scored == "gt_depth")


def test_the_trainer_wires_both_anchor_sources_into_the_step_and_the_eval():
    """Inspected rather than executed, as the refinement and visual-dump pins
    in test_trainer_loop are: main() is .to("cuda") unconditionally. Every
    step_fn call reads the training flag; every eval call reads the training
    flag for its record and the eval source run_training resolved once from
    the args; train_step's one gather spells its own parameter; and the eval's
    scoring gather spells its eval source while the readouts' gather spells
    nothing."""

    tree = ast.parse(Path(train_cli.__file__).read_text())

    def calls_named(root, name):
        return [
            node
            for node in ast.walk(root)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == name
        ]

    def keyword(call, name):
        return next((item.value for item in call.keywords if item.arg == name), None)

    def reads_args(node, attribute):
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and node.attr == attribute
        )

    def is_name(node, name):
        return isinstance(node, ast.Name) and node.id == name

    assert calls_named(tree, "step_fn"), "run_training has no step_fn call"
    for call in calls_named(tree, "step_fn"):
        assert reads_args(keyword(call, "query_anchor_source"), "query_anchor_source")
    assert calls_named(tree, "evaluate_held_out"), "run_training has no eval call"
    for call in calls_named(tree, "evaluate_held_out"):
        assert reads_args(keyword(call, "query_anchor_source"), "query_anchor_source")
        assert is_name(keyword(call, "eval_query_anchor_source"), "eval_anchor_source")
    bound = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if is_name(target, "eval_anchor_source")
    ]
    assert len(bound) == 1
    assert is_name(bound[0].func, "_eval_query_anchor_source")
    assert len(bound[0].args) == 1 and is_name(bound[0].args[0], "args")

    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    (step_gather,) = calls_named(functions["train_step"], "gather_query_anchor_points")
    assert [item.arg for item in step_gather.keywords] == ["source"]
    assert is_name(keyword(step_gather, "source"), "query_anchor_source")
    eval_gathers = calls_named(functions["evaluate_held_out"], "gather_query_anchor_points")
    assert [[item.arg for item in call.keywords] for call in eval_gathers] == [["source"], []]
    assert is_name(keyword(eval_gathers[0], "source"), "eval_query_anchor_source")
    # The extra held-out anchors gather the same way in their own helper, which
    # the eval hands its own source.
    extra_gathers = calls_named(functions["_eval_anchor_rows"], "gather_query_anchor_points")
    assert [[item.arg for item in call.keywords] for call in extra_gathers] == [["source"], []]
    assert is_name(keyword(extra_gathers[0], "source"), "eval_query_anchor_source")
    (extra_call,) = calls_named(functions["evaluate_held_out"], "_eval_anchor_rows")
    assert is_name(keyword(extra_call, "eval_query_anchor_source"), "eval_query_anchor_source")


# ---------------------------------------------------------- the eval flags ---


def _parsed(*argv):
    args = train_cli.build_arg_parser().parse_args(["--manifest", "m.jsonl", *argv])
    train_cli._validate_args(args)
    return args


@pytest.mark.parametrize("trained", QUERY_ANCHOR_SOURCES)
def test_the_eval_source_follows_the_training_source_unless_overridden(trained):
    """Through the real parser: no eval flag scores on the training source;
    the canonical flag and each alias select their reading; an alias beside a
    canonical flag naming the same reading is one request said twice. A
    second validation pass over the same namespace accepts what the first
    produced, and resolves the same."""

    args = _parsed("--query_anchor_source", trained, *_HELD_OUT)
    assert train_cli._eval_query_anchor_source(args) == trained
    for argv, expected in (
        (("--eval_query_anchor_source", "predicted"), "predicted"),
        (("--eval_query_anchor_source", "query_position"), "query_position"),
        (("--eval_query_anchor_source", "gt_depth"), "gt_depth"),
        (("--oracle_query_anchor",), "query_position"),
        (("--ground_truth_query_anchor",), "gt_depth"),
        (("--oracle_query_anchor", "--eval_query_anchor_source", "query_position"), "query_position"),
        (("--ground_truth_query_anchor", "--eval_query_anchor_source", "gt_depth"), "gt_depth"),
    ):
        args = _parsed("--query_anchor_source", trained, *_HELD_OUT, *argv)
        assert train_cli._eval_query_anchor_source(args) == expected, argv
        train_cli._validate_args(args)
        assert train_cli._eval_query_anchor_source(args) == expected, argv


@pytest.mark.parametrize(
    "argv, message",
    (
        (
            ("--oracle_query_anchor", "--eval_query_anchor_source", "predicted"),
            "--oracle_query_anchor (--eval_query_anchor_source query_position) cannot "
            "be combined with --eval_query_anchor_source predicted",
        ),
        (
            ("--oracle_query_anchor", "--eval_query_anchor_source", "gt_depth"),
            "--oracle_query_anchor (--eval_query_anchor_source query_position) cannot "
            "be combined with --eval_query_anchor_source gt_depth",
        ),
        (
            ("--ground_truth_query_anchor", "--eval_query_anchor_source", "predicted"),
            "--ground_truth_query_anchor (--eval_query_anchor_source gt_depth) cannot "
            "be combined with --eval_query_anchor_source predicted",
        ),
        (
            ("--ground_truth_query_anchor", "--eval_query_anchor_source", "query_position"),
            "--ground_truth_query_anchor (--eval_query_anchor_source gt_depth) cannot "
            "be combined with --eval_query_anchor_source query_position",
        ),
        (
            ("--oracle_query_anchor", "--ground_truth_query_anchor"),
            "--oracle_query_anchor cannot be combined with --ground_truth_query_anchor",
        ),
    ),
)
def test_two_spellings_of_different_eval_anchors_are_refused_at_parse_time(argv, message):
    """One eval writes one curve, so two flags naming different readings are
    refused before anything loads, by name."""

    args = train_cli.build_arg_parser().parse_args(
        ["--manifest", "m.jsonl", *_HELD_OUT, *argv]
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        train_cli._validate_args(args)


def test_the_eval_anchor_flag_needs_a_held_out_set_and_the_training_flag_does_not():
    """The eval flag reaches nothing but the held-out eval, so without a
    held-out set it would label a run it never measured; the training flag
    defines what every step trains on and needs no held-out set at all."""

    parser = train_cli.build_arg_parser()
    with pytest.raises(
        ValueError,
        match=(
            r"--eval_query_anchor_source needs a held-out set "
            r"\(--val_scenes_file or --eval_benchmark_dataset\)"
        ),
    ):
        train_cli._validate_args(
            parser.parse_args(["--manifest", "m.jsonl", "--eval_query_anchor_source", "gt_depth"])
        )
    for source in QUERY_ANCHOR_SOURCES:
        train_cli._validate_args(
            parser.parse_args(["--manifest", "m.jsonl", "--query_anchor_source", source])
        )


def test_the_plan_summary_records_both_sources_and_derives_the_booleans(tmp_path):
    """run_summary.json names the training source and the eval's reading, and
    keeps the two booleans its readers predate the names by, derived from the
    eval's reading alone: a ground-truth-trained run scored on another
    reading reports that reading. Without a held-out set the eval reading is
    None, like realized_val_query_anchors, and both booleans are False -- a
    --val_data_root alone included, which the validator accepts and which
    seats no held-out eval."""

    tally = SimpleNamespace(
        planned=[], skipped=[], skip_counts={}, considered=0, threshold_skip_fraction=0.0
    )
    held_out = {"val_scenes_file": "val.json", "val_data_root": "/held/out"}
    for overrides, trained, scored in (
        ({"query_anchor_source": "gt_depth", "oracle_query_anchor": True, **held_out}, "gt_depth", "query_position"),
        ({"query_anchor_source": "gt_depth", **held_out}, "gt_depth", "gt_depth"),
        (
            {"query_anchor_source": "query_position", "eval_query_anchor_source": "predicted", **held_out},
            "query_position",
            "predicted",
        ),
        ({"query_anchor_source": "gt_depth"}, "gt_depth", None),
        ({"query_anchor_source": "gt_depth", "val_data_root": "/held/out"}, "gt_depth", None),
    ):
        args = _validator_args(tmp_path, manifest="m.jsonl", **overrides)
        train_cli._validate_args(args)
        settings = train_cli._plan_summary(tally, args)["settings"]
        assert settings["query_anchor_source"] == trained, overrides
        assert settings["eval_query_anchor_source"] == scored, overrides
        assert settings["oracle_query_anchor"] is (scored == "query_position"), overrides
        assert settings["ground_truth_query_anchor"] is (scored == "gt_depth"), overrides


# ----------------------------------------------------------------- resume ---


@pytest.mark.parametrize(
    "before, after",
    (("predicted", "gt_depth"), ("gt_depth", "query_position"), ("query_position", "predicted")),
)
def test_a_resume_that_changes_the_training_anchor_source_is_refused(tmp_path, before, after):
    """Refused-tier on the loss-weight ground: the source decides what every
    step's position term is computed against, so a segment that switched it
    would report one curve over two objectives while `loss` kept its name."""

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path, query_anchor_source=before))
    assert stored["query_anchor_source"] == before
    with pytest.raises(RuntimeError, match=re.escape(f"carries query_anchor_source={before!r}")):
        train_cli.check_resume_settings(stored, _loop_args(tmp_path, query_anchor_source=after))
    train_cli.check_resume_settings(stored, _loop_args(tmp_path, query_anchor_source=before))


def test_a_checkpoint_predating_the_training_anchor_source_resumes_as_predicted(tmp_path):
    """Absence resolves, it is not tolerated: a pre-flag checkpoint could only
    have composed its loss on its own anchor, so it resumes silently at
    "predicted" and is refused against either ground-truth reading."""

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    del stored["query_anchor_source"]

    train_cli.check_resume_settings(stored, _loop_args(tmp_path))
    for source in _GROUND_TRUTH_SOURCES:
        with pytest.raises(RuntimeError, match="predates 'query_anchor_source'"):
            train_cli.check_resume_settings(
                stored, _loop_args(tmp_path, query_anchor_source=source)
            )
    assert "query_anchor_source" in train_cli._RESUME_SETTINGS_REFUSED
    assert train_cli._RESUME_SETTINGS_ABSENT_DEFAULTS["query_anchor_source"] == "predicted"


# ----------------------------------------------------- vocabulary, surfaces ---


def test_the_source_vocabulary_and_every_signature_that_takes_it():
    """One vocabulary, owned by the gather and exported by the package: both
    flags choose from the tuple itself, an unknown reading never parses, and
    every new parameter is
    keyword-only -- the training one on train_step and the eval's own with
    the default that reproduces today, the eval's record of the training
    source required, like query_anchors. Neither function keeps the two
    booleans the source replaced."""

    assert QUERY_ANCHOR_SOURCES == ("predicted", "query_position", "gt_depth")
    assert "QUERY_ANCHOR_SOURCES" in training_package.__all__
    parser = train_cli.build_arg_parser()
    actions = {action.dest: action for action in parser._actions}
    assert actions["query_anchor_source"].choices is QUERY_ANCHOR_SOURCES
    assert actions["query_anchor_source"].default == "predicted"
    assert actions["eval_query_anchor_source"].choices is QUERY_ANCHOR_SOURCES
    assert actions["eval_query_anchor_source"].default is None
    for flag in ("--query_anchor_source", "--eval_query_anchor_source"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--manifest", "m.jsonl", flag, "oracle"])

    for function, name, default in (
        (gather_query_anchor_points, "source", "predicted"),
        (train_cli.train_step, "query_anchor_source", "predicted"),
        (train_cli.evaluate_held_out, "eval_query_anchor_source", "predicted"),
        (train_cli.evaluate_held_out, "query_anchor_source", inspect.Parameter.empty),
    ):
        parameter = inspect.signature(function).parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (function.__qualname__, name)
        assert parameter.default == default, (function.__qualname__, name)
    for function in (gather_query_anchor_points, train_cli.evaluate_held_out):
        for name in ("oracle_query_anchor", "ground_truth_query_anchor"):
            assert name not in inspect.signature(function).parameters
