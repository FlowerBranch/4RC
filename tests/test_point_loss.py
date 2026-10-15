"""Pins for the point term and the geometry preset: --point_weight.

Nothing trained the reconstruction before this: DualDPT and CameraDec were
frozen under every preset and read only through detached paths. The point
term supervises it directly -- every slot's predicted depth unprojected
through its predicted camera, intrinsics detached, mapped into the stored
world through the step's detached Sim(3) and compared with the ground-truth
points -- and the preset ``temporal_tracking_global_attention_geometry``
trains DualDPT's depth path and CameraDec without its fov head through that
term alone.

The value tests run on the 56-pixel fixture's IDENTITY grid, where the
ground-truth principal point is exactly the image centre the pose encoding
implies, so an exact reconstruction reproduces the ground truth to float
rounding. Camera 1 is yawed and pitched (its depth varies across the image),
and each time gets its own depth, so a target read from the wrong slot, the
wrong time or a transposed grid is a nonzero loss.

The step tests drive the real ``train_step`` with a fake whose tap is a
scalar: both routes -- the step's one backward through the cut, and separate
backwards of the uncut graph -- then add the same numbers in the same order,
so the gradients must be EQUAL, not close. That is float32, not float64: the
real step pins the Sim(3) to float32 and the pose encoding builds its
intrinsics in the default dtype, so a float64 model cannot run through it. The
chain rule itself is pinned in float64 on a nonlinear toy below.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_temporal_tracking as train_cli
from arc.models.arc.arc import Arc
from arc.models.arc.heads.cam_dec import CameraDec
from arc.models.arc.heads.dualdpt import DualDPT
from arc.models.arc.utils.transform import (
    mat_to_quat,
    predicted_pointmaps,
    quat_to_mat,
    unproject_pose_encoding,
)
from arc.training import (
    DetachedSim3,
    build_anchor_correspondences,
    fit_scene_sim3,
    gather_query_anchor_points,
    point_loss,
    reconstruction_point_loss,
    sparse_tracking_loss,
)
from arc.training import runtime
from arc.training.checkpoint import (
    load_temporal_patch_model,
    load_temporal_tracking_checkpoint,
)
from arc.training.manifest_plan import plan_record
from arc.training.sparse_tracking import _metric_pointmap_at_anchor
from scene_fixtures import fixture_scene
from test_manifest_plan import _record
from test_runtime import _meta_arc
from test_temporal_patch_scripts import (
    _fill_distinct,
    _stub_from_pretrained,
    _trainable,
    _with_table,
)
from test_time_indexing import (
    _FakeTrackHead,
    _GlobalAttnTinyArc,
    _GradBackbone,
    _GradCameraDecoder,
    _GradMotionDecoder,
    _GradReconstructionHead,
    _arc_shell,
    _trainer_patch,
)
from test_trainer_loop import (
    _FakeArc,
    _loop_args,
    _micro_step,
    _plans,
    _recording_step,
    _step_scene,
    _toy_model,
    _validator_args,
)
from test_track_refinement import _RecordingScaler, _spy_on_cut

_GEOMETRY = "temporal_tracking_global_attention_geometry"
_GLOBAL = "temporal_tracking_global_attention"
_HUBER = 0.05


# ------------------------------------------------------------- the fixture ---


def _window(**scene_kwargs):
    """Two cameras by four times on the identity grid, camera 1 yawed and
    pitched, and every time its own depth: time t's depth is scaled by
    1 + 0.1 t, so the ground-truth points move with the time read."""

    scene = fixture_scene(
        cameras=(0, 1), times=(0, 1, 2, 3), size=56, rotated_camera=1, **scene_kwargs
    )
    for time in range(scene.depth.shape[1]):
        scene.depth[:, time] *= 1.0 + 0.1 * time
    return scene


def _exact_reconstruction(scene):
    """``(depth, pose_enc)`` reproducing every slot's ground-truth points.

    Each slot's own depth map at its own time on the model grid, its
    ground-truth camera centre and rotation, and the fovs of its ground-truth
    focal lengths. Exact only on the identity grid, where the ground-truth
    principal point is (W/2, H/2), the one the pose encoding implies.
    """

    height, width = scene.views[0]["img"].shape[-2:]
    depth = torch.zeros(1, scene.num_observations, height, width, dtype=torch.float64)
    pose = torch.zeros(1, scene.num_observations, 9, dtype=torch.float64)
    for observation in scene.observations:
        rows, columns = observation.image_transform.output_to_original_indices()
        columns_grid, rows_grid = np.meshgrid(columns, rows)
        surface = scene.surface_depth_map(
            observation.camera, observation.original_time
        ).double().numpy()
        depth[0, observation.slot] = torch.from_numpy(surface[rows_grid, columns_grid])
        world_to_camera = scene.extrinsics_world_to_camera[
            observation.camera, observation.original_time
        ].double()
        rotation = world_to_camera[:3, :3]
        pose[0, observation.slot, :3] = -(rotation.mT @ world_to_camera[:3, 3])
        pose[0, observation.slot, 3:7] = mat_to_quat(rotation.mT)
        intrinsics = scene.intrinsics[observation.camera, observation.original_time].double()
        pose[0, observation.slot, 7] = 2.0 * torch.atan((height / 2.0) / intrinsics[1, 1])
        pose[0, observation.slot, 8] = 2.0 * torch.atan((width / 2.0) / intrinsics[0, 0])
    return depth.float(), pose.float()


def _identity():
    return DetachedSim3(scale=torch.tensor(1.0), rotation=torch.eye(3), translation=torch.zeros(3))


def _rotation(yaw_degrees, pitch_degrees):
    yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
    yaw_matrix = torch.tensor(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=torch.float64,
    )
    pitch_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=torch.float64,
    )
    return pitch_matrix @ yaw_matrix


def _loss(depth, pose, scene, alignment=None):
    return reconstruction_point_loss(
        {"depth": depth, "pose_enc": pose},
        scene,
        _identity() if alignment is None else alignment,
        huber_delta_m=_HUBER,
    )


def _huber(value):
    magnitude = abs(value)
    if magnitude < _HUBER:
        return 0.5 * magnitude * magnitude
    return _HUBER * (magnitude - 0.5 * _HUBER)


def _valid_counts(scene):
    return [
        int(_metric_pointmap_at_anchor(scene, slot)[1].sum())
        for slot in range(scene.num_observations)
    ]


def _leaves(depth, pose):
    return depth.clone().requires_grad_(True), pose.clone().requires_grad_(True)


# ---------------------------------------------------------------- the term ---


def test_the_point_loss_is_zero_at_ground_truth_depth_and_cameras():
    """At the exact reconstruction the term is float rounding, nothing more:
    every slot is paired with its own camera's and its own time's targets,
    on both pixel axes. A second reconstruction planted in a scaled, rotated
    and TRANSLATED gauge is zero again through the matching Sim(3) -- and
    clearly not through the identity, so the gauge is real; the translation
    is what parts apply_points from apply_vectors. Both run at a metric
    factor of 2, so lifting to metres before the Sim(3) instead of after is
    off by the translation."""

    scene = _window()
    scene.track_upscaling_factor = 2.0
    depth, pose = _exact_reconstruction(scene)
    assert _loss(depth, pose, scene).item() < 1e-10

    scale = 1.7
    rotation = _rotation(30.0, 10.0)
    translation = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    gauged_pose = pose.double().clone()
    for slot in range(scene.num_observations):
        centre = gauged_pose[0, slot, :3].clone()
        camera_to_world = quat_to_mat(gauged_pose[0, slot, 3:7])
        gauged_pose[0, slot, :3] = rotation.mT @ (centre - translation) / scale
        gauged_pose[0, slot, 3:7] = mat_to_quat(rotation.mT @ camera_to_world)
    gauged_depth = depth / scale
    alignment = DetachedSim3(
        scale=torch.tensor(scale), rotation=rotation.float(), translation=translation.float()
    )

    assert _loss(gauged_depth, gauged_pose.float(), scene, alignment).item() < 1e-9
    assert _loss(gauged_depth, gauged_pose.float(), scene).item() > 1e-2


def test_the_point_loss_equals_a_planted_offset_in_metres():
    """Two camera centres moved by known offsets move every point of their
    slot by exactly that much: one offset in the Huber's quadratic regime,
    one in its linear regime, both lifted by a track_upscaling_factor of 2.
    The expected value pools the two slots' valid pixels with every other
    slot's zeros over ALL valid pixels, and the second slot has rows of
    invalid depth, so a mean of per-slot means is a different number."""

    scene = _window()
    scene.track_upscaling_factor = 2.0
    quadratic_slot, linear_slot = 1, 6
    camera, time = (
        scene.observations[linear_slot].camera,
        scene.observations[linear_slot].original_time,
    )
    scene.depth[camera, time, 0, :10] = 0.0
    depth, pose = _exact_reconstruction(scene)
    offsets = {
        quadratic_slot: (0.01, -0.02, 0.015),
        linear_slot: (0.1, 0.0, -0.08),
    }
    for slot, offset in offsets.items():
        pose[0, slot, :3] += torch.tensor(offset)

    counts = _valid_counts(scene)
    assert counts[linear_slot] < counts[quadratic_slot]
    expected = sum(
        counts[slot] * sum(_huber(2.0 * component) for component in offset)
        for slot, offset in offsets.items()
    ) / (3 * sum(counts))

    assert _loss(depth, pose, scene).item() == pytest.approx(expected, rel=1e-5)


def test_the_intrinsics_get_no_gradient_and_every_slot_trains():
    """The fov channels of pose_enc get an exactly-zero gradient; depth and
    the translation and rotation channels get one on every slot. Through a
    real CameraDec the fov head is reached only through pose_enc's
    concatenation, which hands it an exact zero, never None -- the reason the
    preset freezes it by name rather than leaving it to a guard."""

    scene = _window()
    depth, pose = _exact_reconstruction(scene)
    pose[0, :, 0] += 0.01 * torch.arange(1, scene.num_observations + 1)
    depth, pose = _leaves(depth, pose)
    _loss(depth, pose, scene).backward()

    assert torch.all(pose.grad[..., 7:] == 0)
    for slot in range(scene.num_observations):
        assert pose.grad[0, slot, :7].abs().sum() > 0, slot
        assert depth.grad[0, slot].abs().sum() > 0, slot

    torch.manual_seed(0)
    decoder = CameraDec(dim_in=8)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.mul_(0.01)
    base_depth, base_pose = _exact_reconstruction(scene)
    pose_from_decoder = base_pose + decoder(torch.randn(1, scene.num_observations, 8))
    _loss(base_depth, pose_from_decoder, scene).backward()

    for parameter in decoder.fc_fov.parameters():
        assert parameter.grad is not None
        assert torch.all(parameter.grad == 0)
    for module in (decoder.backbone, decoder.fc_t, decoder.fc_qvec):
        for parameter in module.parameters():
            assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("which", ["reference", "other"])
def test_reference_and_non_reference_slots_both_contribute(which):
    """An offset on the query slot alone, and on a slot of the other camera at
    another time alone: each raises the loss, and the gradient it adds lands
    on that slot only -- every other slot's gradient is unchanged to the bit,
    since its own residual and the pooled count did not move."""

    scene = _window()
    assert scene.query_observation_slot != 6
    slot = scene.query_observation_slot if which == "reference" else 6
    depth, pose = _exact_reconstruction(scene)
    baseline_depth, baseline_pose = _leaves(depth, pose)
    baseline = _loss(baseline_depth, baseline_pose, scene)
    baseline.backward()
    pose[0, slot, :3] += torch.tensor([0.03, -0.01, 0.02])
    moved_depth, moved_pose = _leaves(depth, pose)
    moved = _loss(moved_depth, moved_pose, scene)
    moved.backward()

    assert moved.item() > baseline.item() + 1e-6
    for other in range(scene.num_observations):
        depth_change = moved_depth.grad[0, other] - baseline_depth.grad[0, other]
        pose_change = moved_pose.grad[0, other] - baseline_pose.grad[0, other]
        if other == slot:
            assert depth_change.abs().sum() > 0 and pose_change.abs().sum() > 0
        else:
            assert torch.all(depth_change == 0) and torch.all(pose_change == 0), other


def test_a_non_finite_depth_outside_the_mask_cannot_poison_the_gradient():
    """The camera-to-world product sums the pose gradient over every pixel,
    so an infinite predicted depth at a pixel with no ground truth would turn
    that camera's gradient NaN through 0 * inf. Zeroed before the
    unprojection, it contributes exactly nothing."""

    scene = _window()
    scene.depth[0, 0, 0, 3, 4] = 0.0
    depth, pose = _exact_reconstruction(scene)
    slot = next(
        observation.slot
        for observation in scene.observations
        if observation.camera == 0 and observation.original_time == 0
    )
    assert not _metric_pointmap_at_anchor(scene, slot)[1][3, 4]
    depth[0, slot, 3, 4] = float("inf")
    pose[0, slot, :3] += torch.tensor([0.02, 0.0, 0.0])
    depth, pose = _leaves(depth, pose)

    loss = _loss(depth, pose, scene)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(depth.grad).all()
    assert torch.isfinite(pose.grad).all()
    assert pose.grad[0, slot, :3].abs().sum() > 0


def test_a_non_finite_prediction_where_there_is_ground_truth_is_refused():
    """Divergence must be loud: a NaN or infinite predicted depth at a pixel
    the term supervises raises, naming the reconstruction -- it is never
    dropped from the mask the way an unsupervised pixel is."""

    scene = _window()
    for value in (float("nan"), float("inf")):
        depth, pose = _exact_reconstruction(scene)
        assert _metric_pointmap_at_anchor(scene, 2)[1][7, 9]
        depth[0, 2, 7, 9] = value
        with pytest.raises(FloatingPointError, match="Predicted reconstruction contains NaN"):
            _loss(depth, pose, scene)


def test_a_reconstruction_with_the_wrong_slot_count_is_refused():
    """One slot's depth or camera would otherwise broadcast over every slot."""

    scene = _window()
    depth, pose = _exact_reconstruction(scene)

    with pytest.raises(ValueError, match=re.escape("depth must have shape (1,8,H,W)")):
        _loss(depth[:, :1], pose, scene)
    with pytest.raises(ValueError, match=re.escape("pose_enc must have shape (1,8,9)")):
        _loss(depth, pose[:, :1], scene)


def test_an_empty_point_mask_names_the_point_loss():
    predicted = torch.zeros(2, 3, 3)
    with pytest.raises(ValueError, match="No pixels selected for the point loss"):
        point_loss(predicted, predicted.clone(), torch.zeros(2, 3, dtype=torch.bool), huber_delta=_HUBER)


def test_one_unprojection_serves_the_detached_and_the_trained_reader():
    """predicted_pointmaps is unproject_pose_encoding under no_grad, bit for
    bit, and still detached; the point term's call keeps the graph."""

    scene = _window()
    depth, pose = _leaves(*_exact_reconstruction(scene))

    detached = predicted_pointmaps({"depth": depth, "pose_enc": pose})
    with torch.no_grad():
        direct = unproject_pose_encoding(depth.float(), pose.float())

    assert torch.equal(detached, direct)
    assert not detached.requires_grad
    assert unproject_pose_encoding(depth, pose).requires_grad


# ----------------------------------------------------- the step's backward ---


class _GeometryFakeArc(_FakeArc):
    """_FakeArc with the reconstruction heads trainable, as the geometry
    preset leaves them.

    The tap is an "encoder block" plus the time table, times a constant one:
    the Mul node saves its operand, so a second backward through the UNCUT
    taps raises as the real encoder's graph does, which is what makes the
    forced cut a behavioural pin. The track field adds the motion decoder and
    the track head to the tap; the reconstruction perturbs the scene's exact
    ground truth by the depth head and camera decoder applied to the tap. So
    the point term reaches head, cam_dec, the block and the table, the track
    loss reaches the motion decoder, the track head, the block and the table,
    and neither reaches the other's heads. Records what reconstruct was
    handed and the depth head's gradient at the first head pass.
    """

    PATCH_SIZE = 14

    def __init__(self, scene):
        height, width = scene.views[0]["img"].shape[-2:]
        super().__init__(scene.num_observations, height, width)
        self.backbone.pretrained.block = nn.Linear(1, 1)
        self.head.requires_grad_(True)
        self.cam_dec.requires_grad_(True)
        self.ground_depth, self.ground_pose = _exact_reconstruction(scene)
        # Distinct per slot and zero on the fovs: the decoder moves every
        # slot's translation and rotation its own way, never the intrinsics.
        steps = torch.arange(1, scene.num_observations + 1, dtype=torch.float32)
        direction = torch.zeros(1, scene.num_observations, 9)
        direction[0, :, 0] = steps
        direction[0, :, 1] = -0.5 * steps
        direction[0, :, 4] = 0.1 * steps
        self.pose_direction = direction
        self.reconstruct_inputs = []
        self.head_grad_at_first_pass = "unset"
        self.depth_graph_freed_at_first_pass = None

    def encode_features(
        self,
        images,
        ref_view_strategy="first",
        time_indices=None,
        depth_maps=None,
        camera_vectors=None,
    ):
        pretrained = self.backbone.pretrained
        scale = (
            pretrained.block.weight.sum()
            + pretrained.block.bias.sum()
            + pretrained.time_index_embedding.weight.sum()
        )
        return [scale * torch.ones(())]

    def reconstruct(self, feats, images):
        self.reconstruct_inputs.append(feats)
        (tap,) = feats
        depth = self.ground_depth * (
            1.0 + 0.01 * torch.tanh(self.head.weight.sum() * tap + self.head.bias.sum())
        )
        shift = 0.01 * torch.tanh(self.cam_dec.weight.sum() * tap + self.cam_dec.bias.sum())
        self.reconstructed_depth = depth
        return {"depth": depth, "pose_enc": self.ground_pose + shift * self.pose_direction}

    def track_for_query(self, feats, images, query_idx, views_per_time=1, merge=False, **kwargs):
        if self.head_grad_at_first_pass == "unset":
            grad = self.head.weight.grad
            self.head_grad_at_first_pass = None if grad is None else grad.clone()
            self.depth_graph_freed_at_first_pass = _graph_freed(self.reconstructed_depth)
        (tap,) = feats
        field = (
            tap
            + self.motion_decoder.weight.sum()
            + self.motion_decoder.bias.sum()
            + self.track_head.weight.sum()
            + self.track_head.bias.sum()
        )
        return super().track_for_query(
            [field], images, query_idx, views_per_time=views_per_time, merge=merge
        )


def _graph_freed(tensor):
    """Whether a backward has already released ``tensor``'s saved tensors:
    reading one then raises, while a graph that was never backwarded, or was
    backwarded with retain_graph, still hands them over."""

    node = tensor.grad_fn
    for name in dir(node):
        if name.startswith("_saved_"):
            try:
                getattr(node, name)
            except RuntimeError:
                return True
    return False


_RECONSTRUCTION = ("head.", "cam_dec.")
_TRACK = ("motion_decoder.", "track_head.")
_ENCODER = ("backbone.",)


def _grads(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _geometry_step(model, scene, *, point_weight, scaler=None, **overrides):
    kwargs = dict(
        model=model,
        scene=scene,
        plan=plan_record(_record(seq_name="0000"), budget=48, stride=2),
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=_RecordingScaler() if scaler is None else scaler,
        precision="32",
        huber_delta_m=_HUBER,
        grad_clip=1e9,
        confidence_weight=0.0,
        confidence_alpha=None,
        sync_weight=0.0,
        velocity_weight=0.0,
        point_weight=point_weight,
        learning_rates=[1e-3],
        step=0,
        accum_steps=1,
        window_start=True,
        window_end=False,
    )
    kwargs.update(overrides)
    return train_cli.train_step(**kwargs)


def _separate_backwards(model, scene, *, point_weight, window_scale=1.0):
    """The step's objective on an identical copy, as separate backwards of
    the UNCUT graph in the step's order -- the point term, then each active
    anchor's weighted total, retain_graph throughout -- so every gradient
    accumulates into .grad by plain addition. Returns the copy, the point
    loss, the gradients after the point backward alone and after everything,
    and the scalars backwarded."""

    reference = copy.deepcopy(model)
    correspondences, _ = build_anchor_correspondences(scene)
    anchor_count = len(scene.anchor_observation_slots)
    counts = runtime.anchor_sample_counts(scene, correspondences, anchor_count)
    images, feats = runtime.encode_views(reference, scene.views)
    recon = reference.reconstruct(feats, images)
    alignment, _ = fit_scene_sim3(recon, scene)
    anchors, frame = gather_query_anchor_points(recon, scene, correspondences)
    point = reconstruction_point_loss(recon, scene, alignment, huber_delta_m=_HUBER)
    scalars = [point * point_weight * window_scale]
    scalars[0].backward(retain_graph=True)
    point_only = _grads(reference)
    for anchor_index, count in enumerate(counts):
        if count == 0:
            continue
        raw = runtime.anchor_tracks(reference, feats, images, scene, anchor_index)
        result = sparse_tracking_loss(
            runtime.tracking_only(raw),
            scene,
            correspondences.select_query_slot(anchor_index),
            alignment,
            anchors[correspondences.anchor_rows(anchor_index)],
            query_anchor_frame=frame,
            huber_delta_m=_HUBER,
            collect_diagnostics=False,
        )
        total = runtime.weighted_anchor_total(
            result,
            position_weight=count / sum(counts),
            confidence_weight=0.0,
            sync_weight=0.0,
            velocity_weight=0.0,
        )
        scalars.append(total * window_scale * 1.0)
        scalars[-1].backward(retain_graph=True)
    return (
        reference,
        point,
        point_only,
        _grads(reference),
        [float(value.detach()) for value in scalars],
    )


def _assert_split(stepped, point_only, combined):
    """Reconstruction heads from the point term alone, track heads from the
    track loss alone, encoder and table the sum -- every one to the bit."""

    for name, gradient in stepped.items():
        if name.startswith(_RECONSTRUCTION):
            assert torch.equal(gradient, point_only[name]), name
        elif name.startswith(_TRACK):
            assert point_only[name] is None, name
        else:
            assert name.startswith(_ENCODER), name
            assert point_only[name] is not None, name
            assert not torch.equal(combined[name], point_only[name]), name
        assert torch.equal(gradient, combined[name]), name


def test_two_anchors_equal_two_separate_backwards(monkeypatch):
    """The live shape: two anchors, the point term first. The step's one
    backward through the cut equals the point backward plus every anchor's,
    each gradient to the bit; the scalars arrive as point·weight·window,
    then each anchor's share; and the breakdown carries the point term."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)
    _, point, point_only, combined, scalars = _separate_backwards(
        model, scene, point_weight=0.5
    )
    scaler = _RecordingScaler()

    outcome = _geometry_step(model, scene, point_weight=0.5, scaler=scaler)

    _assert_split(_grads(model), point_only, combined)
    assert scaler.scaled == scalars
    assert len(scalars) == 3
    assert list(outcome.loss_breakdown) == ["position", "point"]
    assert outcome.loss_breakdown["position"] == outcome.loss
    assert outcome.loss_breakdown["point"] == float(point.item())


def test_one_anchor_forces_the_cut_and_the_reconstruction_reads_it(monkeypatch):
    """At one anchor the step skips the cut unless the point term runs: then
    both backwards reach the taps, so the cut is forced -- without it the
    second would raise -- and reconstruct is handed the cut's leaves, not
    the encoder's own taps."""

    scene = _step_scene(monkeypatch)
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)
    _, _, point_only, combined, scalars = _separate_backwards(model, scene, point_weight=0.5)
    cuts = _spy_on_cut(monkeypatch)
    scaler = _RecordingScaler()

    _geometry_step(model, scene, point_weight=0.5, scaler=scaler)

    assert cuts == [True]
    (tap,) = model.reconstruct_inputs[-1]
    assert tap.is_leaf and tap.requires_grad
    _assert_split(_grads(model), point_only, combined)
    assert scaler.scaled == scalars

    torch.manual_seed(0)
    unweighted = _GeometryFakeArc(scene)
    cuts.clear()
    _geometry_step(unweighted, scene, point_weight=0.0)
    assert cuts == []
    (tap,) = unweighted.reconstruct_inputs[-1]
    assert tap.grad_fn is not None


def test_the_window_scale_rides_the_point_scalar(monkeypatch):
    """Under --grad_accum the point term is a draw of the window like every
    anchor: its scalar is weighted by 1/N, and the gradients match separate
    backwards at the same scale."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)
    _, point, point_only, combined, scalars = _separate_backwards(
        model, scene, point_weight=0.5, window_scale=0.5
    )
    scaler = _RecordingScaler()

    outcome = _geometry_step(model, scene, point_weight=0.5, scaler=scaler, accum_steps=2)

    _assert_split(_grads(model), point_only, combined)
    assert scaler.scaled == scalars
    # Reported undivided, like the position term: 1/N weights the window,
    # not this draw's own value.
    assert outcome.loss_breakdown["point"] == float(point.item())

    # Every micro-step of the window adds the term, not only the first.
    scaler = _RecordingScaler()
    second = _geometry_step(
        model, scene, point_weight=0.5, scaler=scaler, accum_steps=2, step=1,
        window_start=False,
    )
    assert scaler.scaled == scalars
    assert second.loss_breakdown["point"] == float(point.item())


def test_the_point_term_runs_once_per_step_under_refinement(monkeypatch):
    """At --refine_iters 2 every anchor backwards twice, the point term
    once, first, undiscounted; the reconstruction's gradient is still the
    point term's alone."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)
    _, point, point_only, _, _ = _separate_backwards(model, scene, point_weight=0.5)
    scaler = _RecordingScaler()

    _geometry_step(model, scene, point_weight=0.5, scaler=scaler, refine_iters=2)

    assert scaler.scaled[0] == float((point * 0.5).detach())
    assert len(scaler.scaled) == 1 + 2 * 2
    stepped = _grads(model)
    for name, gradient in stepped.items():
        if name.startswith(_RECONSTRUCTION):
            assert torch.equal(gradient, point_only[name]), name


def test_the_track_loss_sends_nothing_into_the_reconstruction():
    """With the term off and the heads trainable, the anchors and the Sim(3)
    are read from the fake's OWN reconstruction -- the pointmap seam is not
    planted here, so a gather that kept its graph would carry the track loss
    into DualDPT and CameraDec -- and their gradients stay None while the
    track loss still reaches the encoder through the tap."""

    scene = fixture_scene(
        time_count=4,
        view_count=2,
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
        query_anchors=((0, 0), (1, 0)),
        invisible=((0, 0, 2),),
    )
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)

    _geometry_step(model, scene, point_weight=0.0)

    stepped = _grads(model)
    for name, gradient in stepped.items():
        if name.startswith(_RECONSTRUCTION):
            assert gradient is None, name
        else:
            assert gradient is not None, name
    assert model.head_grad_at_first_pass is None


def test_the_point_backward_runs_before_the_first_head_pass(monkeypatch):
    """Backwarding the term first is what frees the depth path's graph before
    any head graph allocates: by the first track_for_query the depth head
    already holds its gradient, and the depth's saved tensors are gone -- a
    backward that retained the graph would keep them alive through every
    head pass."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)

    _geometry_step(model, scene, point_weight=0.5)

    assert model.head_grad_at_first_pass is not None
    assert torch.equal(model.head_grad_at_first_pass, model.head.weight.grad)
    assert model.depth_graph_freed_at_first_pass is True


def test_the_cut_sums_both_terms_into_one_encoder_backward_in_float64():
    """The chain rule the step rests on, on a NONLINEAR encoder with two
    taps: the point term and two anchors backwarded onto a cut, then one
    backward through the encoder, equal separate backwards of the uncut
    graph to float64 rounding."""

    torch.manual_seed(0)
    encoder = nn.Sequential(nn.Linear(3, 4), nn.Tanh()).double()
    reconstruction = nn.Linear(4, 2).double()
    track_head = nn.Linear(4, 3).double()
    inputs = torch.randn(5, 3, dtype=torch.float64)
    point_target = torch.randn(5, 2, dtype=torch.float64)
    track_target = torch.randn(5, 3, dtype=torch.float64)
    modules = (encoder, reconstruction, track_head)

    def taps(encoder):
        hidden = encoder(inputs)
        return [hidden, hidden.square()]

    def point(feats, reconstruction):
        return F.huber_loss(reconstruction(feats[0] * feats[1].tanh()), point_target, delta=_HUBER)

    def track(feats, track_head, anchor):
        return F.huber_loss(track_head(feats[0] + anchor * feats[1]), track_target, delta=_HUBER)

    separate = copy.deepcopy(modules)
    feats = taps(separate[0])
    (0.5 * point(feats, separate[1])).backward(retain_graph=True)
    for anchor in (0.3, 0.7):
        track(feats, separate[2], anchor).backward(retain_graph=True)

    feats = taps(encoder)
    cut, pairs = runtime.cut_features(feats)
    (0.5 * point(cut, reconstruction)).backward()
    for anchor in (0.3, 0.7):
        track(cut, track_head, anchor).backward()
    runtime.backward_through_cut(pairs)

    for combined, reference in zip(modules, separate):
        for left, right in zip(combined.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-12, atol=1e-15)


# ------------------------------------------------------------- reporting ---


def test_the_point_term_reaches_the_breakdown_and_the_history(monkeypatch, tmp_path):
    """Unweighted in the breakdown, the reported position loss unmoved by the
    term, and both in the step's history row; the reconstruction's gradient
    norm rides the window's norms only when its heads train."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)
    torch.manual_seed(0)
    unweighted = _GeometryFakeArc(scene)
    _, point, _, _, _ = _separate_backwards(model, scene, point_weight=0.5)

    outcome = _geometry_step(model, scene, point_weight=0.5, window_end=True)
    position_only = _geometry_step(unweighted, scene, point_weight=0.0)

    assert outcome.loss == position_only.loss
    assert position_only.loss_breakdown is None
    assert outcome.loss_breakdown["point"] == float(point.item())
    assert "reconstruction" in outcome.gradient_norms
    path = train_cli.open_step_history(tmp_path, start_step=0)
    train_cli.append_step_history(path, outcome)
    (row,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert row["loss_breakdown"]["point"] == outcome.loss_breakdown["point"]
    assert row["gradient_norms"]["reconstruction"] == outcome.gradient_norms["reconstruction"]

    frozen = _FakeArc(scene.num_observations, *scene.views[0]["img"].shape[-2:])
    frozen_outcome = _micro_step(
        frozen,
        scene,
        torch.optim.AdamW([{"params": list(frozen.parameters()), "lr": 1e-3}]),
        step=0,
        accum_steps=1,
        index=0,
        last=True,
    )
    assert "reconstruction" not in frozen_outcome.gradient_norms


def test_the_reconstruction_norm_is_read_before_the_clip(monkeypatch):
    """The number exists to be read against clipped_total, so it must be the
    window's gradient as the backward left it: equal to the norm of the
    separate backwards' reconstruction gradient, and above a clip that
    binds. A norm taken after clip_grad_norm_ would be scaled down with
    every other group."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)
    reference, _, _, _, _ = _separate_backwards(model, scene, point_weight=0.5)
    clip = 1e-6

    outcome = _geometry_step(model, scene, point_weight=0.5, window_end=True, grad_clip=clip)

    expected = runtime.gradient_norm(
        [p for module in (reference.head, reference.cam_dec) for p in module.parameters()]
    )
    assert outcome.gradient_norms["reconstruction"] == expected
    assert expected > clip
    assert outcome.gradient_norms["clipped_total"] > clip


def test_the_reconstruction_norm_is_read_after_the_unscale(monkeypatch):
    """Under an enabled scaler the backwarded scalars are multiplied by the
    loss scale; the norm must be read from the gradients unscale_ returns,
    which a power-of-two scale makes bit-identical to the unscaled ones."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)
    reference, _, _, _, _ = _separate_backwards(model, scene, point_weight=0.5)

    outcome = _geometry_step(
        model,
        scene,
        point_weight=0.5,
        scaler=torch.amp.GradScaler("cpu", init_scale=1024.0),
        window_end=True,
    )

    assert outcome.gradient_norms["reconstruction"] == runtime.gradient_norm(
        [p for module in (reference.head, reference.cam_dec) for p in module.parameters()]
    )


def test_the_point_term_is_reported_after_every_track_term(monkeypatch):
    """The breakdown's order is the composition order of the anchors' terms
    with the step's own term last, so a history row reads the same way
    whichever terms ran."""

    scene = _step_scene(monkeypatch, query_anchors=((0, 0), (1, 0)), invisible=((0, 0, 2),))
    torch.manual_seed(0)
    model = _GeometryFakeArc(scene)

    outcome = _geometry_step(model, scene, point_weight=0.5, velocity_weight=0.25)

    assert list(outcome.loss_breakdown) == ["position", "velocity", "point"]


def test_point_weight_is_a_required_keyword_of_the_step():
    parameter = inspect.signature(train_cli.train_step).parameters["point_weight"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


# -------------------------------------------------- the preset, the model ---


def test_the_geometry_preset_trains_exactly_the_depth_and_pose_paths():
    """The global-attention set plus every DualDPT parameter outside the ray
    branch and every CameraDec parameter outside the fov head -- selected
    here by name pattern, independently of set_freeze's module list -- and
    exactly the pinned counts."""

    geometry = _meta_arc(_GEOMETRY)
    base = _meta_arc(_GLOBAL)

    trainable = {name for name, p in geometry.named_parameters() if p.requires_grad}
    expected = {name for name, p in base.named_parameters() if p.requires_grad}
    expected |= {f"head.{name}" for name, _ in geometry.head.named_parameters() if "_aux" not in name}
    expected |= {
        f"cam_dec.{name}"
        for name, _ in geometry.cam_dec.named_parameters()
        if not name.startswith("fc_fov.")
    }
    assert trainable == expected
    assert any("_aux" in name for name, _ in geometry.head.named_parameters())

    report = runtime.assert_trainable_parameter_set(
        geometry, freeze_mode=_GEOMETRY, max_time_indices=32
    )
    non_embedding = report["parameter_count"] - 32 * runtime.TIME_EMBEDDING_DIM
    assert (report["tensor_count"], non_embedding) == (553, 765_710_349)
    assert runtime.EXPECTED_TRAINABLE_SETS[_GEOMETRY] == (553, 765_710_349)


@pytest.mark.parametrize(
    "depth_input, camera_input, refine",
    [(True, False, False), (False, True, True), (True, True, True)],
)
def test_the_geometry_preset_prices_the_injections_and_the_refiner(
    depth_input, camera_input, refine
):
    model = _meta_arc()
    model.set_freeze(_GEOMETRY, depth_input=depth_input, camera_input=camera_input, refine=refine)

    runtime.assert_trainable_parameter_set(
        model,
        freeze_mode=_GEOMETRY,
        max_time_indices=32,
        depth_input=depth_input,
        camera_input=camera_input,
        refine=refine,
    )
    assert not model.cam_dec.fc_fov[0].weight.requires_grad


def _dualdpt_missing_a_ray_module():
    model = _GeometryTinyArc(freeze=_GLOBAL)
    del model.head.scratch.refinenet4_aux
    return model


@pytest.mark.parametrize(
    "build",
    [lambda: _GlobalAttnTinyArc(freeze=_GLOBAL), _dualdpt_missing_a_ray_module],
    ids=["stand_in_head", "dualdpt_missing_a_ray_module"],
)
def test_a_head_without_the_named_modules_is_refused_untouched(build):
    """set_freeze resolves the modules it re-freezes before its first
    requires_grad_ call, so a head lacking them raises with every flag, and
    the mode, exactly as they were: a stand-in head without DualDPT's
    surface, and a DualDPT missing one of the modules RAY_BRANCH names."""

    model = build()
    before = {name: p.requires_grad for name, p in model.named_parameters()}

    with pytest.raises(AttributeError):
        model.set_freeze(_GEOMETRY)

    assert {name: p.requires_grad for name, p in model.named_parameters()} == before
    assert model.freeze == _GLOBAL


@pytest.mark.parametrize("freeze", ["none", *Arc.TEMPORAL_FREEZE_MODES])
def test_reconstruct_keeps_its_graph_only_where_a_loss_trains_it(freeze):
    model = _arc_shell(max_time_indices=32)
    model.backbone = _GradBackbone()
    model.head = _GradReconstructionHead()
    model.cam_dec = _GradCameraDecoder()
    model.motion_decoder = _GradMotionDecoder()
    model.track_head = _FakeTrackHead()
    model.freeze = freeze
    images = torch.zeros(1, 4, 3, 2, 2)

    output = model.reconstruct(model.encode_features(images), images)

    keeps = freeze == "none" or freeze in Arc.GEOMETRY_FREEZE_MODES
    assert output["depth"].requires_grad is keeps
    assert output["pose_enc"].requires_grad is keeps
    # Under a caller's no_grad -- the held-out eval and inference -- no mode
    # builds the graph, the geometry preset included: here the taps still
    # carry the backbone's graph, as the geometry heads' own parameters do in
    # the real model, so only the caller's context stands in the way.
    feats = model.encode_features(images)
    with torch.no_grad():
        output = model.reconstruct(feats, images)
    assert not output["depth"].requires_grad
    assert not output["pose_enc"].requires_grad


def test_the_geometry_heads_get_their_own_group_at_the_encoder_rate():
    """Appended after the encoder blocks, holding exactly the trainable
    reconstruction parameters, at the encoder blocks' rate unless
    --geometry_lr says otherwise."""

    model = _meta_arc(_GEOMETRY)

    optimizer, rates, _ = runtime.build_optimizer(model, lr=1e-3)

    assert len(optimizer.param_groups) == 4
    reconstruction = {
        id(p)
        for module in (model.head, model.cam_dec)
        for p in module.parameters()
        if p.requires_grad
    }
    assert {id(p) for p in optimizer.param_groups[3]["params"]} == reconstruction
    assert optimizer.param_groups[3]["lr"] == rates["geometry"] == rates["encoder_blocks"]
    assert rates["geometry"] == pytest.approx(1e-4)
    _, rates, _ = runtime.build_optimizer(model, lr=1e-3, encoder_lr=3e-6)
    assert rates["geometry"] == 3e-6
    optimizer, rates, _ = runtime.build_optimizer(
        model, lr=1e-3, encoder_lr=3e-6, geometry_lr=7e-6
    )
    assert (rates["encoder_blocks"], rates["geometry"]) == (3e-6, 7e-6)
    # The rate the group trains at, not only the one reported.
    assert [group["lr"] for group in optimizer.param_groups[2:]] == [3e-6, 7e-6]


@pytest.mark.parametrize("freeze, groups", [("temporal_tracking", 2), (_GLOBAL, 3)])
def test_other_presets_build_no_geometry_group(freeze, groups):
    model = _meta_arc(freeze)

    optimizer, rates, _ = runtime.build_optimizer(model, lr=1e-3, geometry_lr=7e-6)

    assert len(optimizer.param_groups) == groups
    assert rates["geometry"] is None


class _GeometryTinyArc(_GlobalAttnTinyArc):
    """The tiny global-attention Arc with small REAL reconstruction heads:
    the geometry preset re-freezes DualDPT's ray branch and CameraDec's fov
    head by name, which nn.Linear stand-ins do not have."""

    def __init__(self, freeze="none", max_time_indices=4):
        super().__init__(freeze="none")
        self.head = DualDPT(dim_in=8, output_dim=2, features=8, out_channels=(8, 8, 8, 8))
        self.cam_dec = CameraDec(dim_in=8)
        _with_table(self, max_time_indices)
        self.set_freeze(freeze)


def test_a_geometry_patch_runs_back_as_trained_and_only_under_its_preset(
    tmp_path, monkeypatch
):
    """Through the trainer's own writer and the shared loader: the preset
    comes back from the patch, the reconstruction heads' tensors with it --
    never the ray branch or the fov head -- each under its own name; and the
    same patch is refused under the global-attention preset, by mode."""

    trained = _GeometryTinyArc(max_time_indices=6)
    trained.set_freeze(_GEOMETRY)
    _fill_distinct(trained)
    patch = _trainer_patch(trained, tmp_path / "patch", freeze_mode=_GEOMETRY, point_weight=0.5)
    calls = _stub_from_pretrained(monkeypatch, _GeometryTinyArc)

    model, _ = load_temporal_patch_model("hub/4rc", patch, device=torch.device("cpu"))

    assert calls == [("hub/4rc", {"max_time_indices": 6})]
    assert model.freeze == _GEOMETRY
    expected, loaded = _trainable(trained), _trainable(model)
    assert loaded.keys() == expected.keys()
    assert any(name.startswith("head.") for name in loaded)
    assert any(name.startswith("cam_dec.") for name in loaded)
    assert not any("_aux" in name or "fc_fov" in name for name in loaded)
    for name, value in expected.items():
        assert torch.equal(loaded[name], value), name

    other = _GeometryTinyArc(freeze=_GLOBAL, max_time_indices=6)
    with pytest.raises(ValueError, match=f"trained under freeze mode '{_GEOMETRY}'"):
        load_temporal_tracking_checkpoint(other, patch)


# ------------------------------------------------- the ray branch's graph ---


# Written out here rather than read from DualDPT.RAY_BRANCH, so the tests check
# that list instead of trusting it.
_RAY_MODULES = (
    "refinenet1_aux",
    "refinenet2_aux",
    "refinenet3_aux",
    "refinenet4_aux",
    "output_conv1_aux",
    "output_conv2_aux",
)
_RAY_OUTPUTS = ("ray", "ray_conf", "ray_list", "ray_conf_list")
# 2 x 3 patches, so the two image axes stay distinct.
_IMAGE_SIZE = (28, 42)


def _taps(observations, *, dim=8, size=_IMAGE_SIZE):
    """Four encoder taps as reconstruct reads them -- patch tokens
    (1, S, patches, C) and a camera token (1, S, C) each -- distinct per
    element and requiring grad, as the cut's leaves do in a geometry step.
    Without the grad a frozen ray branch would record no graph whatever the
    gate does, and the tests below would pass vacuously."""

    generator = torch.Generator().manual_seed(0)
    height, width = size
    patches = (height // 14) * (width // 14)
    return [
        (
            torch.randn(1, observations, patches, dim, generator=generator).requires_grad_(),
            torch.randn(1, observations, dim, generator=generator).requires_grad_(),
        )
        for _ in range(4)
    ]


def _images(observations, size=_IMAGE_SIZE):
    return torch.zeros(1, observations, 3, *size)


def _flatten(output, keys):
    """Every tensor under ``keys``, list entries included."""

    for key in keys:
        value = output[key]
        yield from value if isinstance(value, list) else (value,)


def _open_the_gate(model):
    """Through the gate's own condition, not a switch: the ray branch trains."""

    for name in _RAY_MODULES:
        getattr(model.head.scratch, name).requires_grad_(True)


def _as_the_trainer_runs_it(model):
    """train_step's modes: the model in train(), DualDPT and CameraDec in
    eval(), which is the mode every geometry step runs the gate in."""

    model.train()
    model.head.eval()
    model.cam_dec.eval()
    return model


def _reconstruct_recording_saves(model, feats, images):
    """reconstruct, plus how many tensors autograd saved for backward while
    each ray module ran, and in all. The hooks sit on every submodule with a
    nesting counter, because a ModuleList's own forward never runs and both
    output_conv*_aux are ModuleLists."""

    depth = dict.fromkeys(_RAY_MODULES, 0)
    saves = dict.fromkeys(_RAY_MODULES, 0)
    total = 0
    handles = []
    for name in _RAY_MODULES:
        for module in getattr(model.head.scratch, name).modules():

            def enter(module, args, name=name):
                depth[name] += 1

            def leave(module, args, output, name=name):
                depth[name] -= 1

            handles.append(module.register_forward_pre_hook(enter))
            handles.append(module.register_forward_hook(leave))

    def pack(tensor):
        nonlocal total
        total += 1
        for name, level in depth.items():
            if level:
                saves[name] += 1
        return tensor

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = model.reconstruct(feats, images)
    finally:
        for handle in handles:
            handle.remove()
    return output, saves, total


# DualDPT's mode: eval() as train_step runs it, and train() too, so a gate that
# read the module mode fails one of the two.
_HEAD_MODES = pytest.mark.parametrize(
    "head_training", [False, True], ids=["head_eval", "head_train"]
)


@_HEAD_MODES
@pytest.mark.parametrize("observations", [4, 10])
def test_the_geometry_preset_records_no_graph_for_the_ray_branch(observations, head_training):
    """depth and pose_enc keep their graph for the point backward; no ray
    output requires grad, and nothing is saved for backward inside any of
    the six ray modules. The per-module count is what catches one un-gated
    refinenet: output_conv1_aux's gate downstream drops that level's graph
    before _fuse returns, so the outputs alone would still read clean. Four
    observations run DualDPT in one chunk, ten through its chunk loop. The
    recorder's list is pinned against RAY_BRANCH and against the head's own
    *_aux children, so neither can lose a module unseen."""

    model = _GeometryTinyArc(freeze=_GEOMETRY)
    model.head.train(head_training)
    model.cam_dec.train(head_training)
    feats = _taps(observations)

    output, saves, total = _reconstruct_recording_saves(model, feats, _images(observations))

    assert set(_RAY_MODULES) == set(DualDPT.RAY_BRANCH)
    assert set(_RAY_MODULES) == {
        name for name, _ in model.head.scratch.named_children() if name.endswith("_aux")
    }
    assert all(tensor.requires_grad for tap in feats for tensor in tap)
    assert output["depth"].requires_grad
    assert output["pose_enc"].requires_grad
    assert not any(tensor.requires_grad for tensor in _flatten(output, _RAY_OUTPUTS))
    assert saves == dict.fromkeys(_RAY_MODULES, 0)
    assert total > 0


def _missing_reconstruction_gradients(model, output):
    """Backward a loss reading every output; the head and camera-decoder
    parameters it did not reach."""

    sum(tensor.mean() for tensor in _flatten(output, output.keys())).backward()
    reconstruction = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(_RECONSTRUCTION)
    }
    assert any("_aux" in name for name in reconstruction)
    return [name for name, parameter in reconstruction.items() if parameter.grad is None]


@_HEAD_MODES
@pytest.mark.parametrize("observations", [4, 10])
def test_none_keeps_the_ray_branch_graph_and_trains_every_parameter(observations, head_training):
    """Under none the ray branch trains: every ray output requires grad, each
    of the six modules saves tensors (so the recorder above is not blind),
    and a loss reading every output reaches every head and camera-decoder
    parameter. Outputs that merely require grad would let one module run
    under no_grad and silently stop training."""

    model = _GeometryTinyArc(freeze="none")
    model.head.train(head_training)
    model.cam_dec.train(head_training)

    output, saves, _ = _reconstruct_recording_saves(
        model, _taps(observations), _images(observations)
    )

    assert all(tensor.requires_grad for tensor in _flatten(output, _RAY_OUTPUTS))
    assert all(count > 0 for count in saves.values()), saves
    assert _missing_reconstruction_gradients(model, output) == []


def test_the_gate_keys_on_the_ray_parameters_not_on_the_taps():
    """A trainable ray branch records its graph whether or not its input
    requires grad: on detached taps -- a head trained on frozen features --
    every ray output requires grad and every parameter is reached."""

    model = _as_the_trainer_runs_it(_GeometryTinyArc(freeze="none"))
    feats = [tuple(tensor.detach() for tensor in tap) for tap in _taps(10)]

    output = model.reconstruct(feats, _images(10))

    assert all(tensor.requires_grad for tensor in _flatten(output, _RAY_OUTPUTS))
    assert _missing_reconstruction_gradients(model, output) == []


@pytest.mark.parametrize("observations", [4, 10])
def test_the_gate_never_turns_grad_on_under_a_callers_no_grad(observations):
    """The official eval and inference.py run the released model -- every
    parameter requiring grad, in eval() -- under torch.no_grad(). The gate may
    only turn grad off: a bare set_grad_enabled(<ray trains>) would switch it
    back on for the ray branch. Nothing at all is saved for backward, in one
    chunk and through DualDPT's chunk loop -- the outputs alone would miss a
    graph recorded inside the branch and dropped before its end. The taps are
    built outside no_grad, as a backbone with a graph would hand them over, so
    only the caller's context stands in the way."""

    model = _GeometryTinyArc(freeze="none").eval()
    assert all(parameter.requires_grad for parameter in model.parameters())
    feats = _taps(observations)

    with torch.no_grad():
        output, _, total = _reconstruct_recording_saves(model, feats, _images(observations))

    assert not any(tensor.requires_grad for tensor in _flatten(output, output.keys()))
    assert total == 0


@pytest.mark.parametrize("freeze", ["none", *Arc.TEMPORAL_FREEZE_MODES])
def test_the_gate_changes_no_value_in_any_preset(freeze):
    """Gated, and with the gate opened through its own condition, every
    output is EQUAL: the gate decides only whether a graph is recorded. Under
    none the two runs coincide and under the frozen presets reconstruct's
    no_grad decides first; the geometry preset is the pair that differs."""

    model = _GeometryTinyArc()
    model.set_freeze(
        freeze,
        late_global_blocks=1 if freeze == "temporal_tracking_late_global" else None,
    )
    _as_the_trainer_runs_it(model)
    feats = _taps(10)

    gated = model.reconstruct(feats, _images(10))
    _open_the_gate(model)
    opened = model.reconstruct(feats, _images(10))

    if freeze in Arc.GEOMETRY_FREEZE_MODES:
        assert not gated["ray"].requires_grad
        assert opened["ray"].requires_grad
    assert gated.keys() == opened.keys()
    for key in gated:
        pairs = zip(_flatten(gated, [key]), _flatten(opened, [key]), strict=True)
        for left, right in pairs:
            assert torch.equal(left, right), key


def test_the_gate_changes_no_depth_or_pose_gradient():
    """The ray branch never reached the point loss, so cutting its graph
    moves no gradient: from a loss on depth, its confidence and pose_enc,
    every parameter gradient and every tap gradient is EQUAL gated and with
    the gate opened, and the two runs are pinned as gated and opened."""

    model = _as_the_trainer_runs_it(_GeometryTinyArc(freeze=_GEOMETRY))
    pose_weights = torch.arange(1.0, 10.0)

    def gradients():
        feats = _taps(10)
        output = model.reconstruct(feats, _images(10))
        model.zero_grad(set_to_none=True)
        loss = (
            output["depth"].mean()
            + 0.5 * output["depth_conf"].mean()
            + (output["pose_enc"] * pose_weights).sum()
        )
        loss.backward()
        parameters = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        # What reconstruct reads: the four patch taps and the camera token
        # CameraDec takes from the last tap.
        read = [tap[0] for tap in feats] + [feats[-1][1]]
        return parameters, [tensor.grad for tensor in read], output["ray"].requires_grad

    gated_parameters, gated_taps, gated_ray = gradients()
    _open_the_gate(model)
    opened_parameters, opened_taps, opened_ray = gradients()

    assert not gated_ray
    assert opened_ray
    assert any(name.startswith("head.") for name in gated_parameters)
    assert any(name.startswith("cam_dec.") for name in gated_parameters)
    assert gated_parameters.keys() == opened_parameters.keys()
    for name, gradient in gated_parameters.items():
        assert torch.equal(opened_parameters[name], gradient), name
    for gated_tap, opened_tap in zip(gated_taps, opened_taps, strict=True):
        assert torch.equal(gated_tap, opened_tap)


def _shapes(module):
    return {name: tuple(parameter.shape) for name, parameter in module.named_parameters()}


def test_the_geometry_preset_saves_no_ray_branch_tensors_at_production_width():
    """The point of the change, at Arc's own head widths and one 378x504
    observation: gated, the forward saves at least 0.5 GiB less for backward
    than with the gate opened -- the ray branch's share, measured at 0.532
    GiB (0.942 against 0.409 per observation). Relative rather than
    absolute, since the totals move with what a torch release has autograd
    keep for each op. Distinct storages, counted at pack time, without the
    parameters' and the taps' own storage, which exist with or without a
    graph."""

    torch.manual_seed(0)
    model = _GeometryTinyArc()
    model.head = DualDPT(dim_in=3072, output_dim=2, features=256)
    model.cam_dec = CameraDec(dim_in=3072)
    model.set_freeze(_GEOMETRY)
    _as_the_trainer_runs_it(model)
    reference = _meta_arc()
    assert _shapes(model.head) == _shapes(reference.head)
    assert _shapes(model.cam_dec) == _shapes(reference.cam_dec)
    size = (378, 504)
    feats = _taps(1, dim=3072, size=size)
    kept = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
    kept |= {tensor.untyped_storage().data_ptr() for tap in feats for tensor in tap}

    def saved_bytes():
        storages = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in kept:
                storages[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = model.reconstruct(feats, _images(1, size=size))
        assert output["depth"].requires_grad
        return sum(storages.values())

    gated = saved_bytes()
    _open_the_gate(model)
    opened = saved_bytes()

    gib = 2**30
    assert opened - gated >= 0.5 * gib, (
        f"gated {gated / gib:.3f} GiB, opened {opened / gib:.3f} GiB"
    )


@pytest.mark.parametrize("position", [0, -1], ids=["first", "last"])
@pytest.mark.parametrize("name", _RAY_MODULES)
def test_any_one_trainable_ray_parameter_opens_the_gate(name, position):
    """A ray parameter -- not all of them, and not a whole module: under the
    geometry preset with this one tensor trainable, the first or the last of
    its module, the whole branch records its graph again, so every ray
    output requires grad and a loss on them reaches that tensor."""

    model = _as_the_trainer_runs_it(_GeometryTinyArc(freeze=_GEOMETRY))
    parameter = list(getattr(model.head.scratch, name).parameters())[position]
    parameter.requires_grad_(True)

    output = model.reconstruct(_taps(4), _images(4))

    assert all(tensor.requires_grad for tensor in _flatten(output, _RAY_OUTPUTS))
    sum(tensor.mean() for tensor in _flatten(output, _RAY_OUTPUTS)).backward()
    assert parameter.grad is not None


# ---------------------------------------------------------------- the flags ---


def _parsed(*argv):
    args = train_cli.build_arg_parser().parse_args(["--manifest", "m.jsonl", *argv])
    train_cli._validate_args(args)
    return args


def test_the_flags_default_to_no_term_under_the_global_attention_preset():
    args = _parsed()
    assert (args.point_weight, args.geometry_lr, args.freeze_mode) == (0.0, None, _GLOBAL)

    args = _parsed("--freeze_mode", _GEOMETRY, "--point_weight", "0.5", "--geometry_lr", "2e-6")
    assert (args.point_weight, args.geometry_lr) == (0.5, 2e-6)


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--point_weight", "0.5"], f"--point_weight 0.5 needs --freeze_mode {_GEOMETRY}"),
        (
            ["--freeze_mode", "temporal_tracking", "--point_weight", "0.5"],
            f"--point_weight 0.5 needs --freeze_mode {_GEOMETRY}",
        ),
        (["--freeze_mode", _GEOMETRY], f"--freeze_mode {_GEOMETRY} needs --point_weight > 0"),
        (["--geometry_lr", "1e-5"], f"--geometry_lr needs --freeze_mode {_GEOMETRY}"),
        (["--freeze_mode", _GEOMETRY, "--point_weight", "-1"], "--point_weight must be finite"),
        (["--freeze_mode", _GEOMETRY, "--point_weight", "nan"], "--point_weight must be finite"),
        (["--freeze_mode", _GEOMETRY, "--point_weight", "inf"], "--point_weight must be finite"),
        (
            ["--freeze_mode", _GEOMETRY, "--point_weight", "0.5", "--geometry_lr=-1e-5"],
            "--geometry_lr must be finite and positive",
        ),
        (
            ["--freeze_mode", _GEOMETRY, "--point_weight", "0.5", "--geometry_lr", "nan"],
            "--geometry_lr must be finite and positive",
        ),
        (
            ["--freeze_mode", _GEOMETRY, "--point_weight", "0.5", "--geometry_lr", "0"],
            "--geometry_lr must be finite and positive",
        ),
        (
            ["--freeze_mode", _GEOMETRY, "--point_weight", "0.5", "--geometry_lr", "inf"],
            "--geometry_lr must be finite and positive",
        ),
    ],
)
def test_the_point_term_and_the_preset_are_refused_apart(argv, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _parsed(*argv)


def test_a_resume_that_changes_the_point_weight_is_refused(tmp_path):
    stored = train_cli._checkpoint_settings(
        _loop_args(tmp_path, freeze_mode=_GEOMETRY, point_weight=0.5)
    )
    assert stored["point_weight"] == 0.5

    for changed in (0.25, 0.0):
        with pytest.raises(RuntimeError, match=re.escape("carries point_weight=0.5")):
            train_cli.check_resume_settings(stored, _loop_args(tmp_path, point_weight=changed))
    train_cli.check_resume_settings(
        stored, _loop_args(tmp_path, freeze_mode=_GEOMETRY, point_weight=0.5)
    )


def test_a_checkpoint_predating_the_point_term_resumes_without_it(tmp_path):
    """Absence resolves, it is not tolerated: no pre-flag run could have
    trained the reconstruction, so it resumes silently at 0 and is refused
    with the term switched on."""

    stored = train_cli._checkpoint_settings(_loop_args(tmp_path))
    del stored["point_weight"]

    train_cli.check_resume_settings(stored, _loop_args(tmp_path))
    with pytest.raises(RuntimeError, match="predates 'point_weight'"):
        train_cli.check_resume_settings(
            stored, _loop_args(tmp_path, freeze_mode=_GEOMETRY, point_weight=0.5)
        )
    assert "point_weight" in train_cli._RESUME_SETTINGS_REFUSED
    assert train_cli._RESUME_SETTINGS_ABSENT_DEFAULTS["point_weight"] == 0.0


def _toy_run(output_dir, args, freeze):
    model = _toy_model()
    model.freeze = freeze
    train_cli.run_training(
        model=model,
        optimizer=torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-3}]),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        plans=_plans(4),
        args=args,
        scene_provider=lambda plan: SimpleNamespace(name=plan.seq_name),
        step_fn=_recording_step([]),
        output_dir=output_dir,
    )


@pytest.mark.parametrize("trained, resumed", [(_GEOMETRY, _GLOBAL), (_GLOBAL, _GEOMETRY)])
def test_the_patch_loader_refuses_a_resume_across_the_geometry_preset(tmp_path, trained, resumed):
    """The freeze mode is no resume setting: the patch loader refuses it
    against the payload's own field before any state is restored. Reached
    here with every setting equal, so the mode is the only difference."""

    train_cli._STOP_REQUESTED.clear()
    _toy_run(tmp_path / "a", _loop_args(tmp_path, num_steps=4, point_weight=0.5), trained)
    checkpoint = tmp_path / "a" / "train_state.pt"
    assert checkpoint.is_file()

    with pytest.raises(ValueError, match=f"trained under freeze mode '{trained}'"):
        _toy_run(
            tmp_path / "b",
            _loop_args(tmp_path, num_steps=4, point_weight=0.5, resume=str(checkpoint)),
            resumed,
        )


def test_the_plan_summary_records_the_point_weight(tmp_path):
    tally = SimpleNamespace(
        planned=[], skipped=[], skip_counts={}, considered=0, threshold_skip_fraction=0.0
    )
    for freeze, weight in ((_GLOBAL, 0.0), (_GEOMETRY, 0.5)):
        args = _validator_args(tmp_path, manifest="m.jsonl", freeze_mode=freeze, point_weight=weight)
        assert train_cli._plan_summary(tally, args)["settings"]["point_weight"] == weight


def test_the_trainer_wires_the_point_weight_and_the_geometry_rate():
    """Inspected rather than executed, as the other wiring pins are: main()
    is .to("cuda") unconditionally. Every step call reads --point_weight,
    every optimizer build reads --geometry_lr, the coupling reads the preset
    from Arc rather than a second copy of its name, and the step's term reads
    the step's own Huber knee."""

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

    assert calls_named(tree, "step_fn"), "run_training has no step_fn call"
    for call in calls_named(tree, "step_fn"):
        assert reads_args(keyword(call, "point_weight"), "point_weight")
    assert calls_named(tree, "build_optimizer"), "main has no build_optimizer call"
    for call in calls_named(tree, "build_optimizer"):
        assert reads_args(keyword(call, "geometry_lr"), "geometry_lr")

    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    validate = functions["_validate_args"]
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "GEOMETRY_FREEZE_MODES"
        and isinstance(node.value, ast.Name)
        and node.value.id == "Arc"
        for node in ast.walk(validate)
    )
    assert not any(
        isinstance(node, ast.Constant) and node.value == _GEOMETRY for node in ast.walk(validate)
    )
    (term,) = calls_named(functions["train_step"], "reconstruction_point_loss")
    assert [item.arg for item in term.keywords] == ["huber_delta_m"]
    assert isinstance(keyword(term, "huber_delta_m"), ast.Name)
    assert keyword(term, "huber_delta_m").id == "huber_delta_m"
