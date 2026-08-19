"""Pins for the replay's last mile: the provider and the prediction schema.

Three hazards live here, and none of them raises on its own.

A **set-based** track selection silently drops query points, because a record's
``track_indices`` repeat by design. The **covered-timestep** convention in
``query_points[:, 0]`` is a plain integer either way, so writing original frame
numbers produces a well-formed file that scores every query against the wrong row.
And the **axis change** from this repo's camera-major observations to the
scorers' fused timesteps is the kind of transpose that yields plausible numbers
when wrong.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_temporal_tracking as train_cli
from arc.training.predictions import (
    PREDICTION_KEYS,
    build_prediction_arrays,
    write_scene_predictions,
)
from arc.training.scene_provider import (
    SceneProviderError,
    rotation_from_degrees,
    select_tracks_positionally,
)
from test_sparse_tracking import _write_scene


@pytest.fixture
def dumped_scene(tmp_path):
    """A real two-camera window on disk. There is no conftest, so it is local."""

    from arc.training import load_dumped_kubric_scene

    _write_scene(tmp_path, time_count=4, view_count=2, depth_sidecar=True)
    return load_dumped_kubric_scene(
        tmp_path, "0000", cameras=(0, 1), times=(0, 1, 2, 3), size=56
    )


# ------------------------------------------------------------ vendored copy ---


def test_the_vendored_scene_archive_is_byte_identical_to_its_recorded_hash():
    """The layout rule it owns is subtle enough that a local patch would hurt.

    Its docstring spells out that view indices are *positions in a numerically
    sorted list*, never the N in ``view_N`` -- and that both plausible shortcuts
    are silently wrong on real scenes. Two copies drifting apart would mis-index
    cameras with nothing downstream able to notice, so the copy is pinned.
    """

    source = Path(train_cli.__file__).parent / "arc" / "training" / "scene_archive.py"
    text = source.read_text()
    header, _, body = text.partition('"""Read one MV-Kubric scene')
    body = '"""Read one MV-Kubric scene' + body

    recorded = re.findall(r"^#\s*sha256\s*:\s*([0-9a-f]{64})\s*$", header, re.MULTILINE)
    assert recorded, "the vendoring header must record the upstream hash"
    assert hashlib.sha256(body.encode()).hexdigest() == recorded[0]


# ------------------------------------------------------------ track selection ---


def test_repeated_track_ids_yield_repeated_positions():
    """The hazard the manifest schema names, one layer below the planner.

    A repeat is a second, independent column with its own query timestep. The
    planner already keeps duplicates in the record; this is the point where a
    ``set`` would finally drop them -- and the loss would simply supervise fewer
    points than the manifest says, with every count still looking plausible.
    """

    selection = select_tracks_positionally([7, 3, 9, 1], [3, 7, 3, 1, 3])

    assert selection.positions == (1, 0, 1, 3, 1)
    assert selection.found == 5
    assert selection.requested == 5
    assert selection.missing == ()


def test_ids_the_pool_does_not_carry_are_reported_not_skipped():
    """Whether a gap is tolerable is policy; hiding it is never policy."""

    selection = select_tracks_positionally([7, 3], [3, 42, 7, 99])

    assert selection.positions == (1, 0)
    assert selection.missing == (42, 99)
    assert selection.found == 2
    assert selection.requested == 4


def test_a_repeated_id_in_the_pool_itself_maps_deterministically():
    """First occurrence wins, so two runs agree on which column an id means."""

    assert select_tracks_positionally([5, 5, 5], [5]).positions == (0,)


# --------------------------------------------------------------- track route ---


def _provider(**kwargs):
    from arc.training.scene_provider import MVTrackerSceneProvider

    kwargs.setdefault("min_shared_queries", 2)
    return MVTrackerSceneProvider(**kwargs)


class _Plan:
    """Just the two fields the route reads."""

    def __init__(self, track_indices=()):
        self.seq_name = "0000"
        self.track_indices = tuple(track_indices)


def test_the_default_route_supervises_the_pool_and_ignores_the_recorded_draw():
    """V6(c): a recorded draw is not reproducible, so it is not the default.

    The record here names ids the pool does not carry *and* omits ids it does.
    The default route must not consult it at all -- not intersect with it, not
    prefer it, not order by it. Every eligible column is supervised, in pool
    order, and nothing is counted as missing because nothing was requested.
    """

    provider = _provider()

    selection = provider.select_columns(_Plan([42, 99, 7]), [7, 3, 9, 1])

    assert selection.positions == (0, 1, 2, 3)
    assert selection.requested == 4
    assert selection.missing == ()
    assert (provider.requested_track_ids, provider.missing_track_ids) == (0, 0)


def test_the_opt_in_route_counts_missing_ids_and_does_not_raise():
    """A raise here would fire on essentially every real record.

    Two draws of one scene share 739 then 665 of their ~1800-1850 unique ids, so
    "all requested ids present" is not a condition any manifest row satisfies.
    The gap is reported instead -- and it accumulates across steps, because one
    record's shortfall says nothing while a run's total says how far the replay
    drifted.
    """

    provider = _provider(honour_recorded_tracks=True)

    first = provider.select_columns(_Plan([3, 42, 7, 99]), [7, 3])
    provider.select_columns(_Plan([7, 5, 3]), [7, 3])

    assert first.positions == (1, 0)
    assert first.missing == (42, 99)
    assert (provider.requested_track_ids, provider.missing_track_ids) == (7, 3)


def test_the_pool_guard_measures_the_pool_even_when_the_draw_is_honoured():
    """The threshold must never see the intersection, on either route.

    A small intersection is the *expected* case -- jaccard 0.256 and 0.219 over
    two measurements -- so a threshold on matched ids would re-impose exactly
    what dropping the raise removed. Here a healthy pool with an almost-disjoint
    record passes.
    """

    provider = _provider(min_shared_queries=4, honour_recorded_tracks=True)

    selection = provider.select_columns(_Plan([9, 42, 99]), [7, 3, 9, 1])

    assert selection.positions == (2,)
    assert provider.missing_track_ids == 2


def test_a_record_sharing_nothing_with_a_healthy_pool_is_refused_separately():
    """Zero columns is not a small selection; it is an empty gather.

    Kept apart from the pool guard so the message names the real cause: the
    scene is fine, the record simply does not intersect it.
    """

    with pytest.raises(SceneProviderError, match="nothing to supervise"):
        _provider(honour_recorded_tracks=True).select_columns(_Plan([42, 99]), [7, 3])


def test_the_opt_in_route_falls_back_when_a_record_carries_no_draw():
    """An eval plan has no ``track_indices``; the flag must not empty the step."""

    provider = _provider(honour_recorded_tracks=True)

    assert provider.select_columns(_Plan([]), [7, 3, 9]).positions == (0, 1, 2)


def test_the_pool_guard_reports_the_pool_size_not_a_missing_intersection():
    """``--min_shared_queries`` changed meaning with the route, so its message did.

    It now says "this scene is too small to be worth a step". Reading the old
    message on the new route would send someone looking for a manifest mismatch
    that is neither present nor, any longer, checked.
    """

    with pytest.raises(SceneProviderError) as error:
        _provider(min_shared_queries=64).select_columns(_Plan(), [7, 3])

    message = str(error.value)
    assert "only 2 eligible tracks" in message
    assert "not a" in message and "recorded draw" in message


# ------------------------------------------------------------ scene transform ---


def test_the_recorded_rotation_is_applied_x_then_y():
    """Order is not symmetric, and every label moves with it.

    Composing y-then-x gives a different world; the sample would still be
    internally consistent and simply be somewhere else, which no shape check
    catches.
    """

    rotation = rotation_from_degrees(30.0, 45.0)

    x, y = np.radians(30.0), np.radians(45.0)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])

    np.testing.assert_allclose(rotation.numpy(), ry @ rx, atol=1e-6)
    # And it is a proper rotation, which transform_scene asserts on entry.
    np.testing.assert_allclose(
        (rotation @ rotation.T).numpy(), np.eye(3), atol=1e-6
    )
    assert float(torch.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-5)


def test_an_absent_transform_is_a_no_op_rather_than_an_identity_guess():
    """`scene_transform` is null when the augmentation was off; that is legal."""

    from arc.training.scene_provider import apply_recorded_scene_transform

    sentinel = object()
    assert apply_recorded_scene_transform(sentinel, None) is sentinel
    assert apply_recorded_scene_transform(sentinel, {}) is sentinel


class _Sample:
    """The four fields the transform moves, plus the fifth that it must.

    ``track_upscaling_factor`` is 1.0 because that is what the loader actually
    hands us: `scale` is bound to 1.0 at `kubric_multiview_dataset.py:1055` and
    only the augmentation branch or the mutually-exclusive VGGT branch changes
    it, and the provider enables neither.
    """

    def __init__(self):
        self.videodepth = "depth-in"
        self.extrs = "extrs-in"
        self.query_points_3d = "query-in"
        self.trajectory_3d = "traj-in"
        self.track_upscaling_factor = 1.0


@pytest.fixture
def stub_transform_scene(monkeypatch):
    """Stand in for MVTracker's ``transform_scene``, which is not importable here.

    The arithmetic inside it is deliberately MVTracker's -- re-deriving it in this
    repo is the silent divergence the provider exists to avoid. What *is* ours is
    the wiring, and that is what this pins.
    """

    import sys
    import types

    calls = []

    def transform_scene(*args, **kwargs):
        assert not args, "the provider must pass by keyword; a positional slip is silent"
        calls.append(kwargs)
        # Four distinguishable returns, so a write-back in the wrong order fails.
        return ("depth-out", "extrs-out", "query-out", "traj-out", "ignored-tail")

    utils = types.ModuleType("mvtracker.datasets.utils")
    utils.transform_scene = transform_scene
    datasets = types.ModuleType("mvtracker.datasets")
    datasets.utils = utils
    root = types.ModuleType("mvtracker")
    root.datasets = datasets
    monkeypatch.setitem(sys.modules, "mvtracker", root)
    monkeypatch.setitem(sys.modules, "mvtracker.datasets", datasets)
    monkeypatch.setitem(sys.modules, "mvtracker.datasets.utils", utils)
    return calls


# V6(d), job 19740383, the "both on" path -- i.e. what the curve run actually
# produces. Kept as literals because a rounded stand-in would not exercise the
# float path the manifest carries.
REAL_TRANSFORM = {
    "scale": 0.8746830544934553,
    "rot_x_deg": 12.000626399623517,
    "rot_y_deg": 14.784707373444487,
}


def test_a_real_recorded_transform_reaches_transform_scene_intact(stub_transform_scene):
    """The wiring, pinned against a transform the cluster actually produced.

    Until V6(d) ran, every probe saw ``scene_transform=None``, so this path had
    never been exercised with real values. Skipping or mis-wiring it is the one
    failure that leaves the sample internally consistent and simply elsewhere --
    no shape check, no loss curve and no gate can see it.
    """

    from arc.training.scene_provider import apply_recorded_scene_transform

    sample = _Sample()
    returned = apply_recorded_scene_transform(sample, REAL_TRANSFORM)

    assert returned is sample
    assert len(stub_transform_scene) == 1
    call = stub_transform_scene[0]

    # Scale passes through exactly. Measured below 1 here and above 1 on the
    # other path (1.4946), so this is not a magnify-only augmentation and
    # anything that clamped it would be wrong in one direction only.
    assert call["transformation_scale"] == REAL_TRANSFORM["scale"]
    assert isinstance(call["transformation_scale"], float)

    expected = rotation_from_degrees(
        REAL_TRANSFORM["rot_x_deg"], REAL_TRANSFORM["rot_y_deg"]
    )
    np.testing.assert_allclose(
        call["transformation_rotation"].numpy(), expected.numpy(), atol=0
    )

    # The sample's fields go in under the names transform_scene expects...
    assert call["depth"] == "depth-in"
    assert call["extrs"] == "extrs-in"
    assert call["query_points"] == "query-in"
    assert call["traj3d_world"] == "traj-in"

    # ...and the returns come back to the right attributes. A rotation of this
    # write-back is the mis-wiring that survives every other check.
    assert sample.videodepth == "depth-out"
    assert sample.extrs == "extrs-out"
    assert sample.query_points_3d == "query-out"
    assert sample.trajectory_3d == "traj-out"

    # The fifth field, which transform_scene does not touch and the provider must.
    # Every consumer reads X_metres = X_stored * factor, and the transform just
    # multiplied every world quantity by `scale`, so the factor absorbs it.
    assert sample.track_upscaling_factor == pytest.approx(1.0 / REAL_TRANSFORM["scale"])


def test_a_shrinking_transform_scales_the_metric_factor_the_same_way(stub_transform_scene):
    """Both directions, because the measured draws straddle 1.0.

    V6(d) saw scale 1.4946 and 0.8747 -- upstream draws U(0.8, 1.5) -- so a fix
    that only handled magnification would be wrong on roughly half of all scenes
    and right on the other half, which is the hardest kind of wrong to notice.
    A *divide* holds in both directions; assigning `1/scale` only coincides
    because the loader's value is 1.0.
    """

    from arc.training.scene_provider import apply_recorded_scene_transform

    for scale in (1.4945987811164994, 0.8746830544934553, 1.0):
        sample = _Sample()
        sample.track_upscaling_factor = 2.0
        apply_recorded_scene_transform(sample, {"scale": scale})
        assert sample.track_upscaling_factor == pytest.approx(2.0 / scale)


def test_a_record_without_a_transform_leaves_the_metric_factor_alone():
    """The no-op case is a real one: `scene_transform` is null whenever the
    augmentation was off, and a run that quietly rescaled those scenes would
    disagree with the dumped path for no reason."""

    from arc.training.scene_provider import apply_recorded_scene_transform

    sample = _Sample()
    sample.track_upscaling_factor = 3.5
    apply_recorded_scene_transform(sample, None)
    assert sample.track_upscaling_factor == 3.5


def test_the_transform_is_read_by_key_not_by_position(stub_transform_scene):
    """``rot_x``/``rot_y`` are not interchangeable, and the measured pair differ.

    12.0 and 14.8 degrees are close enough that a swap leaves a plausible-looking
    world. Feeding a transform that omits a key must also default rather than
    raise, since the schema's rotations are independently optional.
    """

    from arc.training.scene_provider import apply_recorded_scene_transform

    apply_recorded_scene_transform(_Sample(), {"scale": 2.0, "rot_y_deg": 90.0})

    call = stub_transform_scene[0]
    assert call["transformation_scale"] == 2.0
    np.testing.assert_allclose(
        call["transformation_rotation"].numpy(),
        rotation_from_degrees(0.0, 90.0).numpy(),
        atol=0,
    )


# --------------------------------------------------------- prediction schema ---


def _bundle(time_count=3, track_count=4):
    generator = torch.Generator().manual_seed(5)
    return dict(
        predicted_positions=torch.randn(time_count, track_count, 3, generator=generator),
        ground_truth_positions=torch.randn(time_count, track_count, 3, generator=generator),
        occluded=torch.zeros(time_count, track_count, dtype=torch.bool),
        query_points=torch.cat(
            [
                torch.zeros(track_count, 1),
                torch.randn(track_count, 3, generator=generator),
            ],
            dim=1,
        ),
        visible_any_camera=torch.ones(time_count, track_count, dtype=torch.bool),
    )


def test_the_written_npz_carries_exactly_what_the_cluster_scorer_reads(tmp_path):
    """Asserted against a transcription of `score_official.py`, not our own list.

    Checking the writer's key list against the writer's key list is a test that
    cannot fail for the reason that matters. These are the reads as that file
    performs them, copied by hand:

        d = np.load(p)
        gt, pred = d["gt"].astype(np.float64), d["pred"].astype(np.float64)
        va = np.asarray(d["gt_vis_any"]).astype(bool)
        v0 = va[0]
        gt_f, pred_f, vis_f = gt[:, v0], pred[:, v0], va[:, v0]
    """

    arrays = build_prediction_arrays(**_bundle())
    path = write_scene_predictions(tmp_path / "pred" / "0001.npz", arrays)

    loaded = np.load(path)
    gt, pred = loaded["gt"].astype(np.float64), loaded["pred"].astype(np.float64)
    visible_any = np.asarray(loaded["gt_vis_any"]).astype(bool)
    visible_at_zero = visible_any[0]
    gt_f, pred_f, vis_f = gt[:, visible_at_zero], pred[:, visible_at_zero], visible_any[:, visible_at_zero]

    assert gt_f.shape == pred_f.shape
    assert vis_f.shape == gt_f.shape[:2]
    # score_joint additionally reads occ and query_points, and inverts occ.
    assert set(loaded.files) == set(PREDICTION_KEYS)
    assert loaded["occ"].dtype == bool
    assert (~loaded["occ"]).shape == visible_any.shape
    assert loaded["query_points"].shape == (gt.shape[1], 4)


def test_query_times_outside_the_covered_window_are_refused(tmp_path):
    """Original frame numbers here are the silent version of this bug.

    A window trained on times (0, 2, 4) must write 0, 1, 2. Writing 0, 2, 4
    produces a well-formed file whose every query indexes the wrong row of `gt`,
    and nothing downstream can tell.
    """

    bundle = _bundle(time_count=3, track_count=4)
    bundle["query_points"][:, 0] = torch.tensor([0.0, 2.0, 4.0, 0.0])

    with pytest.raises(ValueError, match="index the 3 covered timesteps"):
        build_prediction_arrays(**bundle)


def test_a_bundle_missing_a_key_is_refused_before_it_reaches_disk(tmp_path):
    """A file the scorer cannot read is worse than no file."""

    arrays = build_prediction_arrays(**_bundle())
    del arrays["occ"]

    with pytest.raises(ValueError, match="missing \\['occ'\\]"):
        write_scene_predictions(tmp_path / "0001.npz", arrays)


# ------------------------------------------------------------- axis fusion ---


def test_cameras_are_fused_per_timestep_by_confidence(dumped_scene):
    """The axis change, checked against a hand-computed weighted mean.

    This repo's `S` is camera-major over cameras x times; the scorers' `T` is
    timesteps with cameras fused. Getting that transpose wrong yields arrays of
    the right shape and plausible magnitude, so it is checked numerically against
    `score_joint.py`'s rule rather than by inspection.
    """

    from arc.training import build_anchor_correspondences, fit_scene_sim3

    scene = dumped_scene
    correspondences, _ = build_anchor_correspondences(scene)
    observations = scene.num_observations
    height, width = scene.views[0]["img"].shape[-2:]

    # Distinct per-slot displacement and confidence, so a fusion that ignored the
    # weights, or averaged the wrong axis, cannot coincidentally match.
    tracks = torch.zeros(1, 1, observations, height, width, 3)
    confidence = torch.ones(1, 1, observations, height, width)
    for slot in range(observations):
        tracks[0, 0, slot] = float(slot + 1)
        confidence[0, 0, slot] = float(slot + 1)
    raw = {
        "track_multi": tracks,
        "conf_track_multi": confidence,
        "track_query_idx": scene.track_query_observation_slots.clone(),
        "depth": torch.ones(1, observations, height, width),
        "pose_enc": torch.zeros(1, observations, 9),
    }

    from arc.training import DetachedSim3, gather_query_anchor_points

    alignment = DetachedSim3(torch.tensor(1.0), torch.eye(3), torch.zeros(3))
    anchors = torch.zeros(correspondences.count, 3)

    arrays = train_cli._prediction_arrays(raw, scene, correspondences, alignment, anchors)

    covered = sorted({int(t) for t in scene.slot_times.tolist()})
    assert arrays["pred"].shape == (len(covered), correspondences.count, 3)
    assert arrays["gt_vis_any"].shape == (len(covered), correspondences.count)

    # First covered time: slots carrying it, fused by confidence weight.
    slots = [i for i, t in enumerate(scene.slot_times.tolist()) if int(t) == covered[0]]
    weights = np.array([slot + 1 for slot in slots], dtype=np.float64)
    values = np.array([slot + 1 for slot in slots], dtype=np.float64)
    expected = float((values * weights).sum() / weights.sum())

    np.testing.assert_allclose(arrays["pred"][0, :, 0], expected, rtol=1e-5)


def test_the_covered_timestep_column_indexes_the_window_not_the_frames(dumped_scene):
    """The convention, checked on a real scene rather than asserted in a comment."""

    from arc.training import DetachedSim3, build_anchor_correspondences

    scene = dumped_scene
    correspondences, _ = build_anchor_correspondences(scene)
    observations = scene.num_observations
    height, width = scene.views[0]["img"].shape[-2:]
    raw = {
        "track_multi": torch.zeros(1, 1, observations, height, width, 3),
        "conf_track_multi": torch.ones(1, 1, observations, height, width),
        "track_query_idx": scene.track_query_observation_slots.clone(),
    }

    arrays = train_cli._prediction_arrays(
        raw,
        scene,
        correspondences,
        DetachedSim3(torch.tensor(1.0), torch.eye(3), torch.zeros(3)),
        torch.zeros(correspondences.count, 3),
    )

    covered = sorted({int(t) for t in scene.slot_times.tolist()})
    times = arrays["query_points"][:, 0]
    assert times.min() >= 0 and times.max() < len(covered)


# ------------------------------------------------------- the dataset kwargs ---


# A realistic copy of what `from_name` returns for an EVALUATION dataset, which is
# what a replay gets: it passes no `training_args`, so the whole override block at
# kubric_multiview_dataset.py:160-190 is skipped and these defaults survive
# verbatim. Deliberately not an opaque sentinel -- an opaque one asserts that the
# data_root override lands while leaving every other inherited default invisible,
# which is exactly how the 30-scene cap and the random view draw got through.
EVAL_DEFAULT_KWARGS = {
    "data_root": "/split/kubric-multiview/test",
    "seq_len": 24,
    "traj_per_sample": 512,
    "seed": 72,
    "sample_vis_1st_frame": False,
    "tune_per_scene": False,
    "max_videos": 30,
    "num_views": 4,
    "views_to_return": None,
    "ratio_dynamic": 0.5,
    "ratio_very_dynamic": 0.25,
}


def test_the_provider_overrides_every_eval_default_that_breaks_a_replay():
    """Three of these eight were shipped-and-fatal, so each is pinned by name.

    `from_name` returns *evaluation* defaults whenever it is handed no
    `training_args`, which is always, here. Inheriting them silently gave a
    30-scene pool and a random four-view draw per sample.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    kwargs = dict(EVAL_DEFAULT_KWARGS)
    kwargs.update(MVTrackerSceneProvider().dataset_overrides("/rows/own/dir"))

    # Verbatim, never a join: the row's data_root is already the directory scenes
    # sit directly under, so re-resolving it produced a path that existed nowhere.
    assert kwargs["data_root"] == "/rows/own/dir"
    # Not 30. The pool is the whole split, or ~99.4% of a 4956-scene manifest
    # misses -- silently, once scene-load failures are skipped rather than fatal.
    assert kwargs["max_videos"] is None
    # -1 with no views_to_return returns EVERY view. At 4 the loader draws four
    # views at random per sample, which breaks the replay's same-view-sets claim.
    assert kwargs["num_views"] == -1
    assert kwargs["views_to_return"] is None
    # Every eligible track: the scene's own pool is what gets supervised.
    assert kwargs["traj_per_sample"] is None
    # Replayed from the record, not redrawn.
    assert kwargs["enable_scene_transform_augs"] is False
    assert kwargs["enable_cropping_augs"] is False
    # The paired run's clip, not the loader's 1000-metre constructor default.
    # This one is invisible in EVAL_DEFAULT_KWARGS on purpose: `from_name` never
    # emits `max_depth` at all, so nothing above would have shown it missing.
    assert kwargs["max_depth"] == 24.0


def test_the_depth_clip_is_the_paired_runs_and_is_overridable():
    """The value mirrors another repo's config, so it must not be a literal here.

    `from_name` never emits `max_depth`, so this is not an inherited kwarg being
    corrected -- it is a constructor default (1000) that would otherwise apply
    because a replay is built without training args. Nothing raises when it is
    wrong; the run simply trains on geometry and labels clipped 40x further out
    than the run it is paired with.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    assert MVTrackerSceneProvider().dataset_overrides("/d")["max_depth"] == 24.0
    # Overridable, because the upstream config it mirrors can move.
    provider = MVTrackerSceneProvider(max_depth=12.5)
    assert provider.dataset_overrides("/d")["max_depth"] == 12.5


def test_the_dynamic_ratios_are_left_alone_deliberately():
    """These LOOK like they should be overridden, and must not be.

    They govern the eligible pool's composition, which after the track-route
    inversion is exactly what gets supervised -- so inheriting an eval value
    would be a systematic difference from the paired run. It is not one:
    `ratio_dynamic` is overridden only under `modes.pretrain_only`
    (kubric_multiview_dataset.py:182-184), and the curve run's training stream
    (cli/train.py:541) does not take that branch, so MVTracker trains at 0.5/0.25
    too. The 0.1/0.0 pair belongs to the pretrain stream, which no manifest row
    replays. Matching is the point; "fixing" these would create the divergence.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    overrides = MVTrackerSceneProvider().dataset_overrides("/rows/own/dir")

    assert "ratio_dynamic" not in overrides
    assert "ratio_very_dynamic" not in overrides


# --------------------------------------------------------- the pool guard ---


@pytest.fixture
def stub_loader(monkeypatch):
    """MVTracker's loader, reduced to the two things ``_dataset`` uses of it.

    Inserted through ``sys.modules`` like ``stub_transform_scene``, which is what
    the provider's deferred import exists to allow. Records each construction so
    a test can count pool scans, the expensive part of a real build.
    """

    import sys
    import types

    state = {"builds": []}

    class _Dataset:
        def __init__(self, **kwargs):
            state["builds"].append(kwargs["data_root"])
            state["kwargs"] = kwargs
            # Two scenes, so any real_len a test asserts on disagrees loudly.
            self.seq_names = ["0000", "0001"]

        @staticmethod
        def from_name(name, dataset_root, just_return_kwargs=False):
            return {"data_root": os.path.join(dataset_root, "kubric-multiview", "test")}

    module = types.ModuleType("mvtracker.datasets.kubric_multiview_dataset")
    module.KubricMultiViewDataset = _Dataset
    datasets = types.ModuleType("mvtracker.datasets")
    datasets.kubric_multiview_dataset = module
    root = types.ModuleType("mvtracker")
    root.datasets = datasets
    monkeypatch.setitem(sys.modules, "mvtracker", root)
    monkeypatch.setitem(sys.modules, "mvtracker.datasets", datasets)
    monkeypatch.setitem(
        sys.modules, "mvtracker.datasets.kubric_multiview_dataset", module
    )
    return state


def test_the_rows_data_root_reaches_the_loader_unjoined(stub_loader):
    """The regression test for the defect that resolved every scene nowhere.

    `from_name` builds its `data_root` by joining `<root>/kubric-multiview/
    <subset>`. Handing it a row's already-resolved path as `dataset_root` made it
    join a second time, so a row naming `.../kubric-multiview/train` opened
    `.../kubric-multiview/train/kubric-multiview/test` -- a path that exists
    nowhere, failing at construction on the first step of every plan.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    MVTrackerSceneProvider()._dataset("/pool/kubric-multiview/train", None)

    used = stub_loader["kwargs"]["data_root"]
    assert used == "/pool/kubric-multiview/train"
    # The join would have appended a second one; nothing may re-resolve the path.
    assert used.count("kubric-multiview") == 1
    # And the value the provider decides reaches the loader, not just the dict.
    assert stub_loader["kwargs"]["max_depth"] == 24.0


def test_a_pool_of_the_wrong_size_is_refused_naming_both_counts():
    """The generalisation of the max_videos bug, rather than a fix for it.

    A cap was one way to open a quietly different pool; a moved split or a
    half-staged copy are others, and the next one will not look like a cap. Each
    loads fine and simply is not the pool the run sampled, after which every
    seq_name resolves against the wrong set. The manifest already records the
    wrap period, so this is a comparison, not an assumption.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    dataset = SimpleNamespace(seq_names=[f"{i:04d}" for i in range(30)])

    with pytest.raises(SceneProviderError) as excinfo:
        MVTrackerSceneProvider()._check_pool_size(dataset, "/pool", 4956)

    # Both counts, because "wrong pool" is not actionable and "30 against 4956"
    # names the cap on sight.
    message = str(excinfo.value)
    assert "30" in message and "4956" in message and "/pool" in message


def test_a_matching_pool_passes_and_an_unrecorded_one_is_not_invented():
    """None is a real answer, not a missing one.

    Held-out plans come from a JSON list of names rather than a manifest row, so
    they carry no drawn-from count. Guessing one -- say, asserting the held-out
    pool matches the training pool -- would fail every correct run.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    provider = MVTrackerSceneProvider()
    dataset = SimpleNamespace(seq_names=["0000", "0001", "0002"])

    provider._check_pool_size(dataset, "/pool", 3)
    provider._check_pool_size(dataset, "/pool", None)


def test_the_pool_guard_reaches_a_cached_dataset_too():
    """A second data_root's plan must not skip the check by hitting the cache.

    The dataset is built once per data_root and reused, so a guard that only ran
    on construction would check the first plan and wave through every later one
    -- including the row whose real_len actually disagrees.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    provider = MVTrackerSceneProvider()
    dataset = SimpleNamespace(seq_names=["0000", "0001"])
    provider._datasets["/pool"] = dataset

    assert provider._dataset("/pool", 2) is dataset
    with pytest.raises(SceneProviderError, match="real_len=99"):
        provider._dataset("/pool", 99)


def test_a_mismatched_pool_is_built_once_however_many_steps_retry(stub_loader):
    """The trainer skips a failed load and tries again, so this must be cheap.

    Construction scans the whole pool -- thousands of scenes, per build. A guard
    that ran before the object was cached would discard it, so every retry before
    the consecutive-skip limit aborts would rescan from scratch. The verdict is
    identical each time, so the object is kept and the retries re-raise off the
    cache. This drives the real build path, not a pre-seeded cache, because the
    ordering it pins is inside that path.
    """

    from arc.training.scene_provider import MVTrackerSceneProvider

    provider = MVTrackerSceneProvider()

    for _ in range(3):
        with pytest.raises(SceneProviderError, match="real_len=4956"):
            provider._dataset("/pool", 4956)

    assert stub_loader["builds"] == ["/pool"], "the pool was rescanned on a retry"
    assert "/pool" in provider._datasets, "a usable loader was thrown away"


@pytest.mark.skipif(
    importlib.util.find_spec("mvtracker") is None,
    reason="upstream not importable; the fixture is pinned by inspection instead",
)
def test_the_loader_default_this_repo_overrides_is_still_1000():
    """The drift alarm for max_depth, which no from_name kwarg would reveal.

    EVAL_DEFAULT_KWARGS cannot carry this: `from_name` does not emit `max_depth`,
    so the value the replay would inherit lives in the constructor signature. If
    upstream ever changes that default, the override stops being a correction and
    this says so on the cluster.
    """

    import inspect

    from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset

    default = inspect.signature(KubricMultiViewDataset.__init__).parameters["max_depth"].default
    assert default == 1000, f"upstream's max_depth default moved to {default!r}"


@pytest.mark.skipif(
    importlib.util.find_spec("mvtracker") is None,
    reason="upstream not importable; the fixture is pinned by inspection instead",
)
def test_the_eval_default_fixture_still_matches_upstream():
    """Keeps the fixture honest, the way the vendored-module hash tests do.

    A fixture that claims to mirror upstream's defaults is only useful while it
    does. This runs on the cluster, where mvtracker is importable, and is skipped
    here -- so it is a drift alarm rather than a local gate.
    """

    from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset

    real = KubricMultiViewDataset.from_name(
        "kubric-multiview-v3", dataset_root="", just_return_kwargs=True
    )
    for name, value in EVAL_DEFAULT_KWARGS.items():
        if name == "data_root":
            continue
        assert name in real, f"{name} vanished from from_name's kwargs"
        assert real[name] == value, f"{name}: fixture {value!r}, upstream {real[name]!r}"


# ------------------------------------------------------------ query anchors ---


def _anchor_plan(**record_overrides):
    """A real StepPlan whose window the anchor slots index into.

    ``views=[1, 0]`` on purpose: slot 0 then names camera id 1, which is what
    separates a *relative* slot from an absolute camera id. ``seq_len=4`` at
    stride 2 seats two times, matching the four-frame fixture scene.
    ``scene_transform=None`` keeps ``__call__`` off the transform path, which
    imports mvtracker; ``real_len=1`` matches the one-scene stub pool below.
    """

    from test_manifest_plan import _record
    from arc.training.manifest_plan import plan_record

    record = _record(
        seq_name="0000",
        views=[1, 0],
        seq_len=4,
        real_len=1,
        scene_transform=None,
        **record_overrides,
    )
    return plan_record(record, budget=48, stride=2)


def test_the_default_provider_spec_resolves_to_build_scenes_own_default():
    """(0, 0) resolves to (cameras[0], times[0]) -- exactly the anchor
    build_scene chooses when query_anchors is omitted, so the default changes
    nothing about existing runs."""

    from arc.training.scene_provider import MVTrackerSceneProvider

    provider = MVTrackerSceneProvider()
    plan = _anchor_plan()

    assert provider.query_anchor_slots == ((0, 0),)
    assert provider.resolve_query_anchors(plan) == (
        (plan.cameras[0], plan.times[0]),
    )


def test_the_provider_resolves_relative_slots_against_the_steps_own_window(tmp_path):
    """Slot 0:0 on a (1, 0) view list anchors camera id 1 -- relative, not
    absolute -- and the resolved pair reaches the scene through
    scene_from_datapoint end to end."""

    from test_scene_sources import _datapoint_from_dump
    from arc.training.scene_provider import MVTrackerSceneProvider

    scene_path = _write_scene(
        tmp_path, time_count=4, view_count=2, depth_sidecar=True
    )
    plan = _anchor_plan()
    assert plan.cameras == (1, 0) and len(plan.times) == 2

    class _Pool:
        seq_names = ["0000"]

        def __getitem__(self, index):
            datapoint = _datapoint_from_dump(scene_path)
            # The provider's column selection reads the loaded sample's own
            # eligible pool; the dump fixture carries three tracks.
            datapoint.sample_track_indices = [0, 1, 2]
            return datapoint, True

    resolved_scenes = []
    for slots, expected in (
        (((0, 0),), ((1, plan.times[0]),)),
        (((1, 1),), ((0, plan.times[1]),)),
        (((0, 0), (1, 0)), ((1, plan.times[0]), (0, plan.times[0]))),
    ):
        provider = MVTrackerSceneProvider(
            min_shared_queries=1, query_anchor_slots=slots
        )
        provider._datasets[plan.data_root] = _Pool()
        scene = provider(plan)
        assert scene.query_anchors == expected, slots
        resolved_scenes.append(scene)

    # The first anchor is primary: it owns the query observation slot.
    primary = resolved_scenes[0].observations[
        resolved_scenes[0].query_observation_slot
    ]
    assert primary.camera_id == 1
    assert primary.original_time == plan.times[0]


def test_an_anchor_slot_outside_the_plans_window_is_a_config_error_not_a_skip():
    """The step loop's skip policy absorbs SceneProviderError as one bad scene;
    a mis-sized anchor spec must kill the run instead, so it raises ValueError.
    (The trainer refuses such a spec at plan time; this is the backstop.)"""

    from arc.training.scene_provider import MVTrackerSceneProvider

    plan = _anchor_plan()
    provider = MVTrackerSceneProvider(query_anchor_slots=((0, 5),))

    with pytest.raises(ValueError, match="0:5") as excinfo:
        provider.resolve_query_anchors(plan)
    assert not isinstance(excinfo.value, SceneProviderError)

    view_provider = MVTrackerSceneProvider(query_anchor_slots=((2, 0),))
    with pytest.raises(ValueError, match="2:0"):
        view_provider.resolve_query_anchors(plan)

    # A negative slot is refused at construction: resolve_query_anchors checks
    # only the upper bound, so -1 would otherwise wrap via Python indexing to
    # the LAST camera and silently anchor a different observation.
    with pytest.raises(ValueError, match="never wrap"):
        MVTrackerSceneProvider(query_anchor_slots=((-1, 0),))
