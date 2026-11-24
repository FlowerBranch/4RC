"""Turn one manifest record into a scene, by replaying MVTracker's own loader.

This is the seam the planner and the training loop were both written around, and
the last piece of the trainer.  Its one design commitment is worth stating
plainly: **the sample assembly comes from MVTracker, not from a reimplementation
here.**  The paired run's entire value is that both models consume identical
samples, and ~200 lines of camera conventions, visibility and depth scaling
re-derived in this repository would be a divergence neither curve could reveal.
So the raw layout is read by the vendored ``scene_archive``, the assembly is
MVTracker's, the world transform is MVTracker's ``transform_scene``, and
:func:`arc.training.scene_from_datapoint` -- which has its own tests -- does the
final mapping.

**The augmentations are loaded off, then replayed on.**  A record's
``scene_transform`` is an *observation* of what the loader drew that step, so
loading with the augmentation enabled would apply a fresh random transform rather
than the recorded one.  The provider therefore disables it, then applies the
record's own ``{scale, rot_x_deg, rot_y_deg}`` through upstream's helper.

**The crop is not replayable, and is not replayed.**  The manifest deliberately
does not record it -- the cropping augmentation draws from the global
``np.random`` stream -- so there is no offset to reproduce.  The provider loads
uncropped, which is deterministic and honest; the resulting frames are 512x512
where MVTracker's run saw 384x512 crops of them.  That is consistent with the
project's ruling that bit-identical *inputs* are not the goal: what the replay
guarantees is the same scenes, windows and view sets -- not the same tracks
and not the same query points (see the class docstring) -- and the comparison that
carries the result is the held-out curve.  It does mean a different aspect ratio
reaches ``compute_image_transform``, which is tested at both shapes.

**The photometric and depth augmentations are not replayed either.**  The paired
run sets ``augmentations.probability 0.8`` with ``rgb`` and ``depth`` true; the
replay inherits ``0.0``/false/false, so upstream's ``augment_this_datapoint`` is
never true and no sample is jittered.  That is a real difference and it is left
in place deliberately.  Without --depth_input, 4RC's view dicts carry no depth,
so depth on this path feeds the Sim(3) fit and the anchor lifting -- it builds
*labels*, and perturbing it would inject noise into the supervision targets
rather than robustness into the inputs.  With --depth_input (and likewise
--camera_input for the pose) the same per-frame depth ALSO becomes a model
input, so the Sim(3) fit and the drift diagnostic stop being input-independent:
the map the model reads and the map the targets are lifted from are one array.
The crop settles the rest by precedent: it is gated by
``enable_cropping_augs`` alone rather than by the probability, so it perturbs
every MVTracker sample more than rgb jitter would, and is already accepted as
unreplayable.  What the pairing controls is which samples arrive in which order,
never the training recipe.

**Dump-format clips are the exception, because no MVTracker loader reads them.**
They are already assembled by 4I4's writers, and two providers read them through
one body, :func:`_scene_from_clip`.  ``--extra_pool`` clips are 4I4's Syn4D
conversion: :class:`DumpClipSceneProvider` reads their arrays and frames, and
shares everything after that with the Kubric path -- MVTracker's
``transform_scene`` through :func:`apply_recorded_scene_transform`, the anchor
slots, the track-pool guard, the label-depth cap and ``build_scene``.
:class:`SceneProviderByRoot` sends each plan to the provider of its pool.
``--eval_benchmark_clips`` is a held-out benchmark 4I4 wrote as a clip set:
:class:`BenchmarkClipSceneProvider` reads it with the same label-depth cap and
``build_scene``, the anchor slots resolved strictly, and neither the track-pool
guard nor a world transform, so that path imports nothing from MVTracker.

Importing MVTracker is deferred to the first load: this module is imported by the
trainer's tests, which run in an environment where ``mvtracker`` is deliberately
absent.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch

from arc.croco.utils.misc import is_main_process


class SceneProviderError(RuntimeError):
    """A record could not be turned into a scene."""


def rotation_from_degrees(rot_x_deg: float, rot_y_deg: float) -> torch.Tensor:
    """The manifest's rotation, applied x then y, as the schema specifies.

    Order matters and is not symmetric: composing y-then-x gives a different
    world, and every label in the sample would move with it.
    """

    x = math.radians(float(rot_x_deg))
    y = math.radians(float(rot_y_deg))
    rotation_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, math.cos(x), -math.sin(x)], [0.0, math.sin(x), math.cos(x)]],
        dtype=torch.float32,
    )
    rotation_y = torch.tensor(
        [[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]],
        dtype=torch.float32,
    )
    return rotation_y @ rotation_x


@dataclass(frozen=True)
class TrackSelection:
    """Where each requested track sits along the loaded sample's track axis."""

    positions: tuple[int, ...]
    requested: int
    missing: tuple[int, ...]

    @property
    def found(self) -> int:
        return len(self.positions)


def select_tracks_positionally(sample_track_indices, requested) -> TrackSelection:
    """Map a record's scene-level track ids onto columns of the loaded sample.

    **Positional, never set-based.**  ``track_indices`` may repeat by design --
    the loader draws from a "dynamic" and a "very dynamic" pool, the second a
    subset of the first -- and a repeat is a second, independent column with its
    own query timestep.  A ``set`` or a dict keyed by id silently drops it, so a
    repeated id here yields two positions, both pointing at the same column.

    Ids the sample does not carry are reported rather than skipped: whether that
    is tolerable is the caller's policy, and it is the question the cluster's
    two-draw probe exists to settle.
    """

    lookup: dict[int, int] = {}
    for position, value in enumerate(sample_track_indices):
        # First occurrence wins, so a pool that itself repeats an id still maps
        # deterministically.
        lookup.setdefault(int(value), position)

    positions = []
    missing = []
    for value in requested:
        position = lookup.get(int(value))
        if position is None:
            missing.append(int(value))
        else:
            positions.append(position)
    return TrackSelection(
        positions=tuple(positions),
        requested=len(list(requested)),
        missing=tuple(missing),
    )


def apply_recorded_scene_transform(sample, scene_transform: dict | None):
    """Apply the record's world similarity to the loaded sample, in place.

    Skipping this trains on different geometry, silently: the transform moves
    ``videodepth``, ``extrs``, ``trajectory_3d`` and ``query_points_3d`` together,
    so a sample without it is internally consistent and simply *elsewhere*.
    """

    if not scene_transform:
        return sample

    from mvtracker.datasets.utils import transform_scene

    scale = float(scene_transform.get("scale", 1.0))
    rotation = rotation_from_degrees(
        scene_transform.get("rot_x_deg", 0.0),
        scene_transform.get("rot_y_deg", 0.0),
    )
    depth, extrs, query_points, traj3d = transform_scene(
        transformation_scale=scale,
        transformation_rotation=rotation,
        depth=sample.videodepth,
        extrs=sample.extrs,
        query_points=sample.query_points_3d,
        traj3d_world=sample.trajectory_3d,
    )[:4]
    sample.videodepth = depth
    sample.extrs = extrs
    sample.query_points_3d = query_points
    sample.trajectory_3d = traj3d
    # The fifth field the transform moves, and the one that does not look like
    # geometry. Every consumer relies on `X_metres = X_stored * factor`, and the
    # transform multiplied every world quantity by `scale`, so the factor must
    # absorb it. Upstream does the same thing from the other side, emitting
    # `1 / scale` when it applies the augmentation itself.
    #
    # Leaving it stale is silent and it is not merely a misreported number: the
    # factor scales residuals before a Huber knee fixed at a physical 0.05 m, and
    # the anchor surface test at a physical 10 cm, so a scene drawn at scale 1.5
    # trains against a knee a third too tight. A divide, not an assignment,
    # because the invariant is the composition -- and it is a no-op at scale 1.0.
    sample.track_upscaling_factor = float(sample.track_upscaling_factor) / scale
    return sample


def checked_anchor_slots(query_anchor_slots) -> tuple[tuple[int, int], ...]:
    """Relative ``(view_slot, time_slot)`` pairs as ints, negatives refused.

    :func:`resolve_anchor_slots` checks only the upper bound, so a -1 would
    otherwise wrap via Python indexing to the LAST camera and anchor a
    different observation than any documented slot meaning, silently.
    """

    slots = []
    for view_slot, time_slot in query_anchor_slots:
        view_slot, time_slot = int(view_slot), int(time_slot)
        if view_slot < 0 or time_slot < 0:
            raise ValueError(
                f"query anchor slot {view_slot}:{time_slot} is negative; "
                "slots index the step's own window and never wrap"
            )
        slots.append((view_slot, time_slot))
    return tuple(slots)


def require_track_pool(seq_name: str, count: int, minimum: int) -> None:
    """Refuse a scene whose own eligible pool is too small to be worth a step.

    The message keeps "eligible tracks", which is how the trainer's
    ``scene_skip_cause`` files it under ``pool_too_small``.
    """

    if count < minimum:
        raise SceneProviderError(
            f"scene {seq_name!r}: only {count} eligible tracks, below "
            f"--min_shared_queries {minimum}. This is a scene "
            "whose own pool is too small to be worth a step, not a failure to "
            "match a recorded draw"
        )


def resolve_anchor_slots(slots, plan, *, adaptive: bool) -> tuple[tuple[int, int], ...]:
    """A plan's absolute ``(camera id, original time)`` anchors, from relative slots.

    ``(plan.cameras[view_slot], plan.times[time_slot])`` per slot, in slot
    order, so the first is primary.  A slot the plan cannot seat is dropped
    under ``adaptive`` and raises ``ValueError`` otherwise -- see
    :meth:`MVTrackerSceneProvider.resolve_query_anchors` for what each mode
    means and why the error is not a ``SceneProviderError``.  One arithmetic
    for the training stream, the held-out eval's extra anchors and the
    benchmark's scenes.
    """

    anchors = []
    for view_slot, time_slot in slots:
        if view_slot >= len(plan.cameras) or time_slot >= len(plan.times):
            if adaptive:
                continue
            raise ValueError(
                f"query anchor slot {view_slot}:{time_slot} does not fit "
                f"scene {plan.seq_name!r} with {len(plan.cameras)} views x "
                f"{len(plan.times)} times; the anchor spec must fit every "
                "planned step"
            )
        anchors.append((plan.cameras[view_slot], plan.times[time_slot]))
    return tuple(anchors)


class MVTrackerSceneProvider:
    """``StepPlan -> DumpedKubricScene``, by replaying a manifest record.

    **Tracks are the scene's own eligible set, not the record's.**  Measured
    rather than chosen, and measured twice on one scene at
    ``traj_per_sample=2048``: two draws shared 739 then 665 of their ~1800-1850
    unique ids, jaccard 0.256 then 0.219.  The ``traj_per_sample=None`` pool is
    itself redrawn -- 5789 entries then 5486, of 18000 scene rows -- so it is not
    a stable superset a recorded draw could be matched into either; it holds
    71-81% of any draw's ids.  Both follow from the pool being rebuilt from
    post-crop visibility on every load, with the crop drawn from the global
    ``np.random`` stream and never recorded, so **no re-load can reproduce a
    recorded draw.**  Honouring ``track_indices`` is therefore opt-in
    (``honour_recorded_tracks``), and even then a missing id is counted, never
    fatal -- a raise would fire on essentially every real record.

    What the replay guarantees: the same scenes, the same windows, the same view
    sets, in the same order at the same steps.

    **Query points are derived here, not replayed** -- a query time is the first
    frame a track is visible, computed after that same crop -- but "derived" is
    not "unrelated".  Of the 665 trajectories in both draws, 52.5% kept their
    query time (an independent redraw over 24 frames would agree ~4%), so about
    a fifth of a step's query points coincide exactly between two runs.  The
    honest claim is neither "identical" nor "each run's own".

    One dataset object is built per ``data_root`` and reused, because
    construction scans the scene pool; the per-sample cost the cluster measured
    (0.41 s at 8 workers, against a 7-14 s step) is the load, not the build.
    """

    def __init__(
        self,
        *,
        dataset_name: str = "kubric-multiview-v3",
        size: int = 512,
        patch_size: int = 14,
        min_shared_queries: int = 64,
        honour_recorded_tracks: bool = False,
        max_depth: float = 24.0,
        query_anchor_slots: Sequence[tuple[int, int]] = ((0, 0),),
        adaptive_query_anchors: bool = False,
        input_depth: bool = False,
        input_camera_vectors: bool = False,
    ):
        # There is deliberately no ``subset``: it was only ever a path component,
        # and after the override above it decides nothing. A held-out set is a
        # different ``data_root``, not a different subset of the same one.
        self.dataset_name = dataset_name
        self.size = size
        self.patch_size = patch_size
        # Mirrors the paired run's `datasets.train.kubric_max_depth`, so it is a
        # constructor parameter rather than a literal in the override dict:
        # the value belongs to someone else's config and has to stay visible and
        # overridable when that config moves. Under --depth_input the same
        # number is reused as BOTH the input channels' invalidity threshold
        # (matching the invalidation MVTracker's loader applies to this
        # provider's label depth, see the override-dict comment below) and the
        # normaliser scale recorded in run_summary.json: one constant applied
        # twice, with the input side pinned by a test.
        self.max_depth = max_depth
        self.input_depth = input_depth
        self.input_camera_vectors = input_camera_vectors
        # Now guards "this scene's own pool is too small to be worth a step",
        # not "too little of a recorded draw survived".
        self.min_shared_queries = min_shared_queries
        self.honour_recorded_tracks = honour_recorded_tracks
        # Relative (view_slot, time_slot) pairs, resolved per step against the
        # plan's own ordered cameras and times, in priority order -- the first
        # is primary and owns the Sim(3). The default resolves to
        # (plan.cameras[0], plan.times[0]), which is build_scene's own default
        # anchor, so it changes nothing. Negatives are refused at construction.
        self.query_anchor_slots = checked_anchor_slots(query_anchor_slots)
        # Whether the spec is a per-step ceiling or a per-run contract; see
        # resolve_query_anchors. Off is the contract every archived run trained
        # under.
        self.adaptive_query_anchors = bool(adaptive_query_anchors)
        # Reported rather than raised on: how much of the recorded draw the
        # opt-in path failed to find. Meaningless when the flag is off.
        self.missing_track_ids = 0
        self.requested_track_ids = 0
        # Keyed on data_root alone, which is now fully determining.
        self._datasets: dict[str, object] = {}

    def _dataset(self, data_root: str, real_len: int | None):
        """The loader for one ``data_root``, built once and reused.

        ``real_len`` is the pool size the manifest recorded for this row, or
        ``None`` for a plan that has no recorded row (the held-out set, which was
        never drawn from a manifest). It is checked rather than trusted: see
        :meth:`_check_pool_size`.
        """

        cached = self._datasets.get(data_root)
        if cached is not None:
            self._check_pool_size(cached, data_root, real_len)
            return cached
        try:
            from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise SceneProviderError(
                "the replay needs MVTracker's loader importable, so that both runs "
                "assemble samples with the same code. Run in RC_ENV with the "
                f"mvtracker checkout on PYTHONPATH ({error})"
            ) from error

        # ``from_name`` parses a dataset *name* into configuration kwargs, which is
        # what it is used for here. Its ``dataset_root``/``subset`` arguments are
        # deliberately given throwaway values: it would join them into a
        # ``data_root``, and that path is overridden below.
        kwargs = KubricMultiViewDataset.from_name(
            self.dataset_name,
            dataset_root="",
            just_return_kwargs=True,
        )
        kwargs.update(self.dataset_overrides(data_root))
        dataset = KubricMultiViewDataset(**kwargs)
        # Cached BEFORE the check, deliberately. The object loaded fine; what is
        # wrong is its pairing with this manifest, and that verdict is the same
        # every step. Checking first would discard it, so the trainer's skip
        # policy would rebuild -- rescanning the whole pool -- once per step until
        # the consecutive-skip limit aborts. Cached, the retries re-raise from the
        # cache hit above and cost nothing.
        self._datasets[data_root] = dataset
        self._check_pool_size(dataset, data_root, real_len)
        return dataset

    @staticmethod
    def _check_pool_size(dataset, data_root: str, real_len: int | None) -> None:
        """The pool this opened must be the pool the manifest drew from.

        The `max_videos` cap was one way to get a quietly different pool; it is
        not the only one, and the next will not look like it. A moved split, a
        half-staged copy, an upstream default that starts filtering scenes -- each
        gives a pool that loads fine and simply is not the one the recorded run
        sampled, and every ``sample_index`` and ``seq_name`` read against it then
        means something else. The manifest already records the wrap period, so
        this is a comparison rather than an assumption, and it costs one ``len``.
        """

        if real_len is None:
            return
        loaded = len(list(getattr(dataset, "seq_names", [])))
        if loaded != real_len:
            raise SceneProviderError(
                f"pool at {data_root!r} holds {loaded} scenes but the manifest "
                f"recorded real_len={real_len}. The replay would draw from a "
                "different pool than the run it is replaying"
            )

    def dataset_overrides(self, data_root: str) -> dict:
        """Everything `from_name`'s *evaluation* defaults get wrong for a replay.

        Split out so it can be tested without MVTracker importable. It is the
        whole of what this repo decides about how scenes are loaded, and three of
        these eight were shipped-and-fatal before they were caught, so the reasons
        are recorded per line rather than in a commit message.

        Not every inherited default is wrong. ``ratio_dynamic`` and
        ``ratio_very_dynamic`` are deliberately absent: they are overridden
        upstream only under ``modes.pretrain_only``, which the paired run never
        enters, so it trains at the same 0.5/0.25 and "fixing" them here would
        *create* the divergence. There is a test saying so.
        """

        return {
            # NEVER a join of it. ``data_root`` in a manifest row is already the
            # resolved directory scenes sit directly under, and upstream stamps it
            # from a path it has joined once already. Passing it as `dataset_root`
            # made `from_name` join it a second time, which is why held-out plans
            # resolved under `train` and the resulting path existed nowhere. The
            # layouts do not even agree (`kubric_multiview_003/train` for the
            # training stream against `kubric-multiview/<subset>` here), so no
            # reconstruction from a root can be right and none is attempted.
            "data_root": data_root,
            # Every eligible track, so the scene's own pool is what gets
            # supervised (V6(c): a recorded draw is not reproducible).
            "traj_per_sample": None,
            # The two augmentations whose draws we replay rather than redraw.
            "enable_scene_transform_augs": False,
            "enable_cropping_augs": False,
            # `from_name` defaults this to 30 for evaluation and only overrides it
            # when handed `training_args`, which a replay does not have. Inherited,
            # it truncates the pool to the split's FIRST THIRTY SCENES, so ~99.4%
            # of a 4956-scene manifest would miss -- and, with the scene-load skip
            # policy in the trainer, would do so silently.
            "max_videos": None,
            # `num_views=4` does not mean "expect four cameras": it makes the
            # loader draw FOUR VIEWS AT RANDOM per sample, from an unrecorded
            # stream. That silently breaks the one thing the replay still
            # guarantees outright -- the same view sets. -1 with no
            # `views_to_return` returns every view, and the provider then indexes
            # the record's own cameras out of the full set, which is what it was
            # written to do. Pinning `-views0123` instead would be deterministic
            # but wrong: rows name their own view sets, per row.
            "num_views": -1,
            "views_to_return": None,
            # The only one of these that `from_name` never mentions: it is the
            # CONSTRUCTOR default (kubric_multiview_dataset.py:212) of 1000, and
            # the override to the paired run's 24 sits at :160, inside the
            # `training_args` block a replay never enters. Depth beyond it is
            # zeroed -- i.e. marked invalid -- unconditionally at :882, so
            # inheriting 1000 trains on different geometry AND different labels
            # than the run this is paired with, on every sample, without raising.
            # Unlike the augmentations there is no draw here and nothing
            # unrecordable: it is a value nobody chose, so it is simply set.
            #
            # KNOWN GAP: nothing verifies 24 is still what the paired run uses.
            # Upstream's own 1000 default is pinned by no test either -- it was
            # last read at kubric_multiview_dataset.py:212 on MVTracker 4002795 --
            # and pinning it would catch the loader changing, not the run's
            # config moving. Do NOT close this
            # by asserting `configs/train.yaml`'s value -- that reads as coverage
            # while checking one of two sources, and the other is live:
            # train_curve.sbatch already overrides `datasets.train.traj_per_sample`
            # from the command line, so a `kubric_max_depth` override alongside it
            # would pass such a test unnoticed. The instrument that actually works
            # is the paired run's own RESOLVED config: hydra writes it to
            # `.hydra/config.yaml` under `experiment_path`, which is the manifest's
            # own directory, so it records what THAT manifest was drawn under,
            # overrides included, and needs no sibling checkout. It would answer
            # the same question for every other assumption this replay makes.
            # Confirmed present at `experiment_path`/.hydra/ (config.yaml,
            # hydra.yaml, overrides.yaml -- the last a direct record of what the
            # job passed). Unbuilt because it is a larger change than this round,
            # not because it is unverified.
            #
            # That config for run 19739167 reads `kubric_max_depth: 24`, so this
            # value is checked against the paired run's OWN resolved config and
            # not merely against a checkout of `train.yaml`. The same file also
            # confirms the `camera_params_noise: false` and
            # `variable_depth_type: false` that the rest of this override set
            # depends on -- which is exactly the leverage that makes building the
            # check worth someone's afternoon.
            "max_depth": self.max_depth,
        }

    def select_columns(self, plan, pool) -> TrackSelection:
        """Which columns of the loaded sample this step supervises.

        Split out of :meth:`__call__` so the route can be tested without
        MVTracker importable -- the branch it picks is the one V6(c) inverted,
        and it would otherwise be reachable only behind a full ``Datapoint``.
        """

        # The guard is on the pool, on both routes. Applying it to the matched
        # count instead would re-impose what the inversion removed -- a small
        # intersection is the *expected* case under V6(c), not a broken scene.
        require_track_pool(plan.seq_name, len(pool), self.min_shared_queries)

        if not (self.honour_recorded_tracks and plan.track_indices):
            return TrackSelection(
                positions=tuple(range(len(pool))), requested=len(pool), missing=()
            )

        # Opt-in, and it can never be complete: see the class docstring.
        # A missing id is counted, never fatal.
        selection = select_tracks_positionally(pool, plan.track_indices)
        self.missing_track_ids += len(selection.missing)
        self.requested_track_ids += selection.requested
        if not selection.positions:
            # Distinct from the guard above: the pool is fine and the record
            # simply shares nothing with it, which leaves no columns to gather
            # and would otherwise index the sample with an empty tensor.
            raise SceneProviderError(
                f"scene {plan.seq_name!r}: none of the {selection.requested} "
                f"recorded track ids are in its {len(pool)}-entry eligible pool, "
                "so --honour_recorded_tracks leaves nothing to supervise"
            )
        return selection

    def resolve_query_anchors(self, plan) -> tuple[tuple[int, int], ...]:
        """The step's absolute (camera id, original time) anchors, from slots.

        Slots are relative to the step's own ordered view list and time window,
        because every replayed step has its own, and the two modes differ only
        in what an unseatable slot means.

        **Strict** (the default): the spec is a per-run contract, and the
        trainer validates that every planned step seats every slot before
        training starts.  A slot that does not fit here is therefore a
        configuration error, and it raises ``ValueError`` rather than
        ``SceneProviderError`` deliberately: the step loop's skip policy
        absorbs ``SceneProviderError`` as one bad scene, and a mis-sized anchor
        spec must kill the run instead.

        **Adaptive** (``--adaptive_query_anchors``): the spec is a per-step
        ceiling, and a slot this step cannot seat is dropped rather than
        vetoing it.  The manifest's steps carry 1-6 views, so under the
        contract a single 2-view step refuses camera slot 3 for the whole
        stream; the alternative, ``--min_views 4``, deletes a quarter of the
        replayable rows instead.  Survivors keep spec order, so the first one
        is primary and owns the Sim(3).  The trainer's plan-time check
        guarantees at least one slot seats on *every* planned step, so this
        never returns empty.
        """

        return resolve_anchor_slots(
            self.query_anchor_slots, plan, adaptive=self.adaptive_query_anchors
        )

    def __call__(self, plan):
        from arc.training import scene_from_datapoint

        dataset = self._dataset(plan.data_root, plan.real_len)
        names = list(getattr(dataset, "seq_names", []))
        if plan.seq_name not in names:
            raise SceneProviderError(
                f"scene {plan.seq_name!r} is not in the pool at {plan.data_root!r} "
                f"({len(names)} scenes). A manifest row naming a scene the pool "
                "does not hold cannot be replayed against it"
            )
        sample, gotit = dataset[names.index(plan.seq_name)]
        if not gotit:
            raise SceneProviderError(f"scene {plan.seq_name!r} failed to load")

        apply_recorded_scene_transform(sample, plan.scene_transform)

        selection = self.select_columns(plan, sample.sample_track_indices or [])
        columns = torch.tensor(selection.positions, dtype=torch.long)
        sample.trajectory_3d = sample.trajectory_3d[:, columns]
        sample.visibility = sample.visibility[:, :, columns]
        sample.query_points_3d = sample.query_points_3d[columns]
        if getattr(sample, "valid", None) is not None:
            sample.valid = sample.valid[:, columns]

        return scene_from_datapoint(
            sample,
            cameras=plan.cameras,
            times=plan.times,
            query_anchors=self.resolve_query_anchors(plan),
            size=self.size,
            patch_size=self.patch_size,
            input_depth_max=self.max_depth if self.input_depth else None,
            input_camera_vectors=self.input_camera_vectors,
        )


def _scene_from_clip(
    path,
    plan,
    *,
    query_anchor_slots,
    adaptive_query_anchors: bool,
    max_depth: float,
    scene_transform,
    min_shared_queries: int | None,
    size: int,
    patch_size: int,
    input_depth_max: float | None,
    input_camera_vectors: bool,
    report_label_depth=None,
):
    """``plan``'s window of the dump-format clip at ``path``: both clip providers' body.

    ``read_clip`` checks the arrays and the frame set against the format
    contract.  The plan's cameras are positions on the clip's view axis; they
    become camera ids through its ``view_ids``, and the anchor slots are
    resolved on those.  Label depth beyond ``max_depth`` is zeroed (strict
    ``>``) before ``scene_transform`` moves the world and divides the factor,
    as MVTracker's loader zeroes Kubric's (kubric_multiview_dataset.py:882)
    before its own transform; under --depth_input ``input_depth_max`` is the
    same number.  ``build_scene`` decodes only the window's frames.

    ``min_shared_queries`` None skips the track-pool guard, and
    ``report_label_depth(name, frames, zeroed, valid)``, when given, hears how
    much label depth the cap zeroed out of the valid pixels, before the scene
    is built; without it nothing is counted.  A clip that breaks the format
    raises :class:`SceneProviderError` naming it, and what that costs is the
    caller's: one skipped step in the training loop, the run on the held-out
    path, where nothing absorbs it.  A slot the plan cannot seat stays a
    ``ValueError``.
    """

    from arc.training.dump_clip import ClipFormatError, open_clip
    from arc.training.dumped_kubric import build_scene, clip_frames, read_clip

    label = f"clip {plan.seq_name!r} at {path}"
    try:
        with open_clip(path) as clip:
            arrays = read_clip(clip, name=plan.seq_name)
            outside = [view for view in plan.cameras if view >= arrays.view_count]
            if outside:
                raise SceneProviderError(
                    f"{label} has {arrays.view_count} views, but the plan draws "
                    f"view position(s) {outside}: the pool index is stale"
                )
            late = [time for time in plan.times if time >= arrays.time_count]
            if late:
                raise SceneProviderError(
                    f"{label} has {arrays.time_count} frames, but the step's "
                    f"window reaches frame(s) {late}"
                )
            if min_shared_queries is not None:
                require_track_pool(plan.seq_name, arrays.track_count, min_shared_queries)
            camera_ids = tuple(int(arrays.view_ids[view]) for view in plan.cameras)
            anchors = resolve_anchor_slots(
                query_anchor_slots,
                replace(plan, cameras=camera_ids),
                adaptive=adaptive_query_anchors,
            )
            depth = arrays.depth
            beyond = depth > max_depth
            if report_label_depth is not None:
                report_label_depth(
                    plan.seq_name,
                    arrays.time_count,
                    int(beyond.sum()),
                    int((depth > 0).sum()),
                )
            depth[beyond] = 0.0
            # The Kubric path's own transform, on the four fields it moves and
            # the factor it divides; both are read back from the namespace.
            sample = SimpleNamespace(
                videodepth=torch.from_numpy(depth),
                extrs=torch.from_numpy(arrays.extrinsics),
                query_points_3d=torch.from_numpy(arrays.query_points),
                trajectory_3d=torch.from_numpy(arrays.trajectories),
                track_upscaling_factor=arrays.track_upscaling_factor,
            )
            apply_recorded_scene_transform(sample, scene_transform)
            archive = clip.frames()
            return build_scene(
                name=plan.seq_name,
                open_frames=lambda positions, times: clip_frames(
                    archive, positions, times, grid=arrays.grid, label=str(path)
                ),
                query_points=sample.query_points_3d,
                trajectories=sample.trajectory_3d,
                visibility=arrays.visibility,
                intrinsics=arrays.intrinsics,
                extrinsics=sample.extrs,
                depth0=sample.videodepth[:, 0],
                depth=sample.videodepth,
                view_ids=arrays.view_ids,
                track_upscaling_factor=float(sample.track_upscaling_factor),
                cameras=camera_ids,
                times=plan.times,
                query_anchors=anchors,
                size=size,
                patch_size=patch_size,
                input_depth_max=input_depth_max,
                input_camera_vectors=input_camera_vectors,
                source=str(path),
            )
    except ClipFormatError as error:
        raise SceneProviderError(f"{label} is malformed: {error}") from error


class DumpClipSceneProvider:
    """``StepPlan -> DumpedKubricScene`` for a pool of dump-format clips (``--extra_pool``).

    The clips are 4I4's Syn4D conversion: arrays and frames already at the
    trainer's working resolution, read through ``arc.training.dump_clip`` and
    checked by ``dumped_kubric.read_clip`` against the format contract at every
    load.  Every malformed clip -- a missing member or key, a shape or dtype
    that breaks the contract, a non-finite camera, a query off its trajectory,
    a frames.zip that does not hold exactly V x T frames, a frame that is not
    RGB at the depth grid -- raises :class:`SceneProviderError` naming the clip,
    so the step loop's skip policy absorbs it as one skipped scene.  A
    mis-sized anchor spec stays a ``ValueError`` and still ends the run, as on
    the Kubric path.

    What it shares with :class:`MVTrackerSceneProvider`, by the same code: the
    anchor slots and their resolution, the ``--min_shared_queries`` guard over
    the clip's own tracks (every track is supervised, as for an empty
    ``track_indices``), the world transform (``apply_recorded_scene_transform``,
    so ``track_upscaling_factor`` absorbs the scale), ``build_scene``, and the
    one depth cap: label depth beyond ``max_depth`` is zeroed before the
    transform, as MVTracker's loader zeroes Kubric's (kubric_multiview_dataset.py
    :882, strict ``>``), and under --depth_input the same number is the input's
    invalidity threshold.  The clip-to-scene body is :func:`_scene_from_clip`,
    which :class:`BenchmarkClipSceneProvider` calls too.

    **Views are positions.**  The sampler draws a clip's views from
    ``range(view_count)``, the index's count of its ``view_ids``, so
    ``plan.cameras`` are positions on the clip's view axis.  They become camera
    ids through ``view_ids`` before ``build_scene`` resolves them back, which is
    the identity for a clip whose ids are 0..V-1 and the right camera for one
    whose converter dropped a view (``[0, 1, 3, 5, 6, 7]``).

    Each pool is listed once, on its first plan, and the listing's size is
    checked against ``plan.real_len`` -- the index's count -- on every plan, as
    the Kubric provider checks its loader's.
    """

    def __init__(
        self,
        *,
        size: int = 512,
        patch_size: int = 14,
        min_shared_queries: int = 64,
        max_depth: float = 24.0,
        query_anchor_slots: Sequence[tuple[int, int]] = ((0, 0),),
        adaptive_query_anchors: bool = False,
        input_depth: bool = False,
        input_camera_vectors: bool = False,
    ):
        self.size = size
        self.patch_size = patch_size
        self.min_shared_queries = min_shared_queries
        self.max_depth = float(max_depth)
        self.query_anchor_slots = checked_anchor_slots(query_anchor_slots)
        self.adaptive_query_anchors = bool(adaptive_query_anchors)
        self.input_depth = input_depth
        self.input_camera_vectors = input_camera_vectors
        self._listings: dict[str, dict[str, str]] = {}

    def clip_path(self, plan) -> str:
        """Where ``plan``'s clip is, from the pool's listing, checked against ``real_len``."""

        from arc.training.dump_clip import CLIP_POOL_KIND
        from arc.training.scene_sampler import ScenePoolError, list_pool

        root = plan.data_root
        listing = self._listings.get(root)
        if listing is None:
            try:
                listing = list_pool(root, kind=CLIP_POOL_KIND)
            except ScenePoolError as error:
                raise SceneProviderError(f"the clip pool at {root!r} cannot be listed: {error}") from error
            self._listings[root] = listing
        if plan.real_len is not None and len(listing) != plan.real_len:
            raise SceneProviderError(
                f"the clip pool at {root!r} holds {len(listing)} clips but the plan "
                f"recorded real_len={plan.real_len}: it was drawn from an index of a "
                "different pool. Rebuild the index with --build_pool_index"
            )
        path = listing.get(plan.seq_name)
        if path is None:
            raise SceneProviderError(
                f"scene {plan.seq_name!r} is not in the pool at {root!r} "
                f"({len(listing)} clips)"
            )
        return path

    def __call__(self, plan):
        # Before any read: a slot this step cannot seat is a configuration error
        # in strict mode (see MVTrackerSceneProvider.resolve_query_anchors).
        resolve_anchor_slots(
            self.query_anchor_slots, plan, adaptive=self.adaptive_query_anchors
        )
        return _scene_from_clip(
            self.clip_path(plan),
            plan,
            query_anchor_slots=self.query_anchor_slots,
            adaptive_query_anchors=self.adaptive_query_anchors,
            max_depth=self.max_depth,
            scene_transform=plan.scene_transform,
            min_shared_queries=self.min_shared_queries,
            size=self.size,
            patch_size=self.patch_size,
            input_depth_max=self.max_depth if self.input_depth else None,
            input_camera_vectors=self.input_camera_vectors,
        )


class SceneProviderByRoot:
    """Route each plan to the provider of the pool it was drawn from.

    Keyed on ``plan.data_root``, which the sampler sets to the pool's own flag
    value.  A root no provider serves is a wiring bug, not a bad scene, so it
    raises ``RuntimeError`` -- never the ``SceneProviderError`` the step loop
    would absorb as a skip.
    """

    def __init__(self, providers: Mapping[str, object]):
        self.providers = dict(providers)

    def __call__(self, plan):
        provider = self.providers.get(plan.data_root)
        if provider is None:
            raise RuntimeError(
                f"no scene provider serves data_root {plan.data_root!r}; this run's "
                f"pools are {sorted(self.providers)}"
            )
        return provider(plan)


# --- Held-out benchmarks. The eval can score an MVTracker benchmark in place of
# the --val_scenes_file scenes, and such a set is opened by the benchmark's OWN
# protocol -- its family's from_name, exactly the call 4I4's
# triton/4rc/multiview/dump_common.py::build_dataset makes (Kubric's at subset
# "test") -- never by the replay's dataset_overrides above: the point is to
# score the queries MVTracker's figures are scored on. One entry per dataset
# family, matched by name prefix: Kubric and DexYCB.
#
# A benchmark 4I4 wrote as a clip set (--eval_benchmark_clips) is the other
# held-out benchmark: BenchmarkClipSceneProvider, at the end of this section,
# reads it with this repo's clip reader and imports nothing from MVTracker.
# Panoptic arrives only that way. The loader entries stay until
# benchmark_clip_parity.py has matched the two readings on the cluster.


def _kubric_benchmark_refusal(name: str) -> str | None:
    """Why a Kubric name cannot be a held-out benchmark, or None.

    Only a ``-cached`` name reads the shipped track set: without it the loader
    redraws every scene's tracks and queries (kubric_multiview_dataset.py:945,
    MVTracker 4002795), so the scored queries would not be the benchmark's.
    """

    if not name.endswith("-cached"):
        return (
            f"--eval_benchmark_dataset {name!r} reads no cached track set: only a "
            "name ending in -cached scores the benchmark's own queries (512 per "
            "scene, shipped with the test data); any other redraws them per scene"
        )
    return None


def _open_kubric_benchmark(name: str, root: str, *, max_depth: float):
    """MVTracker's Kubric test set, by its own protocol plus one label-depth cap.

    The kwargs are ``from_name``'s at subset ``"test"`` with exactly one key
    set on top, ``max_depth``: the constructor's evaluation default is 1000,
    while the held-out scenes load at the run's ``--kubric_max_depth``. It only
    zeroes depth maps (kubric_multiview_dataset.py:882), which feed the Sim(3)
    fit and the anchor depth gate, so without it the benchmark would align and
    admit on far background the held-out curve never sees. The scored tracks,
    visibility and queries come from the track cache, whose key ignores
    ``max_depth`` (kubric_multiview_dataset.py:515-527).

    Refuses a set with a scene whose track cache is missing: the loader would
    redraw that scene's queries and write a new cache file into the dataset
    root on first load (kubric_multiview_dataset.py:925-934).  The location
    rule is the loader's own (:925-931): an archived scene keeps its cache
    under ``<data_root>/_trackcache/<scene>/``, a directory scene under
    ``<scene>/cache/``.
    """

    try:
        from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
        from mvtracker.datasets.scene_archive import ARCHIVE_SUFFIX
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise SceneProviderError(
            "the benchmark needs MVTracker's loader importable, so that it is "
            "opened by the benchmark's own protocol. Run in RC_ENV with the "
            f"mvtracker checkout on PYTHONPATH ({error})"
        ) from error

    kwargs = KubricMultiViewDataset.from_name(
        name, dataset_root=root, subset="test", just_return_kwargs=True
    )
    kwargs["max_depth"] = max_depth
    dataset = KubricMultiViewDataset(**kwargs)
    missing = []
    for scene_name, scene_path in zip(dataset.seq_names, dataset.seq_paths):
        cache_root = (
            os.path.join(dataset.data_root, "_trackcache", scene_name)
            if scene_path.endswith(ARCHIVE_SUFFIX)
            else os.path.join(scene_path, "cache")
        )
        cache_file = os.path.join(cache_root, f"{dataset.cache_name}.npz")
        if not os.path.isfile(cache_file):
            missing.append(cache_file)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(dataset.seq_names)} scenes of {name!r} at "
            f"{kwargs['data_root']!r} have no cached track file (first: "
            f"{missing[0]}). MVTracker would redraw their queries and write the "
            "file into the dataset root on first load, so the scored queries "
            "would not be the benchmark's"
        )
    return dataset, kwargs


# DexYCB's from_name builds its dataset itself, and its loader caps no depth,
# crops no clip and records no camera ids, so the opener reads what the
# provider checks off the data and _FromNameTestSet completes every sample.
# Panoptic's loader has the same shape.

_DEXYCB_PREFIX = "dex-ycb-multiview"
_DEXYCB_BENCHMARK_NAME = re.compile(r"dex-ycb-multiview-views(\d+)-cached")
# from_name's tokens that change what the model sees or what is scored, in its
# parse order (dexycb_multiview_dataset.py:39-87, MVTracker 4002795).
_DEXYCB_REFUSED_TOKENS = (
    (
        r"-(?:duster|vggt)\d*(?:cleaned)?",
        "swaps the sensor depth for an estimate, and that depth is what the "
        "Sim(3) fit and the anchor depth gate read",
    ),
    (r"-novelviews\d*", "loads cameras beyond the returned ones as novel views"),
    (
        r"-removehand",
        "drops the hand's tracks: another scored set, under another track cache",
    ),
    (r"-single", "is MVTracker's one-query-at-a-time predictor mode"),
    (r"-2dpt", "scores 2-D point tracks, not 3-D"),
)


def _dexycb_benchmark_refusal(name: str) -> str | None:
    """Why a DexYCB name cannot be a held-out benchmark, or None.

    Exactly ``dex-ycb-multiview-views<digits>-cached`` is accepted: the sensor
    depth, the cameras named digit by digit and each once, every one of the
    set's tracks, and the shipped track cache. Every other token from_name
    parses changes what the model sees or what is scored, and is named.
    """

    match = _DEXYCB_BENCHMARK_NAME.fullmatch(name)
    if match is not None:
        digits = match.group(1)
        if len(set(digits)) != len(digits):
            return (
                f"--eval_benchmark_dataset {name!r} names a camera twice "
                f"(views{digits}); every returned view must be its own camera"
            )
        return None
    rest = name[len(_DEXYCB_PREFIX):]
    for pattern, reason in _DEXYCB_REFUSED_TOKENS:
        token = re.search(pattern, rest)
        if token is not None:
            return (
                f"--eval_benchmark_dataset {name!r}: {token.group(0)} {reason}; "
                "only dex-ycb-multiview-views<digits>-cached is scored"
            )
    if not rest.endswith("-cached"):
        return (
            f"--eval_benchmark_dataset {name!r} reads no cached track set: only a "
            "name ending in -cached scores the benchmark's own queries (384 per "
            "scene, cached beside each scene's frames); any other draws them "
            "anew and writes a cache file into the dataset root"
        )
    if "-views" not in rest:
        return (
            f"--eval_benchmark_dataset {name!r} names no cameras: write them as "
            "-views<digits> (from_name's default is views0123)"
        )
    return (
        f"--eval_benchmark_dataset {name!r} is not of the form "
        "dex-ycb-multiview-views<digits>-cached"
    )


def _scene_frame_count(name: str, data_root: str, scene_name: str, views) -> int:
    """One DexYCB scene's frame count, read as its loader reads the frames.

    The scene's ``view_*`` entries, sorted and indexed by view number, then
    every file of ``rgb/`` and of ``depth/`` (dexycb_multiview_dataset.py
    :220-239). The loader stacks the returned views (:254-256), so they must
    agree, in both folders; a disagreement or a missing folder would end the
    run at the first load, after the model's.
    """

    scene_dir = os.path.join(data_root, scene_name)
    folders = sorted(
        entry for entry in os.listdir(scene_dir) if entry.startswith("view_")
    )
    counts = {}
    for view in views:
        if view >= len(folders):
            raise ValueError(
                f"scene {scene_name!r} of {name!r} at {data_root!r} has "
                f"{len(folders)} view folders, but the set returns view {view}"
            )
        for kind in ("rgb", "depth"):
            folder = os.path.join(scene_dir, folders[view], kind)
            if not os.path.isdir(folder):
                raise ValueError(
                    f"scene {scene_name!r} of {name!r} has no {kind} folder at "
                    f"{folder!r}"
                )
            counts[f"{folders[view]}/{kind}"] = len(os.listdir(folder))
    if len(set(counts.values())) != 1:
        listed = ", ".join(f"{folder} {count}" for folder, count in counts.items())
        raise ValueError(
            f"scene {scene_name!r} of {name!r} at {data_root!r} holds different "
            f"frame counts across its returned views ({listed}); the loader "
            "stacks them, so it would fail at the first load"
        )
    return next(iter(counts.values()))


def _open_dexycb_benchmark(name: str, root: str, *, max_depth: float):
    """MVTracker's DexYCB test set, by its own protocol, completed per sample.

    ``from_name(name, dataset_root=root)`` builds the dataset itself -- it
    takes no subset and returns no kwargs (dexycb_multiview_dataset.py:22-105,
    MVTracker 4002795) -- so what the provider checks is read off the dataset
    and the data: ``data_root`` (``<root>/dex-ycb-multiview``),
    ``views_to_return``, and ``scene_frames``, each scene's frame count. There
    is no ``seq_len``: the loader loads every frame on disk (:233), and the
    set's scenes run 24 or 25 frames, so the provider holds each scene to the
    window instead.

    Refuses, before any load, what a load would trip over or write: a missing
    dataset directory (the constructor lists it, :151); a scene whose track
    cache is missing -- the loader would draw new queries and write the file
    into the dataset root (:385-387, :464-475); and a scene whose returned
    views disagree on their frames (:func:`_scene_frame_count`).
    """

    try:
        from mvtracker.datasets.dexycb_multiview_dataset import DexYCBMultiViewDataset
    except ImportError as error:
        raise SceneProviderError(
            "the benchmark needs MVTracker's DexYCB loader importable, so that it "
            "is opened by the benchmark's own protocol; the module imports pandas "
            "and matplotlib at module level. Run in RC_ENV with the mvtracker "
            f"checkout on PYTHONPATH ({error})"
        ) from error

    # The loader's own join (:93), checked before from_name lists it.
    data_root = os.path.join(root, _DEXYCB_PREFIX)
    if not os.path.isdir(data_root):
        raise ValueError(
            f"--eval_benchmark_dataset {name!r} has no dataset directory at "
            f"{data_root!r}: --eval_benchmark_root must be the root holding "
            f"{_DEXYCB_PREFIX}"
        )
    dataset = DexYCBMultiViewDataset.from_name(name, dataset_root=root)
    views = [int(view) for view in dataset.views_to_return]
    missing = [
        cache_file
        for cache_file in (
            os.path.join(
                dataset.data_root, scene_name, "cache", f"{dataset.cache_name}.npz"
            )
            for scene_name in dataset.seq_names
        )
        if not os.path.isfile(cache_file)
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(dataset.seq_names)} scenes of {name!r} at "
            f"{dataset.data_root!r} have no cached track file (first: "
            f"{missing[0]}). MVTracker would redraw their queries and write the "
            "file into the dataset root on first load, so the scored queries "
            "would not be the benchmark's"
        )
    scene_frames = {
        str(scene_name): _scene_frame_count(name, dataset.data_root, scene_name, views)
        for scene_name in dataset.seq_names
    }
    return (
        _FromNameTestSet(dataset, views=views, max_depth=max_depth),
        {
            "data_root": dataset.data_root,
            "views_to_return": views,
            "scene_frames": scene_frames,
        },
    )


class _LabelDepthReport:
    """How much label depth the cap zeroed in a benchmark scene, said once per scene.

    One line, out of the scene's valid (positive) pixels, on rank 0, the first
    time the scene loads in this process: the count cannot change between loads,
    and every rank loads every scene at the preflight.  Both benchmark readers
    print it -- a loader-built set's (:class:`_FromNameTestSet`) and a clip set's
    (:class:`BenchmarkClipSceneProvider`) -- so it is spelled here once.
    """

    def __init__(self, max_depth: float):
        self.max_depth = float(max_depth)
        self._reported: set[str] = set()

    def __call__(self, scene_name: str, frames: int, zeroed: int, valid: int) -> None:
        if scene_name in self._reported:
            return
        self._reported.add(scene_name)
        if is_main_process():
            share = 100.0 * zeroed / valid if valid else 0.0
            print(
                f"benchmark scene={scene_name} frames={frames} "
                f"label_depth_zeroed={zeroed}/{valid} ({share:.4f}%) "
                f"beyond {self.max_depth:g}"
            )


class _FromNameTestSet:
    """A benchmark whose ``from_name`` builds its loader, completed per sample.

    DexYCB's loader -- Panoptic's has the same shape -- leaves three things to
    its caller, and every loaded sample gets them here, in place:

    * the label-depth cap: depth beyond ``max_depth`` zeroed, strict ``>``, in
      the loader's own units after its scene transform -- metres x6 on DexYCB
      (dexycb_multiview_dataset.py:478-486), so the default 24 is 4 m -- as
      Kubric's loader zeroes its own (kubric_multiview_dataset.py:882) and the
      clip pool's provider does;
    * ``sample_views``: the loader records none, so ``scene_from_datapoint``
      would number the views 0..V-1, while the video axis is stacked in
      ``views_to_return`` order (:254): each position's camera id is that
      list's entry;
    * one line per scene saying how much label depth the cap zeroed, out of
      the valid (positive) pixels, over every frame the loader loaded
      (:class:`_LabelDepthReport`).
    """

    def __init__(self, dataset, *, views: Sequence[int], max_depth: float):
        self.dataset = dataset
        self.seq_names = dataset.seq_names
        self.views = [int(view) for view in views]
        self.max_depth = float(max_depth)
        self._report = _LabelDepthReport(self.max_depth)

    def __getitem__(self, index):
        sample, gotit = self.dataset[index]
        if not gotit:
            return sample, gotit
        depth = sample.videodepth
        beyond = depth > self.max_depth
        zeroed = int(beyond.sum())
        valid = int((depth > 0).sum())
        depth[beyond] = 0.0
        sample.sample_views = list(self.views)
        self._report(str(sample.seq_name), int(sample.video.shape[1]), zeroed, valid)
        return sample, gotit


# (name prefix, parse-time refusal, opener)
BENCHMARK_DATASETS = (
    ("kubric-multiview-v3", _kubric_benchmark_refusal, _open_kubric_benchmark),
    (_DEXYCB_PREFIX, _dexycb_benchmark_refusal, _open_dexycb_benchmark),
)


def benchmark_name_refusal(name: str) -> str | None:
    """Why ``name`` cannot be a held-out benchmark, or None; imports nothing.

    The parse-time half of the benchmark's checks: what a name alone decides,
    answerable before the model loads and without MVTracker importable.
    """

    for prefix, refusal, _ in BENCHMARK_DATASETS:
        if name.startswith(prefix):
            return refusal(name)
    prefixes = [prefix for prefix, _, _ in BENCHMARK_DATASETS]
    return (
        f"--eval_benchmark_dataset {name!r} is not a supported benchmark; "
        f"supported names start with one of {prefixes}"
    )


class BenchmarkSceneProvider:
    """``StepPlan -> DumpedKubricScene`` for a held-out benchmark's test scenes.

    Built once, before the model loads, so every refusal costs no model load:
    the name, the dataset's own cameras against ``--val_cameras`` (the
    window's camera set, a refused resume key), its clip length against
    ``--val_seq_len``, and -- through the opener -- a missing track cache.
    The clip rule follows the loader. Kubric's crops every clip to its own
    ``seq_len``, which must therefore be the window's. DexYCB's loads every
    frame on disk and its scenes differ in length, so its opener reports
    ``scene_frames`` instead, and every scene must hold the window: a
    shorter one is refused, a longer one is scored on its first frames.

    A scene is the benchmark's sample as loaded: no world transform and no
    column selection, so the scene's tracks and queries are exactly the
    dataset's, in its order.  The anchors are the training spec's seated
    held-out slots, resolved strictly (they were checked against the window at
    parse time).  ``--kubric_max_depth`` caps both depths, as on the held-out
    path: the labels through the opener, the encoder's input under
    ``--depth_input`` through ``input_depth_max``.  A scene that fails to load
    raises: a benchmark scored on a subset of its scenes would not be
    comparable with MVTracker's figures.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_root: str,
        cameras: Sequence[int],
        seq_len: int,
        max_depth: float,
        query_anchor_slots: Sequence[tuple[int, int]],
        size: int = 512,
        patch_size: int = 14,
        input_depth: bool = False,
        input_camera_vectors: bool = False,
    ):
        refusal = benchmark_name_refusal(dataset_name)
        if refusal is not None:
            raise ValueError(refusal)
        opener = next(
            open_benchmark
            for prefix, _, open_benchmark in BENCHMARK_DATASETS
            if dataset_name.startswith(prefix)
        )
        self.dataset, self.kwargs = opener(
            dataset_name, dataset_root, max_depth=max_depth
        )
        views = self.kwargs.get("views_to_return")
        if views is None:
            raise ValueError(
                f"--eval_benchmark_dataset {dataset_name!r} fixes no camera set "
                "(views_to_return is None): the loader would draw the views per "
                "scene, so the held-out window has no cameras to name"
            )
        views = tuple(int(view) for view in views)
        if views != tuple(int(camera) for camera in cameras):
            raise ValueError(
                f"--eval_benchmark_dataset {dataset_name!r} returns cameras "
                f"{list(views)}, but the held-out window is --val_cameras "
                f"{list(cameras)}; the window and its anchors are sized by those"
            )
        scene_frames = self.kwargs.get("scene_frames")
        if scene_frames is None:
            # A loader that crops every clip to its own seq_len (Kubric's).
            if self.kwargs.get("seq_len") != seq_len:
                raise ValueError(
                    f"--eval_benchmark_dataset {dataset_name!r} loads "
                    f"{self.kwargs.get('seq_len')}-frame clips, but the held-out window "
                    f"is planned on --val_seq_len {seq_len}"
                )
        else:
            # A loader that loads every frame on disk (DexYCB's): each scene
            # must seat the window; a longer clip's tail is never scored.
            short = [
                (scene_name, frames)
                for scene_name, frames in scene_frames.items()
                if frames < seq_len
            ]
            if short:
                raise ValueError(
                    f"--eval_benchmark_dataset {dataset_name!r} has {len(short)} of "
                    f"{len(scene_frames)} scenes shorter than the held-out window's "
                    f"--val_seq_len {seq_len} (first: {short[0][0]!r}, "
                    f"{short[0][1]} frames); its loader does not crop, and a window "
                    "past a clip's last frame cannot be seated"
                )
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.data_root = str(self.kwargs["data_root"])
        self.max_depth = float(max_depth)
        self.cameras = views
        self.scene_names = [str(name) for name in self.dataset.seq_names]
        if not self.scene_names:
            raise ValueError(
                f"--eval_benchmark_dataset {dataset_name!r} at {self.data_root!r} "
                "holds no scenes"
            )
        self.query_anchor_slots = tuple(
            (int(view_slot), int(time_slot))
            for view_slot, time_slot in query_anchor_slots
        )
        self.size = size
        self.patch_size = patch_size
        self.input_depth_max = self.max_depth if input_depth else None
        self.input_camera_vectors = input_camera_vectors

    def identity(self) -> dict:
        """What the eval records about the set it scored.

        One record for every family. ``subset`` is the set's role, "test"
        for both: Kubric's from_name takes it as an argument, DexYCB's takes
        none -- MVTracker evaluates on that set and nothing else.
        """

        return {
            "dataset": self.dataset_name,
            "root": str(self.dataset_root),
            "subset": "test",
            "data_root": self.data_root,
            "scene_count": len(self.scene_names),
            "max_depth": self.max_depth,
        }

    def __call__(self, plan):
        from arc.training import scene_from_datapoint

        if plan.seq_name not in self.scene_names:
            raise SceneProviderError(
                f"scene {plan.seq_name!r} is not in the benchmark at "
                f"{self.data_root!r} ({len(self.scene_names)} scenes)"
            )
        sample, gotit = self.dataset[self.scene_names.index(plan.seq_name)]
        if not gotit:
            raise SceneProviderError(
                f"benchmark scene {plan.seq_name!r} failed to load"
            )
        return scene_from_datapoint(
            sample,
            cameras=plan.cameras,
            times=plan.times,
            query_anchors=resolve_anchor_slots(
                self.query_anchor_slots, plan, adaptive=False
            ),
            size=self.size,
            patch_size=self.patch_size,
            input_depth_max=self.input_depth_max,
            input_camera_vectors=self.input_camera_vectors,
        )


# The file at a clip set's root recording the rule that wrote it: an object whose
# "setting" names evaluate_3dpt's setting. The eval records it and reads nothing
# in it; a clip's own source.json is not read at all.
BENCHMARK_PROTOCOL_NAME = "benchmark.json"


class BenchmarkClipSceneProvider:
    """``StepPlan -> DumpedKubricScene`` for a held-out benchmark written as a clip set.

    ``--eval_benchmark_clips DIR``: 4I4's ``prepare_clips.py`` writes a
    benchmark as a directory of dump-format clips -- the format
    :class:`DumpClipSceneProvider` reads for ``--extra_pool`` -- with
    ``benchmark.json`` at the root.  The clips are listed as an extra pool's
    are (``list_pool``: clip directories or ``<name>.zip``, ``.`` and ``_``
    entries skipped, plain string order), and every one is scored, in that
    order.  Nothing here imports MVTracker.

    Built once, before the model loads, so every refusal costs no model load: a
    DIR that is not a directory or holds no clip; a clip ``check_clip`` finds
    malformed or shorter than ``--val_seq_len`` (it reads meta.npz in full and
    depth_full.npz's header, and decodes no frame, so a bad depth value or frame
    is first refused by the preflight's load); a clip whose view count is not
    the window's; and a ``benchmark.json`` that is not a JSON object.

    **Views are positions.**  The window's cameras are ``range(V)``, ``V`` being
    ``len(--val_cameras)``, and each becomes a camera id through the clip's own
    ``view_ids`` (1 7 14 20 on Panoptic), so ``--val_cameras`` is compared by
    count, never by value.  A clip longer than ``--val_seq_len`` is scored on
    the window's frames.

    A scene is the clip as written, through :func:`_scene_from_clip`, the body
    the extra pool's provider calls too, with three differences: no
    ``--min_shared_queries`` guard (a benchmark scores every query it has), the
    anchors resolved strictly (they were checked against the window at parse
    time), and no world transform.  ``--kubric_max_depth`` caps both depths,
    as on the loader path, in the clip's own units -- traj3d_world's: Kubric
    units, metres x6 on DexYCB, metres x2.5 on Panoptic -- and a line per
    scene says how much the cap zeroed (:class:`_LabelDepthReport`).  A clip
    that fails to read raises :class:`SceneProviderError`, which the held-out
    preflight and eval do not absorb: a benchmark scored on a subset of its
    scenes would not be the benchmark.
    """

    def __init__(
        self,
        *,
        clips,
        view_count: int,
        seq_len: int,
        max_depth: float,
        query_anchor_slots: Sequence[tuple[int, int]],
        size: int = 512,
        patch_size: int = 14,
        input_depth: bool = False,
        input_camera_vectors: bool = False,
    ):
        from arc.training.dump_clip import CLIP_POOL_KIND, clip_view_ids
        from arc.training.dumped_kubric import check_clip
        from arc.training.scene_sampler import ScenePoolError, list_pool

        flag = "--eval_benchmark_clips"
        clips = os.fspath(clips)
        # Checked first so the refusal names this flag: list_pool's own names
        # the extra pool's.
        if not os.path.isdir(clips):
            raise ValueError(f"{flag} {clips!r} is not a directory")
        try:
            listing = list_pool(clips, kind=CLIP_POOL_KIND)
        except ScenePoolError as error:
            raise ValueError(f"{flag} {clips!r} cannot be listed: {error}") from error

        failing = {}
        for name, path in listing.items():
            problems = check_clip(path, name=name, min_frames=seq_len)
            if problems:
                failing[name] = problems
        if failing:
            named = "; ".join(
                problem for problems in list(failing.values())[:8] for problem in problems
            )
            raise ValueError(
                f"{flag} {clips!r}: {len(failing)} of {len(listing)} clip(s) fail the "
                f"clip format check at the held-out window's --val_seq_len {seq_len}: "
                f"{named}" + (" ..." if len(failing) > 8 else "")
            )
        miscounted = []
        for name, path in listing.items():
            count = int(clip_view_ids(path).shape[0])
            if count != view_count:
                miscounted.append(f"{name} ({count} views)")
        if miscounted:
            raise ValueError(
                f"{flag} {clips!r}: {len(miscounted)} of {len(listing)} clip(s) do "
                f"not hold the held-out window's {view_count} cameras "
                "(len(--val_cameras); its values are not compared, a clip names "
                "its own through view_ids): "
                + ", ".join(miscounted[:8])
                + (" ..." if len(miscounted) > 8 else "")
            )

        protocol = None
        protocol_path = os.path.join(clips, BENCHMARK_PROTOCOL_NAME)
        if os.path.exists(protocol_path):
            try:
                with open(protocol_path, encoding="utf-8") as handle:
                    protocol = json.load(handle)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"{flag} {clips!r}: {protocol_path} cannot be read as JSON: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if not isinstance(protocol, dict):
                raise ValueError(
                    f"{flag} {clips!r}: {protocol_path} holds a "
                    f"{type(protocol).__name__}, not the JSON object recording the "
                    "rule that wrote the set"
                )

        self.clips = clips
        self.protocol = protocol
        self.data_root = clips
        self.cameras = tuple(range(int(view_count)))
        self.scene_names = list(listing)
        self._paths = dict(listing)
        self.max_depth = float(max_depth)
        self.query_anchor_slots = checked_anchor_slots(query_anchor_slots)
        self.size = size
        self.patch_size = patch_size
        self.input_depth_max = self.max_depth if input_depth else None
        self.input_camera_vectors = input_camera_vectors
        self._report = _LabelDepthReport(self.max_depth)

    def identity(self) -> dict:
        """What the eval records about the set it scored.

        ``clips`` is the flag's value as given and ``name`` the set directory's
        own; ``protocol`` is ``benchmark.json``, the rule that wrote the set,
        or None for a set without one.  ``scene_count`` and ``max_depth`` mean
        what they mean in a loader-built set's record.
        """

        return {
            "clips": self.clips,
            "name": os.path.basename(os.path.abspath(self.clips)),
            "protocol": self.protocol,
            "scene_count": len(self.scene_names),
            "max_depth": self.max_depth,
        }

    def __call__(self, plan):
        path = self._paths.get(plan.seq_name)
        if path is None:
            raise SceneProviderError(
                f"scene {plan.seq_name!r} is not in the benchmark clip set at "
                f"{self.data_root!r} ({len(self.scene_names)} clips)"
            )
        return _scene_from_clip(
            path,
            plan,
            query_anchor_slots=self.query_anchor_slots,
            adaptive_query_anchors=False,
            max_depth=self.max_depth,
            scene_transform=None,
            min_shared_queries=None,
            size=self.size,
            patch_size=self.patch_size,
            input_depth_max=self.input_depth_max,
            input_camera_vectors=self.input_camera_vectors,
            report_label_depth=self._report,
        )
