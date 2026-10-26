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

Importing MVTracker is deferred to the first load: this module is imported by the
trainer's tests, which run in an environment where ``mvtracker`` is deliberately
absent.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


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
        # anchor, so it changes nothing. Negatives are refused at construction:
        # resolve_query_anchors checks only the upper bound, so a -1 would
        # otherwise wrap via Python indexing to the LAST camera and anchor a
        # different observation than any documented slot meaning, silently.
        slots = []
        for view_slot, time_slot in query_anchor_slots:
            view_slot, time_slot = int(view_slot), int(time_slot)
            if view_slot < 0 or time_slot < 0:
                raise ValueError(
                    f"query anchor slot {view_slot}:{time_slot} is negative; "
                    "slots index the step's own window and never wrap"
                )
            slots.append((view_slot, time_slot))
        self.query_anchor_slots = tuple(slots)
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
        if len(pool) < self.min_shared_queries:
            raise SceneProviderError(
                f"scene {plan.seq_name!r}: only {len(pool)} eligible tracks, below "
                f"--min_shared_queries {self.min_shared_queries}. This is a scene "
                "whose own pool is too small to be worth a step, not a failure to "
                "match a recorded draw"
            )

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


# --- Held-out benchmarks. The eval can score an MVTracker benchmark in place of
# the --val_scenes_file scenes, and such a set is opened by the benchmark's OWN
# protocol -- MVTracker's from_name at subset "test", exactly the call 4I4's
# triton/4rc/multiview/dump_common.py::build_dataset makes -- never by the
# replay's dataset_overrides above: the point is to score the queries
# MVTracker's figures are scored on. One entry per dataset family, matched by
# name prefix, so Panoptic or DexYCB is one more entry.


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


# (name prefix, parse-time refusal, opener)
BENCHMARK_DATASETS = (
    ("kubric-multiview-v3", _kubric_benchmark_refusal, _open_kubric_benchmark),
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
        if self.kwargs.get("seq_len") != seq_len:
            raise ValueError(
                f"--eval_benchmark_dataset {dataset_name!r} loads "
                f"{self.kwargs.get('seq_len')}-frame clips, but the held-out window "
                f"is planned on --val_seq_len {seq_len}"
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
        """What the eval records about the set it scored."""

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
