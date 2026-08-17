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

Importing MVTracker is deferred to the first load: this module is imported by the
trainer's tests, which run in an environment where ``mvtracker`` is deliberately
absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    ):
        # There is deliberately no ``subset``: it was only ever a path component,
        # and after the override above it decides nothing. A held-out set is a
        # different ``data_root``, not a different subset of the same one.
        self.dataset_name = dataset_name
        self.size = size
        self.patch_size = patch_size
        # Now guards "this scene's own pool is too small to be worth a step",
        # not "too little of a recorded draw survived".
        self.min_shared_queries = min_shared_queries
        self.honour_recorded_tracks = honour_recorded_tracks
        # Reported rather than raised on: how much of the recorded draw the
        # opt-in path failed to find. Meaningless when the flag is off.
        self.missing_track_ids = 0
        self.requested_track_ids = 0
        # Keyed on data_root alone, which is now fully determining.
        self._datasets: dict[str, object] = {}

    def _dataset(self, data_root: str):
        cached = self._datasets.get(data_root)
        if cached is not None:
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
        self._datasets[data_root] = dataset
        return dataset

    @staticmethod
    def dataset_overrides(data_root: str) -> dict:
        """Everything `from_name`'s *evaluation* defaults get wrong for a replay.

        Split out so it can be tested without MVTracker importable. It is the
        whole of what this repo decides about how scenes are loaded, and three of
        these seven were shipped-and-fatal before they were caught, so the reasons
        are recorded per line rather than in a commit message.
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

    def __call__(self, plan):
        from arc.training import scene_from_datapoint

        dataset = self._dataset(plan.data_root)
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
            size=self.size,
            patch_size=self.patch_size,
        )
