"""Turn one MVTracker manifest record into a concrete 4RC window selection.

This is the replay's decision layer and nothing else: it reads a record, decides
which cameras, which frames and which tracks that step should use, and returns a
plan.  It loads no scene, imports no torch and touches no filesystem beyond the
manifest itself, which is what lets the whole of it be tested on CPU against
hand-written records -- and what keeps it independent of the still-open question
of whether scenes arrive from the live dataset or through the dumper.

Three properties of the schema shape the code, all of them documented in
``sample_manifest``:

* ``track_indices`` **may repeat**, deliberately -- the loader draws from a
  "dynamic" and a "very dynamic" pool, the second a subset of the first.  A
  repeat is a second independent column with its own query timestep, so tracks
  are carried positionally and a ``set`` never touches them.  Keying by id here
  would silently drop query points and nothing downstream could tell.
* ``data_root`` is per **row**, not per run.  A run's static-pretraining epoch
  draws from four other datasets, and those rows are perfectly replayable rows
  that this trainer simply does not want; only a *null* ``data_root`` is
  unreplayable by construction.  The two are counted separately, because a
  legitimate static epoch is hundreds of rows and would trip any threshold set
  for genuine corruption.
* ``scene_transform`` moves the labels, not the pixels, and a replay that skips
  it trains on different geometry.  It is carried through verbatim rather than
  interpreted here.

The frame window is ``frame_start + stride*k``: a fixed stride, so embedding row
``k`` always means the same interval, and offset 0, because the alternative was
measured to halve per-step supervision: query times cluster hard at t=0, so an
odd offset drops from about 81% of queries anchorable to about 16%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Which bound decided T. Reported per step rather than inferred, because the
# three have very different meanings: `budget` is the memory ceiling, `rows` is
# the embedding table, and `window` means the manifest handed us a shorter clip
# than the budget would have allowed.
BOUND_BUDGET = "budget"
BOUND_EMBEDDING_ROWS = "embedding_rows"
BOUND_WINDOW = "window"

# Why a record produced no step. Kept as constants so the counters, the messages
# and the tests all name the same thing.
SKIP_UNREPLAYABLE = "unreplayable"
SKIP_EXCLUDED_DATA_ROOT = "excluded_data_root"
SKIP_TOO_FEW_VIEWS = "too_few_views"
SKIP_WINDOW_TOO_SHORT = "window_too_short"

# Only this one means the manifest is damaged; the rest are policy or geometry.
THRESHOLD_SKIP_CAUSES = frozenset({SKIP_UNREPLAYABLE})


@dataclass(frozen=True)
class StepPlan:
    """One replayable training step, fully decided."""

    step: int
    seq_name: str
    data_root: str
    cameras: tuple[int, ...]
    times: tuple[int, ...]
    frame_start: int
    seq_len: int
    stride: int
    time_bound: str
    track_indices: tuple[int, ...]
    scene_transform: dict | None
    depth_type: str | None
    sample_index: int | None = None
    augmented: bool | None = None

    @property
    def observation_count(self) -> int:
        return len(self.cameras) * len(self.times)

    @property
    def duplicate_track_count(self) -> int:
        """How many ``track_indices`` entries repeat an earlier one.

        Non-zero is normal and expected. It is reported so that a positional
        selection and a set-based one are distinguishable by a number in the
        run's own artifacts, rather than by noticing missing supervision weeks
        later.
        """

        return len(self.track_indices) - len(set(self.track_indices))


@dataclass(frozen=True)
class Skipped:
    """A record that produced no step, and why."""

    step: int | None
    seq_name: str | None
    cause: str
    detail: str

    @property
    def counts_toward_threshold(self) -> bool:
        return self.cause in THRESHOLD_SKIP_CAUSES


@dataclass
class PlanTally:
    """Aggregate outcome of walking a manifest."""

    planned: list[StepPlan] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)

    @property
    def skip_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.skipped:
            counts[entry.cause] = counts.get(entry.cause, 0) + 1
        return counts

    @property
    def threshold_skip_count(self) -> int:
        return sum(1 for entry in self.skipped if entry.counts_toward_threshold)

    @property
    def considered(self) -> int:
        """Rows that were candidates for the threshold.

        Deliberate exclusions are not failures, so they leave the denominator as
        well as the numerator -- otherwise excluding a large static-pretraining
        epoch would push the unreplayable *fraction* down and hide real damage.
        """

        deliberate = sum(
            1 for entry in self.skipped if entry.cause == SKIP_EXCLUDED_DATA_ROOT
        )
        return len(self.planned) + len(self.skipped) - deliberate

    @property
    def threshold_skip_fraction(self) -> float:
        return 0.0 if self.considered == 0 else self.threshold_skip_count / self.considered


class ManifestPlanError(ValueError):
    """The manifest cannot be replayed at all, as opposed to one bad row."""


def require_cameras(
    seq_name: str,
    available_camera_ids: Sequence[int],
    requested: Sequence[int],
) -> tuple[int, ...]:
    """Check requested camera ids against what the scene actually has.

    Membership is checked against ids the caller supplies rather than by catching
    a loader exception: the loader raises ``ValueError`` for non-increasing
    times, a bad anchor and a bad upscaling factor too, and a wrapper that
    reported all of those as a camera problem would send the reader hunting in
    the wrong place.

    ``available_camera_ids`` is injected so this stays testable without a scene
    and works unchanged whichever source ends up providing scenes.
    """

    available = list(dict.fromkeys(int(value) for value in available_camera_ids))
    requested = tuple(int(value) for value in requested)
    missing = [camera for camera in requested if camera not in available]
    if missing:
        raise ManifestPlanError(
            f"scene {seq_name!r} does not have camera(s) {missing}; it has "
            f"{available}. The manifest names original camera ids, and a dump "
            "resolves them through its own view_ids -- never substitute or drop "
            "a camera, because the replay would then train on a different rig."
        )
    return requested


def select_times(
    *,
    frame_start: int,
    seq_len: int,
    view_count: int,
    budget: int,
    stride: int,
    max_time_indices: int,
) -> tuple[tuple[int, ...], str]:
    """The frames this step trains on, and which bound decided how many.

    ``T`` is the smallest of what the observation budget allows at this camera
    count, what the embedding table can index, and what the window physically
    holds.  Times are ``frame_start + stride*k`` -- a *fixed* stride, so row
    ``k`` means the same interval on every step, which is the whole reason the
    embedding can mean anything at all.

    Returns ``(times, bound)`` and never an empty selection; a window too short
    for even one frame is the caller's to reject.
    """

    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if view_count < 1:
        raise ValueError(f"view_count must be at least 1, got {view_count}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")

    from_budget = budget // view_count
    from_window = 1 + (seq_len - 1) // stride
    candidates = (
        (from_budget, BOUND_BUDGET),
        (max_time_indices, BOUND_EMBEDDING_ROWS),
        (from_window, BOUND_WINDOW),
    )
    count, bound = min(candidates, key=lambda pair: (pair[0], _bound_rank(pair[1])))
    if count < 1:
        raise ValueError(
            f"no frames fit: budget {budget} at {view_count} views, "
            f"max_time_indices {max_time_indices}, window {seq_len} at stride {stride}"
        )
    times = tuple(frame_start + stride * k for k in range(count))
    return times, bound


def _bound_rank(bound: str) -> int:
    """Tie-break order when two bounds give the same T.

    Deterministic rather than dict-order, and ordered by how actionable the
    answer is: the budget is a flag the operator sets, the table is a model
    constant, and the window is the manifest's own.
    """

    return {BOUND_BUDGET: 0, BOUND_EMBEDDING_ROWS: 1, BOUND_WINDOW: 2}[bound]


def plan_record(
    record: dict[str, Any],
    *,
    available_cameras=None,
    budget: int = 40,
    stride: int = 2,
    max_time_indices: int = 32,
    min_views: int = 2,
    excluded_data_roots: Iterable[str] = (),
) -> StepPlan | Skipped:
    """Decide one step from one manifest record, or say why there is none.

    ``available_cameras`` is either a mapping from scene name to camera ids or a
    callable taking the scene name; ``None`` skips the camera check, which is
    what ``--plan_only`` does when no scene source is configured yet.
    """

    step = record.get("step")
    seq_name = record.get("seq_name")
    data_root = record.get("data_root")

    # A null data_root is the schema's marker for a row carrying no provenance at
    # all: sample_index, views, frame_start and track_indices are null with it,
    # so there is nothing to replay. Distinct from a row this trainer merely does
    # not want, which is why the two get separate counters.
    if data_root is None:
        return Skipped(
            step=step,
            seq_name=seq_name,
            cause=SKIP_UNREPLAYABLE,
            detail=(
                "data_root is null, so the row carries no provenance stamp and "
                "its loader observations are null with it"
            ),
        )
    if str(data_root) in {str(value) for value in excluded_data_roots}:
        return Skipped(
            step=step,
            seq_name=seq_name,
            cause=SKIP_EXCLUDED_DATA_ROOT,
            detail=f"data_root {data_root!r} is excluded by configuration",
        )

    views = tuple(int(value) for value in record["views"])
    if len(views) < min_views:
        return Skipped(
            step=step,
            seq_name=seq_name,
            cause=SKIP_TOO_FEW_VIEWS,
            detail=(
                f"{len(views)} view(s), need at least {min_views}: a window with "
                "fewer has no synchronized cross-view pair, which is the only "
                "thing the time index can be learned from"
            ),
        )

    if available_cameras is not None:
        ids = (
            available_cameras(seq_name)
            if callable(available_cameras)
            else available_cameras[seq_name]
        )
        require_cameras(seq_name, ids, views)

    frame_start = int(record["frame_start"])
    seq_len = int(record["seq_len"])
    if seq_len < 1:
        return Skipped(
            step=step,
            seq_name=seq_name,
            cause=SKIP_WINDOW_TOO_SHORT,
            detail=f"seq_len {seq_len} holds no frames",
        )

    times, bound = select_times(
        frame_start=frame_start,
        seq_len=seq_len,
        view_count=len(views),
        budget=budget,
        stride=stride,
        max_time_indices=max_time_indices,
    )

    # Positional, order-preserving, duplicates intact. See the module docstring:
    # a repeated id is a second column with its own query timestep, and a set
    # here would drop it without a trace.
    track_indices = tuple(int(value) for value in record["track_indices"])

    return StepPlan(
        step=step,
        seq_name=str(seq_name),
        data_root=str(data_root),
        cameras=views,
        times=times,
        frame_start=frame_start,
        seq_len=seq_len,
        stride=stride,
        time_bound=bound,
        track_indices=track_indices,
        scene_transform=record.get("scene_transform"),
        depth_type=record.get("depth_type"),
        sample_index=record.get("sample_index"),
        augmented=record.get("augmented"),
    )


def plan_manifest(records: Iterable[dict[str, Any]], **kwargs) -> PlanTally:
    """Walk a manifest in file order, planning every record it can."""

    tally = PlanTally()
    for record in records:
        outcome = plan_record(record, **kwargs)
        if isinstance(outcome, StepPlan):
            tally.planned.append(outcome)
        else:
            tally.skipped.append(outcome)
    return tally
