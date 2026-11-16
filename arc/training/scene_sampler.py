"""Draw a training stream from a packed scene pool, MVTracker's way, without a manifest.

The manifest replay has one seam the rest of the trainer was written around:
:func:`arc.training.manifest_plan.plan_record` reads a record and decides a step,
and nothing downstream -- the provider, the loop, the cache, the held-out eval --
asks where that record came from.  This module produces records of that same
schema from a seeded draw over a pool, so a run without a manifest is "a
different record source", not a different trainer.  Records live in memory and
are never written; ``plan.json`` and ``run_summary.json`` say the run was
sampled (``scene_source``), and the checkpoint carries the sampler's identity so a
resume continues the same stream.

**What the draw reproduces.**  MVTracker's loader (kubric_multiview_dataset.py,
checkout 4002795) draws the view count per sample from
``{1: .20, 2: .10, 3: .10, 4: .25, 5: .10, 6: .25}`` (:312-320), the views by
``choice(range(n_views), k, replace=False)`` with the drawn order kept (:621),
returns the whole 24-frame clip at ``frame_start`` 0, and draws the world
transform as rotation x then y, each ``U(-15, 15)`` degrees, and scale
``U(0.8, 1.5)`` (:1061-1063).  The trainer's ``--min_views 2`` removes the
single-view mass from a replay, so the default weights here are the five
multi-view counts, renormalised.  The draw is deliberately NOT bit-for-bit
MVTracker's stream: that stream is per-sample ``RandomState(seed + index)`` with
the crop and the track selection interleaved, which a replay cannot reproduce
either (see ``scene_provider``).  What a sampled run shares with a replayed one
is the distribution -- the same view-count mix, the same transform ranges, every
scene once per epoch -- and every planned step goes through the same planner
with the same arguments.

**Why an index file.**  ``--plan_only`` promises to touch no scene data, and the
loader's own construction scans every archive (one central-directory parse per
scene, ~8 s for the full split).  The pool is therefore read through
``<pool>/pool_index.json``: the scene names in the loader's order, the view count
per scene (the population of the view draw, which is why it is recorded rather
than assumed) and a content hash over both.  :func:`build_pool_index` applies
the loader's listing rule exactly -- ``os.listdir``, entries starting with ``.``
or ``_`` skipped, ``<name>.zip`` archives and directories, a name present as
both refused, ordered by ``int(name)``, zero-view scenes left out -- so the
index lists what the loader will list, and every sampled record carries the
index's scene count as ``real_len`` for the provider to check against the pool
it actually opens.

**A second pool.**  ``--extra_pool`` mixes a pool of dump-format clips into
the stream (4I4's Syn4D converter writes them; ``arc.training.dump_clip`` reads
their layout).  Each step then first draws which pool, ``random_sample() < P``
from the same stream, and takes that pool's next scene from its OWN epoch
shuffle -- so each pool is covered once per its own epoch -- before the count,
views and transform are drawn exactly as for a single pool.  The pool draw
happens only when there is a second pool, so a run without one consumes the
stream exactly as before.  A clip pool's index is listed by name rather than
``int(name)`` and says ``"pool_kind": "dump_clips"`` (:func:`list_pool`).

**RNG.**  ``numpy.random.RandomState``, not ``default_rng``: NumPy freezes the
legacy stream across versions (NEP 19) while ``Generator`` methods may change
between releases, and a resume segment after an environment rebuild must
continue the same stream.  It is also the class the loader draws from.

Imports no torch, like ``manifest_plan``: the CPU suite and ``--build_pool_index``
run it without one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from arc.training.dump_clip import CLIP_POOL_KIND, ClipFormatError, clip_view_ids
from arc.training.manifest_plan import PlanTally, StepPlan, plan_record
from arc.training.sample_manifest import MANIFEST_RECORD_KEYS, MANIFEST_VERSION
from arc.training.scene_archive import ARCHIVE_SUFFIX, scene_view_count

POOL_INDEX_NAME = "pool_index.json"
POOL_INDEX_VERSION = 1
# The two kinds of pool an index can describe. An MV-Kubric pool's index has no
# pool_kind key -- it predates the second kind and stays byte-identical -- so
# KUBRIC_POOL is what an absent key reads as; a clip pool's says CLIP_POOL_KIND.
KUBRIC_POOL = "kubric"
# Which pool a draw came from, by its flag: the label the log line, the history
# row and scene_source's per-pool block all use.
SCENE_POOL = "scene_pool"
EXTRA_POOL = "extra_pool"

# The loader's clip. KubricMultiViewDataset.from_name emits "seq_len": 24 on the
# evaluation path (kubric_multiview_dataset.py:133), the constructor defaults to
# the same (:209), __getitem__ crops every longer clip to exactly seq_len
# (:826-841), and MVTrackerSceneProvider.dataset_overrides deliberately leaves
# seq_len alone -- so every sampled window is [0, 24), and the trainer's held-out
# window default (DEFAULT_VAL_SEQ_LEN) is this same number, imported from here.
# If a seq_len override is ever added to the provider, this must move with it,
# or the planned window and the loaded clip desync silently.
LOADER_CLIP_FRAMES = 24
SAMPLED_FRAME_START = 0
# The pools this trainer samples carry ground-truth depth; the loader's depth
# draw is off in the provider's overrides, so there is no other source to name.
SAMPLED_DEPTH_TYPE = "gt"

# MVTracker's view-count draw at --min_views 2: the single-view mass dropped and
# the rest renormalised per scene by admissible_counts. As a string because the
# flag is one, and the canonical form of a string is what the checkpoint stores.
DEFAULT_VIEW_COUNT_WEIGHTS = "2:0.10,3:0.10,4:0.25,5:0.10,6:0.25"
# The loader's transform ranges (kubric_multiview_dataset.py:1061-1063).
SCALE_RANGE = (0.8, 1.5)
ROTATION_RANGE_DEG = (-15.0, 15.0)

# The StepPlan fields the --sampler_from_manifest diagnostic compares: the window
# and what builds it. Deliberately NOT track_indices, real_len, depth_type,
# sample_index or augmented: those are the manifest's recorded draw and recorded
# pool, which a sampled record has no counterpart for. The like-for-like claim
# is about the window, the cameras and the transform.
WINDOW_FIELDS = (
    "step",
    "seq_name",
    "data_root",
    "cameras",
    "times",
    "frame_start",
    "seq_len",
    "stride",
    "time_bound",
    "scene_transform",
)


class ScenePoolError(ValueError):
    """The pool, its index or the sampler's settings cannot produce a stream."""


@dataclass(frozen=True)
class PoolScene:
    name: str
    view_count: int


@dataclass(frozen=True)
class PoolIndex:
    """The pool as the loader would list it, read from ``pool_index.json``."""

    scenes: tuple[PoolScene, ...]
    content_hash: str

    @property
    def scene_count(self) -> int:
        return len(self.scenes)


# ------------------------------------------------------------------ weights ---


def parse_view_count_weights(text: str) -> dict[int, float]:
    """``"2:0.10,3:0.10"`` -> ``{2: 0.1, 3: 0.1}``, sorted by count.

    Format only: counts are positive integers, weights positive finite floats,
    no count twice.  Whether the counts fit the run (``--min_views``, the budget)
    is :func:`check_view_count_weights`'s question, asked beside those flags.
    """

    weights: dict[int, float] = {}
    text = str(text)
    if not text.strip():
        raise ValueError(
            "--view_count_weights is empty; expected COUNT:WEIGHT pairs such as "
            f"{DEFAULT_VIEW_COUNT_WEIGHTS!r}"
        )
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            raise ValueError(f"--view_count_weights {text!r} has an empty entry")
        count_text, separator, weight_text = entry.partition(":")
        if not separator:
            raise ValueError(
                f"--view_count_weights entry {entry!r} is not COUNT:WEIGHT"
            )
        try:
            count = int(count_text)
        except ValueError:
            raise ValueError(
                f"--view_count_weights entry {entry!r}: count {count_text!r} is "
                "not an integer"
            ) from None
        try:
            weight = float(weight_text)
        except ValueError:
            raise ValueError(
                f"--view_count_weights entry {entry!r}: weight {weight_text!r} is "
                "not a number"
            ) from None
        if count < 1:
            raise ValueError(
                f"--view_count_weights entry {entry!r}: a view count is at least 1"
            )
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"--view_count_weights entry {entry!r}: a weight is finite and "
                "positive; leave a count out rather than weighting it 0"
            )
        if count in weights:
            raise ValueError(
                f"--view_count_weights {text!r} lists view count {count} twice"
            )
        weights[count] = weight
    return dict(sorted(weights.items()))


def check_view_count_weights(
    weights: Mapping[int, float], *, min_views: int, budget: int
) -> None:
    """Refuse a count the run could never train at, naming both flags.

    A count below ``--min_views`` is the subtle one: the planner skips every
    step drawn at it, so the run would train the remaining counts' mix while the
    recorded identity said otherwise -- ``"1:0.20,..."`` at the default
    ``--min_views 2`` trains the plain base distribution.  A count above the
    budget cannot seat one time, and ``select_times`` would raise mid-plan
    rather than at the flag.
    """

    below = [count for count in sorted(weights) if count < min_views]
    if below:
        raise ValueError(
            f"--view_count_weights lists view count(s) {below} below --min_views "
            f"{min_views}: the planner skips every step drawn at those counts, so "
            "the weights would describe a mix the run never trains. Drop them, or "
            "lower --min_views"
        )
    above = [count for count in sorted(weights) if count > budget]
    if above:
        raise ValueError(
            f"--view_count_weights lists view count(s) {above} above "
            f"--observation_budget {budget}, which cannot seat even one time at "
            "that many views"
        )


def format_view_count_weights(weights: Mapping[int, float]) -> str:
    """The canonical string: sorted by count, floats in their shortest repr.

    ``"2:0.10"`` and ``"2:0.1"`` are one setting; the checkpoint stores this
    form so a resume compares settings, not spellings.
    """

    return ",".join(
        f"{int(count)}:{float(weight)!r}" for count, weight in sorted(weights.items())
    )


# -------------------------------------------------------------------- index ---


def index_build_command(pool, *, kind: str) -> str:
    """The command that (re)builds ``pool``'s index, naming the flag the pool goes to."""

    flag = "--extra_pool" if kind == CLIP_POOL_KIND else "--scene_pool"
    pool = shlex.quote(os.fspath(pool))
    return f"python train_temporal_tracking.py {flag} {pool} --build_pool_index"


def _scenes_hash(scenes: list[dict]) -> str:
    """sha256 over the canonical JSON of the scene list -- names and view counts."""

    canonical = json.dumps(scenes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def list_pool(pool, *, kind: str) -> dict[str, str]:
    """``{name: path}`` for every scene or clip under ``pool``, in the pool's order.

    One listing rule for both kinds: ``os.listdir``, entries starting with
    ``.`` or ``_`` skipped, ``<name>.zip`` files and directories, a name present
    as both refused.  An MV-Kubric pool (``KUBRIC_POOL``) is
    kubric_multiview_dataset.py:390-440 (MVTracker 4002795) verbatim besides:
    names must be integers and the order is ``int(name)``, because the loader
    the provider replays orders it so.  A clip pool (``CLIP_POOL_KIND``) is
    read by no loader of MVTracker's, so any name is a clip and the order is a
    plain string sort -- the converter's ``<scene>__<seq>__<start>`` names stay
    traceable to their source.  Reads no scene.
    """

    pool = os.fspath(pool)
    clips = kind == CLIP_POOL_KIND
    flag, noun = ("--extra_pool", "clip") if clips else ("--scene_pool", "scene")
    if not os.path.isdir(pool):
        raise ScenePoolError(f"{flag} {pool!r} is not a directory")

    paths: dict[str, str] = {}
    for entry in os.listdir(pool):
        if entry.startswith(".") or entry.startswith("_"):
            continue
        path = os.path.join(pool, entry)
        if entry.endswith(ARCHIVE_SUFFIX) and os.path.isfile(path):
            name = entry[: -len(ARCHIVE_SUFFIX)]
        elif os.path.isdir(path):
            name = entry
        else:
            continue
        if name in paths:
            raise ScenePoolError(
                f"{noun} {name!r} exists both as a directory and as an archive "
                f"under {pool}; "
                + (
                    "a clip is one or the other, so remove one"
                    if clips
                    else "MVTracker's loader refuses the same pool, so remove one"
                )
            )
        paths[name] = path
    if not paths:
        raise ScenePoolError(
            f"{pool} holds no {noun}s: no <name>{ARCHIVE_SUFFIX} archive and no "
            f"{noun} directory"
        )
    if clips:
        return {name: paths[name] for name in sorted(paths)}

    not_integers = []
    for name in paths:
        try:
            int(name)
        except ValueError:
            not_integers.append(name)
    if not_integers:
        raise ScenePoolError(
            f"scene name(s) {sorted(not_integers)} under {pool} are not integers; "
            "MVTracker's loader orders a pool by int(name) and would refuse it"
        )
    return {name: paths[name] for name in sorted(paths, key=int)}


def build_pool_index(pool) -> tuple[dict, list[str]]:
    """Scan an MV-Kubric ``pool`` as the loader would and return ``(document, zero_view_names)``.

    The listing is :func:`list_pool`'s Kubric rule, and a scene with no
    ``view_*`` folders is left out, as the loader leaves it out.  The names
    the loader would skip are returned beside the document for the caller to
    print; they are not stored and not hashed, because the index is "what the
    loader lists".
    """

    paths = list_pool(pool, kind=KUBRIC_POOL)
    scenes: list[dict] = []
    zero_view: list[str] = []
    for name, path in paths.items():
        try:
            view_count = int(scene_view_count(path))
        except Exception as error:
            raise ScenePoolError(
                f"scene {name!r} at {path} cannot be read: "
                f"{type(error).__name__}: {error}"
            ) from error
        if view_count == 0:
            zero_view.append(name)
            continue
        scenes.append({"name": name, "view_count": view_count})
    if not scenes:
        raise ScenePoolError(f"every scene under {os.fspath(pool)} has zero views")

    document = {
        "pool_index_version": POOL_INDEX_VERSION,
        "scene_count": len(scenes),
        "content_hash": _scenes_hash(scenes),
        "scenes": scenes,
    }
    return document, zero_view


def build_clip_pool_index(pool, *, weights: Mapping[int, float]) -> dict:
    """Scan a pool of dump-format clips and return its index document.

    The listing is :func:`list_pool`'s clip rule; each clip's view count is
    ``len(view_ids)`` read from its meta.npz (``dump_clip.clip_view_ids``),
    the population of its view draw.  Refused, naming every offender and
    writing nothing: a clip whose meta.npz or ``view_ids`` cannot be read, and
    a clip that seats none of ``weights``' counts -- the build command's
    ``--view_count_weights``, parsed for format only, so the build is gated by
    no ``--min_views``.  Every clip of a clip pool is drawable; the draw
    re-checks that against the run's own weights.  The document carries
    ``"pool_kind": "dump_clips"``, which an MV-Kubric index never does.
    """

    paths = list_pool(pool, kind=CLIP_POOL_KIND)
    scenes: list[dict] = []
    unreadable: list[str] = []
    unseatable: list[str] = []
    for name, path in paths.items():
        try:
            view_count = int(clip_view_ids(path).shape[0])
        except ClipFormatError as error:
            unreadable.append(str(error))
            continue
        if not admissible_counts(view_count, weights)[0]:
            unseatable.append(f"{name} ({view_count} views)")
            continue
        scenes.append({"name": name, "view_count": view_count})
    if unreadable:
        raise ScenePoolError(
            f"{len(unreadable)} clip(s) under {os.fspath(pool)} cannot be read: "
            + "; ".join(unreadable[:8])
            + (" ..." if len(unreadable) > 8 else "")
        )
    if unseatable:
        raise ScenePoolError(
            f"{len(unseatable)} clip(s) under {os.fspath(pool)} seat none of the "
            f"listed view counts {sorted(weights)}: "
            + ", ".join(unseatable[:8])
            + (" ..." if len(unseatable) > 8 else "")
            + ". Every clip of the extra pool must be drawable; remove them, or "
            "index with the --view_count_weights the run uses"
        )
    return {
        "pool_index_version": POOL_INDEX_VERSION,
        "pool_kind": CLIP_POOL_KIND,
        "scene_count": len(scenes),
        "content_hash": _scenes_hash(scenes),
        "scenes": scenes,
    }


def write_pool_index(pool, document: dict) -> Path:
    """Write the index beside the scenes, atomically; returns its path."""

    path = Path(os.fspath(pool)) / POOL_INDEX_NAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=1) + "\n")
    os.replace(temporary, path)
    return path


def read_pool_index(pool) -> PoolIndex:
    """An MV-Kubric pool's index, checked rather than trusted (see :func:`_read_pool_index`)."""

    return _read_pool_index(pool, kind=KUBRIC_POOL)


def read_clip_pool_index(pool) -> PoolIndex:
    """A clip pool's index, checked rather than trusted (see :func:`_read_pool_index`)."""

    return _read_pool_index(pool, kind=CLIP_POOL_KIND)


def _read_pool_index(pool, *, kind: str) -> PoolIndex:
    """The pool's index, checked rather than trusted.

    Refuses a missing file (naming the build command), another version, an
    index of the other kind -- an MV-Kubric index carries no ``pool_kind`` and a
    clip index carries ``"dump_clips"``, so a pool handed to the wrong flag
    fails here rather than at every load -- a malformed or hand-edited document
    (the hash is recomputed over the scene list), and a list its listing rule
    would not produce (out of order, repeated names, a non-positive view
    count).  Reads no scene archive.
    """

    pool = os.fspath(pool)
    path = Path(pool) / POOL_INDEX_NAME
    command = index_build_command(pool, kind=kind)
    if not path.is_file():
        raise ScenePoolError(
            f"{path} does not exist. Sampling reads the pool through its index so "
            "that --plan_only touches no scene archive; build it once, on CPU, "
            f"with:\n  {command}"
        )
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ScenePoolError(
            f"{path} is not readable JSON ({error}); rebuild it with:\n  {command}"
        ) from None
    if not isinstance(document, dict):
        raise ScenePoolError(f"{path} is not a JSON object; rebuild it with:\n  {command}")
    version = document.get("pool_index_version")
    if version != POOL_INDEX_VERSION:
        raise ScenePoolError(
            f"{path} has pool_index_version {version!r}; this trainer reads "
            f"{POOL_INDEX_VERSION}. Rebuild it with:\n  {command}"
        )
    stored_kind = document.get("pool_kind", KUBRIC_POOL)
    if stored_kind != kind:
        raise ScenePoolError(
            f"{path} indexes "
            + (
                "a pool of dump-format clips"
                if stored_kind == CLIP_POOL_KIND
                else "an MV-Kubric scene pool"
                if stored_kind == KUBRIC_POOL
                else f"a pool of unknown kind {stored_kind!r}"
            )
            + ", but it was given as "
            + ("--extra_pool" if kind == CLIP_POOL_KIND else "--scene_pool")
            + ", which reads "
            + (
                "dump-format clips"
                if kind == CLIP_POOL_KIND
                else "MV-Kubric scenes through MVTracker's loader"
            )
            + ". Pass the pool to the other flag"
        )
    scenes = document.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ScenePoolError(f"{path} lists no scenes; rebuild it with:\n  {command}")
    if _scenes_hash(scenes) != document.get("content_hash"):
        raise ScenePoolError(
            f"{path}: content_hash does not match its scene list, so the file was "
            f"edited or truncated after it was built. Rebuild it with:\n  {command}"
        )
    if document.get("scene_count") != len(scenes):
        raise ScenePoolError(
            f"{path}: scene_count {document.get('scene_count')!r} disagrees with "
            f"the {len(scenes)} scenes listed; rebuild it with:\n  {command}"
        )

    integer_names = kind == KUBRIC_POOL
    entries: list[PoolScene] = []
    for position, scene in enumerate(scenes):
        try:
            name = str(scene["name"])
            view_count = int(scene["view_count"])
            if integer_names:
                int(name)
            elif not name:
                raise ValueError("empty name")
        except (KeyError, TypeError, ValueError):
            raise ScenePoolError(
                f"{path}: scene entry {position} ({scene!r}) is not "
                + (
                    '{{"name": <integer string>, "view_count": <int>}}'
                    if integer_names
                    else '{{"name": <non-empty string>, "view_count": <int>}}'
                )
                + f"; rebuild it with:\n  {command}"
            ) from None
        if view_count < 1:
            raise ScenePoolError(
                f"{path}: scene {name!r} lists {view_count} views; the loader lists "
                f"no such scene. Rebuild it with:\n  {command}"
            )
        entries.append(PoolScene(name=name, view_count=view_count))
    names = [scene.name for scene in entries]
    if len(set(names)) != len(names):
        raise ScenePoolError(f"{path} lists a scene twice; rebuild it with:\n  {command}")
    if names != (sorted(names, key=int) if integer_names else sorted(names)):
        raise ScenePoolError(
            f"{path} is not in "
            + ("the loader's order (by int(name))" if integer_names else "name order")
            + f"; rebuild it with:\n  {command}"
        )
    return PoolIndex(scenes=tuple(entries), content_hash=str(document["content_hash"]))


# --------------------------------------------------------------------- draw ---


def admissible_counts(
    view_count: int, weights: Mapping[int, float]
) -> tuple[list[int], list[float]]:
    """The listed counts this scene can seat, with their renormalised probabilities.

    Only counts above the scene's own view count are dropped here: a count below
    ``--min_views`` was refused at parse time (:func:`check_view_count_weights`)
    rather than dropped, because dropping it would silently change the mix the
    identity records.  ``([], [])`` when the scene seats none of them.
    """

    counts = [count for count in sorted(weights) if count <= view_count]
    if not counts:
        return [], []
    total = sum(float(weights[count]) for count in counts)
    return counts, [float(weights[count]) / total for count in counts]


@dataclass(frozen=True)
class ExtraPool:
    """The second pool a mixed run draws from, and the share of steps it takes.

    ``index`` is the clip pool's index (:func:`read_clip_pool_index`),
    ``data_root`` the ``--extra_pool`` path every record drawn from it carries,
    and ``weight`` ``--extra_pool_weight``, the per-step probability of drawing
    from it, in (0, 1].
    """

    index: PoolIndex
    data_root: str
    weight: float


@dataclass(frozen=True)
class Draw:
    """One sampled scene before it becomes a record."""

    epoch: int
    seq_name: str
    view_count: int
    views: tuple[int, ...]
    scene_transform: dict | None
    # SCENE_POOL or EXTRA_POOL; ``epoch`` counts that pool's own epochs.
    pool: str


class _EpochShuffle:
    """One pool's scenes, each once per epoch, in an order drawn from the run's stream.

    The permutation runs over EVERY scene of the index, admissible or not, and
    is drawn only when the pool is asked for a scene and has none left in its
    epoch: that is when the single-pool generator drew it too, so one pool's
    consumption of the stream is the one it always was.  A scene that seats no
    listed count is passed over, and draws nothing.
    """

    def __init__(self, index: PoolIndex, weights: Mapping[int, float]):
        self.index = index
        self.admissible = [
            admissible_counts(scene.view_count, weights) for scene in index.scenes
        ]
        self.order: np.ndarray | tuple = ()
        self.cursor = 0
        self.epoch = -1

    def next_scene(self, rng: np.random.RandomState):
        while True:
            if self.cursor == len(self.order):
                self.order = rng.permutation(len(self.index.scenes))
                self.cursor = 0
                self.epoch += 1
            position = int(self.order[self.cursor])
            self.cursor += 1
            counts, probabilities = self.admissible[position]
            if counts:
                return self.index.scenes[position], counts, probabilities


def draw_scenes(
    index: PoolIndex,
    *,
    seed: int,
    weights: Mapping[int, float],
    transform_probability: float,
    min_views: int,
    extra: ExtraPool | None = None,
) -> Iterator[Draw]:
    """The infinite draw: every admissible scene once per epoch, in seeded order.

    Per pool an epoch is a ``RandomState(seed)`` permutation over its index
    (:class:`_EpochShuffle`); per scene the view count from the scene's
    :func:`admissible_counts`, the views without replacement in the drawn
    order, and the transform.  The transform's three values are drawn for
    EVERY step and ``transform_probability`` only decides whether the step
    carries them, so two runs differing in it see the same scenes, views and
    transform values and differ only in which steps are transformed.  A scene
    that seats no listed count is skipped that epoch, and every other scene
    appears exactly once per epoch, so a pool is covered before any scene
    repeats.

    With ``extra``, each step first draws its pool -- ``random_sample() <
    extra.weight`` from the same stream picks the extra pool -- then the
    chosen pool's next scene (a new permutation of that pool if its epoch is
    spent), then the values above.  Without it nothing is drawn for the pool,
    so the stream is the single-pool stream exactly.

    Refused up front, before the first draw, so a direct caller cannot bypass
    them and no outcome depends on how far the stream is drawn: a listed count
    below ``min_views``, a probability outside [0, 1], a pool no scene of which
    seats any listed count, an extra weight outside (0, 1], and any extra clip
    that seats no listed count -- every clip of the extra pool is drawable, as
    its index build already checked against that command's weights.  A count
    above the observation budget is the planner's concern (``_validate_args``
    refuses it by flag; a direct caller gets ``select_times``' own error).
    """

    weights = {int(count): float(weight) for count, weight in weights.items()}
    if not weights:
        raise ScenePoolError("no view-count weights were given")
    below = [count for count in sorted(weights) if count < min_views]
    if below:
        raise ScenePoolError(
            f"view count(s) {below} are below min_views {min_views}: the planner "
            "would skip every step drawn at them"
        )
    if not (math.isfinite(transform_probability) and 0 <= transform_probability <= 1):
        raise ScenePoolError(
            f"transform_probability {transform_probability!r} is not in [0, 1]"
        )
    pools = {SCENE_POOL: _EpochShuffle(index, weights)}
    if not any(counts for counts, _ in pools[SCENE_POOL].admissible):
        widest = max((scene.view_count for scene in index.scenes), default=0)
        raise ScenePoolError(
            f"no scene in the pool seats any listed view count {sorted(weights)}: "
            f"the widest scene has {widest} view(s)"
        )
    if extra is not None:
        weight = float(extra.weight)
        if not (math.isfinite(weight) and 0 < weight <= 1):
            raise ScenePoolError(f"the extra pool's weight {extra.weight!r} is not in (0, 1]")
        pools[EXTRA_POOL] = _EpochShuffle(extra.index, weights)
        unseatable = [
            f"{scene.name} ({scene.view_count} views)"
            for scene, (counts, _) in zip(extra.index.scenes, pools[EXTRA_POOL].admissible)
            if not counts
        ]
        if unseatable:
            raise ScenePoolError(
                f"{len(unseatable)} clip(s) of the extra pool seat none of the listed "
                f"view counts {sorted(weights)}: "
                + ", ".join(unseatable[:8])
                + (" ..." if len(unseatable) > 8 else "")
                + ". Every clip of the extra pool must be drawable; remove them or "
                "change --view_count_weights"
            )

    rng = np.random.RandomState(int(seed))
    while True:
        label = SCENE_POOL
        if extra is not None and rng.random_sample() < extra.weight:
            label = EXTRA_POOL
        shuffle = pools[label]
        scene, counts, probabilities = shuffle.next_scene(rng)
        count = int(rng.choice(counts, p=probabilities))
        views = tuple(
            int(view) for view in rng.choice(scene.view_count, size=count, replace=False)
        )
        # Gate first, then the three values -- in the loader's own order
        # (rot_x, rot_y, scale) -- drawn unconditionally, see the docstring.
        transformed = bool(rng.random_sample() < transform_probability)
        rot_x_deg = float(rng.uniform(*ROTATION_RANGE_DEG))
        rot_y_deg = float(rng.uniform(*ROTATION_RANGE_DEG))
        scale = float(rng.uniform(*SCALE_RANGE))
        yield Draw(
            epoch=shuffle.epoch,
            seq_name=scene.name,
            view_count=scene.view_count,
            views=views,
            scene_transform=(
                {"scale": scale, "rot_x_deg": rot_x_deg, "rot_y_deg": rot_y_deg}
                if transformed
                else None
            ),
            pool=label,
        )


# ------------------------------------------------------------------ records ---

_RECORD_KEYS = frozenset(MANIFEST_RECORD_KEYS)


def build_record(
    *,
    step: int,
    data_root: str,
    seq_name: str,
    views: Iterable[int],
    frame_start: int,
    scene_transform: Mapping[str, Any] | None,
    dataset_seed: int | None = None,
    real_len: int | None = None,
) -> dict[str, Any]:
    """One record of the manifest schema (every ``MANIFEST_RECORD_KEYS`` key).

    What ``plan_record`` reads is set; what only a loader could observe -- the
    rank, batch position, sample index, sample seed, the recorded track draw --
    is ``None`` or empty, exactly as the schema says for an unobserved value.
    ``seq_len`` is the loader's clip; ``frame_start`` is a parameter so the
    diagnostic can feed a manifest row's own, and the sampler passes 0.
    """

    record = {
        "manifest_version": MANIFEST_VERSION,
        "step": int(step),
        "rank": None,
        "batch_index": None,
        "batch_size": None,
        "data_root": str(data_root),
        "seq_name": str(seq_name),
        "sample_index": None,
        "real_len": None if real_len is None else int(real_len),
        "dataset_seed": None if dataset_seed is None else int(dataset_seed),
        "sample_seed": None,
        "views": [int(view) for view in views],
        "frame_start": int(frame_start),
        "seq_len": LOADER_CLIP_FRAMES,
        "track_indices": [],
        "traj_per_sample": 0,
        "traj_per_sample_configured": None,
        "depth_type": SAMPLED_DEPTH_TYPE,
        "augmented": None,
        "scene_transform": None if scene_transform is None else dict(scene_transform),
    }
    # The schema is vendored and may move; a record missing a key must fail at
    # the one place records are built, not plan quietly without it.
    if set(record) != _RECORD_KEYS:
        raise AssertionError(
            "build_record no longer matches the vendored manifest schema: "
            f"missing {sorted(_RECORD_KEYS - set(record))}, "
            f"extra {sorted(set(record) - _RECORD_KEYS)}"
        )
    return record




@dataclass
class PoolTally:
    """What the draw took from one pool to reach the plan, for scene_source."""

    pool: str
    scene_count: int
    admissible_scenes: int
    epochs_drawn: int = 0
    draws: int = 0
    planned_steps: int = 0
    views_per_step: Counter = field(default_factory=Counter)

    def summary(self) -> dict:
        return {
            "pool": self.pool,
            "scene_count": self.scene_count,
            "admissible_scenes": self.admissible_scenes,
            "epochs_drawn": self.epochs_drawn,
            "draws": self.draws,
            "planned_steps": self.planned_steps,
            "views_per_step": dict(sorted(self.views_per_step.items())),
        }


@dataclass
class SampledPlanTally(PlanTally):
    """A ``PlanTally`` plus what the draw covered to reach it.

    ``epochs_drawn``, ``scene_count`` and ``admissible_scenes`` describe the
    ``--scene_pool`` pool, as they always have; ``draws`` counts every draw.
    ``pools`` holds each pool's own :class:`PoolTally`, by label.
    """

    epochs_drawn: int = 0
    draws: int = 0
    scene_count: int = 0
    admissible_scenes: int = 0
    pools: dict = field(default_factory=dict)


def plan_pool(
    index: PoolIndex,
    *,
    data_root: str,
    num_steps: int,
    seed: int,
    weights: Mapping[int, float],
    transform_probability: float,
    min_views: int,
    extra: ExtraPool | None = None,
    **plan_kwargs,
) -> SampledPlanTally:
    """Exactly ``num_steps`` planned steps from the pool(s), through ``plan_record``.

    ``plan_kwargs`` are the planner's own (``available_cameras``, ``budget``,
    ``stride``, ``max_time_indices``, ``excluded_data_roots``), the same dict
    ``main()`` hands ``plan_manifest``; ``min_views`` is named because the draw
    needs it too.  Records are built with ``step = len(planned)`` so
    ``plans[k].step == k``, and with the drawn pool's own root and index scene
    count as ``data_root`` and ``real_len``, so each provider checks the pool it
    opens against the pool that was indexed.

    The same indexes, seed, weights, probabilities and planner arguments give
    the same list, and a larger ``num_steps`` gives the same prefix: the draw
    is consumed in order and nothing wraps.  A pool whose epoch plans nothing
    raises rather than looping.
    """

    if num_steps < 1:
        raise ScenePoolError(f"num_steps must be at least 1, got {num_steps}")
    roots = {SCENE_POOL: (str(data_root), index)}
    if extra is not None:
        roots[EXTRA_POOL] = (str(extra.data_root), extra.index)
    tally = SampledPlanTally(
        scene_count=index.scene_count,
        admissible_scenes=sum(
            1 for scene in index.scenes if admissible_counts(scene.view_count, weights)[0]
        ),
        pools={
            label: PoolTally(
                pool=root,
                scene_count=pool_index.scene_count,
                admissible_scenes=sum(
                    1
                    for scene in pool_index.scenes
                    if admissible_counts(scene.view_count, weights)[0]
                ),
            )
            for label, (root, pool_index) in roots.items()
        },
    )
    current_epoch = {label: -1 for label in roots}
    planned_at_epoch_start = {label: 0 for label in roots}
    for draw in draw_scenes(
        index,
        seed=seed,
        weights=weights,
        transform_probability=transform_probability,
        min_views=min_views,
        extra=extra,
    ):
        pool_tally = tally.pools[draw.pool]
        if draw.epoch != current_epoch[draw.pool]:
            if (
                current_epoch[draw.pool] >= 0
                and pool_tally.planned_steps == planned_at_epoch_start[draw.pool]
            ):
                raise ScenePoolError(
                    ("" if draw.pool == SCENE_POOL else "the extra pool's ")
                    + f"epoch {current_epoch[draw.pool]} drew every admissible scene "
                    f"and planned none of them (skipped: {tally.skip_counts}); the "
                    "pool cannot produce a step under these settings"
                )
            current_epoch[draw.pool] = draw.epoch
            planned_at_epoch_start[draw.pool] = pool_tally.planned_steps
            pool_tally.epochs_drawn = draw.epoch + 1
        tally.draws += 1
        pool_tally.draws += 1
        root, pool_index = roots[draw.pool]
        record = build_record(
            step=len(tally.planned),
            data_root=root,
            seq_name=draw.seq_name,
            views=draw.views,
            frame_start=SAMPLED_FRAME_START,
            scene_transform=draw.scene_transform,
            dataset_seed=seed,
            real_len=pool_index.scene_count,
        )
        outcome = plan_record(record, min_views=min_views, **plan_kwargs)
        if isinstance(outcome, StepPlan):
            tally.planned.append(outcome)
            pool_tally.planned_steps += 1
            pool_tally.views_per_step[len(outcome.cameras)] += 1
        else:
            tally.skipped.append(outcome)
        if len(tally.planned) == num_steps:
            break
    tally.epochs_drawn = tally.pools[SCENE_POOL].epochs_drawn
    return tally


def shard_plans(
    planned: list[StepPlan], *, rank: int, world_size: int, num_steps: int
) -> list[StepPlan]:
    """Rank ``rank``'s share of one ``world_size``-fold draw: every ``world_size``-th plan.

    The draw is ``plan_pool`` at ``world_size * num_steps``, so each shard is
    exactly ``num_steps`` long and every pool -- the extra one included -- mixes
    into every shard at its configured weight, since the pool draw rides in the
    same stream.  :func:`plan_pool`'s prefix guarantee carries over: a longer
    draw's shard begins with the shorter draw's, so a run extended through
    ``--num_steps`` keeps every rank's earlier steps.  Each plan keeps its
    ``step``, the global draw index ``k * world_size + rank``.
    """

    if not 0 <= rank < world_size:
        raise ScenePoolError(f"rank {rank} is outside a world of {world_size}")
    if len(planned) != world_size * num_steps:
        raise ScenePoolError(
            f"{world_size} ranks of {num_steps} steps shard a draw of exactly "
            f"{world_size * num_steps} plans, but the draw holds {len(planned)}"
        )
    return list(planned[rank::world_size])


# --------------------------------------------------------------- diagnostic ---


def compare_sampler_against_manifest(records: Iterable[dict], **plan_kwargs) -> dict:
    """The like-for-like check behind ``--sampler_from_manifest``.

    For every row the manifest planner plans, the row's own scene, views,
    ``frame_start`` and transform are rebuilt through :func:`build_record` and
    planned again with the same arguments; the two plans must agree on every
    ``WINDOW_FIELDS`` entry.  Rows the planner skips (null provenance, an
    excluded root, too few views) have no sampled counterpart and are counted
    by cause under ``inexpressible``.  Returns ``{"compared", "mismatched":
    [(step, seq_name, {field: (manifest, sampler)})], "inexpressible"}``.
    """

    compared = 0
    mismatched: list[tuple] = []
    inexpressible: dict[str, int] = {}
    for record in records:
        reference = plan_record(record, **plan_kwargs)
        if not isinstance(reference, StepPlan):
            inexpressible[reference.cause] = inexpressible.get(reference.cause, 0) + 1
            continue
        rebuilt = build_record(
            step=record["step"],
            data_root=record["data_root"],
            seq_name=record["seq_name"],
            views=record["views"],
            frame_start=record["frame_start"],
            scene_transform=record.get("scene_transform"),
            dataset_seed=record.get("dataset_seed"),
            real_len=record.get("real_len"),
        )
        sampled = plan_record(rebuilt, **plan_kwargs)
        compared += 1
        if not isinstance(sampled, StepPlan):
            outcome = {"outcome": ("planned", f"skipped: {sampled.cause}")}
            mismatched.append((reference.step, reference.seq_name, outcome))
            continue
        differences = {
            name: (getattr(reference, name), getattr(sampled, name))
            for name in WINDOW_FIELDS
            if getattr(reference, name) != getattr(sampled, name)
        }
        if differences:
            mismatched.append((reference.step, reference.seq_name, differences))
    return {"compared": compared, "mismatched": mismatched, "inexpressible": inexpressible}


# ----------------------------------------------------------------- identity ---


def sampler_identity(
    *,
    pool,
    index: PoolIndex,
    sampler_seed: int,
    view_count_weights: str,
    transform_probability: float,
    extra: ExtraPool | None = None,
) -> dict:
    """What, with the planner's own refused keys, decides every sampled step.

    Stored in the checkpoint's settings and refused on a resume that changes
    any of it; plain str/int/float so it round-trips through
    ``torch.load(weights_only=True)`` unchanged.  ``view_count_weights`` is the
    canonical string (:func:`format_view_count_weights`).  With an extra pool it
    also carries that pool's path, index hash and weight; without one it is the
    five keys it always was, so a checkpoint written before the second pool
    existed resumes against an identical dict.
    """

    identity = {
        "pool": str(os.fspath(pool)),
        "index_hash": str(index.content_hash),
        "sampler_seed": int(sampler_seed),
        "view_count_weights": str(view_count_weights),
        "transform_probability": float(transform_probability),
    }
    if extra is not None:
        identity["extra_pool"] = str(os.fspath(extra.data_root))
        identity["extra_pool_index_hash"] = str(extra.index.content_hash)
        identity["extra_pool_weight"] = float(extra.weight)
    return identity
