# VENDORED, DO NOT EDIT. Copy of mvtracker/utils/sample_manifest.py, which is the single
# definition of the manifest format and is stdlib-only precisely so a consuming repo can vendor it.
#
#   upstream : mvtracker @ 4002795 ("fix: correct the manifest replay spec for packed pools
#              and refuse a multi-GPU manifest")
#   sha256   : 456d15fa80ce4a3051daec957d077ee2de2fcf4490e89908248c5a292e517d08
#
# The body below is byte-identical to upstream; only this header is local, so the hash above is
# taken over the file WITHOUT it. `tests/test_manifest_plan.py` recomputes it, which is what turns
# "we vendored it" into a checkable claim rather than a comment. A schema change upstream must be
# re-vendored, not patched here -- two editable copies is how a reader and a writer drift apart
# with no test noticing, which the upstream module's own docstring says in as many words.
"""Per-step sample manifest: a JSONL record of exactly which training samples a run consumed.

Purpose
-------
An MVTracker training run appends one line per training step describing what its dataloader drew.
A run of a different architecture replays that record instead of re-implementing the sampler, so
both runs consume the same scenes, cameras, frame windows and query points in the same order, and
their learning curves are comparable step for step. Identity between the two runs then becomes a
property you can ``diff`` rather than one you have to argue for.

This module is the single definition of the format. It is stdlib-only and imports nothing from
``mvtracker``, so the consuming repo can vendor it verbatim.

File format
-----------
UTF-8 JSONL: one JSON object per line, ``\\n``-terminated, appended in step order. Lines are
canonical -- ``json.dumps(record, sort_keys=True, separators=(",", ":"))`` -- so two manifests
written by two independent implementations can be compared byte for byte.

The writer performs exactly one ``write`` plus one ``flush`` per record, so a killed job leaves a
valid *prefix*: the last step may be missing, but no line is half-written. Nothing is ``fsync``-ed,
so a machine crash (as opposed to a process kill) may lose recently flushed lines.

Record schema (v1)
------------------
Every record carries every key; a value the producer could not observe is ``null``.

manifest_version  int   schema version, currently 1.
step              int   the trainer's ``total_steps`` for this optimizer step, 0-based. The same
                        counter the checkpoints and ``step-<N>_metrics.csv`` use, so a manifest line
                        joins to an eval row by step. A draw that fails (``not all(gotit)``) never
                        reaches the model and never advances the counter, so it gets no line: line k
                        is training step k. NOT necessarily the line number, though -- a resumed run
                        appends from its restored step, so steps can repeat. Build a step -> record
                        map with last-wins if you need one.
rank              int   global rank of the writing process. Only rank 0 writes, so this is 0.
batch_index       int   position of this sample within its batch, 0-based.
batch_size        int   samples in that batch. It is 1 in this project, so there is then exactly one
                        line per step; at B > 1 a step contributes B lines with equal ``step``.
data_root         str   directory the drawing dataset reads its scenes from; ``seq_name`` names a
                        subdirectory of it. Recorded exactly as that dataset was configured, so it
                        may be relative to the run's working directory. It is per row, not per run:
                        one run writes rows drawn from more than one dataset (see below), and
                        resolving every ``seq_name`` against a single root would load the wrong
                        scenes wherever two roots share a scene name.
seq_name          str   scene name (Kubric: "0731"), relative to ``data_root``. The scene is EITHER
                        the directory ``<data_root>/<seq_name>/`` OR the archive
                        ``<data_root>/<seq_name>.zip`` -- an uncompressed (ZIP_STORED) archive of
                        that same tree, members at the archive root. The suffix is deliberately not
                        recorded, so a packed run and a loose one produce comparable manifests; a
                        replaying run must accept both layouts.
sample_index      int   the VIRTUAL dataset index the sampler produced, before the dataset's
                        ``index % real_len`` wrap. Under a ``ConcatDataset`` (the static-pretraining
                        loader) it is the index within the member that drew it, so values repeat
                        across members; read it with that row's ``data_root`` and ``real_len``.
real_len          int   number of distinct scenes in the drawing dataset, i.e. the wrap period.
dataset_seed      int   the drawing dataset's configured ``seed``. ``null`` for an unseeded dataset,
                        which is what the static-pretraining datasets are.
sample_seed       int   the seed this sample was actually drawn with, as reported by the loader.
                        For a seeded dataset it equals ``dataset_seed + sample_index % real_len``;
                        for an unseeded one it was drawn at load time and this is its only record.
views             list[int]  camera ids returned, in the order they appear along the view axis of
                        ``video`` / ``videodepth`` / ``intrs`` / ``extrs``. The length varies
                        between steps when variable-num-views augmentation is enabled.
frame_start       int   index of the first returned frame inside the scene's full clip; 0 when no
                        temporal crop was applied.
seq_len           int   number of frames in the returned window.
track_indices     list[int]  rows of the scene's FULL track array (Kubric: ``tracks_3d.npz``), in
                        the order the returned trajectories appear along the track axis. Element k
                        is the scene-level id of returned track k, so a replaying run can select the
                        same physical points in the same order. Ids MAY REPEAT: the loader draws
                        independently from a "dynamic" and a "very dynamic" pool, the second a
                        subset of the first whenever ``ratio_very_dynamic > 0``, which the training
                        dataset sets. A repeat is a second, independent column with its own query
                        timestep, not a redundant row, so index positionally and never key tracks by
                        id -- a dict or set keyed by id silently drops query points.
traj_per_sample   int   ``len(track_indices)``: how many tracks were actually returned, after every
                        augmentation and clamp. This, not the configured value, matches the tensors.
traj_per_sample_configured  int  the drawing dataset's configured ``traj_per_sample``; context only.
depth_type        str   "gt" | "duster" | "duster_cleaned": which depth source this sample used.
augmented         bool  whether the sample took the augmented branch. It gates the photometric, depth
                        and traj-per-sample augmentations only -- NOT the crop, the scene transform or
                        the camera noise, which run on every sample.
scene_transform   dict  the world-space similarity the loader applied to ``videodepth`` / ``extrs`` /
                        ``trajectory_3d`` / ``query_points_3d``: ``{"scale", "rot_x_deg",
                        "rot_y_deg"}``, rotations applied x then y, no translation. ``null`` when
                        ``augmentations.scene_transform`` is off. A replaying run must apply this to
                        the scene file's coordinates or it trains on different geometry.

Reproducing a recorded draw
---------------------------
For each record, in file order:

1. load scene ``seq_name`` under ``data_root`` -- directory or ``.zip``, see above;
2. take cameras ``views``, in the listed order;
3. take frames ``[frame_start, frame_start + seq_len)``;
4. take tracks ``full_tracks[:, track_indices]``, in the listed order;
5. use depth source ``depth_type``;
6. apply ``scene_transform`` to the resulting geometry.

There is no ``gotit`` key. A draw that fails is dropped whole before it reaches the model, so it never
produces a record -- the field could only ever read true, and its absence is the schema, not an
omission.

Rows from an initial static-pretraining epoch are ordinary, replayable rows and are NOT nulls: that
epoch draws from four other datasets, so those rows carry their own ``data_root`` and ``real_len``
and a ``null`` ``dataset_seed`` (they are unseeded) instead of the training dataset's. Select or
exclude them by ``data_root``; nothing else in the record marks them.

A ``null`` ``data_root`` means the batch carries no provenance stamp at all: it came from a dataset
that does not record one, or from a ``Datapoint`` pickled before this format existed. ``real_len``,
``dataset_seed`` and ``traj_per_sample_configured`` are null on exactly those rows, the four being
one stamp, and the loader observations (``sample_index``, ``views``, ``frame_start``,
``track_indices``) are null with them. Such a row cannot be replayed; the replaying run must decide
whether to skip the step or abort.

``modes.debugging_hotfix_datapoint_path`` replaces the drawn batch with one loaded from disk, so the
row describes the loaded batch rather than the draw. A ``crash_batch_*.pt`` written by a run that
already carried these fields yields a fully populated row -- the provenance of the crashing step,
repeated identically for the rest of the run; a file predating them yields the all-null row above.

What this record does and does not determine
--------------------------------------------
``sample_index`` plus the dataset config does NOT reproduce the sample. It reproduces the
*selection* of scene, cameras and frame window, and nothing beyond that:

* Periodic in the scene index, because the per-sample RNG seed is derived from the POST-wrap index
  (``__getitem__`` wraps before seeding). ``sample_index`` and ``sample_index + real_len`` therefore
  select the same scene, the same ``views``, the same ``frame_start``, the same ``depth_type`` and
  the same ``augmented`` flag. The stream repeats those with period ``real_len``.
* NOT reproducible: ``track_indices`` and ``traj_per_sample``. Kubric's cropping augmentation draws
  its pad / resize / crop-offset from the GLOBAL ``np.random`` stream, not from the per-sample
  generator, and it then masks trajectory visibility to the crop bounds. That post-crop visibility
  is what the track sub-selection is drawn from, so the returned query points vary between two
  samples at the same seed. Under the training config (``augmentations.cropping: true``) the query
  points come from ``track_indices`` in this file or from nowhere.
* NOT recorded: the crop, the photometric augs, the depth augs and the depth noise. Their RNG
  sources differ, and a consumer that wants to neutralise them has to know which is which: the crop
  and the depth noise draw from the GLOBAL ``np.random``, the depth augs from the per-sample
  generators only, and the photometric augs from the per-sample ``rnd_np`` except for their
  ColorJitter and GaussianBlur parameters, which come from the global torch RNG.
* Recorded as of this version, and it moves the LABELS rather than the pixels: ``scene_transform``
  above. ``augmentations.camera_params_noise`` is NOT recorded and perturbs ``intrs`` / ``extrs``
  again afterwards -- Gaussian at std 0.001, drawn from the per-sample generator. It is small beside
  the transform, but two runs that need bit-exact cameras must both set it false.

Consequence for a replaying run: read ``track_indices`` from the file, do not attempt to recompute
it, and do not expect a re-run of MVTracker at the same seed to regenerate an identical manifest.
"""

import json
import os
from typing import Any, Dict, List, TextIO, Union

# Stays 1: this is the first version anything has consumed, so the schema additions since the format
# was drafted (``data_root``, ``scene_transform``) break no existing reader. Bump it the first time a
# key changes meaning or disappears after a consumer exists.
MANIFEST_VERSION = 1

# The record's key set, in schema order. Exported so the writer in ``mvtracker/cli/train.py`` and the
# tests assert against ONE definition -- a hand-copied second copy is how a writer and its schema
# drift apart without any test noticing.
MANIFEST_RECORD_KEYS = (
    "manifest_version",
    "step",
    "rank",
    "batch_index",
    "batch_size",
    "data_root",
    "seq_name",
    "sample_index",
    "real_len",
    "dataset_seed",
    "sample_seed",
    "views",
    "frame_start",
    "seq_len",
    "track_indices",
    "traj_per_sample",
    "traj_per_sample_configured",
    "depth_type",
    "augmented",
    "scene_transform",
)

# One append-mode handle per absolute path, opened lazily and kept for the process lifetime.
_OPEN_HANDLES: Dict[str, TextIO] = {}


def _jsonable(value: Any) -> Any:
    """Fallback encoder for numpy / torch scalars that leak out of a dataloader.

    Anything exposing ``tolist`` (ndarray, tensor) or ``item`` (numpy scalar, 0-d tensor) is
    converted to its Python equivalent; anything else raises, so a genuinely unserializable value is
    a loud bug rather than a silent ``str()``.

    Args:
        value: the object ``json.dumps`` could not encode.

    Returns:
        A JSON-encodable equivalent.

    Raises:
        TypeError: if the value has no obvious Python equivalent.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON-serializable in a manifest record: {type(value).__name__}")


def encode_manifest_line(record: Dict[str, Any]) -> str:
    """Encode one record in its canonical single-line JSON form, without the trailing newline.

    Keys are sorted and separators are tight, so two producers that agree on the values produce
    byte-identical lines.

    Args:
        record: the JSON object for one sample; see the module docstring for the schema.

    Returns:
        The canonical encoding of ``record``.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=_jsonable)


def write_manifest_line(path: Union[str, "os.PathLike[str]"], record: Dict[str, Any]) -> None:
    """Append one record to the JSONL manifest at ``path``.

    The file is opened in append mode on the first call for a given absolute path and the handle is
    then reused for the lifetime of the process. There is deliberately no ``close``: the training
    loop this serves has ``raise`` paths and a ``sys.exit`` from a signal handler, so there is no
    single place to close it, and the per-record ``flush`` already gives the only property closing
    would -- the file on disk is always a valid prefix. The interpreter closes the handle at exit.

    Args:
        path: manifest location, e.g. ``<experiment_path>/manifest.jsonl``. Missing parent
            directories are created.
        record: the JSON object for one sample; see the module docstring for the schema.
    """
    key = os.path.abspath(os.fspath(path))
    handle = _OPEN_HANDLES.get(key)
    if handle is None:
        parent = os.path.dirname(key)
        if parent:
            os.makedirs(parent, exist_ok=True)
        handle = open(key, "a", encoding="utf-8")
        _OPEN_HANDLES[key] = handle
    handle.write(encode_manifest_line(record) + "\n")
    handle.flush()


def entry_for_sample(field: Any, batch_index: int) -> Any:
    """Read entry ``batch_index`` of a collated Datapoint sample-manifest field.

    ``collate_fn`` carries each ``sample_*`` field as a plain list of length B. This returns None for
    anything that is not a long-enough list -- a Datapoint pickled before those fields existed, an
    externally injected batch, or a dataset that does not record them -- so the caller building a
    record never has to special-case those.

    Args:
        field: the collated field, normally a list of length B.
        batch_index: position of the sample within its batch.

    Returns:
        The per-sample value, or None if it was not recorded.
    """
    if isinstance(field, list) and batch_index < len(field):
        return field[batch_index]
    return None


def read_manifest(path: Union[str, "os.PathLike[str]"]) -> List[Dict[str, Any]]:
    """Read a JSONL manifest back into a list of records, in file order.

    Blank lines are skipped. A malformed line raises ``ValueError`` naming the 1-based line number:
    the writer's write-then-flush contract means a killed job truncates *between* lines, so a
    malformed line is real corruption and not something to paper over.

    Args:
        path: manifest location.

    Returns:
        One dict per record, in the order written. A resumed run appends, so ``step`` may repeat.

    Raises:
        ValueError: if a non-blank line is not valid JSON.
    """
    records: List[Dict[str, Any]] = []
    with open(os.fspath(path), "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{os.fspath(path)}:{line_number}: malformed manifest line: {exc}") from exc
    return records
