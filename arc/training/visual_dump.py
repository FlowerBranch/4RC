"""One held-out scene as the eval saw it, written for the multi-view viewer.

``evaluate_held_out`` writes one of these per scene named by
``--eval_visual_scenes``, at ``eval/step-<N>/visual/<scene>.npz``, from the
plain arm only, and ``arc.viz.viser_multiview_eval`` reads it back.  Nothing
here scores anything and nothing reads it during training: it is the picture
behind the two readouts every arm is judged on -- the camera baseline
(``base_ratio``) and the query anchor's error -- next to the tracks that were
scored.

The arrays come from :func:`arc.training.sparse_tracking.visual_geometry`
(the reconstruction, both camera sets, the model's own anchors) and from the
scene's prediction bundle (the scored tracks).  This module only fixes their
names, shapes and dtypes, and checks them on the way out and on the way in, as
:mod:`arc.training.predictions` does for the bundles.

Frame and units
---------------

The stored world frame, every position lifted to metres by the scene's track
upscaling factor -- the frame and units the bundles' ``pred`` and ``gt`` use,
so the ``track_*`` arrays overlay the clouds as written.  Rotations are
camera-to-world, and both camera sets' intrinsics are on the model's ``H x W``
grid (``image_size``).

Shapes
------

``S`` observation slots (camera-major), an ``h x w`` grid of pixels
(``pixel_rows`` by ``pixel_columns``, a fixed stride of the model grid), ``M``
correspondence rows, ``T`` covered timesteps and ``N`` tracks.  ``T`` and ``N``
are the bundle's own, so ``track_query_points[:, 0]`` and ``slot_time_index``
index the same covered-timestep axis.

``depth_conf`` is present exactly when the model emitted one.  ``metadata`` is
JSON text: the run's provenance, the head settings the eval ran, and the
scene's ``base_ratio`` and ``anchor_error_m`` readouts verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Key -> (dtype, shape). A string in a shape is a size resolved per file, which
# every key naming it must agree on; an integer is literal. One table, read by
# the builder and the reader alike, so the two cannot drift apart.
_SCHEMA: dict[str, tuple[type, tuple]] = {
    "slot_camera_id": (np.int64, ("S",)),
    "slot_original_time": (np.int64, ("S",)),
    "slot_time_index": (np.int64, ("S",)),
    "image_size": (np.int64, (2,)),
    "pixel_rows": (np.int64, ("h",)),
    "pixel_columns": (np.int64, ("w",)),
    "pred_points_m": (np.float32, ("S", "h", "w", 3)),
    "rgb": (np.uint8, ("S", "h", "w", 3)),
    "depth_conf": (np.float32, ("S", "h", "w")),
    "gt_points_m": (np.float32, ("S", "h", "w", 3)),
    "gt_points_valid": (np.bool_, ("S", "h", "w")),
    "pred_camera_rotation": (np.float32, ("S", 3, 3)),
    "pred_camera_center_m": (np.float32, ("S", 3)),
    "pred_intrinsics": (np.float32, ("S", 3, 3)),
    "gt_camera_rotation": (np.float32, ("S", 3, 3)),
    "gt_camera_center_m": (np.float32, ("S", 3)),
    "gt_intrinsics": (np.float32, ("S", 3, 3)),
    "anchor_pred_m": (np.float32, ("M", 3)),
    "anchor_true_m": (np.float32, ("M", 3)),
    "anchor_key": (np.str_, ("M",)),
    "track_pred": (np.float32, ("T", "N", 3)),
    "track_gt": (np.float32, ("T", "N", 3)),
    "track_gt_vis_any": (np.bool_, ("T", "N")),
    "track_query_points": (np.float32, ("N", 4)),
    "track_conf": (np.float32, ("T", "N")),
    "metadata": (np.str_, ()),
}

VISUAL_KEYS = tuple(_SCHEMA)

# Absent rather than a sentinel when the model emitted no depth confidence, as
# fit_scene_sim3 reads that channel.
OPTIONAL_VISUAL_KEYS = ("depth_conf",)

# The bundle keys a dump carries, each under ``track_<key>``: what was scored.
VISUAL_TRACK_KEYS = ("pred", "gt", "gt_vis_any", "query_points", "conf")

# The metadata object's keys, exactly: the run's provenance, the head settings
# the eval ran, and the scene's two readouts. The viewer reads every one.
VISUAL_METADATA_KEYS = (
    "scene",
    "step",
    "output_dir",
    "checkpoint_dir",
    "commit",
    "merge_synchronized_slots",
    "refine_iters",
    "depth_input",
    "camera_input",
    "oracle_query_anchor",
    "ground_truth_query_anchor",
    "query_anchors",
    "base_ratio",
    "anchor_error_m",
)


def _validate_visual_arrays(arrays: dict) -> None:
    """The schema above, plus the cross-checks a shape match cannot make."""

    keys = set(arrays)
    missing = set(VISUAL_KEYS) - set(OPTIONAL_VISUAL_KEYS) - keys
    unexpected = keys - set(VISUAL_KEYS)
    if missing or unexpected:
        raise ValueError(
            f"visual dump key mismatch; missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )

    sizes: dict[str, int] = {}
    for key, (dtype, shape) in _SCHEMA.items():
        if key not in arrays:
            continue
        array = arrays[key]
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{key} must be a numpy array, got {type(array).__name__}")
        if dtype is np.str_:
            if array.dtype.kind != "U":
                raise ValueError(f"{key} must hold strings, got {array.dtype}")
        elif array.dtype != np.dtype(dtype):
            raise ValueError(f"{key} must be {np.dtype(dtype)}, got {array.dtype}")
        if array.ndim != len(shape):
            raise ValueError(
                f"{key} must have {len(shape)} axes {shape}, got {array.shape}"
            )
        for axis, (expected, size) in enumerate(zip(shape, array.shape)):
            if isinstance(expected, int):
                if size != expected:
                    raise ValueError(
                        f"{key} axis {axis} must be {expected}, got {array.shape}"
                    )
            elif sizes.setdefault(expected, size) != size:
                raise ValueError(
                    f"{key} axis {axis} is {size}, but {expected} is "
                    f"{sizes[expected]} elsewhere in the dump"
                )

    height, width = (int(value) for value in arrays["image_size"])
    for key, limit in (("pixel_rows", height), ("pixel_columns", width)):
        pixels = arrays[key]
        if pixels.size and (
            pixels.min() < 0 or pixels.max() >= limit or np.any(np.diff(pixels) <= 0)
        ):
            raise ValueError(
                f"{key} must be strictly increasing pixel indices below {limit}"
            )
    time_indices = arrays["slot_time_index"]
    if time_indices.size and (
        time_indices.min() < 0 or time_indices.max() >= sizes["T"]
    ):
        raise ValueError(
            f"slot_time_index must index the {sizes['T']} covered timesteps the "
            f"tracks carry, got [{time_indices.min()}, {time_indices.max()}]"
        )
    metadata = json.loads(arrays["metadata"].item())
    if not isinstance(metadata, dict):
        raise ValueError(
            f"metadata must be a JSON object, got {type(metadata).__name__}"
        )
    if set(metadata) != set(VISUAL_METADATA_KEYS):
        raise ValueError(
            "metadata key mismatch; missing "
            f"{sorted(set(VISUAL_METADATA_KEYS) - set(metadata))}, unexpected "
            f"{sorted(set(metadata) - set(VISUAL_METADATA_KEYS))}"
        )


def build_visual_arrays(*, geometry: dict, tracks: dict, metadata: dict) -> dict:
    """Assemble and validate one scene's dump.

    ``geometry`` is what :func:`arc.training.sparse_tracking.visual_geometry`
    returns.  ``tracks`` is the scene's prediction bundle as
    :func:`arc.training.predictions.build_prediction_arrays` built it; its
    :data:`VISUAL_TRACK_KEYS` are carried unchanged as ``track_<key>``, so the
    dump holds exactly what was scored.  ``metadata`` is serialised to JSON
    text.  Refuses rather than repairs: a key, shape or dtype that disagrees
    with the schema raises here, before anything reaches disk.
    """

    arrays = dict(geometry)
    for key in VISUAL_TRACK_KEYS:
        arrays[f"track_{key}"] = tracks[key]
    arrays["metadata"] = np.array(json.dumps(metadata))
    _validate_visual_arrays(arrays)
    return arrays


def write_visual_dump(path: str | Path, arrays: dict) -> Path:
    """Write one scene's dump, compressed; checked again on the way out."""

    _validate_visual_arrays(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def read_visual_dump(path: str | Path) -> tuple[dict, dict]:
    """Load one dump as ``(arrays, metadata)``, refusing one that breaks the schema."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    try:
        _validate_visual_arrays(arrays)
    except ValueError as error:
        raise ValueError(f"{path} is not a visual dump: {error}") from error
    return arrays, json.loads(arrays["metadata"].item())
