"""Write held-out predictions in the schema the cluster's scorers already read.

Nothing here scores anything.  ``evaluate_3dpt`` lives in MVTracker's environment,
not this one, so the trainer emits files and a companion job scores them.  That
split is why the schema matters more than usual: a file that looks scorable and
is not costs a whole eval cycle to discover.

The consumers, both in the cluster repository:

* ``score_official.py`` reads ``gt``, ``pred`` and ``gt_vis_any``, selects the
  points visible at frame 0 (``gt_vis_any[0]``), and runs upstream's
  ``compute_average_pts_within_thresh`` on that subset.
* ``score.py`` / ``score_joint.py`` write this same schema and pass
  ``evaluate_3dpt(gt, gt_vis_any, pred, ~occ, ...)`` -- note the scorer wants
  predicted **visibility**, so this file stores occlusion and the scorer inverts
  it.  Storing visibility here would silently invert every occlusion metric.

Two details that are easy to get wrong and impossible to notice afterwards:

* ``query_points[:, 0]`` is the index into the **covered timesteps**, not the
  original frame number.  A window that trains on times ``(0, 2, 4)`` writes
  ``0, 1, 2`` here.  Writing the original times instead shifts every query onto
  the wrong row of ``gt``.
* ``occ`` is per (timestep, track) and must cover every covered timestep, not
  only the supervised ones -- the scorer indexes it positionally against ``gt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# The exact key set both scorers expect. Named once so the writer and its test
# assert against one definition rather than two hand-copied lists.
PREDICTION_KEYS = ("pred", "gt", "occ", "query_points", "gt_vis_any")


def _as_array(value, dtype, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def build_prediction_arrays(
    *,
    predicted_positions,
    ground_truth_positions,
    occluded,
    query_points,
    visible_any_camera,
) -> dict[str, np.ndarray]:
    """Validate and cast one scene's prediction bundle.

    Split from the writer so the shape contract can be tested without touching a
    filesystem, and so a caller assembling the arrays gets the error at the point
    it can still fix them.
    """

    pred = _as_array(predicted_positions, np.float32, "pred")
    gt = _as_array(ground_truth_positions, np.float32, "gt")
    occ = _as_array(occluded, bool, "occ")
    queries = _as_array(query_points, np.float32, "query_points")
    visible = _as_array(visible_any_camera, bool, "gt_vis_any")

    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"pred must have shape (T,N,3), got {pred.shape}")
    if gt.shape != pred.shape:
        raise ValueError(f"gt {gt.shape} must match pred {pred.shape}")
    time_count, track_count = pred.shape[:2]
    if occ.shape != (time_count, track_count):
        raise ValueError(
            f"occ must have shape {(time_count, track_count)}, got {occ.shape}"
        )
    if visible.shape != (time_count, track_count):
        raise ValueError(
            f"gt_vis_any must have shape {(time_count, track_count)}, got {visible.shape}"
        )
    if queries.ndim != 2 or queries.shape != (track_count, 4):
        raise ValueError(
            f"query_points must have shape {(track_count, 4)}, got {queries.shape}"
        )
    # The covered-timestep convention, checked rather than trusted: a caller that
    # wrote original frame numbers here would otherwise produce a file that scores
    # every query against the wrong row of gt, with no error anywhere.
    times = queries[:, 0]
    if times.size and (times.min() < 0 or times.max() >= time_count):
        raise ValueError(
            f"query_points[:,0] must index the {time_count} covered timesteps, got "
            f"[{times.min()}, {times.max()}]. These are positions in the selected "
            "window, not original frame numbers"
        )
    if not np.allclose(times, np.rint(times), atol=1e-6):
        raise ValueError("query_points[:,0] must hold integer timestep indices")

    return {
        "pred": pred,
        "gt": gt,
        "occ": occ,
        "query_points": queries,
        "gt_vis_any": visible,
    }


def write_scene_predictions(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write one scene's bundle, compressed, as the scorers expect to find it."""

    missing = set(PREDICTION_KEYS) - set(arrays)
    unexpected = set(arrays) - set(PREDICTION_KEYS)
    if missing or unexpected:
        raise ValueError(
            f"prediction bundle key mismatch; missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
