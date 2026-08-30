"""Write held-out predictions in the schema the cluster's scorers already read.

Nothing here scores anything.  ``evaluate_3dpt`` lives in MVTracker's environment,
not this one, so the trainer emits files and a companion job scores them.  That
split is why the schema matters more than usual: a file that looks scorable and
is not costs a whole eval cycle to discover.

The consumers, all in the cluster repository:

* ``score_official.py`` reads ``gt``, ``pred`` and ``gt_vis_any``, selects the
  points visible at frame 0 (``gt_vis_any[0]``), and runs upstream's
  ``compute_average_pts_within_thresh`` on that subset.
* ``score.py`` / ``score_joint.py`` pass ``evaluate_3dpt(gt, gt_vis_any, pred,
  ~occ, ...)`` -- note the scorer wants predicted **visibility**, so this file
  stores occlusion and the scorer inverts it.  Storing visibility here would
  silently invert every occlusion metric.
* ``score_curve.py`` guards on the five keys above and skips a scene that lacks
  any of them, which is load-bearing here -- see "no operating point" below.

``conf`` is the artifact; ``occ`` is one operating point on it
-------------------------------------------------------------

``occ`` used to be ``~gt_vis_any``: ground truth, inverted, written into the slot
reserved for a prediction.  Every AJ, OA and delta_avg figure produced from those
files was therefore scored with **oracle visibility**, and the track head's
confidence channel was read by no metric at all.

What this module writes now is the raw per-``(T,N)`` confidence, and *thresholding
is a scoring-side decision* -- swept over a grid downstream, not frozen into a
training run.  ``occ`` ships beside it as a single default operating point, at a
threshold recorded in the same file so a figure can never be separated from what
produced it.

``occ`` is now a **prediction** while ``gt_vis_any`` stays ground truth.  The two
are no longer complements and neither may be substituted for the other.

The reference threshold
-----------------------

::

    tau = confidence_alpha / OCCLUSION_DISTANCE_M

Minimising the confidence term ``conf * err - alpha * log(conf)`` over ``conf``
gives ``conf* = alpha / err`` (see :func:`arc.training.losses.resolve_confidence_alpha`),
so a point's *implied* error is ``alpha / conf``, and calling it occluded when that
implied error exceeds ``d`` metres is exactly ``conf <= alpha / d``.

A quantile rule -- "tau is the median confidence of the step" -- was considered and
rejected.  The ground-truth occlusion rate on this held-out set is 10.6%, so a
median threshold labels 50% of entries occluded by construction and scores worse
than a model that never predicts occlusion at all: a broken *threshold* that reads
as a broken *channel*.

**``confidence_alpha`` does not violate the no-ground-truth invariant, and must not
be "fixed" back out of this rule.**  The invariant is that nothing computed *from
the eval's ground truth* may enter the predicted visibility of an eval sample.
``confidence_alpha`` is a training-time constant, resolved once at the first
executed step and carried in the checkpoint -- a model parameter, the same as any
learned scale, reused unchanged at inference.  A per-step quantile over samples
selected by ground-truth visibility would not be.  The test of any future variant
is whether it could run on an unlabelled video.

**An inert default threshold on the first real run is expected, not a bug.**  The
one archived run that resolved an alpha did so at ``1.0624757247242371``; at
``d = 0.10`` that puts tau at 10.6, below the p05 near 36 that
``arc.training.diagnostics`` records for the channel.  So the default ``occ`` may
be all-visible and ``predicted_occluded_fraction`` at or near zero.  That is the
confidence channel being barely supervised, not this file misbehaving.  **Do not
tune ``d`` to make the number look better** -- it is a physical distance threshold,
and 0.10 aligns with the benchmark's own Jaccard thresholds.  Sweep tau over
``conf`` instead; that is what ``conf`` is stored for.

When there is no operating point
--------------------------------

``--confidence_alpha auto`` resolves only inside the confidence term, so a
``--confidence_weight 0`` run never resolves one.  Such a run writes
``confidence_alpha`` and ``tau`` as NaN and **omits ``occ`` entirely**.

Absence, not a sentinel.  An all-occluded array would pass every gate the cluster
scorers have -- ``score_curve.py`` would score it and report OA near 10.6% beside
real numbers, which is the same silently-wrong number this module exists to stop
producing.  With the key absent, that guard skips the scene with a message and any
unguarded ``z["occ"]`` raises ``KeyError`` naming the key.

Recomputing ``occ`` downstream
------------------------------

``occ`` is a total function of two arrays in the same file::

    occ = ~(conf >= tau)

Everything is float32 -- ``conf``, ``tau``, ``confidence_alpha`` and
``tau_distance_m`` -- so this reproduces the stored ``occ`` bit for bit whichever
dtype a reader promotes to.  A float64 threshold against a float32 confidence does
not: the two disagree within one ulp, and NEP 50 weak promotion gives a third
answer again.  ``tau == confidence_alpha / tau_distance_m`` holds exactly on
reload for the same reason.

Two details that are easy to get wrong and impossible to notice afterwards:

* ``query_points[:, 0]`` is the index into the **covered timesteps**, not the
  original frame number.  A window that trains on times ``(0, 2, 4)`` writes
  ``0, 1, 2`` here.  Writing the original times instead shifts every query onto
  the wrong row of ``gt``.
* ``occ`` and ``conf`` are per (timestep, track) and must cover every covered
  timestep, not only the supervised ones -- the scorer indexes them positionally
  against ``gt``.  A timestep's cameras are reduced into ``conf`` with **max**,
  mirroring the ``any`` that reduces them into ``gt_vis_any`` and matching the
  cluster scorer's own ``~(max over cameras of conf > tau)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# The exact key set both scorers expect, then ours. Named once so the writer and
# its test assert against one definition rather than two hand-copied lists.
PREDICTION_KEYS = (
    "pred",
    "gt",
    "occ",
    "query_points",
    "gt_vis_any",
    "conf",
    "tau",
    "confidence_alpha",
    "tau_distance_m",
)

# What a run with no resolved `confidence_alpha` writes: everything but `occ`,
# which such a run has no threshold to compute. Absence rather than a sentinel --
# an unguarded `z["occ"]` in the cluster's scorers must raise rather than score an
# all-occluded array.
UNTHRESHOLDED_PREDICTION_KEYS = tuple(key for key in PREDICTION_KEYS if key != "occ")

# What bundles written before the confidence contract carry. Their `occ` is
# `~gt_vis_any` -- oracle visibility -- and no post-hoc fix recovers a prediction
# that was never made.
LEGACY_PREDICTION_KEYS = ("pred", "gt", "occ", "query_points", "gt_vis_any")

# The distance at which the confidence term's implied error `alpha/conf` is called
# an occlusion. A module constant and not a flag: sweeping tau over the stored
# `conf` is algebraically the same as sweeping this at fixed alpha, so a flag would
# be a second control surface for one quantity and a way for two runs to differ
# invisibly.
OCCLUSION_DISTANCE_M = 0.10


def _as_array(value, dtype: str | type) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _as_float_array(value, name: str) -> np.ndarray:
    """Cast to float32, but **refuse** a non-floating input rather than convert it.

    ``_as_array`` casts unconditionally, which is right for the arrays whose dtype
    carries no information.  It is wrong for confidence: a boolean visibility mask
    handed in here would become 1.0/0.0, threshold to something, and reproduce the
    ground-truth ``occ`` this module exists to stop writing -- the original defect
    wearing the new API.
    """

    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if not value.dtype.is_floating_point:
            raise ValueError(f"{name} must be a floating tensor, got {value.dtype}")
        value = value.float()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must have a floating dtype, got {array.dtype}")
    return array.astype(np.float32, copy=False)


def _alpha_array(confidence_alpha) -> np.float32:
    """The run's confidence alpha as the float32 the bundle stores, or NaN."""

    if confidence_alpha is None:
        return np.float32(np.nan)
    array = _as_float_array(confidence_alpha, "confidence_alpha")
    if array.ndim != 0:
        raise ValueError(
            "confidence_alpha must be a scalar; one threshold belongs to a whole "
            f"run, not to a scene or a timestep, got shape {array.shape}"
        )
    return np.float32(array)


def reference_tau(confidence_alpha) -> np.float32:
    """The confidence at which the term's implied error ``alpha/conf`` reaches ``d``.

    NaN when the run resolved no alpha: ``--confidence_alpha auto`` resolves only
    inside the confidence term, so a ``--confidence_weight 0`` run never has one.
    A NaN here is the honest statement that the run has no reference operating
    point; ``conf`` is still the artifact and is swept downstream.

    Computed in float32 so ``tau == confidence_alpha / tau_distance_m`` holds
    exactly for a reader that reloads the two stored scalars.
    """

    alpha = _alpha_array(confidence_alpha)
    return np.float32(alpha / np.float32(OCCLUSION_DISTANCE_M))


def _occluded_from_confidence(conf: np.ndarray, tau: np.float32) -> np.ndarray:
    """``occ = ~(conf >= tau)``.  Recomputable verbatim from the two stored arrays.

    The negation of the visible side rather than ``conf < tau`` so a NaN confidence
    lands **occluded**; on finite confidence the two are identical, tie included
    (``conf == tau`` is visible).  ``expp1`` is ``1 + exp(x)`` and overflows to
    ``+inf`` in BF16, and an infinite confidence is a real, enormous one -- it
    stays visible.  A NaN is the absence of a call, and crediting it as a confident
    visible call would manufacture a positive.
    """

    return ~(conf >= tau)


def _validate_shapes(
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    conf: np.ndarray,
    queries: np.ndarray,
    visible: np.ndarray,
    occ: np.ndarray | None,
) -> None:
    """The shape contract, shared by the writer's builder and the reader."""

    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"pred must have shape (T,N,3), got {pred.shape}")
    if gt.shape != pred.shape:
        raise ValueError(f"gt {gt.shape} must match pred {pred.shape}")
    time_count, track_count = pred.shape[:2]
    # Checked even though N == the correspondence count: `conf` arrives from
    # `gather_at_correspondences` as (M,S), camera-major over cameras x times, and
    # must be fused first. An unfused conf whose S happened to equal T would
    # transpose the threshold onto the wrong points and still write a valid file.
    if conf.shape != (time_count, track_count):
        raise ValueError(
            f"conf must have shape {(time_count, track_count)}, got {conf.shape}"
        )
    if occ is not None and occ.shape != (time_count, track_count):
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


def build_prediction_arrays(
    *,
    predicted_positions,
    ground_truth_positions,
    confidence,
    confidence_alpha,
    query_points,
    visible_any_camera,
) -> dict[str, np.ndarray]:
    """Validate and cast one scene's prediction bundle.

    Takes the model's confidence channel and the run's ``confidence_alpha``, and
    derives ``tau``, ``tau_distance_m`` and ``occ`` here rather than accepting an
    ``occluded`` array from the caller.  The defect this replaces *was* a caller
    computing ``occ`` from something other than what the file claimed; deriving
    every one of them from a single input makes "``occ`` disagrees with its own
    threshold" unrepresentable, and makes "is there a threshold at all" one
    decision rather than four.

    Split from the writer so the contract can be tested without touching a
    filesystem, and so a caller assembling the arrays gets the error at the point
    it can still fix them.
    """

    pred = _as_array(predicted_positions, np.float32)
    gt = _as_array(ground_truth_positions, np.float32)
    conf = _as_float_array(confidence, "conf")
    queries = _as_array(query_points, np.float32)
    visible = _as_array(visible_any_camera, bool)

    alpha = _alpha_array(confidence_alpha)
    tau = reference_tau(alpha)

    _validate_shapes(
        pred=pred, gt=gt, conf=conf, queries=queries, visible=visible, occ=None
    )

    arrays = {
        "pred": pred,
        "gt": gt,
        "query_points": queries,
        "gt_vis_any": visible,
        "conf": conf,
        "tau": tau,
        "confidence_alpha": alpha,
        "tau_distance_m": np.float32(OCCLUSION_DISTANCE_M),
    }
    if np.isfinite(tau):
        arrays["occ"] = _occluded_from_confidence(conf, tau)
    # Otherwise omitted, deliberately: a run that resolved no `confidence_alpha`
    # has no reference operating point, and absence is the only representation of
    # that which a scorer cannot mistake for a prediction. See the module
    # docstring. Spelled as a branch rather than left to `~(conf >= nan)`, which
    # happens to be True everywhere -- load-bearing behaviour must not depend on
    # NaN comparison semantics a reader has to know to see.
    return arrays


def _expected_keys(arrays) -> tuple[str, ...]:
    """``occ`` is expected exactly when the bundle has a finite reference threshold."""

    tau = arrays.get("tau")
    finite = tau is not None and np.ndim(tau) == 0 and bool(np.isfinite(tau))
    return PREDICTION_KEYS if finite else UNTHRESHOLDED_PREDICTION_KEYS


def write_scene_predictions(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write one scene's bundle, compressed, as the scorers expect to find it.

    The expected key set is derived from the bundle's own ``tau``, so the symmetric
    check below runs in both directions: a bundle with a finite threshold and no
    ``occ`` is refused as missing one, and -- the case that matters -- an ``occ``
    handed in beside a NaN ``tau`` is refused as **unexpected**.  An all-occluded
    array therefore cannot reach disk by any route, not merely by the intended one.
    """

    expected = set(_expected_keys(arrays))
    missing = expected - set(arrays)
    unexpected = set(arrays) - expected
    if missing or unexpected:
        raise ValueError(
            f"prediction bundle key mismatch; missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def read_scene_predictions(
    path: str | Path, *, allow_unscorable: bool = False
) -> dict[str, np.ndarray]:
    """Load one bundle, refusing the two kinds that cannot be scored as they stand.

    Both refusals are on by default and lifted by the same ``allow_unscorable``
    opt-in, because both are about ``occ``: a consumer that sweeps ``tau`` over
    ``conf`` reads neither, and says so by passing the flag.

    * **No ``conf``** -- the bundle predates the confidence contract.  Its ``occ``
      is ``~gt_vis_any``, i.e. oracle visibility, so its OA and AJ are
      oracle-occlusion figures and belong in no table beside a real one.
    * **No ``occ``** -- the run resolved no ``confidence_alpha``, so there is no
      threshold at which to call occlusion.

    Refusing rather than returning a flag is the point.  The failure being fixed is
    a silently-wrong number nobody had reason to look at, and an advisory flag a
    caller forgets to check reproduces it exactly.
    """

    path = Path(path)
    with np.load(path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    keys = set(arrays)

    if "conf" not in keys:
        if not allow_unscorable:
            raise ValueError(
                f"{path} predates the confidence contract: it carries no `conf`, and "
                "its `occ` is `~gt_vis_any` -- oracle visibility, not a prediction. "
                "Any OA, AJ or delta_avg computed from it is an oracle-occlusion "
                "figure. Re-run the eval; pass allow_unscorable=True to load it anyway"
            )
        return arrays

    missing = set(UNTHRESHOLDED_PREDICTION_KEYS) - keys
    unexpected = keys - set(PREDICTION_KEYS)
    if missing or unexpected:
        raise ValueError(
            f"{path} prediction bundle key mismatch; missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )

    # Before either early return below, and deliberately. The caller that opts into
    # an unthresholded bundle is the tau-sweeping one: it reads `conf`, `pred`, `gt`
    # and `gt_vis_any` and indexes them against each other, so it is the caller that
    # most needs the shape contract enforced -- checking it only on the thresholded
    # path would be exactly backwards.
    _validate_shapes(
        pred=arrays["pred"],
        gt=arrays["gt"],
        conf=arrays["conf"],
        queries=arrays["query_points"],
        visible=arrays["gt_vis_any"],
        occ=arrays.get("occ"),
    )

    tau = arrays["tau"]
    if "occ" not in keys:
        if not np.isfinite(tau):
            if not allow_unscorable:
                raise ValueError(
                    f"{path} has no reference operating point: its run resolved no "
                    "`confidence_alpha`, so there is no threshold at which to call "
                    "occlusion and no `occ` was written. Sweep `tau` over `conf` "
                    "instead; pass allow_unscorable=True to load it as it is"
                )
            return arrays
        raise ValueError(
            f"{path} carries a finite tau {float(tau)} but no `occ`; the two are "
            "written together or not at all"
        )
    if not np.isfinite(tau):
        raise ValueError(
            f"{path} carries an `occ` beside a non-finite tau, so its occlusion call "
            "has no threshold behind it"
        )

    # The stored occlusion must still be the one its own threshold implies. Cheap,
    # and it is the only check that would catch a hand-edited or half-migrated file
    # whose `occ` and `tau` have drifted apart.
    implied = _occluded_from_confidence(arrays["conf"], tau)
    if not np.array_equal(arrays["occ"], implied):
        raise ValueError(
            f"{path} stores an `occ` that is not `~(conf >= tau)` at its own recorded "
            f"tau {float(tau)}; the figure and the threshold have been separated"
        )
    return arrays
