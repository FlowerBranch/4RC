#!/usr/bin/env python3
"""Check that a benchmark's clip set builds, scene for scene, what its MVTracker loader builds.

The held-out eval can score a benchmark two ways: ``--eval_benchmark_dataset``
opens it through MVTracker's loader, ``--eval_benchmark_clips`` reads the clip set
4I4's prepare_clips.py exported from that loader.  This script builds every scene
both ways -- :class:`arc.training.scene_provider.BenchmarkSceneProvider` and
:class:`arc.training.scene_provider.BenchmarkClipSceneProvider`, each opened by the
trainer's own ``_open_benchmark`` and planned by its ``_benchmark_plans`` from the
same flags -- and compares them exactly, over what the held-out eval reads:

* the scene name sets (a scene on one side only fails; another order is reported
  with its mapping and is not a failure, since scenes are matched by name and the
  scorers sort the bundles themselves);
* the window's metadata: camera ids and positions, view_ids, times, the slot
  arrays, the anchors and their slots, track_upscaling_factor;
* query_points, in full;
* trajectories_world at the window's times and at every query's start frame,
  where the bundle's fallback rows read it;
* visibility, depth, intrinsics and extrinsics at the window's times (a loader's
  clip can be longer than the export: DexYCB's 25-frame scenes), and depth0;
* every observation (its frame label aside), every view's processed image and
  every tensor build_scene attaches to a view;
* build_anchor_correspondences on both: the trajectory indices, the query slots
  and the eligibility record.

One line per scene, PASS or the first field that differs with its largest absolute
difference; exit 1 on any difference.  CPU only, no model.  Run it in RC_ENV with
the mvtracker checkout on PYTHONPATH, as the loader benchmark runs:

  python benchmark_clip_parity.py \\
      --eval_benchmark_dataset kubric-multiview-v3-views0123-cached \\
      --eval_benchmark_root "$RCMV_DATA_ROOT" \\
      --eval_benchmark_clips "$I4_ROOT/data/4rc/clips/kubric-views0123"

The window flags are the trainer's own, with its defaults, so a recorded command's
values paste in.  ``--query_anchors`` is resolved as the held-out eval resolves it
(``_val_anchor_slots``: a slot the window cannot seat is dropped) and the header
prints what seated.  When the parity holds for Kubric and DexYCB, the loader
entries are deleted, and this script with them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

import train_temporal_tracking as train_cli
from arc.training import build_anchor_correspondences
from arc.training.scene_provider import SceneProviderError

# The trainer's flags that decide the held-out window and how its scenes are built.
WINDOW_FLAGS = (
    "--val_cameras",
    "--val_seq_len",
    "--stride",
    "--observation_budget",
    "--max_time_indices",
    "--query_anchors",
    "--kubric_max_depth",
    "--size",
    "--depth_input",
    "--camera_input",
)
# The window's metadata, compared before any array: a difference here explains
# every one after it.
METADATA_FIELDS = (
    "camera_ids",
    "cameras",
    "view_ids",
    "times",
    "slot_cameras",
    "slot_times",
    "slot_time_indices",
    "query_anchors",
    "query_observation_slot",
    "track_query_observation_slots",
    "track_upscaling_factor",
)
# (V, T, ...) arrays, compared at the window's times.
WINDOW_TIME_FIELDS = ("visibility", "depth", "intrinsics", "extrinsics_world_to_camera")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eval_benchmark_dataset",
        required=True,
        metavar="NAME",
        help="the loader benchmark, as the trainer takes it",
    )
    parser.add_argument(
        "--eval_benchmark_root",
        required=True,
        metavar="DIR",
        help="the dataset root its from_name takes, as the trainer takes it",
    )
    parser.add_argument(
        "--eval_benchmark_clips",
        required=True,
        metavar="DIR",
        help="the clip set 4I4 exported from that loader",
    )
    window = parser.add_argument_group(
        "the held-out window, by the trainer's own flags and defaults"
    )
    # The trainer parser's own actions, so a type or a default cannot drift.
    # argparse offers no public lookup by option string.
    trainer_actions = train_cli.build_arg_parser()._option_string_actions
    for flag in WINDOW_FLAGS:
        window._add_action(trainer_actions[flag])
    return parser


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_difference(expected: np.ndarray, actual: np.ndarray) -> str | None:
    """How two arrays differ -- shape, dtype, or values -- or None when identical.

    Exact equality, with NaN equal to NaN in the same place.
    """

    if expected.shape != actual.shape:
        return f"shape {expected.shape} vs {actual.shape}"
    if expected.dtype != actual.dtype:
        return f"dtype {expected.dtype} vs {actual.dtype}"
    floating = np.issubdtype(expected.dtype, np.floating)
    if np.array_equal(expected, actual, equal_nan=floating):
        return None
    if floating:
        nan_expected, nan_actual = np.isnan(expected), np.isnan(actual)
        if not np.array_equal(nan_expected, nan_actual):
            return f"NaN in {int((nan_expected != nan_actual).sum())} other place(s)"
        compared = ~nan_expected
        differs = (expected != actual) & compared
        gap = np.abs(
            expected[compared].astype(np.float64) - actual[compared].astype(np.float64)
        ).max()
    else:
        differs = expected != actual
        gap = np.abs(expected.astype(np.float64) - actual.astype(np.float64)).max()
    return (
        f"max |diff| {gap:.6g}, {int(differs.sum())} of {expected.size} elements differ"
    )


def _difference(expected, actual) -> str | None:
    """How two values of one field differ, or None when identical."""

    if isinstance(expected, (torch.Tensor, np.ndarray)) or isinstance(
        actual, (torch.Tensor, np.ndarray)
    ):
        return _array_difference(_numpy(expected), _numpy(actual))
    numbers = (int, float, np.integer, np.floating)
    if (
        isinstance(expected, numbers)
        and isinstance(actual, numbers)
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        if expected == actual or (np.isnan(expected) and np.isnan(actual)):
            return None
        return f"{expected!r} vs {actual!r} (|diff| {abs(float(expected) - float(actual)):.6g})"
    if expected == actual:
        return None
    return f"{expected!r} vs {actual!r}"


def _record_difference(expected, actual, path: str) -> tuple[str, str] | None:
    """The first leaf of two JSON-like records that differs, named by its path."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return path, f"keys {sorted(expected)} vs {sorted(actual)}"
        for key in expected:
            found = _record_difference(expected[key], actual[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return path, f"{len(expected)} vs {len(actual)} entries"
        for index, (mine, theirs) in enumerate(zip(expected, actual)):
            found = _record_difference(mine, theirs, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    detail = _difference(expected, actual)
    return None if detail is None else (path, detail)


def first_difference(expected, actual) -> tuple[str, str] | None:
    """The first field in which scene ``actual`` differs from ``expected``, and how.

    None when every field the held-out eval reads is identical; see the module
    docstring for the fields and the times each is compared at.
    """

    for name in METADATA_FIELDS:
        detail = _difference(getattr(expected, name), getattr(actual, name))
        if detail is not None:
            return name, detail
    detail = _difference(expected.query_points, actual.query_points)
    if detail is not None:
        return "query_points", detail

    window = [int(time) for time in expected.times]
    starts = np.rint(_numpy(expected.query_points)[:, 0]).astype(np.int64).tolist()
    read = sorted(set(window) | set(starts))
    held = min(expected.trajectories_world.shape[0], actual.trajectories_world.shape[0])
    if read and read[-1] >= held:
        return (
            "trajectories_world",
            f"time {read[-1]} is read, but one side holds {held} frames",
        )
    detail = _difference(
        expected.trajectories_world[read], actual.trajectories_world[read]
    )
    if detail is not None:
        return "trajectories_world", detail
    for name in WINDOW_TIME_FIELDS:
        mine, theirs = getattr(expected, name), getattr(actual, name)
        held = min(mine.shape[1], theirs.shape[1])
        if window[-1] >= held:
            return name, f"time {window[-1]} is in the window, but one side holds {held} frames"
        detail = _difference(mine[:, window], theirs[:, window])
        if detail is not None:
            return name, detail
    detail = _difference(expected.depth0, actual.depth0)
    if detail is not None:
        return "depth0", detail

    if len(expected.observations) != len(actual.observations):
        return "observations", f"{len(expected.observations)} vs {len(actual.observations)}"
    for slot, (mine, theirs) in enumerate(zip(expected.observations, actual.observations)):
        # The frame label names where the pixels came from, not what they are.
        for field in fields(mine):
            if field.name == "path":
                continue
            detail = _difference(getattr(mine, field.name), getattr(theirs, field.name))
            if detail is not None:
                return f"observations[{slot}].{field.name}", detail

    if len(expected.views) != len(actual.views):
        return "views", f"{len(expected.views)} vs {len(actual.views)}"
    for slot, (mine, theirs) in enumerate(zip(expected.views, actual.views)):
        if set(mine) != set(theirs):
            return f"views[{slot}]", f"keys {sorted(mine)} vs {sorted(theirs)}"
        for key in ["img", *sorted(key for key in mine if key != "img")]:
            detail = _difference(mine[key], theirs[key])
            if detail is not None:
                return f"views[{slot}].{key}", detail

    expected_correspondences, expected_eligibility = build_anchor_correspondences(expected)
    actual_correspondences, actual_eligibility = build_anchor_correspondences(actual)
    for name in ("trajectory_indices", "query_slots"):
        detail = _difference(
            getattr(expected_correspondences, name), getattr(actual_correspondences, name)
        )
        if detail is not None:
            return f"correspondences.{name}", detail
    return _record_difference(expected_eligibility, actual_eligibility, "eligibility")


def _mvtracker_commit() -> str | None:
    """HEAD of the mvtracker checkout this process imported, or None."""

    location = getattr(sys.modules.get("mvtracker"), "__file__", None)
    if not location:
        return None
    return train_cli._fork_commit(Path(location).resolve().parent)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # One namespace per provider, so each is opened exactly as the trainer opens
    # it: the trainer refuses the two sources together.
    loader_args = argparse.Namespace(**{**vars(args), "eval_benchmark_clips": None})
    clip_args = argparse.Namespace(
        **{**vars(args), "eval_benchmark_dataset": None, "eval_benchmark_root": None}
    )
    try:
        times, bound = train_cli._val_window(args)
        seated = train_cli._val_anchor_spec(args)
        clips = train_cli._open_benchmark(clip_args)
        loader = train_cli._open_benchmark(loader_args)
        loader_plans = train_cli._benchmark_plans(args, loader)
        clip_plans = train_cli._benchmark_plans(args, clips)
    except (ValueError, SceneProviderError) as error:
        parser.error(str(error))

    print(
        f"parity window: {len(args.val_cameras)} cameras x times {list(times)} "
        f"(bound {bound}), anchors {seated}, size {args.size}, "
        f"depth_input {args.depth_input}, camera_input {args.camera_input}, "
        f"max_depth {args.kubric_max_depth:g}"
    )
    print(f"parity loader: {json.dumps(loader.identity())}")
    print(
        f"parity clips: {json.dumps(clips.identity())} "
        f"mvtracker_checkout={_mvtracker_commit()}"
    )

    loader_names = [plan.seq_name for plan in loader_plans]
    by_name = {plan.seq_name: plan for plan in loader_plans}
    clip_by_name = {plan.seq_name: plan for plan in clip_plans}
    failed = 0
    for name in loader_names:
        if name not in clip_by_name:
            print(f"parity scene={name} FAIL scene: in the loader's set, not in the clip set")
            failed += 1
    for plan in clip_plans:
        if plan.seq_name not in by_name:
            print(f"parity scene={plan.seq_name} FAIL scene: in the clip set, not in the loader's")
            failed += 1
    shared = [name for name in loader_names if name in clip_by_name]
    clip_order = [plan.seq_name for plan in clip_plans if plan.seq_name in by_name]
    if clip_order != shared:
        position = {name: index for index, name in enumerate(clip_order)}
        mapping = " ".join(
            f"{name}:{index}->{position[name]}" for index, name in enumerate(shared)
        )
        print(
            "parity order: the clip set lists the shared scenes in another order "
            f"(scene:loader->clips) {mapping}"
        )

    passed = 0
    for name in shared:
        built = {}
        for side, provider, plan in (
            ("loader", loader, by_name[name]),
            ("clips", clips, clip_by_name[name]),
        ):
            # Any failure to build is this scene's difference, reported and
            # counted; the check goes on to the next scene.
            try:
                built[side] = provider(plan)
            except Exception as error:
                print(f"parity scene={name} FAIL load ({side}): {type(error).__name__}: {error}")
                break
        if len(built) < 2:
            failed += 1
            continue
        frames = (
            f"{built['loader'].trajectories_world.shape[0]}/"
            f"{built['clips'].trajectories_world.shape[0]}"
        )
        found = first_difference(built["loader"], built["clips"])
        if found is None:
            print(f"parity scene={name} PASS frames={frames}")
            passed += 1
        else:
            field, detail = found
            print(f"parity scene={name} FAIL {field}: {detail} (frames={frames})")
            failed += 1
        del built
    total = len(set(loader_names) | set(clip_by_name))
    print(f"parity: {passed} of {total} scenes identical, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
