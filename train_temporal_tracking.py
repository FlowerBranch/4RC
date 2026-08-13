#!/usr/bin/env python3
"""Multi-scene temporal-tracking trainer, replaying MVTracker's sample stream.

**Landing 2 of four: only ``--plan_only`` works.** The record-to-selection
planner is complete and tested; scene loading, the training step, the schedule,
checkpoint/resume and the held-out eval are not written yet, and every path that
would need them raises rather than pretending. That is deliberate -- the planner
is fully determined by the manifest schema and testable without a GPU, while how
scenes are loaded is still waiting on a cluster measurement, so the two landed
separately instead of the second guessing at the first.

``--plan_only`` is worth running on its own: it walks a real manifest and reports
what every step *would* select, before a GPU is allocated. A manifest whose rows
this trainer cannot replay, or whose windows are shorter than the budget assumes,
shows up at submit time instead of twelve hours in.

  python train_temporal_tracking.py --manifest run/manifest.jsonl --plan_only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from arc.training.manifest_plan import (
    SKIP_EXCLUDED_DATA_ROOT,
    ManifestPlanError,
    StepPlan,
    plan_manifest,
)
from arc.training.sample_manifest import MANIFEST_VERSION, read_manifest


# The committed first-run window: 4 cameras x 10 times at stride 2. The budget is
# a measured memory ceiling (peak ~= 2.86*N + 7.2 GiB, N = cameras x times), not
# a preference, so it is a flag with a default rather than a constant.
DEFAULT_OBSERVATION_BUDGET = 40
DEFAULT_STRIDE = 2
# The time-index embedding's row count. A window may not carry more times than
# the table can index, whatever the budget allows.
DEFAULT_MAX_TIME_INDICES = 32
# Above this share of unreplayable rows the manifest is damaged rather than
# merely untidy, and training on the remainder would be training on a fraction of
# the recorded stream while every other number looked healthy.
DEFAULT_MAX_UNREPLAYABLE_FRACTION = 0.02


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay an MVTracker sample manifest to train 4RC temporal tracking"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="MVTracker manifest.jsonl to replay, in file order",
    )
    parser.add_argument(
        "--observation_budget",
        type=int,
        default=DEFAULT_OBSERVATION_BUDGET,
        help=(
            "Maximum cameras x times per step. Peak memory depends only on this "
            "product, not on how it splits, so it is the one knob that bounds a "
            "step's footprint (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=(
            "Frames between consecutive selected times. Fixed for the whole run "
            "so embedding row k always means k*stride frames after the anchor; a "
            "run at a different stride trains a different embedding and its "
            "patch is not loadable here (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max_time_indices",
        type=int,
        default=DEFAULT_MAX_TIME_INDICES,
        help="Rows in the time-index embedding (default: %(default)s)",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=2,
        help=(
            "Records with fewer views are skipped. Below two there is no "
            "synchronized cross-view pair, so the run cannot measure the thing "
            "temporal indexing exists for (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--exclude_data_root",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Skip records whose data_root matches, repeatable. A run's "
            "static-pretraining epoch draws from other datasets and is marked by "
            "nothing else; these are replayable rows this trainer declines, not "
            "damage, and they are counted separately"
        ),
    )
    parser.add_argument(
        "--max_unreplayable_fraction",
        type=float,
        default=DEFAULT_MAX_UNREPLAYABLE_FRACTION,
        help=(
            "Abort if more than this share of considered records are "
            "unreplayable. Deliberate exclusions leave both sides of the ratio, "
            "so excluding a large epoch cannot mask real damage "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help=(
            "Report what every step would select and exit. Needs no GPU, no "
            "checkpoint and no scene data. Currently the only supported mode"
        ),
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Plan only the first N records; for eyeballing a long manifest",
    )
    parser.add_argument("--json_out", help="Write the plan report as JSON")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.observation_budget < 1:
        raise ValueError("--observation_budget must be positive")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.max_time_indices < 1:
        raise ValueError("--max_time_indices must be positive")
    if args.min_views < 1:
        raise ValueError("--min_views must be at least 1")
    if not 0 <= args.max_unreplayable_fraction <= 1:
        raise ValueError("--max_unreplayable_fraction must be in [0, 1]")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max_records must be positive")
    if args.observation_budget < args.min_views:
        raise ValueError(
            f"--observation_budget {args.observation_budget} cannot seat even one "
            f"time at --min_views {args.min_views}"
        )


def _plan_summary(tally, args) -> dict:
    strides = Counter(plan.stride for plan in tally.planned)
    bounds = Counter(plan.time_bound for plan in tally.planned)
    observations = Counter(plan.observation_count for plan in tally.planned)
    views = Counter(len(plan.cameras) for plan in tally.planned)
    duplicates = [plan.duplicate_track_count for plan in tally.planned]
    tracks = [len(plan.track_indices) for plan in tally.planned]
    transformed = sum(1 for plan in tally.planned if plan.scene_transform)
    return {
        "manifest_version_expected": MANIFEST_VERSION,
        "records_read": len(tally.planned) + len(tally.skipped),
        "planned_steps": len(tally.planned),
        "skipped": tally.skip_counts,
        "unreplayable_fraction": tally.threshold_skip_fraction,
        "considered_for_threshold": tally.considered,
        "distinct_scenes": len({plan.seq_name for plan in tally.planned}),
        "views_per_step": dict(sorted(views.items())),
        "observations_per_step": dict(sorted(observations.items())),
        "time_bound": dict(bounds),
        "stride": dict(strides),
        "tracks_per_step": (
            None
            if not tracks
            else {"min": min(tracks), "max": max(tracks), "mean": sum(tracks) / len(tracks)}
        ),
        # Non-zero is normal: the loader draws from overlapping pools. Reported so
        # a positional selection and a set-based one differ by a number here
        # rather than by supervision quietly going missing.
        "duplicate_track_ids": (
            None
            if not duplicates
            else {"total": sum(duplicates), "max_in_a_step": max(duplicates)}
        ),
        "records_with_scene_transform": transformed,
        "settings": {
            "observation_budget": args.observation_budget,
            "stride": args.stride,
            "max_time_indices": args.max_time_indices,
            "min_views": args.min_views,
            "excluded_data_roots": list(args.exclude_data_root),
        },
    }


def _print_plan(tally, summary, *, limit: int = 20) -> None:
    for plan in tally.planned[:limit]:
        times = list(plan.times)
        shown = times if len(times) <= 6 else times[:5] + ["..."] + times[-1:]
        print(
            f"step={plan.step} scene={plan.seq_name} "
            f"views={list(plan.cameras)} "
            f"window=[{plan.frame_start},{plan.frame_start + plan.seq_len}) "
            f"times={shown} stride={plan.stride} bound={plan.time_bound} "
            f"obs={plan.observation_count} "
            f"tracks={len(plan.track_indices)}"
            f"(+{plan.duplicate_track_count} dup) "
            f"transform={'yes' if plan.scene_transform else 'no'}"
        )
    if len(tally.planned) > limit:
        print(f"... {len(tally.planned) - limit} more planned steps")

    for entry in tally.skipped[:limit]:
        print(f"SKIP step={entry.step} scene={entry.seq_name} {entry.cause}: {entry.detail}")
    if len(tally.skipped) > limit:
        print(f"... {len(tally.skipped) - limit} more skipped records")

    print()
    for key in (
        "records_read",
        "planned_steps",
        "distinct_scenes",
        "views_per_step",
        "observations_per_step",
        "time_bound",
        "tracks_per_step",
        "duplicate_track_ids",
        "records_with_scene_transform",
        "skipped",
        "unreplayable_fraction",
    ):
        print(f"{key}={summary[key]}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.plan_only:
        raise NotImplementedError(
            "Only --plan_only is implemented. The training step, LR schedule, "
            "checkpoint/resume, signal handling and held-out eval land next, "
            "together with scene loading -- which is still waiting on the "
            "cluster probe that decides whether scenes come from the live "
            "MVTracker dataset or through the dumper. Re-run with --plan_only."
        )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")
    records = read_manifest(manifest_path)
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        parser.error(f"{manifest_path} holds no records")

    versions = {record.get("manifest_version") for record in records}
    if versions != {MANIFEST_VERSION}:
        print(
            f"WARNING manifest_version {sorted(versions, key=str)} against the "
            f"vendored schema's {MANIFEST_VERSION}; a key may have changed "
            "meaning. Re-vendor arc/training/sample_manifest.py from mvtracker "
            "rather than reading the difference by hand.",
            file=sys.stderr,
        )

    try:
        tally = plan_manifest(
            records,
            available_cameras=None,
            budget=args.observation_budget,
            stride=args.stride,
            max_time_indices=args.max_time_indices,
            min_views=args.min_views,
            excluded_data_roots=args.exclude_data_root,
        )
    except ManifestPlanError as exc:
        parser.error(str(exc))

    summary = _plan_summary(tally, args)
    _print_plan(tally, summary)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"summary={args.json_out}")

    # stderr is unbuffered and stdout is not, so without this the verdict lands
    # above the report it is a verdict on whenever the two are merged.
    sys.stdout.flush()

    if tally.threshold_skip_fraction > args.max_unreplayable_fraction:
        print(
            f"FAIL {tally.threshold_skip_count}/{tally.considered} considered "
            f"records are unreplayable "
            f"({tally.threshold_skip_fraction:.4%} > "
            f"{args.max_unreplayable_fraction:.4%}). Deliberate exclusions "
            f"({tally.skip_counts.get(SKIP_EXCLUDED_DATA_ROOT, 0)}) are not "
            "counted here.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not tally.planned:
        print("FAIL no record produced a replayable step", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS planned {len(tally.planned)} steps from {manifest_path}")


if __name__ == "__main__":
    main()
