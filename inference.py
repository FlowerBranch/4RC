import argparse
import glob
import os
import time
from numbers import Integral

import numpy as np
import torch

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
TIME_INDEX_KEY = "time_index"
TIME_EMBEDDING_KEY = "backbone.pretrained.time_index_embedding.weight"


def collect_images(input_path):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTS:
            from arc.viz.video_utils import extract_frames_from_video
            tmp_dir = f"/tmp/4rc_frames_{int(time.time() * 1000)}"
            os.makedirs(tmp_dir, exist_ok=True)
            paths = extract_frames_from_video(input_path, tmp_dir)
            return paths, True
        return [input_path], False
    if os.path.isdir(input_path):
        paths = sorted(
            p for p in glob.glob(os.path.join(input_path, "*"))
            if os.path.splitext(p)[1].lower() in IMAGE_EXTS
        )
        return paths, False
    raise ValueError(f"Input not found: {input_path}")


def validate_time_indices(time_indices, max_time_indices=None):
    if time_indices is None:
        return None

    validated = []
    for position, value in enumerate(time_indices):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"time index at position {position} must be an integer, got {type(value).__name__}"
            )
        value = int(value)
        if value < 0:
            raise ValueError(
                f"time index at position {position} must be non-negative, got {value}"
            )
        if max_time_indices is not None and value >= max_time_indices:
            raise ValueError(
                f"time index at position {position} must be in [0, {max_time_indices - 1}], "
                f"got {value}"
            )
        validated.append(value)
    return validated


def select_input_frames(paths, time_indices=None, *, is_video=False, max_frames=30):
    paths = list(paths)
    time_indices = validate_time_indices(time_indices)
    if time_indices is not None and len(time_indices) != len(paths):
        raise ValueError(
            "--time_indices requires one value per collected input frame before "
            f"subsampling: expected {len(paths)}, got {len(time_indices)}"
        )

    if is_video or len(paths) > max_frames:
        selected_indices = np.linspace(
            0,
            len(paths) - 1,
            max_frames,
            dtype=int,
        )
        paths = [paths[index] for index in selected_indices]
        if time_indices is not None:
            time_indices = [time_indices[index] for index in selected_indices]

    return paths, time_indices


def attach_frame_metadata(imgs, track_query_idx, time_indices=None):
    time_indices = validate_time_indices(time_indices)
    if time_indices is not None and len(time_indices) != len(imgs):
        raise ValueError(
            "Every loaded image must have exactly one time index after subsampling: "
            f"loaded {len(imgs)} images but have {len(time_indices)} indices"
        )

    query_indices = [
        index if 0 <= index < len(imgs) else len(imgs) // 2
        for index in track_query_idx
    ]
    for view_idx, img in enumerate(imgs):
        img["track_query_idx"] = torch.tensor(query_indices, dtype=torch.long)
        if time_indices is not None:
            img[TIME_INDEX_KEY] = torch.tensor(
                [time_indices[view_idx]],
                dtype=torch.long,
            )
    return query_indices


def save_npz(output_dict, path):
    flat = {"n_frames": np.array(len(output_dict["preds"]))}

    for i, pred in enumerate(output_dict["preds"]):
        for k, v in pred.items():
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            if isinstance(arr, np.ndarray):
                flat[f"pred_{i}_{k}"] = arr

    for i, view in enumerate(output_dict["views"]):
        for k, v in view.items():
            if k == TIME_INDEX_KEY:
                continue
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            if isinstance(arr, np.ndarray):
                flat[f"view_{i}_{k}"] = arr

    if output_dict.get("track_text_mask"):
        masks = output_dict["track_text_mask"]
        flat["track_text_mask_n"] = np.array(len(masks))
        for i, m in enumerate(masks):
            if m is not None:
                flat[f"track_text_mask_{i}"] = m

    for k in ("track_dynamic_objects_text", "track_query_img_uint8"):
        if k in output_dict and output_dict[k] is not None:
            flat[k] = np.array(output_dict[k])

    flat["refine_track_visual"] = np.array(bool(output_dict.get("refine_track_visual", False)))

    np.savez_compressed(path, **flat)
    print(f"Saved → {path}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image folder, single image, or video file")
    parser.add_argument("--save", required=True, help="Output .npz path")
    parser.add_argument("--checkpoint_dir", default="Luo-Yihang/4RC")
    parser.add_argument("--track_query_idx", type=int, nargs="+", default=[-1], help="Frame index/indices for tracking query; -1 = middle frame")
    parser.add_argument(
        "--time_indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One semantic time index per collected input frame, before any 30-frame "
            "subsampling. Repeated values mark synchronized observations and must not "
            "encode camera identity."
        ),
    )
    parser.add_argument(
        "--temporal_patch",
        default=None,
        help=(
            "Optional temporal_tracking.pt written by overfit_temporal_tracking.py. "
            "Overlays the finetuned time embedding, MotionDecoder and track head onto "
            "the base checkpoint. Without it the time embedding stays zero-initialized "
            "and --time_indices has no effect on the output."
        ),
    )
    parser.add_argument("--refine_track_visualization", action="store_true", default=False)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, is_video = collect_images(args.input)
    try:
        paths, time_indices = select_input_frames(
            paths,
            args.time_indices,
            is_video=is_video,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Processing {len(paths)} frames...")

    from arc.models.arc import Arc
    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import load_images

    # Bound-check against the default table size before paying for the ~1B-param
    # checkpoint load. The post-load check below still runs, because a checkpoint
    # config.json may declare a different max_time_indices.
    try:
        validate_time_indices(time_indices, max_time_indices=Arc.MAX_TIME_INDICES)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    model = Arc.from_pretrained(args.checkpoint_dir).to(device)
    if args.temporal_patch:
        from arc.training.checkpoint import load_temporal_tracking_checkpoint

        # set_freeze first: the loader requires it and keys the expected
        # parameter set off requires_grad.
        model.set_freeze("temporal_tracking")
        load_temporal_tracking_checkpoint(model, args.temporal_patch)
        print(f"Loaded temporal-tracking patch: {args.temporal_patch}")
    elif TIME_EMBEDDING_KEY in getattr(model, "consumed_legacy_missing_keys", frozenset()):
        print(
            "Warning: this checkpoint has no time-index embedding, so it was "
            "zero-initialized. --time_indices will not change the output. Pass "
            "--temporal_patch to load a finetuned embedding."
        )
    model = model.eval()
    imgs = load_images(paths, size=512, verbose=True, patch_size=14)

    try:
        time_indices = validate_time_indices(
            time_indices,
            max_time_indices=model.max_time_indices,
        )
        attach_frame_metadata(
            imgs,
            args.track_query_idx,
            time_indices,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    output_dict, profiling = inference(
        imgs, model, device, dtype="bf16-mixed", verbose=True, profiling=True, use_center_as_anchor=False
    )
    print(f"Inference: {profiling['total_time']:.2f}s")

    for pred in output_dict["preds"]:
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                pred[k] = v.cpu()
    for view in output_dict["views"]:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_dict["refine_track_visual"] = args.refine_track_visualization
    if args.refine_track_visualization:
        from arc.viz.motion_seg import prepare_refine_mask
        prepare_refine_mask(output_dict, device, refine_track_visual=True)

    save_npz(output_dict, args.save)


if __name__ == "__main__":
    main()
