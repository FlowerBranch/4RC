import os
import sys
import argparse
import torch
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(__file__))

from arc.models.arc import Arc


def get_args_parser():
    parser = argparse.ArgumentParser(description="Tracking evaluation (TapVid3D-style).")
    parser.add_argument("--checkpoint_dir", default="", type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory for logs")
    parser.add_argument("--data_root", type=str, default="./data", help="root of eval npz data")
    parser.add_argument("--data_type", type=str, default=None, help="dataset type (adt_mini/po_mini/pstudio_mini/ds_mini)")
    parser.add_argument("--num_frames", type=int, default=64, help="number of frames to evaluate per sequence")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="kept for compatibility; not used")
    parser.add_argument("--pose_eval_stride", type=int, default=1, help="frame stride for evaluation")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size used for cropping")
    parser.add_argument("--save_predictions", action="store_true", help="save per-sequence predictions to npy")
    parser.add_argument("--size", type=int, default=512, help="size of the image")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--model_type", type=str, default="4rc", help="model type (4rc, vggt, pi3, da3)")
    parser.add_argument("--align", type=str, default="sim3", choices=["global", "pertraj", "sim3", "sim3_closed"], help="alignment method for evaluation metrics")
    parser.add_argument("--track_query_idx", type=int, default=11, help="frame index used as tracking query")
    parser.add_argument(
        "--temporal_patch",
        default=None,
        help=(
            "Optional train_state.pt written by train_temporal_tracking.py. "
            "Builds the base model at the patch's time-table size, overlays the "
            "patch's trained parameters, and runs the forward the patch trained "
            "under: its refine_iters and, for a run trained with "
            "--merge_synchronized_slots, the merged head, here at one camera. A "
            "patch trained with --depth_input or --camera_input is refused: this "
            "eval feeds neither input. Not an exact replay of training: the eval "
            "attaches no time_index, so the patch's time-index table is never "
            "read. The trainer numbers each window's times from 0, so only the "
            "first rows ever train, and numbering this eval's frames would read "
            "rows that never trained."
        ),
    )
    return parser


def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint_dir if args.checkpoint_dir else "Luo-Yihang/4RC"
    if args.temporal_patch:
        from arc.training.checkpoint import load_temporal_patch_model

        model, forward_kwargs = load_temporal_patch_model(
            checkpoint, args.temporal_patch, device=device
        )
    else:
        model = Arc.from_pretrained(checkpoint).to(device)
        forward_kwargs = {}
    model.eval()

    track_eval_data_types = [args.data_type]

    for data_type in track_eval_data_types:
        if "waymo" in data_type or "kubric" in data_type:
            from track_eval_dense import eval_ours_tapvid3d
        else:
            from track_eval import eval_ours_tapvid3d
        eval_ours_tapvid3d(
            args,
            model,
            device,
            data_type=data_type,
            save_predictions=args.save_predictions,
            align=args.align,
            data_root=args.data_root,
            forward_kwargs=forward_kwargs,
        )

if __name__ == "__main__":
    main()
