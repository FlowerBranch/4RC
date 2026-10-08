"""Multi-view held-out eval viewer: one scene as a trained checkpoint saw it.

    python -m arc.viz.viser_multiview_eval <visual npz> [--port 8020]

Reads one ``eval/step-<N>/visual/<scene>.npz``, which ``evaluate_held_out``
writes under ``--eval_visual_scenes`` (schema: :mod:`arc.training.visual_dump`),
and draws the two errors every arm is judged on next to its tracks:

* the camera baseline: predicted (orange) and ground-truth (green) frusta for
  every camera, with a segment joining each pair of centres;
* the query anchor's error: the model's own anchor against the tracked point's
  true position, one segment per correspondence row, coloured by anchor camera.

Per time index, on a slider with Play, each ``/cams/t{i}`` frame holds that
time's nodes and the slider shows one frame at a time, the pattern upstream's
``viser_visualizer_track.py`` uses: every camera's predicted cloud (RGB or one
colour per camera), the optional ground-truth cloud, both frusta, the baseline
segments, and the scored tracks with trails -- each track hidden before its
query time, its ground truth dimmed where no camera sees it.  The GUI states
the model on screen: source file, run, step, checkpoint, commit, head
settings, and ``base_ratio`` and the anchor error per anchor.

CPU only, no model load.  Reading the dump imports ``arc.training``, which
loads torch.  viser is pinned to 1.0.30, the version ``$RC_ENV`` runs, so
lines take ``line_width`` (1.1 renamed it ``thickness``).  The server binds
127.0.0.1: the login node is shared, and the SSH tunnel targets localhost.
When ``--port`` is taken, viser binds the next free port; the URL printed is
the bound one, with a warning, and the tunnel must target that port.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf

from arc.training.visual_dump import read_visual_dump

HOST = "127.0.0.1"
PREDICTED = (255, 140, 0)
GROUND_TRUTH = (40, 200, 90)
GROUND_TRUTH_OCCLUDED = (35, 85, 50)
# tab10 without its orange and green, which mean predicted and ground truth.
CAMERA_COLORS = (
    (31, 119, 180),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)
COLOR_MODES = ("RGB", "Per camera")
# Each GUI toggle, the per-time layers it shows, and whether it starts on.
LAYER_TOGGLES = {
    "Predicted cloud": (("pred_points",), True),
    "Ground-truth cloud": (("gt_points",), False),
    "Predicted cameras": (("pred_frusta",), True),
    "Ground-truth cameras": (("gt_frusta",), True),
    "Baselines": (("baselines",), True),
    "Tracks": (("gt_tracks", "pred_tracks", "gt_trails", "pred_trails"), True),
}
TIME_LAYERS = tuple(layer for layers, _ in LAYER_TOGGLES.values() for layer in layers)


@dataclass
class EvalViewer:
    """Every handle the viewer created, so a caller can drive it headless."""

    server: viser.ViserServer
    time_count: int
    # One /cams/t{i} frame per time index; the slider shows exactly one.
    frames: list
    # Per time index, layer name -> that time's nodes in the layer.
    layers: list
    # The static anchor layer: "pred", "true" and "error".
    anchors: dict
    gui: dict


def _scene_extent(points: np.ndarray) -> float:
    """The 20th-80th percentile span of the finite points, as upstream sizes it."""

    points = points.reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return 1.0
    span = np.percentile(points, 80, axis=0) - np.percentile(points, 20, axis=0)
    extent = float(span.max())
    return extent if extent > 0 else 1.0


def _frustum_pose(rotation, center, intrinsics, height: int, width: int):
    """``(fov, aspect, wxyz, position)`` of one camera, or None if any part is non-finite."""

    if not (
        np.isfinite(rotation).all()
        and np.isfinite(center).all()
        and np.isfinite(intrinsics).all()
    ):
        return None
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    if fx <= 0 or fy <= 0:
        return None
    return (
        float(2.0 * np.arctan2(height / 2.0, fy)),
        float((width / fx) / (height / fy)),
        vtf.SO3.from_matrix(rotation.astype(np.float64)).wxyz,
        center.astype(np.float64),
    )


def _finite_rows(points: np.ndarray, colors: np.ndarray):
    keep = np.isfinite(points).all(axis=-1)
    return points[keep], colors[keep]


def _trails(track: np.ndarray, visible: np.ndarray, query_index: np.ndarray, time_index: int):
    """Segments from each track's query time up to ``time_index``.

    Returns ``(segments, visible)``: ``(K, 2, 3)`` segments, step ``j-1 -> j``
    for every ``j <= time_index`` a track has reached, and the ground-truth
    visibility at each segment's later end.  Non-finite segments are dropped.
    """

    segments, segment_visible = [], []
    for later in range(1, time_index + 1):
        live = np.flatnonzero(query_index <= later - 1)
        segments.append(np.stack([track[later - 1, live], track[later, live]], axis=1))
        segment_visible.append(visible[later, live])
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros(0, dtype=bool)
    segments = np.concatenate(segments)
    segment_visible = np.concatenate(segment_visible)
    keep = np.isfinite(segments).all(axis=(1, 2))
    return segments[keep], segment_visible[keep]


def _visibility_colors(visible: np.ndarray, *, pairs: bool = False) -> np.ndarray:
    colors = np.where(
        visible[:, None],
        np.array(GROUND_TRUTH, dtype=np.uint8),
        np.array(GROUND_TRUTH_OCCLUDED, dtype=np.uint8),
    ).astype(np.uint8)
    return np.repeat(colors[:, None], 2, axis=1) if pairs else colors


def _head_summary(metadata: dict) -> str:
    inputs = [
        name
        for name, on in (("depth", metadata["depth_input"]), ("camera", metadata["camera_input"]))
        if on
    ]
    if metadata["oracle_query_anchor"]:
        scored = "oracle anchors"
    elif metadata["ground_truth_query_anchor"]:
        scored = "ground-truth-depth anchors"
    else:
        scored = "model anchors"
    return (
        f"{'merged' if metadata['merge_synchronized_slots'] else 'per-slot'} head, "
        f"K={metadata['refine_iters']}, inputs: {'+'.join(inputs) or 'none'}, "
        f"tracks scored on {scored}"
    )


def _anchor_error_text(summary: dict) -> str:
    if not summary["count"]:
        return "- (n=0)"
    return f"median {summary['median']:.3f} m (n={summary['count']})"


def build_viewer(
    server: viser.ViserServer, arrays: dict, metadata: dict, *, source: str
) -> EvalViewer:
    """Draw one visual dump on ``server``; starts no thread, so it runs headless."""

    time_count = int(arrays["track_pred"].shape[0])
    height, width = (int(value) for value in arrays["image_size"])
    slot_cameras = arrays["slot_camera_id"]
    slot_times = arrays["slot_time_index"]
    palette = {
        camera: CAMERA_COLORS[index % len(CAMERA_COLORS)]
        for index, camera in enumerate(sorted(set(slot_cameras.tolist())))
    }
    extent = _scene_extent(arrays["pred_points_m"])
    point_size = 0.004 * extent
    camera_percent = 5.0
    query_index = np.rint(arrays["track_query_points"][:, 0]).astype(np.int64)
    original_time = {
        int(index): int(time) for index, time in zip(slot_times, arrays["slot_original_time"])
    }

    server.gui.set_panel_label(f"{metadata['scene']} @ step {metadata['step']}")
    gui: dict = {}
    with server.gui.add_folder("Model on screen"):
        for key, label, value in (
            ("source", "Source", source),
            ("run", "Run", metadata["output_dir"]),
            ("step", "Step", str(metadata["step"])),
            ("scene", "Scene", metadata["scene"]),
            ("checkpoint", "Checkpoint", str(metadata["checkpoint_dir"])),
            ("commit", "Commit", metadata["commit"] or "unknown"),
            ("head", "Head", _head_summary(metadata)),
        ):
            gui[key] = server.gui.add_text(label, initial_value=value, disabled=True)
    with server.gui.add_folder("Readouts"):
        for key, ratio in metadata["base_ratio"].items():
            gui[f"base_ratio {key}"] = server.gui.add_text(
                f"base_ratio {key}",
                initial_value="-" if ratio is None else f"{ratio:.4f}",
                disabled=True,
            )
        for key, summary in metadata["anchor_error_m"].items():
            gui[f"anchor error {key}"] = server.gui.add_text(
                f"anchor err {key}",
                initial_value=_anchor_error_text(summary),
                disabled=True,
            )
    with server.gui.add_folder("Playback"):
        gui["time"] = server.gui.add_slider(
            "Time index", min=0, max=max(time_count - 1, 0), step=1, initial_value=0
        )
        gui["original_time"] = server.gui.add_text(
            "Original time", initial_value=str(original_time.get(0, "")), disabled=True
        )
        gui["play"] = server.gui.add_checkbox("Play", initial_value=False)
        gui["fps"] = server.gui.add_slider(
            "FPS", min=0.5, max=10.0, step=0.5, initial_value=2.0
        )
        gui["previous"] = server.gui.add_button("Previous")
        gui["next"] = server.gui.add_button("Next")
    with server.gui.add_folder("Layers"):
        gui["color"] = server.gui.add_dropdown(
            "Cloud colour", COLOR_MODES, initial_value=COLOR_MODES[0]
        )
        for label, (_, initially) in LAYER_TOGGLES.items():
            gui[label] = server.gui.add_checkbox(label, initial_value=initially)
        gui["Anchors"] = server.gui.add_checkbox("Anchors", initial_value=True)
        gui["point_size"] = server.gui.add_slider(
            "Point size",
            min=0.0,
            max=10 * point_size,
            step=point_size / 20,
            initial_value=point_size,
        )
        gui["camera_size"] = server.gui.add_slider(
            "Camera size (%)", min=0.5, max=25.0, step=0.5, initial_value=camera_percent
        )

    # Up is the mean ground-truth camera's -Y, the OpenCV camera's up.
    up = -np.mean(arrays["gt_camera_rotation"][:, :, 1], axis=0)
    if np.isfinite(up).all() and np.linalg.norm(up) > 1e-6:
        server.scene.set_up_direction(tuple(float(value) for value in up / np.linalg.norm(up)))

    visible = {label: on for label, (_, on) in LAYER_TOGGLES.items()}
    shown = {layer: visible[label] for label, (layers, _) in LAYER_TOGGLES.items() for layer in layers}
    frames: list = []
    layers: list = []
    # Per time index, (handle, per-point RGB, camera colour) for every predicted cloud.
    clouds: list = []
    for time_index in range(time_count):
        prefix = f"/cams/t{time_index}"
        frames.append(
            server.scene.add_frame(prefix, show_axes=False, visible=time_index == 0)
        )
        nodes: dict = {layer: [] for layer in TIME_LAYERS}
        time_clouds = []
        baselines, baseline_colors = [], []
        for slot in np.flatnonzero(slot_times == time_index):
            camera = int(slot_cameras[slot])
            name = f"cam{camera}"
            color = palette[camera]
            points, rgb = _finite_rows(
                arrays["pred_points_m"][slot].reshape(-1, 3),
                arrays["rgb"][slot].reshape(-1, 3),
            )
            handle = server.scene.add_point_cloud(
                f"{prefix}/pred_points/{name}",
                points=points,
                colors=rgb,
                point_size=point_size,
                point_shape="rounded",
                precision="float32",
                visible=shown["pred_points"],
            )
            nodes["pred_points"].append(handle)
            time_clouds.append((handle, rgb, color))
            truth = arrays["gt_points_m"][slot].reshape(-1, 3)
            truth = truth[arrays["gt_points_valid"][slot].reshape(-1) & np.isfinite(truth).all(axis=1)]
            nodes["gt_points"].append(
                server.scene.add_point_cloud(
                    f"{prefix}/gt_points/{name}",
                    points=truth,
                    colors=GROUND_TRUTH,
                    point_size=point_size,
                    point_shape="rounded",
                    precision="float32",
                    visible=shown["gt_points"],
                )
            )
            for side, layer, frustum_color, image in (
                ("pred", "pred_frusta", PREDICTED, arrays["rgb"][slot]),
                ("gt", "gt_frusta", GROUND_TRUTH, None),
            ):
                pose = _frustum_pose(
                    arrays[f"{side}_camera_rotation"][slot],
                    arrays[f"{side}_camera_center_m"][slot],
                    arrays[f"{side}_intrinsics"][slot],
                    height,
                    width,
                )
                if pose is None:
                    print(f"{side} camera {camera} at time index {time_index} is non-finite; not drawn")
                    continue
                fov, aspect, wxyz, position = pose
                nodes[layer].append(
                    server.scene.add_camera_frustum(
                        f"{prefix}/{layer}/{name}",
                        fov=fov,
                        aspect=aspect,
                        scale=extent * camera_percent / 100.0,
                        line_width=2.0,
                        color=frustum_color,
                        image=image,
                        wxyz=wxyz,
                        position=position,
                        visible=shown[layer],
                    )
                )
            pair = np.stack(
                [arrays["pred_camera_center_m"][slot], arrays["gt_camera_center_m"][slot]]
            )
            if np.isfinite(pair).all():
                baselines.append(pair)
                baseline_colors.append([color, color])
        nodes["baselines"].append(
            server.scene.add_line_segments(
                f"{prefix}/baselines",
                points=np.array(baselines, dtype=np.float32).reshape(-1, 2, 3),
                colors=np.array(baseline_colors, dtype=np.uint8).reshape(-1, 2, 3),
                line_width=3.0,
                visible=shown["baselines"],
            )
        )

        live = np.flatnonzero(query_index <= time_index)
        truth, truth_colors = _finite_rows(
            arrays["track_gt"][time_index, live],
            _visibility_colors(arrays["track_gt_vis_any"][time_index, live]),
        )
        nodes["gt_tracks"].append(
            server.scene.add_point_cloud(
                f"{prefix}/gt_tracks",
                points=truth,
                colors=truth_colors,
                point_size=3 * point_size,
                point_shape="circle",
                precision="float32",
                visible=shown["gt_tracks"],
            )
        )
        predicted = arrays["track_pred"][time_index, live]
        predicted = predicted[np.isfinite(predicted).all(axis=1)]
        nodes["pred_tracks"].append(
            server.scene.add_point_cloud(
                f"{prefix}/pred_tracks",
                points=predicted,
                colors=PREDICTED,
                point_size=3 * point_size,
                point_shape="circle",
                precision="float32",
                visible=shown["pred_tracks"],
            )
        )
        for layer, track, colors in (
            ("gt_trails", arrays["track_gt"], None),
            ("pred_trails", arrays["track_pred"], PREDICTED),
        ):
            segments, segment_visible = _trails(
                track, arrays["track_gt_vis_any"], query_index, time_index
            )
            nodes[layer].append(
                server.scene.add_line_segments(
                    f"{prefix}/{layer}",
                    points=segments,
                    colors=(
                        _visibility_colors(segment_visible, pairs=True)
                        if colors is None
                        else colors
                    ),
                    line_width=1.5,
                    visible=shown[layer],
                )
            )
        layers.append(nodes)
        clouds.append(time_clouds)

    anchor_colors = np.array(
        [
            palette.get(int(key.partition(":")[0]), (255, 255, 255))
            for key in arrays["anchor_key"]
        ],
        dtype=np.uint8,
    ).reshape(-1, 3)
    anchors = {
        "pred": server.scene.add_point_cloud(
            "/anchors/pred",
            points=arrays["anchor_pred_m"],
            colors=anchor_colors,
            point_size=4 * point_size,
            point_shape="diamond",
            precision="float32",
        ),
        "true": server.scene.add_point_cloud(
            "/anchors/true",
            points=arrays["anchor_true_m"],
            colors=anchor_colors,
            point_size=4 * point_size,
            point_shape="circle",
            precision="float32",
        ),
        "error": server.scene.add_line_segments(
            "/anchors/error",
            points=np.stack([arrays["anchor_pred_m"], arrays["anchor_true_m"]], axis=1),
            colors=np.repeat(anchor_colors[:, None], 2, axis=1),
            line_width=2.0,
        ),
    }

    @gui["time"].on_update
    def _(_) -> None:
        current = int(gui["time"].value)
        with server.atomic():
            for time_index, frame in enumerate(frames):
                frame.visible = time_index == current
        gui["original_time"].value = str(original_time.get(current, ""))

    @gui["previous"].on_click
    def _(_) -> None:
        gui["time"].value = (int(gui["time"].value) - 1) % time_count

    @gui["next"].on_click
    def _(_) -> None:
        gui["time"].value = (int(gui["time"].value) + 1) % time_count

    @gui["play"].on_update
    def _(_) -> None:
        for key in ("time", "previous", "next"):
            gui[key].disabled = gui["play"].value

    @gui["color"].on_update
    def _(_) -> None:
        per_camera = gui["color"].value == COLOR_MODES[1]
        with server.atomic():
            for time_clouds in clouds:
                for handle, rgb, color in time_clouds:
                    handle.colors = (
                        np.broadcast_to(np.array(color, dtype=np.uint8), rgb.shape).copy()
                        if per_camera
                        else rgb
                    )

    for label, (toggled, _) in LAYER_TOGGLES.items():

        def toggle(_, label=label, toggled=toggled) -> None:
            with server.atomic():
                for nodes in layers:
                    for layer in toggled:
                        for handle in nodes[layer]:
                            handle.visible = gui[label].value

        gui[label].on_update(toggle)

    @gui["Anchors"].on_update
    def _(_) -> None:
        with server.atomic():
            for handle in anchors.values():
                handle.visible = gui["Anchors"].value

    @gui["point_size"].on_update
    def _(_) -> None:
        size = float(gui["point_size"].value)
        with server.atomic():
            for nodes in layers:
                for layer, scale in (
                    ("pred_points", 1),
                    ("gt_points", 1),
                    ("gt_tracks", 3),
                    ("pred_tracks", 3),
                ):
                    for handle in nodes[layer]:
                        handle.point_size = scale * size
            for key in ("pred", "true"):
                anchors[key].point_size = 4 * size

    @gui["camera_size"].on_update
    def _(_) -> None:
        scale = extent * float(gui["camera_size"].value) / 100.0
        with server.atomic():
            for nodes in layers:
                for layer in ("pred_frusta", "gt_frusta"):
                    for handle in nodes[layer]:
                        handle.scale = scale

    reference = np.flatnonzero(slot_times == 0)
    look_at = arrays["pred_points_m"][reference].reshape(-1, 3)
    look_at = look_at[np.isfinite(look_at).all(axis=1)]
    if len(look_at) and len(reference) and np.isfinite(
        arrays["gt_camera_center_m"][reference[0]]
    ).all():
        target = np.median(look_at, axis=0)
        position = arrays["gt_camera_center_m"][reference[0]].astype(np.float64)

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = target

    return EvalViewer(
        server=server,
        time_count=time_count,
        frames=frames,
        layers=layers,
        anchors=anchors,
        gui=gui,
    )


def _play(viewer: EvalViewer, stop: threading.Event) -> None:
    """Advance the time slider while Play is ticked, until ``stop`` is set."""

    gui = viewer.gui
    while not stop.wait(1.0 / float(gui["fps"].value)):
        if gui["play"].value:
            gui["time"].value = (int(gui["time"].value) + 1) % viewer.time_count


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="View one held-out scene's visual dump (--eval_visual_scenes) with viser"
    )
    parser.add_argument(
        "npz", type=Path, help="a dump: <output_dir>/eval/step-<N>/visual/<scene>.npz"
    )
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args(argv)

    arrays, metadata = read_visual_dump(args.npz)
    # Loopback only: viser binds every interface by default, and the login node
    # is shared. Reach it through the SSH tunnel.
    server = viser.ViserServer(host=HOST, port=args.port)
    # viser moves to the next free port when the requested one is taken -- by
    # a second viewer comparing another step or arm, or by upstream's viewer,
    # also on 8020 -- and a URL or tunnel on the requested port would then
    # open that other server. So the port printed is the one bound.
    port = server.get_port()
    if port != args.port:
        print(
            f"warning: port {args.port} is taken, so this viewer is on port {port}; "
            f"tunnel to it with ssh -L {port}:localhost:{port}, not {args.port}",
            file=sys.stderr,
        )
    viewer = build_viewer(server, arrays, metadata, source=str(args.npz))
    stop = threading.Event()
    threading.Thread(target=_play, args=(viewer, stop), daemon=True).start()
    print(f"viewer running at http://{HOST}:{port}; Ctrl+C to stop")
    try:
        _wait_for_interrupt(stop)
    finally:
        stop.set()
        server.stop()


def _wait_for_interrupt(stop: threading.Event) -> None:
    """Block until Ctrl+C."""

    try:
        while not stop.wait(3600.0):
            pass
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
