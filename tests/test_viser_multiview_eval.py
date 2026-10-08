"""The multi-view eval viewer, built headless on an in-process viser server.

No browser: ``build_viewer`` draws a visual dump on a server bound to a free
loopback port, and the handles it returns are read back.  viser 1.0.30 reports
port 0 for such a server, so nothing here reads the port.
"""

from __future__ import annotations

import json
import re
import socket

import numpy as np
import pytest
import viser

import arc.viz.viser_multiview_eval as viewer_module
from arc.training import build_visual_arrays, read_visual_dump, write_visual_dump
from arc.viz.viser_multiview_eval import (
    CAMERA_COLORS,
    GROUND_TRUTH,
    GROUND_TRUTH_OCCLUDED,
    PREDICTED,
    TIME_LAYERS,
    build_viewer,
)
from test_trainer_loop import _FOUR_ANCHOR_SPEC, _FOUR_CAMERA_WINDOW, _GroundTruthPoseArc
from test_visual_dump import _dumped, _dumping, _evaluate, _scaled_scene

# Two cameras, three time indices, camera-major slots.
_CAMERAS = (0, 1)
_TIMES = (0, 2, 4)
# Four tracks queried at time indices 0, 0, 1 and 2: 2, 3 and 4 live per time,
# and 0, 2 and 5 trail segments -- each live track one per step it has made.
_QUERY_INDEX = (0, 0, 1, 2)
_LIVE = (2, 3, 4)
_TRAILS = (0, 2, 5)
_METADATA = {
    "scene": "0042",
    "step": 6,
    "output_dir": "/runs/arm-b",
    "checkpoint_dir": "/weights/released",
    "commit": "0123abcd",
    "merge_synchronized_slots": True,
    "refine_iters": 4,
    "depth_input": True,
    "camera_input": False,
    "oracle_query_anchor": False,
    "ground_truth_query_anchor": False,
    "query_anchors": ["0:0", "1:0"],
    "base_ratio": {"0:0": None, "1:0": 0.98134},
    "anchor_error_m": {
        "0:0": {"median": 0.12, "mean": 0.12, "p90": 0.12, "count": 1},
        "1:0": {"median": 0.33, "mean": 0.34, "p90": 0.4, "count": 2},
    },
}


def _rotation_about_y(degrees):
    angle = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


def _synthetic_dump(path):
    """A dump with every count known: written and read back through the schema."""

    slot_count, height, width = len(_CAMERAS) * len(_TIMES), 8, 10
    rows, columns = np.arange(0, height, 2), np.arange(0, width, 2)
    grid = (slot_count, rows.size, columns.size)
    camera_of_slot = np.repeat(_CAMERAS, len(_TIMES))
    time_of_slot = np.tile(np.arange(len(_TIMES)), len(_CAMERAS))
    columns_grid, rows_grid = np.meshgrid(columns, rows)
    plane = np.stack(
        [columns_grid * 0.1, rows_grid * 0.1, np.full(rows_grid.shape, 5.0)], axis=-1
    )
    points = np.stack([plane + [0.5 * camera, 0.0, 0.0] for camera in camera_of_slot])
    rgb = np.zeros((*grid, 3), dtype=np.uint8)
    rgb[...] = (np.arange(slot_count) * 40)[:, None, None, None]
    intrinsics = np.array([[8.0, 0.0, 5.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    geometry = {
        "slot_camera_id": camera_of_slot.astype(np.int64),
        "slot_original_time": np.array(_TIMES)[time_of_slot].astype(np.int64),
        "slot_time_index": time_of_slot.astype(np.int64),
        "image_size": np.array([height, width], dtype=np.int64),
        "pixel_rows": rows.astype(np.int64),
        "pixel_columns": columns.astype(np.int64),
        "pred_points_m": (points * 0.98).astype(np.float32),
        "rgb": rgb,
        "gt_points_m": points.astype(np.float32),
        "gt_points_valid": np.ones(grid, dtype=bool),
        "pred_camera_rotation": np.stack(
            [_rotation_about_y(3.0 * camera) for camera in camera_of_slot]
        ).astype(np.float32),
        "pred_camera_center_m": np.stack(
            [[0.98 * camera, 0.0, 0.0] for camera in camera_of_slot]
        ).astype(np.float32),
        "pred_intrinsics": np.repeat(intrinsics[None], slot_count, axis=0).astype(np.float32),
        "gt_camera_rotation": np.repeat(np.eye(3)[None], slot_count, axis=0).astype(
            np.float32
        ),
        "gt_camera_center_m": np.stack(
            [[1.0 * camera, 0.0, 0.0] for camera in camera_of_slot]
        ).astype(np.float32),
        "gt_intrinsics": np.repeat(intrinsics[None], slot_count, axis=0).astype(np.float32),
        "anchor_pred_m": np.array(
            [[0.0, 0.0, 5.1], [1.0, 0.2, 5.3], [1.2, 0.4, 4.8]], dtype=np.float32
        ),
        "anchor_true_m": np.array(
            [[0.0, 0.0, 5.0], [1.0, 0.2, 5.0], [1.2, 0.4, 5.0]], dtype=np.float32
        ),
        "anchor_key": np.array(["0:0", "1:0", "1:0"]),
    }
    time_count, track_count = len(_TIMES), len(_QUERY_INDEX)
    track_gt = (
        np.arange(time_count)[:, None, None] * 0.1
        + np.arange(track_count)[None, :, None]
        + np.array([0.0, 0.0, 5.0])
    ).astype(np.float32)
    visible = np.ones((time_count, track_count), dtype=bool)
    visible[2, 0] = False
    tracks = {
        "pred": track_gt + np.float32(0.05),
        "gt": track_gt,
        "gt_vis_any": visible,
        "query_points": np.concatenate(
            [np.array(_QUERY_INDEX, dtype=np.float32)[:, None], track_gt[0]], axis=1
        ),
        "conf": np.ones((time_count, track_count), dtype=np.float32),
    }
    write_visual_dump(
        path, build_visual_arrays(geometry=geometry, tracks=tracks, metadata=_METADATA)
    )
    return read_visual_dump(path)


@pytest.fixture
def server():
    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        yield server
    finally:
        server.stop()


def test_the_viewer_builds_one_frame_per_time_with_every_layer(tmp_path, server):
    path = tmp_path / "0042.npz"
    arrays, metadata = _synthetic_dump(path)

    viewer = build_viewer(server, arrays, metadata, source=str(path))

    assert viewer.time_count == 3
    assert [frame.name for frame in viewer.frames] == ["/cams/t0", "/cams/t1", "/cams/t2"]
    assert [frame.visible for frame in viewer.frames] == [True, False, False]
    for time_index, nodes in enumerate(viewer.layers):
        assert set(nodes) == set(TIME_LAYERS)
        assert {layer: len(handles) for layer, handles in nodes.items()} == {
            "pred_points": 2,
            "gt_points": 2,
            "pred_frusta": 2,
            "gt_frusta": 2,
            "baselines": 1,
            "gt_tracks": 1,
            "pred_tracks": 1,
            "gt_trails": 1,
            "pred_trails": 1,
        }
        # Every node of a time index follows that index's frame.
        for handles in nodes.values():
            for handle in handles:
                assert handle.name.startswith(f"/cams/t{time_index}/")
        assert nodes["baselines"][0].points.shape == (2, 2, 3)
        assert nodes["gt_tracks"][0].points.shape == (_LIVE[time_index], 3)
        assert nodes["pred_tracks"][0].points.shape == (_LIVE[time_index], 3)
        assert nodes["gt_trails"][0].points.shape == (_TRAILS[time_index], 2, 3)
        assert nodes["pred_trails"][0].points.shape == (_TRAILS[time_index], 2, 3)
        for handle in nodes["gt_frusta"]:
            np.testing.assert_array_equal(handle.color, GROUND_TRUTH)
        for handle in nodes["pred_frusta"]:
            np.testing.assert_array_equal(handle.color, PREDICTED)

    # Track 0 is occluded in every camera at time index 2: its marker dims.
    gt_markers = viewer.layers[2]["gt_tracks"][0].colors
    np.testing.assert_array_equal(gt_markers[0], GROUND_TRUTH_OCCLUDED)
    np.testing.assert_array_equal(gt_markers[1:], np.tile(GROUND_TRUTH, (3, 1)))
    # The baseline segment joins each camera's predicted and true centres.
    np.testing.assert_allclose(
        viewer.layers[0]["baselines"][0].points[1],
        [[0.98, 0.0, 0.0], [1.0, 0.0, 0.0]],
        atol=1e-6,
    )

    assert viewer.anchors["pred"].points.shape == (3, 3)
    assert viewer.anchors["true"].points.shape == (3, 3)
    assert viewer.anchors["error"].points.shape == (3, 2, 3)
    np.testing.assert_array_equal(
        viewer.anchors["pred"].colors,
        [CAMERA_COLORS[0], CAMERA_COLORS[1], CAMERA_COLORS[1]],
    )


def test_the_gui_names_the_model_on_screen(tmp_path, server):
    path = tmp_path / "0042.npz"
    arrays, metadata = _synthetic_dump(path)

    gui = build_viewer(server, arrays, metadata, source=str(path)).gui

    assert {key: gui[key].value for key in ("source", "run", "step", "scene")} == {
        "source": str(path),
        "run": "/runs/arm-b",
        "step": "6",
        "scene": "0042",
    }
    assert gui["checkpoint"].value == "/weights/released"
    assert gui["commit"].value == "0123abcd"
    assert gui["head"].value == (
        "merged head, K=4, inputs: depth, tracks scored on model anchors"
    )
    assert gui["base_ratio 0:0"].value == "-"
    assert gui["base_ratio 1:0"].value == "0.9813"
    assert gui["anchor error 0:0"].value == "median 0.120 m (n=1)"
    assert gui["anchor error 1:0"].value == "median 0.330 m (n=2)"
    assert gui["original_time"].value == "0"


def test_the_controls_drive_the_scene(tmp_path, server):
    path = tmp_path / "0042.npz"
    arrays, metadata = _synthetic_dump(path)
    viewer = build_viewer(server, arrays, metadata, source=str(path))
    gui = viewer.gui

    gui["time"].value = 2
    assert [frame.visible for frame in viewer.frames] == [False, False, True]
    assert gui["original_time"].value == "4"
    gui["next"].value = True
    assert int(gui["time"].value) == 0

    cloud = viewer.layers[1]["pred_points"][1]
    rgb = cloud.colors.copy()
    gui["color"].value = "Per camera"
    np.testing.assert_array_equal(cloud.colors, np.tile(CAMERA_COLORS[1], (len(rgb), 1)))
    gui["color"].value = "RGB"
    np.testing.assert_array_equal(cloud.colors, rgb)

    assert not any(
        handle.visible for nodes in viewer.layers for handle in nodes["gt_points"]
    )
    gui["Ground-truth cloud"].value = True
    assert all(handle.visible for nodes in viewer.layers for handle in nodes["gt_points"])
    gui["Tracks"].value = False
    for nodes in viewer.layers:
        for layer in ("gt_tracks", "pred_tracks", "gt_trails", "pred_trails"):
            assert not nodes[layer][0].visible
    gui["Anchors"].value = False
    assert not any(handle.visible for handle in viewer.anchors.values())


def test_a_non_finite_camera_is_skipped_not_drawn(tmp_path, server):
    path = tmp_path / "0042.npz"
    arrays, metadata = _synthetic_dump(path)
    # Camera 1 at time index 2 is slot 5, camera-major.
    arrays["pred_camera_rotation"][5] = np.nan

    viewer = build_viewer(server, arrays, metadata, source=str(path))

    assert [len(nodes["pred_frusta"]) for nodes in viewer.layers] == [2, 2, 1]
    assert [len(nodes["gt_frusta"]) for nodes in viewer.layers] == [2, 2, 2]


def test_main_binds_loopback_only(tmp_path, monkeypatch):
    """The login node is shared and viser binds every interface by default;
    the SSH tunnel targets localhost, so main() must ask for 127.0.0.1."""

    path = tmp_path / "0042.npz"
    _synthetic_dump(path)
    bound = []

    class _Stop(Exception):
        pass

    def stub(**kwargs):
        bound.append(kwargs)
        raise _Stop

    monkeypatch.setattr(viewer_module.viser, "ViserServer", stub)

    with pytest.raises(_Stop):
        viewer_module.main([str(path)])
    with pytest.raises(_Stop):
        viewer_module.main([str(path), "--port", "9123"])
    assert bound == [
        {"host": "127.0.0.1", "port": 8020},
        {"host": "127.0.0.1", "port": 9123},
    ]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_main_prints_the_port_viser_bound_and_warns_when_it_moved(
    tmp_path, monkeypatch, capsys
):
    """viser 1.0.30 moves to the next free port when the requested one is
    taken -- by a second viewer comparing another step or arm, or by
    upstream's, also on 8020 -- so the URL printed, and the tunnel it names,
    must be the bound port, or they open that other server. Real servers on
    both sides; only the Ctrl+C wait is stubbed out, so main() returns."""

    path = tmp_path / "0042.npz"
    _synthetic_dump(path)
    monkeypatch.setattr(viewer_module, "_wait_for_interrupt", lambda stop: None)

    def launch(port):
        viewer_module.main([str(path), "--port", str(port)])
        captured = capsys.readouterr()
        url = re.search(r"viewer running at http://127\.0\.0\.1:(\d+)", captured.out)
        return int(url.group(1)), captured.err

    free = _free_port()
    bound, err = launch(free)
    assert bound == free
    assert "warning" not in err

    holder = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    try:
        taken = holder.get_port()
        bound, err = launch(taken)
    finally:
        holder.stop()
    assert bound != taken
    assert f"port {taken} is taken, so this viewer is on port {bound}" in err
    assert f"ssh -L {bound}:localhost:{bound}" in err


def test_a_dump_the_eval_writes_builds_in_the_viewer(tmp_path, monkeypatch, server):
    """The round trip: the eval's own dump, at the live four-anchor spec, read
    back and drawn, its readouts in the GUI as metrics.json holds them."""

    scene = _scaled_scene(monkeypatch, **_FOUR_CAMERA_WINDOW)
    model = _GroundTruthPoseArc([scene], [{1: 0.9, 2: 1.25, 3: 0.6}])
    metrics = _evaluate(
        tmp_path, {"0000": scene}, model, query_anchors=_FOUR_ANCHOR_SPEC, **_dumping("0000")
    )
    arrays, metadata = _dumped(tmp_path)

    viewer = build_viewer(server, arrays, metadata, source="dump.npz")

    assert viewer.time_count == 4
    for nodes in viewer.layers:
        assert len(nodes["pred_points"]) == len(nodes["pred_frusta"]) == 4
        assert len(nodes["gt_frusta"]) == 4
    drift = metrics["per_scene"][0]["drift"]
    for key, ratio in drift["base_ratio"].items():
        assert viewer.gui[f"base_ratio {key}"].value == (
            "-" if ratio is None else f"{ratio:.4f}"
        )
    assert json.loads(json.dumps(metadata["anchor_error_m"])) == metrics["per_scene"][0][
        "anchor_error_m"
    ]
