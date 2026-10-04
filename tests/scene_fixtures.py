"""One small Kubric-like world, built in memory through ``build_scene``.

Every scene here is the same world: three tracks drifting across the plane
``z = _PLANE_Z``, seen by ``view_count`` cameras on a pure-x baseline at 56x56,
one of them optionally yawed, pitched and offset (``_world_to_camera``). It is
assembled by :func:`arc.training.build_scene`, the core ``scene_from_datapoint``
uses, so a test runs the same slot arithmetic, anchor resolution and frame
cross-checks a live sample does.

Per-frame depth is the analytic render of the plane, the same at every time
because the plane is static, and ``depth0`` is ``depth[:, 0]``, exactly as
``scene_from_datapoint`` derives it. ``surface_depth_map`` reads ``depth``, so a
test that plants a depth value plants it there.

Frames are PNG bytes decoded by PIL, so a scene holds the pixels a decoded file
would. The fill is distinct per (view, time), which is what lets a frame landing
in the wrong slot show.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from arc.training.dumped_kubric import build_scene

# Every track lies on this world plane, so depth can be rendered analytically
# for any camera pose instead of being hard-coded to a constant.
_PLANE_Z = 5.0
_GRID = 56
_TRACK_COUNT = 3


def _yaw_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cos, 0.0, sin],
            [0.0, 1.0, 0.0],
            [-sin, 0.0, cos],
        ],
        dtype=np.float64,
    )


def _pitch_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos, -sin],
            [0.0, sin, cos],
        ],
        dtype=np.float64,
    )


def _world_to_camera(camera: int, rotated_camera: int | None):
    """Return world-to-camera (R, t) with ``X_cam = R @ X_world + t``.

    The default poses are identity-rotation with a pure-x baseline, which makes
    a w2c/c2w flip and an R/R^T transpose numerically invisible. ``rotated_camera``
    opts one camera into a real yaw and a z offset so those mistakes change the
    projected pixels.
    """

    if camera != rotated_camera:
        rotation = np.eye(3, dtype=np.float64)
        centre = np.array([float(camera), 0.0, 0.0])
    else:
        # Yaw and pitch together, plus y and z offsets, so the projected pixels
        # move non-uniformly in both axes and camera-space z stops being constant.
        rotation = _pitch_rotation(-12.0) @ _yaw_rotation(25.0)
        centre = np.array([float(camera), 0.45, -0.6])
    return rotation, -rotation @ centre


def _render_plane_depth(rotation, translation, intrinsics, height, width):
    """Per-pixel camera-space z of the world plane ``z = _PLANE_Z``.

    For an identity camera with no z offset this is exactly the constant
    ``_PLANE_Z``.
    """

    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack(
        (columns, rows, np.ones_like(columns)),
        axis=-1,
    ).astype(np.float64)
    # X_cam = depth * direction, and X_world = R^T (X_cam - t).
    directions = pixels @ np.linalg.inv(intrinsics).T
    normal = rotation[:, 2]  # third row of R^T
    return ((_PLANE_Z + normal @ translation) / (directions @ normal)).astype(
        np.float32
    )


def _frame_png_bytes(camera: int, time_index: int, height: int, width: int) -> bytes:
    """The PNG one frame holds, filled distinctly per (camera, time)."""

    pixels = np.full(
        (height, width, 3),
        fill_value=20 * camera + time_index,
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def scene_arrays(
    *,
    scene_name: str = "0000",
    time_count: int = 4,
    view_count: int = 2,
    rotated_camera: int | None = None,
    view_ids=None,
    query_times=None,
    invisible=(),
) -> dict:
    """The ``build_scene`` arguments for one fixture scene, frame source included.

    ``query_times`` gives each track its own query frame, ``view_ids`` records
    original camera ids that need not be ``range(view_count)``, and
    ``invisible`` marks ``(camera, time, track)`` triples as occluded.
    """

    height = width = _GRID
    initial_points = np.array(
        [
            [-1.0, -0.5, 5.0],
            [0.0, 0.4, 5.0],
            [1.0, -0.2, 5.0],
        ],
        dtype=np.float32,
    )
    trajectory = np.stack(
        [
            initial_points + np.array([0.1 * time, 0.02 * time, 0.0])
            for time in range(time_count)
        ],
        axis=0,
    ).astype(np.float32)
    if query_times is None:
        query_times = np.zeros(_TRACK_COUNT, dtype=np.int64)
    query_times = np.asarray(query_times, dtype=np.int64)
    query_points = np.concatenate(
        (
            query_times.astype(np.float32)[:, None],
            trajectory[query_times, np.arange(_TRACK_COUNT)],
        ),
        axis=-1,
    )
    visibility = np.ones((view_count, time_count, _TRACK_COUNT), dtype=bool)
    for camera, time_index, track in invisible:
        visibility[camera, time_index, track] = False
    intrinsics = np.zeros((view_count, time_count, 3, 3), dtype=np.float32)
    intrinsics[..., 0, 0] = 30.0
    intrinsics[..., 1, 1] = 30.0
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = height / 2
    intrinsics[..., 2, 2] = 1.0
    extrinsics = np.zeros((view_count, time_count, 3, 4), dtype=np.float32)
    depth = np.zeros((view_count, time_count, 1, height, width), dtype=np.float32)
    for camera in range(view_count):
        rotation, translation = _world_to_camera(camera, rotated_camera)
        extrinsics[camera, :, :3, :3] = rotation.astype(np.float32)
        extrinsics[camera, :, :3, 3] = translation.astype(np.float32)
        # Depth must agree with the pose: build_anchor_correspondences gates on
        # |depth - camera_z| <= 10 cm, so a pose change without a matching
        # depth render rejects every candidate. The plane is static, so the
        # same render is the truth at every time.
        depth[camera, :, 0] = _render_plane_depth(
            rotation,
            translation,
            intrinsics[camera, 0].astype(np.float64),
            height,
            width,
        )

    def open_frames(view_positions, times):
        for camera in view_positions:
            for time_index in times:
                image = Image.open(
                    io.BytesIO(_frame_png_bytes(camera, time_index, height, width))
                )
                image.load()
                yield (
                    camera,
                    time_index,
                    f"{scene_name}/view_{camera}/{time_index:04d}.png",
                    image,
                )

    return {
        "name": scene_name,
        "open_frames": open_frames,
        "query_points": query_points,
        "trajectories": trajectory,
        "visibility": visibility,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "depth0": depth[:, 0].copy(),
        "depth": depth,
        "view_ids": list(range(view_count)) if view_ids is None else list(view_ids),
        "track_upscaling_factor": 1.0,
    }


def fixture_scene(
    *,
    cameras,
    times,
    size,
    query_anchors=None,
    input_depth_max=None,
    input_camera_vectors=False,
    **scene_parameters,
):
    """One window of the fixture world; ``scene_parameters`` go to ``scene_arrays``."""

    return build_scene(
        **scene_arrays(**scene_parameters),
        cameras=cameras,
        times=times,
        query_anchors=query_anchors,
        size=size,
        input_depth_max=input_depth_max,
        input_camera_vectors=input_camera_vectors,
    )


def fixture_datapoint(**scene_parameters) -> SimpleNamespace:
    """A Datapoint-shaped stand-in for the fixture world, for ``scene_from_datapoint``.

    Duck-typed because MVTracker is not importable here: it carries exactly the
    attributes the live front-end reads. ``video`` is decoded from the same PNG
    bytes the fixture's frame source decodes, so a live scene and
    :func:`fixture_scene` see the same pixels.
    """

    arrays = scene_arrays(**scene_parameters)
    view_count, time_count = arrays["visibility"].shape[:2]
    frames = np.zeros((view_count, time_count, 3, _GRID, _GRID), dtype=np.uint8)
    for camera, time_index, _, image in arrays["open_frames"](
        range(view_count), range(time_count)
    ):
        frames[camera, time_index] = np.asarray(image).transpose(2, 0, 1)
    return SimpleNamespace(
        seq_name=arrays["name"],
        video=torch.from_numpy(frames),
        videodepth=torch.from_numpy(arrays["depth"]),
        query_points_3d=torch.from_numpy(arrays["query_points"]),
        trajectory_3d=torch.from_numpy(arrays["trajectories"]),
        visibility=torch.from_numpy(arrays["visibility"]),
        intrs=torch.from_numpy(arrays["intrinsics"]),
        extrs=torch.from_numpy(arrays["extrinsics"]),
        track_upscaling_factor=float(arrays["track_upscaling_factor"]),
        sample_views=list(arrays["view_ids"]),
    )
