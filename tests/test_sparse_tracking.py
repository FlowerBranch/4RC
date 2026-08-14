from __future__ import annotations

import ast
import io
import json
import sys
import zipfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import arc.training.sparse_tracking as sparse_module
import overfit_temporal_tracking as overfit_cli
from arc.models.arc.arc import Arc
from arc.training import (
    DetachedSim3,
    SparseCorrespondences,
    SparseTrackingLossResult,
    build_anchor_correspondences,
    fit_scene_sim3,
    gather_query_anchor_points,
    load_dumped_kubric_scene,
    load_temporal_tracking_checkpoint,
    reconstruction_drift_report,
    save_temporal_tracking_checkpoint,
    sparse_tracking_loss,
)
from arc.models.arc.heads.dpt_head import DPTHead
from arc.models.arc.heads.head_act import activate_head
from arc.models.arc.utils.transform import mat_to_quat, quat_to_mat
from arc.training import (
    ELIGIBILITY_ASSIGNMENT_RULE,
    compose_tracking_loss,
    ELIGIBILITY_REJECTION_STAGES,
    ELIGIBILITY_ROLLUP_RULE,
    sparse_targets,
)
from arc.training.dumped_kubric import compute_image_transform


# Every track lies on this world plane, so depth0 can be rendered analytically
# for any camera pose instead of being hard-coded to a constant.
_PLANE_Z = 5.0


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
    ``_PLANE_Z``, so the unrotated fixture is unchanged.
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
    """The PNG one frame holds, identical whichever layout stores it.

    Both layouts are written from this one function, so a packed-versus-loose
    comparison measures the loader rather than two encodings that happened to
    agree.  The fill is distinct per (camera, time), which is what lets such a
    comparison catch a frame landing in the wrong slot.
    """

    pixels = np.full(
        (height, width, 3),
        fill_value=20 * camera + time_index,
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_scene(
    root: Path,
    *,
    scene_name: str = "0000",
    time_count: int = 4,
    view_count: int = 2,
    rotated_camera: int | None = None,
    depth_sidecar: bool = False,
    sidecar_dtype=np.float32,
    view_ids=None,
    query_times=None,
    invisible=(),
    packed: bool = False,
) -> Path:
    """Write a dump.

    ``depth_sidecar`` also emits ``depth_full.npz``.  The fixture plane is
    static, so per-frame depth is the same analytic render at every time --
    which is the point: ``depth0`` is taken from ``depth[:, 0]`` here exactly as
    the producing side derives it, so the fixture cannot drift away from the
    contract it is used to test.

    ``query_times`` gives each track its own query frame, ``view_ids`` records
    original camera ids that need not be ``range(view_count)``, and
    ``invisible`` marks ``(camera, time, track)`` triples as occluded.

    ``packed`` writes the frames into one ``frames.zip`` instead of loose
    ``view_<v>/`` directories, mirroring the training dump.  This is not a
    fixture convenience: both layouts are permanently live -- the benchmark dump
    stays loose while the training dump packs to fit the cluster's file-count
    quota -- so the flag reproduces a real bimodality the loader must handle.
    ``meta.npz`` and ``depth_full.npz`` stay loose either way.
    """

    scene_path = root / scene_name
    track_count = 3
    height = width = 56

    if packed:
        scene_path.mkdir(parents=True)
        # STORED matches the producing side: PNGs are already compressed, so
        # deflating them again buys nothing.
        with zipfile.ZipFile(
            scene_path / "frames.zip", "w", zipfile.ZIP_STORED
        ) as archive:
            for camera in range(view_count):
                for time_index in range(time_count):
                    archive.writestr(
                        f"view_{camera}/{time_index:04d}.png",
                        _frame_png_bytes(camera, time_index, height, width),
                    )
    else:
        for camera in range(view_count):
            view_path = scene_path / f"view_{camera}"
            view_path.mkdir(parents=True)
            for time_index in range(time_count):
                (view_path / f"{time_index:04d}.png").write_bytes(
                    _frame_png_bytes(camera, time_index, height, width)
                )

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
        query_times = np.zeros(track_count, dtype=np.int64)
    query_times = np.asarray(query_times, dtype=np.int64)
    query_points = np.concatenate(
        (
            query_times.astype(np.float32)[:, None],
            trajectory[query_times, np.arange(track_count)],
        ),
        axis=-1,
    )
    visibility = np.ones(
        (view_count, time_count, track_count),
        dtype=bool,
    )
    for camera, time_index, track in invisible:
        visibility[camera, time_index, track] = False
    intrinsics = np.zeros((view_count, time_count, 3, 3), dtype=np.float32)
    intrinsics[..., 0, 0] = 30.0
    intrinsics[..., 1, 1] = 30.0
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = height / 2
    intrinsics[..., 2, 2] = 1.0
    extrinsics = np.zeros((view_count, time_count, 3, 4), dtype=np.float32)
    depth_full = np.zeros(
        (view_count, time_count, 1, height, width),
        dtype=np.float32,
    )
    for camera in range(view_count):
        rotation, translation = _world_to_camera(camera, rotated_camera)
        extrinsics[camera, :, :3, :3] = rotation.astype(np.float32)
        extrinsics[camera, :, :3, 3] = translation.astype(np.float32)
        # Depth must agree with the pose: build_anchor_correspondences gates on
        # |depth - camera_z| <= 10 cm, so a pose change without a matching
        # depth render rejects every candidate. The plane is static, so the
        # same render is the truth at every time.
        depth_full[camera, :, 0] = _render_plane_depth(
            rotation,
            translation,
            intrinsics[camera, 0].astype(np.float64),
            height,
            width,
        )
    depth_full = depth_full.astype(sidecar_dtype)
    # Exactly how the producing side derives it, which is what makes
    # depth[:, 0] == depth0 an invariant rather than a coincidence.
    depth0 = depth_full[:, 0]

    meta = {
        "query_points": query_points,
        "traj3d_world": trajectory,
        "visibility": visibility,
        "intrs": intrinsics,
        "extrs": extrinsics,
        "depth0": depth0,
        "track_upscaling_factor": np.float64(1.0),
    }
    if view_ids is not None:
        meta["view_ids"] = np.asarray(view_ids, dtype=np.int64)
    np.savez_compressed(scene_path / "meta.npz", **meta)
    if depth_sidecar:
        np.savez_compressed(
            scene_path / "depth_full.npz",
            depth=depth_full,
            seq_name=scene_name,
        )
    return scene_path


@pytest.fixture
def dumped_scene(tmp_path, monkeypatch):
    _write_scene(tmp_path)

    def fake_preprocess_images(
        frames,
        size,
        square_ok,
        verbose,
        patch_size,
    ):
        result = []
        for index, (_name, image) in enumerate(frames):
            width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "true_shape": np.int32(
                        [[transform.output_height, transform.output_width]]
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result

    monkeypatch.setattr(
        "arc.dust3r.utils.image.preprocess_images",
        fake_preprocess_images,
    )
    return load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )


def _identity_alignment() -> DetachedSim3:
    return DetachedSim3(
        scale=torch.tensor(1.0),
        rotation=torch.eye(3),
        translation=torch.zeros(3),
    )


def _perfect_raw_tracks(scene, correspondences):
    height, width = scene.views[0]["img"].shape[-2:]
    tracks = torch.zeros(
        1,
        1,
        scene.num_observations,
        height,
        width,
        3,
    )
    for item in range(correspondences.count):
        trajectory_index = int(correspondences.trajectory_indices[item])
        query_time = int(correspondences.query_times[item])
        row = int(correspondences.rows[item])
        column = int(correspondences.columns[item])
        for slot, original_time in enumerate(scene.slot_times.tolist()):
            tracks[0, 0, slot, row, column] = (
                scene.trajectories_world[original_time, trajectory_index]
                - scene.trajectories_world[query_time, trajectory_index]
            )
    query_anchors = scene.trajectories_world[
        correspondences.query_times,
        correspondences.trajectory_indices,
    ].clone()
    return {
        "track_multi": tracks,
        "track_query_idx": scene.track_query_observation_slots.clone(),
    }, query_anchors


def test_square_512_preprocessing_geometry_is_exact():
    transform = compute_image_transform(512, 512, size=512, patch_size=14)

    assert (
        transform.crop_left,
        transform.crop_top,
        transform.output_width,
        transform.output_height,
    ) == (4, 67, 504, 378)
    mapped = transform.original_to_output(np.array([[256.0, 256.0]]))
    np.testing.assert_allclose(mapped, [[252.0, 189.0]])


def test_production_crop_geometry_is_exact():
    """Cover the geometry the overfit actually runs.

    Every adapter test uses 56x56 at size=56, where crop is (0,0) and scale is 1,
    so output_to_original_indices degenerates to the identity and a crop-offset
    sign flip or a crop_top/crop_left swap is invisible. The real dump is
    384x512 at size=512 -> 378x504 with crop (3,4), asserted nowhere else.
    """

    transform = compute_image_transform(384, 512, size=512, patch_size=14)

    assert (
        transform.crop_top,
        transform.crop_left,
        transform.output_height,
        transform.output_width,
    ) == (3, 4, 378, 504)

    rows, columns = transform.output_to_original_indices()
    np.testing.assert_array_equal(rows, np.arange(378) + 3)
    np.testing.assert_array_equal(columns, np.arange(504) + 4)

    # The forward map must be the inverse of the grid above, not merely close.
    np.testing.assert_allclose(
        transform.original_to_output(np.array([[4.0, 3.0], [507.0, 380.0]])),
        [[0.0, 0.0], [503.0, 377.0]],
    )


def _gradient_png(path: Path, height: int, width: int) -> None:
    """Write a non-uniform, asymmetric image.

    The asymmetry is the point.  A transposed or flipped output carries the same
    sum and standard deviation as the original, so only a spatially varying
    pattern tells them apart -- and rotation is one of the things below is for.
    """

    rows = np.arange(height, dtype=np.float64)[:, None]
    columns = np.arange(width, dtype=np.float64)[None, :]
    pixels = np.stack(
        [
            (rows * 3 + columns * 5) % 256,
            (rows * 7 + columns * 2) % 256,
            (rows * columns) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def _probe_points(tensor):
    """Four corners and two off-centre interior pixels."""

    height, width = tensor.shape[-2:]
    picks = [
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
        (height // 3, width // 4),
        (2 * height // 3, 3 * width // 4),
    ]
    return [
        [float(tensor[0, channel, row, column]) for channel in range(3)]
        for row, column in picks
    ]


# Captured from `load_images` *before* `preprocess_images` was split out of it, so
# these measure the split against what the loader did rather than against itself.
# The cases cover each geometry branch a bad extraction could break: long-edge
# resize with the patch crop, the `size <= 392` short-edge and square crop, and
# the `square_ok=False and W == H` case that sets `halfh = 3 * halfw / 4`.
# `rotate_clockwise_90` and `crop_to_landscape` are app.py's alone and were
# covered nowhere; `landscape_crop` pins that cropping an already-4:3 image is a
# no-op, while `tall_crop` pins that cropping a portrait one is not.
_LOAD_IMAGES_GOLDENS = {
    "landscape_512": {
        "source": (384, 512),
        "kwargs": {"size": 512, "patch_size": 14},
        "shape": (1, 3, 378, 504),
        "total": -2122.8164,
        "std": 0.5800428,
        "probes": [
            [-0.772549, -0.772549, -0.9058824],
            [0.8823529, -0.9137255, 0.8901961],
            [0.0666667, -0.1529412, 0.8823529],
            [-0.2862745, -0.2941176, 0.1607844],
            [-0.8980392, 0.0901961, 0.0196079],
            [-0.0980392, 0.9215686, 0.0196079],
        ],
    },
    "portrait_512": {
        "source": (512, 384),
        "kwargs": {"size": 512, "patch_size": 14},
        "shape": (1, 3, 504, 378),
        "total": -2229.2319,
        "std": 0.5800789,
        "probes": [
            [-0.7882353, -0.7333333, -0.9058824],
            [-0.0588235, -0.8431373, 0.8823529],
            [-1.0, 0.7803922, 0.8901961],
            [-0.2705882, 0.6705883, 0.1607844],
            [0.827451, -0.0745098, -0.654902],
            [0.1450981, 0.0666667, 0.6941177],
        ],
    },
    "square_512": {
        "source": (256, 256),
        "kwargs": {"size": 512, "patch_size": 14},
        "shape": (1, 3, 378, 504),
        "total": -755.418,
        "std": 0.5315253,
        "probes": [
            [-0.145098, 0.8588235, -0.5372549],
            [-0.3254902, 0.7882353, 0.2862746],
            [0.2705883, -0.8431373, 0.5921569],
            [0.0901961, -0.9529412, -0.2549019],
            [-0.2078431, -0.7176471, -0.3568627],
            [0.1921569, 0.7019608, 0.0117648],
        ],
    },
    "square_512_square_ok": {
        "source": (256, 256),
        "kwargs": {"size": 512, "patch_size": 14, "square_ok": True},
        "shape": (1, 3, 504, 504),
        "total": -770.9363,
        "std": 0.5347646,
        "probes": [
            [-0.8901961, -0.8745098, -0.9686275],
            [0.9764706, -0.9921569, 1.0],
            [-0.6862745, 0.8901961, 1.0],
            [0.8352941, 0.8196079, -0.945098],
            [-0.4588235, 0.7098039, 0.6156863],
            [0.4352942, -0.7333333, 0.1843138],
        ],
    },
    "small_224": {
        "source": (96, 64),
        "kwargs": {"size": 224, "patch_size": 14},
        "shape": (1, 3, 224, 224),
        "total": -126.9702,
        "std": 0.5591598,
        "probes": [
            [-0.6392157, -0.145098, -1.0],
            [-0.1607843, 0.8431373, 0.8666667],
            [0.8588235, -0.654902, -1.0],
            [-0.6705883, 0.3333334, 0.1529412],
            [0.4745098, -0.7411765, -0.5137255],
            [0.2313726, 0.9686275, 0.5921569],
        ],
    },
    "rotated": {
        "source": (384, 512),
        "kwargs": {"size": 512, "patch_size": 14, "rotate_clockwise_90": True},
        "shape": (1, 3, 504, 378),
        "total": -2122.8159,
        "std": 0.5800428,
        "probes": [
            [0.0666667, -0.1529412, 0.8823529],
            [-0.772549, -0.772549, -0.9058824],
            [-0.2862745, -0.2941176, 0.1607844],
            [0.8823529, -0.9137255, 0.8901961],
            [0.427451, -0.6705883, -0.6862745],
            [0.5607843, -0.3803921, 0.6627451],
        ],
    },
    # `size <= 392` combined with a rotate is the only case where reading W1/H1
    # before the rotate rather than after it changes the resize target, so
    # without this case that transposition is invisible.
    "rotated_224": {
        "source": (384, 512),
        "kwargs": {"size": 224, "patch_size": 14, "rotate_clockwise_90": True},
        "shape": (1, 3, 224, 224),
        "total": -628.5382,
        "std": 0.4839001,
        "probes": [
            [0.4666667, 0.9058824, -0.0431373],
            [-0.4980392, 0.0196079, -0.8509804],
            [-0.6235294, 0.8901961, -0.1686274],
            [0.4196079, -0.0117647, -0.2784314],
            [-0.8196079, -0.3411765, -0.5843138],
            [-0.3254902, -0.7960784, -0.2784314],
        ],
    },
    "landscape_crop": {
        "source": (384, 512),
        "kwargs": {"size": 512, "patch_size": 14, "crop_to_landscape": True},
        "shape": (1, 3, 378, 504),
        "total": -2122.8164,
        "std": 0.5800428,
        "probes": [
            [-0.772549, -0.772549, -0.9058824],
            [0.8823529, -0.9137255, 0.8901961],
            [0.0666667, -0.1529412, 0.8823529],
            [-0.2862745, -0.2941176, 0.1607844],
            [-0.8980392, 0.0901961, 0.0196079],
            [-0.0980392, 0.9215686, 0.0196079],
        ],
    },
    "tall_crop": {
        "source": (512, 384),
        "kwargs": {"size": 512, "patch_size": 14, "crop_to_landscape": True},
        "shape": (1, 3, 378, 504),
        "total": -2258.3513,
        "std": 0.5363215,
        "probes": [
            [-0.2156863, -0.7098039, -0.2705882],
            [0.5294118, -0.8196079, -0.490196],
            [0.4196079, 0.7568628, 0.254902],
            [-0.8431373, 0.6470588, -0.3019608],
            [-0.3019608, -0.0666667, -0.3568627],
            [-0.7098039, 0.0588236, 0.2],
        ],
    },
}


@pytest.mark.parametrize("case", sorted(_LOAD_IMAGES_GOLDENS))
def test_load_images_is_unchanged_by_the_preprocess_split(tmp_path, case):
    """`load_images` still does exactly what it did before it was split.

    The adapter now calls `preprocess_images` directly so it can hand over a frame
    it already decoded, and `load_images` is the path-opening wrapper left around
    it. Four other callers -- inference.py, app.py, and the two eval launchers --
    still go through the wrapper and must not be able to tell.
    """

    from arc.dust3r.utils.image import load_images

    golden = _LOAD_IMAGES_GOLDENS[case]
    height, width = golden["source"]
    path = tmp_path / f"{case}.png"
    _gradient_png(path, height, width)

    views = load_images([str(path)], verbose=False, **golden["kwargs"])

    assert len(views) == 1
    view = views[0]
    assert tuple(view["img"].shape) == golden["shape"]
    assert view["true_shape"].tolist() == [list(golden["shape"][-2:])]
    assert view["idx"] == 0
    assert view["instance"] == "0"
    assert float(view["img"].sum()) == pytest.approx(golden["total"], abs=1e-2)
    assert float(view["img"].std()) == pytest.approx(golden["std"], abs=1e-6)
    np.testing.assert_allclose(
        _probe_points(view["img"]),
        golden["probes"],
        atol=1e-6,
    )


def test_load_images_still_honours_exif_orientation(tmp_path):
    """A dropped `exif_transpose` is invisible on the dump's own frames.

    Kubric writes PNGs with no EXIF, so every other case here passes with the
    call removed -- and it sits on the line the split moved between functions.
    Orientation 6 means "rotate on display", so the loader must hand back the
    transpose of what is stored.
    """

    from arc.dust3r.utils.image import load_images

    path = tmp_path / "sideways.png"
    _gradient_png(path, 64, 96)
    with Image.open(path) as stored:
        assert stored.size == (96, 64)
    exif = Image.Exif()
    exif[0x0112] = 6
    with Image.open(path) as stored:
        stored.save(path, exif=exif)

    view = load_images([str(path)], size=224, patch_size=14, verbose=False)[0]

    # 96x64 stored, so 64x96 after the transpose: short side 64 resized to 224
    # gives 336x224, square-cropped to 224x224. Without exif_transpose the same
    # arithmetic runs on 96x64 and lands on 224x224 too -- so compare pixels, not
    # only the shape.
    assert tuple(view["img"].shape) == (1, 3, 224, 224)
    upright = load_images(
        [str(_rewritten_without_exif(tmp_path, path))],
        size=224,
        patch_size=14,
        verbose=False,
    )[0]
    assert not torch.equal(view["img"], upright["img"])


def _rewritten_without_exif(tmp_path: Path, source: Path) -> Path:
    """The same pixels as ``source``, stripped of its EXIF block."""

    destination = tmp_path / f"upright_{source.name}"
    with Image.open(source) as image:
        Image.fromarray(np.asarray(image)).save(destination)
    return destination


def test_load_images_still_converts_modes_and_numbers_its_output(tmp_path):
    """Non-RGB input becomes RGB, and `idx` counts the surviving images.

    Kubric renders RGBA, so `convert("RGB")` is load-bearing rather than
    defensive -- without it `ImgNorm` meets a 4- or 1-channel tensor and its
    3-channel normalisation is wrong or raises. `idx` is what the adapter's
    out-of-step guard reads, and a single-image call cannot tell `len(imgs)`
    from a constant 0.
    """

    from arc.dust3r.utils.image import load_images

    rgba = tmp_path / "a_rgba.png"
    grey = tmp_path / "b_grey.png"
    _gradient_png(rgba, 64, 96)
    with Image.open(rgba) as image:
        image.convert("RGBA").save(rgba)
        Image.fromarray(np.asarray(image.convert("L"))).save(grey)

    views = load_images([str(rgba), str(grey)], size=224, patch_size=14, verbose=False)

    assert len(views) == 2
    assert [view["idx"] for view in views] == [0, 1]
    assert [view["instance"] for view in views] == ["0", "1"]
    assert all(view["img"].shape[1] == 3 for view in views)
    # A greyscale source broadcast to RGB has three identical channels; the
    # colour one must not.
    red, green, blue = views[1]["img"][0]
    assert torch.equal(red, green) and torch.equal(green, blue)
    assert not torch.equal(views[0]["img"][0][0], views[0]["img"][0][1])


def test_load_images_still_rejects_a_folder_with_no_images(tmp_path):
    """The empty case names the root, which only the wrapper knows."""

    from arc.dust3r.utils.image import load_images

    (tmp_path / "notes.txt").write_text("not an image")

    with pytest.raises(AssertionError, match=f"no images found at {tmp_path}"):
        load_images(str(tmp_path), size=512, patch_size=14, verbose=False)


def test_adapter_rejects_a_loader_that_reorders_its_output(tmp_path, monkeypatch):
    """Slot s must hold the pixels of paths[s].

    _validate_scene_layout re-derives camera and time from the same slot
    arithmetic the loader used, so it is an identity over the whole input space
    and cannot see a permuted image list.
    """

    _write_scene(tmp_path)

    def reversing_preprocess_images(frames, size, square_ok, verbose, patch_size):
        result = []
        for index, (_name, image) in enumerate(frames):
            width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "true_shape": np.int32(
                        [[transform.output_height, transform.output_width]]
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result[::-1]

    monkeypatch.setattr(
        "arc.dust3r.utils.image.preprocess_images",
        reversing_preprocess_images,
    )

    with pytest.raises(RuntimeError, match="out of step"):
        load_dumped_kubric_scene(
            tmp_path,
            "0000",
            cameras=(0, 1),
            times=(0, 1, 2, 3),
            size=56,
        )


def test_adapter_keeps_eight_camera_major_observations(dumped_scene):
    assert dumped_scene.num_observations == 8
    assert dumped_scene.time_indices == (0, 1, 2, 3, 0, 1, 2, 3)
    assert [
        (observation.slot, observation.camera, observation.original_time)
        for observation in dumped_scene.observations
    ] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
        (3, 0, 3),
        (4, 1, 0),
        (5, 1, 1),
        (6, 1, 2),
        (7, 1, 3),
    ]
    assert [
        int(view["time_index"].item()) for view in dumped_scene.views
    ] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert dumped_scene.query_observation_slot == 0
    assert all(
        view["track_query_idx"].tolist() == [0]
        for view in dumped_scene.views
    )


class _ObservationAxisArc(Arc):
    """Exercise Arc's public input plumbing without constructing ViT-G."""

    def __init__(self):
        nn.Module.__init__(self)
        self.max_time_indices = 32
        self.seen_time_indices = None

    def _forward(
        self,
        images,
        track_query_idx,
        inference_track=True,
        time_indices=None,
        **kwargs,
    ):
        self.seen_time_indices = time_indices.detach().clone()
        batch, observations, _, height, width = images.shape
        return {
            "track_multi": torch.zeros(
                batch,
                len(track_query_idx),
                observations,
                height,
                width,
                3,
            ),
            "track_query_idx": torch.tensor(track_query_idx),
        }


def test_arc_public_forward_keeps_all_eight_observations(dumped_scene):
    model = _ObservationAxisArc()

    output = model(dumped_scene.views, force_no_output_conversion=True)

    assert output["track_multi"].shape[:3] == (1, 1, 8)
    assert torch.equal(
        model.seen_time_indices,
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
    )
    assert torch.equal(output["track_query_idx"], torch.tensor([0]))


def test_single_camera_window_runs_through_arc_and_sparse_loss(tmp_path):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(1,),
        times=(0, 2, 3),
        size=56,
    )
    overfit_cli._validate_scene_layout(scene)

    assert scene.num_observations == 3
    assert scene.time_indices == (0, 1, 2)
    assert scene.slot_cameras.tolist() == [1, 1, 1]
    assert scene.slot_times.tolist() == [0, 2, 3]
    output = _ObservationAxisArc()(
        scene.views,
        force_no_output_conversion=True,
    )
    assert output["track_multi"].shape[:3] == (1, 1, 3)

    correspondences, _ = build_anchor_correspondences(scene)
    raw, query_anchors = _perfect_raw_tracks(scene, correspondences)
    result = sparse_tracking_loss(
        raw,
        scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    assert result.sample_count == correspondences.count * 3


def test_nonfirst_query_camera_owns_alignment_correspondence(tmp_path):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_anchors=((1, 0),),
        size=56,
    )
    query = scene.observations[scene.query_observation_slot]
    assert (query.slot, query.camera, query.original_time) == (2, 1, 0)
    assert all(view["track_query_idx"].tolist() == [2] for view in scene.views)

    # If correspondence construction accidentally uses camera 0, every
    # candidate will fail its depth-consistency gate.
    scene.depth0[0].fill_(100.0)
    correspondences, _ = build_anchor_correspondences(scene)

    assert correspondences.count > 0
    assert len(set(zip(
        correspondences.rows.tolist(),
        correspondences.columns.tolist(),
    ))) == correspondences.count


def test_selected_camera_order_is_preserved_camera_major(tmp_path):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(1, 0),
        times=(0, 3),
        size=56,
    )
    overfit_cli._validate_scene_layout(scene)

    assert [
        (observation.camera, observation.original_time)
        for observation in scene.observations
    ] == [(1, 0), (1, 3), (0, 0), (0, 3)]
    assert scene.time_indices == (0, 1, 0, 1)
    assert scene.query_observation_slot == 0


def test_adapter_supports_more_than_two_selected_cameras(tmp_path):
    _write_scene(tmp_path, time_count=3, view_count=4)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(3, 1, 0),
        times=(0, 2),
        size=56,
    )
    overfit_cli._validate_scene_layout(scene)

    assert scene.num_observations == 6
    assert [
        (observation.camera, observation.original_time)
        for observation in scene.observations
    ] == [(3, 0), (3, 2), (1, 0), (1, 2), (0, 0), (0, 2)]
    assert scene.time_indices == (0, 1, 0, 1, 0, 1)
    assert scene.slot_cameras.tolist() == [3, 3, 1, 1, 0, 0]


def test_adapter_uses_the_real_image_loader(tmp_path):
    _write_scene(tmp_path)

    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    assert scene.num_observations == 8
    assert all(view["img"].shape == (1, 3, 56, 56) for view in scene.views)
    assert scene.time_indices == (0, 1, 2, 3, 0, 1, 2, 3)


def test_packed_and_loose_dumps_load_identically(tmp_path):
    """The layout on disk must not reach the tensors.

    Deliberately unmocked: a fake preprocessor would skip the pixel path, which
    is the only thing this test is about.  Both dumps are written from the same
    `_frame_png_bytes`, so any difference is the loader's doing.
    """

    _write_scene(tmp_path, scene_name="loose")
    _write_scene(tmp_path, scene_name="packed", packed=True)

    assert (tmp_path / "packed" / "frames.zip").is_file()
    assert not (tmp_path / "packed" / "view_0").exists()
    assert (tmp_path / "packed" / "meta.npz").is_file()

    window = dict(cameras=(0, 1), times=(0, 1, 2, 3), size=56)
    loose = load_dumped_kubric_scene(tmp_path, "loose", **window)
    packed = load_dumped_kubric_scene(tmp_path, "packed", **window)

    assert len(packed.views) == len(loose.views) == 8
    for slot, (from_loose, from_packed) in enumerate(zip(loose.views, packed.views)):
        assert torch.equal(from_loose["img"], from_packed["img"]), (
            f"slot {slot} pixels differ between layouts"
        )
        np.testing.assert_array_equal(
            from_loose["true_shape"], from_packed["true_shape"]
        )
        assert from_loose["idx"] == from_packed["idx"] == slot
        assert from_loose["instance"] == from_packed["instance"]
        assert torch.equal(from_loose["time_index"], from_packed["time_index"])
        assert torch.equal(
            from_loose["track_query_idx"], from_packed["track_query_idx"]
        )

    # Equal tensors would prove nothing if every frame looked alike, so confirm
    # the eight are mutually distinct -- that is what makes the loop above able
    # to catch a frame landing in the wrong slot.
    assert len({float(view["img"].mean()) for view in loose.views}) == 8

    assert packed.slot_cameras.tolist() == loose.slot_cameras.tolist()
    assert packed.slot_times.tolist() == loose.slot_times.tolist()
    assert packed.time_indices == loose.time_indices
    assert torch.equal(packed.depth0, loose.depth0)

    # `path` is the one field that must differ: it records where the frame was
    # read from, and for a packed scene that is a member inside the archive.
    assert loose.observations[0].path == tmp_path / "loose" / "view_0" / "0000.png"
    assert (
        packed.observations[0].path
        == tmp_path / "packed" / "frames.zip" / "view_0" / "0000.png"
    )


def test_a_missing_packed_frame_names_the_scene_and_the_member(tmp_path):
    """A gap in the archive must say which frame of which scene is missing.

    `ZipFile.read` raises a bare `KeyError` naming neither, which would surface
    as an unhandled key error halfway through a cluster run.
    """

    _write_scene(tmp_path, scene_name="gappy", packed=True)
    archive_path = tmp_path / "gappy" / "frames.zip"

    with zipfile.ZipFile(archive_path) as source:
        names = source.namelist()
        kept = [
            (name, source.read(name)) for name in names if name != "view_1/0002.png"
        ]
    # Without this the test would still pass if the writer's member spelling ever
    # stopped matching the name above -- having deleted no frame at all.
    assert len(kept) == len(names) - 1

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as target:
        for name, payload in kept:
            target.writestr(name, payload)

    # FileNotFoundError, not KeyError: same failure, same type, as a loose dump
    # missing the same frame.
    with pytest.raises(FileNotFoundError) as raised:
        load_dumped_kubric_scene(
            tmp_path,
            "gappy",
            cameras=(0, 1),
            times=(0, 1, 2, 3),
            size=56,
        )

    message = str(raised.value)
    assert "view_1/0002.png" in message
    assert str(archive_path) in message


def test_the_frame_archive_wins_when_both_layouts_are_present(tmp_path):
    """A scene carrying both layouts reads the archive.

    Packing is a transition: a scene can be packed before its loose frames are
    swept away.  Preferring the loose copy would silently read whichever half
    went stale, so the rule is that the archive decides wherever it exists.
    """

    scene_path = _write_scene(tmp_path, scene_name="both")
    # Offset fills, so which layout was read is visible in the pixels.
    with zipfile.ZipFile(
        scene_path / "frames.zip", "w", zipfile.ZIP_STORED
    ) as archive:
        for camera in range(2):
            for time_index in range(4):
                archive.writestr(
                    f"view_{camera}/{time_index:04d}.png",
                    _frame_png_bytes(camera + 5, time_index, 56, 56),
                )

    scene = load_dumped_kubric_scene(
        tmp_path,
        "both",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    # A 56x56 frame at size=56 is resized and cropped by the identity, and ImgNorm
    # maps a constant fill f to 2f/255 - 1.
    expected = [
        2 * (20 * (camera + 5) + time_index) / 255 - 1
        for camera in range(2)
        for time_index in range(4)
    ]
    np.testing.assert_allclose(
        [float(view["img"].mean()) for view in scene.views],
        expected,
        atol=1e-6,
    )
    assert "frames.zip" in str(scene.observations[0].path)


def test_adapter_parses_nonzero_window_but_sparse_supervision_rejects_it(
    tmp_path,
):
    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0,),
        times=(1, 2, 3),
        size=56,
    )

    assert scene.num_observations == 3
    assert scene.time_indices == (0, 1, 2)
    query = scene.observations[scene.query_observation_slot]
    assert (query.camera, query.original_time) == (0, 1)
    assert not scene.has_time_varying_depth
    # The dump is complete-looking without the opt-in sidecar, so the failure
    # has to name the flag that produces it rather than just refusing.
    with pytest.raises(ValueError, match="RCMV_DUMP_DEPTH=1"):
        build_anchor_correspondences(scene)
    with pytest.raises(ValueError, match="RCMV_DUMP_DEPTH=1"):
        scene.surface_depth_map(0, 1)
    # Time 0 still resolves from meta.npz's depth0 alone.
    assert scene.surface_depth_map(0, 0).shape == scene.depth0.shape[-2:]


def test_overfit_cli_accepts_dynamic_layouts_and_its_flag_requirements():
    """Camera/time layouts, and which flags a run may omit.

    Named for what it checks now: --checkpoint_dir and --output_dir are required
    for a training run and exempted by both --parse_only and --eligibility_only,
    --max_time_indices bounds the semantic times, and an off-t0 anchor is
    *accepted* here because whether it is supportable depends on the per-frame
    depth sidecar, which only the loaded scene knows about.
    """

    parser = overfit_cli.build_arg_parser()
    one_camera = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--checkpoint_dir",
            "checkpoint",
            "--output_dir",
            "output",
            "--cameras",
            "1",
            "--times",
            "0",
            "2",
            "5",
        ]
    )
    overfit_cli._validate_args(one_camera)

    missing_checkpoint = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--cameras",
            "1",
            "--times",
            "0",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="checkpoint_dir"):
        overfit_cli._validate_args(missing_checkpoint)

    parse_only = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--cameras",
            "1",
            "--times",
            "1",
            "2",
            "5",
            "--parse_only",
        ]
    )
    overfit_cli._validate_args(parse_only)
    assert parse_only.checkpoint_dir is None
    assert parse_only.output_dir is None

    # --eligibility_only exempts them too, and the messages must say so.
    eligibility_only = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--cameras",
            "1",
            "--times",
            "1",
            "2",
            "5",
            "--eligibility_only",
        ]
    )
    overfit_cli._validate_args(eligibility_only)
    assert eligibility_only.checkpoint_dir is None
    with pytest.raises(ValueError, match="--parse_only or --eligibility_only"):
        overfit_cli._validate_args(missing_checkpoint)

    # An off-t0 anchor is no longer refused from the flags alone: whether it is
    # supportable depends on the per-frame depth sidecar, which only the loaded
    # scene knows about. The guard moved to DumpedKubricScene.surface_depth_map
    # and names RCMV_DUMP_DEPTH=1 there; see
    # test_adapter_parses_nonzero_window_but_sparse_supervision_rejects_it.
    off_t0_anchor = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--checkpoint_dir",
            "checkpoint",
            "--output_dir",
            "output",
            "--cameras",
            "1",
            "--times",
            "1",
            "2",
            "5",
        ]
    )
    overfit_cli._validate_args(off_t0_anchor)
    assert overfit_cli._resolve_query_anchors(off_t0_anchor) == ((1, 1),)

    too_many_times = parser.parse_args(
        [
            "--data_root",
            "data",
            "--scene",
            "1",
            "--cameras",
            "1",
            "--times",
            "0",
            "2",
            "5",
            "--max_time_indices",
            "2",
            "--parse_only",
        ]
    )
    with pytest.raises(ValueError, match="Selected 3 semantic times"):
        overfit_cli._validate_args(too_many_times)


def _rotated_two_camera_scene(tmp_path, monkeypatch, *, query_camera=1):
    """A scene whose camera 1 has a real yaw/pitch and a matching depth0 render."""

    _write_scene(tmp_path, rotated_camera=1)

    def fake_preprocess_images(frames, size, square_ok, verbose, patch_size):
        result = []
        for index, (_name, image) in enumerate(frames):
            width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "true_shape": np.int32(
                        [[transform.output_height, transform.output_width]]
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result

    monkeypatch.setattr(
        "arc.dust3r.utils.image.preprocess_images", fake_preprocess_images
    )
    return load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_anchors=((query_camera, 0),),
        size=56,
    )


def test_metric_pointmap_lifts_depth0_onto_the_known_world_plane(
    tmp_path,
    monkeypatch,
):
    """Check the world lift against ground truth the fixture knows independently.

    The old Sim(3) test built its source by inverting a known transform applied to
    this same function's output, so the world-lift convention cancelled out and a
    transposed lift or an axis swap passed. Here depth0 renders the plane
    z = _PLANE_Z, so every lifted point must land on that plane -- an invariant
    that R vs R^T and any axis permutation break immediately.
    """

    scene = _rotated_two_camera_scene(tmp_path, monkeypatch)
    world_points, valid = sparse_module._metric_pointmap_at_anchor(
        scene,
        scene.query_observation_slot,
    )

    assert valid.all()
    np.testing.assert_allclose(world_points[..., 2], _PLANE_Z, atol=1e-4)

    # Each anchor pixel must lift back onto its own track, up to the half-pixel
    # rounding in the anchor choice (about 0.18 world units per pixel here).
    correspondences, _ = build_anchor_correspondences(scene)
    for item in range(correspondences.count):
        row = int(correspondences.rows[item])
        column = int(correspondences.columns[item])
        track = int(correspondences.trajectory_indices[item])
        expected = scene.trajectories_world[0, track].numpy()
        assert np.linalg.norm(world_points[row, column] - expected) < 0.15


def test_fit_scene_sim3_reads_the_query_observation_not_slot_zero(
    tmp_path,
    monkeypatch,
):
    """Only the query observation's pointmap may drive the alignment."""

    scene = _rotated_two_camera_scene(tmp_path, monkeypatch)
    query_slot = scene.query_observation_slot
    assert query_slot != 0

    target, _ = sparse_module._metric_pointmap_at_anchor(scene, query_slot)
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 2.25
    translation = np.array([0.3, -0.2, 1.1], dtype=np.float64)

    # Only the query slot carries the correctly pre-transformed source; every
    # other slot holds a decoy that fits a different transform entirely.
    pointmaps = torch.zeros(
        1,
        scene.num_observations,
        target.shape[0],
        target.shape[1],
        3,
    )
    for slot in range(scene.num_observations):
        pointmaps[0, slot] = torch.from_numpy(target).float() * 5.0 + 11.0
    pointmaps[0, query_slot] = torch.from_numpy(
        ((target - translation) @ rotation) / scale
    ).float()

    monkeypatch.setattr(
        sparse_module,
        "_predicted_pointmaps",
        lambda raw: pointmaps,
    )
    raw = {
        "depth_conf": torch.ones(
            1,
            scene.num_observations,
            target.shape[0],
            target.shape[1],
        )
    }

    fitted, report = fit_scene_sim3(raw, scene, confidence_percentile=0)

    assert fitted.scale.item() == pytest.approx(scale, rel=1e-4)
    np.testing.assert_allclose(fitted.rotation.numpy(), rotation, atol=1e-4)
    np.testing.assert_allclose(fitted.translation.numpy(), translation, atol=1e-4)
    assert report["median_residual_metric"] < 1e-5


def test_detached_sim3_rejects_improper_and_non_orthonormal_rotations():
    """These guards exist but no test triggered them."""

    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant"):
        DetachedSim3(
            scale=torch.tensor(1.0),
            rotation=torch.from_numpy(reflection).float(),
            translation=torch.zeros(3),
        )

    with pytest.raises(ValueError, match="orthonormal"):
        DetachedSim3(
            scale=torch.tensor(1.0),
            rotation=torch.tensor(
                [[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            translation=torch.zeros(3),
        )

    with pytest.raises(ValueError, match="scale must be positive"):
        DetachedSim3(
            scale=torch.tensor(-1.0),
            rotation=torch.eye(3),
            translation=torch.zeros(3),
        )


def test_fit_scene_sim3_rejects_collinear_predictions(tmp_path, monkeypatch):
    """The collinearity guard exists but no test reached it."""

    scene = _rotated_two_camera_scene(tmp_path, monkeypatch)
    target, _ = sparse_module._metric_pointmap_at_anchor(
        scene,
        scene.query_observation_slot,
    )
    height, width = target.shape[:2]

    # Every coordinate carries the identical ramp, so the points lie exactly on
    # the (1,1,1) diagonal even in float32. Scaling the axes differently would
    # let float32 rounding lift the cloud off the line and defeat the guard.
    line = torch.zeros(1, scene.num_observations, height, width, 3)
    ramp = torch.linspace(0.0, 1.0, height * width).reshape(height, width)
    for axis in range(3):
        line[0, :, :, :, axis] = ramp

    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: line)
    raw = {"depth_conf": torch.ones(1, scene.num_observations, height, width)}

    with pytest.raises(ValueError, match="collinear"):
        fit_scene_sim3(raw, scene, confidence_percentile=0)


def _project_expected_anchors(scene, camera, rotated_camera, time_index):
    """Independent pinhole oracle for the query anchors.

    Derived from the pose and intrinsics directly, not from
    ``build_anchor_correspondences`` or ``ImageTransform``, so a w2c/c2w flip or
    an R/R^T transpose inside the adapter changes one side but not the other.
    """

    rotation, translation = _world_to_camera(camera, rotated_camera)
    intrinsics = scene.intrinsics[camera, time_index].numpy().astype(np.float64)
    expected = []
    for point in scene.trajectories_world[time_index].numpy().astype(np.float64):
        camera_point = rotation @ point + translation
        homogeneous = intrinsics @ camera_point
        u, v = homogeneous[:2] / homogeneous[2]
        expected.append((int(np.rint(v)), int(np.rint(u))))
    return expected


def test_rotated_query_camera_projects_to_independently_derived_pixels(tmp_path):
    """A real rotation makes the world-to-camera convention observable.

    With every camera at identity rotation and a pure-x baseline, inverting the
    extrinsics or transposing R leaves the projected pixels unchanged (or shifted
    uniformly), and the constant-depth plane means the 10 cm depth gate never
    fires. Both mistakes move these pixels.
    """

    _write_scene(tmp_path, rotated_camera=1)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_anchors=((1, 0),),
        size=56,
    )
    assert (
        scene.observations[scene.query_observation_slot].slot,
        scene.observations[scene.query_observation_slot].camera,
    ) == (2, 1)

    correspondences, _ = build_anchor_correspondences(scene)

    expected = _project_expected_anchors(scene, camera=1, rotated_camera=1, time_index=0)
    # Hand-derived from R = pitch(-12) @ yaw(25), C = (1, 0.45, -0.6), fx=fy=30,
    # cx=cy=28: distinct in both axes, unlike the identity-camera fixture.
    assert expected == [(30, 31), (34, 36), (30, 42)]

    actual = list(
        zip(correspondences.rows.tolist(), correspondences.columns.tolist())
    )
    assert actual == expected


def test_sparse_loss_is_zero_for_a_nonzero_query_slot(tmp_path):
    """Exercise the query-slot -> observation indirection off its identity.

    ``build_anchor_correspondences`` emits query_slot 0 (an index into the track
    query list) while the anchors live at ``scene.query_observation_slot`` (an
    index into the observation axis). Every other numeric test uses a scene where
    those are both 0, so dropping the indirection is numerically invisible.
    """

    _write_scene(tmp_path, rotated_camera=1)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_anchors=((1, 0),),
        size=56,
    )
    assert scene.query_observation_slot == 2

    correspondences, _ = build_anchor_correspondences(scene)
    raw, query_anchors = _perfect_raw_tracks(scene, correspondences)

    result = sparse_tracking_loss(
        raw,
        scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.sample_count == correspondences.count * scene.num_observations
    assert float(result.loss.item()) == pytest.approx(0.0, abs=1e-8)
    assert float(result.metric_error.item()) == pytest.approx(0.0, abs=1e-8)


def test_query_anchor_gather_follows_the_observation_slot(tmp_path):
    """Anchors must be read from the query observation, not from slot 0."""

    _write_scene(tmp_path, rotated_camera=1)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_anchors=((1, 0),),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    height, width = scene.views[0]["img"].shape[-2:]

    # Give every observation a distinct constant pointmap so the gather's choice
    # of observation is readable straight off the returned value.
    pointmaps = torch.zeros(1, scene.num_observations, height, width, 3)
    for slot in range(scene.num_observations):
        pointmaps[0, slot] = float(slot + 1)

    import arc.training.sparse_tracking as module

    original = module._predicted_pointmaps
    try:
        module._predicted_pointmaps = lambda raw: pointmaps
        anchors = gather_query_anchor_points(
            {"track_query_idx": scene.track_query_observation_slots},
            scene,
            correspondences,
        )
    finally:
        module._predicted_pointmaps = original

    expected_value = float(scene.query_observation_slot + 1)
    assert torch.allclose(anchors, torch.full_like(anchors, expected_value))


def test_exit_gate_requires_a_real_margin_not_just_any_decrease():
    """A bare `final < initial` is dominated by reconstruction drift.

    The time embedding is injected at alt_start and every head-feeding out-layer
    is downstream of it, so refitting the Sim(3) moves the loss even when the
    track head improved by exactly nothing. Require a margin.
    """

    common = dict(
        baseline_loss=1.0,
        final_shuffled_loss=None,
        min_improvement=0.01,
        min_index_advantage=0.0,
    )

    # Passes: a 5% drop clears the 1% default margin.
    assert overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=0.95,
        embedding_change=0.5,
        **common,
    ) is None

    # Fails: a 0.1% drop would have passed the old zero-margin comparison.
    reason = overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=0.999,
        embedding_change=0.5,
        **common,
    )
    assert reason is not None
    assert "like-for-like position loss" in reason

    # Fails: loss went up.
    assert overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=1.5,
        embedding_change=0.5,
        **common,
    ) is not None

    # A frozen embedding fails regardless of the loss.
    reason = overfit_cli._exit_criteria_failure(
        initial_loss=1.0,
        final_loss=0.1,
        embedding_change=0.0,
        **common,
    )
    assert reason == "Temporal embedding did not change"


def test_exit_gate_stays_on_position_but_names_the_confidence_term():
    """The gate must not soften when a second term is added, only explain itself.

    A confidence term that buys calibration by giving up track accuracy should
    still read as a failure here -- but a reader who sees a bare position-loss
    failure has no reason to suspect the term they just switched on.
    """

    references = dict(
        baseline_loss=1.0,
        final_shuffled_loss=None,
        min_improvement=0.01,
        min_index_advantage=0.0,
    )
    common = dict(
        initial_loss=1.0, final_loss=0.999, embedding_change=0.5, **references
    )

    position_only = overfit_cli._exit_criteria_failure(**common)
    with_confidence = overfit_cli._exit_criteria_failure(
        **common, confidence_weight=0.001
    )

    # Same verdict either way: the threshold is untouched by the new term.
    assert position_only is not None and with_confidence is not None
    assert with_confidence.startswith(position_only)
    assert "--confidence_weight=0.001" in with_confidence
    assert "final_loss_breakdown" in with_confidence

    # A passing run is silent about it, and a frozen embedding still wins.
    assert overfit_cli._exit_criteria_failure(
        initial_loss=1.0, final_loss=0.5, embedding_change=0.5,
        confidence_weight=1.0, **references,
    ) is None
    assert overfit_cli._exit_criteria_failure(
        initial_loss=1.0, final_loss=0.1, embedding_change=0.0,
        confidence_weight=1.0, **references,
    ) == "Temporal embedding did not change"


def test_min_improvement_is_validated():
    parser = overfit_cli.build_arg_parser()
    base = [
        "--data_root", "root", "--scene", "0000",
        "--checkpoint_dir", "ckpt", "--output_dir", "out",
    ]
    args = parser.parse_args(base)
    assert args.min_improvement == 0.01
    overfit_cli._validate_args(args)

    for bad in ("-0.1", "1.0", "nan"):
        rejected = parser.parse_args(base + ["--min_improvement", bad])
        with pytest.raises(ValueError, match="--min_improvement"):
            overfit_cli._validate_args(rejected)


def test_confidence_flags_default_off_and_are_validated():
    """Defaulting off is what keeps existing invocations byte-for-byte unaffected."""

    parser = overfit_cli.build_arg_parser()
    base = [
        "--data_root", "root", "--scene", "0000",
        "--checkpoint_dir", "ckpt", "--output_dir", "out",
    ]
    args = parser.parse_args(base)
    assert args.confidence_weight == 0.0
    assert args.confidence_alpha == "auto"
    overfit_cli._validate_args(args)
    assert overfit_cli._parse_confidence_alpha(args.confidence_alpha) is None

    for bad in ("-1.0", "nan"):
        rejected = parser.parse_args(base + ["--confidence_weight", bad])
        with pytest.raises(ValueError, match="--confidence_weight"):
            overfit_cli._validate_args(rejected)

    for bad in ("0", "-2", "nan", "sometimes"):
        rejected = parser.parse_args(base + ["--confidence_alpha", bad])
        with pytest.raises(ValueError, match="--confidence_alpha"):
            overfit_cli._validate_args(rejected)

    accepted = parser.parse_args(
        base + ["--confidence_weight", "0.5", "--confidence_alpha", "330"]
    )
    overfit_cli._validate_args(accepted)
    assert overfit_cli._parse_confidence_alpha(accepted.confidence_alpha) == 330.0


# The fields run_summary.json carried before the confidence term existed. Anything
# consuming an archived summary keeps working only if these all survive, so the
# rule is add-only. Checked by reading the source: writing a real summary needs a
# checkpoint and a GPU, and this must stay runnable in CI.
_BASELINE_RUN_SUMMARY_FIELDS = frozenset({
    "scene", "cameras", "times", "observation_count", "time_indices",
    "max_time_indices", "query_observation_slot", "query_camera", "query_time",
    "eligible_query_count", "initial_alignment", "initial_alignment_scale",
    "initial_alignment_rotation", "initial_alignment_translation",
    "final_alignment", "final_alignment_scale", "final_alignment_rotation",
    "final_alignment_translation", "success", "failure_reason", "min_improvement",
    "initial_position_loss", "final_position_loss", "final_position_loss_refit",
    "initial_metric_error_m", "final_metric_error_m", "final_metric_error_refit_m",
    "initial_track_confidence", "final_track_confidence",
    "initial_temporal_embedding_norm", "final_temporal_embedding_norm",
    "temporal_embedding_change", "gradient_norms", "trainable_tensor_count",
    "trainable_parameter_count", "peak_gpu_memory_bytes", "checkpoint_path",
    "seed", "precision", "steps", "learning_rate",
})


def test_run_summary_fields_are_only_ever_added_to():
    source = Path(overfit_cli.__file__).read_text()
    written = None
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "summary"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            written = {key.value for key in node.value.keys}
    assert written is not None, "could not find the run_summary dict literal"

    assert _BASELINE_RUN_SUMMARY_FIELDS <= written
    assert {
        "confidence_weight",
        "confidence_alpha",
        "confidence_alpha_mode",
        "initial_confidence_loss",
        "final_confidence_loss",
        "confidence_gradient_norms",
        "confidence_sample_count",
        "implied_optimal_confidence",
        "initial_confidence_diagnostics",
        "final_confidence_diagnostics",
        "initial_loss_breakdown",
        "final_loss_breakdown",
        "initial_confidence_dropped",
        "final_confidence_dropped",
    } <= written
    # The freeze mask a run actually trained under: the mode name alone does
    # not fix it, and gpu_name is what makes a mixed-hardware comparison
    # detectable in the archive rather than being read as a result.
    assert {"late_global_blocks", "gpu_name"} <= written


def test_confidence_gradient_norms_split_the_shared_output_conv():
    """Row 3 and rows 0-2 of the final conv are reported apart, with no extra backward."""

    conv = nn.Conv2d(2, 4, kernel_size=1)
    conv.weight.grad = torch.zeros_like(conv.weight)
    conv.bias.grad = torch.zeros_like(conv.bias)
    conv.weight.grad[3].fill_(3.0)
    conv.bias.grad[0].fill_(4.0)
    model = SimpleNamespace(
        track_head=SimpleNamespace(
            scratch=SimpleNamespace(output_conv2=[None, None, conv])
        )
    )

    norms = overfit_cli._confidence_gradient_norms(model)

    # Row 3 holds two weights of 3.0 and a zero bias.
    assert norms["track_head_output_conv_confidence_row"] == pytest.approx(
        (2 * 3.0**2) ** 0.5
    )
    assert norms["track_head_output_conv_position_rows"] == pytest.approx(4.0)


def test_confidence_gradient_norms_resolve_against_a_real_dpt_head():
    """The attribute path is the untested part, so walk a real DPTHead, not a stub.

    `_confidence_gradient_norms` reaches `track_head.scratch.output_conv2[2]` and
    assumes it is the 4-channel xyz+conf conv. The sibling test above checks the
    arithmetic against a bare Conv2d; this checks that the path and the channel
    count are actually what the shipped head builds.
    """

    head = DPTHead(
        dim_in=8,
        output_dim=4,
        features=16,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
    )
    output_conv = head.scratch.output_conv2[2]
    assert isinstance(output_conv, nn.Conv2d)
    assert output_conv.out_channels == 4
    assert output_conv.kernel_size == (1, 1)

    output_conv.weight.grad = torch.zeros_like(output_conv.weight)
    output_conv.bias.grad = torch.zeros_like(output_conv.bias)
    output_conv.weight.grad[3].fill_(2.0)
    output_conv.bias.grad[1].fill_(5.0)

    norms = overfit_cli._confidence_gradient_norms(
        SimpleNamespace(track_head=head)
    )

    confidence_elements = output_conv.weight[3].numel()
    assert norms["track_head_output_conv_confidence_row"] == pytest.approx(
        (confidence_elements * 2.0**2) ** 0.5
    )
    assert norms["track_head_output_conv_position_rows"] == pytest.approx(5.0)


def test_confidence_gradient_norms_reject_a_head_that_is_not_xyz_plus_conf():
    """The split is meaningless for another output_dim, so it must fail loudly."""

    head = DPTHead(
        dim_in=8,
        output_dim=2,
        features=16,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
    )

    with pytest.raises(RuntimeError, match="4-channel track output conv"):
        overfit_cli._confidence_gradient_norms(SimpleNamespace(track_head=head))


def test_diagnostics_can_be_skipped_without_touching_the_loss(dumped_scene):
    """The training loop discards the report, and it costs a device sync per figure."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    raw["track_multi"] = raw["track_multi"] + 0.3
    raw["conf_track_multi"] = torch.full(raw["track_multi"].shape[:-1], 120.0)
    call = dict(
        confidence_weight=1.0,
        confidence_alpha=5.0,
    )

    with_report = sparse_tracking_loss(
        raw, dumped_scene, correspondences, _identity_alignment(), query_anchors,
        **call,
    )
    without_report = sparse_tracking_loss(
        raw, dumped_scene, correspondences, _identity_alignment(), query_anchors,
        collect_diagnostics=False, **call,
    )

    assert with_report.diagnostics is not None
    assert without_report.diagnostics is None
    # Skipping the report must not change a single trained quantity.
    assert torch.equal(without_report.loss, with_report.loss)
    assert torch.equal(without_report.total_loss, with_report.total_loss)
    assert torch.equal(without_report.confidence_loss, with_report.confidence_loss)
    assert without_report.confidence_sample_count == with_report.confidence_sample_count
    assert without_report.confidence_dropped == with_report.confidence_dropped


def test_collecting_the_report_moves_no_gradient(dumped_scene):
    """Diagnostics are read-only measurement and must not be able to move a weight.

    The loss values are pinned above; this pins the backward too, which is what the
    exit gates ultimately read through the trained model.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)

    def run(collect):
        raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
        tracks = (raw["track_multi"] + 0.3).detach().requires_grad_(True)
        confidence = torch.full(
            raw["track_multi"].shape[:-1], 120.0, requires_grad=True
        )
        result = sparse_tracking_loss(
            {**raw, "track_multi": tracks, "conf_track_multi": confidence},
            dumped_scene,
            correspondences,
            _identity_alignment(),
            query_anchors,
            confidence_weight=1.0,
            confidence_alpha=5.0,
            collect_diagnostics=collect,
        )
        result.total_loss.backward()
        return result, tracks.grad, confidence.grad

    with_report, with_track_grad, with_confidence_grad = run(True)
    without_report, without_track_grad, without_confidence_grad = run(False)

    assert with_report.diagnostics is not None
    assert without_report.diagnostics is None
    assert torch.equal(with_track_grad, without_track_grad)
    assert torch.equal(with_confidence_grad, without_confidence_grad)


def test_the_resolved_alpha_anchors_the_reported_relative_grid(dumped_scene):
    """Auto-alpha is resolved inside the loss and is what sets the run's own
    confidence scale.  If it did not reach the report, the relative grid would have
    nothing to anchor to and would silently go missing."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    raw["track_multi"] = raw["track_multi"] + 0.3
    raw["conf_track_multi"] = torch.full(raw["track_multi"].shape[:-1], 120.0)

    result = sparse_tracking_loss(
        raw, dumped_scene, correspondences, _identity_alignment(), query_anchors,
        confidence_weight=1.0,
    )
    report = result.diagnostics

    assert result.confidence_alpha is not None
    assert report["implied_optimal_confidence"] == pytest.approx(
        result.confidence_alpha / report["mean_error"]
    )
    first = report["relative_tau_grid"][0]
    assert first["tau"] == pytest.approx(
        first["multiple"] * report["implied_optimal_confidence"]
    )


def test_nonfinite_confidence_samples_are_dropped_and_counted(dumped_scene):
    """`expp1` overflows to inf in BF16, so filtering is right -- but never silent."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    confidence = torch.full(raw["track_multi"].shape[:-1], 80.0)
    row = int(correspondences.rows[0])
    column = int(correspondences.columns[0])
    confidence[0, 0, 3, row, column] = float("inf")
    confidence[0, 0, 5, row, column] = float("nan")
    raw["conf_track_multi"] = confidence

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
        confidence_weight=1.0,
        confidence_alpha=1.0,
    )

    assert result.confidence_dropped["confidence_nonfinite"] == 2
    assert result.confidence_dropped["total"] == 2
    assert result.confidence_dropped["target_nonfinite"] == 0
    assert result.confidence_dropped["prediction_nonfinite"] == 0
    assert result.confidence_sample_count == correspondences.count * 8 - 2
    assert torch.isfinite(result.confidence_loss)


def test_a_clean_run_reports_zero_dropped_confidence_samples(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    raw["conf_track_multi"] = torch.full(raw["track_multi"].shape[:-1], 80.0)

    result = sparse_tracking_loss(
        raw, dumped_scene, correspondences, _identity_alignment(), query_anchors,
        confidence_weight=1.0, confidence_alpha=1.0,
    )

    assert result.confidence_dropped["total"] == 0
    assert result.confidence_sample_count == correspondences.count * 8


def test_the_loss_modules_introduce_no_trainable_parameters():
    """The 231-tensor / 314,600,740-parameter freeze set must stay exact.

    A future term with a learnable temperature would silently break that count, so
    assert the loss surface is parameter-free rather than assuming it.
    """

    import arc.training.diagnostics as diagnostics_module
    import arc.training.losses as losses_module

    for module in (losses_module, diagnostics_module):
        for name in dir(module):
            attribute = getattr(module, name)
            assert not isinstance(attribute, nn.Module), name
            assert not isinstance(attribute, nn.Parameter), name


def test_evaluate_scores_like_for_like_against_the_initial_alignment(monkeypatch):
    """The gated number must reuse the initial alignment and anchors.

    Otherwise `initial_loss` and `final_loss` are measured under two different
    transforms and their difference is not attributable to tracking.
    """

    initial_alignment = _identity_alignment()
    initial_anchors = torch.full((3, 3), 7.0)
    refit_alignment = _identity_alignment()
    refit_anchors = torch.full((3, 3), -1.0)

    calls = []

    # The real dataclass rather than a hand-rolled stub: a stub has to be updated
    # every time the result grows a field, and silently fails the test when it is
    # not. This test is about alignment reuse, so the confidence fields default off.
    def _recorded_result(value):
        return SparseTrackingLossResult(
            loss=torch.tensor(value),
            metric_error=torch.tensor(value * 2),
            sample_count=0,
        )

    def fake_loss(raw, scene, correspondences, alignment, anchors, **kwargs):
        calls.append((alignment, anchors))
        return _recorded_result(0.25 if len(calls) == 1 else 0.75)

    monkeypatch.setattr(overfit_cli, "sparse_tracking_loss", fake_loss)
    monkeypatch.setattr(
        overfit_cli,
        "fit_scene_sim3",
        lambda raw, scene: (refit_alignment, {"pair_count": 1}),
    )
    monkeypatch.setattr(
        overfit_cli,
        "gather_query_anchor_points",
        lambda raw, scene, correspondences: refit_anchors,
    )
    monkeypatch.setattr(
        overfit_cli,
        "_tracking_only",
        lambda raw, keep_confidence=False: raw,
    )
    monkeypatch.setattr(
        overfit_cli,
        "synchronized_consistency_stats",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        overfit_cli,
        "reconstruction_drift_report",
        lambda *args, **kwargs: None,
    )

    class _StubModel:
        def eval(self):
            return self

        def __call__(self, views, **kwargs):
            return {
                "conf_track_multi": torch.full((1, 1, 2, 2, 2), 3.0),
                "track_multi": torch.zeros(1, 1, 2, 2, 2, 3),
            }

    evaluation = overfit_cli._evaluate(
        _StubModel(),
        SimpleNamespace(views=[], slot_time_indices=torch.zeros(0, dtype=torch.long)),
        object(),
        "32",
        0.05,
        initial_alignment,
        initial_anchors,
        sync_metric_scale=1.0,
        shuffled_views=None,
    )

    assert len(calls) == 2
    # First call is the refit (diagnostic), second is the gated like-for-like.
    assert calls[0][0] is refit_alignment and calls[0][1] is refit_anchors
    assert calls[1][0] is initial_alignment and calls[1][1] is initial_anchors
    assert evaluation["loss_refit"] == pytest.approx(0.25)
    assert evaluation["loss"] == pytest.approx(0.75)
    assert evaluation["confidence"]["mean"] == pytest.approx(3.0)


def test_parse_only_main_needs_neither_cuda_nor_checkpoint(
    tmp_path,
    monkeypatch,
    capsys,
):
    _write_scene(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overfit_temporal_tracking.py",
            "--data_root",
            str(tmp_path),
            "--scene",
            "0000",
            "--cameras",
            "1",
            "--times",
            "1",
            "2",
            "3",
            "--parse_only",
        ],
    )

    def unexpected_cuda_check():
        raise AssertionError("parse-only mode must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)
    monkeypatch.setattr(
        Arc,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail(
            "parse-only mode must not load an Arc checkpoint"
        ),
    )

    overfit_cli.main()

    output = capsys.readouterr().out
    assert "observations=3" in output
    assert "time_indices=[0, 1, 2]" in output
    assert "query_observation=slot 0, camera 1, original_time 1" in output
    assert "PASS mvtracker dump parsing" in output


def test_nonconsecutive_frames_keep_local_semantic_time_indices(
    tmp_path,
    monkeypatch,
):
    _write_scene(tmp_path, time_count=7)

    def fake_preprocess_images(frames, size, square_ok, verbose, patch_size):
        result = []
        for index, (_name, image) in enumerate(frames):
            width, height = image.size
            transform = compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            result.append(
                {
                    "img": torch.zeros(
                        1,
                        3,
                        transform.output_height,
                        transform.output_width,
                    ),
                    "idx": index,
                    "instance": str(index),
                }
            )
        return result

    monkeypatch.setattr(
        "arc.dust3r.utils.image.preprocess_images",
        fake_preprocess_images,
    )
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 2, 4, 6),
        size=56,
    )

    assert scene.time_indices == (0, 1, 2, 3, 0, 1, 2, 3)
    assert scene.slot_time_indices.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert scene.slot_times.tolist() == [0, 2, 4, 6, 0, 2, 4, 6]
    assert [
        observation.original_time for observation in scene.observations
    ] == [0, 2, 4, 6, 0, 2, 4, 6]


def test_known_sim3_is_recovered_and_detached(dumped_scene, monkeypatch):
    target, _ = sparse_module._metric_pointmap_at_anchor(dumped_scene, 0)
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 2.25
    translation = np.array([0.3, -0.2, 1.1], dtype=np.float64)
    source = ((target - translation) @ rotation) / scale
    pointmaps = torch.zeros(
        1,
        dumped_scene.num_observations,
        target.shape[0],
        target.shape[1],
        3,
        requires_grad=True,
    )
    pointmaps_data = pointmaps.detach().clone()
    pointmaps_data[0, 0] = torch.from_numpy(source).float()
    pointmaps_data.requires_grad_(True)
    monkeypatch.setattr(
        sparse_module,
        "_predicted_pointmaps",
        lambda raw: pointmaps_data,
    )
    raw = {
        "depth_conf": torch.ones(
            1,
            dumped_scene.num_observations,
            target.shape[0],
            target.shape[1],
        )
    }

    fitted, report = fit_scene_sim3(
        raw,
        dumped_scene,
        confidence_percentile=0,
    )

    assert fitted.scale.item() == pytest.approx(scale, rel=1e-4)
    np.testing.assert_allclose(fitted.rotation.numpy(), rotation, atol=1e-4)
    np.testing.assert_allclose(
        fitted.translation.numpy(),
        translation,
        atol=1e-4,
    )
    assert report["median_residual_metric"] < 1e-5
    assert not fitted.scale.requires_grad
    assert not fitted.rotation.requires_grad
    assert not fitted.translation.requires_grad


def test_predicted_pointmaps_stay_float32_inside_bfloat16_autocast(monkeypatch):
    depth = torch.ones(1, 2, 4, 5, dtype=torch.bfloat16)
    pose_encoding = torch.ones(1, 2, 9, dtype=torch.bfloat16)

    def fake_pose_conversion(converted_pose, image_shape):
        assert converted_pose.dtype == torch.float32
        assert image_shape == (4, 5)
        camera_to_world = torch.eye(4).expand(1, 2, 4, 4).clone()
        intrinsics = torch.eye(3).expand(1, 2, 3, 3).clone()
        return camera_to_world, intrinsics

    def fake_unproject(converted_depth, intrinsics, camera_to_world):
        assert converted_depth.dtype == torch.float32
        # Matmul is deliberately autocast-sensitive. The alignment helper must
        # disable the caller's BF16 autocast before reaching this operation.
        marker = torch.ones(1, 1, dtype=torch.float32)
        marker = marker @ marker
        return marker * torch.ones(1, 2, 4, 5, 3, dtype=torch.float32)

    monkeypatch.setattr(
        sparse_module,
        "pose_encoding_to_extri_intri",
        fake_pose_conversion,
    )
    monkeypatch.setattr(sparse_module, "as_homogeneous", lambda value: value)
    monkeypatch.setattr(sparse_module, "unproject_depth", fake_unproject)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pointmaps = sparse_module._predicted_pointmaps(
            {"depth": depth, "pose_enc": pose_encoding}
        )

    assert pointmaps.dtype == torch.float32
    assert pointmaps.numpy().dtype == np.float32


def test_detached_sim3_stays_float32_inside_bfloat16_autocast():
    points = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    vectors = torch.tensor(
        [[0.25, -0.5, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        alignment = DetachedSim3(
            scale=torch.tensor(2.0, dtype=torch.float32),
            rotation=torch.eye(3, dtype=torch.float32),
            translation=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        ).to(device=torch.device("cpu"), dtype=torch.float32)
        transformed_points = alignment.apply_points(points)
        transformed_vectors = alignment.apply_vectors(vectors)
        loss = transformed_points.sum() + transformed_vectors.sum()

    assert transformed_points.dtype == torch.float32
    assert transformed_vectors.dtype == torch.float32
    loss.backward()
    assert points.grad is not None
    assert vectors.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(vectors.grad).all()


def test_correspondence_is_direct_projection_and_detached(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)

    assert correspondences.count == 3
    assert correspondences.trajectory_indices.tolist() == [0, 1, 2]
    assert correspondences.query_slots.tolist() == [0, 0, 0]
    assert correspondences.query_times.tolist() == [0, 0, 0]
    assert not correspondences.rows.requires_grad
    assert not correspondences.columns.requires_grad
    assert correspondences.rows.tolist() == [25, 30, 27]
    assert correspondences.columns.tolist() == [22, 28, 34]
    assert len(set(zip(
        correspondences.rows.tolist(),
        correspondences.columns.tolist(),
    ))) == 3


def test_query_pointmap_anchor_is_gathered_and_detached(
    dumped_scene,
    monkeypatch,
):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    pointmaps = torch.arange(
        dumped_scene.num_observations * height * width * 3,
        dtype=torch.float32,
    ).reshape(1, dumped_scene.num_observations, height, width, 3)
    pointmaps.requires_grad_(True)
    monkeypatch.setattr(
        sparse_module,
        "_predicted_pointmaps",
        lambda raw: pointmaps,
    )
    raw = {
        "track_query_idx": dumped_scene.track_query_observation_slots,
    }

    anchors = gather_query_anchor_points(
        raw,
        dumped_scene,
        correspondences,
    )

    expected = pointmaps[
        0,
        0,
        correspondences.rows,
        correspondences.columns,
    ]
    torch.testing.assert_close(anchors, expected)
    assert not anchors.requires_grad
    # The anchor gather reads reconstruction only -- depth and pose_enc -- so
    # one shared forward serves every anchor. It indexes by the adapter's anchor
    # list, never by a forward's track_query_idx, which is why a dict carrying
    # no track queries at all still works.
    torch.testing.assert_close(
        gather_query_anchor_points({}, dumped_scene, correspondences),
        expected,
    )
    stray = SparseCorrespondences(
        trajectory_indices=correspondences.trajectory_indices,
        query_slots=torch.ones_like(correspondences.query_slots),
        query_times=correspondences.query_times,
        rows=correspondences.rows,
        columns=correspondences.columns,
    )
    with pytest.raises(ValueError, match="exceeds the adapter's query observations"):
        gather_query_anchor_points(raw, dumped_scene, stray)


def test_sparse_loss_rejects_queries_that_are_not_declared_anchors(dumped_scene):
    """The query-vs-anchor check lives where the tracks are scored.

    Supervising several anchors runs one head pass per anchor, so a forward
    carries a subsequence of the adapter's anchors rather than all of them. What
    must still be impossible is scoring one anchor's field against another's
    pixels.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)

    raw["track_query_idx"] = torch.tensor([1])
    with pytest.raises(ValueError, match="ordered subsequence"):
        sparse_tracking_loss(
            raw,
            dumped_scene,
            correspondences,
            _identity_alignment(),
            query_anchors,
        )


def test_perfect_sparse_tracks_have_numerical_zero_loss(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    assert result.metric_error.item() == pytest.approx(0.0, abs=1e-8)
    assert result.sample_count == correspondences.count * 8


def test_known_perturbation_increases_loss(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    perfect, query_anchors = _perfect_raw_tracks(
        dumped_scene,
        correspondences,
    )
    baseline = sparse_tracking_loss(
        perfect,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )
    perturbed_tracks = perfect["track_multi"].clone()
    perturbed_tracks[
        0,
        0,
        7,
        correspondences.rows[0],
        correspondences.columns[0],
        0,
    ] += 0.2
    perturbed = sparse_tracking_loss(
        {
            "track_multi": perturbed_tracks,
            "track_query_idx": perfect["track_query_idx"],
        },
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert perturbed.loss.item() > baseline.loss.item()
    assert perturbed.metric_error.item() > baseline.metric_error.item()


def test_invisible_target_does_not_contribute(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    trajectory_index = int(correspondences.trajectory_indices[0])
    row = int(correspondences.rows[0])
    column = int(correspondences.columns[0])
    # Slot 7 is camera 1 / time 3.
    dumped_scene.visibility[1, 3, trajectory_index] = False
    raw["track_multi"][0, 0, 7, row, column, 0] += 100.0

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    assert result.metric_error.item() == pytest.approx(0.0, abs=1e-8)
    assert result.sample_count == correspondences.count * 8 - 1


def test_loss_preserves_eight_observation_axis(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    with pytest.raises(ValueError, match="observation slots"):
        sparse_tracking_loss(
            {
                "track_multi": raw["track_multi"][:, :, :7],
                "track_query_idx": raw["track_query_idx"],
            },
            dumped_scene,
            correspondences,
            _identity_alignment(),
            query_anchors,
        )


def test_nonidentity_sim3_absolute_position_loss_is_zero(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    angle = torch.tensor(0.35)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    alignment = DetachedSim3(
        scale=torch.tensor(1.7),
        rotation=rotation,
        translation=torch.tensor([0.4, -0.8, 1.2]),
    )
    dumped_scene.track_upscaling_factor = 2.5
    query_anchors = torch.tensor(
        [
            [-0.4, 0.2, 1.1],
            [0.3, -0.7, 2.0],
            [1.2, 0.1, -0.5],
        ],
        requires_grad=True,
    )
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    tracks = torch.zeros(
        1,
        1,
        dumped_scene.num_observations,
        height,
        width,
        3,
    )
    for item, trajectory_index in enumerate(
        correspondences.trajectory_indices.tolist()
    ):
        row = int(correspondences.rows[item])
        column = int(correspondences.columns[item])
        for slot, original_time in enumerate(dumped_scene.slot_times.tolist()):
            target = dumped_scene.trajectories_world[
                original_time,
                trajectory_index,
            ]
            target_in_predicted_world = (
                (target - alignment.translation) @ alignment.rotation
            ) / alignment.scale
            tracks[0, 0, slot, row, column] = (
                target_in_predicted_world - query_anchors.detach()[item]
            )
    tracks.requires_grad_(True)
    raw = {
        "track_multi": tracks,
        "track_query_idx": dumped_scene.track_query_observation_slots,
    }

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        alignment,
        query_anchors,
    )
    result.loss.backward()

    assert result.loss.item() == pytest.approx(0.0, abs=2e-7)
    assert result.metric_error.item() == pytest.approx(0.0, abs=2e-6)
    assert query_anchors.grad is None
    assert tracks.grad is not None


def test_depth_inconsistent_rounded_anchor_is_rejected(dumped_scene):
    all_correspondences, _ = build_anchor_correspondences(dumped_scene)
    row = int(all_correspondences.rows[0])
    column = int(all_correspondences.columns[0])
    transform = dumped_scene.observations[0].image_transform
    original_row = int(np.rint((row + transform.crop_top) / transform.scale_y))
    original_column = int(
        np.rint((column + transform.crop_left) / transform.scale_x)
    )
    dumped_scene.depth0[0, 0, original_row, original_column] = 8.0

    filtered, _ = build_anchor_correspondences(dumped_scene)

    assert filtered.trajectory_indices.tolist() == [1, 2]


def test_duplicate_dense_anchor_keeps_the_best_depth_match(dumped_scene):
    farther = torch.tensor([-1.0, -0.5, 5.05])
    nearer = torch.tensor([-1.0, -0.5, 5.0])
    dumped_scene.query_points[0, 1:] = farther
    dumped_scene.trajectories_world[0, 0] = farther
    dumped_scene.query_points[1, 1:] = nearer
    dumped_scene.trajectories_world[0, 1] = nearer

    correspondences, _ = build_anchor_correspondences(dumped_scene)

    assert correspondences.trajectory_indices.tolist() == [1, 2]
    pixels = list(zip(
        correspondences.rows.tolist(),
        correspondences.columns.tolist(),
    ))
    assert len(pixels) == len(set(pixels))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query_slots", -1, "query slots must be non-negative"),
        ("trajectory_indices", -1, "trajectory index"),
        ("query_times", -1, "query time"),
    ],
)
def test_sparse_loss_rejects_wrapping_negative_indices(
    dumped_scene,
    field,
    value,
    message,
):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    values = {
        name: getattr(correspondences, name).clone()
        for name in (
            "trajectory_indices",
            "query_slots",
            "query_times",
            "rows",
            "columns",
        )
    }
    values[field][0] = value
    invalid = SparseCorrespondences(**values)

    with pytest.raises(ValueError, match=message):
        sparse_tracking_loss(
            raw,
            dumped_scene,
            invalid,
            _identity_alignment(),
            query_anchors,
        )


class _TinyPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_index_embedding = nn.Embedding(4, 2)
        self.frozen_backbone_weight = nn.Parameter(torch.ones(1))


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = _TinyPretrained()


def _tiny_arc():
    model = Arc.__new__(Arc)
    nn.Module.__init__(model)
    model.backbone = _TinyBackbone()
    model.head = nn.Linear(1, 1)
    model.cam_dec = nn.Linear(1, 1)
    model.motion_decoder = nn.Linear(1, 1)
    model.track_head = nn.Linear(1, 1)
    model.set_freeze("temporal_tracking")
    return model


def test_only_temporal_tracking_parameters_receive_gradients(dumped_scene):
    model = _tiny_arc()
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    value = (
        model.backbone.pretrained.time_index_embedding.weight.sum()
        + model.backbone.pretrained.frozen_backbone_weight.sum()
        + model.head.weight.sum()
        + model.head.bias.sum()
        + model.cam_dec.weight.sum()
        + model.cam_dec.bias.sum()
        + model.motion_decoder.weight.sum()
        + model.motion_decoder.bias.sum()
        + model.track_head.weight.sum()
        + model.track_head.bias.sum()
    )
    tracks = value.expand(
        1,
        1,
        dumped_scene.num_observations,
        height,
        width,
        3,
    )
    query_anchors = dumped_scene.trajectories_world[
        correspondences.query_times,
        correspondences.trajectory_indices,
    ]
    result = sparse_tracking_loss(
        {
            "track_multi": tracks,
            "track_query_idx": dumped_scene.track_query_observation_slots,
        },
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )
    result.loss.backward()

    expected_prefixes = (
        "backbone.pretrained.time_index_embedding.",
        "motion_decoder.",
        "track_head.",
    )
    for name, parameter in model.named_parameters():
        should_train = name.startswith(expected_prefixes)
        assert parameter.requires_grad is should_train
        assert (parameter.grad is not None) is should_train


def _shared_conv_predictions(scene, seed=0):
    """Reproduce the track head's real output split: one conv, then `activate_head`.

    The whole reason the confidence channel is delicate is that xyz and confidence
    come off the *same* ``Conv2d(_, 4, 1)`` and are only separated afterwards in
    tensor space.  Testing against a hand-built tensor pair would not exercise that;
    this drives the actual conv and the actual ``inv_log``/``expp1`` activations, so
    the per-row gradient claims are about the real mechanism.
    """

    torch.manual_seed(seed)
    height, width = scene.views[0]["img"].shape[-2:]
    conv = nn.Conv2d(2, 4, kernel_size=1)
    features = torch.randn(scene.num_observations, 2, height, width)
    track, confidence = activate_head(
        conv(features),
        activation="inv_log",
        conf_activation="expp1",
    )
    raw = {
        "track_multi": track[None, None],
        "conf_track_multi": confidence[None, None],
        "track_query_idx": scene.track_query_observation_slots.clone(),
    }
    return conv, raw


def _anchors_for(scene, correspondences):
    return scene.trajectories_world[
        correspondences.query_times,
        correspondences.trajectory_indices,
    ].clone()


def test_confidence_gradient_reaches_the_confidence_row_and_only_it(dumped_scene):
    """The deliverable's central claim, tested on the real shared conv.

    Row 3 of the output conv must move only because of the confidence term, and
    rows 0-2 must be exactly what position-only training would have produced.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    conv, raw = _shared_conv_predictions(dumped_scene)
    anchors = _anchors_for(dumped_scene, correspondences)

    position_only = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
    )
    position_only.total_loss.backward(retain_graph=True)
    position_rows = conv.weight.grad[:3].clone()
    position_bias = conv.bias.grad[:3].clone()
    # The position term cannot reach the confidence channel at all.
    assert torch.count_nonzero(conv.weight.grad[3]) == 0
    conv.zero_grad(set_to_none=True)

    with_confidence = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
        confidence_weight=1.0,
        confidence_alpha=100.0,
    )
    with_confidence.total_loss.backward(retain_graph=True)

    assert torch.count_nonzero(conv.weight.grad[3]) > 0
    torch.testing.assert_close(conv.weight.grad[:3], position_rows)
    torch.testing.assert_close(conv.bias.grad[:3], position_bias)

    # The converse: the confidence term on its own contributes nothing to xyz.
    weight_grad, bias_grad = torch.autograd.grad(
        with_confidence.confidence_loss,
        [conv.weight, conv.bias],
        retain_graph=True,
    )
    assert torch.count_nonzero(weight_grad[:3]) == 0
    assert torch.count_nonzero(bias_grad[:3]) == 0
    assert torch.count_nonzero(weight_grad[3]) > 0


def test_total_loss_is_the_position_loss_when_confidence_is_disabled(dumped_scene):
    """Off means never built, not multiplied by zero, so archived runs reproduce."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    assert "conf_track_multi" not in raw

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
    )

    assert result.total_loss is result.loss
    assert result.confidence_loss is None
    assert result.confidence_alpha is None
    assert result.confidence_sample_count is None
    assert result.diagnostics is None
    assert result.loss_breakdown is None


def test_confidence_term_supervises_samples_the_position_mask_drops(dumped_scene):
    """Occluded points are the signal for low confidence, so they must be included."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    raw["conf_track_multi"] = torch.full(raw["track_multi"].shape[:-1], 50.0)
    trajectory_index = int(correspondences.trajectory_indices[0])
    # Slot 7 is camera 1 / time 3.
    dumped_scene.visibility[1, 3, trajectory_index] = False

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
        confidence_weight=1.0,
        confidence_alpha=1.0,
    )

    assert result.sample_count == correspondences.count * 8 - 1
    assert result.confidence_sample_count == correspondences.count * 8
    assert result.diagnostics["occluded_count"] == 1
    assert result.diagnostics["visible_count"] == correspondences.count * 8 - 1


def test_auto_alpha_puts_the_optimum_at_the_gathered_operating_point(dumped_scene):
    """Resolved from this call's own samples, so the two statistics are commensurate."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    raw["track_multi"] = raw["track_multi"] + 0.4
    raw["conf_track_multi"] = torch.full(raw["track_multi"].shape[:-1], 200.0)

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        query_anchors,
        confidence_weight=1.0,
    )
    diagnostics = result.diagnostics

    assert result.confidence_alpha == pytest.approx(
        diagnostics["mean_confidence"] * diagnostics["mean_error"]
    )
    assert result.confidence_alpha / diagnostics["mean_error"] == pytest.approx(
        diagnostics["mean_confidence"],
        rel=1e-5,
    )


def test_confidence_term_requires_the_confidence_prediction(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)

    with pytest.raises(KeyError, match="conf_track_multi"):
        sparse_tracking_loss(
            raw,
            dumped_scene,
            correspondences,
            _identity_alignment(),
            query_anchors,
            confidence_weight=1.0,
            confidence_alpha=1.0,
        )


def test_confidence_term_rejects_a_mismatched_confidence_shape(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)
    raw["conf_track_multi"] = torch.ones(1, 1, 2, 3, 4)

    with pytest.raises(ValueError, match="conf_track_multi"):
        sparse_tracking_loss(
            raw,
            dumped_scene,
            correspondences,
            _identity_alignment(),
            query_anchors,
            confidence_weight=1.0,
            confidence_alpha=1.0,
        )


def test_negative_confidence_weight_is_rejected(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw, query_anchors = _perfect_raw_tracks(dumped_scene, correspondences)

    with pytest.raises(ValueError, match="confidence_weight"):
        sparse_tracking_loss(
            raw,
            dumped_scene,
            correspondences,
            _identity_alignment(),
            query_anchors,
            confidence_weight=-1.0,
        )


def test_confidence_term_leaves_frozen_parameters_without_gradients(dumped_scene):
    """The freeze invariant must survive the extra term, not just the position one."""

    model = _tiny_arc()
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    value = (
        model.backbone.pretrained.time_index_embedding.weight.sum()
        + model.backbone.pretrained.frozen_backbone_weight.sum()
        + model.head.weight.sum()
        + model.head.bias.sum()
        + model.cam_dec.weight.sum()
        + model.cam_dec.bias.sum()
        + model.motion_decoder.weight.sum()
        + model.motion_decoder.bias.sum()
        + model.track_head.weight.sum()
        + model.track_head.bias.sum()
    )
    shape = (1, 1, dumped_scene.num_observations, height, width)
    result = sparse_tracking_loss(
        {
            "track_multi": value.expand(*shape, 3),
            # Mirror `expp1`: strictly positive whatever the parameters are.
            "conf_track_multi": (1 + value.exp()).expand(*shape),
            "track_query_idx": dumped_scene.track_query_observation_slots,
        },
        dumped_scene,
        correspondences,
        _identity_alignment(),
        _anchors_for(dumped_scene, correspondences),
        confidence_weight=1.0,
        confidence_alpha=10.0,
    )
    result.total_loss.backward()

    expected_prefixes = (
        "backbone.pretrained.time_index_embedding.",
        "motion_decoder.",
        "track_head.",
    )
    for name, parameter in model.named_parameters():
        should_train = name.startswith(expected_prefixes)
        assert parameter.requires_grad is should_train
        assert (parameter.grad is not None) is should_train


def test_save_reload_preserves_temporal_embedding(tmp_path):
    source = _tiny_arc()
    with torch.no_grad():
        source.backbone.pretrained.time_index_embedding.weight.fill_(3.25)
        source.motion_decoder.weight.fill_(1.5)
        source.track_head.bias.fill_(-2.0)
    checkpoint = save_temporal_tracking_checkpoint(
        source,
        tmp_path / "temporal_tracking.pt",
    )
    target = _tiny_arc()

    load_temporal_tracking_checkpoint(target, checkpoint)

    torch.testing.assert_close(
        target.backbone.pretrained.time_index_embedding.weight,
        source.backbone.pretrained.time_index_embedding.weight,
    )
    torch.testing.assert_close(
        target.motion_decoder.weight,
        source.motion_decoder.weight,
    )
    torch.testing.assert_close(
        target.track_head.bias,
        source.track_head.bias,
    )


# ------------------------------------------------------------------------------
# synchronized-pair consistency term in sparse_tracking_loss
# ------------------------------------------------------------------------------


def test_sparse_loss_sync_term_composes_and_defaults_off(dumped_scene):
    correspondences, _ = build_anchor_correspondences(dumped_scene)
    _, raw = _shared_conv_predictions(dumped_scene)
    anchors = _anchors_for(dumped_scene, correspondences)

    base = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
    )
    with_sync = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
        sync_weight=0.5,
    )

    # Default path untouched: no sync graph is built at weight 0.
    assert base.sync_loss is None
    assert base.total_loss is base.loss

    # 2 cameras x 4 times: one synchronized pair per time.
    assert with_sync.sync_pair_count == 4
    assert with_sync.sync_loss is not None
    assert with_sync.sync_loss.item() > 0
    torch.testing.assert_close(with_sync.loss, base.loss)
    torch.testing.assert_close(
        with_sync.total_loss,
        with_sync.loss + 0.5 * with_sync.sync_loss,
    )
    assert set(with_sync.loss_breakdown) == {"position", "sync"}


def test_sparse_loss_sync_term_is_zero_for_view_consistent_fields(
    dumped_scene,
):
    """Identical dP fields for synchronized slots cost exactly nothing."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    per_time = torch.randn(1, 1, len(dumped_scene.times), height, width, 3)
    # Camera-major layout: repeat the per-time fields for the second camera.
    tracks = per_time.repeat(1, 1, len(dumped_scene.cameras), 1, 1, 1)
    raw = {
        "track_multi": tracks,
        "track_query_idx": dumped_scene.track_query_observation_slots.clone(),
    }
    anchors = _anchors_for(dumped_scene, correspondences)

    result = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
        sync_weight=1.0,
    )

    assert result.sync_loss.item() == 0.0
    torch.testing.assert_close(result.total_loss, result.loss)


# ------------------------------------------------------------------------------
# reconstruction drift vs. dump ground truth
# ------------------------------------------------------------------------------


def _ground_truth_raw_reconstruction(scene):
    """depth and pose_enc that reproduce the dump exactly under identity Sim(3)."""

    height, width = scene.views[0]["img"].shape[-2:]
    depth = torch.full((1, scene.num_observations, height, width), 5.0)
    pose_encoding = torch.zeros(1, scene.num_observations, 9)
    for observation in scene.observations:
        if observation.original_time == 0:
            rows, columns = observation.image_transform.output_to_original_indices()
            columns_grid, rows_grid = np.meshgrid(columns, rows)
            sampled = scene.depth0[observation.camera, 0].numpy()[
                rows_grid, columns_grid
            ]
            depth[0, observation.slot] = torch.from_numpy(sampled).float()
        world_to_camera = scene.extrinsics_world_to_camera[
            observation.camera, observation.original_time
        ].double()
        rotation = world_to_camera[:3, :3]
        camera_to_world_rotation = rotation.mT
        centre = -(rotation.mT @ world_to_camera[:3, 3])
        pose_encoding[0, observation.slot, :3] = centre.float()
        pose_encoding[0, observation.slot, 3:7] = mat_to_quat(
            camera_to_world_rotation
        ).float()
        pose_encoding[0, observation.slot, 7:9] = 1.0
    return {"depth": depth, "pose_enc": pose_encoding}


def test_drift_report_is_zero_for_ground_truth_predictions(dumped_scene):
    raw = _ground_truth_raw_reconstruction(dumped_scene)

    report = reconstruction_drift_report(
        raw,
        dumped_scene,
        _identity_alignment(),
    )

    assert set(report["depth"]) == {"0", "1"}
    for camera_report in report["depth"].values():
        assert camera_report["median_relative_error"] < 1e-5
        assert camera_report["p90_relative_error"] < 1e-5
    assert report["pose"]["rotation_error_deg"]["max"] < 1e-3
    assert report["pose"]["camera_center_error_m"]["max"] < 1e-4
    # The anchor-referenced figures agree, and both groups are populated: the
    # window has cameras apart from the anchor's and timesteps of its own.
    pose = report["pose"]
    for figure in ("relative_rotation_deg", "relative_center_error_m"):
        for group in ("cross_camera", "static_camera"):
            assert pose[figure][group] is not None
            assert pose[figure][group]["max"] < 1e-3
    assert pose["baseline_scale"] == pytest.approx(1.0, rel=1e-5)


def test_drift_report_reads_a_depth_inflation_as_relative_error(dumped_scene):
    raw = _ground_truth_raw_reconstruction(dumped_scene)
    raw["depth"] = raw["depth"] * 1.1

    report = reconstruction_drift_report(
        raw,
        dumped_scene,
        _identity_alignment(),
    )

    for camera_report in report["depth"].values():
        assert camera_report["median_relative_error"] == pytest.approx(0.1, rel=1e-3)


def test_drift_report_reads_a_camera_translation_as_center_error(dumped_scene):
    raw = _ground_truth_raw_reconstruction(dumped_scene)
    raw["pose_enc"][0, :, 0] += 0.25  # move every camera centre 25 cm in x

    report = reconstruction_drift_report(
        raw,
        dumped_scene,
        _identity_alignment(),
    )

    assert report["pose"]["camera_center_error_m"]["mean"] == pytest.approx(
        0.25, rel=1e-4
    )
    assert report["pose"]["rotation_error_deg"]["max"] < 1e-3
    # Moving the whole rig is a change of gauge, not a pose error. This is the
    # difference the anchor-referenced figures exist to draw: the alignment-
    # composed number above reads 25 cm, these read nothing.
    pose = report["pose"]
    for group in ("cross_camera", "static_camera"):
        assert pose["relative_center_error_m"][group]["max"] < 1e-4
        assert pose["relative_rotation_deg"][group]["max"] < 1e-3
    assert pose["baseline_scale"] == pytest.approx(1.0, rel=1e-5)


def _apply_predicted_gauge(raw, rotation, scale):
    """Rotate and rescale the *predicted* world frame, leaving the dump alone."""

    pose_encoding = raw["pose_enc"].clone()
    rotation = torch.as_tensor(rotation, dtype=torch.float64)
    for slot in range(pose_encoding.shape[1]):
        centre = pose_encoding[0, slot, :3].double()
        camera_to_world = quat_to_mat(pose_encoding[0, slot, 3:7].double())
        pose_encoding[0, slot, :3] = (scale * (rotation @ centre)).float()
        pose_encoding[0, slot, 3:7] = mat_to_quat(
            rotation @ camera_to_world
        ).float()
    return {**raw, "pose_enc": pose_encoding}


def _moved_center(raw, slot, offset):
    """A copy of the raw prediction with one camera centre displaced."""

    pose_encoding = raw["pose_enc"].clone()
    pose_encoding[0, slot, :3] += torch.tensor(offset, dtype=pose_encoding.dtype)
    return {**raw, "pose_enc": pose_encoding}


def test_drift_report_relative_pose_is_gauge_invariant(dumped_scene):
    """A global rotation and rescale of the prediction must not be an error.

    Applied to a prediction that is already wrong, so the figures being
    compared are nonzero and an implementation that merely returned constants
    could not pass.  ``baseline_scale`` moves *inversely*: scaling the predicted
    offsets by sigma scales the fit's numerator by sigma and its denominator by
    sigma squared.
    """

    scale = 2.5
    rotation = _yaw_rotation(25.0) @ _pitch_rotation(-12.0)
    raw = _moved_center(
        _ground_truth_raw_reconstruction(dumped_scene),
        slot=4,
        offset=(0.0, 0.25, 0.0),
    )

    plain = reconstruction_drift_report(raw, dumped_scene, _identity_alignment())
    gauged = reconstruction_drift_report(
        _apply_predicted_gauge(raw, rotation, scale),
        dumped_scene,
        _identity_alignment(),
    )

    assert plain["pose"]["relative_center_error_m"]["cross_camera"]["max"] > 0.1
    for figure in ("relative_rotation_deg", "relative_center_error_m"):
        for group in ("cross_camera", "static_camera"):
            for statistic in ("mean", "max"):
                assert gauged["pose"][figure][group][statistic] == pytest.approx(
                    plain["pose"][figure][group][statistic],
                    abs=1e-4,
                )
    assert gauged["pose"]["baseline_scale"] == pytest.approx(
        plain["pose"]["baseline_scale"] / scale,
        rel=1e-4,
    )


def test_drift_report_relative_center_ignores_a_depth_contaminated_alignment(
    dumped_scene,
):
    """The alignment carries the depth error; the relative figures must not.

    This is the whole reason the new numbers exist, so the alignment here is the
    real one -- fitted by ``fit_scene_sim3`` from the inflated pointmaps -- not a
    hand-built stand-in.
    """

    clean = _ground_truth_raw_reconstruction(dumped_scene)
    raw = {**clean, "depth": clean["depth"] * 1.1}
    clean_alignment, _ = fit_scene_sim3(clean, dumped_scene)
    alignment, _ = fit_scene_sim3(raw, dumped_scene)

    report = reconstruction_drift_report(raw, dumped_scene, alignment)

    # The fixture's own scale is not 1, so the readable statement is the ratio:
    # the fit absorbed the depth inflation exactly, which is what contaminates
    # every figure composed through it.
    assert float(alignment.scale.item()) == pytest.approx(
        float(clean_alignment.scale.item()) / 1.1, rel=1e-4
    )
    assert report["pose"]["camera_center_error_m"]["max"] > 0.05
    assert (
        report["pose"]["relative_center_error_m"]["cross_camera"]["max"] < 1e-4
    )
    # Fitted from camera centres, so it does not follow the depth inflation the
    # Sim(3) scale above absorbed. The two disagreeing is the contamination.
    assert report["pose"]["baseline_scale"] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("offset", (0.25, 4.0))
def test_drift_report_static_wander_does_not_move_the_baseline_scale(
    dumped_scene,
    offset,
):
    """Zero-baseline slots are measured but do not vote on the gauge factor.

    Such a slot has a zero ground-truth offset and a nonzero predicted one -- the
    wander itself -- so a fit taken over every slot would add nothing to the
    numerator and the full squared norm to the denominator, dragging the scalar
    down harder as the drift grows.  Hence the two offsets: the larger one must
    move the scalar no more than the smaller.
    """

    raw = _moved_center(
        _ground_truth_raw_reconstruction(dumped_scene),
        slot=1,  # camera 0 at time 1: the anchor's own camera, zero GT baseline
        offset=(offset, 0.0, 0.0),
    )

    pose = reconstruction_drift_report(
        raw,
        dumped_scene,
        _identity_alignment(),
    )["pose"]

    assert pose["baseline_scale"] == pytest.approx(1.0, rel=1e-5)
    assert pose["relative_center_error_m"]["static_camera"]["max"] == pytest.approx(
        offset, rel=1e-4
    )
    assert pose["relative_center_error_m"]["cross_camera"]["max"] < 1e-4


def test_drift_report_reads_a_cross_camera_center_move(dumped_scene):
    """A term invariant to everything would pass every test above but this one."""

    raw = _moved_center(
        _ground_truth_raw_reconstruction(dumped_scene),
        slot=4,  # camera 1 at time 0: genuinely separated from the anchor
        offset=(0.0, 0.25, 0.0),
    )

    pose = reconstruction_drift_report(
        raw,
        dumped_scene,
        _identity_alignment(),
    )["pose"]

    assert pose["relative_center_error_m"]["cross_camera"]["max"] > 0.2
    assert pose["camera_center_error_m"]["max"] == pytest.approx(0.25, rel=1e-4)


def test_drift_report_relative_figures_stay_finite_under_wander(dumped_scene):
    """The 2-camera fixture keeps three zero-baseline slots; none may go NaN."""

    raw = _moved_center(
        _moved_center(
            _ground_truth_raw_reconstruction(dumped_scene),
            slot=2,
            offset=(0.3, -0.1, 0.2),
        ),
        slot=5,
        offset=(-0.2, 0.4, 0.1),
    )

    pose = reconstruction_drift_report(
        raw,
        dumped_scene,
        _identity_alignment(),
    )["pose"]

    assert np.isfinite(pose["baseline_scale"])
    for figure in ("relative_rotation_deg", "relative_center_error_m"):
        for group in ("cross_camera", "static_camera"):
            for value in pose[figure][group].values():
                assert np.isfinite(value)


def test_drift_report_relative_rotation_survives_a_single_camera_window(tmp_path):
    """One camera means no baseline anywhere, and rotation must outlive that.

    A fit taken over every slot would make the numerator identically zero here,
    so ``baseline_scale`` would be 0 and every centre residual would collapse to
    ``||0 * c_pred - 0|| == 0`` -- the report would call a drifting model perfect.
    Restricting the fit empties its input instead, which reaches ``None``.
    """

    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(1,),
        times=(0, 2, 3),
        size=56,
    )
    # The centre must actually wander, or the merged-fit denominator would be
    # zero too and this would pass for the wrong reason. With it, that fit finds
    # a nonzero denominator against an identically-zero numerator and reports
    # baseline_scale 0.0 and perfect centres.
    raw = _moved_center(
        _ground_truth_raw_reconstruction(scene),
        slot=1,
        offset=(0.35, -0.2, 0.15),
    )
    raw["pose_enc"][0, 1, 3:7] = mat_to_quat(
        torch.from_numpy(_yaw_rotation(5.0))
    ).float()

    pose = reconstruction_drift_report(raw, scene, _identity_alignment())["pose"]

    assert pose["baseline_scale"] is None
    assert pose["relative_center_error_m"]["cross_camera"] is None
    assert pose["relative_center_error_m"]["static_camera"] is None
    assert pose["relative_rotation_deg"]["cross_camera"] is None
    # The scale-free half still reads the drift the centre half cannot.
    assert pose["relative_rotation_deg"]["static_camera"]["max"] == pytest.approx(
        5.0, rel=1e-3
    )


def test_drift_report_relative_pose_ignores_the_alignment_entirely(dumped_scene):
    """The relative figures must not move when the Sim(3) does.

    The alignment is the channel the depth error arrives through, so the claim
    worth testing is the strong one: swapping it for an arbitrary rotation,
    scale and translation changes neither figure at all.  The first two
    assertions keep that non-vacuous by confirming the swapped alignment does
    move the numbers that are composed through it.
    """

    raw = _moved_center(
        _ground_truth_raw_reconstruction(dumped_scene),
        slot=4,
        offset=(0.0, 0.25, 0.0),
    )
    skewed = DetachedSim3(
        scale=torch.tensor(0.37),
        rotation=torch.from_numpy(
            _yaw_rotation(31.0) @ _pitch_rotation(17.0)
        ).float(),
        translation=torch.tensor([0.8, -1.3, 2.0]),
    )

    plain = reconstruction_drift_report(
        raw, dumped_scene, _identity_alignment()
    )["pose"]
    composed = reconstruction_drift_report(raw, dumped_scene, skewed)["pose"]

    assert composed["camera_center_error_m"]["max"] != pytest.approx(
        plain["camera_center_error_m"]["max"], rel=1e-3
    )
    assert composed["rotation_error_deg"]["max"] != pytest.approx(
        plain["rotation_error_deg"]["max"], abs=1e-2
    )
    assert composed["baseline_scale"] == pytest.approx(
        plain["baseline_scale"], rel=1e-6
    )
    for figure in ("relative_rotation_deg", "relative_center_error_m"):
        for group in ("cross_camera", "static_camera"):
            for statistic in ("mean", "max"):
                assert composed[figure][group][statistic] == pytest.approx(
                    plain[figure][group][statistic], abs=1e-5
                )


def test_drift_report_relative_pose_is_anchored_at_the_query_observation(tmp_path):
    """The reference is the anchor, not slot 0.

    ``rotated_camera=1`` gives the anchor camera a real yaw, pitch and offset, so
    a transposed rotation moves the numbers.  The wander is placed on the
    anchor's *own* camera, which is what makes the choice of reference readable:
    slot 3 shares camera 1 with the anchor and so owes zero baseline, but under a
    slot-0 reference it would be a separated slot instead and the two groups
    below would swap.
    """

    _write_scene(tmp_path, rotated_camera=1)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 3),
        query_anchors=((1, 0),),
        size=56,
    )
    assert scene.query_observation_slot == 2
    clean = _ground_truth_raw_reconstruction(scene)

    exact = reconstruction_drift_report(clean, scene, _identity_alignment())["pose"]
    for figure in ("relative_rotation_deg", "relative_center_error_m"):
        for group in ("cross_camera", "static_camera"):
            assert exact[figure][group]["max"] < 1e-3
    assert exact["baseline_scale"] == pytest.approx(1.0, rel=1e-4)

    wandered = reconstruction_drift_report(
        _moved_center(clean, slot=3, offset=(0.3, 0.0, 0.0)),
        scene,
        _identity_alignment(),
    )["pose"]

    assert wandered["relative_center_error_m"]["static_camera"]["max"] == (
        pytest.approx(0.3, rel=1e-4)
    )
    assert wandered["relative_center_error_m"]["cross_camera"]["max"] < 1e-4
    assert wandered["baseline_scale"] == pytest.approx(1.0, rel=1e-4)


# ------------------------------------------------------------------------------
# the three-reference exit gate and the shuffled-index control
# ------------------------------------------------------------------------------


def test_exit_gate_requires_beating_the_baseline():
    passing = overfit_cli._exit_criteria_failure(
        baseline_loss=1.0,
        initial_loss=2.0,
        final_loss=0.9,
        final_shuffled_loss=None,
        embedding_change=0.5,
        min_improvement=0.01,
        min_index_advantage=0.0,
    )
    assert passing is None

    # Beats the inflated initial handily, but not the released baseline: the
    # improvement was recovery from a disruptive init, and the gate says so.
    recovery_only = overfit_cli._exit_criteria_failure(
        baseline_loss=1.0,
        initial_loss=2.0,
        final_loss=0.999,
        final_shuffled_loss=None,
        embedding_change=0.5,
        min_improvement=0.01,
        min_index_advantage=0.0,
    )
    assert recovery_only is not None
    assert "zero-embedding baseline" in recovery_only


def test_exit_gate_requires_an_index_advantage():
    passing = overfit_cli._exit_criteria_failure(
        baseline_loss=1.0,
        initial_loss=1.0,
        final_loss=0.5,
        final_shuffled_loss=0.6,
        embedding_change=0.5,
        min_improvement=0.01,
        min_index_advantage=0.01,
    )
    assert passing is None

    # Shuffling the indices barely hurts: the improvement is decoder
    # adaptation, and the run must fail even though both loss gates pass.
    unexploited = overfit_cli._exit_criteria_failure(
        baseline_loss=1.0,
        initial_loss=1.0,
        final_loss=0.5,
        final_shuffled_loss=0.502,
        embedding_change=0.5,
        min_improvement=0.01,
        min_index_advantage=0.01,
    )
    assert unexploited is not None
    assert "Shuffling" in unexploited

    # None skips the check: single-camera or single-time windows have no
    # synchronization to break.
    assert overfit_cli._exit_criteria_failure(
        baseline_loss=1.0,
        initial_loss=1.0,
        final_loss=0.5,
        final_shuffled_loss=None,
        embedding_change=0.5,
        min_improvement=0.01,
        min_index_advantage=0.5,
    ) is None


def _shuffle_stub_scene(cameras, times):
    observations = []
    views = []
    for camera in cameras:
        for position in range(len(times)):
            observations.append(
                SimpleNamespace(camera=camera, semantic_time_index=position)
            )
            views.append(
                {
                    "img": torch.zeros(1),
                    "time_index": torch.tensor([position]),
                    "track_query_idx": torch.tensor([0]),
                }
            )
    return SimpleNamespace(
        cameras=tuple(cameras),
        times=tuple(times),
        observations=tuple(observations),
        views=views,
    )


def test_shuffled_index_views_reverse_only_secondary_cameras():
    scene = _shuffle_stub_scene([0, 1], [0, 1, 2])

    shuffled = overfit_cli._shuffled_index_views(scene)

    for position in range(3):
        # Primary camera keeps its indices; the copies share the same tensors.
        assert torch.equal(
            shuffled[position]["time_index"], torch.tensor([position])
        )
        # Secondary camera is reversed.
        assert shuffled[3 + position]["time_index"].item() == 2 - position
        # The scene's own views must be untouched.
        assert scene.views[3 + position]["time_index"].item() == position
        assert shuffled[3 + position] is not scene.views[3 + position]


def test_shuffled_index_views_skip_windows_with_nothing_to_break():
    assert overfit_cli._shuffled_index_views(
        _shuffle_stub_scene([0], [0, 1, 2])
    ) is None
    assert overfit_cli._shuffled_index_views(
        _shuffle_stub_scene([0, 1], [0])
    ) is None


def test_evaluate_scores_the_shuffled_arm_against_the_initial_references(
    monkeypatch,
):
    """The control arm must be scored exactly like the gated number."""

    initial_alignment = _identity_alignment()
    initial_anchors = torch.full((3, 3), 7.0)

    calls = []

    def _recorded_result(value):
        return SparseTrackingLossResult(
            loss=torch.tensor(value),
            metric_error=torch.tensor(value * 2),
            sample_count=0,
        )

    def fake_loss(raw, scene, correspondences, alignment, anchors, **kwargs):
        calls.append((alignment, anchors))
        return _recorded_result(0.25 * len(calls))

    monkeypatch.setattr(overfit_cli, "sparse_tracking_loss", fake_loss)
    monkeypatch.setattr(
        overfit_cli,
        "fit_scene_sim3",
        lambda raw, scene: (_identity_alignment(), {"pair_count": 1}),
    )
    monkeypatch.setattr(
        overfit_cli,
        "gather_query_anchor_points",
        lambda raw, scene, correspondences: torch.zeros(3, 3),
    )
    monkeypatch.setattr(
        overfit_cli,
        "_tracking_only",
        lambda raw, keep_confidence=False: raw,
    )
    monkeypatch.setattr(
        overfit_cli,
        "synchronized_consistency_stats",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        overfit_cli,
        "reconstruction_drift_report",
        lambda *args, **kwargs: None,
    )

    class _StubModel:
        def eval(self):
            return self

        def __call__(self, views, **kwargs):
            return {
                "conf_track_multi": torch.full((1, 1, 2, 2, 2), 3.0),
                "track_multi": torch.zeros(1, 1, 2, 2, 2, 3),
            }

    shuffled_views = [{"time_index": torch.tensor([1])}]
    evaluation = overfit_cli._evaluate(
        _StubModel(),
        SimpleNamespace(views=[], slot_time_indices=torch.zeros(0, dtype=torch.long)),
        object(),
        "32",
        0.05,
        initial_alignment,
        initial_anchors,
        sync_metric_scale=1.0,
        shuffled_views=shuffled_views,
    )

    # refit, like-for-like, then the shuffled arm.
    assert len(calls) == 3
    assert calls[1][0] is initial_alignment and calls[1][1] is initial_anchors
    assert calls[2][0] is initial_alignment and calls[2][1] is initial_anchors
    assert evaluation["loss"] == pytest.approx(0.5)
    assert evaluation["loss_shuffled"] == pytest.approx(0.75)


def test_new_training_flags_default_to_the_archived_behaviour():
    """Sync off, temporal_tracking mode: an old command line trains the same
    parameters it always did; only the init default is deliberately new."""

    parser = overfit_cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--data_root", "root", "--scene", "0000",
            "--checkpoint_dir", "ckpt", "--output_dir", "out",
        ]
    )
    overfit_cli._validate_args(args)

    assert args.freeze_mode == "temporal_tracking"
    assert args.sync_weight == 0.0
    assert args.time_embedding_init == "orthogonal"
    assert args.time_embedding_init_scale == 0.1
    assert args.embedding_lr is None
    assert args.encoder_lr is None
    assert args.min_index_advantage == 0.01
    assert args.late_global_blocks == overfit_cli.DEFAULT_LATE_GLOBAL_BLOCKS == 4

    for flag, bad in (
        ("--time_embedding_init_scale", "0"),
        ("--sync_weight", "-1"),
        ("--min_index_advantage", "1.0"),
        ("--embedding_lr", "nan"),
        ("--encoder_lr", "0"),
        ("--late_global_blocks", "0"),
        ("--late_global_blocks", "15"),
    ):
        rejected = parser.parse_args(
            [
                "--data_root", "root", "--scene", "0000",
                "--checkpoint_dir", "ckpt", "--output_dir", "out",
                flag, bad,
            ]
        )
        with pytest.raises(ValueError, match=flag.lstrip("-").replace("-", "_")):
            overfit_cli._validate_args(rejected)


def _optimizer_args(lr=1e-5, embedding_lr=None, encoder_lr=None):
    return SimpleNamespace(lr=lr, embedding_lr=embedding_lr, encoder_lr=encoder_lr)


def _optimizer_model(freeze, late_global_blocks=None):
    from test_time_indexing import _GlobalAttnTinyArc

    return _GlobalAttnTinyArc(
        freeze=freeze,
        late_global_blocks=late_global_blocks,
    )


_ENCODER_FREEZE_MODES = (
    ("temporal_tracking_global_attention", None),
    ("temporal_tracking_late_global", 1),
)


@pytest.mark.parametrize("freeze,late_global_blocks", _ENCODER_FREEZE_MODES)
def test_build_optimizer_gives_every_mode_the_same_encoder_rate_rule(
    freeze,
    late_global_blocks,
):
    """encoder_lr must reach the middle rung exactly as it reaches the full one.

    _build_optimizer selects the encoder group by module membership and a name
    filter, never by freeze mode or block index, so this holds by construction
    -- but "by construction" is what silently stops being true under a
    refactor, and a sweep over encoder_lr is worthless if the flag misses.
    """

    model = _optimizer_model(freeze, late_global_blocks)

    _, defaulted, encoder_parameters = overfit_cli._build_optimizer(
        model,
        _optimizer_args(lr=1e-5),
    )
    assert defaulted["decoder"] == 1e-5
    assert defaulted["embedding"] == 1e-5
    assert defaulted["encoder_blocks"] == pytest.approx(1e-6)
    assert encoder_parameters

    optimizer, explicit, _ = overfit_cli._build_optimizer(
        model,
        _optimizer_args(lr=1e-5, encoder_lr=3e-6, embedding_lr=2e-5),
    )
    assert explicit["encoder_blocks"] == 3e-6
    assert explicit["embedding"] == 2e-5
    # The rate reaches the optimizer itself, not just the reported dict.
    assert sorted(group["lr"] for group in optimizer.param_groups) == [
        3e-6,
        1e-5,
        2e-5,
    ]


def test_build_optimizer_leaves_the_narrow_mode_without_an_encoder_group():
    """No unfrozen encoder block means no group and no rate to report."""

    model = _optimizer_model("temporal_tracking")
    optimizer, learning_rates, encoder_parameters = overfit_cli._build_optimizer(
        model,
        _optimizer_args(lr=1e-5, encoder_lr=3e-6),
    )
    assert encoder_parameters == []
    assert learning_rates["encoder_blocks"] is None
    assert len(optimizer.param_groups) == 2


def test_build_optimizer_encoder_group_is_the_unfrozen_blocks_only():
    """The embedding lives in its own group, never in the encoder one."""

    model = _optimizer_model("temporal_tracking_late_global", late_global_blocks=1)
    _, _, encoder_parameters = overfit_cli._build_optimizer(
        model,
        _optimizer_args(),
    )

    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    blocks = model.backbone.pretrained.blocks
    assert encoder_ids == {id(parameter) for parameter in blocks[3].parameters()}
    embedding = model.backbone.pretrained.time_index_embedding.weight
    assert id(embedding) not in encoder_ids


@pytest.mark.parametrize(
    "freeze,late_global_blocks",
    (("temporal_tracking", None), *_ENCODER_FREEZE_MODES),
)
def test_build_optimizer_groups_cover_every_trainable_parameter(
    freeze,
    late_global_blocks,
):
    model = _optimizer_model(freeze, late_global_blocks)
    optimizer, _, _ = overfit_cli._build_optimizer(model, _optimizer_args())

    grouped = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert grouped == trainable


def test_build_optimizer_rejects_a_parameter_outside_every_group():
    """The coverage guard is the only thing standing between a silent typo in
    the group construction and a run that trains fewer parameters than it
    reports."""

    model = _optimizer_model("temporal_tracking_late_global", late_global_blocks=1)
    model.head.requires_grad_(True)

    with pytest.raises(RuntimeError, match="escaped every group"):
        overfit_cli._build_optimizer(model, _optimizer_args())


def test_expected_trainable_set_derives_every_k():
    """The pinned k=4 entry and the per-block constant must span the range.

    At k=14 the derivation has to land exactly on the independently pinned
    temporal_tracking_global_attention entry: that is what proves the middle
    rung and the full preset describe one mask, not two that happen to agree
    at the default.
    """

    narrow_tensors, narrow_parameters = overfit_cli.EXPECTED_TRAINABLE_SETS[
        "temporal_tracking"
    ]
    per_block_tensors, per_block_parameters = overfit_cli.LATE_GLOBAL_PER_BLOCK

    for k in range(1, overfit_cli.MAX_LATE_GLOBAL_BLOCKS + 1):
        assert overfit_cli._expected_trainable_set(
            "temporal_tracking_late_global", k
        ) == (
            narrow_tensors + per_block_tensors * k,
            narrow_parameters + per_block_parameters * k,
        )

    assert overfit_cli._expected_trainable_set(
        "temporal_tracking_late_global",
        overfit_cli.MAX_LATE_GLOBAL_BLOCKS,
    ) == overfit_cli.EXPECTED_TRAINABLE_SETS["temporal_tracking_global_attention"]

    # The k-less modes ignore k entirely.
    for mode in ("temporal_tracking", "temporal_tracking_global_attention"):
        assert (
            overfit_cli._expected_trainable_set(mode, None)
            == overfit_cli.EXPECTED_TRAINABLE_SETS[mode]
        )

    assert (
        "temporal_tracking_late_global"
        in overfit_cli.build_arg_parser()
        ._option_string_actions["--freeze_mode"]
        .choices
    )


def test_run_summary_includes_the_baseline_and_control_fields():
    source = Path(overfit_cli.__file__).read_text()
    written = None
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "summary"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            written = {key.value for key in node.value.keys}
    assert written is not None

    assert {
        "freeze_mode",
        "time_embedding_init",
        "time_embedding_init_scale",
        "time_embedding_target_row_norm",
        "learning_rates",
        "sync_weight",
        "min_index_advantage",
        "baseline_position_loss",
        "baseline_metric_error_m",
        "baseline_track_confidence",
        "final_position_loss_shuffled",
        "final_metric_error_shuffled_m",
        "baseline_sync_consistency",
        "initial_sync_consistency",
        "final_sync_consistency",
        "temporal_injection",
        "reconstruction_shift",
        "baseline_reconstruction_drift",
        "final_reconstruction_drift",
    } <= written


# ---------------------------------------------------------------------------
# Per-frame depth sidecar
# ---------------------------------------------------------------------------


def test_depth0_only_dump_is_unchanged(tmp_path):
    """No sidecar must mean exactly today's behaviour.

    A dump taken before the sidecar existed, or without RCMV_DUMP_DEPTH=1, is
    still the common case, and it must keep producing the same correspondences
    it always has.
    """

    _write_scene(tmp_path)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    assert scene.depth is None
    assert not scene.has_time_varying_depth
    assert scene.depth_sidecar_path is None

    correspondences, report = build_anchor_correspondences(scene)
    assert correspondences.trajectory_indices.tolist() == [0, 1, 2]
    assert correspondences.query_slots.tolist() == [0, 0, 0]
    assert correspondences.query_times.tolist() == [0, 0, 0]
    assert correspondences.rows.tolist() == [25, 30, 27]
    assert correspondences.columns.tolist() == [22, 28, 34]
    assert report["eligible_query_count"] == 3
    assert report["supervised_pair_count"] == 3
    assert report["anchor_count"] == 1


def test_time_varying_depth_sidecar_is_loaded_beside_meta(tmp_path):
    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    assert scene.has_time_varying_depth
    assert scene.depth.shape == (2, 4, 1, 56, 56)
    assert scene.depth_sidecar_path == tmp_path / "0000" / "depth_full.npz"
    # depth[:, 0] is what meta.npz stores as depth0, by construction on the
    # producing side; the adapter relies on that to anchor time 0 identically
    # whether or not the sidecar is present.
    assert torch.equal(scene.depth[:, 0], scene.depth0)
    for time_index in range(4):
        assert torch.equal(
            scene.surface_depth_map(0, time_index),
            scene.depth[0, time_index, 0],
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_sidecar_dtype_is_read_not_asserted(tmp_path, dtype):
    """Kubric's depth TIFFs decide the dtype and the dump passes it through.

    Anything the producing side emits must load, and must produce the same
    correspondences, so nothing may hardcode or assert float32.
    """

    _write_scene(tmp_path, depth_sidecar=True, sidecar_dtype=dtype)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    assert scene.depth.dtype == torch.float32
    assert torch.equal(scene.depth[:, 0], scene.depth0)
    correspondences, _ = build_anchor_correspondences(scene)
    assert correspondences.trajectory_indices.tolist() == [0, 1, 2]
    assert correspondences.rows.tolist() == [25, 30, 27]
    assert correspondences.columns.tolist() == [22, 28, 34]


def test_sidecar_disagreeing_with_depth0_is_rejected(tmp_path):
    scene_path = _write_scene(tmp_path, depth_sidecar=True)
    with np.load(scene_path / "depth_full.npz") as sidecar:
        depth = np.array(sidecar["depth"])
    depth[0, 0, 0, 5, 5] += 1.0
    np.savez_compressed(scene_path / "depth_full.npz", depth=depth, seq_name="0000")

    with pytest.raises(ValueError, match="different dumps"):
        load_dumped_kubric_scene(
            tmp_path,
            "0000",
            cameras=(0, 1),
            times=(0, 1, 2, 3),
            size=56,
        )


def test_sidecar_shape_mismatch_is_rejected(tmp_path):
    scene_path = _write_scene(tmp_path, depth_sidecar=True)
    with np.load(scene_path / "depth_full.npz") as sidecar:
        depth = np.array(sidecar["depth"])
    np.savez_compressed(
        scene_path / "depth_full.npz",
        depth=depth[:, :2],
        seq_name="0000",
    )

    with pytest.raises(ValueError, match="views x .* frames"):
        load_dumped_kubric_scene(
            tmp_path,
            "0000",
            cameras=(0, 1),
            times=(0, 1, 2, 3),
            size=56,
        )


def test_view_ids_resolve_anchor_cameras(tmp_path):
    """``--cameras`` and anchors speak original camera ids, not view positions.

    Dumps are taken with an ascending, complete view list so the two coincide
    today. Resolving through the recorded ``view_ids`` is what stops that
    convention from being load-bearing.
    """

    _write_scene(tmp_path, view_ids=[4, 7], depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(4, 7),
        times=(0, 1, 2, 3),
        query_anchors=((7, 0),),
        size=56,
    )

    assert scene.view_ids.tolist() == [4, 7]
    assert scene.camera_ids == (4, 7)
    # Arrays stay indexed by view position; only the ids the user types differ.
    assert scene.cameras == (0, 1)
    query = scene.observations[scene.query_observation_slot]
    assert (query.camera, query.camera_id) == (1, 7)
    assert scene.query_observation_slot == 4

    with pytest.raises(ValueError, match="not among the dumped cameras"):
        load_dumped_kubric_scene(
            tmp_path,
            "0000",
            cameras=(0, 1),
            times=(0, 1),
            size=56,
        )


# ---------------------------------------------------------------------------
# Anchors at any camera and any time
# ---------------------------------------------------------------------------


def test_nonzero_query_time_anchor_supervises_with_the_sidecar(tmp_path):
    """A query that starts at t=2 is anchored at t=2, not discarded."""

    _write_scene(tmp_path, depth_sidecar=True, query_times=[2, 2, 2])
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 2),),
        size=56,
    )

    correspondences, report = build_anchor_correspondences(scene)

    assert correspondences.count == 3
    assert correspondences.query_times.tolist() == [2, 2, 2]
    assert report["eligible_query_count"] == 3
    expected = _project_expected_anchors(
        scene,
        camera=0,
        rotated_camera=None,
        time_index=2,
    )
    assert list(zip(correspondences.rows.tolist(), correspondences.columns.tolist())) == expected

    # The same window anchored at t=0 reaches nothing: the queries are not at
    # frame 0, which is exactly the loss the sidecar exists to recover. That is
    # reported as an empty set with a split explaining it, not raised -- an
    # anchor set buying nothing is the case most worth measuring.
    at_time_zero = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0),),
        size=56,
    )
    empty, empty_report = build_anchor_correspondences(at_time_zero)
    assert empty.count == 0
    assert empty_report["eligible_query_count"] == 0
    assert empty_report["rejected"]["query_time_mismatch"] == 3


def test_nonzero_query_time_needs_the_sidecar(tmp_path):
    _write_scene(tmp_path, query_times=[2, 2, 2])
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 2),),
        size=56,
    )

    with pytest.raises(ValueError, match="RCMV_DUMP_DEPTH=1"):
        build_anchor_correspondences(scene)


def test_nonzero_query_camera_anchor_uses_its_own_depth(tmp_path):
    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((1, 0),),
        size=56,
    )
    # If the anchor accidentally read camera 0, every candidate fails its gate.
    scene.depth[0].fill_(100.0)
    scene.depth0[0].fill_(100.0)

    correspondences, report = build_anchor_correspondences(scene)

    assert correspondences.count == 3
    assert report["per_anchor"][0]["camera"] == 1
    assert scene.query_observation_slot == 4


def test_a_query_visible_from_two_anchors_yields_one_row(tmp_path):
    """One row per trajectory, whichever anchor wins it.

    Both anchors here can see every query, so under a per-(query, anchor) rule
    this scene would produce six rows and the doubly-visible queries would carry
    twice the gradient of a singly-visible one. Anchor multiplicity is a property
    of where the cameras point, not of how much a point matters, so exactly one
    anchor supervises each query and the totals stay balanced.
    """

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )

    correspondences, report = build_anchor_correspondences(scene)

    # One row per trajectory, never one per (trajectory, anchor) pair: anchor
    # multiplicity is a property of where the cameras point, not of how much a
    # point matters, so a doubly-visible query must not outweigh a singly-visible
    # one.
    assert report["eligible_query_count"] == 3
    assert report["supervised_pair_count"] == 3
    assert correspondences.count == 3
    assert sorted(correspondences.trajectory_indices.tolist()) == [0, 1, 2]
    assert ELIGIBILITY_ASSIGNMENT_RULE == report["assignment_rule"]
    assert ELIGIBILITY_ROLLUP_RULE == report["rollup_rule"]
    # Both anchors could take every query here, so the tiebreak decides and the
    # totals still balance.
    assert [anchor["eligible"] for anchor in report["per_anchor"]] == [3, 3]
    assert sum(anchor["assigned"] for anchor in report["per_anchor"]) == 3
    assert sum(anchor["sole_anchor"] for anchor in report["per_anchor"]) == 0


def test_best_fitting_anchor_wins_over_an_earlier_worse_one(tmp_path):
    """Selection is by fit, not by declaration order.

    Anchor 0 is declared first but its depth map is nudged off the query's
    surface, so its anchor-depth error is larger than anchor 1's while still
    inside the 10 cm gate. The later, better-fitting anchor must win -- that is
    what distinguishes best-fit from first-eligible.
    """

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    # Tie-break alone would give every query to anchor 0.
    baseline, _ = build_anchor_correspondences(scene)
    assert baseline.query_slots.tolist() == [0, 0, 0]

    # 5 cm of depth error at camera 0: still passes the gate, but is a worse fit
    # than camera 1's exact render.
    scene.depth[0, 0, 0] += 0.05
    scene.depth0[0, 0] += 0.05

    correspondences, report = build_anchor_correspondences(scene)

    assert correspondences.query_slots.tolist() == [1, 1, 1]
    assert report["per_anchor"][0]["eligible"] == 3
    assert report["per_anchor"][0]["assigned"] == 0
    assert report["per_anchor"][1]["assigned"] == 3


def test_second_anchor_recovers_a_query_the_first_cannot(tmp_path):
    """The occlusion recovery, measured.

    Track 1 is invisible in camera 0 at t=0, so no camera-0 anchor can reach
    it -- its pixel there belongs to whatever occludes it. Camera 1 sees it.
    """

    _write_scene(tmp_path, depth_sidecar=True, invisible=[(0, 0, 1)])
    single = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0),),
        size=56,
    )
    both = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )

    _, single_report = build_anchor_correspondences(single)
    _, both_report = build_anchor_correspondences(both)

    assert single_report["eligible_query_count"] == 2
    assert single_report["rejected"]["not_visible_in_anchor"] == 1
    assert both_report["eligible_query_count"] == 3
    assert both_report["rejected"]["not_visible_in_anchor"] == 0
    # Camera 1 is the only anchor that can reach track 1 -- that is the recovery,
    # and sole_anchor names it without depending on declaration order.
    assert both_report["per_anchor"][1]["sole_anchor"] == 1
    assert both_report["per_anchor"][0]["sole_anchor"] == 0
    assert both_report["per_anchor"][0]["assigned"] == 2
    assert both_report["per_anchor"][1]["assigned"] == 1
    assert both_report["per_anchor"][1]["rejected"]["not_visible_in_anchor"] == 0


def test_eligibility_split_is_exclusive_and_exhaustive(tmp_path):
    """Every query is accounted for exactly once, whatever the anchor set."""

    _write_scene(
        tmp_path,
        depth_sidecar=True,
        query_times=[0, 2, 2],
        invisible=[(0, 0, 0)],
    )
    for anchors in (((0, 0),), ((0, 0), (1, 0)), ((0, 0), (1, 0), (0, 2))):
        scene = load_dumped_kubric_scene(
            tmp_path,
            "0000",
            cameras=(0, 1),
            times=(0, 1, 2, 3),
            query_anchors=anchors,
            size=56,
        )
        _, report = build_anchor_correspondences(scene)
        assert set(report["rejected"]) == set(ELIGIBILITY_REJECTION_STAGES)
        accounted = report["eligible_query_count"] + sum(report["rejected"].values())
        assert accounted == report["total_query_count"] == 3


def test_anchor_depth_gate_still_rejects_at_a_nonzero_time(tmp_path):
    """The 10 cm gate is unchanged, and applies to per-frame depth too."""

    _write_scene(tmp_path, depth_sidecar=True, query_times=[2, 2, 2])
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 2),),
        size=56,
    )
    baseline, _ = build_anchor_correspondences(scene)
    assert baseline.trajectory_indices.tolist() == [0, 1, 2]

    observation = scene.observations[scene.query_observation_slot]
    rows, columns = observation.image_transform.output_to_original_indices()
    # Move the surface a metre away under track 0's anchor pixel, at t=2 only.
    scene.depth[0, 2, 0, int(rows[baseline.rows[0]]), int(columns[baseline.columns[0]])] = 8.0

    filtered, report = build_anchor_correspondences(scene)

    assert filtered.trajectory_indices.tolist() == [1, 2]
    assert report["rejected"]["anchor_depth_gate"] == 1


def test_out_of_bounds_projection_is_rejected(tmp_path):
    """A query projecting outside the crop is counted, not silently kept."""

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )
    # Push track 0 far off the image plane; the trajectory and query point must
    # move together or the adapter's own consistency check fires first.
    scene.trajectories_world[:, 0, 0] = 500.0
    scene.query_points[0, 1] = 500.0

    correspondences, report = build_anchor_correspondences(scene)

    assert correspondences.trajectory_indices.tolist() == [1, 2]
    assert report["rejected"]["projection"] == 1
    assert report["eligible_query_count"] == 2


def test_select_query_slot_slices_and_rebases(tmp_path):
    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)

    for anchor_index in (0, 1):
        anchor = correspondences.select_query_slot(anchor_index)
        keep = correspondences.query_slots == anchor_index
        assert anchor.count == int(keep.sum())
        # Rebased to zero, because one anchor's head pass produces Q=1.
        assert anchor.query_slots.tolist() == [0] * anchor.count
        assert anchor.trajectory_indices.tolist() == (
            correspondences.trajectory_indices[keep].tolist()
        )
        assert anchor.rows.tolist() == correspondences.rows[keep].tolist()

    assert correspondences.select_query_slot(9).count == 0
    with pytest.raises(ValueError, match="non-negative"):
        correspondences.select_query_slot(-1)


def test_anchor_sample_counts_come_from_the_loss_masking(tmp_path):
    """The per-anchor weights must be the loss's own mask, not a re-derivation."""

    # Track 1 is reachable only from camera 1, so each anchor owns rows and the
    # counts cannot both come from the same anchor.
    _write_scene(
        tmp_path,
        depth_sidecar=True,
        invisible=[(1, 2, 0), (0, 0, 1)],
    )
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)

    counts = overfit_cli._anchor_sample_counts(scene, correspondences, 2)

    for anchor_index, count in enumerate(counts):
        anchor = correspondences.select_query_slot(anchor_index)
        _, _, _, mask = sparse_targets(scene, anchor)
        assert count == int(mask.sum())
    # Anchor 0 owns tracks 0 and 2 (one of track 0's 8 observations occluded),
    # anchor 1 owns track 1 (one occluded).
    assert counts == [15, 7]
    assert sum(counts) > 0


# ---------------------------------------------------------------------------
# The encoder/head graph cut
# ---------------------------------------------------------------------------


class _CutToy(nn.Module):
    """A backbone-shaped stand-in: shared trunk, per-query head.

    ``feats`` is a list of tuples of tensors, matching the real backbone's tap
    structure, so ``_cut_features`` is exercised on the shape it must actually
    walk rather than on a flat tensor.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 3)

    def encode(self, x):
        first = self.encoder(x)
        second = self.encoder(x * 0.5)
        return [(first, second, first + second), (first * 2.0, second - 1.0, first)]

    def anchor_loss(self, feats, anchor_index):
        tap = feats[anchor_index % len(feats)][anchor_index % 3]
        return self.head(tap).pow(2).mean()


def test_graph_cut_accumulation_equals_one_combined_backward():
    """Per-anchor backward across the cut must equal the single big backward.

    This is the claim the whole multi-anchor design rests on: the cut exists so
    that only one track-head graph is alive at a time and the encoder is
    differentiated once, and it is only admissible because summing gradients at
    a cut is exactly the chain rule. The assertion deliberately covers
    parameters *upstream* of the cut, which is where a mis-wired
    ``torch.autograd.backward(feats, grad_tensors=...)`` would show up.
    """

    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    weights = [0.5, 0.3, 0.2]

    combined = _CutToy()
    feats = combined.encode(inputs)
    total = sum(
        weight * combined.anchor_loss(feats, anchor_index)
        for anchor_index, weight in enumerate(weights)
    )
    total.backward()
    expected = {
        name: parameter.grad.clone()
        for name, parameter in combined.named_parameters()
    }

    cut_model = _CutToy()
    cut_model.load_state_dict(combined.state_dict())
    feats = cut_model.encode(inputs)
    cut_feats, pairs = overfit_cli._cut_features(feats)
    assert pairs, "the cut must find differentiable taps to detach"
    cut_total = 0.0
    for anchor_index, weight in enumerate(weights):
        loss = weight * cut_model.anchor_loss(cut_feats, anchor_index)
        # Backward per anchor: this anchor's head graph is freed here, before
        # the next anchor allocates its own.
        loss.backward()
        cut_total += float(loss.detach())
    assert cut_model.head.weight.grad is not None
    assert cut_model.encoder.weight.grad is None, (
        "nothing may reach the encoder until the cut is backwarded"
    )
    overfit_cli._backward_through_cut(pairs)

    assert cut_total == pytest.approx(float(total.detach()), rel=1e-6)
    for name, parameter in cut_model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], msg=name)


def test_graph_cut_leaves_untouched_taps_at_zero():
    """A tap no anchor read still needs a gradient of the right shape."""

    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    model = _CutToy()
    feats = model.encode(inputs)
    cut_feats, pairs = overfit_cli._cut_features(feats)

    # Read exactly one tap, so every other leaf keeps grad None.
    model.anchor_loss(cut_feats, 0).backward()
    assert any(leaf.grad is None for _, leaf in pairs)

    overfit_cli._backward_through_cut(pairs)

    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()


def test_graph_cut_passes_non_differentiable_values_through():
    feats = [(torch.ones(2), None, "tap"), 7]

    cut, pairs = overfit_cli._cut_features(feats)

    assert pairs == []
    assert cut[0][1] is None and cut[0][2] == "tap" and cut[1] == 7
    # No pairs means nothing to push back; this must be a no-op, not a crash.
    overfit_cli._backward_through_cut(pairs)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_query_anchor_parsing_and_defaults():
    parser = overfit_cli.build_arg_parser()
    base = [
        "--data_root", "data", "--scene", "1",
        "--cameras", "0", "1",
        "--times", "0", "2", "6",
        "--parse_only",
    ]

    default = parser.parse_args(base)
    overfit_cli._validate_args(default)
    assert default.query_anchor is None
    assert overfit_cli._resolve_query_anchors(default) == ((0, 0),)

    several = parser.parse_args(base + ["--query_anchor", "0:0", "1:0", "0:6"])
    overfit_cli._validate_args(several)
    assert overfit_cli._resolve_query_anchors(several) == ((0, 0), (1, 0), (0, 6))

    for flags, message in (
        (["--query_anchor", "3:0"], "camera 3 is not in --cameras"),
        (["--query_anchor", "0:1"], "time 1 is not in --times"),
        (["--query_anchor", "0:0", "0:0"], "listed more than once"),
        (["--query_anchor", "0"], "CAMERA:TIME"),
        (["--query_anchor", "0:0:0"], "CAMERA:TIME"),
        (["--query_anchor", "a:0"], "integers"),
        (["--query_anchor=-1:0"], "non-negative"),
    ):
        args = parser.parse_args(base + flags)
        with pytest.raises(ValueError, match=message):
            overfit_cli._validate_args(args)


def test_eligibility_only_main_needs_neither_cuda_nor_checkpoint(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The recovery must be measurable without a GPU allocation."""

    _write_scene(tmp_path, depth_sidecar=True, invisible=[(0, 0, 1)])
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overfit_temporal_tracking.py",
            "--data_root", str(tmp_path),
            "--scene", "0000",
            "--cameras", "0", "1",
            "--times", "0", "1", "2", "3",
            "--query_anchor", "0:0", "1:0",
            "--output_dir", str(output_dir),
            "--eligibility_only",
        ],
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("eligibility-only mode must not inspect CUDA"),
    )
    monkeypatch.setattr(
        Arc,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail(
            "eligibility-only mode must not load an Arc checkpoint"
        ),
    )

    overfit_cli.main()

    printed = capsys.readouterr().out
    assert "eligible_queries=3/3" in printed
    # N is stated wherever a count or a share is, so a 512-query benchmark
    # split can never be read against the training dump's 2048.
    assert "total_query_count=3" in printed
    assert "PASS eligibility report" in printed

    written = json.loads((output_dir / "eligibility.json").read_text())
    assert written["query_anchors"] == [[0, 0], [1, 0]]
    assert written["time_varying_depth"]["present"] is True
    report = written["eligibility"]
    assert report["total_query_count"] == 3
    assert report["eligible_query_count"] == 3
    # One row per query, whatever the overlap, so pairs == queries.
    assert report["supervised_pair_count"] == 3
    # Camera 1 is the only anchor that can reach the query occluded in camera 0.
    # Which anchor wins the other two is decided by sub-pixel fit and so depends
    # on the crop; only the totals and the sole-anchor count are invariant.
    assert report["per_anchor"][1]["sole_anchor"] == 1
    assert report["per_anchor"][0]["sole_anchor"] == 0
    assert sum(anchor["assigned"] for anchor in report["per_anchor"]) == 3
    assert report["per_anchor"][1]["assigned"] >= 1
    assert report["assignment_rule"] == ELIGIBILITY_ASSIGNMENT_RULE
    assert report["rollup_rule"] == ELIGIBILITY_ROLLUP_RULE


def test_run_summary_includes_the_anchor_and_eligibility_fields():
    source = Path(overfit_cli.__file__).read_text()
    written = None
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "summary"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            written = {key.value for key in node.value.keys}
    assert written is not None, "could not find the run_summary dict literal"

    assert {
        "query_anchors",
        "query_anchor_slots",
        "anchor_count",
        "active_anchor_count",
        "anchor_sample_counts",
        "anchor_weights",
        "eligibility",
        "view_ids",
        "time_varying_depth",
    } <= written
    # The pre-existing keys keep their meaning; the rule is still add-only.
    assert _BASELINE_RUN_SUMMARY_FIELDS <= written


def test_per_anchor_weighted_supervision_equals_one_combined_loss(tmp_path):
    """The training step's arithmetic, pinned against the loss it stands in for.

    A multi-anchor step never forms the stacked Q=A loss: it scores one anchor
    at a time and backwards each, weighted by that anchor's share of the
    supervised samples. This asserts the two are the same number and the same
    gradient -- which is what makes the memory saving free rather than a change
    of objective.
    """

    # (0,0,1) makes track 1 reachable only from camera 1, so both anchors own
    # rows; (1,2,0) puts an occluded target in the mix so the masks differ.
    _write_scene(
        tmp_path,
        depth_sidecar=True,
        invisible=[(1, 2, 0), (0, 0, 1)],
    )
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    anchor_slots = scene.anchor_observation_slots
    counts = overfit_cli._anchor_sample_counts(scene, correspondences, len(anchor_slots))
    assert all(count > 0 for count in counts), "both anchors must own rows"
    weights = [count / sum(counts) for count in counts]
    anchors = _anchors_for(scene, correspondences)
    alignment = _identity_alignment()

    generator = torch.Generator().manual_seed(7)
    field = torch.randn(
        1,
        len(anchor_slots),
        scene.num_observations,
        56,
        56,
        3,
        generator=generator,
    ) * 0.05

    combined_field = field.clone().requires_grad_(True)
    combined = sparse_tracking_loss(
        {
            "track_multi": combined_field,
            "track_query_idx": scene.track_query_observation_slots,
        },
        scene,
        correspondences,
        alignment,
        anchors,
    )
    combined.loss.backward()

    split_field = field.clone().requires_grad_(True)
    accumulated = None
    for anchor_index, (slot, weight) in enumerate(zip(anchor_slots, weights)):
        rows = torch.nonzero(correspondences.query_slots == anchor_index).flatten()
        result = sparse_tracking_loss(
            {
                "track_multi": split_field[:, anchor_index : anchor_index + 1],
                "track_query_idx": torch.tensor([slot]),
            },
            scene,
            correspondences.select_query_slot(anchor_index),
            alignment,
            anchors[rows],
        )
        overfit_cli._weighted_anchor_total(
            result,
            position_weight=weight,
            confidence_weight=0.0,
            sync_weight=0.0,
        ).backward()
        accumulated = overfit_cli._accumulate(accumulated, result.loss, weight)

    assert accumulated == pytest.approx(float(combined.loss.detach()), rel=1e-6)
    torch.testing.assert_close(split_field.grad, combined_field.grad, rtol=1e-5, atol=1e-7)
    assert combined.sample_count == sum(counts)


def test_per_anchor_confidence_weighting_equals_one_combined_loss(tmp_path):
    """The confidence term needs its own shares, not the position term's.

    It deliberately drops the visibility mask -- occluded samples are where the
    error is large, so they are where a low confidence is learned -- and so it
    reduces over a strictly larger set. Weighting it by the position share would
    make the multi-anchor objective quietly differ from the stacked-Q one.
    """

    _write_scene(
        tmp_path,
        depth_sidecar=True,
        invisible=[(1, 2, 0), (0, 1, 2), (0, 0, 1)],
    )
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    anchor_slots = scene.anchor_observation_slots
    anchor_count = len(anchor_slots)
    position_counts = overfit_cli._anchor_sample_counts(
        scene, correspondences, anchor_count
    )
    confidence_counts = overfit_cli._anchor_confidence_counts(
        scene, correspondences, anchor_count
    )
    # The two masks really are different sets, or this test proves nothing.
    assert confidence_counts != position_counts
    assert all(count > 0 for count in position_counts), "both anchors must own rows"
    position_weights = [c / sum(position_counts) for c in position_counts]
    confidence_weights = [c / sum(confidence_counts) for c in confidence_counts]
    anchors = _anchors_for(scene, correspondences)
    alignment = _identity_alignment()
    alpha = 2.0

    generator = torch.Generator().manual_seed(11)
    field = torch.randn(
        1, anchor_count, scene.num_observations, 56, 56, 3, generator=generator
    ) * 0.05
    confidence = 1.0 + torch.rand(
        1, anchor_count, scene.num_observations, 56, 56, generator=generator
    )

    def score(track, conf, corr, query_idx, anchor_points):
        return sparse_tracking_loss(
            {
                "track_multi": track,
                "conf_track_multi": conf,
                "track_query_idx": query_idx,
            },
            scene,
            corr,
            alignment,
            anchor_points,
            confidence_weight=1.0,
            confidence_alpha=alpha,
        )

    combined_conf = confidence.clone().requires_grad_(True)
    combined = score(
        field,
        combined_conf,
        correspondences,
        scene.track_query_observation_slots,
        anchors,
    )
    combined.confidence_loss.backward()

    split_conf = confidence.clone().requires_grad_(True)
    accumulated = None
    for anchor_index, slot in enumerate(anchor_slots):
        rows = torch.nonzero(correspondences.query_slots == anchor_index).flatten()
        result = score(
            field[:, anchor_index : anchor_index + 1],
            split_conf[:, anchor_index : anchor_index + 1],
            correspondences.select_query_slot(anchor_index),
            torch.tensor([slot]),
            anchors[rows],
        )
        overfit_cli._weighted_anchor_total(
            result,
            position_weight=position_weights[anchor_index],
            confidence_weight=confidence_weights[anchor_index],
            sync_weight=0.0,
        ).backward()
        accumulated = overfit_cli._accumulate(
            accumulated,
            result.confidence_loss,
            confidence_weights[anchor_index],
        )

    assert accumulated == pytest.approx(
        float(combined.confidence_loss.detach()), rel=1e-6
    )
    torch.testing.assert_close(
        split_conf.grad, combined_conf.grad, rtol=1e-5, atol=1e-8
    )


def test_an_anchor_set_that_reaches_nothing_is_reported_not_raised(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The case most worth measuring must not be the one that crashes.

    An anchor set recovering nothing is a finding about the anchor set. The
    report has to survive it; only training refuses to start, and it says why.
    """

    _write_scene(tmp_path, depth_sidecar=True, query_times=[2, 2, 2])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overfit_temporal_tracking.py",
            "--data_root", str(tmp_path),
            "--scene", "0000",
            "--cameras", "0", "1",
            "--times", "0", "1", "2", "3",
            "--query_anchor", "0:0",
            "--eligibility_only",
        ],
    )

    overfit_cli.main()

    printed = capsys.readouterr().out
    assert "eligible_queries=0/3" in printed
    assert "rejected.query_time_mismatch=3/3" in printed
    assert "accounted=3/3" in printed
    assert "PASS eligibility report" in printed


def test_anchor_rows_pairs_the_gather_with_the_rebased_correspondences(tmp_path):
    """``query_slots`` mean the anchor list to the gather, Q to the loss.

    ``select_query_slot`` rebases to 0 for the loss, so a rebased set handed to
    ``gather_query_anchor_points`` would read anchor 0's pointmaps for anchor
    k's pixels and raise nothing. ``anchor_rows`` is the supported pairing, and
    this pins that it selects the same rows in the same order.
    """

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    height = width = 56
    pointmaps = torch.arange(
        scene.num_observations * height * width * 3,
        dtype=torch.float32,
    ).reshape(1, scene.num_observations, height, width, 3)

    for anchor_index, slot in enumerate(scene.anchor_observation_slots):
        rows = correspondences.anchor_rows(anchor_index)
        rebased = correspondences.select_query_slot(anchor_index)
        assert rows.dtype == torch.bool
        assert int(rows.sum()) == rebased.count
        assert correspondences.rows[rows].tolist() == rebased.rows.tolist()
        assert correspondences.columns[rows].tolist() == rebased.columns.tolist()
        # The gather reads slot k's pointmap for anchor k, which is exactly what
        # the rebased set can no longer express on its own.
        expected = pointmaps[0, slot, rebased.rows, rebased.columns]
        assert expected.shape == (rebased.count, 3)

    with pytest.raises(ValueError, match="non-negative"):
        correspondences.anchor_rows(-1)


def test_per_anchor_sync_weighting_equals_one_combined_loss(tmp_path):
    """The sync term decomposes over anchors at 1/A, and that was unpinned.

    ``synchronized_consistency_loss`` reduces with ``reduction="mean"`` spanning
    the Q axis, and every anchor contributes the same element count, so the
    stacked loss is the plain mean of the per-anchor ones. The step loop relies
    on that; the other two equivalence tests both run at sync_weight=0.
    """

    _write_scene(
        tmp_path,
        depth_sidecar=True,
        invisible=[(1, 2, 0), (0, 0, 1)],
    )
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    anchor_slots = scene.anchor_observation_slots
    anchor_count = len(anchor_slots)
    anchors = _anchors_for(scene, correspondences)
    alignment = _identity_alignment()

    generator = torch.Generator().manual_seed(13)
    field = torch.randn(
        1, anchor_count, scene.num_observations, 56, 56, 3, generator=generator
    ) * 0.05

    def score(track, corr, query_idx, anchor_points):
        return sparse_tracking_loss(
            {"track_multi": track, "track_query_idx": query_idx},
            scene,
            corr,
            alignment,
            anchor_points,
            sync_weight=1.0,
        )

    combined_field = field.clone().requires_grad_(True)
    combined = score(
        combined_field,
        correspondences,
        scene.track_query_observation_slots,
        anchors,
    )
    assert combined.sync_loss is not None
    combined.sync_loss.backward()

    split_field = field.clone().requires_grad_(True)
    accumulated = None
    for anchor_index, slot in enumerate(anchor_slots):
        rows = correspondences.anchor_rows(anchor_index)
        result = score(
            split_field[:, anchor_index : anchor_index + 1],
            correspondences.select_query_slot(anchor_index),
            torch.tensor([slot]),
            anchors[rows],
        )
        overfit_cli._weighted_anchor_total(
            result,
            position_weight=0.0,
            confidence_weight=0.0,
            sync_weight=1.0 / anchor_count,
        ).backward()
        accumulated = overfit_cli._accumulate(
            accumulated,
            result.sync_loss,
            1.0 / anchor_count,
        )

    assert accumulated == pytest.approx(float(combined.sync_loss.detach()), rel=1e-6)
    torch.testing.assert_close(
        split_field.grad, combined_field.grad, rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("confidence_weight", [0.0, 0.75])
@pytest.mark.parametrize("sync_weight", [0.0, 0.5])
def test_single_anchor_total_is_bit_identical_to_the_unsplit_loss(
    tmp_path,
    confidence_weight,
    sync_weight,
):
    """A single-anchor step must be the pre-change path, exactly.

    Every archived run is single-anchor. At one anchor the position and
    confidence shares are both 1.0 and the sync share is 1/1, so
    ``_weighted_anchor_total`` must reproduce ``sparse_tracking_loss``'s own
    ``total_loss`` -- and bit-for-bit, not merely close: ``compose_tracking_loss``
    sums in insertion order and float addition is not associative, so the two
    must also agree on the order of their terms.
    """

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    assert len(scene.anchor_observation_slots) == 1

    conv, raw = _shared_conv_predictions(scene)
    result = sparse_tracking_loss(
        raw,
        scene,
        correspondences,
        _identity_alignment(),
        _anchors_for(scene, correspondences),
        confidence_weight=confidence_weight,
        confidence_alpha=3.0,
        sync_weight=sync_weight,
    )

    combined = overfit_cli._weighted_anchor_total(
        result,
        position_weight=1.0,
        confidence_weight=confidence_weight,
        sync_weight=sync_weight,
    )

    assert torch.equal(combined, result.total_loss)


def test_single_anchor_step_bypasses_the_cut_without_changing_gradients():
    """The bypass must be a memory saving only, never a behaviour change.

    With one anchor the cut buys nothing -- there is a single backward -- but it
    does hold an accumulated ``.grad`` on every tap for the whole step. Skipping
    it must leave the same parameters with the same gradients.
    """

    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    cut_model = _CutToy()
    feats = cut_model.encode(inputs)
    cut_feats, pairs = overfit_cli._cut_features(feats)
    assert pairs
    cut_model.anchor_loss(cut_feats, 0).backward()
    overfit_cli._backward_through_cut(pairs)
    through_cut = {
        name: parameter.grad.clone()
        for name, parameter in cut_model.named_parameters()
    }

    direct_model = _CutToy()
    direct_model.load_state_dict(cut_model.state_dict())
    feats = direct_model.encode(inputs)
    # The bypass: feats pass straight through, and there is nothing to push back.
    bypass_feats, bypass_pairs = feats, []
    assert bypass_pairs == []
    direct_model.anchor_loss(bypass_feats, 0).backward()
    overfit_cli._backward_through_cut(bypass_pairs)

    assert {
        name for name, p in direct_model.named_parameters() if p.grad is not None
    } == {name for name, p in cut_model.named_parameters() if p.grad is not None}
    for name, parameter in direct_model.named_parameters():
        torch.testing.assert_close(parameter.grad, through_cut[name], msg=name)


def test_weighted_anchor_total_sums_in_the_loss_s_own_term_order(
    tmp_path,
    monkeypatch,
):
    """Term order is load-bearing, because float addition is not associative.

    ``compose_tracking_loss`` sums in dict insertion order, so a single-anchor
    ``_weighted_anchor_total`` reproduces ``sparse_tracking_loss``'s own
    ``total_loss`` bit for bit only if both insert their terms in the same order.
    Asserted on the order itself rather than on a numeric difference: whether two
    orders actually disagree depends on the values, and on real losses they
    usually happen to coincide -- which would make a value-based test pass while
    the hazard stayed.
    """

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    _, raw = _shared_conv_predictions(scene)

    seen = []

    def recording_compose(terms, weights):
        seen.append(list(terms))
        return compose_tracking_loss(terms, weights)

    monkeypatch.setattr(sparse_module, "compose_tracking_loss", recording_compose)
    monkeypatch.setattr(overfit_cli, "compose_tracking_loss", recording_compose)

    result = sparse_tracking_loss(
        raw,
        scene,
        correspondences,
        _identity_alignment(),
        _anchors_for(scene, correspondences),
        confidence_weight=0.75,
        confidence_alpha=3.0,
        sync_weight=0.5,
    )
    overfit_cli._weighted_anchor_total(
        result,
        position_weight=1.0,
        confidence_weight=0.75,
        sync_weight=0.5,
    )

    loss_order, anchor_order = seen
    assert loss_order == ["position", "sync", "confidence"]
    assert anchor_order == loss_order


def test_negative_stage_would_mislabel_which_is_why_the_rollup_guards_it():
    """Why the roll-up raises on a negative stage rather than indexing with it.

    ``furthest_stage`` starts at -1, and -1 is a valid Python index landing on
    the LAST stage, so an unaccounted query would inflate that bucket with a
    plausible number instead of failing. The guard is unreachable today and
    cannot be driven through the public function -- ``_anchor_candidate`` returns
    either an eligible candidate or a failure stage, never neither, so every
    query is always one or the other. This pins the hazard the guard exists for:
    if the stage list is ever reordered, -1 still silently names a real bucket.
    """

    assert ELIGIBILITY_REJECTION_STAGES[-1] in ELIGIBILITY_REJECTION_STAGES
    assert ELIGIBILITY_REJECTION_STAGES[-1] == "pixel_dedup"


def test_report_records_the_label_quality_each_anchor_won(tmp_path):
    """Best-fit's justification, made visible without a training run.

    The rule exists because the surviving label is the least noisy one. Counts
    alone cannot show that, so each anchor reports the depth error of the labels
    it won and how much it beat the runner-up where there was one.
    """

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    # Push camera 0 off its own surface by 5 cm: still inside the 10 cm gate, so
    # it stays eligible, but camera 1 now fits every query better and should win
    # them all by a measurable margin.
    scene.depth[0, 0, 0] += 0.05
    scene.depth0[0, 0] += 0.05

    _, report = build_anchor_correspondences(scene)

    loser, winner = report["per_anchor"]
    assert winner["assigned"] == 3
    assert loser["assigned"] == 0
    # An anchor that won nothing has no labels to describe, but its contested
    # count is still a number rather than missing data.
    assert loser["assigned_depth_error_m"] is None
    assert loser["contested_assigned"] == 0
    assert loser["contested_depth_error_margin_m"] is None

    errors = winner["assigned_depth_error_m"]
    assert errors is not None
    assert errors["median"] >= 0.0
    assert errors["p95"] >= errors["median"]
    # Every win was contested here, and the margin is the runner-up's depth error
    # minus the winner's -- in metres, so it should land near the 5 cm offset.
    assert winner["contested_assigned"] == 3
    assert winner["contested_depth_error_margin_m"] == pytest.approx(0.05, abs=5e-3)
    assert winner["sole_anchor"] == 0


def test_uncontested_wins_report_a_zero_contested_count(tmp_path):
    """No runner-up anywhere must read as 'uncontested', not as missing data."""

    _write_scene(tmp_path, depth_sidecar=True)
    scene = load_dumped_kubric_scene(
        tmp_path,
        "0000",
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    _, report = build_anchor_correspondences(scene)

    only = report["per_anchor"][0]
    assert only["assigned"] == 3
    assert only["sole_anchor"] == 3
    assert only["contested_assigned"] == 0
    assert only["contested_depth_error_margin_m"] is None
    assert only["assigned_depth_error_m"]["median"] == pytest.approx(0.0, abs=1e-6)
