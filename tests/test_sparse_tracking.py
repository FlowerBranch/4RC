from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import arc.models.arc.utils.transform as transform_module
import arc.training.runtime as runtime_module
import arc.training.sparse_tracking as sparse_module
from arc.models.arc.arc import Arc
from arc.training import (
    DetachedSim3,
    SparseCorrespondences,
    adjacent_pair_indices,
    aligned_query_anchors,
    build_anchor_correspondences,
    build_scene,
    capture_rng_state,
    compose_predicted_metric,
    fit_scene_sim3,
    gather_query_anchor_points,
    load_temporal_tracking_checkpoint,
    query_anchor_errors,
    reconstruction_drift_report,
    sparse_tracking_loss,
)
from arc.models.arc.heads.head_act import activate_head
from arc.models.arc.utils.transform import mat_to_quat, quat_to_mat
from arc.training import (
    ELIGIBILITY_ASSIGNMENT_RULE,
    compose_tracking_loss,
    ELIGIBILITY_REJECTION_STAGES,
    ELIGIBILITY_ROLLUP_RULE,
    camera_major_layout,
    sparse_targets,
    sparse_targets_per_time,
)
from arc.training.runtime import anchor_sample_counts, anchor_velocity_counts
from arc.training.dumped_kubric import compute_image_transform
from scene_fixtures import (
    _PLANE_Z,
    _pitch_rotation,
    _world_to_camera,
    _yaw_rotation,
    fixture_scene,
    scene_arrays,
)
from test_time_indexing import _LinearMotionDecoder, _trainer_patch


@pytest.fixture
def dumped_scene(monkeypatch):
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
    return fixture_scene(cameras=(0, 1), times=(0, 1, 2, 3), size=56)


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
    """Cover the geometry a real Kubric frame takes.

    Every adapter test uses 56x56 at size=56, where crop is (0,0) and scale is 1,
    so output_to_original_indices degenerates to the identity and a crop-offset
    sign flip or a crop_top/crop_left swap is invisible. A real Kubric frame is
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
    """A dropped `exif_transpose` is invisible on the fixture's own frames.

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


def test_adapter_rejects_a_loader_that_reorders_its_output(monkeypatch):
    """Slot s must hold the pixels of paths[s].

    Re-deriving camera and time from the slot arithmetic is an identity over
    the whole input space and cannot see a permuted image list; the loader-index
    check is what does.
    """

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
        fixture_scene(
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


def test_single_camera_window_runs_through_arc_and_sparse_loss():
    scene = fixture_scene(
        cameras=(1,),
        times=(0, 2, 3),
        size=56,
    )

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


def test_nonfirst_query_camera_owns_alignment_correspondence():
    scene = fixture_scene(
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
    scene.depth[0].fill_(100.0)
    scene.depth0[0].fill_(100.0)
    correspondences, _ = build_anchor_correspondences(scene)

    assert correspondences.count > 0
    assert len(set(zip(
        correspondences.rows.tolist(),
        correspondences.columns.tolist(),
    ))) == correspondences.count


def test_selected_camera_order_is_preserved_camera_major():
    scene = fixture_scene(
        cameras=(1, 0),
        times=(0, 3),
        size=56,
    )

    assert [
        (observation.camera, observation.original_time)
        for observation in scene.observations
    ] == [(1, 0), (1, 3), (0, 0), (0, 3)]
    assert scene.time_indices == (0, 1, 0, 1)
    assert scene.query_observation_slot == 0


def test_adapter_supports_more_than_two_selected_cameras():
    scene = fixture_scene(
        time_count=3,
        view_count=4,
        cameras=(3, 1, 0),
        times=(0, 2),
        size=56,
    )

    assert scene.num_observations == 6
    assert [
        (observation.camera, observation.original_time)
        for observation in scene.observations
    ] == [(3, 0), (3, 2), (1, 0), (1, 2), (0, 0), (0, 2)]
    assert scene.time_indices == (0, 1, 0, 1, 0, 1)
    assert scene.slot_cameras.tolist() == [3, 3, 1, 1, 0, 0]


def test_adapter_uses_the_real_image_loader():
    scene = fixture_scene(
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )

    assert scene.num_observations == 8
    assert all(view["img"].shape == (1, 3, 56, 56) for view in scene.views)
    assert scene.time_indices == (0, 1, 2, 3, 0, 1, 2, 3)


def _rotated_two_camera_scene(monkeypatch, *, query_camera=1):
    """A scene whose camera 1 has a real yaw/pitch and a matching depth render."""

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
    return fixture_scene(
        rotated_camera=1,
        cameras=(0, 1),
        times=(0, 3),
        query_anchors=((query_camera, 0),),
        size=56,
    )


def test_metric_pointmap_lifts_depth0_onto_the_known_world_plane(monkeypatch):
    """Check the world lift against ground truth the fixture knows independently.

    The old Sim(3) test built its source by inverting a known transform applied to
    this same function's output, so the world-lift convention cancelled out and a
    transposed lift or an axis swap passed. Here depth0 renders the plane
    z = _PLANE_Z, so every lifted point must land on that plane -- an invariant
    that R vs R^T and any axis permutation break immediately.
    """

    scene = _rotated_two_camera_scene(monkeypatch)
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


def test_fit_scene_sim3_reads_the_query_observation_not_slot_zero(monkeypatch):
    """Only the query observation's pointmap may drive the alignment."""

    scene = _rotated_two_camera_scene(monkeypatch)
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


def test_fit_scene_sim3_rejects_collinear_predictions(monkeypatch):
    """The collinearity guard exists but no test reached it."""

    scene = _rotated_two_camera_scene(monkeypatch)
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


def test_rotated_query_camera_projects_to_independently_derived_pixels():
    """A real rotation makes the world-to-camera convention observable.

    With every camera at identity rotation and a pure-x baseline, inverting the
    extrinsics or transposing R leaves the projected pixels unchanged (or shifted
    uniformly), and the constant-depth plane means the 10 cm depth gate never
    fires. Both mistakes move these pixels.
    """

    scene = fixture_scene(
        rotated_camera=1,
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


def test_sparse_loss_is_zero_for_a_nonzero_query_slot():
    """Exercise the query-slot -> observation indirection off its identity.

    ``build_anchor_correspondences`` emits query_slot 0 (an index into the track
    query list) while the anchors live at ``scene.query_observation_slot`` (an
    index into the observation axis). Every other numeric test uses a scene where
    those are both 0, so dropping the indirection is numerically invisible.
    """

    scene = fixture_scene(
        rotated_camera=1,
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


def test_query_anchor_gather_follows_the_observation_slot():
    """Anchors must be read from the query observation, not from slot 0."""

    scene = fixture_scene(
        rotated_camera=1,
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
        anchors, anchor_frame = gather_query_anchor_points(
            {"track_query_idx": scene.track_query_observation_slots},
            scene,
            correspondences,
        )
    finally:
        module._predicted_pointmaps = original

    expected_value = float(scene.query_observation_slot + 1)
    assert torch.allclose(anchors, torch.full_like(anchors, expected_value))
    assert anchor_frame == "model"


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


def test_nonconsecutive_frames_keep_local_semantic_time_indices(monkeypatch):
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
    scene = fixture_scene(
        time_count=7,
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

    # Planted where the body resolves them: the helper moved to transform.py
    # (the model reads its own cloud through it), and sparse_tracking keeps
    # only the alias, which this call goes through so the alias is pinned to
    # the moved function as well.
    assert sparse_module._predicted_pointmaps is transform_module.predicted_pointmaps
    monkeypatch.setattr(
        transform_module,
        "pose_encoding_to_extri_intri",
        fake_pose_conversion,
    )
    monkeypatch.setattr(transform_module, "as_homogeneous", lambda value: value)
    monkeypatch.setattr(transform_module, "unproject_depth", fake_unproject)

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

    anchors, anchor_frame = gather_query_anchor_points(
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
    assert anchor_frame == "model"
    # The anchor gather reads reconstruction only -- depth and pose_enc -- so
    # one shared forward serves every anchor. It indexes by the adapter's anchor
    # list, never by a forward's track_query_idx, which is why a dict carrying
    # no track queries at all still works.
    torch.testing.assert_close(
        gather_query_anchor_points({}, dumped_scene, correspondences)[0],
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


def test_the_oracle_anchor_returns_the_ground_truth_query_positions():
    """With the oracle on, the anchor IS the tracked point's true position.

    Queries at t=2 rather than t=0, so the time index is load-bearing: the
    fixture moves every point 0.1 m per frame, and an anchor read off
    ``trajectories_world[0]`` would be 20 cm out.
    """

    scene = fixture_scene(
        query_times=[2, 2, 2],
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 2),),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    assert correspondences.count == 3

    anchors, frame = gather_query_anchor_points(
        {}, scene, correspondences, oracle_query_anchor=True
    )

    expected = scene.trajectories_world[
        correspondences.query_times, correspondences.trajectory_indices
    ]
    assert torch.equal(anchors, expected)
    assert anchors.shape == (correspondences.count, 3)
    assert anchors.dtype == torch.float32
    assert frame == "world"
    assert anchors.device == scene.trajectories_world.device
    assert not anchors.requires_grad
    # Not the time-0 positions: the query time selected the row.
    assert not torch.equal(
        anchors, scene.trajectories_world[0, correspondences.trajectory_indices]
    )


def test_the_oracle_anchor_reads_no_prediction(dumped_scene):
    """The oracle path never touches the reconstruction.

    An empty prediction dict is the sharpest way to say so: off, the gather
    reaches ``_predicted_pointmaps`` and dies on the missing alignment fields;
    on, it returns the ground truth from the scene alone. The stray-slot
    contract is the same either way -- the slot check sits ahead of both paths.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)

    with pytest.raises(KeyError, match="missing alignment fields"):
        gather_query_anchor_points({}, dumped_scene, correspondences)

    anchors, frame = gather_query_anchor_points(
        {}, dumped_scene, correspondences, oracle_query_anchor=True
    )
    assert torch.equal(
        anchors,
        dumped_scene.trajectories_world[
            correspondences.query_times, correspondences.trajectory_indices
        ],
    )
    assert frame == "world"

    stray = SparseCorrespondences(
        trajectory_indices=correspondences.trajectory_indices,
        query_slots=torch.ones_like(correspondences.query_slots),
        query_times=correspondences.query_times,
        rows=correspondences.rows,
        columns=correspondences.columns,
    )
    with pytest.raises(ValueError, match="exceeds the adapter's query observations"):
        gather_query_anchor_points({}, dumped_scene, stray, oracle_query_anchor=True)


def test_the_oracle_anchor_is_off_by_default(dumped_scene, monkeypatch):
    """Off, and off by default, is the predicted path unchanged.

    The bare call, an explicit ``oracle_query_anchor=False`` and an explicit
    ``ground_truth_query_anchor=False`` all return ``(points, frame)`` with
    the pointmap gather at the query pixels and frame ``"model"`` -- the
    snapshot the test above pins.  Both parameters are keyword-only with a
    False default, which is what keeps ``train_step``'s numerics
    byte-identical: its calls stay bare, and only the mechanical tuple unpack
    changed.
    """

    import inspect

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    height, width = dumped_scene.views[0]["img"].shape[-2:]
    pointmaps = torch.arange(
        dumped_scene.num_observations * height * width * 3,
        dtype=torch.float32,
    ).reshape(1, dumped_scene.num_observations, height, width, 3)
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)
    raw = {"track_query_idx": dumped_scene.track_query_observation_slots}
    snapshot = pointmaps[0, 0, correspondences.rows, correspondences.columns]

    bare = gather_query_anchor_points(raw, dumped_scene, correspondences)
    assert torch.equal(bare[0], snapshot) and bare[1] == "model"
    for kwargs in (
        {"oracle_query_anchor": False},
        {"ground_truth_query_anchor": False},
    ):
        points, frame = gather_query_anchor_points(
            raw, dumped_scene, correspondences, **kwargs
        )
        assert torch.equal(points, snapshot) and frame == "model"

    for name in ("oracle_query_anchor", "ground_truth_query_anchor"):
        parameter = inspect.signature(gather_query_anchor_points).parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is False


def test_a_world_frame_anchor_is_composed_in_world_not_double_transformed(
    dumped_scene, monkeypatch
):
    """The geometric test that would have caught the double transform.

    The planted pointmaps are the exact similarity preimage of the anchor's
    metric pointmap, so the fitted gauge is known and far from identity
    (scale 2.25, rotation and translation both nonzero), and the model-gauge
    anchors and the oracle anchors describe the same physical points up to a
    deliberate 2 cm perturbation.  The displacement field is exact in the
    model gauge, so the world-frame arm must cancel to zero while the
    predicted arm carries exactly the perturbation's image, scale * |eps| *
    tuf.  Under the pre-fix composition the world arm computed
    apply_points(gt + d), off by |s*R*gt + t - gt| * tuf -- scene magnitudes
    rather than centimetres; at the identity gauge every older fixture
    planted, the two compositions coincide, which is why the bug survived.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    dumped_scene.track_upscaling_factor = 2.5

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
    source = ((target - np.array([0.3, -0.2, 1.1])) @ rotation) / 2.25
    pointmaps = torch.zeros(
        1, dumped_scene.num_observations, target.shape[0], target.shape[1], 3
    )
    pointmaps[0, 0] = torch.from_numpy(source).float()
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)
    raw_fit = {
        "depth_conf": torch.ones(
            1, dumped_scene.num_observations, target.shape[0], target.shape[1]
        )
    }
    alignment, _ = fit_scene_sim3(raw_fit, dumped_scene, confidence_percentile=0)
    assert alignment.scale.item() == pytest.approx(2.25, rel=1e-4)

    oracle_anchors, oracle_frame = gather_query_anchor_points(
        {}, dumped_scene, correspondences, oracle_query_anchor=True
    )
    assert oracle_frame == "world"
    # The same physical points in the model gauge, plus a deliberate error so
    # "oracle strictly closer" cannot pass as a tie.
    eps = torch.full((correspondences.count, 3), 0.02)
    model_anchors = (
        (oracle_anchors - alignment.translation) @ alignment.rotation
    ) / alignment.scale + eps

    height, width = dumped_scene.views[0]["img"].shape[-2:]
    tracks = torch.zeros(1, 1, dumped_scene.num_observations, height, width, 3)
    for item, trajectory_index in enumerate(
        correspondences.trajectory_indices.tolist()
    ):
        row = int(correspondences.rows[item])
        column = int(correspondences.columns[item])
        query_time = int(correspondences.query_times[item])
        for slot, original_time in enumerate(dumped_scene.slot_times.tolist()):
            motion = (
                dumped_scene.trajectories_world[original_time, trajectory_index]
                - dumped_scene.trajectories_world[query_time, trajectory_index]
            )
            # apply_vectors of this is exactly the ground-truth world motion.
            tracks[0, 0, slot, row, column] = (
                motion @ alignment.rotation
            ) / alignment.scale
    raw = {
        "track_multi": tracks,
        "track_query_idx": dumped_scene.track_query_observation_slots,
    }

    predicted = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        alignment,
        model_anchors,
        query_anchor_frame="model",
    )
    oracle = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        alignment,
        oracle_anchors,
        query_anchor_frame="world",
    )

    # Every predicted-arm sample is off by exactly the perturbation's image,
    # scale * |eps| * tuf (a rotation preserves the norm), identical for
    # every sample, so the masked mean equals it whatever the mask.
    expected_error = (
        float(alignment.scale.item())
        * float(torch.linalg.vector_norm(eps[0]))
        * float(dumped_scene.track_upscaling_factor)
    )
    assert predicted.metric_error.item() == pytest.approx(expected_error, rel=1e-3)
    assert oracle.metric_error.item() < 1e-4
    assert oracle.metric_error.item() < predicted.metric_error.item()

    with pytest.raises(ValueError, match="anchor_frame must be"):
        compose_predicted_metric(
            oracle_anchors,
            torch.zeros_like(oracle_anchors)[:, None, :],
            alignment,
            anchor_frame="stored",
            metric_factor=1.0,
        )


def test_the_ground_truth_anchor_unprojects_the_anchor_depth():
    """--ground_truth_query_anchor is the pixel's GT-depth unprojection.

    Two anchors on purpose -- track 1 is invisible to camera 0 at t=0, so
    the correspondences span two distinct anchor slots and the per-slot loop
    is load-bearing: an all-S materialisation with wrong slot bookkeeping,
    a slot-0-for-everything gather, or a second unprojection all change the
    camera-1 row's exact value.  The empty raw dict proves no prediction is
    read.  Against the oracle the pair differs (pixel quantisation is real),
    but the along-ray half of the difference is bounded by the very
    depth_error_m that anchor_depth_gate admitted -- the gathered anchor
    sits on the rounded pixel's ray with camera-z equal to the depth map's
    value there; nothing is claimed about the lateral component.
    """

    scene = fixture_scene(
        invisible=[(0, 0, 1)],
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    assert correspondences.count == 3
    anchor_slots = scene.track_query_observation_slots[correspondences.query_slots]
    assert len(set(anchor_slots.tolist())) == 2

    anchors, frame = gather_query_anchor_points(
        {}, scene, correspondences, ground_truth_query_anchor=True
    )
    oracle, oracle_frame = gather_query_anchor_points(
        {}, scene, correspondences, oracle_query_anchor=True
    )

    assert frame == "world" and oracle_frame == "world"
    assert anchors.shape == (3, 3)
    assert anchors.dtype == torch.float32
    assert anchors.device.type == "cpu"
    assert not anchors.requires_grad
    for item in range(correspondences.count):
        slot = int(anchor_slots[item])
        world_points, _ = sparse_module._metric_pointmap_at_anchor(scene, slot)
        expected = torch.from_numpy(
            world_points[
                int(correspondences.rows[item]), int(correspondences.columns[item])
            ]
        ).to(torch.float32)
        assert torch.equal(anchors[item], expected)

    assert not torch.equal(anchors, oracle)
    # 0.10 m is anchor_depth_tolerance_m; the eligibility comparison's own
    # 1e-5 slack is in STORED units, so it scales with tuf.
    tuf = float(scene.track_upscaling_factor)
    for item in range(correspondences.count):
        slot = int(anchor_slots[item])
        observation = scene.observations[slot]
        world_to_camera = scene.extrinsics_world_to_camera[
            observation.camera, observation.original_time
        ].double()

        def camera_z(point):
            return float(
                (world_to_camera[:3, :3] @ point.double() + world_to_camera[:3, 3])[2]
            )

        assert (
            abs(camera_z(anchors[item]) - camera_z(oracle[item])) * tuf
            <= 0.10 + 1e-5 * tuf
        )


def test_the_ground_truth_anchor_refuses_an_invalid_depth_pixel(
    dumped_scene, monkeypatch
):
    """Raise, never drop: a dropped row would shrink the correspondence set
    out from under the comparison curve.  Eligibility gated on this same
    depth map, so an invalid pixel here means the correspondences do not
    belong to this scene."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    real = sparse_module._metric_pointmap_at_anchor

    def invalidating(scene, slot):
        world_points, valid = real(scene, slot)
        valid = valid.copy()
        valid[int(correspondences.rows[0]), int(correspondences.columns[0])] = False
        return world_points, valid

    monkeypatch.setattr(sparse_module, "_metric_pointmap_at_anchor", invalidating)

    with pytest.raises(ValueError, match="anchor_depth_gate"):
        gather_query_anchor_points(
            {}, dumped_scene, correspondences, ground_truth_query_anchor=True
        )


def test_the_anchor_diagnostics_are_refused_together(dumped_scene):
    """Each flag replaces the same anchor with a different ground-truth
    reading; the raise inside the gather covers callers that bypass the
    trainer's parse-time refusal."""

    correspondences, _ = build_anchor_correspondences(dumped_scene)

    with pytest.raises(ValueError, match="cannot be combined"):
        gather_query_anchor_points(
            {},
            dumped_scene,
            correspondences,
            oracle_query_anchor=True,
            ground_truth_query_anchor=True,
        )


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
    dumped_scene.depth[0, 0, 0, original_row, original_column] = 8.0
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
    model.motion_decoder = _LinearMotionDecoder(1, 1)
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
        # The refiner rides inside motion_decoder but set_freeze re-freezes it
        # by name unless refine=True, so it neither trains nor takes gradient.
        should_train = name.startswith(expected_prefixes) and not name.startswith(
            "motion_decoder.refiner."
        )
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
        # The refiner rides inside motion_decoder but set_freeze re-freezes it
        # by name unless refine=True, so it neither trains nor takes gradient.
        should_train = name.startswith(expected_prefixes) and not name.startswith(
            "motion_decoder.refiner."
        )
        assert parameter.requires_grad is should_train
        assert (parameter.grad is not None) is should_train


def test_save_reload_preserves_temporal_embedding(tmp_path):
    source = _tiny_arc()
    with torch.no_grad():
        source.backbone.pretrained.time_index_embedding.weight.fill_(3.25)
        source.motion_decoder.weight.fill_(1.5)
        source.track_head.bias.fill_(-2.0)
    checkpoint = _trainer_patch(source, tmp_path / "patch")
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


def test_sparse_loss_velocity_term_composes_and_defaults_off(dumped_scene):
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
    with_velocity = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
        velocity_weight=0.5,
    )

    # Default path untouched: no velocity graph is built at weight 0.
    assert base.velocity_loss is None
    assert base.velocity_pair_count is None
    assert base.total_loss is base.loss

    assert with_velocity.velocity_loss is not None
    assert with_velocity.velocity_loss.item() > 0
    assert with_velocity.velocity_pair_count > 0
    torch.testing.assert_close(with_velocity.loss, base.loss)
    torch.testing.assert_close(
        with_velocity.total_loss,
        with_velocity.loss + 0.5 * with_velocity.velocity_loss,
    )
    assert set(with_velocity.loss_breakdown) == {"position", "velocity"}


def test_the_velocity_residual_is_reported_at_weight_zero(dumped_scene):
    """The whole point of putting it before the fast path.

    --confidence_weight was trained with no observable that could see it. A term
    measurable without being trained lets a weight-0 run record the baseline, so
    turning the term on can be judged rather than hoped about.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    _, raw = _shared_conv_predictions(dumped_scene)
    anchors = _anchors_for(dumped_scene, correspondences)

    off = sparse_tracking_loss(
        raw, dumped_scene, correspondences, _identity_alignment(), anchors
    )
    assert off.velocity_loss is None, "not trained"
    assert off.velocity_stats is not None, "but measured"
    assert off.velocity_stats["mean_m"] >= 0.0
    # 2 cameras x 4 times, paired within each camera: 2 * (4 - 1).
    assert off.velocity_stats["pair_count"] == 6

    # And skipped entirely when the caller says it is not reading diagnostics --
    # which is what every training step says.
    stepwise = sparse_tracking_loss(
        raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
        collect_diagnostics=False,
    )
    assert stepwise.velocity_stats is None


def test_velocity_pairs_never_span_two_cameras(dumped_scene):
    """Eq. 8 is monocular, and an ungrouped pair would smuggle sync in.

    A cross-camera pair equals the same-camera velocity plus the
    synchronized-consistency residual at the later time. The scene here is
    camera-major 2x4, so grouping is the difference between 6 pairs -- (T-1)*V --
    and 12 -- (T-1)*V**2. At the committed 4x12 window it is 44 against 176, i.e.
    three quarters of the pairs would be measuring the other term's quantity.
    """

    times = dumped_scene.slot_time_indices.reshape(-1)
    cameras = dumped_scene.slot_cameras.reshape(-1)

    grouped_first, grouped_second, _ = adjacent_pair_indices(times, cameras)
    assert len(grouped_first) == 6
    for earlier, later in zip(grouped_first, grouped_second):
        assert cameras[earlier] == cameras[later]

    assert len(adjacent_pair_indices(times)[0]) == 12


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
# reconstruction drift vs. ground truth
# ------------------------------------------------------------------------------


def _ground_truth_raw_reconstruction(scene):
    """depth and pose_enc reproducing the ground truth exactly under identity Sim(3)."""

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
    # The window's one anchor is the reference, which has no offset to measure.
    assert report["base_ratio"] == {"0:0": None}


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
    """Rotate and rescale the *predicted* world frame; the ground truth stays put."""

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


def test_drift_report_relative_rotation_survives_a_single_camera_window():
    """One camera means no baseline anywhere, and rotation must outlive that.

    A fit taken over every slot would make the numerator identically zero here,
    so ``baseline_scale`` would be 0 and every centre residual would collapse to
    ``||0 * c_pred - 0|| == 0`` -- the report would call a drifting model perfect.
    Restricting the fit empties its input instead, which reaches ``None``.
    """

    scene = fixture_scene(
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


def test_drift_report_relative_pose_is_anchored_at_the_query_observation():
    """The reference is the anchor, not slot 0.

    ``rotated_camera=1`` gives the anchor camera a real yaw, pitch and offset, so
    a transposed rotation moves the numbers.  The wander is placed on the
    anchor's *own* camera, which is what makes the choice of reference readable:
    slot 3 shares camera 1 with the anchor and so owes zero baseline, but under a
    slot-0 reference it would be a separated slot instead and the two groups
    below would swap.
    """

    scene = fixture_scene(
        rotated_camera=1,
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
# base_ratio and the query anchor's error, the held-out readouts
# ------------------------------------------------------------------------------


def _steered_to(cameras_by_track, view_count=4):
    """``invisible=`` triples leaving track k visible at time 0 in one camera.

    Every fixture query starts at time 0 and the planes tie on depth error and
    rounding, so assignment falls to anchor order and every row lands on the
    first anchor; hiding each track from all cameras but one steers it.
    """

    return tuple(
        (camera, 0, track)
        for track, target in cameras_by_track.items()
        for camera in range(view_count)
        if camera != target
    )


def _four_camera_scene(**scene_parameters):
    """The fixture world from four cameras, one anchor each at time 0: the live spec."""

    return fixture_scene(
        view_count=4,
        cameras=(0, 1, 2, 3),
        times=(0, 1),
        size=56,
        query_anchors=((0, 0), (1, 0), (2, 0), (3, 0)),
        **scene_parameters,
    )


def _skewed_alignment(scale: float = 0.37) -> DetachedSim3:
    """Scale, rotation and translation all non-trivial, so points and vectors part."""

    return DetachedSim3(
        scale=torch.tensor(scale),
        rotation=torch.from_numpy(_yaw_rotation(31.0) @ _pitch_rotation(17.0)).float(),
        translation=torch.tensor([0.8, -1.3, 2.0]),
    )


def _preimage(alignment: DetachedSim3, points: torch.Tensor) -> torch.Tensor:
    """The model-gauge points ``alignment.apply_points`` carries onto ``points``.

    Inverted exactly in float64: the float32 rotation is orthonormal only to
    about 1e-7, so its transpose is not quite its inverse.
    """

    rotation = alignment.rotation.double()
    return (
        (points.double() - alignment.translation.double()) / alignment.scale.double()
    ) @ torch.linalg.inv(rotation.mT)


def test_base_ratio_is_one_for_ground_truth_cameras_and_pointmaps(monkeypatch):
    """Stage 0A's readout at its fixed point, at the live four-anchor spec.

    The reference pointmap is planted as the scene's own metric one, so the
    Sim(3) is fitted from ground-truth pointmaps -- scale exactly 1 -- and the
    cameras are ground truth too, so every baseline has its true length.
    """

    scene = _four_camera_scene()
    target, _ = sparse_module._metric_pointmap_at_anchor(
        scene, scene.query_observation_slot
    )
    pointmaps = torch.from_numpy(target).float().expand(
        1, scene.num_observations, *target.shape
    ).contiguous()
    monkeypatch.setattr(sparse_module, "_predicted_pointmaps", lambda raw: pointmaps)
    raw = _ground_truth_raw_reconstruction(scene)
    alignment, _ = fit_scene_sim3(raw, scene)
    assert float(alignment.scale.item()) == pytest.approx(1.0, rel=1e-6)

    ratios = reconstruction_drift_report(raw, scene, alignment)["base_ratio"]

    assert list(ratios) == ["0:0", "1:0", "2:0", "3:0"]
    assert ratios["0:0"] is None
    for key in ("1:0", "2:0", "3:0"):
        assert ratios[key] == pytest.approx(1.0, rel=1e-6)


def test_base_ratio_scales_linearly_and_lands_on_its_own_anchor():
    """Each anchor reads its own camera's baseline length times the Sim(3) scale.

    Every non-reference anchor's centre moves along its baseline by its own
    factor, so the ratios are distinct and one filed under the wrong anchor
    fails. The cameras are recorded as ids 10-13 and the spec is permuted,
    putting the reference on the second view: a key built from view indices
    or spec positions, or anchors paired in camera order, cannot pass. Only
    the alignment's scale may enter -- not its rotation or translation -- and
    not the track upscaling factor (2.5 here), which cancels between lengths.
    """

    scene = dataclasses.replace(
        fixture_scene(
            view_count=4,
            view_ids=(10, 11, 12, 13),
            cameras=(10, 11, 12, 13),
            times=(0, 1),
            size=56,
            query_anchors=((11, 0), (13, 0), (10, 0), (12, 0)),
        ),
        track_upscaling_factor=2.5,
    )
    factors = {"13:0": 0.6, "10:0": 1.25, "12:0": 0.9}
    raw = _ground_truth_raw_reconstruction(scene)
    pose_encoding = raw["pose_enc"].clone()
    reference = pose_encoding[0, scene.query_observation_slot, :3].clone()
    for (camera, time), slot in zip(
        scene.query_anchors, scene.anchor_observation_slots
    ):
        factor = factors.get(f"{camera}:{time}")
        if factor is not None:
            pose_encoding[0, slot, :3] = reference + factor * (
                pose_encoding[0, slot, :3] - reference
            )

    ratios = reconstruction_drift_report(
        {**raw, "pose_enc": pose_encoding}, scene, _skewed_alignment(scale=2.0)
    )["base_ratio"]

    assert list(ratios) == ["11:0", "13:0", "10:0", "12:0"]
    assert ratios["11:0"] is None
    for key, factor in factors.items():
        assert ratios[key] == pytest.approx(2.0 * factor, rel=1e-5)


def test_base_ratio_is_none_without_a_baseline_and_zero_for_a_collapsed_camera():
    """None where there is no length to compare against; 0.0 is a measurement.

    The reference anchor has no offset of its own. Anchor 0:3 shares its
    camera, with the ground-truth centre nudged 1.5e-6 off the reference: a
    zero baseline under the fit's own test, 1e-6 of the largest baseline (2 m,
    camera 2's), though neither under ``> 0`` nor under an absolute 1e-6, each
    of which would divide the planted 30 cm wander by it. Anchor 1:3's
    predicted centre is collapsed onto the reference: its baseline exists and
    the prediction measures zero along it -- and still does when the whole rig
    collapses and the fit has no baseline_scale left. Original time 3 sits at
    semantic slot 2 of this window, so a key built from the slot cannot pass.
    """

    scene = fixture_scene(
        view_count=3,
        cameras=(0, 1, 2),
        times=(0, 2, 3),
        size=56,
        query_anchors=((0, 0), (0, 3), (1, 0), (1, 3)),
    )
    extrinsics = scene.extrinsics_world_to_camera.clone()
    extrinsics[0, 3, 0, 3] -= 1.5e-6  # camera 0 at time 3: centre x = +1.5e-6
    scene.extrinsics_world_to_camera = extrinsics
    slots = dict(zip(scene.query_anchors, scene.anchor_observation_slots))
    raw = _moved_center(
        _ground_truth_raw_reconstruction(scene),
        slot=slots[(0, 3)],
        offset=(0.3, 0.0, 0.0),
    )
    pose_encoding = raw["pose_enc"].clone()
    pose_encoding[0, slots[(1, 3)], :3] = pose_encoding[0, slots[(0, 0)], :3]

    ratios = reconstruction_drift_report(
        {**raw, "pose_enc": pose_encoding}, scene, _identity_alignment()
    )["base_ratio"]

    assert list(ratios) == ["0:0", "0:3", "1:0", "1:3"]
    assert ratios["0:0"] is None
    assert ratios["0:3"] is None
    assert ratios["1:0"] == pytest.approx(1.0, rel=1e-6)
    assert ratios["1:3"] == 0.0

    collapsed = raw["pose_enc"].clone()
    collapsed[0, :, :3] = collapsed[0, slots[(0, 0)], :3]
    report = reconstruction_drift_report(
        {**raw, "pose_enc": collapsed}, scene, _identity_alignment()
    )
    assert report["pose"]["baseline_scale"] is None
    assert report["base_ratio"] == {"0:0": None, "0:3": None, "1:0": 0.0, "1:3": 0.0}


def test_base_ratio_reads_the_offsets_length_not_its_projection():
    """Stage 0A's ratio is of lengths, so a camera moved across its baseline reads long.

    Camera 1 keeps its 1 m along the baseline and gains 0.75 m across it: the
    offset grows to 1.25 m (a 3-4-5 triangle) while its projection onto the
    baseline stays 1 m.
    """

    scene = fixture_scene(
        cameras=(0, 1), times=(0, 1), size=56, query_anchors=((0, 0), (1, 0))
    )
    slots = dict(zip(scene.query_anchors, scene.anchor_observation_slots))
    raw = _moved_center(
        _ground_truth_raw_reconstruction(scene),
        slot=slots[(1, 0)],
        offset=(0.0, 0.75, 0.0),
    )

    ratios = reconstruction_drift_report(raw, scene, _identity_alignment())["base_ratio"]

    assert ratios["1:0"] == pytest.approx(1.25, rel=1e-6)


def test_anchor_errors_refuse_misfiled_rows_and_take_an_empty_set(dumped_scene):
    """One anchor per row, and every row's slot a seated anchor; nothing dropped.

    A short anchor tensor would broadcast against the truth, and a slot past
    the last anchor would file its rows under none -- both silently -- so both
    refuse. An empty set is no error: every anchor reads an empty array.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    truth = _anchors_for(dumped_scene, correspondences)

    with pytest.raises(ValueError, match="model_anchors must have shape"):
        query_anchor_errors(
            truth[:1], dumped_scene, correspondences, _identity_alignment()
        )
    stray = dataclasses.replace(
        correspondences,
        query_slots=torch.full_like(
            correspondences.query_slots, len(dumped_scene.query_anchors)
        ),
    )
    with pytest.raises(ValueError, match="must index the scene's"):
        query_anchor_errors(truth, dumped_scene, stray, _identity_alignment())

    empty = SparseCorrespondences(*(torch.empty(0, dtype=torch.long),) * 5)
    errors = query_anchor_errors(
        torch.empty(0, 3), dumped_scene, empty, _identity_alignment()
    )
    assert list(errors) == ["0:0"]
    assert errors["0:0"].size == 0


def test_drift_depth_reads_the_surface_depth_map():
    """Time-0 depth comes through surface_depth_map, like every other consumer's.

    build_scene refuses a depth0 that differs from depth[:, 0], so the two are
    split after construction: each camera's time-0 map is inflated by its own
    factor in ``depth`` alone. The prediction is depth0's values, so each
    camera reads exactly its own inflation -- and nothing, were depth0 read.
    """

    scene = fixture_scene(cameras=(0, 1), times=(0, 1), size=56)
    raw = _ground_truth_raw_reconstruction(scene)
    factors = {0: 1.1, 1: 1.25}
    depth = scene.depth.clone()
    for camera, factor in factors.items():
        depth[camera, 0] *= factor
    scene.depth = depth

    report = reconstruction_drift_report(raw, scene, _identity_alignment())

    for camera, factor in factors.items():
        assert report["depth"][str(camera)]["median_relative_error"] == (
            pytest.approx(abs(1.0 - factor) / factor, rel=1e-5)
        )


def test_anchor_error_is_zero_at_the_true_position(dumped_scene):
    """An anchor that IS the tracked point scores zero, through any gauge.

    The skewed alignment's preimage of the truth comes back exactly through
    apply_points; apply_vectors would drop the translation and miss by 2.5 m.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    truth = _anchors_for(dumped_scene, correspondences)
    skewed = _skewed_alignment()

    exact = query_anchor_errors(
        truth, dumped_scene, correspondences, _identity_alignment()
    )
    gauged = query_anchor_errors(
        _preimage(skewed, truth), dumped_scene, correspondences, skewed
    )

    for errors in (exact, gauged):
        assert list(errors) == ["0:0"]
        assert errors["0:0"].shape == (correspondences.count,)
    assert np.all(exact["0:0"] == 0.0)
    np.testing.assert_allclose(gauged["0:0"], 0.0, atol=1e-9)


def test_anchor_error_reads_a_planted_offset_in_metres(dumped_scene):
    """Each row reads its own planted miss, lifted to metres by the factor.

    The factor is 2.5, not the fixture's 1.0, which would hide a missing
    multiplication; every row's offset is distinct, so a row measured against
    another row's truth fails too.
    """

    scene = dataclasses.replace(dumped_scene, track_upscaling_factor=2.5)
    correspondences, _ = build_anchor_correspondences(scene)
    offsets = torch.tensor(
        [[0.01 * (row + 1), -0.02 * row, 0.005] for row in range(correspondences.count)],
        dtype=torch.float64,
    )
    skewed = _skewed_alignment()
    anchors = _preimage(skewed, _anchors_for(scene, correspondences).double() + offsets)

    errors = query_anchor_errors(anchors, scene, correspondences, skewed)["0:0"]

    np.testing.assert_allclose(
        errors, torch.linalg.vector_norm(offsets, dim=-1).numpy() * 2.5, rtol=1e-9
    )


def test_anchor_errors_are_the_pre_move_formula_bit_for_bit(dumped_scene):
    """The composition moved into aligned_query_anchors, and no figure moved.

    The visual dump shares the readout's anchor composition through
    aligned_query_anchors, so the readout's numbers must be exactly the inline
    formula it used before the move -- written out here as the expectation,
    through a gauge where every term is non-trivial, at a factor that is not 1,
    from float32 anchors on the model's side of the fit.
    """

    scene = dataclasses.replace(dumped_scene, track_upscaling_factor=2.5)
    correspondences, _ = build_anchor_correspondences(scene)
    skewed = _skewed_alignment()
    anchors = _preimage(skewed, _anchors_for(scene, correspondences).double() + 0.01).float()

    stored = skewed.to(device=torch.device("cpu"), dtype=torch.float64)
    truth = scene.trajectories_world.detach().to(device="cpu", dtype=torch.float64)[
        correspondences.query_times.cpu(),
        correspondences.trajectory_indices.cpu(),
    ]
    aligned = stored.apply_points(anchors.detach().to(device="cpu", dtype=torch.float64))
    expected = (
        torch.linalg.vector_norm(aligned - truth, dim=-1) * float(scene.track_upscaling_factor)
    ).numpy()

    np.testing.assert_array_equal(
        query_anchor_errors(anchors, scene, correspondences, skewed)["0:0"], expected
    )
    shared, true_positions = aligned_query_anchors(anchors, scene, correspondences, skewed)
    for tensor in (shared, true_positions):
        assert tensor.dtype == torch.float64
        assert tensor.device.type == "cpu"
    assert torch.equal(shared, aligned)
    assert torch.equal(true_positions, truth)


@pytest.mark.parametrize(
    ("scene_parameters", "counts"),
    (
        # Camera ids and original times that are neither view indices nor
        # semantic slots: track 1's query time 2 lands it on anchor (11, 2),
        # whose view index is 1 and semantic time 1.
        (
            dict(
                view_ids=(10, 11),
                cameras=(10, 11),
                times=(0, 2),
                query_anchors=((10, 0), (11, 2)),
                query_times=(0, 2, 0),
            ),
            {"10:0": 2, "11:2": 1},
        ),
        # The live spec, rows steered off the reference, which then
        # supervises nothing.
        (
            dict(
                view_count=4,
                cameras=(0, 1, 2, 3),
                times=(0, 1),
                query_anchors=((0, 0), (1, 0), (2, 0), (3, 0)),
                invisible=_steered_to({0: 3, 1: 1, 2: 2}),
            ),
            {"0:0": 0, "1:0": 1, "2:0": 1, "3:0": 1},
        ),
    ),
)
def test_anchor_error_lands_on_each_anchor(scene_parameters, counts):
    """Rows group by query slot and file under that anchor's own camera:time.

    Each anchor's rows carry that anchor's own planted miss, so a row filed
    under another anchor -- or a group keyed by anything but the anchor's
    camera id and original time -- fails. Every seated anchor is listed, in
    spec order, an empty one included.
    """

    scene = fixture_scene(size=56, **scene_parameters)
    correspondences, _ = build_anchor_correspondences(scene)
    misses = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
    planted = _anchors_for(scene, correspondences).double()
    planted[:, 0] += misses[correspondences.query_slots]

    errors = query_anchor_errors(
        planted, scene, correspondences, _identity_alignment()
    )

    assert list(errors) == list(counts)
    assert {key: values.size for key, values in errors.items()} == counts
    assert sum(values.size for values in errors.values()) == correspondences.count
    for index, values in enumerate(errors.values()):
        np.testing.assert_allclose(values, misses[index].item(), rtol=1e-9)


def _assert_same_rng_state(after: dict, before: dict) -> None:
    """Every stream ``capture_rng_state`` records, compared field by field."""

    after, before = dict(after), dict(before)
    assert torch.equal(after.pop("torch"), before.pop("torch"))
    cuda_after = after.pop("torch_cuda", [])
    cuda_before = before.pop("torch_cuda", [])
    assert len(cuda_after) == len(cuda_before)
    assert all(torch.equal(a, b) for a, b in zip(cuda_after, cuda_before))
    assert after == before


def test_the_readouts_draw_no_randomness(dumped_scene):
    """No draw at all, rather than one the eval's RNG restore would hide.

    evaluate_held_out restores every stream in a ``finally``, so an eval-level
    check cannot see a draw made inside these; here nothing restores between
    the two snapshots.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    raw = _ground_truth_raw_reconstruction(dumped_scene)
    before = capture_rng_state()

    reconstruction_drift_report(raw, dumped_scene, _identity_alignment())
    query_anchor_errors(
        _anchors_for(dumped_scene, correspondences),
        dumped_scene,
        correspondences,
        _identity_alignment(),
    )

    _assert_same_rng_state(capture_rng_state(), before)


# ------------------------------------------------------------------------------
# the shuffled-index control
# ------------------------------------------------------------------------------


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

    shuffled = runtime_module.shuffled_index_views(scene)

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
    assert runtime_module.shuffled_index_views(
        _shuffle_stub_scene([0], [0, 1, 2])
    ) is None
    assert runtime_module.shuffled_index_views(
        _shuffle_stub_scene([0, 1], [0])
    ) is None


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

    build_optimizer selects the encoder group by module membership and a name
    filter, never by freeze mode or block index, so this holds by construction
    -- but "by construction" is what silently stops being true under a
    refactor, and a sweep over encoder_lr is worthless if the flag misses.
    """

    model = _optimizer_model(freeze, late_global_blocks)

    _, defaulted, encoder_parameters = runtime_module.build_optimizer(model, lr=1e-5)
    assert defaulted["decoder"] == 1e-5
    assert defaulted["embedding"] == 1e-5
    assert defaulted["encoder_blocks"] == pytest.approx(1e-6)
    assert encoder_parameters

    optimizer, explicit, _ = runtime_module.build_optimizer(
        model,
        lr=1e-5,
        encoder_lr=3e-6,
        embedding_lr=2e-5,
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
    optimizer, learning_rates, encoder_parameters = runtime_module.build_optimizer(
        model,
        lr=1e-5,
        encoder_lr=3e-6,
    )
    assert encoder_parameters == []
    assert learning_rates["encoder_blocks"] is None
    assert len(optimizer.param_groups) == 2


def test_build_optimizer_encoder_group_is_the_unfrozen_blocks_only():
    """The embedding lives in its own group, never in the encoder one."""

    model = _optimizer_model("temporal_tracking_late_global", late_global_blocks=1)
    _, _, encoder_parameters = runtime_module.build_optimizer(model, lr=1e-5)

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
    optimizer, _, _ = runtime_module.build_optimizer(model, lr=1e-5)

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
        runtime_module.build_optimizer(model, lr=1e-5)


def test_expected_trainable_set_derives_every_k():
    """The pinned k=4 entry and the per-block constant must span the range.

    At k=14 the derivation has to land exactly on the independently pinned
    temporal_tracking_global_attention entry: that is what proves the middle
    rung and the full preset describe one mask, not two that happen to agree
    at the default.
    """

    narrow_tensors, narrow_parameters = runtime_module.EXPECTED_TRAINABLE_SETS[
        "temporal_tracking"
    ]
    per_block_tensors, per_block_parameters = runtime_module.LATE_GLOBAL_PER_BLOCK
    # Every global-attention block the vitg encoder has (odd blocks from
    # alt_start=13 to depth 40);
    # test_late_global_at_full_k_reproduces_the_global_attention_mask pins the
    # count on the real model.
    all_global_blocks = 14

    for k in range(1, all_global_blocks + 1):
        assert runtime_module.expected_trainable_set(
            "temporal_tracking_late_global", k
        ) == (
            narrow_tensors + per_block_tensors * k,
            narrow_parameters + per_block_parameters * k,
        )

    assert runtime_module.expected_trainable_set(
        "temporal_tracking_late_global",
        all_global_blocks,
    ) == runtime_module.EXPECTED_TRAINABLE_SETS["temporal_tracking_global_attention"]

    # The k-less modes ignore k entirely.
    for mode in ("temporal_tracking", "temporal_tracking_global_attention"):
        assert (
            runtime_module.expected_trainable_set(mode, None)
            == runtime_module.EXPECTED_TRAINABLE_SETS[mode]
        )


# ---------------------------------------------------------------------------
# Per-frame depth
# ---------------------------------------------------------------------------


def test_per_frame_depth_disagreeing_with_depth0_is_rejected():
    arrays = scene_arrays()
    arrays["depth"][0, 0, 0, 5, 5] += 1.0

    with pytest.raises(ValueError, match="differs from depth0"):
        build_scene(**arrays, cameras=(0, 1), times=(0, 1, 2, 3), size=56)


def test_per_frame_depth_shape_mismatch_is_rejected():
    arrays = scene_arrays()
    arrays["depth"] = arrays["depth"][:, :2]

    with pytest.raises(ValueError, match="views x .* frames"):
        build_scene(**arrays, cameras=(0, 1), times=(0, 1, 2, 3), size=56)


def test_view_ids_resolve_anchor_cameras():
    """Cameras and anchors speak original camera ids, not view positions.

    The two coincide only for an ascending, complete view list. Resolving
    through the recorded ``view_ids`` is what stops that convention from being
    load-bearing.
    """

    scene = fixture_scene(
        view_ids=[4, 7],
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
        fixture_scene(
            view_ids=[4, 7],
            cameras=(0, 1),
            times=(0, 1),
            size=56,
        )


# ---------------------------------------------------------------------------
# Anchors at any camera and any time
# ---------------------------------------------------------------------------


def test_nonzero_query_time_anchor_supervises_from_its_own_frame():
    """A query that starts at t=2 is anchored at t=2, not discarded."""

    scene = fixture_scene(
        query_times=[2, 2, 2],
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
    # frame 0, which is exactly the loss per-frame depth recovers. That is
    # reported as an empty set with a split explaining it, not raised -- an
    # anchor set buying nothing is the case most worth measuring.
    at_time_zero = fixture_scene(
        query_times=[2, 2, 2],
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0),),
        size=56,
    )
    empty, empty_report = build_anchor_correspondences(at_time_zero)
    assert empty.count == 0
    assert empty_report["eligible_query_count"] == 0
    assert empty_report["rejected"]["query_time_mismatch"] == 3


def test_nonzero_query_camera_anchor_uses_its_own_depth():
    scene = fixture_scene(
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


def test_a_query_visible_from_two_anchors_yields_one_row():
    """One row per trajectory, whichever anchor wins it.

    Both anchors here can see every query, so under a per-(query, anchor) rule
    this scene would produce six rows and the doubly-visible queries would carry
    twice the gradient of a singly-visible one. Anchor multiplicity is a property
    of where the cameras point, not of how much a point matters, so exactly one
    anchor supervises each query and the totals stay balanced.
    """

    scene = fixture_scene(
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


def test_best_fitting_anchor_wins_over_an_earlier_worse_one():
    """Selection is by fit, not by declaration order.

    Anchor 0 is declared first but its depth map is nudged off the query's
    surface, so its anchor-depth error is larger than anchor 1's while still
    inside the 10 cm gate. The later, better-fitting anchor must win -- that is
    what distinguishes best-fit from first-eligible.
    """

    scene = fixture_scene(
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


def test_second_anchor_recovers_a_query_the_first_cannot():
    """The occlusion recovery, measured.

    Track 1 is invisible in camera 0 at t=0, so no camera-0 anchor can reach
    it -- its pixel there belongs to whatever occludes it. Camera 1 sees it.
    """

    single = fixture_scene(
        invisible=[(0, 0, 1)],
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0),),
        size=56,
    )
    both = fixture_scene(
        invisible=[(0, 0, 1)],
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


def test_eligibility_split_is_exclusive_and_exhaustive():
    """Every query is accounted for exactly once, whatever the anchor set."""

    for anchors in (((0, 0),), ((0, 0), (1, 0)), ((0, 0), (1, 0), (0, 2))):
        scene = fixture_scene(
            query_times=[0, 2, 2],
            invisible=[(0, 0, 0)],
            cameras=(0, 1),
            times=(0, 1, 2, 3),
            query_anchors=anchors,
            size=56,
        )
        _, report = build_anchor_correspondences(scene)
        assert set(report["rejected"]) == set(ELIGIBILITY_REJECTION_STAGES)
        accounted = report["eligible_query_count"] + sum(report["rejected"].values())
        assert accounted == report["total_query_count"] == 3


def test_anchor_depth_gate_still_rejects_at_a_nonzero_time():
    """The 10 cm gate is unchanged, and applies to per-frame depth too."""

    scene = fixture_scene(
        query_times=[2, 2, 2],
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


def test_out_of_bounds_projection_is_rejected():
    """A query projecting outside the crop is counted, not silently kept."""

    scene = fixture_scene(
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


def test_select_query_slot_slices_and_rebases():
    scene = fixture_scene(
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


def test_anchor_sample_counts_come_from_the_loss_masking():
    """The per-anchor weights must be the loss's own mask, not a re-derivation."""

    # Track 1 is reachable only from camera 1, so each anchor owns rows and the
    # counts cannot both come from the same anchor.
    scene = fixture_scene(
        invisible=[(1, 2, 0), (0, 0, 1)],
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)

    counts = anchor_sample_counts(scene, correspondences, 2)

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
    cut_feats, pairs = runtime_module.cut_features(feats)
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
    runtime_module.backward_through_cut(pairs)

    assert cut_total == pytest.approx(float(total.detach()), rel=1e-6)
    for name, parameter in cut_model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], msg=name)


def test_graph_cut_leaves_untouched_taps_at_zero():
    """A tap no anchor read still needs a gradient of the right shape."""

    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    model = _CutToy()
    feats = model.encode(inputs)
    cut_feats, pairs = runtime_module.cut_features(feats)

    # Read exactly one tap, so every other leaf keeps grad None.
    model.anchor_loss(cut_feats, 0).backward()
    assert any(leaf.grad is None for _, leaf in pairs)

    runtime_module.backward_through_cut(pairs)

    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()


def test_graph_cut_passes_non_differentiable_values_through():
    feats = [(torch.ones(2), None, "tap"), 7]

    cut, pairs = runtime_module.cut_features(feats)

    assert pairs == []
    assert cut[0][1] is None and cut[0][2] == "tap" and cut[1] == 7
    # No pairs means nothing to push back; this must be a no-op, not a crash.
    runtime_module.backward_through_cut(pairs)


def test_per_anchor_weighted_supervision_equals_one_combined_loss():
    """The training step's arithmetic, pinned against the loss it stands in for.

    A multi-anchor step never forms the stacked Q=A loss: it scores one anchor
    at a time and backwards each, weighted by that anchor's share of the
    supervised samples. This asserts the two are the same number and the same
    gradient -- which is what makes the memory saving free rather than a change
    of objective.
    """

    # (0,0,1) makes track 1 reachable only from camera 1, so both anchors own
    # rows; (1,2,0) puts an occluded target in the mix so the masks differ.
    scene = fixture_scene(
        invisible=[(1, 2, 0), (0, 0, 1)],
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    anchor_slots = scene.anchor_observation_slots
    counts = anchor_sample_counts(scene, correspondences, len(anchor_slots))
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
        runtime_module.weighted_anchor_total(
            result,
            position_weight=weight,
            confidence_weight=0.0,
            sync_weight=0.0,
            velocity_weight=0.0,
        ).backward()
        accumulated = runtime_module.accumulate_weighted(
            accumulated, result.loss, weight
        )

    assert accumulated == pytest.approx(float(combined.loss.detach()), rel=1e-6)
    torch.testing.assert_close(split_field.grad, combined_field.grad, rtol=1e-5, atol=1e-7)
    assert combined.sample_count == sum(counts)


def test_per_anchor_confidence_weighting_equals_one_combined_loss():
    """The confidence term needs its own shares, not the position term's.

    It deliberately drops the visibility mask -- occluded samples are where the
    error is large, so they are where a low confidence is learned -- and so it
    reduces over a strictly larger set. Weighting it by the position share would
    make the multi-anchor objective quietly differ from the stacked-Q one.
    """

    scene = fixture_scene(
        invisible=[(1, 2, 0), (0, 1, 2), (0, 0, 1)],
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        query_anchors=((0, 0), (1, 0)),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    anchor_slots = scene.anchor_observation_slots
    anchor_count = len(anchor_slots)
    position_counts = anchor_sample_counts(
        scene, correspondences, anchor_count
    )
    confidence_counts = runtime_module.anchor_confidence_counts(
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
        runtime_module.weighted_anchor_total(
            result,
            position_weight=position_weights[anchor_index],
            confidence_weight=confidence_weights[anchor_index],
            sync_weight=0.0,
            velocity_weight=0.0,
        ).backward()
        accumulated = runtime_module.accumulate_weighted(
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


def test_anchor_rows_pairs_the_gather_with_the_rebased_correspondences():
    """``query_slots`` mean the anchor list to the gather, Q to the loss.

    ``select_query_slot`` rebases to 0 for the loss, so a rebased set handed to
    ``gather_query_anchor_points`` would read anchor 0's pointmaps for anchor
    k's pixels and raise nothing. ``anchor_rows`` is the supported pairing, and
    this pins that it selects the same rows in the same order.
    """

    scene = fixture_scene(
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


def test_per_anchor_sync_weighting_equals_one_combined_loss():
    """The sync term decomposes over anchors at 1/A, and that was unpinned.

    ``synchronized_consistency_loss`` reduces with ``reduction="mean"`` spanning
    the Q axis, and every anchor contributes the same element count, so the
    stacked loss is the plain mean of the per-anchor ones. The step loop relies
    on that; the other two equivalence tests both run at sync_weight=0.
    """

    scene = fixture_scene(
        invisible=[(1, 2, 0), (0, 0, 1)],
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
        runtime_module.weighted_anchor_total(
            result,
            position_weight=0.0,
            confidence_weight=0.0,
            sync_weight=1.0 / anchor_count,
            velocity_weight=0.0,
        ).backward()
        accumulated = runtime_module.accumulate_weighted(
            accumulated,
            result.sync_loss,
            1.0 / anchor_count,
        )

    assert accumulated == pytest.approx(float(combined.sync_loss.detach()), rel=1e-6)
    torch.testing.assert_close(
        split_field.grad, combined_field.grad, rtol=1e-5, atol=1e-8
    )


def test_per_anchor_velocity_weighting_equals_one_combined_loss():
    """The claim the sample-share rests on, pinned on values and gradients.

    ``velocity_consistency_loss`` reduces with ``reduction="mean"`` over its
    own masked pair selection, so a stacked forward divides by the TOTAL pair
    count while a per-anchor one divides by that anchor's. Weighting each
    anchor by its share of the pairs is what makes the two equal --
    sum_a (K_a/K) * (S_a/K_a) = (sum_a S_a)/K. Sync's flat 1/A would not:
    the fixture's invisible samples make the per-anchor pair counts differ.
    """

    scene = fixture_scene(
        invisible=[(1, 2, 0), (0, 0, 1)],
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

    counts = anchor_velocity_counts(scene, correspondences, anchor_count)
    assert min(counts) > 0 and len(set(counts)) > 1, (
        "the fixture must give the anchors different pair counts, or the "
        "flat share would pass too"
    )
    shares = [count / sum(counts) for count in counts]

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
            velocity_weight=1.0,
        )

    combined_field = field.clone().requires_grad_(True)
    combined = score(
        combined_field,
        correspondences,
        scene.track_query_observation_slots,
        anchors,
    )
    assert combined.velocity_loss is not None
    assert combined.velocity_pair_count == sum(counts)
    combined.velocity_loss.backward()

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
        # The counts computed from the scene alone must be exactly what the
        # loss reduced over -- nothing in them reads a prediction.
        assert result.velocity_pair_count == counts[anchor_index]
        runtime_module.weighted_anchor_total(
            result,
            position_weight=0.0,
            confidence_weight=0.0,
            sync_weight=0.0,
            velocity_weight=shares[anchor_index],
        ).backward()
        accumulated = runtime_module.accumulate_weighted(
            accumulated,
            result.velocity_loss,
            shares[anchor_index],
        )

    assert accumulated == pytest.approx(
        float(combined.velocity_loss.detach()), rel=1e-6
    )
    torch.testing.assert_close(
        split_field.grad, combined_field.grad, rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("confidence_weight", [0.0, 0.75])
@pytest.mark.parametrize("sync_weight", [0.0, 0.5])
@pytest.mark.parametrize("velocity_weight", [0.0, 0.25])
def test_single_anchor_total_is_bit_identical_to_the_unsplit_loss(
    confidence_weight,
    sync_weight,
    velocity_weight,
):
    """A single-anchor step must be the pre-change path, exactly.

    Every archived run is single-anchor. At one anchor the position and
    confidence shares are both 1.0 and the sync share is 1/1, so
    ``_weighted_anchor_total`` must reproduce ``sparse_tracking_loss``'s own
    ``total_loss`` -- and bit-for-bit, not merely close: ``compose_tracking_loss``
    sums in insertion order and float addition is not associative, so the two
    must also agree on the order of their terms.
    """

    scene = fixture_scene(
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
        velocity_weight=velocity_weight,
    )

    combined = runtime_module.weighted_anchor_total(
        result,
        position_weight=1.0,
        confidence_weight=confidence_weight,
        sync_weight=sync_weight,
        velocity_weight=velocity_weight,
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
    cut_feats, pairs = runtime_module.cut_features(feats)
    assert pairs
    cut_model.anchor_loss(cut_feats, 0).backward()
    runtime_module.backward_through_cut(pairs)
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
    runtime_module.backward_through_cut(bypass_pairs)

    assert {
        name for name, p in direct_model.named_parameters() if p.grad is not None
    } == {name for name, p in cut_model.named_parameters() if p.grad is not None}
    for name, parameter in direct_model.named_parameters():
        torch.testing.assert_close(parameter.grad, through_cut[name], msg=name)


def test_weighted_anchor_total_sums_in_the_loss_s_own_term_order(monkeypatch):
    """Term order is load-bearing, because float addition is not associative.

    ``compose_tracking_loss`` sums in dict insertion order, so a single-anchor
    ``_weighted_anchor_total`` reproduces ``sparse_tracking_loss``'s own
    ``total_loss`` bit for bit only if both insert their terms in the same order.
    Asserted on the order itself rather than on a numeric difference: whether two
    orders actually disagree depends on the values, and on real losses they
    usually happen to coincide -- which would make a value-based test pass while
    the hazard stayed.
    """

    scene = fixture_scene(
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
    # The helper lives in arc.training.runtime and resolves compose_tracking_loss
    # from ITS module globals, so that is where it is patched.
    monkeypatch.setattr(runtime_module, "compose_tracking_loss", recording_compose)

    result = sparse_tracking_loss(
        raw,
        scene,
        correspondences,
        _identity_alignment(),
        _anchors_for(scene, correspondences),
        confidence_weight=0.75,
        confidence_alpha=3.0,
        sync_weight=0.5,
        velocity_weight=0.25,
    )
    runtime_module.weighted_anchor_total(
        result,
        position_weight=1.0,
        confidence_weight=0.75,
        sync_weight=0.5,
        velocity_weight=0.25,
    )

    loss_order, anchor_order = seen
    assert loss_order == ["position", "sync", "velocity", "confidence"]
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


def test_report_records_the_label_quality_each_anchor_won():
    """Best-fit's justification, made visible without a training run.

    The rule exists because the surviving label is the least noisy one. Counts
    alone cannot show that, so each anchor reports the depth error of the labels
    it won and how much it beat the runner-up where there was one.
    """

    scene = fixture_scene(
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


def test_uncontested_wins_report_a_zero_contested_count():
    """No runner-up anywhere must read as 'uncontested', not as missing data."""

    scene = fixture_scene(
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


# ------------------ merged synchronized slots: layout proof, targets, loss ---


def test_camera_major_layout_proves_the_grid(dumped_scene):
    """The one place slot = camera*T + t is proved rather than assumed.

    Every merged reduction reshapes an (M, S) axis into (M, V, T) on this
    helper's say-so; a permuted or transposed grid that slipped past it would
    pool the wrong slots with no shape error anywhere downstream.
    """

    assert camera_major_layout(dumped_scene) == (2, 4)

    with pytest.raises(ValueError, match="slot_time_indices"):
        camera_major_layout(
            dataclasses.replace(
                dumped_scene,
                slot_time_indices=dumped_scene.slot_time_indices.flip(0),
            )
        )
    with pytest.raises(ValueError, match="slot_cameras"):
        camera_major_layout(
            dataclasses.replace(
                dumped_scene,
                slot_cameras=dumped_scene.slot_cameras.flip(0),
            )
        )
    with pytest.raises(ValueError, match="slot_times"):
        camera_major_layout(
            dataclasses.replace(
                dumped_scene,
                slot_times=dumped_scene.slot_times + 1,
            )
        )
    with pytest.raises(ValueError, match="camera-major grid"):
        camera_major_layout(dataclasses.replace(dumped_scene, cameras=(0,)))


def test_camera_major_layout_handles_non_contiguous_times(dumped_scene):
    """slot_times repeats scene.times verbatim, not an arithmetic guess.

    A stride-2 window's times are (0, 2, ...) while slot_time_indices stays
    arange(T); a helper that conflated the two would refuse every strided
    window or, worse, accept a tampered one.
    """

    scene = fixture_scene(
        cameras=(0, 1),
        times=(0, 2, 3),
        size=56,
    )
    assert camera_major_layout(scene) == (2, 3)


def test_sparse_targets_per_time_reduces_visibility_with_any(dumped_scene):
    """Occluded in one camera of two is still visible merged; in both, not.

    This is gt_vis_any's convention, and the merged loss masks on it -- a
    reduction that took camera 0's visibility (the way it takes camera 0's
    positions, which ARE camera-independent) would silently unsupervise every
    sample the other camera still sees.
    """

    scene = fixture_scene(
        scene_name="occl",
        invisible=((0, 2, 2),),
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )
    correspondences, _ = build_anchor_correspondences(scene)
    positions, visible, finite, mask = sparse_targets(scene, correspondences)
    merged = sparse_targets_per_time(scene, correspondences)
    count = positions.shape[0]

    assert torch.equal(merged[0], positions.view(count, 2, 4, 3)[:, 0])
    assert torch.equal(merged[1], visible.view(count, 2, 4).any(dim=1))
    assert torch.equal(merged[2], finite.view(count, 2, 4)[:, 0])
    assert torch.equal(merged[3], merged[1] & merged[2])

    row = (correspondences.trajectory_indices == 2).nonzero(as_tuple=True)[0]
    assert row.numel() == 1
    # Camera 0 lost track 2 at time 2, camera 1 did not: per-slot says so, and
    # the merged column keeps the sample.
    assert not visible[row, 0 * 4 + 2]
    assert visible[row, 1 * 4 + 2]
    assert merged[1][row, 2]

    both = fixture_scene(
        scene_name="occl2",
        invisible=((0, 2, 2), (1, 2, 2)),
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=56,
    )
    both_correspondences, _ = build_anchor_correspondences(both)
    both_merged = sparse_targets_per_time(both, both_correspondences)
    both_row = (both_correspondences.trajectory_indices == 2).nonzero(as_tuple=True)[0]
    assert not both_merged[1][both_row, 2]


def _merged_conv_predictions(scene, seed=0):
    """A merged head's raw dict: one field per time, through the real split."""

    torch.manual_seed(seed)
    height, width = scene.views[0]["img"].shape[-2:]
    conv = nn.Conv2d(2, 4, kernel_size=1)
    features = torch.randn(len(scene.times), 2, height, width)
    track, confidence = activate_head(
        conv(features),
        activation="inv_log",
        conf_activation="expp1",
    )
    return {
        "track_multi": track[None, None],
        "conf_track_multi": confidence[None, None],
        "track_query_idx": scene.track_query_observation_slots.clone(),
    }


def test_the_merged_loss_scores_one_field_per_time(dumped_scene):
    """merge_synchronized_slots flips the loss's observation-axis contract.

    Both mismatches must refuse -- a T-wide grid without the flag and an
    S-wide grid with it -- or a mis-threaded flag would gather displacements
    against the wrong axis and train on garbage; and the sample count must be
    the REDUCED mask's, since it is what weights multi-anchor backwards.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    merged_raw = _merged_conv_predictions(dumped_scene)
    anchors = _anchors_for(dumped_scene, correspondences)

    result = sparse_tracking_loss(
        merged_raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
        merge_synchronized_slots=True,
    )
    reduced_mask = sparse_targets_per_time(dumped_scene, correspondences)[3]
    assert result.sample_count == int(reduced_mask.sum().item())
    assert torch.isfinite(result.loss)

    with pytest.raises(ValueError, match="observation slots"):
        sparse_tracking_loss(
            merged_raw,
            dumped_scene,
            correspondences,
            _identity_alignment(),
            anchors,
        )
    _, per_slot_raw = _shared_conv_predictions(dumped_scene)
    with pytest.raises(ValueError, match="merged head owes"):
        sparse_tracking_loss(
            per_slot_raw,
            dumped_scene,
            correspondences,
            _identity_alignment(),
            anchors,
            merge_synchronized_slots=True,
        )
    # Belt and braces under the parse-time refusal: a caller that bypasses
    # _validate_args must not silently build a sync term with no pairs.
    with pytest.raises(ValueError, match="merge_synchronized_slots"):
        sparse_tracking_loss(
            merged_raw,
            dumped_scene,
            correspondences,
            _identity_alignment(),
            anchors,
            sync_weight=0.5,
            merge_synchronized_slots=True,
        )


def test_the_merged_velocity_pairing_matches_the_anchor_counts(dumped_scene):
    """The three velocity-pairing copies must move together under the merge.

    The loss (sparse_tracking_loss), the diagnostic (temporal_velocity_stats)
    and the multi-anchor weights (anchor_velocity_counts) each derive the
    pairing from the scene; if one kept the per-slot camera grouping while the
    others merged, the per-anchor shares would silently stop matching the loss
    they weight.
    """

    correspondences, _ = build_anchor_correspondences(dumped_scene)
    merged_raw = _merged_conv_predictions(dumped_scene)
    anchors = _anchors_for(dumped_scene, correspondences)

    result = sparse_tracking_loss(
        merged_raw,
        dumped_scene,
        correspondences,
        _identity_alignment(),
        anchors,
        velocity_weight=0.5,
        merge_synchronized_slots=True,
    )
    counts = anchor_velocity_counts(
        dumped_scene,
        correspondences,
        len(dumped_scene.anchor_observation_slots),
        merge_synchronized_slots=True,
    )
    assert result.velocity_pair_count == counts[0] > 0
    # One camera axis left: adjacent pairs over the distinct times, T - 1 of
    # them, against the per-slot path's (T - 1) * V.
    assert result.velocity_stats["pair_count"] == 3

    sample_counts = anchor_sample_counts(
        dumped_scene,
        correspondences,
        len(dumped_scene.anchor_observation_slots),
        merge_synchronized_slots=True,
    )
    reduced_mask = sparse_targets_per_time(
        dumped_scene, correspondences.select_query_slot(0)
    )[3]
    assert sample_counts == [int(reduced_mask.sum().item())]
