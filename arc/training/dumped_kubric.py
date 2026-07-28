"""Adapter for one scene produced by ``triton/4rc/multiview/dump_eval.py``.

The adapter is intentionally camera-major.  It keeps every selected camera/time
pair as a separate 4RC observation while assigning equal ``time_index`` values
to synchronized observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ImageTransform:
    """The deterministic resize and centre crop performed by ``load_images``."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    output_height: int
    output_width: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.original_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.original_height

    def original_to_output(self, xy: np.ndarray) -> np.ndarray:
        """Map original-image ``(x,y)`` coordinates to the track-head grid."""

        xy = np.asarray(xy)
        result = xy.astype(np.float64, copy=True)
        result[..., 0] = result[..., 0] * self.scale_x - self.crop_left
        result[..., 1] = result[..., 1] * self.scale_y - self.crop_top
        return result

    def output_to_original_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Nearest original pixel represented by every output-grid pixel."""

        columns = (
            np.arange(self.output_width, dtype=np.float64) + self.crop_left
        ) / self.scale_x
        rows = (
            np.arange(self.output_height, dtype=np.float64) + self.crop_top
        ) / self.scale_y
        columns = np.clip(
            np.rint(columns).astype(np.int64),
            0,
            self.original_width - 1,
        )
        rows = np.clip(
            np.rint(rows).astype(np.int64),
            0,
            self.original_height - 1,
        )
        return rows, columns


@dataclass(frozen=True)
class Observation:
    slot: int
    camera: int
    original_time: int
    semantic_time_index: int
    path: Path
    image_transform: ImageTransform


@dataclass
class DumpedKubricScene:
    """One deterministic camera-major window and its sparse metric metadata."""

    name: str
    views: list[dict]
    observations: tuple[Observation, ...]
    cameras: tuple[int, ...]
    times: tuple[int, ...]
    slot_cameras: torch.Tensor
    slot_times: torch.Tensor
    slot_time_indices: torch.Tensor
    query_observation_slot: int
    track_query_observation_slots: torch.Tensor
    query_points: torch.Tensor
    trajectories_world: torch.Tensor
    visibility: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics_world_to_camera: torch.Tensor
    depth0: torch.Tensor
    track_upscaling_factor: float

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    @property
    def time_indices(self) -> tuple[int, ...]:
        return tuple(
            observation.semantic_time_index
            for observation in self.observations
        )


def compute_image_transform(
    original_height: int,
    original_width: int,
    *,
    size: int = 512,
    patch_size: int = 14,
    square_ok: bool = False,
) -> ImageTransform:
    """Mirror the geometry in ``arc.dust3r.utils.image.load_images``."""

    if original_height <= 0 or original_width <= 0:
        raise ValueError("Image dimensions must be positive")
    if size <= 0 or patch_size <= 0:
        raise ValueError("size and patch_size must be positive")

    if size <= 392:
        requested_long_edge = round(
            size * max(
                original_width / original_height,
                original_height / original_width,
            )
        )
    else:
        requested_long_edge = size

    longest_edge = max(original_width, original_height)
    resized_width = int(round(original_width * requested_long_edge / longest_edge))
    resized_height = int(round(original_height * requested_long_edge / longest_edge))
    center_x = resized_width // 2
    center_y = resized_height // 2

    if size <= 392:
        half_width = half_height = min(center_x, center_y)
    else:
        half_width = ((2 * center_x) // patch_size) * patch_size // 2
        half_height = ((2 * center_y) // patch_size) * patch_size // 2
        if not square_ok and resized_width == resized_height:
            half_height_float = 3 * half_width / 4
            if not float(half_height_float).is_integer():
                raise ValueError(
                    "The square-image crop is not integral for "
                    f"size={size}, patch_size={patch_size}"
                )
            half_height = int(half_height_float)

    crop_left = center_x - half_width
    crop_top = center_y - half_height
    return ImageTransform(
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=crop_top,
        crop_left=crop_left,
        output_height=2 * half_height,
        output_width=2 * half_width,
    )


def _as_unique_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = []
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"{name}[{position}] must be an integer, got {type(value).__name__}"
            )
        value = int(value)
        if value < 0:
            raise ValueError(f"{name}[{position}] must be non-negative, got {value}")
        result.append(value)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates: {result}")
    return tuple(result)


def _validate_meta(
    meta: np.lib.npyio.NpzFile,
    *,
    scene_path: Path,
) -> None:
    expected = {
        "query_points",
        "traj3d_world",
        "visibility",
        "intrs",
        "extrs",
        "depth0",
        "track_upscaling_factor",
    }
    missing = expected - set(meta.files)
    if missing:
        raise ValueError(
            f"{scene_path / 'meta.npz'} is missing required fields: {sorted(missing)}"
        )

    query_points = meta["query_points"]
    trajectories = meta["traj3d_world"]
    visibility = meta["visibility"]
    intrinsics = meta["intrs"]
    extrinsics = meta["extrs"]
    depth0 = meta["depth0"]

    if query_points.ndim != 2 or query_points.shape[1] != 4:
        raise ValueError(
            f"query_points must have shape (N,4), got {query_points.shape}"
        )
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError(
            f"traj3d_world must have shape (T,N,3), got {trajectories.shape}"
        )
    time_count, track_count = trajectories.shape[:2]
    if query_points.shape[0] != track_count:
        raise ValueError(
            "query_points and traj3d_world disagree on track count: "
            f"{query_points.shape[0]} versus {track_count}"
        )
    if visibility.ndim != 3 or visibility.shape[1:] != (time_count, track_count):
        raise ValueError(
            "visibility must have shape (V,T,N) matching traj3d_world, got "
            f"{visibility.shape}"
        )
    view_count = visibility.shape[0]
    if intrinsics.shape != (view_count, time_count, 3, 3):
        raise ValueError(
            f"intrs must have shape {(view_count, time_count, 3, 3)}, "
            f"got {intrinsics.shape}"
        )
    if extrinsics.shape != (view_count, time_count, 3, 4):
        raise ValueError(
            f"extrs must have shape {(view_count, time_count, 3, 4)}, "
            f"got {extrinsics.shape}"
        )
    if depth0.ndim != 4 or depth0.shape[:2] != (view_count, 1):
        raise ValueError(
            f"depth0 must have shape (V,1,H,W), got {depth0.shape}"
        )


def load_dumped_kubric_scene(
    data_root: str | Path,
    scene_name: str,
    *,
    cameras: Sequence[int] = (0, 1),
    times: Sequence[int] = (0, 1, 2, 3),
    query_camera: int | None = None,
    query_time: int | None = None,
    size: int = 512,
    patch_size: int = 14,
    square_ok: bool = False,
    verbose: bool = False,
) -> DumpedKubricScene:
    """Load one camera-major window from the existing evaluation dump.

    Parsing and ordinary Arc forwarding may select any dumped cameras and
    times.  Sparse metric supervision remains separately restricted to a
    time-zero query because the current dump only contains ``depth0``.
    """

    cameras = _as_unique_int_tuple("cameras", cameras)
    times = _as_unique_int_tuple("times", times)
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(
            f"times must be strictly increasing to define temporal order, got {times}"
        )
    query_camera = cameras[0] if query_camera is None else query_camera
    query_time = times[0] if query_time is None else query_time
    if (
        isinstance(query_camera, bool)
        or not isinstance(query_camera, (int, np.integer))
    ):
        raise TypeError("query_camera must be an integer")
    if (
        isinstance(query_time, bool)
        or not isinstance(query_time, (int, np.integer))
    ):
        raise TypeError("query_time must be an integer")
    query_camera = int(query_camera)
    query_time = int(query_time)
    if query_camera not in cameras:
        raise ValueError(
            f"query_camera {query_camera} is not in selected cameras {cameras}"
        )
    if query_time not in times:
        raise ValueError(f"query_time {query_time} is not in selected times {times}")

    scene_path = Path(data_root) / scene_name
    meta_path = scene_path / "meta.npz"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Scene metadata not found: {meta_path}")

    with np.load(meta_path, allow_pickle=False) as meta:
        _validate_meta(meta, scene_path=scene_path)
        query_points = np.array(meta["query_points"], dtype=np.float32, copy=True)
        trajectories = np.array(meta["traj3d_world"], dtype=np.float32, copy=True)
        visibility = np.array(meta["visibility"], dtype=bool, copy=True)
        intrinsics = np.array(meta["intrs"], dtype=np.float32, copy=True)
        extrinsics = np.array(meta["extrs"], dtype=np.float32, copy=True)
        depth0 = np.array(meta["depth0"], dtype=np.float32, copy=True)
        track_upscaling_factor = float(
            np.asarray(meta["track_upscaling_factor"]).reshape(()).item()
        )

    view_count, time_count = visibility.shape[:2]
    for camera in cameras:
        if camera >= view_count:
            raise ValueError(
                f"Camera {camera} is out of range for scene with {view_count} views"
            )
    for time_index in times:
        if time_index >= time_count:
            raise ValueError(
                f"Time {time_index} is out of range for scene with {time_count} frames"
            )
    if not np.isfinite(track_upscaling_factor) or track_upscaling_factor <= 0:
        raise ValueError(
            "track_upscaling_factor must be finite and positive, got "
            f"{track_upscaling_factor}"
        )

    paths = []
    transforms = []
    for camera in cameras:
        for time_index in times:
            path = scene_path / f"view_{camera}" / f"{time_index:04d}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Observation image not found: {path}")
            with Image.open(path) as image:
                width, height = image.size
            if depth0.shape[-2:] != (height, width):
                raise ValueError(
                    f"depth0 grid {depth0.shape[-2:]} does not match dumped PNG "
                    f"grid {(height, width)} for {path}"
                )
            paths.append(str(path))
            transforms.append(
                compute_image_transform(
                    height,
                    width,
                    size=size,
                    patch_size=patch_size,
                    square_ok=square_ok,
                )
            )

    # Import lazily so metadata/geometry tests do not require torchvision.
    from arc.dust3r.utils.image import load_images

    views = load_images(
        paths,
        size=size,
        square_ok=square_ok,
        verbose=verbose,
        patch_size=patch_size,
    )
    if len(views) != len(paths):
        raise RuntimeError(
            f"Loaded {len(views)} images for {len(paths)} selected observations"
        )

    observations = []
    slot_cameras = []
    slot_times = []
    slot_time_indices = []
    for slot, (view, path, transform) in enumerate(zip(views, paths, transforms)):
        camera = cameras[slot // len(times)]
        semantic_time_index = slot % len(times)
        original_time = times[semantic_time_index]
        actual_height, actual_width = view["img"].shape[-2:]
        if (actual_height, actual_width) != (
            transform.output_height,
            transform.output_width,
        ):
            raise RuntimeError(
                f"Preprocessing geometry mismatch for {path}: predicted "
                f"{(transform.output_height, transform.output_width)}, loaded "
                f"{(actual_height, actual_width)}"
            )
        # load_images numbers its outputs from its own counter, so this catches a
        # loader that reordered or dropped images. The shape check above cannot:
        # every observation in a window is required to share one processed shape,
        # so a permuted list passes it while every slot holds the wrong picture.
        if view.get("idx") != slot:
            raise RuntimeError(
                f"Observation slot {slot} carries loader index "
                f"{view.get('idx')!r} for {path}; the loaded images and the "
                "camera/time grid are out of step"
            )
        view["time_index"] = torch.tensor(
            [semantic_time_index],
            dtype=torch.long,
        )
        observations.append(
            Observation(
                slot=slot,
                camera=camera,
                original_time=original_time,
                semantic_time_index=semantic_time_index,
                path=Path(path),
                image_transform=transform,
            )
        )
        slot_cameras.append(camera)
        slot_times.append(original_time)
        slot_time_indices.append(semantic_time_index)

    query_observation_slot = next(
        observation.slot
        for observation in observations
        if observation.camera == query_camera
        and observation.original_time == query_time
    )
    # Keep one dense query field while leaving every selected observation in S.
    # A later-time query is valid for parsing/inference; sparse metric
    # supervision will reject it until full per-time depth is available.
    track_query_observation_slots = torch.tensor(
        [query_observation_slot],
        dtype=torch.long,
    )
    for view in views:
        view["track_query_idx"] = track_query_observation_slots.clone()

    output_shapes = {tuple(view["img"].shape[-2:]) for view in views}
    if len(output_shapes) != 1:
        raise ValueError(
            "The bounded overfit harness requires all observations to have the "
            f"same processed shape, got {sorted(output_shapes)}"
        )

    return DumpedKubricScene(
        name=scene_name,
        views=views,
        observations=tuple(observations),
        cameras=cameras,
        times=times,
        slot_cameras=torch.tensor(slot_cameras, dtype=torch.long),
        slot_times=torch.tensor(slot_times, dtype=torch.long),
        slot_time_indices=torch.tensor(slot_time_indices, dtype=torch.long),
        query_observation_slot=query_observation_slot,
        track_query_observation_slots=track_query_observation_slots,
        query_points=torch.from_numpy(query_points),
        trajectories_world=torch.from_numpy(trajectories),
        visibility=torch.from_numpy(visibility),
        intrinsics=torch.from_numpy(intrinsics),
        extrinsics_world_to_camera=torch.from_numpy(extrinsics),
        depth0=torch.from_numpy(depth0),
        track_upscaling_factor=track_upscaling_factor,
    )
