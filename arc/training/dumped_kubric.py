"""Adapter for one scene produced by the cluster repository's dump scripts.

The adapter is intentionally camera-major.  It keeps every selected camera/time
pair as a separate 4RC observation while assigning equal ``time_index`` values
to synchronized observations.

Two files are read.  ``meta.npz`` is required and carries the sparse metric
metadata.  ``depth_full.npz`` is an optional sibling holding per-frame depth; it
is emitted only when the dump ran with ``RCMV_DUMP_DEPTH=1``, so a dump made
without it is complete-looking but supports anchors at original time 0 only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# The optional per-frame depth sidecar. Its ``t`` axis is 1:1 with the PNG
# names, and ``depth[:, 0]`` is what ``meta.npz`` stores as ``depth0``.
DEPTH_SIDECAR_NAME = "depth_full.npz"
DEPTH_SIDECAR_KEY = "depth"
DEPTH_SIDECAR_FLAG = "RCMV_DUMP_DEPTH=1"


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
    """One selected camera/time pair.

    ``camera`` is the *view index*: the position along the ``V`` axis of every
    dumped array and the ``view_<camera>`` directory number.  ``camera_id`` is
    the original camera this view was rendered from, recorded by ``view_ids``.
    The two coincide for every dump taken with an ascending, complete view
    list, which is all of them so far, but only ``camera`` may index an array.
    """

    slot: int
    camera: int
    camera_id: int
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
    camera_ids: tuple[int, ...]
    view_ids: torch.Tensor
    times: tuple[int, ...]
    slot_cameras: torch.Tensor
    slot_times: torch.Tensor
    slot_time_indices: torch.Tensor
    query_anchors: tuple[tuple[int, int], ...]
    query_observation_slot: int
    track_query_observation_slots: torch.Tensor
    query_points: torch.Tensor
    trajectories_world: torch.Tensor
    visibility: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics_world_to_camera: torch.Tensor
    depth0: torch.Tensor
    depth: torch.Tensor | None
    depth_sidecar_path: Path | None
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

    @property
    def has_time_varying_depth(self) -> bool:
        return self.depth is not None

    @property
    def anchor_observation_slots(self) -> tuple[int, ...]:
        return tuple(
            int(slot) for slot in self.track_query_observation_slots.tolist()
        )

    def surface_depth_map(self, camera: int, original_time: int) -> torch.Tensor:
        """The ``(H,W)`` camera-z depth map anchoring queries in one observation.

        This is the single place the depth0/sidecar branch lives, so every
        consumer -- the Sim(3) target pointmap and the sparse anchor projection
        alike -- reads the same map for the same observation, and neither has to
        know which file it came from.
        """

        camera = int(camera)
        original_time = int(original_time)
        if not 0 <= camera < self.depth0.shape[0]:
            raise ValueError(
                f"Camera view index {camera} is out of range for "
                f"{self.depth0.shape[0]} dumped views"
            )
        if self.depth is None:
            if original_time != 0:
                raise ValueError(
                    f"Anchoring at original time {original_time} needs per-frame "
                    f"depth, but {DEPTH_SIDECAR_NAME} is absent for scene "
                    f"'{self.name}'. The sidecar is opt-in: re-run the dump with "
                    f"{DEPTH_SIDECAR_FLAG} to emit it. Without it only original "
                    "time 0 can be anchored, because meta.npz carries depth0 alone."
                )
            return self.depth0[camera, 0]
        if not 0 <= original_time < self.depth.shape[1]:
            raise ValueError(
                f"Original time {original_time} is out of range for "
                f"{self.depth.shape[1]} frames of per-frame depth"
            )
        return self.depth[camera, original_time, 0]


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


def _as_query_anchor_pairs(
    values: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    result = []
    for position, pair in enumerate(values):
        try:
            camera, time_index = pair
        except (TypeError, ValueError):
            raise TypeError(
                f"query_anchors[{position}] must be a (camera, time) pair, got "
                f"{pair!r}"
            ) from None
        for name, value in (("camera", camera), ("time", time_index)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"query_anchors[{position}] {name} must be an integer, got "
                    f"{type(value).__name__}"
                )
            if int(value) < 0:
                raise ValueError(
                    f"query_anchors[{position}] {name} must be non-negative, got "
                    f"{int(value)}"
                )
        result.append((int(camera), int(time_index)))
    if not result:
        raise ValueError("query_anchors must contain at least one (camera, time) pair")
    if len(set(result)) != len(result):
        raise ValueError(f"query_anchors must not contain duplicates: {result}")
    return tuple(result)


def _resolve_view_indices(
    camera_ids: Sequence[int],
    view_ids: np.ndarray,
) -> tuple[int, ...]:
    """Map original camera ids onto positions along the dumped ``V`` axis.

    Every dumped array is indexed by view position, not by camera id.  The two
    coincide whenever the dump was taken with an ascending, complete view list,
    which is every dump so far -- but resolving through the recorded ``view_ids``
    makes ``--cameras`` and ``--query_anchor`` rest on a recorded fact instead of
    on that convention holding forever.
    """

    lookup: dict[int, int] = {}
    for position, value in enumerate(view_ids.tolist()):
        value = int(value)
        if value in lookup:
            raise ValueError(
                f"view_ids contains duplicate camera id {value} at positions "
                f"{lookup[value]} and {position}"
            )
        lookup[value] = position
    resolved = []
    for camera_id in camera_ids:
        if int(camera_id) not in lookup:
            raise ValueError(
                f"Camera {int(camera_id)} is not among the dumped cameras "
                f"{sorted(lookup)}"
            )
        resolved.append(lookup[int(camera_id)])
    return tuple(resolved)


def _load_view_ids(
    meta: np.lib.npyio.NpzFile,
    *,
    view_count: int,
) -> np.ndarray:
    """``view_ids`` when the dump records it, else the identity it replaced.

    Dumps predating the field are indistinguishable from ones whose views are
    cameras 0..V-1, which is exactly what the identity default expresses.
    """

    if "view_ids" not in meta.files:
        return np.arange(view_count, dtype=np.int64)
    view_ids = np.asarray(meta["view_ids"]).reshape(-1).astype(np.int64)
    if view_ids.shape != (view_count,):
        raise ValueError(
            f"view_ids must have shape ({view_count},) matching the dumped "
            f"views, got {view_ids.shape}"
        )
    if np.any(view_ids < 0):
        raise ValueError(f"view_ids must be non-negative, got {view_ids.tolist()}")
    return view_ids


def _validate_depth_sidecar(
    depth: np.ndarray,
    *,
    sidecar_path: Path,
    view_count: int,
    time_count: int,
    depth0: np.ndarray,
) -> None:
    """Check the per-frame depth sidecar against the metadata it extends.

    The dtype is deliberately not checked.  It comes from Kubric's depth TIFFs
    through imageio and the dump passes it through untouched, so anything from
    float16 to float64 is legitimate; both sides are cast identically on load,
    which is what keeps the ``depth[:, 0] == depth0`` comparison below exact.
    """

    if depth.ndim != 5 or depth.shape[2] != 1:
        raise ValueError(
            f"{sidecar_path} '{DEPTH_SIDECAR_KEY}' must have shape (V,T,1,H,W), "
            f"got {depth.shape}"
        )
    if depth.shape[:2] != (view_count, time_count):
        raise ValueError(
            f"{sidecar_path} '{DEPTH_SIDECAR_KEY}' covers "
            f"{depth.shape[0]} views x {depth.shape[1]} frames, but meta.npz "
            f"describes {view_count} x {time_count}"
        )
    if depth.shape[-2:] != depth0.shape[-2:]:
        raise ValueError(
            f"{sidecar_path} depth grid {depth.shape[-2:]} does not match "
            f"meta.npz depth0 grid {depth0.shape[-2:]}"
        )
    # depth0 is assigned depth[:, 0] on the producing side, so this is an
    # invariant rather than a tolerance question; a mismatch means the two files
    # came from different runs and nothing downstream would be trustworthy.
    if not np.array_equal(depth[:, 0], depth0):
        raise ValueError(
            f"{sidecar_path} depth[:, 0] differs from meta.npz depth0; the "
            "sidecar and the metadata are from different dumps"
        )


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
    query_anchors: Sequence[tuple[int, int]] | None = None,
    size: int = 512,
    patch_size: int = 14,
    square_ok: bool = False,
    verbose: bool = False,
) -> DumpedKubricScene:
    """Load one camera-major window from the existing evaluation dump.

    ``cameras`` and the camera half of ``query_anchors`` are original camera
    ids, resolved against the dump's ``view_ids``; every array is indexed by the
    resolved view position.  ``query_anchors`` names the observations that own a
    dense query field, in priority order -- the first is the primary anchor and
    owns the scene Sim(3).  It defaults to the first selected camera and time.

    Anchoring at an original time other than 0 needs the per-frame depth
    sidecar; :meth:`DumpedKubricScene.surface_depth_map` is where that is
    enforced, so parsing and ordinary Arc forwarding stay unrestricted.
    """

    cameras = _as_unique_int_tuple("cameras", cameras)
    times = _as_unique_int_tuple("times", times)
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(
            f"times must be strictly increasing to define temporal order, got {times}"
        )
    if query_anchors is None:
        query_anchors = ((cameras[0], times[0]),)
    query_anchors = _as_query_anchor_pairs(query_anchors)
    for position, (anchor_camera, anchor_time) in enumerate(query_anchors):
        if anchor_camera not in cameras:
            raise ValueError(
                f"query_anchors[{position}] camera {anchor_camera} is not in "
                f"selected cameras {cameras}"
            )
        if anchor_time not in times:
            raise ValueError(
                f"query_anchors[{position}] time {anchor_time} is not in "
                f"selected times {times}"
            )

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
        view_ids = _load_view_ids(meta, view_count=visibility.shape[0])
        track_upscaling_factor = float(
            np.asarray(meta["track_upscaling_factor"]).reshape(()).item()
        )

    view_count, time_count = visibility.shape[:2]
    camera_ids = cameras
    cameras = _resolve_view_indices(camera_ids, view_ids)
    anchor_view_pairs = tuple(
        (_resolve_view_indices((anchor_camera,), view_ids)[0], anchor_time)
        for anchor_camera, anchor_time in query_anchors
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

    # Optional per-frame depth. Absent is the normal case for a dump taken
    # without RCMV_DUMP_DEPTH=1, and leaves every path below at depth0 only.
    sidecar_path = scene_path / DEPTH_SIDECAR_NAME
    depth = None
    depth_sidecar_path = None
    if sidecar_path.exists():
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            if DEPTH_SIDECAR_KEY not in sidecar.files:
                raise ValueError(
                    f"{sidecar_path} is missing the '{DEPTH_SIDECAR_KEY}' array; "
                    f"found {sorted(sidecar.files)}"
                )
            # Cast before comparing: the sidecar's dtype is whatever Kubric's
            # depth TIFFs carried, and depth0 was already cast the same way.
            depth = np.array(
                sidecar[DEPTH_SIDECAR_KEY],
                dtype=np.float32,
                copy=True,
            )
        _validate_depth_sidecar(
            depth,
            sidecar_path=sidecar_path,
            view_count=view_count,
            time_count=time_count,
            depth0=depth0,
        )
        depth_sidecar_path = sidecar_path

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
                camera_id=int(view_ids[camera]),
                original_time=original_time,
                semantic_time_index=semantic_time_index,
                path=Path(path),
                image_transform=transform,
            )
        )
        slot_cameras.append(camera)
        slot_times.append(original_time)
        slot_time_indices.append(semantic_time_index)

    # One dense query field per anchor, in the caller's priority order, while
    # every selected observation stays in S. Anchor 0 is primary: it owns the
    # scene Sim(3) and the reconstruction drift report.
    anchor_slots = [
        next(
            observation.slot
            for observation in observations
            if observation.camera == anchor_camera
            and observation.original_time == anchor_time
        )
        for anchor_camera, anchor_time in anchor_view_pairs
    ]
    query_observation_slot = anchor_slots[0]
    track_query_observation_slots = torch.tensor(anchor_slots, dtype=torch.long)
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
        camera_ids=camera_ids,
        view_ids=torch.from_numpy(view_ids),
        times=times,
        slot_cameras=torch.tensor(slot_cameras, dtype=torch.long),
        slot_times=torch.tensor(slot_times, dtype=torch.long),
        slot_time_indices=torch.tensor(slot_time_indices, dtype=torch.long),
        query_anchors=query_anchors,
        query_observation_slot=query_observation_slot,
        track_query_observation_slots=track_query_observation_slots,
        query_points=torch.from_numpy(query_points),
        trajectories_world=torch.from_numpy(trajectories),
        visibility=torch.from_numpy(visibility),
        intrinsics=torch.from_numpy(intrinsics),
        extrinsics_world_to_camera=torch.from_numpy(extrinsics),
        depth0=torch.from_numpy(depth0),
        depth=None if depth is None else torch.from_numpy(depth),
        depth_sidecar_path=depth_sidecar_path,
        track_upscaling_factor=track_upscaling_factor,
    )
