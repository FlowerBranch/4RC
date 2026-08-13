"""Adapter for one scene produced by the cluster repository's dump scripts.

The adapter is intentionally camera-major.  It keeps every selected camera/time
pair as a separate 4RC observation while assigning equal ``time_index`` values
to synchronized observations.

Two files are read.  ``meta.npz`` is required and carries the sparse metric
metadata.  ``depth_full.npz`` is an optional sibling holding per-frame depth; it
is emitted only when the dump ran with ``RCMV_DUMP_DEPTH=1``, so a dump made
without it is complete-looking but supports anchors at original time 0 only.

Frames arrive in one of two layouts and the adapter reads either.  The training
dump packs them into a single ``frames.zip``; the benchmark dump leaves them
loose under ``view_<v>/``.  Both are permanently live, and which one applies is
decided by what is on disk -- there is no flag and no config key.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# The optional per-frame depth sidecar. Its ``t`` axis is 1:1 with the frame
# names, and ``depth[:, 0]`` is what ``meta.npz`` stores as ``depth0``.
DEPTH_SIDECAR_NAME = "depth_full.npz"
DEPTH_SIDECAR_KEY = "depth"
DEPTH_SIDECAR_FLAG = "RCMV_DUMP_DEPTH=1"

# The packed frame layout. Present for a dump whose frames were packed, absent
# for one that left them loose; nothing else distinguishes the two.
FRAMES_ARCHIVE_NAME = "frames.zip"


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

    ``path`` is a *diagnostic label, not a locator*, and is not guaranteed to
    exist on disk: a frame read out of ``frames.zip`` is labelled
    ``<scene>/frames.zip/view_<v>/<t>.png``, naming the archive and the member
    it came from.  Nothing stats or reopens it today.  A consumer that needs to
    is the moment to give this field an explicit archive spelling and change its
    type -- with that caller in hand, not before.
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


def _frame_member_name(camera: int, time_index: int) -> str:
    """Where one frame lives, in the single spelling both layouts share.

    Zip member names are ``/``-separated by specification, so this is built as a
    string; the loose layout then derives its path from it, which is what keeps
    the packed and loose names from ever drifting apart.  ``camera`` is the
    resolved view index -- the position along the dumped ``V`` axis -- exactly as
    it is for the loose directory name.
    """

    return f"view_{camera}/{time_index:04d}.png"


def _iter_frames(
    scene_path: Path,
    cameras: Sequence[int],
    times: Sequence[int],
):
    """Yield ``(camera, time, path, image)`` for the selected grid, camera-major.

    The camera and time travel with the frame so ``build_scene`` can check each
    one against the slot it is filling.  They are redundant for this source,
    which generates the grid itself -- but the core accepts any frame source, and
    a source that yields in the wrong order would otherwise produce a scene whose
    every slot holds the right metadata and the wrong picture.

    Frames are packed into ``frames.zip`` or loose under ``view_<v>/``, and which
    it is depends on nothing but what is on disk.  Packing exists because the
    cluster quota caps file *count*: a ten-view scene costs 241 loose files
    against three packed.  The benchmark dump stays loose, so both layouts are
    live simultaneously and the archive wins wherever it is present.

    Each image is decoded here rather than handed on unread, so its source handle
    is released before the next frame opens instead of one being held open per
    observation.
    """

    archive_path = scene_path / FRAMES_ARCHIVE_NAME
    if archive_path.is_file():
        try:
            archive = zipfile.ZipFile(archive_path)
        except zipfile.BadZipFile as error:
            raise ValueError(
                f"{archive_path} is not a readable zip archive ({error}); the "
                "packed frames are truncated or were not transferred completely"
            ) from error
        # One handle for the whole window. Opening per member would re-read the
        # archive's central directory once per observation.
        with archive:
            for camera in cameras:
                for time_index in times:
                    member = _frame_member_name(camera, time_index)
                    try:
                        payload = archive.read(member)
                    except KeyError:
                        raise FileNotFoundError(
                            f"Observation image not found: member '{member}' is "
                            f"missing from {archive_path}"
                        ) from None
                    image = Image.open(io.BytesIO(payload))
                    image.load()
                    yield camera, time_index, archive_path / member, image
    else:
        for camera in cameras:
            for time_index in times:
                path = scene_path / _frame_member_name(camera, time_index)
                if not path.is_file():
                    raise FileNotFoundError(f"Observation image not found: {path}")
                image = Image.open(path)
                image.load()
                yield camera, time_index, path, image


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
    sidecar_path: Path | str,
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

    _validate_scene_arrays(
        query_points=meta["query_points"],
        trajectories=meta["traj3d_world"],
        visibility=meta["visibility"],
        intrinsics=meta["intrs"],
        extrinsics=meta["extrs"],
        depth0=meta["depth0"],
    )


def _validate_scene_arrays(
    *,
    query_points,
    trajectories,
    visibility,
    intrinsics,
    extrinsics,
    depth0,
) -> None:
    """The shape contract, checked the same way whatever the arrays came from.

    Split out of ``_validate_meta`` so a scene assembled in memory is held to the
    identical contract as one read off disk.  A live sample that has been
    transposed or has picked up a batch axis fails here rather than three frames
    later inside the slot arithmetic, where the message would name a grid
    mismatch instead of the actual fault.
    """

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


def _as_numpy(value, dtype):
    """Accept a torch tensor or anything array-like, always return a fresh copy."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def build_scene(
    *,
    name: str,
    open_frames,
    query_points,
    trajectories,
    visibility,
    intrinsics,
    extrinsics,
    depth0,
    track_upscaling_factor: float,
    view_ids=None,
    depth=None,
    depth_sidecar_path: Path | None = None,
    cameras: Sequence[int] = (0, 1),
    times: Sequence[int] = (0, 1, 2, 3),
    query_anchors: Sequence[tuple[int, int]] | None = None,
    size: int = 512,
    patch_size: int = 14,
    square_ok: bool = False,
    verbose: bool = False,
    source: str = "<arrays>",
) -> DumpedKubricScene:
    """Assemble one camera-major window from arrays and a frame source.

    Everything that makes a :class:`DumpedKubricScene` a *scene* rather than a
    pile of arrays lives here: camera-id resolution, the camera-major slot
    arithmetic, the anchor slots, and the two cross-checks that catch a frame
    source which reordered or dropped images.  None of it depends on where the
    pixels came from, which is the whole point -- a dump on disk and a live
    MVTracker sample must not be able to disagree about what a window *is*.

    ``open_frames(view_positions, times)`` yields ``(label, PIL image)`` in
    camera-major order.  It is a callable rather than an iterable because the
    resolution from original camera ids to view positions happens here, and the
    frame source needs the resolved positions.  ``label`` is used for error
    messages and for ``Observation.path``; it need not name a real file.

    Arrays may be numpy or torch, and are copied.  ``view_ids`` defaults to the
    identity, which is what a dump predating the field means.
    """

    query_points = _as_numpy(query_points, np.float32)
    trajectories = _as_numpy(trajectories, np.float32)
    visibility = _as_numpy(visibility, bool)
    intrinsics = _as_numpy(intrinsics, np.float32)
    extrinsics = _as_numpy(extrinsics, np.float32)
    depth0 = _as_numpy(depth0, np.float32)
    if depth is not None:
        depth = _as_numpy(depth, np.float32)
    _validate_scene_arrays(
        query_points=query_points,
        trajectories=trajectories,
        visibility=visibility,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth0=depth0,
    )

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

    view_count, time_count = visibility.shape[:2]
    if view_ids is None:
        view_ids = np.arange(view_count, dtype=np.int64)
    else:
        view_ids = _as_numpy(view_ids, np.int64).reshape(-1)
        if view_ids.shape != (view_count,):
            raise ValueError(
                f"view_ids must have shape ({view_count},) matching the dumped "
                f"views, got {view_ids.shape}"
            )
        if np.any(view_ids < 0):
            raise ValueError(f"view_ids must be non-negative, got {view_ids.tolist()}")

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
    if depth is not None:
        _validate_depth_sidecar(
            depth,
            sidecar_path=depth_sidecar_path if depth_sidecar_path is not None else source,
            view_count=view_count,
            time_count=time_count,
            depth0=depth0,
        )

    # Each frame is read and decoded exactly once here: its size feeds the
    # transform, and the same decoded image is then handed to the preprocessor.
    paths = []
    images = []
    transforms = []
    for slot, (camera, time_index, path, image) in enumerate(
        open_frames(cameras, times)
    ):
        # The frame source declares which observation each image is, and it is
        # checked against the slot arithmetic rather than trusted. The loader-index
        # check further down cannot stand in for this: `preprocess_images` numbers
        # its outputs by receipt order, so a source that yields the grid in the
        # wrong order satisfies it while every slot holds the wrong picture.
        if slot >= len(cameras) * len(times):
            raise RuntimeError(
                f"Frame source yielded more than {len(cameras) * len(times)} "
                "frames for the selected camera/time grid"
            )
        expected_camera = cameras[slot // len(times)]
        expected_time = times[slot % len(times)]
        if (int(camera), int(time_index)) != (expected_camera, expected_time):
            raise RuntimeError(
                f"Frame source is out of step at slot {slot}: got camera "
                f"{int(camera)} time {int(time_index)} for {path}, expected "
                f"camera {expected_camera} time {expected_time}. Frames must "
                "arrive camera-major over the selected grid."
            )
        width, height = image.size
        if depth0.shape[-2:] != (height, width):
            raise ValueError(
                f"depth0 grid {depth0.shape[-2:]} does not match dumped frame "
                f"grid {(height, width)} for {path}"
            )
        paths.append(str(path))
        images.append(image)
        transforms.append(
            compute_image_transform(
                height,
                width,
                size=size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
        )

    expected_frames = len(cameras) * len(times)
    if len(paths) != expected_frames:
        raise RuntimeError(
            f"Frame source yielded {len(paths)} frames for a {len(cameras)}x"
            f"{len(times)} camera/time grid ({expected_frames} expected)"
        )

    # Import lazily so metadata/geometry tests do not require torchvision.
    from arc.dust3r.utils.image import preprocess_images

    views = preprocess_images(
        zip(paths, images),
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
        name=name,
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


def scene_from_datapoint(
    sample,
    *,
    cameras: Sequence[int] | None = None,
    times: Sequence[int] | None = None,
    query_anchors: Sequence[tuple[int, int]] | None = None,
    size: int = 512,
    patch_size: int = 14,
    square_ok: bool = False,
    verbose: bool = False,
) -> DumpedKubricScene:
    """Build a scene from a live MVTracker ``Datapoint``, with no dump on disk.

    The same :func:`build_scene` core as the dump reader, so a window assembled
    here and the same window read back from a dump are the same object -- which
    is asserted, not assumed, by the live-vs-dump equivalence test.

    A live sample always carries per-frame depth (``videodepth``), so
    :attr:`DumpedKubricScene.has_time_varying_depth` is true and an anchor at a
    time other than 0 needs no sidecar.  ``cameras`` are original camera ids and
    default to the sample's own ``sample_views``; ``times`` default to every
    frame the sample holds.
    """

    video = sample.video
    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError(
            f"Datapoint.video must have shape (V,T,3,H,W), got {tuple(video.shape)}"
        )
    view_count, time_count = int(video.shape[0]), int(video.shape[1])
    view_ids = getattr(sample, "sample_views", None)
    if view_ids is None:
        view_ids = list(range(view_count))
    view_ids = [int(value) for value in view_ids]
    if len(view_ids) != view_count:
        raise ValueError(
            f"sample_views has {len(view_ids)} entries for {view_count} video views"
        )
    if cameras is None:
        cameras = tuple(view_ids)
    if times is None:
        times = tuple(range(time_count))

    depth = sample.videodepth
    if depth is None:
        raise ValueError(
            "Datapoint.videodepth is None; a live scene needs per-frame depth to "
            "anchor queries at all"
        )

    def open_frames(view_positions, selected_times):
        for view_position in view_positions:
            for time_index in selected_times:
                frame = video[view_position, time_index]
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu()
                array = np.asarray(frame).transpose(1, 2, 0).astype(np.uint8)
                # Labelled the way a packed dump labels its members, so an error
                # message reads the same whichever source produced the frame.
                yield (
                    view_position,
                    time_index,
                    f"{sample.seq_name}/<live>/{_frame_member_name(view_position, time_index)}",
                    Image.fromarray(array),
                )

    return build_scene(
        name=str(sample.seq_name),
        open_frames=open_frames,
        query_points=sample.query_points_3d,
        trajectories=sample.trajectory_3d,
        visibility=sample.visibility,
        intrinsics=sample.intrs,
        extrinsics=sample.extrs,
        depth0=depth[:, 0],
        depth=depth,
        view_ids=view_ids,
        track_upscaling_factor=float(sample.track_upscaling_factor),
        cameras=cameras,
        times=times,
        query_anchors=query_anchors,
        size=size,
        patch_size=patch_size,
        square_ok=square_ok,
        verbose=verbose,
        source=f"<live sample {sample.seq_name}>",
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

    The disk front-end of :func:`build_scene`: it owns ``meta.npz``, the optional
    depth sidecar and the frames-or-archive branch, and nothing else.  Every
    derivation the scene depends on lives in the core, so this and
    :func:`scene_from_datapoint` cannot drift apart.

    ``cameras`` and the camera half of ``query_anchors`` are original camera
    ids, resolved against the dump's ``view_ids``; every array is indexed by the
    resolved view position.  ``query_anchors`` names the observations that own a
    dense query field, in priority order -- the first is the primary anchor and
    owns the scene Sim(3).  It defaults to the first selected camera and time.

    Anchoring at an original time other than 0 needs the per-frame depth
    sidecar; :meth:`DumpedKubricScene.surface_depth_map` is where that is
    enforced, so parsing and ordinary Arc forwarding stay unrestricted.
    """

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
        depth_sidecar_path = sidecar_path

    return build_scene(
        name=scene_name,
        open_frames=lambda view_positions, selected_times: _iter_frames(
            scene_path,
            view_positions,
            selected_times,
        ),
        query_points=query_points,
        trajectories=trajectories,
        visibility=visibility,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth0=depth0,
        depth=depth,
        depth_sidecar_path=depth_sidecar_path,
        view_ids=view_ids,
        track_upscaling_factor=track_upscaling_factor,
        cameras=cameras,
        times=times,
        query_anchors=query_anchors,
        size=size,
        patch_size=patch_size,
        square_ok=square_ok,
        verbose=verbose,
        source=str(scene_path),
    )
