"""The camera-major scene a training step consumes, and the one builder for it.

The scene is intentionally camera-major.  It keeps every selected camera/time
pair as a separate 4RC observation while assigning equal ``time_index`` values
to synchronized observations.

:func:`build_scene` owns every derivation that turns arrays plus frames into a
scene; :func:`scene_from_datapoint` is its front-end for a live MVTracker
``Datapoint``.  A live sample always carries per-frame depth, so an anchor may
sit at any original time.  (The ``dumped_kubric`` and ``DumpedKubricScene``
names predate the live path; nothing here reads a dump.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# The optional geometry-input view keys --depth_input / --camera_input attach.
# Spelled to match Arc.DEPTH_KEY / Arc.CAMERA_VECTOR_KEY; duplicated rather
# than imported so this loader stays import-light, and pinned against drift by
# a test.
DEPTH_INPUT_KEY = "depth_map"
CAMERA_VECTOR_KEY = "camera_vector"


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
    scene array and the ``view_<camera>`` in the frame's label.  ``camera_id``
    is the original camera this view was rendered from, recorded by
    ``view_ids``.  The two coincide whenever the view list is ascending and
    complete, but only ``camera`` may index an array.

    ``path`` is a *diagnostic label, not a locator*: a live frame is labelled
    ``<scene>/<live>/view_<v>/<t>.png`` and names no file.  Nothing stats or
    reopens it.
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
    depth: torch.Tensor
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
    def anchor_observation_slots(self) -> tuple[int, ...]:
        return tuple(
            int(slot) for slot in self.track_query_observation_slots.tolist()
        )

    def surface_depth_map(self, camera: int, original_time: int) -> torch.Tensor:
        """The ``(H,W)`` camera-z depth map anchoring queries in one observation.

        The Sim(3) target pointmap, the sparse anchor projection and the depth
        input all read depth through here, so they read the same map for the
        same observation.
        """

        camera = int(camera)
        original_time = int(original_time)
        if not 0 <= camera < self.depth0.shape[0]:
            raise ValueError(
                f"Camera view index {camera} is out of range for "
                f"{self.depth0.shape[0]} dumped views"
            )
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
    """Map original camera ids onto positions along the scene's ``V`` axis.

    Every scene array is indexed by view position, not by camera id.  The two
    coincide whenever the view list is ascending and complete, but a live
    sample's ``sample_views`` need be neither, so cameras and anchors are
    resolved through the recorded ``view_ids`` rather than assumed to be
    positions.
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
    """The ``view_<v>/<t>.png`` part of a frame's label.

    ``camera`` is the resolved view index -- the position along the scene's
    ``V`` axis -- not the original camera id.
    """

    return f"view_{camera}/{time_index:04d}.png"


def _validate_depth(
    depth: np.ndarray,
    *,
    source: str,
    view_count: int,
    time_count: int,
    depth0: np.ndarray,
) -> None:
    """Check per-frame depth against the window's arrays and ``depth0``.

    The dtype is deliberately not checked: ``build_scene`` casts both depth
    arrays to float32 the same way before this runs, which is what keeps the
    ``depth[:, 0] == depth0`` comparison below exact.
    """

    if depth.ndim != 5 or depth.shape[2] != 1:
        raise ValueError(
            f"{source} depth must have shape (V,T,1,H,W), got {depth.shape}"
        )
    if depth.shape[:2] != (view_count, time_count):
        raise ValueError(
            f"{source} depth covers {depth.shape[0]} views x {depth.shape[1]} "
            f"frames, but the scene arrays describe {view_count} x {time_count}"
        )
    if depth.shape[-2:] != depth0.shape[-2:]:
        raise ValueError(
            f"{source} depth grid {depth.shape[-2:]} does not match the depth0 "
            f"grid {depth0.shape[-2:]}"
        )
    # depth0 is depth[:, 0] by construction (scene_from_datapoint slices it), so
    # this is an invariant rather than a tolerance question; a mismatch means
    # the two arrays describe different frames.
    if not np.array_equal(depth[:, 0], depth0):
        raise ValueError(
            f"{source} depth[:, 0] differs from depth0; the two arrays describe "
            "different frames"
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
    """The shape contract every scene's arrays are held to.

    A live sample that has been transposed or has picked up a batch axis fails
    here rather than three frames later inside the slot arithmetic, where the
    message would name a grid mismatch instead of the actual fault.
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


def _attach_view_geometry(scene, *, input_depth_max, input_camera_vectors):
    """Attach the geometry-input view keys the trainer's flags request.

    Depth rides the SAME index gather every anchor pointmap takes
    (``output_to_original_indices``) -- a gather, never interpolation, so a
    depth discontinuity cannot be blended into a depth that exists nowhere.
    ``input_depth_max`` is an INVALIDITY threshold, not a normalisation
    ceiling: the live loader zeroes label depth beyond it unconditionally, so
    a beyond-max pixel reads invalid here too rather than saturating to
    1.0/valid. After invalidation every valid depth is in (0, max], so
    channel 0 maps to (0, 1] and the clip is a saturation guard at the
    boundary, and channel 1 is the validity. Metric scale is kept: no per-view normalisation.

    The camera vector is the fork's own 9-dim pose encoding of the
    camera-to-world pose (the scene stores world-to-camera; ``affine_inverse``
    inverts it) with model-grid intrinsics, plus the principal point the
    9-dim format discards -- it discards it only because ``cam_dec`` must
    predict INTO the format, and a projection has no such obligation. The
    model-grid intrinsics are derived exclusively through
    ``ImageTransform.original_to_output`` so no second spelling of the
    scale/crop affine exists; a test pins the construction against the direct
    fields formula.
    """

    if input_depth_max is not None:
        if not np.isfinite(input_depth_max) or input_depth_max <= 0:
            raise ValueError(
                f"input_depth_max must be finite and positive, got {input_depth_max}"
            )
        for observation, view in zip(scene.observations, scene.views):
            depth = (
                scene.surface_depth_map(observation.camera, observation.original_time)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            rows, columns = observation.image_transform.output_to_original_indices()
            columns_grid, rows_grid = np.meshgrid(columns, rows)
            sampled = depth[rows_grid, columns_grid]
            invalid = (
                ~np.isfinite(sampled)
                | (sampled <= 1e-6)
                | (sampled > input_depth_max)
            )
            ch0 = np.where(
                invalid, 0.0, np.clip(sampled, 0.0, input_depth_max) / input_depth_max
            )
            view[DEPTH_INPUT_KEY] = torch.from_numpy(
                np.stack([ch0, (~invalid).astype(np.float64)]).astype(np.float32)
            )[None]

    if input_camera_vectors:
        from arc.models.arc.utils.transform import (
            affine_inverse,
            extri_intri_to_pose_encoding,
        )

        shared_transform = scene.observations[0].image_transform
        output_height = shared_transform.output_height
        output_width = shared_transform.output_width
        model_intrinsics = []
        world_to_camera_rows = []
        principal_points = []
        for observation in scene.observations:
            transform = observation.image_transform
            intrinsics = (
                scene.intrinsics[observation.camera, observation.original_time]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            principal = transform.original_to_output(
                np.array([[intrinsics[0, 2], intrinsics[1, 2]]])
            )[0]
            fx_out = (
                transform.original_to_output(
                    np.array(
                        [[intrinsics[0, 2] + intrinsics[0, 0], intrinsics[1, 2]]]
                    )
                )[0][0]
                - principal[0]
            )
            fy_out = (
                transform.original_to_output(
                    np.array(
                        [[intrinsics[0, 2], intrinsics[1, 2] + intrinsics[1, 1]]]
                    )
                )[0][1]
                - principal[1]
            )
            model_intrinsics.append(
                np.array(
                    [
                        [fx_out, 0.0, principal[0]],
                        [0.0, fy_out, principal[1]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            )
            world_to_camera_rows.append(
                scene.extrinsics_world_to_camera[
                    observation.camera, observation.original_time
                ].double()
            )
            principal_points.append(
                (principal[0] / output_width, principal[1] / output_height)
            )
        camera_to_world = affine_inverse(torch.stack(world_to_camera_rows)[None])
        pose9 = extri_intri_to_pose_encoding(
            camera_to_world,
            torch.from_numpy(np.stack(model_intrinsics))[None],
            (output_height, output_width),
        )
        # Terms 0-8 are the pose encoding's own layout: translation, the xyzw
        # quaternion (real part last), then the (h, w)-ORDERED fovs. Terms
        # 9-10 switch convention to the (x, y)-ordered principal point,
        # normalised by output width and height -- deliberate; both halves
        # are asserted index by index in the tests.
        principal_terms = torch.tensor(principal_points, dtype=torch.float64)[None]
        vectors = torch.cat([pose9.double(), principal_terms], dim=-1).float()
        for slot, view in enumerate(scene.views):
            view[CAMERA_VECTOR_KEY] = vectors[:, slot]


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
    view_ids,
    depth,
    cameras: Sequence[int] = (0, 1),
    times: Sequence[int] = (0, 1, 2, 3),
    query_anchors: Sequence[tuple[int, int]] | None = None,
    size: int = 512,
    patch_size: int = 14,
    square_ok: bool = False,
    input_depth_max: float | None = None,
    input_camera_vectors: bool = False,
    verbose: bool = False,
    source: str = "<arrays>",
) -> DumpedKubricScene:
    """Assemble one camera-major window from arrays and a frame source.

    Everything that makes a :class:`DumpedKubricScene` a *scene* rather than a
    pile of arrays lives here: camera-id resolution, the camera-major slot
    arithmetic, the anchor slots, and the two cross-checks that catch a frame
    source which reordered or dropped images.  None of it depends on where the
    pixels came from, so every frame source gets the same window from the same
    arrays.

    ``open_frames(view_positions, times)`` yields ``(camera, time, label, PIL
    image)`` in camera-major order.  It is a callable rather than an iterable
    because the resolution from original camera ids to view positions happens
    here, and the frame source needs the resolved positions.  ``label`` is used
    for error messages and for ``Observation.path``; it need not name a real
    file.

    Arrays may be numpy or torch, and are copied.  ``depth`` is per-frame,
    ``(V,T,1,H,W)``, and ``depth0`` is its first frame.
    """

    query_points = _as_numpy(query_points, np.float32)
    trajectories = _as_numpy(trajectories, np.float32)
    visibility = _as_numpy(visibility, bool)
    intrinsics = _as_numpy(intrinsics, np.float32)
    extrinsics = _as_numpy(extrinsics, np.float32)
    depth0 = _as_numpy(depth0, np.float32)
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
    _validate_depth(
        depth,
        source=source,
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
            "A window requires all observations to have the same processed "
            f"shape, got {sorted(output_shapes)}"
        )

    scene = DumpedKubricScene(
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
        depth=torch.from_numpy(depth),
        track_upscaling_factor=track_upscaling_factor,
    )
    # Attached post-construction so the helper reuses surface_depth_map and
    # scene.observations instead of re-deriving the depth lookup and the slot
    # arithmetic.
    if input_depth_max is not None or input_camera_vectors:
        _attach_view_geometry(
            scene,
            input_depth_max=input_depth_max,
            input_camera_vectors=input_camera_vectors,
        )
    return scene


def scene_from_datapoint(
    sample,
    *,
    cameras: Sequence[int] | None = None,
    times: Sequence[int] | None = None,
    query_anchors: Sequence[tuple[int, int]] | None = None,
    size: int = 512,
    patch_size: int = 14,
    square_ok: bool = False,
    input_depth_max: float | None = None,
    input_camera_vectors: bool = False,
    verbose: bool = False,
) -> DumpedKubricScene:
    """Build a scene from a live MVTracker ``Datapoint``.

    A thin front-end over :func:`build_scene`: frames come out of ``video``,
    camera ids out of ``sample_views``, and ``depth0`` is ``videodepth``'s first
    frame, so a window built here is the one its arrays make directly -- which
    is asserted, not assumed, by ``tests/test_scene_sources.py``.

    A live sample always carries per-frame depth (``videodepth``), so an anchor
    may sit at any time.  ``cameras`` are original camera ids and default to the
    sample's own ``sample_views``; ``times`` default to every frame the sample
    holds.
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
                # A label for error messages and Observation.path, not a file.
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
        input_depth_max=input_depth_max,
        input_camera_vectors=input_camera_vectors,
        verbose=verbose,
        source=f"<live sample {sample.seq_name}>",
    )


