"""Small, explicit training utilities for bounded 4RC experiments."""

from .checkpoint import (
    load_temporal_tracking_checkpoint,
    save_temporal_tracking_checkpoint,
)
from .dumped_kubric import (
    DumpedKubricScene,
    ImageTransform,
    Observation,
    load_dumped_kubric_scene,
)
from .sparse_tracking import (
    DetachedSim3,
    SparseCorrespondences,
    SparseTrackingLossResult,
    build_anchor_correspondences,
    fit_scene_sim3,
    gather_query_anchor_points,
    sparse_tracking_loss,
)

__all__ = [
    "DetachedSim3",
    "DumpedKubricScene",
    "ImageTransform",
    "Observation",
    "SparseCorrespondences",
    "SparseTrackingLossResult",
    "build_anchor_correspondences",
    "fit_scene_sim3",
    "gather_query_anchor_points",
    "load_temporal_tracking_checkpoint",
    "load_dumped_kubric_scene",
    "save_temporal_tracking_checkpoint",
    "sparse_tracking_loss",
]
