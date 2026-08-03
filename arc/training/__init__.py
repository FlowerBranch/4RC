"""Small, explicit training utilities for bounded 4RC experiments."""

from .checkpoint import (
    load_temporal_tracking_checkpoint,
    save_temporal_tracking_checkpoint,
)
from .diagnostics import (
    DEFAULT_CONFIDENCE_TAUS,
    confidence_occlusion_diagnostics,
)
from .dumped_kubric import (
    DumpedKubricScene,
    ImageTransform,
    Observation,
    load_dumped_kubric_scene,
)
from .losses import (
    compose_tracking_loss,
    per_sample_huber_error,
    resolve_confidence_alpha,
    track_confidence_loss,
    track_metric_error,
    track_position_loss,
)
from .sparse_tracking import (
    DetachedSim3,
    SparseCorrespondences,
    SparseTrackingLossResult,
    build_anchor_correspondences,
    fit_scene_sim3,
    gather_at_correspondences,
    gather_query_anchor_points,
    sparse_tracking_loss,
)

__all__ = [
    "DEFAULT_CONFIDENCE_TAUS",
    "DetachedSim3",
    "DumpedKubricScene",
    "ImageTransform",
    "Observation",
    "SparseCorrespondences",
    "SparseTrackingLossResult",
    "build_anchor_correspondences",
    "compose_tracking_loss",
    "confidence_occlusion_diagnostics",
    "fit_scene_sim3",
    "gather_at_correspondences",
    "gather_query_anchor_points",
    "load_temporal_tracking_checkpoint",
    "load_dumped_kubric_scene",
    "per_sample_huber_error",
    "resolve_confidence_alpha",
    "save_temporal_tracking_checkpoint",
    "sparse_tracking_loss",
    "track_confidence_loss",
    "track_metric_error",
    "track_position_loss",
]
