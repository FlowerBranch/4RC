"""Small, explicit training utilities for bounded 4RC experiments."""

from .checkpoint import (
    load_temporal_tracking_checkpoint,
    read_temporal_patch_metadata,
    save_temporal_tracking_checkpoint,
)
from .diagnostics import (
    DEFAULT_CONFIDENCE_TAUS,
    confidence_occlusion_diagnostics,
    reconstruction_shift_report,
    synchronized_consistency_stats,
    temporal_injection_report,
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
    synchronized_consistency_loss,
    synchronized_pair_indices,
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
    reconstruction_drift_report,
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
    "read_temporal_patch_metadata",
    "reconstruction_drift_report",
    "reconstruction_shift_report",
    "per_sample_huber_error",
    "resolve_confidence_alpha",
    "save_temporal_tracking_checkpoint",
    "sparse_tracking_loss",
    "synchronized_consistency_loss",
    "synchronized_consistency_stats",
    "synchronized_pair_indices",
    "temporal_injection_report",
    "track_confidence_loss",
    "track_metric_error",
    "track_position_loss",
]
