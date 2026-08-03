from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from arc.training import (
    compose_tracking_loss,
    confidence_occlusion_diagnostics,
    per_sample_huber_error,
    resolve_confidence_alpha,
    track_confidence_loss,
    track_metric_error,
    track_position_loss,
)


# The whole point of the pure-tensor boundary is that these need no scene, no dump
# and no model -- just tensors of the right shape.
def _sample(count=4, slots=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    predicted = torch.randn(count, slots, 3, generator=generator)
    target = torch.randn(count, slots, 3, generator=generator)
    mask = torch.ones(count, slots, dtype=torch.bool)
    return predicted, target, mask


# ------------------------------------------------------------------ position ---
def test_position_loss_matches_the_prerefactor_expression():
    """The refactor must not move the number: archived runs are reproduced against it."""

    predicted, target, mask = _sample()
    mask[1, 2] = False

    expected = F.huber_loss(
        predicted[mask],
        target[mask],
        reduction="mean",
        delta=0.05,
    )

    assert torch.equal(
        track_position_loss(predicted, target, mask, huber_delta=0.05),
        expected,
    )


def test_position_loss_is_a_mean_over_coordinates_not_over_samples():
    """A mean of per-sample means is equal in exact arithmetic but not bit-for-bit.

    Pinning the distinction keeps anyone from "simplifying" the reduction into the
    per-sample path and silently changing an archived number.
    """

    predicted, target, mask = _sample(count=32, slots=8, seed=3)
    per_sample = per_sample_huber_error(predicted, target, huber_delta=0.05)

    loss = track_position_loss(predicted, target, mask, huber_delta=0.05)

    assert torch.allclose(loss, per_sample[mask].mean(), atol=1e-6)


def test_position_loss_ignores_masked_out_samples():
    predicted, target, mask = _sample()
    mask[0, 0] = False
    predicted[0, 0] += 1000.0

    loss = track_position_loss(predicted, target, mask, huber_delta=0.05)

    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_metric_error_is_the_mean_l2_residual():
    predicted = torch.zeros(2, 1, 3)
    target = torch.tensor([[[3.0, 4.0, 0.0]], [[0.0, 0.0, 1.0]]])
    mask = torch.ones(2, 1, dtype=torch.bool)

    assert track_metric_error(predicted, target, mask).item() == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(huber_delta=0.0), "huber_delta"),
        (dict(huber_delta=float("inf")), "huber_delta"),
    ],
)
def test_position_loss_rejects_a_nonpositive_delta(kwargs, message):
    predicted, target, mask = _sample()

    with pytest.raises(ValueError, match=message):
        track_position_loss(predicted, target, mask, **kwargs)


def test_position_loss_rejects_an_empty_mask():
    predicted, target, mask = _sample()

    with pytest.raises(ValueError, match="No samples selected"):
        track_position_loss(predicted, target, torch.zeros_like(mask), huber_delta=0.05)


# ---------------------------------------------------------------- confidence ---
def test_confidence_loss_is_minimized_at_alpha_over_error():
    """conf* = alpha / err is the whole reason this term calibrates confidence."""

    error = torch.full((1, 1), 0.25)
    mask = torch.ones(1, 1, dtype=torch.bool)
    alpha = 3.0
    optimum = alpha / error.item()

    at_optimum = track_confidence_loss(
        torch.full((1, 1), optimum),
        error,
        mask,
        alpha=alpha,
    )
    for offset in (0.5, 2.0):
        assert (
            track_confidence_loss(
                torch.full((1, 1), optimum * offset),
                error,
                mask,
                alpha=alpha,
            ).item()
            > at_optimum.item()
        )


def test_confidence_loss_detaches_the_error():
    """Without the detach the term would also reshape the position gradient."""

    # Deliberately away from the optimum conf*=alpha/err, where the gradient would
    # be zero for reasons that have nothing to do with the detach.
    confidence = torch.full((2, 2), 4.0, requires_grad=True)
    error = torch.full((2, 2), 0.5, requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)

    track_confidence_loss(confidence, error, mask, alpha=1.0).backward()

    assert confidence.grad is not None
    assert torch.count_nonzero(confidence.grad) > 0
    assert error.grad is None


def test_confidence_loss_rewards_low_confidence_where_the_error_is_large():
    mask = torch.ones(1, 2, dtype=torch.bool)
    error = torch.tensor([[0.1, 10.0]])
    confidence = torch.tensor([[20.0, 20.0]], requires_grad=True)

    # alpha=4 puts the optimum at 40 for the accurate sample and 0.4 for the
    # inaccurate one, so the shared starting confidence of 20 is between them.
    track_confidence_loss(confidence, error, mask, alpha=4.0).backward()

    # d/dconf = err - alpha/conf: positive gradient means "push this one down".
    low_error_gradient, high_error_gradient = confidence.grad[0].tolist()
    assert low_error_gradient < 0 < high_error_gradient


def test_confidence_loss_accepts_samples_the_position_mask_would_drop():
    """The confidence mask is deliberately a superset. Occluded points are the signal."""

    error = torch.tensor([[0.1, 5.0]])
    confidence = torch.tensor([[50.0, 50.0]])
    position_mask = torch.tensor([[True, False]])
    confidence_mask = torch.tensor([[True, True]])

    narrow = track_confidence_loss(confidence, error, position_mask, alpha=1.0)
    wide = track_confidence_loss(confidence, error, confidence_mask, alpha=1.0)

    assert not torch.isclose(narrow, wide)


def test_confidence_loss_rejects_nonpositive_confidence():
    """`expp1` cannot produce this, but log() must not be reached if something else does."""

    mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="strictly positive"):
        track_confidence_loss(
            torch.zeros(1, 1),
            torch.ones(1, 1),
            mask,
            alpha=1.0,
        )


def test_confidence_loss_rejects_a_nonpositive_alpha():
    mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="alpha"):
        track_confidence_loss(torch.full((1, 1), 2.0), torch.ones(1, 1), mask, alpha=0.0)


def test_confidence_loss_stays_finite_under_bfloat16_autocast():
    """The term runs inside the training autocast; log() must not be taken in bf16."""

    if not hasattr(torch, "autocast"):  # pragma: no cover - torch is always new enough
        pytest.skip("torch.autocast unavailable")
    confidence = torch.full((4, 4), 244.0)
    error = torch.full((4, 4), 1.36)
    mask = torch.ones(4, 4, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = track_confidence_loss(confidence, error, mask, alpha=330.0)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


# --------------------------------------------------------------------- alpha ---
def test_auto_alpha_puts_the_optimum_at_the_current_operating_point():
    """This is what stops the term from level-shifting a released calibration."""

    mean_confidence = 244.158
    mean_error = 1.360591

    alpha = resolve_confidence_alpha(mean_confidence, mean_error)

    assert alpha / mean_error == pytest.approx(mean_confidence)


def test_the_reference_alpha_of_one_would_collapse_confidence_below_the_expp1_floor():
    """Documents why `alpha=1` from eval/mv_recon/criterion.py is not the default."""

    mean_error = 1.360591

    assert 1.0 / mean_error < 1.0
    assert resolve_confidence_alpha(244.158, mean_error) / mean_error > 200.0


@pytest.mark.parametrize(
    "mean_confidence, mean_error",
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (float("nan"), 1.0), (1.0, float("inf"))],
)
def test_auto_alpha_rejects_degenerate_statistics(mean_confidence, mean_error):
    with pytest.raises(ValueError):
        resolve_confidence_alpha(mean_confidence, mean_error)


# ------------------------------------------------------------------- compose ---
def test_compose_returns_a_single_unit_weighted_term_unchanged():
    """This identity is what makes a confidence-disabled run bit-identical."""

    position = torch.tensor(0.25)

    total, breakdown = compose_tracking_loss(
        {"position": position, "confidence": torch.tensor(9.0)},
        {"position": 1.0, "confidence": 0.0},
    )

    assert total is position
    assert list(breakdown) == ["position"]
    assert breakdown["position"].item() == pytest.approx(0.25)


def test_compose_skips_zero_weight_terms_rather_than_multiplying_by_zero():
    """A multiplied-by-zero term still builds a graph; a skipped one cannot."""

    ablated = torch.tensor(3.0, requires_grad=True)
    kept = torch.tensor(2.0, requires_grad=True)

    total, breakdown = compose_tracking_loss(
        {"kept": kept, "ablated": ablated},
        {"kept": 1.0, "ablated": 0.0},
    )
    total.backward()

    assert "ablated" not in breakdown
    assert ablated.grad is None
    assert kept.grad is not None


def test_compose_weights_each_term():
    total, breakdown = compose_tracking_loss(
        {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
        {"a": 1.0, "b": 0.5},
    )

    assert total.item() == pytest.approx(2.0)
    # The breakdown reports unweighted values so runs at different weights compare.
    assert {k: v.item() for k, v in breakdown.items()} == {
        "a": pytest.approx(1.0),
        "b": pytest.approx(2.0),
    }


def test_compose_breakdown_holds_detached_tensors_not_floats():
    """This runs every training step; returning floats would force a device sync."""

    a = torch.tensor(1.0, requires_grad=True)
    _, breakdown = compose_tracking_loss({"a": a}, {"a": 1.0})

    assert isinstance(breakdown["a"], torch.Tensor)
    assert not breakdown["a"].requires_grad


def test_compose_rejects_weights_for_unknown_terms():
    with pytest.raises(KeyError, match="unknown loss terms"):
        compose_tracking_loss({"a": torch.tensor(1.0)}, {"a": 1.0, "typo": 1.0})


def test_compose_rejects_an_entirely_ablated_loss():
    with pytest.raises(ValueError, match="at least one term"):
        compose_tracking_loss({"a": torch.tensor(1.0)}, {"a": 0.0})


# --------------------------------------------------------------- diagnostics ---
def test_diagnostics_split_confidence_by_visibility():
    confidence = torch.tensor([[10.0, 200.0]])
    error = torch.tensor([[5.0, 0.1]])
    visible = torch.tensor([[False, True]])

    report = confidence_occlusion_diagnostics(confidence, error, visible)

    assert report["mean_confidence_visible"] == pytest.approx(200.0)
    assert report["mean_confidence_occluded"] == pytest.approx(10.0)
    assert report["error_separation"] == pytest.approx(4.9)
    assert report["visible_count"] == 1
    assert report["occluded_count"] == 1


def test_diagnostics_tau_grid_recovers_a_separating_threshold():
    confidence = torch.tensor([[10.0, 200.0, 12.0, 180.0]])
    error = torch.tensor([[5.0, 0.1, 4.0, 0.2]])
    visible = torch.tensor([[False, True, False, True]])

    report = confidence_occlusion_diagnostics(
        confidence,
        error,
        visible,
        taus=(1.5, 100.0, 300.0),
    )
    by_tau = {row["tau"]: row for row in report["tau_grid"]}

    # tau=1.5 is below every `expp1` value, so everything is called visible -- the
    # inert-threshold failure mode the cluster scorer is in today.
    assert by_tau[1.5]["occlusion_accuracy_for_vis1"] == pytest.approx(1.0)
    assert by_tau[1.5]["occlusion_accuracy_for_vis0"] == pytest.approx(0.0)
    assert by_tau[100.0]["occlusion_accuracy"] == pytest.approx(1.0)
    assert by_tau[300.0]["occlusion_accuracy_for_vis1"] == pytest.approx(0.0)


def test_diagnostics_report_none_rather_than_nan_for_an_empty_side():
    confidence = torch.tensor([[10.0, 20.0]])
    error = torch.tensor([[1.0, 2.0]])
    visible = torch.ones(1, 2, dtype=torch.bool)

    report = confidence_occlusion_diagnostics(confidence, error, visible)

    assert report["mean_confidence_occluded"] is None
    assert report["error_separation"] is None
    assert all(row["occlusion_accuracy_for_vis0"] is None for row in report["tau_grid"])


def test_diagnostics_honour_the_valid_mask():
    confidence = torch.tensor([[10.0, 999.0]])
    error = torch.tensor([[1.0, 2.0]])
    visible = torch.tensor([[True, True]])
    valid = torch.tensor([[True, False]])

    report = confidence_occlusion_diagnostics(confidence, error, visible, valid)

    assert report["sample_count"] == 1
    assert report["mean_confidence"] == pytest.approx(10.0)


def test_diagnostics_mean_error_is_the_denominator_of_the_implied_optimum():
    """run_summary divides alpha by this, so it must be the term's own mean."""

    confidence = torch.tensor([[100.0, 300.0]])
    error = torch.tensor([[1.0, 3.0]])
    visible = torch.tensor([[True, False]])

    report = confidence_occlusion_diagnostics(confidence, error, visible)

    assert report["mean_error"] == pytest.approx(2.0)
    assert math.isclose(report["mean_confidence"], 200.0)
