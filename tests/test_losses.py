from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from arc.training import (
    CONFIDENCE_DIAGNOSTICS_VERSION,
    DEFAULT_CONFIDENCE_TAU_MULTIPLES,
    DEFAULT_CONFIDENCE_TAUS,
    compose_tracking_loss,
    confidence_occlusion_diagnostics,
    per_sample_huber_error,
    resolve_confidence_alpha,
    synchronized_consistency_loss,
    synchronized_consistency_stats,
    synchronized_pair_indices,
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


# ------------------------------------------------ relative tau grid ---
# A tau is a property of a model state, not of the protocol: the confidence term's
# optimum is conf*=alpha/err, so the whole channel level climbs as training drives the
# error down.  The archived weight sweep moved mean visible confidence from 6175 to
# 25452, which is why a grid fixed in absolute units samples a different part of every
# run's curve and their occlusion accuracies cannot be compared.
def _scale_free_sample(scale=1.0):
    """A separating visible/occluded split whose confidences rescale wholesale.

    Powers of two throughout so rescaling only shifts exponents and the comparisons
    stay bit-exact -- the test is about the grid, not about float rounding.
    """

    visible = torch.tensor([[True] * 6 + [False] * 4])
    confidence = scale * torch.tensor(
        [[1024.0, 2048.0, 2048.0, 4096.0, 4096.0, 8192.0, 64.0, 128.0, 128.0, 256.0]]
    )
    # mean = 1.0, so alpha passes straight through as implied_optimal_confidence.
    error = torch.tensor([[0.1] * 6 + [2.35] * 4])
    return confidence, error, visible


def test_relative_grid_is_invariant_to_the_runs_confidence_scale():
    """The whole point: two runs at different confidence scales become comparable.

    Scaling the channel and alpha together scales the implied optimum with them, so a
    grid anchored to it lands on the same points of each run's curve.  Every accuracy
    must be identical and only the absolute taus may move.
    """

    base = confidence_occlusion_diagnostics(
        *_scale_free_sample(1.0), confidence_alpha=100.0
    )
    scaled = confidence_occlusion_diagnostics(
        *_scale_free_sample(1024.0), confidence_alpha=1024 * 100.0
    )

    assert scaled["implied_optimal_confidence"] == 1024 * base["implied_optimal_confidence"]
    assert len(scaled["relative_tau_grid"]) == len(base["relative_tau_grid"])
    for base_row, scaled_row in zip(base["relative_tau_grid"], scaled["relative_tau_grid"]):
        assert scaled_row["multiple"] == base_row["multiple"]
        assert scaled_row["tau"] == 1024 * base_row["tau"]
        for key in (
            "occlusion_accuracy",
            "occlusion_accuracy_for_vis1",
            "occlusion_accuracy_for_vis0",
            "predicted_visible_fraction",
        ):
            assert scaled_row[key] == base_row[key], key
    assert scaled["best"]["relative"]["occlusion_accuracy"] == (
        base["best"]["relative"]["occlusion_accuracy"]
    )

    # The fixed grid is what this replaces: same two runs, different answers.
    assert scaled["best"]["absolute"]["occlusion_accuracy"] != (
        base["best"]["absolute"]["occlusion_accuracy"]
    )


def test_relative_grid_extends_past_the_optimum_instead_of_stopping_on_it():
    """The archived defect: accuracy still rising at the last tau, so the best
    reported value was set by where the grid stopped rather than by the model."""

    report = confidence_occlusion_diagnostics(
        *_scale_free_sample(1.0), confidence_alpha=100.0
    )
    grid = report["relative_tau_grid"]

    # The top of the grid is above every confidence, so nothing is called visible and
    # accuracy has fallen to the all-occluded value -- it is provably past the turn.
    assert grid[-1]["predicted_visible_fraction"] == pytest.approx(0.0)
    assert grid[-1]["occlusion_accuracy"] == pytest.approx(0.4)
    assert grid[0]["predicted_visible_fraction"] == pytest.approx(1.0)
    assert grid[0]["occlusion_accuracy"] == pytest.approx(0.6)

    best = report["best"]["relative"]
    assert best["occlusion_accuracy"] == pytest.approx(1.0)
    assert not best["at_grid_edge"]
    assert best["basis"] == report["relative_tau_grid_basis"]


def test_a_grid_that_stops_before_the_optimum_says_so():
    """The flag that would have caught the archived runs, where the argmax sat on the
    last point of a grid whose top was 0.35x the run's own implied optimum."""

    report = confidence_occlusion_diagnostics(
        *_scale_free_sample(1.0),
        confidence_alpha=100.0,
        multiples=(0.01, 0.1, 1.0),
    )

    best = report["best"]["relative"]
    assert best["tau"] == pytest.approx(100.0)
    assert best["at_grid_edge"]
    # Six visible above it, and only 64 of the four occluded below it.
    assert best["occlusion_accuracy"] == pytest.approx(0.7)
    # Still rising where it stopped, which is exactly the archived failure.
    assert best["occlusion_accuracy"] > report["relative_tau_grid"][-2]["occlusion_accuracy"]


def test_the_argmax_is_reported_against_the_trivial_all_visible_baseline():
    """Calling everything visible needs no model, so it is the bar a threshold must
    clear; at the released checkpoint none did (0.684 against 0.700)."""

    confidence = torch.tensor([[300.0, 20.0, 400.0, 30.0, 25.0]])
    error = torch.tensor([[0.1, 1.0, 0.1, 1.0, 1.0]])
    visible = torch.tensor([[True, False, True, False, True]])

    report = confidence_occlusion_diagnostics(confidence, error, visible)

    assert report["trivial_all_visible_occlusion_accuracy"] == pytest.approx(3 / 5)
    best = report["best"]["absolute"]
    assert best["occlusion_accuracy"] == max(
        row["occlusion_accuracy"] for row in report["tau_grid"]
    )
    assert best["tau"] in [row["tau"] for row in report["tau_grid"]]
    assert best["multiple"] is None


def test_the_trivial_baseline_counts_only_valid_samples():
    confidence = torch.tensor([[10.0, 20.0, 999.0]])
    error = torch.tensor([[1.0, 1.0, 1.0]])
    visible = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])

    report = confidence_occlusion_diagnostics(confidence, error, visible, valid)

    assert report["trivial_all_visible_occlusion_accuracy"] == pytest.approx(0.5)


def test_a_degenerate_all_equal_confidence_still_produces_a_grid():
    """Every sample at one confidence is the case a quantile-derived grid collapses
    on -- every quantile is the same number.  Anchoring to alpha/mean_error instead
    depends on no spread at all, so the grid survives and simply reports that no
    threshold separates anything."""

    confidence = torch.full((1, 8), 500.0)
    error = torch.tensor([[0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]])
    visible = torch.tensor([[True] * 5 + [False] * 3])

    report = confidence_occlusion_diagnostics(confidence, error, visible, confidence_alpha=100.0)
    grid = report["relative_tau_grid"]

    assert len(grid) == len(DEFAULT_CONFIDENCE_TAU_MULTIPLES)
    assert all(row["occlusion_accuracy"] is not None for row in grid)
    assert all(math.isfinite(row["occlusion_accuracy"]) for row in grid)
    # One flat step from all-visible to all-occluded, and nothing in between.
    assert {row["predicted_visible_fraction"] for row in grid} == {0.0, 1.0}
    assert report["best"]["relative"] is not None
    # Nothing was located, and the flat grid ties at its first row, so it reads as an
    # edge rather than as a discovered optimum.
    assert report["best"]["relative"]["at_grid_edge"]


def test_a_zero_error_run_reports_no_relative_grid_rather_than_a_nan_one():
    """conf* = alpha/err has no finite value once the error reaches zero.  A missing
    grid is diagnosable; a grid of infinities silently poisons every row."""

    confidence = torch.tensor([[100.0, 200.0]])
    error = torch.zeros(1, 2)
    visible = torch.tensor([[True, False]])

    report = confidence_occlusion_diagnostics(confidence, error, visible, confidence_alpha=1.0)

    assert report["implied_optimal_confidence"] is None
    assert report["relative_tau_grid"] is None
    assert report["relative_tau_grid_basis"] is None
    assert report["best"]["relative"] is None
    assert report["best"]["absolute"] is not None


def test_without_alpha_there_is_no_relative_grid_and_the_report_says_so():
    """Rather than quietly falling back to the absolute grid under a relative name."""

    report = confidence_occlusion_diagnostics(*_scale_free_sample(1.0))

    assert report["implied_optimal_confidence"] is None
    assert report["relative_tau_grid"] is None
    assert report["tau_grid"] is not None
    assert report["confidence_diagnostics_version"] == CONFIDENCE_DIAGNOSTICS_VERSION


def test_the_absolute_grid_keeps_its_archived_thresholds_and_basis():
    """Four archived cluster runs were scored on exactly these taus.  Moving them
    would strand those numbers, so the relative grid is added beside this one."""

    report = confidence_occlusion_diagnostics(
        *_scale_free_sample(1.0), confidence_alpha=100.0
    )

    assert [row["tau"] for row in report["tau_grid"]] == list(DEFAULT_CONFIDENCE_TAUS)
    assert "absolute" in report["tau_grid_basis"]
    assert "implied_optimal_confidence" in report["relative_tau_grid_basis"]
    assert report["best"]["absolute"]["basis"] == report["tau_grid_basis"]


# ----------------------------------------------------- synchronized pairs ---
def test_synchronized_pair_indices_pair_camera_major_layouts():
    first, second = synchronized_pair_indices(torch.tensor([0, 1, 2, 0, 1, 2]))
    assert list(zip(first, second)) == [(0, 3), (1, 4), (2, 5)]

    # Three cameras of one instant: all pairs of the triple.
    first, second = synchronized_pair_indices(torch.tensor([0, 0, 0]))
    assert list(zip(first, second)) == [(0, 1), (0, 2), (1, 2)]

    # Monocular video: nothing is synchronized.
    assert synchronized_pair_indices(torch.tensor([0, 1, 2])) == ([], [])

    with pytest.raises(ValueError, match="one-dimensional"):
        synchronized_pair_indices(torch.zeros(2, 2))


def _paired_fields(seed=0):
    """(1,Q,S,H,W,3) fields for a 2-camera x 2-time window, exactly consistent."""

    generator = torch.Generator().manual_seed(seed)
    fields = torch.randn(1, 1, 4, 2, 2, 3, generator=generator)
    slot_time_indices = torch.tensor([0, 1, 0, 1])
    fields[:, :, 2] = fields[:, :, 0]
    fields[:, :, 3] = fields[:, :, 1]
    return fields, slot_time_indices


def test_sync_loss_is_zero_for_consistent_fields_and_positive_otherwise():
    fields, slot_time_indices = _paired_fields()

    loss = synchronized_consistency_loss(
        fields, slot_time_indices, huber_delta=0.05
    )
    assert loss.item() == 0.0

    perturbed = fields.clone()
    perturbed[:, :, 3] += 0.01
    assert synchronized_consistency_loss(
        perturbed, slot_time_indices, huber_delta=0.05
    ).item() > 0.0


def test_sync_loss_ignores_differences_between_different_times():
    """Only same-instant disagreement is a defect; motion across times is not."""

    fields, slot_time_indices = _paired_fields()
    moved = fields.clone()
    # Move both time-1 slots identically: cross-time difference changes, the
    # same-index pairs stay equal.
    moved[:, :, 1] += 5.0
    moved[:, :, 3] += 5.0

    loss = synchronized_consistency_loss(moved, slot_time_indices, huber_delta=0.05)
    assert loss.item() == 0.0


def test_sync_loss_metric_scale_is_quadratic_below_the_huber_knee():
    fields, slot_time_indices = _paired_fields()
    perturbed = fields.clone()
    perturbed[:, :, 2] += 1e-3

    small = synchronized_consistency_loss(
        perturbed, slot_time_indices, huber_delta=1.0, metric_scale=1.0
    )
    doubled = synchronized_consistency_loss(
        perturbed, slot_time_indices, huber_delta=1.0, metric_scale=2.0
    )
    assert doubled.item() == pytest.approx(4.0 * small.item(), rel=1e-5)


def test_sync_loss_gradient_reaches_only_paired_slots():
    generator = torch.Generator().manual_seed(1)
    fields = torch.randn(1, 1, 3, 2, 2, 3, generator=generator).requires_grad_(True)
    # Slot 1 shares its index with nobody, so no pair touches it.
    slot_time_indices = torch.tensor([0, 1, 0])

    synchronized_consistency_loss(
        fields, slot_time_indices, huber_delta=10.0
    ).backward()

    assert fields.grad is not None
    assert fields.grad[:, :, 0].abs().sum() > 0
    assert fields.grad[:, :, 2].abs().sum() > 0
    assert fields.grad[:, :, 1].abs().sum() == 0


def test_sync_loss_validates_inputs():
    fields, slot_time_indices = _paired_fields()

    with pytest.raises(ValueError, match=r"\(1,Q,S,H,W,3\)"):
        synchronized_consistency_loss(
            fields[0], slot_time_indices, huber_delta=0.05
        )
    with pytest.raises(ValueError, match="one entry per observation slot"):
        synchronized_consistency_loss(
            fields, slot_time_indices[:3], huber_delta=0.05
        )
    with pytest.raises(ValueError, match="No synchronized observation pairs"):
        synchronized_consistency_loss(
            fields, torch.tensor([0, 1, 2, 3]), huber_delta=0.05
        )
    with pytest.raises(ValueError, match="huber_delta"):
        synchronized_consistency_loss(fields, slot_time_indices, huber_delta=0.0)
    with pytest.raises(ValueError, match="metric_scale"):
        synchronized_consistency_loss(
            fields, slot_time_indices, huber_delta=0.05, metric_scale=0.0
        )


def test_sync_stats_report_metric_disagreement_and_skip_unpaired_windows():
    fields, slot_time_indices = _paired_fields()
    perturbed = fields.clone()
    # A uniform 3 mm offset on one synchronized slot: every pixel of that pair
    # disagrees by exactly sqrt(3 * 0.003^2) after the metric lift below.
    perturbed[:, :, 2] += 0.003

    stats = synchronized_consistency_stats(
        perturbed, slot_time_indices, metric_scale=2.0
    )
    expected = 2.0 * (3 * 0.003**2) ** 0.5
    assert stats["pair_count"] == 2
    assert stats["p90_m"] >= stats["median_m"] >= 0.0
    assert stats["mean_m"] == pytest.approx(expected / 2.0, rel=1e-5)

    assert synchronized_consistency_stats(
        fields, torch.tensor([0, 1, 2, 3])
    ) is None
