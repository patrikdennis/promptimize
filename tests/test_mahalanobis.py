"""Tests for lib/mahalanobis.py."""
import math

from config import load_config
from mahalanobis import compute_mahalanobis_to_optimal


def _run(spec, ctx, struct, eff):
    return {
        "score_specificity": spec,
        "score_context": ctx,
        "score_structure": struct,
        "score_efficiency": eff,
    }


def test_returns_none_below_min_history_runs():
    cfg = load_config()["mahalanobis"]
    too_few = [_run(50, 50, 50, 50) for _ in range(cfg["min_history_runs"] - 1)]
    result = compute_mahalanobis_to_optimal(too_few, {"specificity": 50, "context": 50, "structure": 50, "efficiency": 50})
    assert result is None


def test_perfect_score_history_gives_zero_distance_at_optimal():
    cfg = load_config()["mahalanobis"]
    history = [_run(100, 100, 100, 100) for _ in range(cfg["min_history_runs"] + 2)]
    # History has zero variance; current position is exactly at the optimum.
    result = compute_mahalanobis_to_optimal(history, {"specificity": 100, "context": 100, "structure": 100, "efficiency": 100})
    assert result is not None
    assert math.isclose(result.distance, 0.0, abs_tol=1e-6)
    assert math.isclose(result.euclidean_distance, 0.0, abs_tol=1e-9)


def test_euclidean_distance_is_independent_of_covariance():
    cfg = load_config()["mahalanobis"]
    history = [_run(60 + i, 20 + i, 40 + i, 90 + i) for i in range(cfg["min_history_runs"] + 3)]
    current = {"specificity": 70, "context": 30, "structure": 45, "efficiency": 95}
    result = compute_mahalanobis_to_optimal(history, current)
    assert result is not None
    expected_euclidean = math.sqrt((100 - 70) ** 2 + (100 - 30) ** 2 + (100 - 45) ** 2 + (100 - 95) ** 2)
    assert math.isclose(result.euclidean_distance, expected_euclidean, abs_tol=1e-6)


def test_distance_is_finite_and_nonnegative_with_near_singular_history():
    # Historical runs where two axes are perfectly correlated (structure and
    # efficiency both increase together) -> a near-singular but NOT exactly
    # zero-variance raw covariance; shrinkage regularization must keep this
    # well-behaved (this reproduces the original numerical-instability bug
    # found in dev, which used real run history with this kind of structure).
    cfg = load_config()["mahalanobis"]
    n = max(cfg["min_history_runs"], 5)
    history = [_run(50 + i, 10 + i * 0.1, 30 + i * 2, 80 + i * 2) for i in range(n)]
    current = {"specificity": 55, "context": 12, "structure": 45, "efficiency": 95}
    result = compute_mahalanobis_to_optimal(history, current)
    assert result is not None
    assert result.distance >= 0
    assert math.isfinite(result.distance)
    # Should not blow up to an absurd magnitude relative to euclidean distance.
    assert result.distance < result.euclidean_distance * 50 + 1000


def test_exactly_zero_variance_history_still_returns_finite_distance():
    # Degenerate edge case: every historical run identical (zero variance on
    # every axis). Shrinkage floors the average variance at a small epsilon
    # rather than dividing by zero, so the result must stay a finite number
    # (even though it will be large, since any deviation from a historically
    # constant baseline is, correctly, maximally surprising).
    cfg = load_config()["mahalanobis"]
    history = [_run(50, 10, 30, 80) for _ in range(cfg["min_history_runs"])]
    current = {"specificity": 55, "context": 12, "structure": 31, "efficiency": 82}
    result = compute_mahalanobis_to_optimal(history, current)
    assert result is not None
    assert result.distance >= 0
    assert math.isfinite(result.distance)


def test_mean_reflects_history_average():
    cfg = load_config()["mahalanobis"]
    history = [_run(10, 20, 30, 40), _run(30, 40, 50, 60)] * ((cfg["min_history_runs"] // 2) + 1)
    result = compute_mahalanobis_to_optimal(history, {"specificity": 50, "context": 50, "structure": 50, "efficiency": 50})
    assert result is not None
    assert math.isclose(result.mean[0], 20.0, abs_tol=1e-9)
    assert math.isclose(result.mean[1], 30.0, abs_tol=1e-9)
