"""Tests for lib/anomaly.py: z-score anomaly classification."""
from config import load_config
from anomaly import compute_anomalies, TRACKED_METRICS


def _history(n, value):
    """n prior runs, all with the same metric value (zero historical variance)."""
    return [{"correction_rate": value} for _ in range(n)]


def test_returns_empty_below_min_history_runs():
    cfg = load_config()["anomaly_detection"]
    history = _history(cfg["min_history_runs"] - 1, 0.1)
    result = compute_anomalies(history, {"correction_rate": 0.1})
    assert result == []


def test_flags_large_deviation_as_flagged():
    cfg = load_config()["anomaly_detection"]
    n = max(cfg["min_history_runs"], 6)
    # Historical correction_rate oscillates narrowly around 0.05.
    history = [{"correction_rate": v} for v in ([0.04, 0.05, 0.06, 0.05, 0.04, 0.06] * 2)[:n]]
    current = {"correction_rate": 0.9}  # wildly higher than history
    results = compute_anomalies(history, current)
    by_key = {a.metric_key: a for a in results}
    assert "correction_rate" in by_key
    flag = by_key["correction_rate"]
    assert flag.status == "flagged"
    assert flag.z_score > 0
    # correction_rate is "higher_is_bad" -> a large positive z is concerning.
    assert flag.concerning is True


def test_normal_when_close_to_historical_mean():
    cfg = load_config()["anomaly_detection"]
    n = max(cfg["min_history_runs"], 6)
    history = [{"correction_rate": v} for v in ([0.04, 0.05, 0.06, 0.05, 0.04, 0.06] * 2)[:n]]
    current = {"correction_rate": 0.05}
    results = compute_anomalies(history, current)
    by_key = {a.metric_key: a for a in results}
    assert by_key["correction_rate"].status == "normal"


def test_zero_variance_history_with_matching_current_is_normal():
    cfg = load_config()["anomaly_detection"]
    n = max(cfg["min_history_runs"], 5)
    history = _history(n, 0.1)
    current = {"correction_rate": 0.1}
    results = compute_anomalies(history, current)
    by_key = {a.metric_key: a for a in results}
    assert by_key["correction_rate"].status == "normal"
    assert by_key["correction_rate"].z_score == 0.0


def test_zero_variance_history_with_any_deviation_is_flagged():
    cfg = load_config()["anomaly_detection"]
    n = max(cfg["min_history_runs"], 5)
    history = _history(n, 0.1)
    current = {"correction_rate": 0.15}
    results = compute_anomalies(history, current)
    by_key = {a.metric_key: a for a in results}
    assert by_key["correction_rate"].status == "flagged"


def test_skips_metric_missing_from_current_run():
    cfg = load_config()["anomaly_detection"]
    n = max(cfg["min_history_runs"], 5)
    history = _history(n, 0.1)
    results = compute_anomalies(history, {})
    assert results == []


def test_all_tracked_metrics_have_valid_direction_label():
    for key, label, direction in TRACKED_METRICS:
        assert direction in ("higher_is_bad", "lower_is_bad")
        assert isinstance(key, str) and key
        assert isinstance(label, str) and label


def test_results_sorted_by_absolute_z_descending():
    cfg = load_config()["anomaly_detection"]
    n = max(cfg["min_history_runs"], 6)
    history = [
        {"correction_rate": v, "vague_rate": w}
        for v, w in zip([0.04, 0.05, 0.06, 0.05, 0.04, 0.06] * 2, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] * 2)
    ][:n]
    current = {"correction_rate": 0.9, "vague_rate": 0.11}
    results = compute_anomalies(history, current)
    abs_zs = [abs(a.z_score) for a in results]
    assert abs_zs == sorted(abs_zs, reverse=True)
