"""Tests for lib/history.py: objective status logic and run persistence."""
import sqlite3
from pathlib import Path

import history


def _obj(metric_key="vague_rate", direction="lower", baseline=0.20, healthy=0.08):
    # A plain dict works fine here: evaluate_objectives only ever does
    # dict-style __getitem__ access (o["metric_key"], etc.), so this doubles
    # for a sqlite3.Row without needing a live DB connection.
    return {
        "rec_id": "rec-1",
        "title": "Reduce vague language",
        "metric_key": metric_key,
        "direction": direction,
        "baseline_value": baseline,
        "healthy_threshold": healthy,
    }


def test_lower_is_better_achieved_when_at_or_below_healthy():
    result = history.evaluate_objectives([_obj()], {"vague_rate": 0.05})
    assert result[0].status == "achieved"


def test_lower_is_better_improving_when_below_band_but_not_yet_healthy():
    # baseline 0.20, band 5% -> "improving" needs current < 0.20 * 0.95 = 0.19,
    # but not yet <= healthy (0.08).
    result = history.evaluate_objectives([_obj()], {"vague_rate": 0.15})
    assert result[0].status == "improving"


def test_lower_is_better_regressed_when_above_band():
    result = history.evaluate_objectives([_obj()], {"vague_rate": 0.25})
    assert result[0].status == "regressed"


def test_lower_is_better_no_change_within_band():
    result = history.evaluate_objectives([_obj()], {"vague_rate": 0.20})
    assert result[0].status == "no_change"


def test_higher_is_better_achieved_when_at_or_above_healthy():
    obj = _obj(metric_key="file_ref_rate", direction="higher", baseline=0.30, healthy=0.55)
    result = history.evaluate_objectives([obj], {"file_ref_rate": 0.60})
    assert result[0].status == "achieved"


def test_higher_is_better_regressed_when_below_band():
    obj = _obj(metric_key="file_ref_rate", direction="higher", baseline=0.30, healthy=0.55)
    result = history.evaluate_objectives([obj], {"file_ref_rate": 0.20})
    assert result[0].status == "regressed"


def test_skips_objective_when_metric_missing_from_current_run():
    result = history.evaluate_objectives([_obj()], {})
    assert result == []


def test_relative_change_pct_handles_zero_baseline():
    obj = _obj(baseline=0.0)
    result = history.evaluate_objectives([obj], {"vague_rate": 0.0})
    assert result[0].relative_change_pct == 0.0
    obj2 = _obj(baseline=0.0)
    result2 = history.evaluate_objectives([obj2], {"vague_rate": 0.1})
    assert result2[0].relative_change_pct == 100.0


def test_record_run_and_get_run_history_roundtrip(tmp_path):
    db_path = Path(tmp_path) / "history.db"
    con = history.connect(db_path)
    run_id = history.record_run(
        con,
        period_label="this week",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-01-08T00:00:00+00:00",
        agents=["copilot"],
        total_prompts=10,
        total_sessions=2,
        scores={"specificity": 80, "context": 60, "structure": 70, "efficiency": 90},
        metric_values={"vague_rate": 0.1},
        new_objectives=[{"rec_id": "r1", "title": "Test objective", "metric_key": "vague_rate", "baseline_value": 0.1}],
    )
    assert isinstance(run_id, int)

    rows = history.get_run_history(con)
    assert len(rows) == 1
    assert rows[0]["score_specificity"] == 80

    objectives = history.get_objectives_for_run(con, run_id)
    assert len(objectives) == 1
    assert objectives[0]["metric_key"] == "vague_rate"
    # direction/healthy_threshold should come from METRIC_DIRECTION/config, not
    # from the (unspecified) objective dict.
    assert objectives[0]["direction"] == "lower"

    con.close()
