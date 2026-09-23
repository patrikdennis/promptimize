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


def test_history_only_compares_runs_with_current_scoring_version(tmp_path):
    db_path = Path(tmp_path) / "history.db"
    con = history.connect(db_path)
    con.execute(
        """
        INSERT INTO runs (
            run_at, period_label, period_start, period_end, agents,
            total_prompts, total_sessions,
            score_specificity, score_context, score_structure, score_efficiency, score_overall,
            metrics_json, scoring_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-01-01T00:00:00+00:00", "legacy", "2026-01-01", "2026-01-02", "copilot",
            1, 1, 1, 1, 1, 1, 1, "{}", 1,
        ),
    )
    history.record_run(
        con,
        period_label="current",
        period_start="2026-01-02T00:00:00+00:00",
        period_end="2026-01-03T00:00:00+00:00",
        agents=["copilot"],
        total_prompts=1,
        total_sessions=1,
        scores={"specificity": 80, "context": 70, "structure": 60, "efficiency": 90},
        metric_values={"vague_rate": 0.1},
        new_objectives=[],
    )

    rows = history.get_run_history(con)
    assert len(rows) == 1
    assert rows[0]["period_label"] == "current"
    assert rows[0]["scoring_version"] == history.SCORING_SCHEMA_VERSION
    assert history.get_previous_run(con)["period_label"] == "current"
    con.close()


def test_connect_migrates_legacy_runs_table_with_version_one_rows(tmp_path):
    db_path = Path(tmp_path) / "legacy.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            period_label TEXT,
            period_start TEXT,
            period_end TEXT,
            agents TEXT,
            total_prompts INTEGER,
            total_sessions INTEGER,
            score_specificity REAL,
            score_context REAL,
            score_structure REAL,
            score_efficiency REAL,
            score_overall REAL,
            metrics_json TEXT
        )
        """
    )
    con.execute(
        "INSERT INTO runs (run_at, total_prompts, total_sessions) VALUES (?, ?, ?)",
        ("2026-01-01T00:00:00+00:00", 1, 1),
    )
    con.commit()
    con.close()

    migrated = history.connect(db_path)
    row = migrated.execute("SELECT scoring_version FROM runs").fetchone()
    assert row[0] == 1
    assert history.get_run_history(migrated) == []
    migrated.close()


# --- Leveling: seen_turns dedup + skill XP award ----------------------------

def _mk_feats(session_id, texts):
    from datetime import datetime, timedelta, timezone
    from backends import Session, Turn
    from metrics import extract_all_features

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns = [
        Turn(agent="copilot", session_id=session_id, turn_index=i, timestamp=base + timedelta(minutes=i),
             user_text=text, assistant_text=None)
        for i, text in enumerate(texts)
    ]
    session = Session(agent="copilot", session_id=session_id, cwd="/tmp/p", repository="acme/repo", turns=turns)
    return extract_all_features([session])


def test_award_skill_xp_awards_on_first_run(tmp_path):
    con = history.connect(Path(tmp_path) / "history.db")
    feats = _mk_feats("s1", ["fix the bug in `src/app.py`, make sure tests still pass"])
    totals, gained = history.award_skill_xp_for_new_turns(con, feats)
    assert totals["context_anchoring"] > 0
    assert gained["context_anchoring"] > 0
    con.close()


def test_award_skill_xp_never_double_counts_same_turn(tmp_path):
    con = history.connect(Path(tmp_path) / "history.db")
    feats = _mk_feats("s1", ["fix the bug in `src/app.py`, make sure tests still pass"])

    totals1, gained1 = history.award_skill_xp_for_new_turns(con, feats)
    # Re-run with the exact same turns (simulating an overlapping period
    # re-analysis) -- no new XP should be gained, and totals must be identical.
    totals2, gained2 = history.award_skill_xp_for_new_turns(con, feats)

    assert totals1 == totals2
    assert all(v == 0.0 for v in gained2.values())
    con.close()


def test_award_skill_xp_accumulates_across_distinct_new_turns(tmp_path):
    con = history.connect(Path(tmp_path) / "history.db")
    feats1 = _mk_feats("s1", ["fix the bug in `src/app.py`, make sure tests still pass"])
    totals1, _ = history.award_skill_xp_for_new_turns(con, feats1)

    feats2 = _mk_feats("s2", ["add a new endpoint for exporting reports"])
    totals2, gained2 = history.award_skill_xp_for_new_turns(con, feats2)

    assert totals2["efficiency"] >= totals1["efficiency"] + gained2["efficiency"]
    con.close()


def test_get_skill_xp_totals_defaults_all_skills_to_zero(tmp_path):
    con = history.connect(Path(tmp_path) / "history.db")
    totals = history.get_skill_xp_totals(con)
    import skills
    assert set(totals.keys()) == set(skills.SKILL_KEYS)
    assert all(v == 0.0 for v in totals.values())
    con.close()
