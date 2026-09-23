"""
Persistent run history: stores each analysis run's scores and metric values,
and tracks "objectives" — the metric-linked recommendations issued in a run —
so the *next* run can report whether they were achieved, partially improved,
unchanged, or regressed. This is what makes the tool a closed feedback loop
rather than a one-off report.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import load_config
import skills as skills_lib

SCORING_SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
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
    metrics_json TEXT,
    scoring_version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS objectives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    rec_id TEXT,
    title TEXT,
    metric_key TEXT,
    direction TEXT,       -- 'lower' or 'higher' is better
    baseline_value REAL,
    healthy_threshold REAL,
    created_at TEXT
);
-- Ledger of every (agent, session, turn) whose skill XP has already been
-- awarded, so re-analyzing an overlapping period (e.g. "this week" run again
-- after also running "last 30 days") never double-counts XP. This is what
-- makes the Leveling tab's totals monotonic and cumulative across runs.
CREATE TABLE IF NOT EXISTS seen_turns (
    agent TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL,
    PRIMARY KEY (agent, session_id, turn_index)
);
-- Cumulative XP per skill (see lib/skills.py for the skill list and curve).
CREATE TABLE IF NOT EXISTS skill_xp (
    skill_key TEXT PRIMARY KEY,
    xp REAL NOT NULL DEFAULT 0
);
-- Distinct significant words ever contributed, across all history, used to
-- detect genuinely *new* vocabulary for the Vocabulary skill (an incremental
-- analogue of the Heaps' Law vocabulary-growth curve -- see Appendix B).
CREATE TABLE IF NOT EXISTS seen_vocabulary (
    word TEXT PRIMARY KEY
);
"""

# Direction + a "healthy" reference threshold used to judge whether an
# objective has been achieved (crossed into the healthy zone), independent of
# the exact threshold that triggered the recommendation in the first place.
# Sourced from config.yaml -> objective_thresholds (falls back to built-in
# defaults if the config file is missing or a key isn't present there).
def _metric_direction() -> dict:
    cfg = load_config()
    return {k: (v["direction"], v["healthy"]) for k, v in cfg["objective_thresholds"].items()}


METRIC_DIRECTION = _metric_direction()


@dataclass
class ObjectiveProgress:
    rec_id: str
    title: str
    metric_key: str
    direction: str
    baseline_value: float
    current_value: float
    healthy_threshold: float
    delta: float
    relative_change_pct: float
    status: str  # 'achieved' | 'improving' | 'no_change' | 'regressed'


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    columns = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
    if "scoring_version" not in columns:
        # Existing rows used the original score definitions. Preserve them as
        # version 1, then prevent them from being compared with version 2
        # rather than inventing a migration from incomplete aggregate data.
        con.execute("ALTER TABLE runs ADD COLUMN scoring_version INTEGER NOT NULL DEFAULT 1")
        con.commit()
    return con


def get_previous_run(
    con: sqlite3.Connection, scoring_version: int = SCORING_SCHEMA_VERSION
) -> sqlite3.Row | None:
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM runs WHERE scoring_version = ? ORDER BY run_at DESC LIMIT 1",
        (scoring_version,),
    ).fetchone()
    return row


def get_run_history(
    con: sqlite3.Connection, limit: int = 100, scoring_version: int = SCORING_SCHEMA_VERSION
) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return list(
        con.execute(
            "SELECT * FROM runs WHERE scoring_version = ? ORDER BY run_at ASC LIMIT ?",
            (scoring_version, limit),
        ).fetchall()
    )


def get_objectives_for_run(con: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return list(con.execute("SELECT * FROM objectives WHERE run_id = ?", (run_id,)).fetchall())


def evaluate_objectives(objectives: list[sqlite3.Row], current_metric_values: dict) -> list[ObjectiveProgress]:
    band = load_config()["objective_improvement_band_pct"] / 100.0
    results = []
    for o in objectives:
        metric_key = o["metric_key"]
        current = current_metric_values.get(metric_key)
        if current is None:
            continue
        baseline = o["baseline_value"]
        direction = o["direction"]
        healthy = o["healthy_threshold"]
        delta = current - baseline

        if baseline == 0:
            relative_change_pct = 0.0 if current == 0 else 100.0
        else:
            relative_change_pct = (delta / abs(baseline)) * 100

        if direction == "lower":
            achieved = current <= healthy
            improved = current < baseline * (1 - band)
            regressed = current > baseline * (1 + band)
        else:
            achieved = current >= healthy
            improved = current > baseline * (1 + band)
            regressed = current < baseline * (1 - band)

        if achieved:
            status = "achieved"
        elif improved:
            status = "improving"
        elif regressed:
            status = "regressed"
        else:
            status = "no_change"

        results.append(ObjectiveProgress(
            rec_id=o["rec_id"],
            title=o["title"],
            metric_key=metric_key,
            direction=direction,
            baseline_value=baseline,
            current_value=current,
            healthy_threshold=healthy,
            delta=delta,
            relative_change_pct=relative_change_pct,
            status=status,
        ))
    return results


def record_run(
    con: sqlite3.Connection,
    period_label: str,
    period_start: str,
    period_end: str,
    agents: list[str],
    total_prompts: int,
    total_sessions: int,
    scores: dict,
    metric_values: dict,
    new_objectives: list[dict],
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = con.execute(
        """
        INSERT INTO runs (run_at, period_label, period_start, period_end, agents,
                           total_prompts, total_sessions,
                           score_specificity, score_context, score_structure, score_efficiency, score_overall,
                           metrics_json, scoring_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now, period_label, period_start, period_end, ",".join(agents),
            total_prompts, total_sessions,
            scores.get("specificity", 0), scores.get("context", 0),
            scores.get("structure", 0), scores.get("efficiency", 0), scores.get("overall", 0),
            json.dumps(metric_values),
            SCORING_SCHEMA_VERSION,
        ),
    )
    run_id = cur.lastrowid

    for obj in new_objectives:
        metric_key = obj["metric_key"]
        direction, healthy = METRIC_DIRECTION.get(metric_key, ("lower", 0.1))
        con.execute(
            """
            INSERT INTO objectives (run_id, rec_id, title, metric_key, direction, baseline_value, healthy_threshold, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, obj["rec_id"], obj["title"], metric_key, direction, obj["baseline_value"], healthy, now),
        )
    con.commit()
    return run_id


def get_skill_xp_totals(con: sqlite3.Connection) -> dict:
    """Returns {skill_key: cumulative_xp} for all skills, defaulting unseen
    skills to 0.0 (e.g. right after a schema upgrade, or the very first run)."""
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT skill_key, xp FROM skill_xp").fetchall()
    totals = {row["skill_key"]: row["xp"] for row in rows}
    for key in skills_lib.SKILL_KEYS:
        totals.setdefault(key, 0.0)
    return totals


def award_skill_xp_for_new_turns(con: sqlite3.Connection, feats: list) -> tuple:
    """Awards Leveling-tab XP for every turn in `feats` not already present in
    `seen_turns`, then persists the updated cumulative totals. Returns
    (updated_totals, xp_gained_this_run) -- both {skill_key: xp} dicts -- so
    the report can both show current levels and call out "XP gained this
    run" as positive reinforcement for the just-analyzed period.

    Turns are looked up by (agent, session_id, turn_index), which is stable
    across re-runs with different/overlapping period filters, so cumulative
    totals are monotonic: analyzing "this week" and then "last 30 days"
    never double-counts the same turn's XP twice."""
    cfg = load_config()
    gained = {key: 0.0 for key in skills_lib.SKILL_KEYS}

    for f in feats:
        t = f.turn
        already_seen = con.execute(
            "SELECT 1 FROM seen_turns WHERE agent = ? AND session_id = ? AND turn_index = ?",
            (t.agent, t.session_id, t.turn_index),
        ).fetchone()
        if already_seen:
            continue

        boolean_xp = skills_lib.compute_boolean_skill_xp(f, cfg)
        for key, xp in boolean_xp.items():
            gained[key] += xp

        new_word_count = 0
        for word in skills_lib.turn_candidate_words(t.user_text):
            cur = con.execute("INSERT OR IGNORE INTO seen_vocabulary(word) VALUES (?)", (word,))
            if cur.rowcount:
                new_word_count += 1
        gained["vocabulary"] += skills_lib.compute_vocabulary_xp(new_word_count, cfg)

        con.execute(
            "INSERT INTO seen_turns (agent, session_id, turn_index, first_seen_at) VALUES (?, ?, ?, ?)",
            (t.agent, t.session_id, t.turn_index, datetime.now(timezone.utc).isoformat()),
        )

    totals = get_skill_xp_totals(con)
    for key, xp_gained in gained.items():
        if xp_gained:
            totals[key] += xp_gained
            con.execute(
                "INSERT INTO skill_xp (skill_key, xp) VALUES (?, ?) "
                "ON CONFLICT(skill_key) DO UPDATE SET xp = xp + excluded.xp",
                (key, xp_gained),
            )
    con.commit()
    return totals, gained
