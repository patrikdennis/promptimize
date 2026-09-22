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
    metrics_json TEXT
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
    return con


def get_previous_run(con: sqlite3.Connection) -> sqlite3.Row | None:
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM runs ORDER BY run_at DESC LIMIT 1").fetchone()
    return row


def get_run_history(con: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return list(con.execute("SELECT * FROM runs ORDER BY run_at ASC LIMIT ?", (limit,)).fetchall())


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
                           metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now, period_label, period_start, period_end, ",".join(agents),
            total_prompts, total_sessions,
            scores.get("specificity", 0), scores.get("context", 0),
            scores.get("structure", 0), scores.get("efficiency", 0), scores.get("overall", 0),
            json.dumps(metric_values),
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
