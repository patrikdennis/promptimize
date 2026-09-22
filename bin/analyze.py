#!/usr/bin/env python3
"""
prompt-optimizer: analyze your own Copilot CLI / Claude Code prompt history for a
time period and produce a multi-tab HTML report with quantitative, evidence-backed
metrics, recommendations, and progress tracking across runs.

Usage:
    python3 bin/analyze.py                      # defaults to "this week", both agents
    python3 bin/analyze.py --period "last 30 days"
    python3 bin/analyze.py --period "2026-09-01..2026-09-21"
    python3 bin/analyze.py --agent copilot       # copilot | claude | all (default all)
    python3 bin/analyze.py --out ~/Desktop/report.html
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from backends import load_all_sessions, flatten_prompts, filter_sessions_by_range  # noqa: E402
from metrics import build_aggregate, extract_all_features  # noqa: E402
from recommendations import build_recommendations, discover_installed_skills  # noqa: E402
from report import render_html  # noqa: E402
from advanced_metrics import compute_all_advanced_metrics  # noqa: E402
from anomaly import compute_anomalies  # noqa: E402
from mahalanobis import compute_mahalanobis_to_optimal  # noqa: E402
from export import write_run_exports, build_export_payload  # noqa: E402
import history  # noqa: E402

DEFAULT_COPILOT_DB = Path.home() / ".copilot" / "session-store.db"
DEFAULT_CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
DEFAULT_OUT_DIR = ROOT / "out"
DEFAULT_HISTORY_DB = ROOT / "data" / "history.db"

# How many top recommendations become tracked objectives for the next run.
OBJECTIVE_COUNT = 6


def _start_of_week(dt: datetime) -> datetime:
    return (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def parse_period(period: str, now: datetime) -> tuple[datetime, datetime, str]:
    period = (period or "this week").strip().lower()

    m = re.match(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$", period)
    if m:
        start = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(m.group(2)).replace(tzinfo=timezone.utc) + timedelta(days=1)
        return start, end, f"{m.group(1)} to {m.group(2)}"

    m = re.match(r"^last (\d+) days?$", period)
    if m:
        n = int(m.group(1))
        return now - timedelta(days=n), now, f"last {n} days"

    if period in ("today",):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now, "today"
    if period in ("yesterday",):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1), "yesterday"
    if period in ("this week",):
        start = _start_of_week(now)
        return start, now, "this week"
    if period in ("last week",):
        this_start = _start_of_week(now)
        return this_start - timedelta(days=7), this_start, "last week"
    if period in ("this month",):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, now, "this month"
    if period in ("last month",):
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = this_month_start
        last_month_start = (this_month_start - timedelta(days=1)).replace(day=1)
        return last_month_start, last_month_end, "last month"
    if period in ("all", "all time", "everything"):
        return datetime(2000, 1, 1, tzinfo=timezone.utc), now, "all time"

    start = _start_of_week(now)
    return start, now, f"this week (unrecognized period '{period}', defaulted)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze prompt history and generate an optimization report.")
    ap.add_argument("--period", default="this week", help="e.g. 'today', 'this week', 'last week', 'this month', 'last 30 days', 'all', or 'YYYY-MM-DD..YYYY-MM-DD'")
    ap.add_argument("--agent", default="all", choices=["all", "copilot", "claude"], help="Which agent's history to include")
    ap.add_argument("--copilot-db", default=str(DEFAULT_COPILOT_DB))
    ap.add_argument("--claude-projects", default=str(DEFAULT_CLAUDE_PROJECTS))
    ap.add_argument("--out", default=None, help="Output HTML path (default: out/report-<timestamp>.html next to this script)")
    ap.add_argument("--history-db", default=str(DEFAULT_HISTORY_DB), help="Path to the local run-history SQLite database")
    ap.add_argument("--no-record", action="store_true", help="Don't record this run in history (useful for dry runs)")
    ap.add_argument("--open", action="store_true", help="Open the report in the default browser when done")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    start, end, period_label = parse_period(args.period, now)

    agents = ["copilot", "claude"] if args.agent == "all" else [args.agent]
    all_sessions = load_all_sessions(agents, Path(args.copilot_db), Path(args.claude_projects))
    sessions_in_range = filter_sessions_by_range(all_sessions, start, end)

    feats = extract_all_features(sessions_in_range)
    agg = build_aggregate(feats, sessions_in_range)
    adv = compute_all_advanced_metrics(sessions_in_range, feats)

    installed_skills = discover_installed_skills()
    recs = build_recommendations(agg, installed_skills)

    prompts_in_range = flatten_prompts(sessions_in_range)
    agents_included = sorted(set(p.agent for p in prompts_in_range)) or agents

    # --- History: evaluate objectives from the previous run, then record this one ---
    con = history.connect(Path(args.history_db))
    previous_run_row = history.get_previous_run(con)
    previous_run = dict(previous_run_row) if previous_run_row else None

    objective_progress = []
    if previous_run_row is not None:
        prev_objectives = history.get_objectives_for_run(con, previous_run_row["id"])
        objective_progress = history.evaluate_objectives(prev_objectives, agg.metric_values)

    # Captured BEFORE this run is recorded, so anomaly/Mahalanobis baselines
    # are estimated purely from prior runs and never trivially include the
    # value they are being compared against.
    prior_run_rows = [dict(r) for r in history.get_run_history(con)]
    prior_metric_values = [json.loads(r["metrics_json"]) for r in prior_run_rows if r.get("metrics_json")]
    anomalies = compute_anomalies(prior_metric_values, agg.metric_values)
    mahalanobis_result = compute_mahalanobis_to_optimal(prior_run_rows, agg.scores)

    run_id = None
    if not args.no_record:
        new_objectives = [
            {"rec_id": r.id, "title": r.title, "metric_key": r.metric_key, "baseline_value": r.current_value}
            for r in recs[:OBJECTIVE_COUNT]
            if r.metric_key is not None and r.current_value is not None
        ]
        run_id = history.record_run(
            con,
            period_label=period_label,
            period_start=start.isoformat(),
            period_end=end.isoformat(),
            agents=agents,
            total_prompts=agg.total_prompts,
            total_sessions=agg.total_sessions,
            scores=agg.scores,
            metric_values=agg.metric_values,
            new_objectives=new_objectives,
        )

    run_history = [dict(r) for r in history.get_run_history(con)]
    con.close()

    html = render_html(
        period_label=period_label,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        agg=agg,
        recs=recs,
        installed_skills=installed_skills,
        agents_included=agents_included,
        objective_progress=objective_progress,
        run_history=run_history,
        previous_run=previous_run,
        adv=adv,
        anomalies=anomalies,
        mahalanobis_result=mahalanobis_result,
    )

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = now.strftime("%Y%m%d-%H%M%S")
        out_path = DEFAULT_OUT_DIR / f"report-{ts}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    export_payload = build_export_payload(
        period_label=period_label,
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        agents=agents_included,
        agg=agg,
        adv=adv,
        anomalies=anomalies,
        mahalanobis_result=mahalanobis_result,
        run_id=run_id,
        generated_at=now.isoformat(),
    )
    json_path, csv_path = write_run_exports(out_path.with_suffix(""), export_payload)

    print(f"Analyzed {agg.total_prompts} turns across {agg.total_sessions} sessions ({period_label}).")
    print(f"Overall score: {agg.scores.get('overall', 0):.0f}/100")
    if previous_run:
        print(f"Previous run overall score: {previous_run['score_overall']:.0f}/100 (change: {agg.scores.get('overall', 0) - previous_run['score_overall']:+.1f})")
    print(f"Report written to: {out_path}")
    print(f"Machine-readable exports: {json_path.name}, {csv_path.name}")

    if args.open:
        webbrowser.open(f"file://{out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
