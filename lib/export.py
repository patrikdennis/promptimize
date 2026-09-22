"""
Machine-readable export of a single run's results: JSON (full structured
dump) and CSV (flat metric/value rows) written alongside the HTML report.

This exists so the tool's output isn't locked inside a human-readable HTML
page: the JSON is what `bin/aggregate_team.py` consumes to build anonymized
team-level rollups (see team_mode docs in README), and both formats are
suitable for feeding into an external dashboard, a CI quality gate that
diffs metrics between runs, or a spreadsheet.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _to_jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def build_export_payload(
    *,
    period_label: str,
    period_start: str,
    period_end: str,
    agents: list[str],
    agg,
    adv,
    anomalies: list,
    mahalanobis_result,
    run_id: int | None,
    generated_at: str,
    tool_version: str = "1.0",
) -> dict:
    """Builds the full structured export payload. Intentionally excludes any
    raw prompt/reply text -- only aggregate rates, scores, and derived
    statistics are exported, so this file is safe to share for team rollups."""
    return {
        "schema_version": 1,
        "tool_version": tool_version,
        "generated_at": generated_at,
        "run_id": run_id,
        "period": {"label": period_label, "start": period_start, "end": period_end},
        "agents": agents,
        "totals": {"total_prompts": agg.total_prompts, "total_sessions": agg.total_sessions},
        "scores": agg.scores,
        "metric_values": agg.metric_values,
        "conditional_stats": agg.conditional_stats,
        "advanced_metrics": {
            "markov": _to_jsonable(adv.markov) if adv.markov else None,
            "windings": _to_jsonable(adv.windings),
            "burstiness": _to_jsonable(adv.burstiness) if adv.burstiness else None,
            "compression": _to_jsonable(adv.compression) if adv.compression else None,
            "heaps": _to_jsonable(adv.heaps) if adv.heaps else None,
        },
        "anomalies": [_to_jsonable(a) for a in anomalies],
        "mahalanobis": _to_jsonable(mahalanobis_result) if mahalanobis_result else None,
    }


def write_run_exports(out_path_base: Path, payload: dict) -> tuple[Path, Path]:
    """Writes <out_path_base>.json and <out_path_base>.csv. Returns the two paths."""
    json_path = out_path_base.with_suffix(".json")
    csv_path = out_path_base.with_suffix(".csv")

    json_path.write_text(json.dumps(payload, indent=2, default=str))

    rows = [("category", "metric", "value")]
    for k, v in payload["scores"].items():
        rows.append(("score", k, v))
    for k, v in payload["metric_values"].items():
        rows.append(("metric", k, v))
    rows.append(("total", "total_prompts", payload["totals"]["total_prompts"]))
    rows.append(("total", "total_sessions", payload["totals"]["total_sessions"]))

    adv = payload["advanced_metrics"]
    if adv.get("markov"):
        for state, val in adv["markov"]["expected_turns_to_resolution"].items():
            rows.append(("markov_expected_turns", state, val))
    if adv.get("burstiness"):
        rows.append(("advanced", "burstiness", adv["burstiness"]["burstiness"]))
        rows.append(("advanced", "memory", adv["burstiness"]["memory"]))
    if adv.get("compression"):
        rows.append(("advanced", "compression_ratio", adv["compression"]["ratio"]))
    if adv.get("heaps"):
        rows.append(("advanced", "heaps_beta", adv["heaps"]["beta"]))
        rows.append(("advanced", "heaps_r_squared", adv["heaps"]["r_squared"]))
    if payload.get("mahalanobis"):
        rows.append(("advanced", "mahalanobis_distance", payload["mahalanobis"]["distance"]))
        rows.append(("advanced", "euclidean_distance", payload["mahalanobis"]["euclidean_distance"]))
    for a in payload["anomalies"]:
        rows.append((f"anomaly_{a['status']}", a["metric_key"], a["z_score"]))

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return json_path, csv_path
