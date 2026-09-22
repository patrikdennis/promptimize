#!/usr/bin/env python3
"""
prompt-optimizer: anonymized team/aggregate report.

Ingests multiple people's exported JSON files (each produced locally by
`bin/analyze.py`, one per person, shared voluntarily -- this tool never
transmits anything anywhere on its own) and produces a single self-contained
HTML summary: score distribution across the group, which axes are weakest
on average, aggregate anomaly-flag frequency, and aggregate advanced-metrics
summaries. No raw prompt/reply text ever enters an export file in the first
place (see lib/export.py), so nothing sensitive can leak through this step
either.

Usage:
    python3 bin/aggregate_team.py out/*.json
    python3 bin/aggregate_team.py alice.json bob.json carol.json --open
    python3 bin/aggregate_team.py out/*.json --out ~/Desktop/team-report.html

Contributors are always labeled "Contributor 1", "Contributor 2", ... in the
order their files were given on the command line -- never by filename,
username, or any other identifying string (config: team_mode.anonymize_labels).
A minimum of team_mode.min_users_for_aggregate (default 2) exports is
required, since an "aggregate" of one person is not actually anonymous.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from config import load_config  # noqa: E402

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

AXES = ["specificity", "context", "structure", "efficiency"]
AXIS_LABELS = {
    "specificity": "Specificity",
    "context": "Context anchoring",
    "structure": "Structure / acceptance criteria",
    "efficiency": "Efficiency",
}


def load_exports(paths: list[Path]) -> list[dict]:
    payloads = []
    for p in paths:
        try:
            payloads.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: skipping {p} ({e})", file=sys.stderr)
    return payloads


def weakest_axis(scores: dict) -> str:
    return min(AXES, key=lambda a: scores.get(a, 0))


def build_aggregate(payloads: list[dict], anonymize: bool) -> dict:
    n = len(payloads)
    labels = [f"Contributor {i+1}" for i in range(n)] if anonymize else [
        p.get("period", {}).get("label", f"Contributor {i+1}") for i, p in enumerate(payloads)
    ]

    overall_scores = [p["scores"]["overall"] for p in payloads]
    axis_scores = {a: [p["scores"].get(a, 0) for p in payloads] for a in AXES}
    weakest_counts = {a: 0 for a in AXES}
    for p in payloads:
        weakest_counts[weakest_axis(p["scores"])] += 1

    anomaly_counts: dict[str, dict[str, int]] = {}
    for p in payloads:
        for a in p.get("anomalies", []):
            key = a["label"]
            anomaly_counts.setdefault(key, {"watch": 0, "flagged": 0, "normal": 0})
            status = a.get("status", "normal")
            anomaly_counts[key][status] = anomaly_counts[key].get(status, 0) + 1

    mahal = [p["mahalanobis"]["distance"] for p in payloads if p.get("mahalanobis")]

    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "median": None, "stdev": None, "min": None, "max": None}
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "n": n,
        "labels": labels,
        "overall_scores": overall_scores,
        "axis_scores": axis_scores,
        "weakest_counts": weakest_counts,
        "anomaly_counts": anomaly_counts,
        "mahalanobis_distances": mahal,
        "overall_stats": stats(overall_scores),
        "axis_stats": {a: stats(v) for a, v in axis_scores.items()},
        "mahalanobis_stats": stats(mahal),
    }


def _fmt(x, digits=1):
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_html(agg: dict, generated_at: str) -> str:
    n = agg["n"]
    labels_json = json.dumps(agg["labels"])
    overall_json = json.dumps(agg["overall_scores"])

    axis_bar_traces = ",\n".join(
        f"""{{ x: {labels_json}, y: {json.dumps(agg['axis_scores'][a])}, name: {json.dumps(AXIS_LABELS[a])}, type: 'bar' }}"""
        for a in AXES
    )

    weakest_labels = json.dumps([AXIS_LABELS[a] for a in AXES])
    weakest_values = json.dumps([agg["weakest_counts"][a] for a in AXES])

    anomaly_items = sorted(
        agg["anomaly_counts"].items(),
        key=lambda kv: (kv[1]["flagged"], kv[1]["watch"]),
        reverse=True,
    )
    anomaly_labels = json.dumps([k for k, _ in anomaly_items])
    anomaly_flagged = json.dumps([v["flagged"] for _, v in anomaly_items])
    anomaly_watch = json.dumps([v["watch"] for _, v in anomaly_items])

    stats_rows = "".join(
        f"<tr><td>{AXIS_LABELS[a]}</td>"
        f"<td class='num'>{_fmt(agg['axis_stats'][a]['mean'])}</td>"
        f"<td class='num'>{_fmt(agg['axis_stats'][a]['median'])}</td>"
        f"<td class='num'>{_fmt(agg['axis_stats'][a]['stdev'])}</td>"
        f"<td class='num'>{_fmt(agg['axis_stats'][a]['min'])}</td>"
        f"<td class='num'>{_fmt(agg['axis_stats'][a]['max'])}</td></tr>"
        for a in AXES
    )

    mahal_note = ""
    if agg["mahalanobis_distances"]:
        ms = agg["mahalanobis_stats"]
        mahal_note = (
            f"<p class='caption'>Across the {len(agg['mahalanobis_distances'])} contributors "
            f"who had enough personal run history to compute a Mahalanobis distance, the group "
            f"mean distance-to-optimal is {_fmt(ms['mean'])} (range {_fmt(ms['min'])} - "
            f"{_fmt(ms['max'])}). Each contributor's own distance is scaled by their own "
            f"score covariance (see Appendix A.6 in the main README), so these are not directly "
            f"comparable across people as an absolute ranking -- only as a relative signal of how "
            f"close each person is to their own optimum.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Team Aggregate Prompt-Optimization Report</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{
    --bg: #0f1115; --panel: #171a21; --text: #e8eaed;
    --muted: #9aa3b2; --accent: #6ee7b7; --accent2: #60a5fa;
    --border: #262b38;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 0 60px 0; background: radial-gradient(1200px 800px at 10% -10%, #1b2130 0%, var(--bg) 60%);
    color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 0 28px; overflow-x: hidden; }}
  h1 {{ font-size: 24px; margin: 0 0 4px 0; }}
  h2 {{ margin-top:0; font-size: 17px; }}
  .sub {{ color: var(--muted); margin-bottom: 20px; font-size: 13.5px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 22px; min-width: 0; overflow: hidden; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 14px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
  .card .big {{ font-size: 26px; font-weight: 700; }}
  .card .label {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; }}
  table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align:left; padding: 7px 9px; border-bottom: 1px solid var(--border); vertical-align:top; }}
  th {{ color: var(--muted); font-weight:600; text-transform:uppercase; font-size:10.5px; letter-spacing:.04em; }}
  th.num, td.num {{ text-align:center; }}
  .caption {{ color: var(--muted); font-size: 12.5px; margin-top: 8px; line-height:1.5; border-left: 2px solid var(--border); padding-left: 10px; }}
  .footer {{ color: var(--muted); font-size:12px; margin-top: 24px; padding: 0 28px;}}
  .plot {{ width: 100%; height: 380px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Team Aggregate Prompt-Optimization Report</h1>
  <div class="sub">{n} contributors - anonymized labels - generated {generated_at}</div>

  <div class="grid">
    <div class="card"><div class="big">{_fmt(agg['overall_stats']['mean'])}</div><div class="label">Mean overall score</div></div>
    <div class="card"><div class="big">{_fmt(agg['overall_stats']['median'])}</div><div class="label">Median overall score</div></div>
    <div class="card"><div class="big">{_fmt(agg['overall_stats']['stdev'])}</div><div class="label">Std. dev across group</div></div>
    <div class="card"><div class="big">{n}</div><div class="label">Contributors aggregated</div></div>
  </div>

  <div class="panel">
    <h2>Overall score distribution</h2>
    <div id="overall-chart" class="plot"></div>
    <p class="caption">One bar per contributor (anonymized), sorted by the order exports were
    supplied on the command line. This shows spread, not a ranking meant to be shared back
    to identifiable individuals.</p>
  </div>

  <div class="panel">
    <h2>Axis scores per contributor</h2>
    <div id="axis-chart" class="plot"></div>
    <table>
      <thead><tr><th>Axis</th><th class="num">Mean</th><th class="num">Median</th><th class="num">Std dev</th><th class="num">Min</th><th class="num">Max</th></tr></thead>
      <tbody>{stats_rows}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Weakest axis, by frequency across the group</h2>
    <div id="weakest-chart" class="plot"></div>
    <p class="caption">For each contributor, the axis with the lowest score is counted once.
    A tall bar means that axis is the group's most common bottleneck -- the best candidate
    for a shared team-wide recommendation or skill.</p>
  </div>

  <div class="panel">
    <h2>Aggregate anomaly flags</h2>
    <div id="anomaly-chart" class="plot"></div>
    <p class="caption">Counts of "watch" and "flagged" statistical anomalies (see Appendix A.7)
    across all contributors' most recent run, grouped by metric. A metric flagged for several
    people in the same run often points at something environmental (a shared tool change, a
    stressful period) rather than an individual habit.</p>
  </div>

  {f'<div class="panel"><h2>Distance to optimal (Mahalanobis)</h2>{mahal_note}</div>' if mahal_note else ''}

  <div class="footer">
    prompt-optimizer team aggregate - built from {n} locally-generated JSON exports -
    no raw prompt or reply text was ever included in those exports.
  </div>
</div>
<script>
Plotly.newPlot('overall-chart', [{{
  x: {labels_json}, y: {overall_json}, type: 'bar',
  marker: {{ color: '#6ee7b7' }}
}}], {{
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ color: '#e8eaed' }}, margin: {{ t: 10, l: 40, r: 10, b: 60 }},
  yaxis: {{ range: [0, 100], gridcolor: '#262b38' }}, xaxis: {{ gridcolor: '#262b38' }}
}}, {{displayModeBar: false, responsive: true}});

Plotly.newPlot('axis-chart', [
{axis_bar_traces}
], {{
  barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ color: '#e8eaed' }}, margin: {{ t: 10, l: 40, r: 10, b: 60 }},
  legend: {{ orientation: 'h', y: -0.2 }},
  yaxis: {{ range: [0, 100], gridcolor: '#262b38' }}, xaxis: {{ gridcolor: '#262b38' }}
}}, {{displayModeBar: false, responsive: true}});

Plotly.newPlot('weakest-chart', [{{
  x: {weakest_labels}, y: {weakest_values}, type: 'bar',
  marker: {{ color: '#f87171' }}
}}], {{
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ color: '#e8eaed' }}, margin: {{ t: 10, l: 40, r: 10, b: 60 }},
  yaxis: {{ gridcolor: '#262b38', dtick: 1 }}, xaxis: {{ gridcolor: '#262b38' }}
}}, {{displayModeBar: false, responsive: true}});

Plotly.newPlot('anomaly-chart', [
  {{ x: {anomaly_labels}, y: {anomaly_watch}, name: 'Watch', type: 'bar', marker: {{ color: '#fbbf24' }} }},
  {{ x: {anomaly_labels}, y: {anomaly_flagged}, name: 'Flagged', type: 'bar', marker: {{ color: '#f87171' }} }}
], {{
  barmode: 'stack', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ color: '#e8eaed' }}, margin: {{ t: 10, l: 40, r: 10, b: 110 }},
  legend: {{ orientation: 'h', y: -0.35 }},
  yaxis: {{ gridcolor: '#262b38', dtick: 1 }}, xaxis: {{ gridcolor: '#262b38' }}
}}, {{displayModeBar: false, responsive: true}});
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="+", help="Exported JSON files from bin/analyze.py (one per contributor)")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: out/team-aggregate-<timestamp>.html)")
    parser.add_argument("--open", action="store_true", help="Open the report in a browser after generating")
    args = parser.parse_args()

    cfg = load_config()["team_mode"]

    paths = [Path(p) for p in args.exports]
    payloads = load_exports(paths)

    if len(payloads) < cfg["min_users_for_aggregate"]:
        print(
            f"error: {len(payloads)} valid export(s) supplied, but "
            f"team_mode.min_users_for_aggregate={cfg['min_users_for_aggregate']} in config.yaml. "
            f"An aggregate of fewer contributors than this would not be meaningfully anonymous.",
            file=sys.stderr,
        )
        return 1

    agg = build_aggregate(payloads, anonymize=cfg["anonymize_labels"])
    generated_at = datetime.now(timezone.utc).isoformat()
    html = render_html(agg, generated_at)

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_dir = ROOT / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"team-aggregate-{ts}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote {out_path} ({len(payloads)} contributors aggregated)")

    if args.open:
        webbrowser.open(f"file://{out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
