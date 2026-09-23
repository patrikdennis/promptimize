#!/usr/bin/env python3
"""
prompt-optimizer: opt-in "Hiscores" comparison.

This is an *addition* to `bin/aggregate_team.py`, not a replacement for it:
`aggregate_team.py` rolls up score distributions/weakest axes/anomaly and
Mahalanobis statistics across a group's exported JSON files; this script
instead ranks the same (or a different) set of exports by their Leveling-tab
data -- Prompt Level, Total level, and each of the 7 skill levels -- in a
leaderboard-style table. Run either script, both, or neither; they
read the same export JSON and don't depend on each other.

Every contributor's export was produced entirely locally by `bin/analyze.py`
(see lib/export.py) and contains only numeric levels/XP, never raw prompt or
reply text. Levels are directly comparable across people because every
export was computed with the exact same deterministic XP rules and level
curve (lib/skills.py) -- there is no live/centralized server involved; you
compare by collecting everyone's exported JSON yourselves (e.g. in a shared
folder or a Slack thread) and running this script locally, the same opt-in
pattern as `aggregate_team.py`.

Usage:
    python3 bin/hiscores.py out/*.json
    python3 bin/hiscores.py alice.json bob.json carol.json --open
    python3 bin/hiscores.py out/*.json --out ~/Desktop/hiscores.html

Contributors are labeled "Contributor 1", "Contributor 2", ... in the order
their files were given on the command line, controlled by the same
team_mode.anonymize_labels config flag used by aggregate_team.py. A minimum
of team_mode.min_users_for_aggregate exports is required for the same reason
(an "aggregate"/leaderboard of one person is not meaningfully anonymous).
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from config import load_config  # noqa: E402
from skills import SKILL_DEFINITIONS  # noqa: E402

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"


def load_exports(paths: list[Path]) -> list[dict]:
    payloads = []
    for p in paths:
        try:
            payloads.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: skipping {p} ({e})", file=sys.stderr)
    return payloads


def build_hiscores(payloads: list[dict], anonymize: bool) -> list[dict]:
    """Returns one row per contributor who has a `skills` payload (older
    exports predating this feature simply won't have one, and are skipped
    with a warning), ranked by Total level descending (ties broken by
    Prompt Level)."""
    rows = []
    for i, p in enumerate(payloads):
        skills_payload = p.get("skills")
        if not skills_payload:
            print(
                f"warning: export #{i+1} has no 'skills' data (generated before the Leveling "
                f"feature, or run with --no-record before this feature's history was populated) "
                f"-- skipping it for the Hiscores table.",
                file=sys.stderr,
            )
            continue
        label = f"Contributor {i+1}" if anonymize else p.get("period", {}).get("label", f"Contributor {i+1}")
        rows.append({
            "label": label,
            "total_level": skills_payload["total_level"],
            "prompt_level": skills_payload["prompt_level"],
            "skills": skills_payload["skills"],
        })
    rows.sort(key=lambda r: (r["total_level"], r["prompt_level"]), reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows


def _fmt(x, digits=0):
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_html(rows: list[dict], generated_at: str) -> str:
    n = len(rows)
    skill_keys = [s["key"] for s in SKILL_DEFINITIONS]
    skill_labels = [s["label"] for s in SKILL_DEFINITIONS]

    header_cells = "".join(f"<th class='num'>{lbl}</th>" for lbl in skill_labels)

    def row_html(r: dict) -> str:
        skill_cells = "".join(
            f"<td class='num'>{r['skills'].get(k, {}).get('level', 'n/a')}</td>" for k in skill_keys
        )
        crown = ' class="rank-1"' if r["rank"] == 1 else ""
        return (
            f"<tr{crown}><td class='num'>{r['rank']}</td><td>{r['label']}</td>"
            f"<td class='num'>{r['total_level']}</td><td class='num'>{r['prompt_level']}</td>"
            f"{skill_cells}</tr>"
        )

    table_rows = "".join(row_html(r) for r in rows) or "<tr><td colspan='99' class='muted'>No contributors with Leveling data.</td></tr>"

    labels_json = json.dumps([r["label"] for r in rows])
    total_level_json = json.dumps([r["total_level"] for r in rows])
    prompt_level_json = json.dumps([r["prompt_level"] for r in rows])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Prompt-Optimizer Hiscores</title>
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
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 0 28px; overflow-x: auto; }}
  h1 {{ font-size: 24px; margin: 0 0 4px 0; }}
  h2 {{ margin-top:0; font-size: 17px; }}
  .sub {{ color: var(--muted); margin-bottom: 20px; font-size: 13.5px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 22px; min-width: 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 14px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
  .card .big {{ font-size: 26px; font-weight: 700; }}
  .card .label {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; }}
  table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align:left; padding: 7px 9px; border-bottom: 1px solid var(--border); vertical-align:top; white-space:nowrap; }}
  th {{ color: var(--muted); font-weight:600; text-transform:uppercase; font-size:10.5px; letter-spacing:.04em; }}
  th.num, td.num {{ text-align:center; }}
  tr.rank-1 td {{ color: var(--accent); font-weight: 700; }}
  .caption {{ color: var(--muted); font-size: 12.5px; margin-top: 8px; line-height:1.5; border-left: 2px solid var(--border); padding-left: 10px; }}
  .footer {{ color: var(--muted); font-size:12px; margin-top: 24px; padding: 0 28px;}}
  .plot {{ width: 100%; height: 380px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Prompt-Optimizer Hiscores</h1>
  <div class="sub">{n} contributors ranked - generated {generated_at}</div>

  <div class="panel">
    <h2>Total level and Prompt Level, by contributor</h2>
    <div id="level-chart" class="plot"></div>
    <p class="caption">Total level is the sum of all 7 skill levels (max 693). Prompt Level is a single
    weighted-average composite across the same 7 skills (config: skills.prompt_level_weights) -- analogous
    to a combat level. Both are computed identically for every contributor (lib/skills.py), so they are
    directly comparable across people, unlike Mahalanobis distance in the team-aggregate report (which is
    scaled per-person against their own score covariance).</p>
  </div>

  <div class="panel">
    <h2>Hiscores table</h2>
    <table>
      <thead><tr><th>Rank</th><th>Contributor</th><th class="num">Total level</th><th class="num">Prompt Level</th>{header_cells}</tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    <p class="caption">Sorted by Total level (ties broken by Prompt Level). Per-skill columns show each
    contributor's current level (1-99) in that skill; see the main report's Leveling tab, or README Appendix
    B, for what each skill measures and how XP is earned.</p>
  </div>

  <div class="footer">
    prompt-optimizer Hiscores - built from {n} locally-generated JSON exports, voluntarily shared -
    no raw prompt or reply text was ever included in those exports. This is a point-in-time snapshot
    comparison, not a live/centralized leaderboard: re-run this script with fresher exports to see movement.
  </div>
</div>
<script>
Plotly.newPlot('level-chart', [
  {{ x: {labels_json}, y: {total_level_json}, name: 'Total level', type: 'bar', marker: {{ color: '#6ee7b7' }} }},
  {{ x: {labels_json}, y: {prompt_level_json}, name: 'Prompt Level', type: 'bar', marker: {{ color: '#60a5fa' }} }}
], {{
  barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ color: '#e8eaed' }}, margin: {{ t: 10, l: 40, r: 10, b: 60 }},
  legend: {{ orientation: 'h', y: -0.2 }},
  yaxis: {{ gridcolor: '#262b38' }}, xaxis: {{ gridcolor: '#262b38' }}
}}, {{displayModeBar: false, responsive: true}});
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="+", help="Exported JSON files from bin/analyze.py (one per contributor)")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: out/hiscores-<timestamp>.html)")
    parser.add_argument("--open", action="store_true", help="Open the report in a browser after generating")
    args = parser.parse_args()

    cfg = load_config()["team_mode"]

    paths = [Path(p) for p in args.exports]
    payloads = load_exports(paths)

    if len(payloads) < cfg["min_users_for_aggregate"]:
        print(
            f"error: {len(payloads)} valid export(s) supplied, but "
            f"team_mode.min_users_for_aggregate={cfg['min_users_for_aggregate']} in config.yaml. "
            f"A leaderboard of fewer contributors than this would not be meaningfully anonymous.",
            file=sys.stderr,
        )
        return 1

    rows = build_hiscores(payloads, anonymize=cfg["anonymize_labels"])
    generated_at = datetime.now(timezone.utc).isoformat()
    html = render_html(rows, generated_at)

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_dir = ROOT / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"hiscores-{ts}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote {out_path} ({len(rows)} contributor(s) ranked)")

    if args.open:
        webbrowser.open(f"file://{out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
