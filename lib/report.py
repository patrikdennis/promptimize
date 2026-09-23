"""
Renders the self-contained, multi-tab HTML report (Plotly via CDN, no other
runtime dependencies). Every chart is followed by a caption explaining what
is plotted and how it was computed; every score has a linked definition in
the Methodology tab.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

from metrics import Aggregate, METRIC_DEFINITIONS
from recommendations import Recommendation
from history import ObjectiveProgress
from advanced_metrics import AdvancedMetrics, STATES
from config import load_config
from skills import SKILL_DEFINITIONS

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

AXIS_ORDER = ["specificity", "context", "structure", "efficiency"]
AXIS_LABEL = {
    "specificity": "Specificity",
    "context": "Context & rationale",
    "structure": "Structure & acceptance criteria",
    "efficiency": "Efficiency (corrections, clarifications, restatement)",
    "overall": "Overall",
}
AXIS_FORMULA = {
    "specificity": "100 x (0.55 x (1 - vague_rate) + 0.45 x share of turns with at least 6 words)",
    "context": "100 x (0.65 x file_ref_rate + 0.35 x rationale_rate)",
    "structure": "100 x (0.4 x share of turns with 2+ numbered/bulleted steps + 0.35 x acceptance_rate + 0.25 x (1 - multi_ask_rate))",
    "efficiency": "100 x (0.5 x (1 - correction_rate) + 0.25 x (1 - clarification_rate) + 0.25 x (1 - context_restatement_rate))",
}

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

STATUS_LABEL = {
    "achieved": "Achieved",
    "improving": "Improving",
    "no_change": "No material change",
    "regressed": "Regressed",
}


def _build_3d_path(agg: Aggregate, recs: list[Recommendation]):
    """Steepest-ascent path from the current position toward the optimal
    corner (100, 100, 100) across specificity / context / efficiency,
    applying recommendations in priority order."""
    axes = ["specificity", "context", "efficiency"]
    cur = {a: agg.scores[a] for a in axes}
    xs, ys, zs = [cur["specificity"]], [cur["context"]], [cur["efficiency"]]
    labels = ["Current position"]
    markers = ["S"]

    running = dict(cur)
    for i, r in enumerate(recs):
        if r.axis in running:
            running[r.axis] = min(100.0, running[r.axis] + r.impact)
            xs.append(running["specificity"])
            ys.append(running["context"])
            zs.append(running["efficiency"])
            labels.append(r.title)
            markers.append(str(len(markers)))

    return {
        "xs": xs, "ys": ys, "zs": zs, "labels": labels, "markers": markers,
        "optimal": {"x": 100, "y": 100, "z": 100},
    }


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _fmt_num(x: float | None, digits=1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_html(
    period_label: str,
    start,
    end,
    agg: Aggregate,
    recs: list[Recommendation],
    installed_skills: list[dict],
    agents_included: list[str],
    objective_progress: list[ObjectiveProgress],
    run_history: list[dict],
    previous_run: dict | None,
    adv: AdvancedMetrics,
    anomalies: list | None = None,
    mahalanobis_result=None,
    skill_progress: dict | None = None,
    total_level: int | None = None,
    prompt_level: int | None = None,
    prompt_level_before: int | None = None,
    skill_xp_gained: dict | None = None,
) -> str:
    anomalies = anomalies or []
    skill_progress = skill_progress or {}
    skill_xp_gained = skill_xp_gained or {}
    scores = agg.scores
    radar_axes = AXIS_ORDER
    radar_values = [scores[a] for a in radar_axes] + [scores[radar_axes[0]]]
    radar_labels = [AXIS_LABEL[a] for a in radar_axes] + [AXIS_LABEL[radar_axes[0]]]

    days_sorted = sorted(agg.by_day.items())
    day_labels = [d for d, _ in days_sorted]
    day_values = [v for _, v in days_sorted]

    bucket_order = ["1-5", "6-15", "16-40", "41-80", "80+"]
    bucket_values = [agg.word_count_buckets.get(b, 0) for b in bucket_order]

    session_bucket_order = ["1", "2-3", "4-7", "8-15", "16+"]
    session_bucket_values = [agg.session_turn_buckets.get(b, 0) for b in session_bucket_order]

    heat = [[0] * 24 for _ in range(7)]
    for f in agg.features:
        ts = f.turn.timestamp
        heat[ts.weekday()][ts.hour] += 1

    repo_items = agg.by_repo_cwd.most_common(10)
    repo_labels = [r for r, _ in repo_items]
    repo_values = [v for _, v in repo_items]

    agent_items = agg.by_agent.most_common()

    path3d = _build_3d_path(agg, recs)
    path3d_legend_rows = "".join(
        f"<tr><td class='num'><span class='task-check' style='display:inline-flex;width:22px;height:22px;font-size:11px;'>{m}</span></td><td>{lbl}</td></tr>"
        for m, lbl in zip(path3d["markers"], path3d["labels"])
    )

    # ---- Recommendations & tasks (detailed cards) --------------------------
    def rec_card(r: Recommendation, idx: int) -> str:
        target_str = ""
        if r.metric_key and r.target_value is not None and r.current_value is not None:
            target_str = f"<div class='kv'><span>Current</span><b>{_fmt_pct(r.current_value) if r.current_value <= 1 else _fmt_num(r.current_value)}</b></div><div class='kv'><span>Target</span><b>{_fmt_pct(r.target_value) if r.target_value <= 1 else _fmt_num(r.target_value)}</b></div>"
        return f"""
        <div class="rec-card">
          <div class="rec-head">
            <span class="rec-num">{idx}</span>
            <span class="rec-title">{r.title}</span>
            <span class="pill pill-{r.priority}">Priority {r.priority}</span>
            <span class="pill pill-axis">{AXIS_LABEL.get(r.axis, r.axis)}</span>
            <span class="pill pill-impact">Est. +{r.impact:.0f} pts</span>
          </div>
          <div class="rec-body">
            <div class="rec-col">
              <div class="rec-label">Evidence</div>
              <div class="rec-text">{r.evidence}</div>
            </div>
            <div class="rec-col">
              <div class="rec-label">Why this affects the agent's output</div>
              <div class="rec-text">{r.mechanism}</div>
            </div>
            <div class="rec-col">
              <div class="rec-label">Action</div>
              <div class="rec-text">{r.action}</div>
            </div>
          </div>
          <div class="rec-foot">
            <span>Confidence: {r.confidence}</span>
            {target_str}
          </div>
        </div>
        """

    rec_cards = "".join(rec_card(r, i + 1) for i, r in enumerate(recs)) or "<p class='muted'>No recommendations triggered for this period.</p>"

    # ---- Evidence & correlations tab ---------------------------------------
    def cond_rows(segment: str, stat: dict, with_label: str, without_label: str) -> str:
        w, wo = stat.get("with", {}), stat.get("without", {})
        if w.get("n", 0) == 0 and wo.get("n", 0) == 0:
            return ""
        return f"""
            <tr class="segment-first"><td>{segment}</td><td>{with_label}</td><td class="num">{_fmt_pct(w.get('correction_rate'))}</td><td class="num">{_fmt_pct(w.get('clarification_rate'))}</td><td class="num">{w.get('n', 0)}</td></tr>
            <tr><td></td><td>{without_label}</td><td class="num">{_fmt_pct(wo.get('correction_rate'))}</td><td class="num">{_fmt_pct(wo.get('clarification_rate'))}</td><td class="num">{wo.get('n', 0)}</td></tr>
        """

    cs = agg.conditional_stats
    evidence_body = "".join([
        cond_rows("File/code reference", cs.get("correction_by_file_ref", {}), "With reference", "Without reference"),
        cond_rows("Vague language", cs.get("correction_by_vague", {}), "Vague", "Not vague"),
        cond_rows("Stated rationale (why)", cs.get("correction_by_rationale", {}), "Rationale stated", "No rationale"),
        cond_rows("Stated acceptance criteria", cs.get("correction_by_acceptance", {}), "Stated", "Not stated"),
        cond_rows("2+ action verbs bundled", cs.get("correction_by_multi_ask", {}), "Bundled", "Single-focus"),
    ])
    evidence_tables = (
        f"""
        <table class="evidence-table">
          <thead><tr><th>Segment</th><th>Group</th><th class="num">Correction rate</th><th class="num">Clarification rate</th><th class="num">n</th></tr></thead>
          <tbody>{evidence_body}</tbody>
        </table>
        """
        if evidence_body else ""
    )

    # ---- Methodology tab: metric definitions --------------------------------
    def metric_def_row(key: str) -> str:
        d = METRIC_DEFINITIONS.get(key)
        if not d:
            return ""
        current = agg.metric_values.get(key)
        current_str = _fmt_pct(current) if isinstance(current, float) and current <= 1 else _fmt_num(current)
        return f"""
        <div class="metric-def">
          <div class="metric-def-head"><span class="metric-def-label">{d['label']}</span><span class="metric-def-current">Current: {current_str}</span></div>
          <div class="metric-def-formula"><code>{d['formula']}</code></div>
          <div class="metric-def-detail">{d['detail']}</div>
          <div class="metric-def-why"><b>Why it matters:</b> {d['why_it_matters']}</div>
        </div>
        """

    metric_defs_html = "".join(metric_def_row(k) for k in METRIC_DEFINITIONS)

    # ---- Progress over time tab ---------------------------------------------
    hist_labels = [r["run_at"][:16].replace("T", " ") for r in run_history]
    hist_overall = [r["score_overall"] for r in run_history]
    hist_spec = [r["score_specificity"] for r in run_history]
    hist_ctx = [r["score_context"] for r in run_history]
    hist_struct = [r["score_structure"] for r in run_history]
    hist_eff = [r["score_efficiency"] for r in run_history]

    def objective_row(o: ObjectiveProgress) -> str:
        def fmt(v):
            return f"{v:.1%}" if abs(v) <= 1.5 else f"{v:.1f}"
        return f"""
        <tr class="status-{o.status}">
          <td>{o.title}</td>
          <td>{fmt(o.baseline_value)}</td>
          <td>{fmt(o.current_value)}</td>
          <td>{o.relative_change_pct:+.1f}%</td>
          <td><span class="status-pill status-pill-{o.status}">{STATUS_LABEL.get(o.status, o.status)}</span></td>
        </tr>
        """

    objectives_html = "".join(objective_row(o) for o in objective_progress)
    has_history = len(run_history) > 1
    has_objectives = len(objective_progress) > 0

    if previous_run:
        prev_summary = (
            f"Previous report: {previous_run['period_label']} (run at {previous_run['run_at'][:16].replace('T', ' ')}), "
            f"overall score {previous_run['score_overall']:.0f}/100."
        )
        delta_overall = scores.get("overall", 0) - previous_run["score_overall"]
        prev_summary += f" Change since then: {delta_overall:+.1f} points."
    else:
        prev_summary = "No prior report uses the current scoring model, so there is no comparable baseline yet. Run this analysis again later to start tracking progress."

    skills_rows = "".join(
        f"<tr><td>{s['name']}</td><td>{s['location']}</td><td class='detail'>{s['description']}</td></tr>"
        for s in installed_skills
    ) or "<tr><td colspan='3' class='detail'>No personal skills installed yet.</td></tr>"

    def skill_card(key: str) -> str:
        sp = skill_progress.get(key)
        if sp is None:
            return ""
        gained = skill_xp_gained.get(key, 0.0)
        gained_html = f'<div class="skill-gained">+{gained:.0f} XP this run</div>' if gained else ""
        if sp.is_maxed:
            progress_label = f"MAX LEVEL &mdash; {sp.xp:,.0f} XP total"
        else:
            progress_label = f"{sp.xp_into_level:,.0f} / {sp.xp_for_next_level:,.0f} XP to level {sp.level + 1}"
        return f"""
        <div class="skill-card{' skill-maxed' if sp.is_maxed else ''}">
          <div class="skill-head">
            <span class="skill-level">{sp.level}</span>
            <span class="skill-name">{sp.label}</span>
          </div>
          <div class="skill-bar-track">
            <div class="skill-bar-fill" style="width:{sp.progress_pct:.1f}%"></div>
          </div>
          <div class="skill-progress-label">{progress_label}</div>
          {gained_html}
          <div class="skill-desc">{sp.description}</div>
        </div>
        """

    skill_cards_html = "".join(skill_card(s["key"]) for s in SKILL_DEFINITIONS)
    prompt_level_delta = (
        (prompt_level - prompt_level_before) if (prompt_level is not None and prompt_level_before is not None) else 0
    )
    prompt_level_delta_html = (
        f'<div class="skill-gained">+{prompt_level_delta} this run</div>' if prompt_level_delta > 0 else ""
    )

    def task_item(r: Recommendation, idx: int) -> str:
        return f"""
        <div class="task-item">
          <div class="task-check">{idx}</div>
          <div class="task-body">
            <div class="task-title-row">
              <span class="task-title">{r.title}</span>
              <span class="pill pill-{r.priority}">P{r.priority}</span>
              <span class="pill pill-axis">{AXIS_LABEL.get(r.axis, r.axis)}</span>
            </div>
            <div class="task-action">{r.action}</div>
          </div>
        </div>
        """

    task_items = "".join(task_item(r, i + 1) for i, r in enumerate(recs[:8])) or "<p class='muted'>No action items triggered for this period.</p>"

    agent_badges = " ".join(f"<span class='badge'>{a}</span>" for a in agents_included)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- Advanced Analytics tab data ---------------------------------------
    markov = adv.markov
    markov_states = STATES
    markov_values = [markov.expected_turns_to_resolution[s] for s in markov_states] if markov else []
    markov_available = markov is not None

    windings = adv.windings
    winding_values = [w.winding_number for w in windings]
    winding_abs_mean = (sum(abs(w) for w in winding_values) / len(winding_values)) if winding_values else None
    winding_circ_threshold = load_config()["advanced_metrics"]["winding_circular_threshold"]
    circular_sessions = [w for w in windings if abs(w.winding_number) >= winding_circ_threshold]
    circular_count = len(circular_sessions)
    winding_available = len(windings) >= 3

    burst = adv.burstiness
    comp = adv.compression
    heaps = adv.heaps

    heaps_fit_x, heaps_fit_y = [], []
    if heaps:
        xs_plot = [10 * (1.4 ** i) for i in range(0, 18)]
        xs_plot = [x for x in xs_plot if x <= heaps.n_tokens]
        heaps_fit_x = xs_plot
        heaps_fit_y = [heaps.k * (x ** heaps.beta) for x in xs_plot]

    def fmt_signed(v: float, digits=3) -> str:
        return f"{v:+.{digits}f}"

    markov_js = ""
    if markov_available:
        markov_js = (
            "Plotly.newPlot('markovchart', [{"
            f"type:'bar', x: {json.dumps(markov_states)}, y: {json.dumps(markov_values)}, "
            "marker:{color:['#60a5fa','#fbbf24','#c084fc','#f87171']}"
            "}], Object.assign({}, dark, { xaxis:Object.assign({}, gridAxis, {title:'Starting state'}), "
            "yaxis:Object.assign({}, gridAxis, {title:'Expected turns to resolution'}), "
            "margin:{t:24,r:24,b:48,l:56} }), {displayModeBar:false, responsive:true});"
        )

    winding_js = ""
    if winding_available:
        winding_js = (
            "Plotly.newPlot('windingchart', [{"
            f"type:'histogram', x: {json.dumps(winding_values)}, marker:{{color:'#c084fc'}}, xbins:{{size:0.1}}"
            "}], Object.assign({}, dark, { xaxis:Object.assign({}, gridAxis, {title:'Winding number W per session'}), "
            "yaxis:Object.assign({}, gridAxis, {title:'Sessions'}), "
            f"shapes:[{{type:'line', x0:{winding_circ_threshold}, x1:{winding_circ_threshold}, y0:0, y1:1, yref:'paper', line:{{color:'#f87171', dash:'dot'}}}}, "
            f"{{type:'line', x0:{-winding_circ_threshold}, x1:{-winding_circ_threshold}, y0:0, y1:1, yref:'paper', line:{{color:'#f87171', dash:'dot'}}}}], "
            "margin:{t:24,r:24,b:48,l:56} }), {displayModeBar:false, responsive:true});"
        )

    burst_js = ""
    if burst:
        b_color = "#f87171" if burst.burstiness > 0.15 else ("#60a5fa" if burst.burstiness < -0.15 else "#6ee7b7")
        burst_js = (
            "Plotly.newPlot('burstchart', ["
            f"{{type:'bar', orientation:'h', x:[{burst.burstiness}], y:['Burstiness B'], "
            f"marker:{{color:'{b_color}'}}, showlegend:false}}, "
            f"{{type:'bar', orientation:'h', x:[{burst.memory}], y:['Memory M'], "
            "marker:{color:'#fbbf24'}, showlegend:false}"
            "], Object.assign({}, dark, { xaxis:Object.assign({}, gridAxis, {title:'Value', range:[-1,1]}), "
            "yaxis:Object.assign({}, gridAxis), "
            "shapes:[{type:'line', x0:0, x1:0, y0:-0.5, y1:1.5, line:{color:'#4b5568', dash:'dot'}}], "
            "margin:{t:24,r:24,b:40,l:90} }), {displayModeBar:false, responsive:true});"
        )

    heaps_js = ""
    if heaps:
        heaps_js = (
            "Plotly.newPlot('heapschart', [{"
            "type:'scatter', mode:'lines+markers', name:'Fitted curve V = K\u00b7n^\u03b2', "
            f"x:{json.dumps(heaps_fit_x)}, y:{json.dumps(heaps_fit_y)}, "
            "marker:{color:'#6ee7b7', size:7}, line:{color:'#6ee7b7'}"
            "}], Object.assign({}, dark, { xaxis:Object.assign({}, gridAxis, "
            "{title:'Corpus length n (tokens, log scale)', type:'log'}), "
            "yaxis:Object.assign({}, gridAxis, {title:'Distinct vocabulary V(n), log scale', type:'log'}), "
            "margin:{t:24,r:24,b:48,l:64} }), {displayModeBar:false, responsive:true});"
        )

    # ---- Mahalanobis distance-to-optimal (Overview tab) --------------------
    def mahalanobis_block() -> str:
        if mahalanobis_result is None:
            return (
                "<p class='muted'>Needs at least a few prior recorded runs to estimate how your own axis "
                "scores co-vary before a Mahalanobis distance is statistically meaningful. Showing the plain "
                "Euclidean distance below in the meantime.</p>"
                f"<div class='stat-card'><div class='stat-value'>{math.sqrt(sum((100-agg.scores.get(a,0))**2 for a in AXIS_ORDER)):.1f}</div>"
                "<div class='stat-label'>Euclidean distance to optimal (unweighted, ignores correlation between axes)</div></div>"
            )
        m = mahalanobis_result
        pct_tighter = (1 - m.distance / m.euclidean_distance) * 100 if m.euclidean_distance > 0 else 0.0
        return f"""
        <div class="row2">
          <div class='stat-card'><div class='stat-value'>{m.distance:.2f}</div><div class='stat-label'>Mahalanobis distance to optimal (covariance-weighted, {m.n_history_runs} prior runs)</div></div>
          <div class='stat-card'><div class='stat-value'>{m.euclidean_distance:.1f}</div><div class='stat-label'>Plain Euclidean distance (unweighted, for comparison)</div></div>
        </div>
        <div class="caption">
          The Euclidean distance treats a point gap on any axis as equally significant and ignores that your
          own axes tend to move together. The Mahalanobis distance
          D<sub>M</sub>(x) = &radic;((x&minus;&mu;)<sup>T</sup> &Sigma;<sup>&minus;1</sup> (x&minus;&mu;)) rescales
          the gap by the inverse of your own historical covariance matrix &Sigma;, estimated from
          {m.n_history_runs} previously recorded runs, so an unusual *combination* of axis values counts for
          more than the same total gap spread across axes that normally move in lock-step. See README Appendix
          A.6 for the derivation. {"This run's position is about " + f"{pct_tighter:.0f}% closer, in covariance-adjusted terms, than the raw Euclidean number suggests." if pct_tighter > 1 else ("This run's position is about " + f"{abs(pct_tighter):.0f}% farther, in covariance-adjusted terms, than the raw Euclidean number suggests." if pct_tighter < -1 else "The two measures roughly agree for this run.")}
        </div>
        """

    mahalanobis_html = mahalanobis_block()

    # ---- Anomaly detection (Progress Over Time tab) -------------------------
    def anomaly_row(a) -> str:
        badge_class = {"flagged": "status-pill-regressed", "watch": "status-pill-no_change", "normal": "status-pill-achieved"}[a.status]
        direction_note = "concerning direction" if a.concerning and a.status != "normal" else ("favorable direction" if a.status != "normal" else "&mdash;")
        return f"""
        <tr>
          <td>{a.label}</td>
          <td class="num">{a.current_value:.3f}</td>
          <td class="num">{a.historical_mean:.3f} &plusmn; {a.historical_std:.3f}</td>
          <td class="num">{a.z_score:+.2f}</td>
          <td class="num">{a.n_history}</td>
          <td><span class="status-pill {badge_class}">{a.status.upper()}</span></td>
          <td>{direction_note}</td>
        </tr>
        """

    anomaly_rows_html = "".join(anomaly_row(a) for a in anomalies)
    flagged_count = sum(1 for a in anomalies if a.status == "flagged")
    watch_count = sum(1 for a in anomalies if a.status == "watch")
    _anom_cfg = load_config()["anomaly_detection"]
    anomaly_watch_z = _anom_cfg["z_score_watch_threshold"]
    anomaly_flag_z = _anom_cfg["z_score_flag_threshold"]
    anomaly_min_runs = _anom_cfg["min_history_runs"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Prompt Optimization Report - {period_label}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{
    --bg: #0f1115; --panel: #171a21; --panel2: #1d212b; --text: #e8eaed;
    --muted: #9aa3b2; --accent: #6ee7b7; --accent2: #60a5fa; --warn: #fbbf24; --bad: #f87171;
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
  .badge {{ display:inline-block; background:#26304a; color:var(--accent2); border-radius:999px; padding:2px 10px; font-size:12px; margin-right:6px;}}

  nav.tabs {{
    position: sticky; top: 0; z-index: 10; background: #12141a; border-bottom: 1px solid var(--border);
    display:flex; gap: 4px; padding: 0 28px; overflow-x:auto;
  }}
  nav.tabs button {{
    background: none; border: none; color: var(--muted); padding: 14px 16px; font-size: 13.5px;
    cursor:pointer; border-bottom: 2px solid transparent; white-space:nowrap;
  }}
  nav.tabs button.active {{ color: var(--text); border-bottom: 2px solid var(--accent2); }}
  nav.tabs button:hover {{ color: var(--text); }}

  .tabpanel {{ display:none; }}
  .tabpanel.active {{ display:block; }}

  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 14px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
  .card .big {{ font-size: 26px; font-weight: 700; }}
  .card .label {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; }}
  .score-overall {{ color: var(--accent); }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 22px; min-width: 0; overflow: hidden; }}
  .row2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 18px; min-width: 0; }}
  table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align:left; padding: 7px 9px; border-bottom: 1px solid var(--border); vertical-align:top; }}
  th {{ color: var(--muted); font-weight:600; text-transform:uppercase; font-size:10.5px; letter-spacing:.04em; }}
  th.num, td.num {{ text-align:center; }}
  .evidence-table td:first-child {{ color: var(--text); font-weight:600; }}
  .evidence-table tr.segment-first td {{ border-top: 1px solid var(--border); padding-top: 12px; }}
  .evidence-table tr:not(.segment-first) td {{ border-bottom: 1px solid var(--border); }}
  .evidence-table td, .evidence-table th {{ width: 20%; }}
  .evidence-table td:first-child, .evidence-table th:first-child {{ width: 22%; }}
  td.detail {{ color: #cbd3e1; }}
  p.detail {{ color: #cbd3e1; font-size: 12.5px; margin-top: 10px; }}
  .muted {{ color: var(--muted); }}
  .caption {{ color: var(--muted); font-size: 12.5px; margin-top: 8px; line-height:1.5; border-left: 2px solid var(--border); padding-left: 10px; }}
  .stat-card {{ background: #14161d; border: 1px solid var(--border); border-radius: 10px; padding: 22px; text-align:center; margin-top: 6px; }}
  .stat-value {{ font-size: 40px; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }}
  .stat-label {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
  .task-list {{ display:flex; flex-direction:column; gap:10px; }}
  .task-item {{ display:flex; gap:14px; align-items:flex-start; background:#14161d; border:1px solid var(--border); border-radius:10px; padding:12px 14px; }}
  .task-check {{ flex:0 0 auto; width:26px; height:26px; border-radius:50%; background:#26304a; color:var(--accent2); font-size:12.5px; font-weight:700; display:flex; align-items:center; justify-content:center; }}
  .task-body {{ flex:1; min-width:0; }}
  .task-title-row {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
  .task-title {{ font-weight:600; font-size:14px; }}
  .task-action {{ color: #cbd3e1; font-size:13px; margin-top:5px; line-height:1.5; }}
  .footer {{ color: var(--muted); font-size:12px; margin-top: 24px; padding: 0 28px;}}
  @media (max-width: 900px) {{ .row2 {{ grid-template-columns: 1fr; }} }}

  .rec-card {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }}
  .rec-head {{ display:flex; align-items:center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
  .rec-num {{ background:#2a3040; color:var(--accent2); border-radius:999px; width:22px; height:22px; display:inline-flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; }}
  .rec-title {{ font-weight:600; font-size:14.5px; flex: 1 1 260px; }}
  .pill {{ font-size: 11px; padding: 2px 9px; border-radius: 999px; background:#26304a; color:var(--muted); }}
  .pill-1 {{ background:#3a2230; color:#f87171; }}
  .pill-2 {{ background:#3a3320; color:var(--warn); }}
  .pill-3 {{ background:#233a2e; color:var(--accent); }}
  .pill-4 {{ background:#20303a; color:var(--accent2); }}
  .pill-axis {{ background:#232838; }}
  .pill-impact {{ background:#1e2a1e; color:var(--accent); }}
  .rec-body {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .rec-label {{ font-size: 10.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-bottom: 4px; }}
  .rec-text {{ font-size: 13px; line-height: 1.5; color: #d7dbe4; }}
  .rec-foot {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); display:flex; gap:20px; flex-wrap:wrap; }}
  .rec-foot .kv {{ display:flex; gap: 6px; }}
  .rec-foot .kv span {{ color: var(--muted); }}
  .rec-foot .kv b {{ color: var(--text); }}
  @media (max-width: 900px) {{ .rec-body {{ grid-template-columns: 1fr; }} }}

  .metric-def {{ border-bottom: 1px solid var(--border); padding: 14px 0; }}
  .metric-def:last-child {{ border-bottom: none; }}
  .metric-def-head {{ display:flex; justify-content: space-between; align-items:baseline; }}
  .metric-def-label {{ font-weight: 600; font-size: 14.5px; }}
  .metric-def-current {{ color: var(--accent2); font-size: 13px; }}
  .metric-def-formula {{ margin: 6px 0; color: var(--accent); font-size: 12.5px; }}
  .metric-def-formula code {{ background:#0d0f14; padding: 3px 8px; border-radius: 6px; }}
  .metric-def-detail {{ color: #cbd3e1; font-size: 13px; line-height: 1.5; margin-bottom: 4px; }}
  .metric-def-why {{ color: var(--muted); font-size: 12.5px; line-height: 1.5; }}

  .status-pill {{ padding: 2px 9px; border-radius: 999px; font-size: 11.5px; }}
  .status-pill-achieved {{ background:#1e2a1e; color:var(--accent); }}
  .status-pill-improving {{ background:#20303a; color:var(--accent2); }}
  .status-pill-no_change {{ background:#2a2a2a; color:var(--muted); }}
  .status-pill-regressed {{ background:#3a2230; color:var(--bad); }}

  .optimal-explainer {{ display:grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .optimal-explainer ul {{ margin: 6px 0 0 18px; padding:0; font-size: 13px; line-height: 1.7; color:#d7dbe4;}}

  .level-hero {{ text-align:center; }}
  .level-hero-num {{ color: var(--accent); font-size: 44px; }}
  .skill-gained {{ color: var(--accent); font-size: 11.5px; font-weight:700; margin-top: 4px; }}
  .skill-grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; margin-top: 12px; }}
  .skill-card {{ background:#1c202b; border:1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .skill-card.skill-maxed {{ border-color: var(--accent); box-shadow: 0 0 0 1px rgba(110,231,183,0.25) inset; }}
  .skill-head {{ display:flex; align-items:center; gap:10px; margin-bottom: 8px; }}
  .skill-level {{ flex:0 0 auto; width:32px; height:32px; border-radius:8px; background:#26304a; color:var(--accent2); font-size:14px; font-weight:700; display:flex; align-items:center; justify-content:center; }}
  .skill-maxed .skill-level {{ background:#1e2a1e; color:var(--accent); }}
  .skill-name {{ font-size: 14px; font-weight:600; }}
  .skill-bar-track {{ background:#12141a; border-radius:999px; height:8px; overflow:hidden; }}
  .skill-bar-fill {{ background: linear-gradient(90deg, var(--accent2), var(--accent)); height:100%; }}
  .skill-progress-label {{ color: var(--muted); font-size: 11px; margin-top: 6px; font-variant-numeric: tabular-nums; }}
  .skill-desc {{ color: var(--muted); font-size: 12px; line-height:1.5; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Prompt Optimization Report</h1>
    <div class="sub">Period: <b>{period_label}</b> ({start} to {end}) &middot; Generated {generated_at} &middot; Agents included: {agent_badges or '<span class="badge">none</span>'}</div>
  </div>

  <nav class="tabs" id="tabnav">
    <button data-tab="overview" class="active">Overview</button>
    <button data-tab="metrics">Metric Deep-Dive</button>
    <button data-tab="evidence">Evidence &amp; Correlations</button>
    <button data-tab="recommendations">Recommendations &amp; Tasks</button>
    <button data-tab="progress">Progress Over Time</button>
    <button data-tab="advanced">Advanced Analytics</button>
    <button data-tab="skills">Skills &amp; Capabilities</button>
    <button data-tab="leveling">Leveling</button>
    <button data-tab="methodology">Methodology</button>
  </nav>

  <div class="wrap">

    <div class="tabpanel active" id="tab-overview">
      <div class="grid">
        <div class="card"><div class="big score-overall">{scores.get('overall',0):.0f}</div><div class="label">Overall score (0-100, equal-weighted average of the 4 axes below)</div></div>
        <div class="card"><div class="big">{agg.total_prompts}</div><div class="label">Turns analyzed</div></div>
        <div class="card"><div class="big">{agg.total_sessions}</div><div class="label">Sessions</div></div>
        <div class="card"><div class="big">{agg.avg_word_count:.1f}</div><div class="label">Avg prompt length (words)</div></div>
        <div class="card"><div class="big">{agg.correction_rate:.0%}</div><div class="label">Correction rate</div></div>
        <div class="card"><div class="big">{agg.clarification_rate:.0%}</div><div class="label">Clarification-loop rate</div></div>
      </div>

      <div class="panel">
        <h2>Position vs. the optimal point (3D)</h2>
        <div id="path3d" style="height:520px; width:100%;" data-plotly></div>
        <div class="caption">
          Axes: Specificity, Context & rationale, Efficiency (see Methodology tab for exact formulas). The gold
          diamond marks the optimal corner (100, 100, 100) &mdash; the point where every underlying rate is at
          its theoretical best (0% vague language, 100% of turns anchored to a file/reference, 0% corrections/
          clarifications/restatement). It is a mathematical ceiling defined by how the metrics are constructed,
          not an arbitrary target: each axis is already scaled to [0, 100], so 100 is reachable only if every
          contributing rate hits its ideal value simultaneously. The blue line is a steepest-ascent path: each
          numbered step applies one recommendation from the Recommendations tab, in priority order, adding its
          estimated point gain to the axis it targets. Hover a point on the chart for its full detail; the
          numbers are matched to the list below since inline 3D text labels become unreadable once several
          points are close together.
        </div>
        {f"<table style='margin-top:14px; max-width:520px;'><thead><tr><th style='width:40px;'>Step</th><th>Recommendation applied</th></tr></thead><tbody>{path3d_legend_rows}</tbody></table>" if len(path3d['labels']) > 1 else ""}
      </div>

      <div class="panel">
        <h2>Distance to optimal: covariance-adjusted (Mahalanobis)</h2>
        {mahalanobis_html}
      </div>

      <div class="row2">
        <div class="panel">
          <h2>Score profile</h2>
          <div id="radar" style="height:360px; width:100%;" data-plotly></div>
          <div class="caption">Each axis is a weighted combination of measured rates; see the Methodology tab for the exact formula behind each one.</div>
        </div>
        <div class="panel">
          <h2>What "optimal" means here</h2>
          <div class="optimal-explainer">
            <div>
              <b>The point (100, 100, 100)</b> is not a subjective ideal &mdash; it is what the score formulas
              evaluate to when every input rate is at its best possible value:
              <ul>
                <li>0% of turns contain vague/hedging language</li>
                <li>100% of turns anchor to a specific file, symbol, or code reference</li>
                <li>0% of turns require a correction, a clarification loop, or restate prior context</li>
              </ul>
            </div>
            <div>
              No real workflow sustains exactly 0%/100% on every axis indefinitely &mdash; genuinely novel or
              exploratory work will always cost some clarification and iteration. The chart is a navigational
              aid: it shows direction and estimated magnitude of improvement, not a pass/fail bar.
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Progress since your last report</h2>
        <div class="caption" style="margin-top:0; margin-bottom: 10px;">{prev_summary}</div>
      </div>
    </div>

    <div class="tabpanel" id="tab-metrics">
      <div class="row2">
        <div class="panel">
          <h2>Turns per day</h2>
          <div id="perday" style="height:340px; width:100%;" data-plotly></div>
          <div class="caption">Count of analyzed turns bucketed by calendar day (UTC) within the selected period.</div>
        </div>
        <div class="panel">
          <h2>Prompt length distribution</h2>
          <div id="lenbuckets" style="height:340px; width:100%;" data-plotly></div>
          <div class="caption">Turns bucketed by word count. The specificity axis gives length credit to any turn with at least 6 words; it deliberately has no upper ceiling, because useful context, rationale, examples, and constraints should not be penalized for making a prompt longer. Long multi-part requests are best made easier to verify with numbered or bulleted steps, rather than shortened arbitrarily.</div>
        </div>
      </div>
      <div class="row2">
        <div class="panel">
          <h2>When you prompt (day x hour)</h2>
          <div id="heatmap" style="height:340px; width:100%;" data-plotly></div>
          <div class="caption">Turn counts by day-of-week and hour-of-day (UTC). Useful for noticing whether corrections/clarifications cluster at certain times (e.g. late-session fatigue).</div>
        </div>
        <div class="panel">
          <h2>Session length distribution</h2>
          <div id="sessionbuckets" style="height:340px; width:100%;" data-plotly></div>
          <div class="caption">Sessions bucketed by number of turns. A "1" session resolved in a single exchange; see single_shot_rate in Methodology for how this feeds the efficiency picture.</div>
        </div>
      </div>
      <div class="panel">
        <h2>Top projects / working directories</h2>
        <div id="repos" style="height:360px; width:100%;" data-plotly></div>
        <div class="caption">Turn counts grouped by git repository (or working directory if no repository was recorded).</div>
      </div>
    </div>

    <div class="tabpanel" id="tab-evidence">
      <div class="panel">
        <h2>Segmented correction and clarification rates</h2>
        <div class="caption" style="margin-top:0; margin-bottom:14px;">
          Each table below splits your own turns into two groups by one property, then reports the correction
          rate and clarification-loop rate measured within each group. This is the quantitative basis cited by
          the corresponding recommendations &mdash; not an external benchmark, but your own transcript data,
          segmented. Groups with fewer than 5 turns are omitted as unreliable.
        </div>
        {evidence_tables or "<p class='muted'>Not enough turns in this period to compute segmented statistics reliably.</p>"}
      </div>
    </div>

    <div class="tabpanel" id="tab-recommendations">
      <div class="panel">
        <h2>Top tasks for this period</h2>
        <div class="caption" style="margin-top:0; margin-bottom:14px;">The highest-priority actions from the recommendations below, ordered so the largest estimated score gains come first.</div>
        <div class="task-list">{task_items}</div>
      </div>
      <div class="panel">
        <h2>Full recommendation detail ({len(recs)} total)</h2>
        {rec_cards}
      </div>
    </div>

    <div class="tabpanel" id="tab-progress">
      <div class="panel">
        <h2>Score history across all runs</h2>
        <div id="scorehistory" style="height:380px; width:100%;" data-plotly></div>
        <div class="caption">Every time this tool is run, the resulting scores are appended to a local history file (`data/history.db`). This chart plots all recorded runs to date, regardless of the period each one covered.</div>
        {'' if has_history else '<p class="muted">Only one run recorded so far &mdash; run this analysis again in the future to build a trend line.</p>'}
      </div>
      <div class="panel">
        <h2>Objectives set in the previous report</h2>
        <div class="caption" style="margin-top:0; margin-bottom: 12px;">
          The top recommendations from the previous run become tracked objectives: this table compares each
          one's baseline value (measured at the time it was recommended) against its current value in this
          report. "Achieved" means the metric crossed into its healthy reference zone; "Improving"/"Regressed"
          are relative to the prior baseline; "No material change" is within a 5% relative band of the baseline.
        </div>
        {"<table><thead><tr><th>Objective</th><th>Baseline</th><th>Current</th><th>Relative change</th><th>Status</th></tr></thead><tbody>" + objectives_html + "</tbody></table>" if has_objectives else "<p class='muted'>No objectives were recorded in a previous run yet &mdash; they will appear starting from your next report.</p>"}
      </div>
      <div class="panel">
        <h2>Statistical anomaly detection</h2>
        <div class="caption" style="margin-top:0; margin-bottom:12px;">
          Each tracked metric's value this run is compared against the mean and standard deviation of that
          same metric across your own prior recorded runs (never including this run in its own baseline), via
          a z-score z&#8201;=&#8201;(current&#8722;mean)/std &mdash; the standard control-chart technique for
          detecting when a new observation is unusual relative to an established personal baseline, rather
          than an absolute target. WATCH means |z|&#8201;&#8805;&#8201;{anomaly_watch_z:g}; FLAGGED means
          |z|&#8201;&#8805;&#8201;{anomaly_flag_z:g}. Requires at least {anomaly_min_runs} prior runs per metric.
        </div>
        {f"<p><b>{flagged_count}</b> metric(s) flagged, <b>{watch_count}</b> under watch, out of {len(anomalies)} evaluated.</p>" if anomalies else ""}
        {"<table><thead><tr><th>Metric</th><th class='num'>This run</th><th class='num'>Historical mean &plusmn; std</th><th class='num'>z-score</th><th class='num'>n</th><th>Status</th><th>Direction</th></tr></thead><tbody>" + anomaly_rows_html + "</tbody></table>" if anomalies else "<p class='muted'>Not enough prior runs recorded yet to establish a personal baseline for anomaly detection (needs at least " + str(anomaly_min_runs) + " prior runs per metric). This will populate automatically as you keep running reports over time.</p>"}
      </div>
    </div>

    <div class="tabpanel" id="tab-advanced">
      <div class="panel">
        <h2>Absorbing Markov chain: expected turns to resolution</h2>
        {"<div id='markovchart' style='height:340px; width:100%;' data-plotly></div>" if markov_available else "<p class='muted'>Not enough turn transitions recorded yet (need at least 20) to fit a reliable Markov chain. This will populate as more history accumulates.</p>"}
        <div class="caption">
          Every turn is classified into one of four transient states &mdash; <b>Normal</b>, <b>Vague</b> (low
          specificity), <b>Clarification</b> (assistant asked a question back), or <b>Correction</b> (user
          corrected a prior answer) &mdash; plus an absorbing <b>Resolved</b> state entered at the last turn of
          each session. Pooling all observed transitions across every session in this period into a single
          transition matrix P, the transient block Q gives the fundamental matrix N&#8201;=&#8201;(I&#8722;Q)<sup>&#8722;1</sup>;
          each row sum of N is the expected number of further turns before resolution, starting from that state.
          This is a property of the <i>pooled empirical model</i>, not a live prediction for any single ongoing
          conversation &mdash; see Appendix A in the README for the full derivation. A lower value for
          Correction/Vague relative to Normal indicates that once a conversation goes sideways, it tends to
          need more turns than one that stays on track.
        </div>
        {f"<p class='detail'>Fitted from {markov.n_transitions} observed transitions.</p>" if markov_available else ""}
      </div>

      <div class="panel">
        <h2>Discrete winding number: circularity of multi-turn sessions</h2>
        {"<div id='windingchart' style='height:340px; width:100%;' data-plotly></div>" if winding_available else "<p class='muted'>Not enough multi-turn sessions (need at least 4 turns each) to compute winding numbers this period.</p>"}
        <div class="caption">
          Each session with 4+ turns is embedded as a 2D lattice path: the x-coordinate accumulates the sign of
          each turn's word-count trend (expanding vs. contracting replies), the y-coordinate accumulates a
          running correction-minus-acceptance balance. The discrete winding number W is the sum of the signed
          turning angles between consecutive path segments (about the path's own centroid), divided by 2&#960;.
          |W| near 0 means the conversation moved roughly monotonically toward resolution; |W| &#8805; 0.75
          means the path looped back on itself at least three-quarters of a full turn &mdash; a geometric
          signature of the conversation revisiting the same ground (re-explaining, re-correcting) rather than
          progressing. See Appendix A for the full definition and edge-case handling.
        </div>
        {f"<p class='detail'>Mean |W| across {len(winding_values)} sessions: {winding_abs_mean:.3f}. Sessions flagged as circular (|W| &#8805; {winding_circ_threshold:g}): {circular_count}.</p>" if winding_available else ""}
      </div>

      <div class="panel">
        <div class="row2">
          <div>
            <h2>Burstiness &amp; memory of turn timing</h2>
            {"<div id='burstchart' style='height:300px; width:100%;' data-plotly></div>" if burst else "<p class='muted'>Not enough timestamped turns (need at least 10) to compute burstiness this period.</p>"}
            <div class="caption">
              Goh &amp; Barabási's burstiness parameter B&#8201;=&#8201;(&#963;<sub>&#964;</sub>&#8722;&#956;<sub>&#964;</sub>)/(&#963;<sub>&#964;</sub>+&#956;<sub>&#964;</sub>)
              is computed on the inter-arrival times &#964; between consecutive turns, pooled across all
              sessions. B&#8201;=&#8201;&#8722;1 is perfectly periodic (metronomic) timing, B&#8201;=&#8201;0 is a memoryless
              Poisson process, B&#8201;&#8594;&#8201;1 is maximally bursty (long idle gaps punctuated by rapid-fire
              turns). The memory coefficient M is the Pearson correlation between consecutive intervals
              (&#964;<sub>i</sub>, &#964;<sub>i+1</sub>): M&#8201;&gt;&#8201;0 means short intervals tend to follow short
              intervals (self-reinforcing rapid exchanges); M&#8201;&lt;&#8201;0 means short and long intervals tend to
              alternate.
            </div>
            {f"<p class='detail'>B = {fmt_signed(burst.burstiness)}, M = {fmt_signed(burst.memory)}, from {burst.n_intervals} intervals.</p>" if burst else ""}
          </div>
          <div>
            <h2>Compression-ratio complexity proxy</h2>
            {f"<div class='stat-card'><div class='stat-value'>{comp.ratio:.3f}</div><div class='stat-label'>compressed bytes / raw bytes (gzip -9)</div></div>" if comp else "<p class='muted'>Not enough transcript text yet (need at least 200 bytes) to compute this metric.</p>"}
            <div class="caption">
              This is a normalised-compression-distance-style proxy for the Kolmogorov complexity of your own
              prompt-and-reply corpus for this period (Cilibrasi &amp; Vitányi's NCD, using gzip as the
              practical compressor stand-in for the uncomputable Kolmogorov complexity). A ratio closer to 0
              means your text this period was highly repetitive/templated (the compressor found and exploited
              lots of redundancy); a ratio closer to 1 means the text was closer to incompressible
              (high lexical/structural entropy, little repeated phrasing). This value is only meaningful
              compared against your own value from a different period &mdash; see Progress Over Time.
            </div>
            {f"<p class='detail'>{comp.raw_bytes:,} raw bytes &#8594; {comp.compressed_bytes:,} compressed bytes.</p>" if comp else ""}
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Heaps&#8217; law: vocabulary growth exponent</h2>
        {"<div id='heapschart' style='height:360px; width:100%;' data-plotly></div>" if heaps else "<p class='muted'>Not enough tokens yet (need at least 200) to fit a Heaps&#8217; law curve this period.</p>"}
        <div class="caption">
          Heaps' law models vocabulary size V as a power of corpus length n: V(n)&#8201;=&#8201;K&#183;n<sup>&#946;</sup>.
          Fitting log(V) against log(n) by ordinary least squares over exponentially-spaced samples of your
          own transcript text gives &#946; (the exponent) and R&#178; (fit quality). Natural-language corpora
          typically show &#946; in the 0.4&#8211;0.6 range as new distinct words are introduced at a
          steadily-decreasing rate; a &#946; noticeably below that band suggests heavy reuse of a narrow,
          templated vocabulary (fewer distinct concepts introduced per prompt), while a high R&#178; confirms
          the power-law relationship actually holds for your data rather than being noise. See Appendix A for
          the OLS derivation.
        </div>
        {f"<p class='detail'>&#946; = {heaps.beta:.3f}, R&#178; = {heaps.r_squared:.3f}, fit over {heaps.n_samples} samples spanning {heaps.n_tokens:,} tokens.</p>" if heaps else ""}
      </div>
    </div>

    <div class="tabpanel" id="tab-skills">
      <div class="panel">
        <h2>Installed personal skills</h2>
        <table>
          <thead><tr><th>Skill</th><th>Location</th><th>Description</th></tr></thead>
          <tbody>{skills_rows}</tbody>
        </table>
        <div class="caption">Scanned from `~/.copilot/skills/*/SKILL.md` and `~/.claude/skills/*/SKILL.md`.</div>
      </div>
    </div>

    <div class="tabpanel" id="tab-leveling">
      <div class="grid">
        <div class="card level-hero">
          <div class="big level-hero-num">{prompt_level if prompt_level is not None else '&mdash;'}</div>
          <div class="label">Prompt Level (composite, weighted average of the 7 skills below)</div>
          {prompt_level_delta_html}
        </div>
        <div class="card level-hero">
          <div class="big level-hero-num">{total_level if total_level is not None else '&mdash;'}</div>
          <div class="label">Total level (sum of all 7 skill levels, max {7 * 99})</div>
        </div>
      </div>
      <div class="panel">
        <h2>Skills</h2>
        <div class="skill-grid">
          {skill_cards_html}
        </div>
        <div class="caption">
          Each skill's XP accrues permanently across every analysis run &mdash; a turn is only ever counted
          once (tracked in a local `seen_turns` ledger keyed by session and turn index), so re-running this
          tool on an overlapping period never double-counts XP. Levels 1&#8211;99 follow the same geometric
          shape as a classic MMO XP curve (slow grind to reach the highest levels), rescaled so the
          early/mid levels are reachable within weeks of normal usage. Full formulas, the per-skill XP rules,
          and the derivation of the level curve are in README Appendix B.
        </div>
      </div>
      <div class="panel">
        <h2>Hiscores (opt-in, cross-colleague comparison)</h2>
        <div class="rec-text">
          This run's JSON export (written alongside the HTML report) now includes your Prompt Level, Total
          level, and per-skill levels/XP &mdash; nothing else new, and still no raw prompt/reply text. If you
          and your colleagues each export and share that JSON, <code>bin/hiscores.py</code> merges any number
          of exports into one ranked table using the exact same deterministic formula for everyone, so levels
          are genuinely comparable. This is additive to, and independent from, the existing
          <code>bin/aggregate_team.py</code> team rollup (which aggregates score distributions, weakest axes,
          and anomaly/Mahalanobis statistics) &mdash; run either or both, on the same or different sets of
          exports. See README "Hiscores" section for usage.
        </div>
      </div>
    </div>

    <div class="tabpanel" id="tab-methodology">
      <div class="panel">
        <h2>Score axis formulas</h2>
        <table>
          <thead><tr><th>Axis</th><th>Formula</th></tr></thead>
          <tbody>
            {"".join(f"<tr><td>{AXIS_LABEL[a]}</td><td class='detail'><code>{AXIS_FORMULA[a]}</code></td></tr>" for a in AXIS_ORDER)}
          </tbody>
        </table>
        <div class="caption">Overall score is the unweighted mean of the four axis scores. All inputs are turn-level boolean/rate features defined below, computed only from your own transcript text (and, where noted, the assistant's replies to your turns).</div>
      </div>
      <div class="panel">
        <h2>Metric definitions</h2>
        {metric_defs_html}
      </div>
      <div class="panel">
        <h2>Cross-agent guidance boundary</h2>
        <div class="rec-text">
          The scored defaults retain only prompt practices that overlap in official guidance for both
          Claude and GitHub Copilot and can be detected reliably from a local transcript: clear
          requirements, concrete artifact grounding, motivation/rationale, acceptance criteria,
          structured multi-step work, and validation. The Context &amp; rationale axis deliberately
          measures both <i>where</i> work belongs (file/code references) and <i>why</i> it matters
          (because/so-that/in-order-to style context).
          <br><br>
          The report deliberately does <b>not</b> score provider- or surface-specific tactics such as
          Claude XML tags, named roles, hidden-thinking settings, Copilot IDE participants/keywords,
          or which files happen to be open. Those can be useful in their respective products, but they
          are neither universal quality signals nor reliably inferable from exported transcript data.
          Nor does this report award points merely for starting with a verb. See the source guidance:
          <a href="https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices">Anthropic prompting best practices</a>,
          <a href="https://docs.github.com/en/copilot/concepts/prompting/prompt-engineering">GitHub Copilot prompt engineering</a>, and
          <a href="https://docs.github.com/en/copilot/get-started/best-practices">GitHub Copilot best practices</a>.
        </div>
      </div>
      <div class="panel">
        <h2>Evidentiary standard and limitations</h2>
        <div class="rec-text">
          Two kinds of justification are used in this report, and each recommendation states which applies:
          <ul>
            <li><b>Self-referential statistics:</b> a rate or segmented comparison computed directly from your
            own transcript data, with a sample size (n) and a confidence label (high: n&ge;80, moderate:
            n&ge;20, directional: below that).</li>
            <li><b>Structural/mechanistic reasoning:</b> an explanation of how LLM context processing behaves
            (token windows, ambiguity resolution over recent turns, lack of an enforced check on multi-clause
            requests) used to explain *why* a self-referential pattern would be expected to matter. This is
            reasoning about mechanism, not a numeric external citation &mdash; no claim in this report cites an
            external study or benchmark that this tool has not itself computed.</li>
          </ul>
          All detection (vague language, artifact/rationale context, structured steps, corrections,
          acceptance criteria, clarification loops, and context restatement) is pattern- and
          heuristic-based, not a judgment by the agent that produced the analyzed responses. Treat
          scores as a directional signal for reflection, not a precise measurement.
        </div>
      </div>
    </div>

  </div>

  <div class="footer">Generated by prompt-optimizer &mdash; a local, agent-agnostic analysis of your own Copilot CLI / Claude Code prompt history. No data leaves your machine.</div>

<script>
const dark = {{
  paper_bgcolor: '#171a21', plot_bgcolor: '#171a21',
  font: {{ color: '#c7cdd9', family: "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif", size: 12 }},
  margin: {{ t: 28, r: 24, b: 48, l: 56 }},
  hoverlabel: {{ bgcolor: '#232838', bordercolor: '#3a4256', font: {{ color: '#e8eaed' }} }}
}};
const gridAxis = {{ gridcolor: '#262b38', zerolinecolor: '#333a4a', color: '#9aa3b2', tickfont: {{ size: 11 }} }};

document.getElementById('tabnav').addEventListener('click', (e) => {{
  const btn = e.target.closest('button[data-tab]');
  if (!btn) return;
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tabpanel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  const panel = document.getElementById('tab-' + btn.dataset.tab);
  panel.classList.add('active');
  // Plotly sizes charts to their container's width at creation time; charts
  // created while their tab was hidden (display:none => width 0) need an
  // explicit resize once the tab becomes visible, since a bare window
  // 'resize' event only helps plots created with config.responsive AND
  // already has a nonzero container width.
  requestAnimationFrame(() => {{
    panel.querySelectorAll('[data-plotly]').forEach(div => {{
      try {{ Plotly.Plots.resize(div); }} catch (err) {{ /* not yet rendered */ }}
    }});
  }});
}});

const p3 = {json.dumps(path3d)};
Plotly.newPlot('path3d', [
  {{
    type: 'scatter3d', mode: 'lines+markers+text',
    x: p3.xs, y: p3.ys, z: p3.zs, text: p3.markers, textposition: 'top center',
    textfont: {{ color: '#e8eaed', size: 12, family: 'monospace' }},
    hovertext: p3.labels, hovertemplate: 'Step %{{text}}: %{{hovertext}}<br>Specificity %{{x:.0f}} / Context %{{y:.0f}} / Efficiency %{{z:.0f}}<extra></extra>',
    line: {{ width: 6, color: '#60a5fa' }},
    marker: {{ size: 12, color: '#1d4ed8', line: {{ color: '#e8eaed', width: 1.5 }} }},
    name: 'Path'
  }},
  {{
    type: 'scatter3d', mode: 'markers+text',
    x: [p3.optimal.x], y: [p3.optimal.y], z: [p3.optimal.z], text: ['OPT'], textposition: 'top center',
    textfont: {{ color: '#fbbf24', size: 12 }},
    hovertext: ['Optimal (100, 100, 100)'], hovertemplate: '%{{hovertext}}<extra></extra>',
    marker: {{ size: 11, color: '#fbbf24', symbol: 'diamond', line: {{ color: '#171a21', width: 1 }} }},
    name: 'Optimal'
  }}
], Object.assign({{}}, dark, {{
  scene: {{
    xaxis: {{ title: 'Specificity', range: [0,100], color:'#9aa3b2', gridcolor: '#2a3040', backgroundcolor: '#14161d' }},
    yaxis: {{ title: 'Context & rationale', range: [0,100], color:'#9aa3b2', gridcolor: '#2a3040', backgroundcolor: '#14161d' }},
    zaxis: {{ title: 'Efficiency', range: [0,100], color:'#9aa3b2', gridcolor: '#2a3040', backgroundcolor: '#14161d' }},
    bgcolor: '#171a21',
    camera: {{ eye: {{ x: 1.5, y: 1.5, z: 1.1 }} }}
  }},
  margin: {{ t: 10, r: 10, b: 10, l: 10 }},
  showlegend: false
}}), {{displayModeBar: false, responsive: true}});

Plotly.newPlot('radar', [{{
  type: 'scatterpolar',
  r: {json.dumps(radar_values)},
  theta: {json.dumps(radar_labels)},
  fill: 'toself',
  fillcolor: 'rgba(110, 231, 183, 0.18)',
  line: {{ color: '#6ee7b7', width: 2 }},
  marker: {{ size: 5, color: '#6ee7b7' }}
}}], Object.assign({{}}, dark, {{
  margin: {{ t: 70, r: 70, b: 70, l: 70 }},
  polar: {{
    radialaxis: {{ visible: true, range: [0,100], color:'#9aa3b2', gridcolor: '#262b38' }},
    angularaxis: {{ color: '#c7cdd9', gridcolor: '#262b38', rotation: 90 }},
    bgcolor:'#171a21'
  }},
  showlegend: false
}}), {{displayModeBar: false, responsive: true}});

Plotly.newPlot('perday', [{{
  type: 'bar', x: {json.dumps(day_labels)}, y: {json.dumps(day_values)},
  marker: {{ color: '#60a5fa', line: {{ width: 0 }} }}
}}], Object.assign({{}}, dark, {{ xaxis:Object.assign({{}}, gridAxis), yaxis:Object.assign({{}}, gridAxis, {{title:'Turns'}}), bargap: 0.25 }}), {{displayModeBar: false, responsive: true}});

Plotly.newPlot('lenbuckets', [{{
  type: 'bar', x: {json.dumps(bucket_order)}, y: {json.dumps(bucket_values)},
  marker: {{ color: '#6ee7b7' }}
}}], Object.assign({{}}, dark, {{ xaxis:Object.assign({{}}, gridAxis, {{title:'Words per turn'}}), yaxis:Object.assign({{}}, gridAxis, {{title:'Turns'}}), bargap: 0.25 }}), {{displayModeBar: false, responsive: true}});

Plotly.newPlot('sessionbuckets', [{{
  type: 'bar', x: {json.dumps(session_bucket_order)}, y: {json.dumps(session_bucket_values)},
  marker: {{ color: '#fbbf24' }}
}}], Object.assign({{}}, dark, {{ xaxis:Object.assign({{}}, gridAxis, {{title:'Turns per session'}}), yaxis:Object.assign({{}}, gridAxis, {{title:'Sessions'}}), bargap: 0.25 }}), {{displayModeBar: false, responsive: true}});

Plotly.newPlot('heatmap', [{{
  type: 'heatmap', z: {json.dumps(heat)}, x: [...Array(24).keys()], y: {json.dumps(WEEKDAY_NAMES)},
  colorscale: 'Viridis', showscale: true, colorbar: {{ tickfont: {{ color: '#9aa3b2', size: 10 }}, thickness: 12 }}
}}], Object.assign({{}}, dark, {{ xaxis:Object.assign({{}}, gridAxis, {{title:'Hour of day', dtick: 2}}), yaxis:Object.assign({{}}, gridAxis) }}), {{displayModeBar: false, responsive: true}});

Plotly.newPlot('repos', [{{
  type: 'bar', orientation: 'h', y: {json.dumps(repo_labels[::-1])}, x: {json.dumps(repo_values[::-1])},
  marker: {{ color: '#fbbf24' }}
}}], Object.assign({{}}, dark, {{ xaxis:Object.assign({{}}, gridAxis, {{title:'Turns'}}), yaxis:Object.assign({{}}, gridAxis, {{automargin:true}}), margin: {{ t: 28, r: 24, b: 48, l: 20 }} }}), {{displayModeBar: false, responsive: true}});

Plotly.newPlot('scorehistory', [
  {{ type:'scatter', mode:'lines+markers', name:'Overall', x: {json.dumps(hist_labels)}, y: {json.dumps(hist_overall)}, line:{{color:'#6ee7b7', width:3}} }},
  {{ type:'scatter', mode:'lines+markers', name:'Specificity', x: {json.dumps(hist_labels)}, y: {json.dumps(hist_spec)}, line:{{color:'#60a5fa'}} }},
  {{ type:'scatter', mode:'lines+markers', name:'Context', x: {json.dumps(hist_labels)}, y: {json.dumps(hist_ctx)}, line:{{color:'#fbbf24'}} }},
  {{ type:'scatter', mode:'lines+markers', name:'Structure', x: {json.dumps(hist_labels)}, y: {json.dumps(hist_struct)}, line:{{color:'#f87171'}} }},
  {{ type:'scatter', mode:'lines+markers', name:'Efficiency', x: {json.dumps(hist_labels)}, y: {json.dumps(hist_eff)}, line:{{color:'#c084fc'}} }},
], Object.assign({{}}, dark, {{ xaxis:Object.assign({{}}, gridAxis), yaxis:Object.assign({{}}, gridAxis, {{range:[0,100], title:'Score'}}), legend:{{orientation:'h', y:-0.25, font:{{color:'#c7cdd9'}}}}, margin: {{ t: 20, r: 24, b: 70, l: 56 }} }}), {{displayModeBar: false, responsive: true}});

{markov_js}

{winding_js}

{burst_js}

{heaps_js}
</script>
</body>
</html>
"""
