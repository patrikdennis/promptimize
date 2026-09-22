---
name: optimize-prompts
description: 'Analyze the user''s own recent AI-agent prompt history (Copilot CLI and/or Claude Code) for a time period (e.g. "this week", "last 30 days") and produce an HTML report with prompting statistics, visualizations, and concrete tasks to improve prompting. Use when the user asks to optimize prompts, review their prompting habits, or asks something like "/optimize-prompts this week".'
argument-hint: 'A time period, e.g. "this week", "last 30 days", "this month", or a date range "YYYY-MM-DD..YYYY-MM-DD". Defaults to "this week" if omitted.'
---

# Optimize Prompts

This skill is agent-agnostic: it works identically whether it is invoked from
GitHub Copilot CLI or Claude Code. It never needs the current project's repo
checked out — it reads **your own local prompt history** directly from disk:

- GitHub Copilot CLI: `~/.copilot/session-store.db` (SQLite)
- Claude Code: `~/.claude/projects/**/*.jsonl`

All of the actual logic (data loading, normalization, scoring, chart/report
generation) lives in a standalone tool at:

```
~/Documents/prompt-optimizer/bin/analyze.py
```

Nothing here should be re-implemented inline — just invoke that script.

## How to run it

1. Parse the user's requested period from their message (e.g. "this week",
   "last 30 days", "this month", "yesterday", "all", or an explicit
   `YYYY-MM-DD..YYYY-MM-DD` range). Default to `"this week"` if they didn't
   specify one.
2. Run:

   ```bash
   python3 ~/Documents/prompt-optimizer/bin/analyze.py --period "<period>" --agent all
   ```

   (`--agent` can be narrowed to `copilot` or `claude` if the user asks for
   just one tool's history.)
3. The script prints the number of turns/sessions analyzed, the overall
   0-100 score, the previous run's score and delta (if one exists), and the
   path to the generated HTML report (written under
   `~/Documents/prompt-optimizer/out/report-<timestamp>.html`). Each run is
   also recorded to `~/Documents/prompt-optimizer/data/history.db`, so
   running this repeatedly over time builds a progress trend and evaluates
   whether the previous report's top objectives were met.
4. Open the report for the user (e.g. `open <path>` on macOS) and summarize
   in chat:
   - The overall score and the 4 axis scores (specificity, context
     anchoring, structure/acceptance criteria, efficiency).
   - Whether this run's score improved, regressed, or held steady versus
     the previous run, and which of the previous run's tracked objectives
     were achieved/improving/regressed (see the report's "Progress Over
     Time" tab).
   - The top 5-8 recommended tasks, each with its cited evidence (a
     self-referential statistic from the user's own data, not a generic
     claim) and the concrete action to take.
   - Point out that the 3D chart (Overview tab) shows the steepest-ascent
     path toward the optimal point — call out which single change would
     move the needle most.
5. If the user wants, offer to turn the top recommendations into actual
   todo items for them to work through.

## Report structure

The generated report has separate tabs — mention which tab has what the
user is asking about:

- **Overview**: score cards, the 3D position-vs-optimal chart, score radar,
  an explanation of what the optimal point (100,100,100) actually means,
  and (once enough run history exists) a covariance-adjusted Mahalanobis
  distance-to-optimal panel (see README Appendix A.6).
- **Metric Deep-Dive**: turns-per-day, prompt-length distribution, day x hour
  heatmap, session-length distribution, top projects/repos.
- **Evidence & Correlations**: segmented statistics (e.g. correction rate on
  turns with vs. without a file reference), each with sample sizes — this is
  the quantitative backing behind the recommendations.
- **Recommendations & Tasks**: full detail per recommendation — evidence,
  mechanism (why it affects LLM output quality), concrete action, priority,
  estimated point gain, confidence level.
- **Progress Over Time**: score history across all runs, a table of whether
  each objective set in the previous report was achieved, and a statistical
  anomaly-detection panel flagging metrics that are z-score outliers versus
  the user's own prior-run baseline (see README Appendix A.7).
- **Advanced Analytics**: Markov-chain expected turns to resolution,
  discrete winding number per session (flags "circular"/unresolved
  conversations), burstiness & memory of turn timing, a compression-ratio
  complexity proxy, and a Heaps' law vocabulary-growth fit — see README
  Appendix A for full derivations.
- **Skills & Capabilities**: installed personal skills and suggested gaps.
- **Methodology**: exact formula for every axis and metric, plus the
  evidentiary standard used (self-referential statistics vs. structural
  reasoning about LLM context processing) and stated limitations.

## Notes

- This tool only reads local files; nothing is sent anywhere. Safe to run
  freely.
- If both `~/.copilot/session-store.db` and `~/.claude/projects/` exist, both
  are included by default so the report reflects the user's real cross-tool
  habits.
- Requires only Python 3 stdlib — no pip install needed. Every run also
  writes a machine-readable JSON + CSV export next to the HTML report
  (containing no raw prompt/reply text); `bin/aggregate_team.py` can turn
  several people's exports into one anonymized team-level report.
- `config.yaml` at the repo root controls every scoring weight, threshold,
  and sample-size cutoff used by the tool — mention it if the user wants to
  retune anything rather than treating the numbers as fixed.
- If the script reports 0 turns for a period, try a wider period (e.g.
  `"all"`) before concluding there is no data.
- Use `--no-record` if the user wants a report without it counting as an
  official run for progress-tracking purposes (e.g. testing).
