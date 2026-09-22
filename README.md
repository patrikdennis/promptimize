# prompt-optimizer

A local, **agent-agnostic** tool that analyzes your own prompt history from
GitHub Copilot CLI and/or Claude Code and produces a multi-tab HTML report
with quantitative, evidence-backed metrics, a ranked list of concrete
recommendations, and progress tracking across repeated runs.

Nothing leaves your machine — it only reads local session files that already
exist on disk:

- **GitHub Copilot CLI**: `~/.copilot/session-store.db` (SQLite)
- **Claude Code**: `~/.claude/projects/**/*.jsonl`

It lives outside any git repo, at `~/Documents/prompt-optimizer`, on purpose —
it's a personal tool, not project code.

## Usage

```bash
python3 ~/Documents/prompt-optimizer/bin/analyze.py --period "this week"
python3 ~/Documents/prompt-optimizer/bin/analyze.py --period "last 30 days" --open
python3 ~/Documents/prompt-optimizer/bin/analyze.py --period "2026-09-01..2026-09-21"
python3 ~/Documents/prompt-optimizer/bin/analyze.py --agent claude   # only Claude Code history
python3 ~/Documents/prompt-optimizer/bin/analyze.py --no-record      # don't count this as an official run
```

Periods understood: `today`, `yesterday`, `this week`, `last week`,
`this month`, `last month`, `last N days`, `all`, or an explicit
`YYYY-MM-DD..YYYY-MM-DD` range. Defaults to `this week`.

Via chat, in either Copilot CLI or Claude Code (once the skill below is
installed): say something like **"optimize-prompts this week"** and the
agent runs the tool, opens the report, and summarizes the key findings.

Reports are written to `out/report-<timestamp>.html` (self-contained, uses
Plotly via CDN for charts — open in any browser).

## What's in the report

The report is organized into tabs:

- **Overview** — overall 0-100 score, the 4 axis scores (specificity,
  context anchoring, structure/acceptance criteria, efficiency), a **3D
  chart** plotting your current position against the optimal point
  (100, 100, 100), and an explicit explanation of what that point means and
  why it's mathematically the ceiling of the score formulas (not an
  arbitrary target). Once at least a few prior runs exist, this tab also
  shows a **covariance-adjusted (Mahalanobis) distance to optimal** panel —
  see "Distance metrics" below and Appendix A.6.
- **Metric Deep-Dive** — turns-per-day, prompt-length distribution, a
  day x hour heatmap, session-length distribution, top projects/repos, each
  with a caption explaining exactly what's plotted.
- **Evidence & Correlations** — segmented statistics computed from your own
  data (e.g. correction rate on turns with vs. without a file reference),
  with sample sizes — the quantitative backing behind every recommendation.
- **Recommendations & Tasks** — a detailed card per recommendation: the
  cited evidence, the mechanism (why it affects an LLM's ability to resolve
  your request correctly on the first attempt), a concrete action, priority,
  estimated point gain, and a confidence label based on sample size.
- **Progress Over Time** — a score-history chart across every run recorded
  so far, a table showing whether each objective set in your *previous*
  report was achieved, improving, unchanged, or regressed, and a
  **statistical anomaly detection** panel that flags any metric whose
  current-run value is a z-score outlier relative to your own prior-run
  distribution — see "Distance metrics" below and Appendix A.7.
- **Advanced Analytics** — five metrics from probability theory, complex
  analysis, and information theory, applied to your own transcripts: an
  absorbing Markov chain over turn states, a discrete winding number per
  session, Goh-Barabási burstiness/memory of turn timing, a compression-ratio
  proxy for Kolmogorov complexity, and a Heaps' law vocabulary-growth fit.
  Full derivations are in Appendix A below.
- **Skills & Capabilities** — your installed personal skills (across both
  `~/.copilot/skills` and `~/.claude/skills`) plus suggested gaps based on
  workflow patterns detected in your prompts.
- **Methodology** — the exact formula behind every axis and metric, and the
  evidentiary standard used (see below).

## How the scoring and evidence work

All metrics are rule-based text-pattern matching over your own prompt *and*
assistant-reply text (vague-language keywords, file/code references, action
verbs, correction phrases, acceptance-criteria phrases, clarifying-question
detection in replies, word-overlap between consecutive turns). Every
recommendation states which of two evidentiary standards backs it:

1. **Self-referential statistics** — a rate or segmented comparison computed
   directly from your own transcript data, with a sample size and a
   confidence label (high: n>=80, moderate: n>=20, directional otherwise).
2. **Structural/mechanistic reasoning** — an explanation of how LLM context
   processing behaves (ambiguity resolution, lack of an enforced check on
   multi-clause requests) used to explain *why* a pattern would be expected
   to matter. This is reasoning about mechanism, not a numeric external
   citation — nothing here cites an external study this tool has not itself
   computed.

Treat scores as a directional signal for reflection, not a precise
measurement. Tune the keyword lists and thresholds in `lib/metrics.py` and
`lib/recommendations.py` if you want to adjust what counts.

## Progress tracking across runs

Each run is appended to a local SQLite database at `data/history.db`. The
top recommendations from a run become "objectives" with a recorded baseline
value; the *next* run evaluates each objective's current value against that
baseline and against a healthy reference threshold, and reports the result
in the Progress Over Time tab. This is what turns the tool from a one-off
report into a closed feedback loop for gradually improving your prompting.

## Configuration

`config.yaml` (plain text, hand-editable, no dependency needed to parse it —
see `lib/config.py`) controls every threshold used by the tool: scoring
formula weights, objective-achieved/regressed bands, the minimum sample
sizes required before each Advanced Analytics metric is computed, the
anomaly-detection z-score cutoffs, the Mahalanobis shrinkage intensity, and
team-mode settings. If the file is missing or a key is absent, built-in
defaults (mirrored in `lib/config.py::DEFAULTS`) are used, so the tool always
runs even with no config present. Edit `config.yaml` directly; there is no
need to touch any `lib/*.py` file to retune a threshold.

## Distance metrics: Mahalanobis and anomaly detection

Two statistically-grounded additions go beyond simple thresholds:

- **Mahalanobis distance to optimal** (Overview tab) measures how far your
  current 4-axis score position is from the (100, 100, 100, 100) optimum,
  *scaled by how your own axis scores actually co-vary across your run
  history* — rather than treating all axes as independent and
  equally-scaled, as a plain Euclidean distance would. Requires a handful of
  prior runs; falls back to reporting Euclidean distance with an explicit
  note when history is too short. Derivation: Appendix A.6.
- **Anomaly detection** (Progress tab) computes a z-score for every tracked
  metric in the current run against the mean/standard deviation of *all
  prior* runs (the current run is never included in its own baseline), and
  labels each metric normal / watch / flagged based on configurable
  thresholds. This surfaces genuine regressions or spikes that a simple
  "went up or down" comparison would miss because it accounts for how much
  that metric normally fluctuates for you. Derivation: Appendix A.7.

## Exports

Every run writes, alongside the HTML report in `out/`:

- `report-<timestamp>.json` — the full structured payload (scores, metric
  values, conditional/evidence stats, advanced-metrics results, anomaly
  flags, Mahalanobis result) with **no raw prompt or reply text**, so it is
  safe to share or hand to a downstream aggregator.
- `report-<timestamp>.csv` — the same data flattened to
  `category,metric,value` rows for quick spreadsheet analysis.

These exports are also the input format for team/aggregate mode below.

## Team / aggregate mode

`bin/aggregate_team.py` ingests multiple people's exported JSON files (each
produced locally and shared voluntarily — nothing is transmitted anywhere by
this tool) and produces one anonymized aggregate HTML summary: distribution
of scores across the group, which axes are weakest on average, common
flagged anomalies, and aggregate recommendation frequency. Individual
contributors are labeled `Contributor 1, 2, ...` (never by filename or any
identifying string) per `team_mode.anonymize_labels` in `config.yaml`, and
the aggregate is only produced once at least `team_mode.min_users_for_aggregate`
exports are supplied, to avoid trivially de-anonymizing a single person.

```bash
python3 ~/Documents/prompt-optimizer/bin/aggregate_team.py out/*.json --open
```

## Project layout

```text
bin/analyze.py              CLI entry point (period parsing, orchestration)
bin/aggregate_team.py       Anonymized team/aggregate report from exported JSON files
lib/backends.py              Reads + normalizes Copilot CLI / Claude Code history into sessions/turns
lib/metrics.py               Turn-level feature extraction, aggregate stats, scoring, metric definitions
lib/recommendations.py       Evidence-backed recommendation engine + skill discovery
lib/history.py                Persistent run history + objective tracking (SQLite)
lib/report.py                 Self-contained multi-tab HTML report renderer (Plotly)
lib/advanced_metrics.py       Markov/winding/burstiness/compression/Heaps' law (see Appendix A)
lib/mahalanobis.py            Covariance-shrinkage Mahalanobis distance to optimal (Appendix A.6)
lib/anomaly.py                Z-score anomaly detection against prior-run baselines (Appendix A.7)
lib/linalg.py                 Shared pure-Python matrix helpers (no numpy dependency)
lib/config.py                 Dependency-free config.yaml loader + defaults
lib/export.py                 JSON/CSV export writer
config.yaml                   Editable thresholds/weights (see Configuration above)
skill/SKILL.md                The skill definition, symlinked into both agents
data/history.db                Local run-history database (created on first run)
out/                            Generated reports (HTML + JSON + CSV) land here
tests/                          Pytest suite covering config, math, and scoring logic
```

## Installing the skill (already done for you)

The skill is symlinked so both agents see the same source:

```bash
ln -sfn ~/Documents/prompt-optimizer/skill ~/.copilot/skills/optimize-prompts
ln -sfn ~/Documents/prompt-optimizer/skill ~/.claude/skills/optimize-prompts
```

Editing `skill/SKILL.md` here updates both automatically.

## Requirements

Python 3 stdlib only — no `pip install` needed. Charts render via the Plotly
CDN, so opening the report requires internet access for the visuals (the
data itself is 100% local).

## Running the tests

The tool itself needs zero dependencies, but the test suite uses `pytest`,
which isn't installed system-wide (this machine's Python is externally
managed). A local virtual environment keeps that dependency isolated from
everything else:

```bash
cd ~/Documents/prompt-optimizer
python3 -m venv .venv
./.venv/bin/pip install pytest
./.venv/bin/pytest tests/ -v
```

The suite (`tests/`) covers the config loader/parser, the pure-Python matrix
math, the Mahalanobis distance (including the degenerate near-zero-variance
edge case that originally motivated the shrinkage fix), z-score anomaly
classification, objective achieved/improving/regressed logic, the 4 axis
scoring formulas, and all 5 Advanced Analytics metrics — using small,
hand-computable synthetic inputs rather than your real prompt history.

## Appendix A: mathematical derivations for the Advanced Analytics metrics

Everything below is implemented in pure Python (no numpy) in
`lib/advanced_metrics.py`. Every metric here is computed only from your own
transcript data — none of them rely on an external benchmark or an
unverifiable "typical" comparison value.

### A.1 Absorbing Markov chain over turn states

**Setup.** Every turn is classified into exactly one of four transient
states by `classify_state()`, with priority Correction > Clarification >
Vague > Normal:

- **Normal** — an ordinary turn (no vagueness, no correction, no
  clarification signal).
- **Vague** — the prompt matched one or more low-specificity heuristics
  (e.g. very short, no concrete noun/identifier).
- **Clarification** — the assistant's reply pattern-matches an actual
  question posed back to the user.
- **Correction** — the user's turn pattern-matches a correction of a prior
  answer ("no, that's wrong", "that didn't work", etc).

Plus one absorbing state, **Resolved**, which every session's *final*
observed turn transitions into (by construction, a session ends when the
user stops responding, so its last turn is treated as reaching the
absorbing state).

**The chain.** Let the transient states be indexed $1, \dots, t$ (here
$t=4$) and let $r$ be the number of absorbing states (here $r=1$). Every
observed turn $\to$ next-turn (or turn $\to$ end-of-session) pair across
*every session in the reporting period* is pooled into raw transition
counts $c_{ij}$, and the transition matrix is row-normalized:

$$P_{ij} = \frac{c_{ij}}{\sum_k c_{ik}}$$

In canonical form, a finite absorbing Markov chain's transition matrix can
be arranged as

$$P = \begin{pmatrix} Q & R \\ 0 & I_r \end{pmatrix}$$

where $Q$ ( $t \times t$ ) holds transient-to-transient probabilities, and
$R$ ( $t \times r$ ) holds transient-to-absorbing probabilities. The
**fundamental matrix** is

$$N = (I - Q)^{-1} = \sum_{k=0}^{\infty} Q^k$$

which converges because every transient state has probability mass leaking
to the absorbing state eventually (the chain is guaranteed absorbing since
every session terminates). Entry $N_{ij}$ is the expected number of visits
to transient state $j$ before absorption, given the chain started in state
$i$. Standard absorbing-Markov-chain theory (Kemeny & Snell, *Finite Markov
Chains*, 1960, ch. 3) gives the **expected number of steps before
absorption** starting from state $i$ as the $i$-th row sum of $N$:

$$\tau_i = \sum_{j=1}^{t} N_{ij} = (N \mathbf{1})_i$$

`_invert_matrix()` computes $N$ via Gauss-Jordan elimination on
$(I - Q \mid I)$ in pure Python (no numpy available in this environment).
The chain is only fitted if there are at least 20 observed transitions,
since a 4-state transition matrix has up to 16 free parameters and fewer
observations would make $Q$, and therefore $N$, unreliable.

**Interpretation caveat (important).** $\tau_i$ is a property of the
*pooled empirical model fitted over the whole reporting period*, not a
live forecast for any single ongoing conversation — it answers "if turns
behaved on average like the pooled transition frequencies observed this
period, how many more turns would a state-$i$ turn need before the session
ends?" It is still a useful comparative statistic: a large gap between
$\tau_{\text{Correction}}$ and $\tau_{\text{Normal}}$ means that, in your
own history, once a conversation needs a correction it structurally tends
to take longer to close out than one that never did.

### A.2 Discrete winding number for session trajectories

**Motivation.** A conversation that is making steady forward progress
should look, geometrically, like a path that moves outward from its
starting point. A conversation that is going in circles — repeatedly
re-explaining the same requirement, alternating between acceptance and
correction — should look like a path that loops back on itself. The
winding number from complex analysis / algebraic topology is the natural
invariant for "how many times does a closed(ish) path loop around its own
centroid," so it's repurposed here as a discrete diagnostic.

**Embedding.** For a session with turns $1, \dots, n$ ( $n \ge 4$
required), each turn $k$ contributes a 2D step:

- $\Delta x_k = \operatorname{sign}(\text{words}_k - \text{words}_{k-1})$
  — did the reply/turn expand or contract relative to the previous one.
- $\Delta y_k = +1$ if turn $k$ carries a correction signal, $-1$ if it
  carries an acceptance/resolution signal, $0$ otherwise.

The path is the cumulative sum $\left(x_k, y_k\right) = \left(\sum_{i\le k}
\Delta x_i, \sum_{i \le k} \Delta y_i\right)$, and the **centroid**
$(\bar{x}, \bar{y})$ is its mean. Let $\theta_k =
\operatorname{atan2}(y_k - \bar{y}, x_k - \bar{x})$ be the angle from the
centroid to point $k$ (points coincident with the centroid are dropped,
since angle is undefined there; at least 3 valid points are required).

**Winding number.** The discrete winding number is the total signed
turning angle around the centroid, divided by $2\pi$:

$$W = \frac{1}{2\pi} \sum_{k} \operatorname{wrap}(\theta_{k+1} - \theta_k)$$

where $\operatorname{wrap}(\cdot)$ maps an angle difference into
$(-\pi, \pi]$ before summing (implemented in `_wrap_angle()`) — this is
exactly the standard discrete definition of the winding number of a closed
polygonal path around an interior point (see e.g. do Carmo, *Differential
Geometry of Curves and Surfaces*, or any complex-analysis treatment of
$\oint \frac{dz}{z - z_0} = 2\pi i \, W$, whose discrete analogue is this
sum of wrapped angle increments).

**Interpretation.** $W \approx 0$ means the path's angular position around
its own centroid didn't complete a meaningful loop — consistent with
monotonic progress. $|W| \ge 0.75$ (three-quarters of a full revolution)
is flagged in the report as a **circular session**: the conversation's
word-count/correction dynamics revisited a similar configuration relative
to where it started, which in practice tends to correlate with re-explained
requirements or oscillating accept/reject cycles.

### A.3 Burstiness and memory of turn timing

Using the exact formulation from Goh & Barabási, *"Burstiness and memory in
complex systems"* (EPL, 2008): let $\tau_1, \dots, \tau_{m}$ be the $m$
inter-arrival times between your consecutive turns (pooled across all
sessions in the period, requires $m \ge 9$, i.e. at least 10 timestamps),
with sample mean $\mu_\tau$ and sample standard deviation $\sigma_\tau$.

**Burstiness parameter:**

$$B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau} \in [-1, 1]$$

$B = -1$ corresponds to a perfectly periodic process (all intervals
identical, $\sigma_\tau = 0$); $B = 0$ corresponds to a Poisson
(memoryless, exponentially-distributed-interval) process where $\sigma_\tau
= \mu_\tau$; $B \to 1$ corresponds to maximally bursty behavior — a
Cauchy-like, heavy-tailed distribution of intervals (long idle stretches
punctuated by rapid-fire clusters of turns).

**Memory coefficient:** the Pearson correlation coefficient between
consecutive intervals $(\tau_i, \tau_{i+1})$ for $i = 1, \dots, m-1$:

$$M = \frac{\frac{1}{m-1}\sum_i (\tau_i - \bar\tau_{(1)})(\tau_{i+1} -
\bar\tau_{(2)})}{\sigma_{\tau_{(1)}} \, \sigma_{\tau_{(2)}}}$$

where $\bar\tau_{(1)}, \sigma_{\tau_{(1)}}$ are the mean/std of
$\tau_1,\dots,\tau_{m-1}$ and $\bar\tau_{(2)}, \sigma_{\tau_{(2)}}$ of
$\tau_2,\dots,\tau_m$. $M > 0$ means short intervals cluster with short
intervals (and long with long) — a self-reinforcing rhythm; $M < 0$ means
short and long intervals tend to alternate.

### A.4 Compression-ratio complexity proxy (Kolmogorov complexity / NCD)

The Kolmogorov complexity $K(x)$ of a string $x$ — the length of the
shortest program that outputs $x$ — is not computable in general, but
Cilibrasi & Vitányi's **normalized compression distance** framework
(*"Clustering by compression,"* IEEE Trans. Information Theory, 2005)
establishes that any real-world compressor $C$ gives a practical, provably
bounded approximation of $K$, since $C(x) \ge K(x) - O(\log|x|)$ for any
lossless compressor. Here, gzip level 9 stands in for $C$, applied to the
UTF-8 concatenation of every turn's text (chronological order, requires
$\ge 200$ raw bytes):

$$\text{NCR} = \frac{|C(x)|}{|x|}$$

A low NCR means the corpus is highly self-similar/templated (the LZ77 +
Huffman backend of gzip found long repeated substrings across turns); a
high NCR (approaching 1, incompressible) means the corpus has close to
maximal lexical/structural entropy relative to its own length. Because
"typical" NCR values are corpus- and language-dependent with no universally
agreed reference point, this report deliberately does not compare your NCR
against an external benchmark — only against your own value from a
different reporting period (visible by re-running the tool over time).

### A.5 Heaps' law vocabulary-growth exponent

Heaps' law (Heaps, *Information Retrieval: Computational and Theoretical
Aspects*, 1978) is the empirical observation that the number of *distinct*
words $V$ in a corpus grows sub-linearly with the total token count $n$:

$$V(n) = K \cdot n^{\beta}, \qquad 0 < \beta < 1$$

**Fitting procedure.** Tokens are extracted via the regex `[A-Za-z']+`,
lowercased, from every turn's text in chronological order (requires
$\ge 200$ tokens total). $V(n)$ is sampled at $n$ values spaced
exponentially (base 50, ratio 1.35, up to 24 points, each $\le$ total token
count) rather than linearly, since a power law is linear in log-log space
and exponential spacing gives evenly-distributed leverage across orders of
magnitude. Taking logs of both sides linearizes the model:

$$\log V(n) = \log K + \beta \log n$$

which is fit by ordinary least squares (closed-form simple linear
regression — no numpy, just the standard sums) against $u = \log n$,
$v = \log V(n)$:

$$\beta = \frac{\sum_i (u_i - \bar u)(v_i - \bar v)}{\sum_i (u_i -
\bar u)^2}, \qquad \log K = \bar v - \beta \bar u$$

$$R^2 = 1 - \frac{\sum_i (v_i - \hat v_i)^2}{\sum_i (v_i - \bar v)^2}$$

At least 5 valid sample points are required to fit. Empirically, natural
English text tends to show $\beta$ roughly in the 0.4–0.6 range (this
range is cited only as a rough qualitative anchor from the wider
information-retrieval literature — the report itself never compares your
$\beta$ against it numerically, only reports your own $\beta$ and $R^2$).
A low $R^2$ would mean the power-law model is a poor fit for your data
(vocabulary growth is noisy/irregular rather than smoothly sub-linear); a
$\beta$ well below the typical band suggests unusually repetitive,
templated language relative to corpus length.

### A.6 Covariance-shrinkage Mahalanobis distance to the optimal point

Let $\mathbf{x} = (s_1, s_2, s_3, s_4)$ be the current run's four axis
scores (specificity, context anchoring, structure, efficiency) and
$\boldsymbol{\mu}_{\text{opt}} = (100, 100, 100, 100)$ the optimal corner
(the provable ceiling of the scoring formulas — see the Methodology tab).
A naive Euclidean distance $\lVert \mathbf{x} - \boldsymbol{\mu}_{\text{opt}}
\rVert$ implicitly assumes the four axes are independent and equally
scaled, which is false in practice: e.g. structure and specificity tend to
move together for a given user because both respond to the same underlying
habit (writing detailed, well-scoped prompts). The Mahalanobis distance
corrects for this by using the empirical covariance of the user's own axis
scores across their run history, $\Sigma$, estimated from $n$ prior runs'
score vectors $\mathbf{x}_1, \dots, \mathbf{x}_n$:

$$\Sigma_{\text{sample}} = \frac{1}{n-1}\sum_{i=1}^{n} (\mathbf{x}_i -
\bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^{\mathsf T}$$

With only a handful of historical runs (as is typical early on), $n \ll 4$
degrees of freedom makes $\Sigma_{\text{sample}}$ ill-conditioned or
singular, so a pure inverse is numerically unstable (this was observed
directly during development — an unregularized inverse produced distances
in the tens of thousands). The fix is Ledoit-Wolf-style linear shrinkage
toward a scaled identity matrix, with shrinkage intensity $\alpha \in
[0, 1]$ (`mahalanobis.ridge_epsilon` in `config.yaml`, default $0.25$):

$$\Sigma_{\text{reg}} = (1 - \alpha)\, \Sigma_{\text{sample}} + \alpha\,
\bar{\sigma}^2 I, \qquad \bar{\sigma}^2 = \frac{\operatorname{tr}(
\Sigma_{\text{sample}})}{4}$$

This blends the sample covariance with an isotropic (axis-independent)
estimate whose scale matches the average observed variance, guaranteeing
$\Sigma_{\text{reg}}$ is well-conditioned and invertible for any $\alpha >
0$, while still recovering the raw sample covariance's directional
structure when $\alpha$ is small and history is long. The reported distance
is then the standard quadratic form, with matrix inversion done via
Gauss-Jordan elimination in `lib/linalg.py` (no numpy dependency):

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu}_{\text{opt}})^{
\mathsf T} \Sigma_{\text{reg}}^{-1} (\mathbf{x} -
\boldsymbol{\mu}_{\text{opt}})}$$

At least `mahalanobis.min_history_runs` prior runs (default 3) are required
before this is computed at all; below that, the report shows the plain
Euclidean distance instead with an explicit note, since shrinkage cannot
meaningfully compensate for near-zero history. A smaller $D_M$ means your
current position is close to optimal *relative to your own typical
variability* — moving along the direction your scores naturally co-vary in
costs less distance than moving against it, which is the entire point of
using the Mahalanobis metric over Euclidean distance here.

### A.7 Z-score anomaly detection against prior-run baselines

For each tracked metric $m$ (turn count, prompt length, correction rate,
file-mention rate, session length, etc. — see `lib/anomaly.py::
TRACKED_METRICS`), let $v_1, \dots, v_n$ be that metric's values from every
*prior* run (explicitly excluding the run currently being scored, so a
metric can never be judged anomalous against a baseline that already
contains it) and $v_{\text{now}}$ the current run's value. The z-score is:

$$z = \frac{v_{\text{now}} - \bar v}{s}, \qquad \bar v = \frac{1}{n}
\sum_{i=1}^n v_i, \qquad s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (v_i - \bar
v)^2}$$

requiring $n \ge$ `anomaly_detection.min_runs` (default 3) prior runs, and
guarding against $s = 0$ (a metric that has never varied) by treating any
non-zero deviation from a constant baseline as flagged regardless of the
formal z-score. Each metric is classified using two configurable
thresholds, `anomaly_detection.watch_z` (default 1.5) and
`anomaly_detection.flag_z` (default 2.5):

$$
\text{status}(z) =
\begin{cases}
\text{normal} & |z| < \text{watch\_z} \\
\text{watch} & \text{watch\_z} \le |z| < \text{flag\_z} \\
\text{flagged} & |z| \ge \text{flag\_z}
\end{cases}
$$

For metrics where a *known* direction is undesirable (e.g. correction rate
going up, or file-mention rate going down — see `TRACKED_METRICS`'s
recorded "bad direction" per metric), the flag/watch label additionally
notes whether the deviation is in the concerning direction or the
beneficial one, so a large positive swing in, say, task-completion rate is
still surfaced as a statistically notable event but not miscast as a
regression. This is a two-sided within-subject control-chart approach
(closely related to Shewhart control charts / statistical process control
control limits), applied per-user so that "normal" is always calibrated to
that person's own historical variability rather than a fixed universal
cutoff.
