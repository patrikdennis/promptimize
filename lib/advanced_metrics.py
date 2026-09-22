"""
Advanced, mathematically-grounded metrics.

Unlike the rate-based heuristics in metrics.py, every metric here is derived
from an explicit, named piece of mathematics (absorbing Markov chains,
discrete winding numbers, point-process burstiness, compression-based
complexity, Heaps' law vocabulary growth) rather than an arbitrary weighted
formula. Full derivations for each are in README.md, Appendix A.

All functions degrade gracefully (return None / empty structures) when there
isn't enough data for the underlying math to be meaningful, and each result
carries the sample size it was computed from.
"""
from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass, field

from backends import Session
from metrics import TurnFeatures
from config import load_config
from linalg import invert_matrix

WORD_RE = re.compile(r"[A-Za-z']+")

# ---------------------------------------------------------------------------
# 1. Absorbing Markov chain over turn states -> expected turns to resolution
# ---------------------------------------------------------------------------
# States are mutually exclusive and assigned by priority: a turn that is both
# a "correction" and "vague" is classified as Correction, since correction is
# the more consequential signal. Every session's last turn transitions to the
# absorbing state Resolved (the session ended there).

STATES = ["Normal", "Vague", "Clarification", "Correction"]
ABSORBING = "Resolved"


def classify_state(f: TurnFeatures) -> str:
    if f.correction_signal:
        return "Correction"
    if f.clarification_signal:
        return "Clarification"
    if f.vague_hits > 0:
        return "Vague"
    return "Normal"


@dataclass
class MarkovResult:
    states: list[str]
    transition_counts: dict
    transition_probs: dict
    expected_turns_to_resolution: dict  # state -> float
    n_transitions: int


def compute_markov_absorption(sessions: list[Session], feats_by_key: dict) -> MarkovResult | None:
    """feats_by_key maps (session_id, turn_index) -> TurnFeatures."""
    counts = {s: {s2: 0 for s2 in STATES + [ABSORBING]} for s in STATES}
    n_transitions = 0

    for s in sessions:
        ordered = sorted(s.turns, key=lambda t: (t.timestamp, t.turn_index))
        states_seq = []
        for t in ordered:
            f = feats_by_key.get((s.session_id, t.turn_index))
            if f is None:
                continue
            states_seq.append(classify_state(f))
        for i in range(len(states_seq) - 1):
            counts[states_seq[i]][states_seq[i + 1]] += 1
            n_transitions += 1
        if states_seq:
            counts[states_seq[-1]][ABSORBING] += 1
            n_transitions += 1

    if n_transitions < load_config()["advanced_metrics"]["markov_min_transitions"]:
        return None

    probs = {}
    for s in STATES:
        total = sum(counts[s].values())
        if total == 0:
            # No observed outgoing transitions from this state: assume it goes
            # straight to Resolved (conservative / avoids a divide-by-zero row).
            probs[s] = {s2: (1.0 if s2 == ABSORBING else 0.0) for s2 in STATES + [ABSORBING]}
        else:
            probs[s] = {s2: counts[s][s2] / total for s2 in STATES + [ABSORBING]}

    # Q: transient-to-transient sub-matrix, in STATES order.
    q = [[probs[si][sj] for sj in STATES] for si in STATES]
    identity_minus_q = [[(1.0 if i == j else 0.0) - q[i][j] for j in range(len(STATES))] for i in range(len(STATES))]
    fundamental = invert_matrix(identity_minus_q)
    if fundamental is None:
        return None

    # Expected steps to absorption from state i = row sum of N = (I-Q)^-1.
    expected = {STATES[i]: sum(fundamental[i]) for i in range(len(STATES))}

    return MarkovResult(
        states=STATES,
        transition_counts=counts,
        transition_probs=probs,
        expected_turns_to_resolution=expected,
        n_transitions=n_transitions,
    )


# ---------------------------------------------------------------------------
# 2. Discrete winding number -> circular-argument detector
# ---------------------------------------------------------------------------
# Each session is embedded as a 2D lattice path:
#   x_t = cumulative sign(word_count_t - word_count_{t-1})       (length-trend walk)
#   y_t = cumulative (+1 correction, -1 acceptance-criteria, 0 otherwise)
# The winding number of this path about its own centroid is the sum of
# consecutive signed turning angles divided by 2*pi (discrete Hopf Umlaufsatz /
# turning-angle theorem). A session whose path winds fully around its own
# centroid (|W| >= 1) has, in this embedding, returned to a similar relative
# state repeatedly without net escape -- a geometric signature of circular,
# unresolved back-and-forth.

@dataclass
class SessionWinding:
    session_id: str
    n_turns: int
    winding_number: float


def _wrap_angle(delta: float) -> float:
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return delta


def compute_winding_numbers(sessions: list[Session], feats_by_key: dict) -> list[SessionWinding]:
    min_turns = load_config()["advanced_metrics"]["winding_min_turns"]
    results = []
    for s in sessions:
        ordered = sorted(s.turns, key=lambda t: (t.timestamp, t.turn_index))
        if len(ordered) < min_turns:
            continue
        xs, ys = [0.0], [0.0]
        prev_wc = None
        for t in ordered:
            f = feats_by_key.get((s.session_id, t.turn_index))
            if f is None:
                continue
            wc = f.word_count
            dx = 0.0 if prev_wc is None else (1.0 if wc > prev_wc else (-1.0 if wc < prev_wc else 0.0))
            dy = 1.0 if f.correction_signal else (-1.0 if f.acceptance_signal else 0.0)
            xs.append(xs[-1] + dx)
            ys.append(ys[-1] + dy)
            prev_wc = wc

        if len(xs) < 4:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        angles = []
        for x, y in zip(xs, ys):
            rx, ry = x - cx, y - cy
            if abs(rx) < 1e-9 and abs(ry) < 1e-9:
                continue
            angles.append(math.atan2(ry, rx))
        if len(angles) < 3:
            continue

        total_turn = sum(_wrap_angle(angles[i + 1] - angles[i]) for i in range(len(angles) - 1))
        winding_number = total_turn / (2 * math.pi)
        results.append(SessionWinding(session_id=s.session_id, n_turns=len(ordered), winding_number=winding_number))
    return results


# ---------------------------------------------------------------------------
# 3. Burstiness and memory of the prompting point process (Goh & Barabasi, 2008)
# ---------------------------------------------------------------------------
# For inter-event times {tau_i} with mean m and std deviation s:
#   B = (s - m) / (s + m)                         burstiness, in [-1, 1]
#   M = corr(tau_i, tau_{i+1})  for consecutive pairs   memory coefficient
# B = -1: perfectly periodic: B = 0: Poisson (memoryless); B -> 1: heavy-tailed/bursty.

@dataclass
class BurstinessResult:
    n_intervals: int
    mean_interval_s: float
    std_interval_s: float
    burstiness: float
    memory: float | None


def compute_burstiness(timestamps: list) -> BurstinessResult | None:
    min_ts = load_config()["advanced_metrics"]["burstiness_min_timestamps"]
    ts = sorted(timestamps)
    if len(ts) < min_ts:
        return None
    intervals = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
    intervals = [x for x in intervals if x >= 0]
    n = len(intervals)
    if n < 9:
        return None
    mean = sum(intervals) / n
    var = sum((x - mean) ** 2 for x in intervals) / n
    std = math.sqrt(var)
    if (std + mean) == 0:
        burstiness = 0.0
    else:
        burstiness = (std - mean) / (std + mean)

    memory = None
    if n >= 10:
        a = intervals[:-1]
        b = intervals[1:]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        sa = math.sqrt(sum((x - ma) ** 2 for x in a) / len(a))
        sb = math.sqrt(sum((x - mb) ** 2 for x in b) / len(b))
        if sa > 0 and sb > 0:
            cov = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a))) / len(a)
            memory = cov / (sa * sb)

    return BurstinessResult(
        n_intervals=n, mean_interval_s=mean, std_interval_s=std,
        burstiness=burstiness, memory=memory,
    )


# ---------------------------------------------------------------------------
# 4. Compression-ratio complexity proxy (Kolmogorov complexity, approximated)
# ---------------------------------------------------------------------------
# Kolmogorov complexity K(x) (the length of the shortest program producing x)
# is uncomputable; gzip compressed size is a standard, well-established
# computable upper-bound proxy for it (used e.g. in Normalized Compression
# Distance, Cilibrasi & Vitanyi 2005). NCR = |gzip(x)| / |x| : lower means
# more algorithmically redundant (repetitive phrasing across turns), higher
# means more novel/information-dense text turn to turn.

@dataclass
class CompressionResult:
    raw_bytes: int
    compressed_bytes: int
    ratio: float


def compute_compression_ratio(texts: list[str]) -> CompressionResult | None:
    corpus = "\n".join(texts).encode("utf-8")
    if len(corpus) < load_config()["advanced_metrics"]["compression_min_bytes"]:
        return None
    compressed = gzip.compress(corpus, compresslevel=9)
    return CompressionResult(raw_bytes=len(corpus), compressed_bytes=len(compressed), ratio=len(compressed) / len(corpus))


# ---------------------------------------------------------------------------
# 5. Heaps' law -> vocabulary growth exponent
# ---------------------------------------------------------------------------
# Empirical linguistic law: distinct vocabulary size V(n) as a function of
# total tokens seen n follows V(n) = K * n^beta. beta is estimated by
# ordinary least squares on log(V) vs log(n) over log-spaced sample points.
# beta near 1 => vocabulary keeps expanding turn to turn (little repeated
# phrasing); beta near 0 => vocabulary saturates quickly (heavy reuse of the
# same words/templates).

@dataclass
class HeapsResult:
    beta: float
    k: float
    r_squared: float
    n_tokens: int
    n_samples: int


def compute_heaps_law(texts_in_order: list[str]) -> HeapsResult | None:
    am_cfg = load_config()["advanced_metrics"]
    min_tokens = am_cfg["heaps_min_tokens"]
    min_samples = am_cfg["heaps_min_samples"]
    base = am_cfg["heaps_sample_base"]
    ratio = am_cfg["heaps_sample_ratio"]
    points = int(am_cfg["heaps_sample_points"])

    tokens = []
    for text in texts_in_order:
        tokens.extend(w.lower() for w in WORD_RE.findall(text))
    n_total = len(tokens)
    if n_total < min_tokens:
        return None

    sample_ns = sorted(set(
        min(n_total, max(10, int(round(base * (ratio ** i)))))
        for i in range(0, points)
    ))
    sample_ns = [n for n in sample_ns if 10 <= n <= n_total]
    if len(sample_ns) < min_samples:
        return None

    seen = set()
    vocab_at = {}
    sample_set = set(sample_ns)
    for i, tok in enumerate(tokens, start=1):
        seen.add(tok)
        if i in sample_set:
            vocab_at[i] = len(seen)
    if len(vocab_at) < min_samples:
        return None

    xs = [math.log(n) for n in vocab_at.keys()]
    ys = [math.log(v) for v in vocab_at.values()]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    if sxx == 0:
        return None
    beta = sxy / sxx
    log_k = my - beta * mx
    k = math.exp(log_k)

    y_pred = [log_k + beta * x for x in xs]
    ss_res = sum((ys[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return HeapsResult(beta=beta, k=k, r_squared=r_squared, n_tokens=n_total, n_samples=n)


@dataclass
class AdvancedMetrics:
    markov: MarkovResult | None
    windings: list[SessionWinding]
    burstiness: BurstinessResult | None
    compression: CompressionResult | None
    heaps: HeapsResult | None


def compute_all_advanced_metrics(sessions: list[Session], feats: list[TurnFeatures]) -> AdvancedMetrics:
    feats_by_key = {(f.turn.session_id, f.turn.turn_index): f for f in feats}
    ordered_feats = sorted(feats, key=lambda f: f.turn.timestamp)
    texts_in_order = [f.turn.user_text for f in ordered_feats]
    timestamps = [f.turn.timestamp for f in ordered_feats]

    return AdvancedMetrics(
        markov=compute_markov_absorption(sessions, feats_by_key),
        windings=compute_winding_numbers(sessions, feats_by_key),
        burstiness=compute_burstiness(timestamps),
        compression=compute_compression_ratio(texts_in_order),
        heaps=compute_heaps_law(texts_in_order),
    )
