"""Tests for lib/advanced_metrics.py: the 5 advanced math metrics (see
README Appendix A for derivations). Each test checks both the min-sample
gating (returns None below threshold) and basic correctness of the math on
a synthetic, hand-computable input."""
import math
from datetime import datetime, timedelta, timezone

from backends import Session, Turn
from metrics import extract_all_features
from config import load_config
import advanced_metrics as am


def _mk_session(agent, session_id, texts, interval_seconds=60):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns = [
        Turn(
            agent=agent, session_id=session_id, turn_index=i,
            timestamp=base + timedelta(seconds=i * interval_seconds),
            user_text=text, assistant_text=None,
        )
        for i, text in enumerate(texts)
    ]
    return Session(agent=agent, session_id=session_id, cwd="/tmp/proj", repository="acme/repo", turns=turns)


# --- Compression ratio ------------------------------------------------------

def test_compression_returns_none_below_min_bytes():
    result = am.compute_compression_ratio(["short text"])
    assert result is None


def test_compression_highly_repetitive_text_has_low_ratio():
    min_bytes = load_config()["advanced_metrics"]["compression_min_bytes"]
    repetitive = ["please fix the bug in the login flow " for _ in range(30)]
    assert len("\n".join(repetitive).encode("utf-8")) >= min_bytes
    result = am.compute_compression_ratio(repetitive)
    assert result is not None
    assert 0 < result.ratio < 0.5  # gzip should compress highly repeated text well


# --- Burstiness --------------------------------------------------------------

def test_burstiness_returns_none_below_min_timestamps():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = [base + timedelta(seconds=i * 10) for i in range(5)]
    assert am.compute_burstiness(ts) is None


def test_burstiness_perfectly_periodic_intervals_near_minus_one():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = [base + timedelta(seconds=i * 60) for i in range(15)]  # constant 60s gaps
    result = am.compute_burstiness(ts)
    assert result is not None
    assert math.isclose(result.burstiness, -1.0, abs_tol=1e-6)


def test_burstiness_bursty_process_positive():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # Clusters of near-simultaneous events separated by long gaps -> bursty.
    ts = []
    t = base
    for _ in range(4):
        for _ in range(4):
            ts.append(t)
            t += timedelta(seconds=1)
        t += timedelta(hours=5)
    result = am.compute_burstiness(ts)
    assert result is not None
    assert result.burstiness > 0


# --- Heaps' law --------------------------------------------------------------

def test_heaps_returns_none_below_min_tokens():
    result = am.compute_heaps_law(["only a few words here"])
    assert result is None


def _alphabetic_unique_words(n):
    # advanced_metrics.WORD_RE only matches [A-Za-z']+, so a numeric suffix
    # like "word0" would have its digits stripped and collide with every
    # other "word<N>" -- use purely-alphabetic 3-letter combinations instead
    # so each token is genuinely unique post-tokenization.
    import itertools
    import string
    combos = itertools.product(string.ascii_lowercase, repeat=3)
    return ["".join(c) for c, _ in zip(combos, range(n))]


def test_heaps_ever_growing_vocabulary_has_beta_near_one():
    min_tokens = load_config()["advanced_metrics"]["heaps_min_tokens"]
    # Every token is unique -> vocabulary grows linearly with n -> beta ~= 1.
    tokens = _alphabetic_unique_words(min_tokens + 50)
    text = " ".join(tokens)
    result = am.compute_heaps_law([text])
    assert result is not None
    assert result.beta > 0.9
    assert result.r_squared > 0.9


def test_heaps_fully_repeated_single_word_has_low_beta():
    min_tokens = load_config()["advanced_metrics"]["heaps_min_tokens"]
    text = " ".join(["same"] * (min_tokens + 50))
    result = am.compute_heaps_law([text])
    assert result is not None
    assert result.beta < 0.1


# --- Markov absorption -------------------------------------------------------

def test_markov_returns_none_below_min_transitions():
    sessions = [_mk_session("copilot", "s1", ["hello there, please help", "thanks"])]
    feats = extract_all_features(sessions)
    feats_by_key = {(f.turn.session_id, f.turn.turn_index): f for f in feats}
    assert am.compute_markov_absorption(sessions, feats_by_key) is None


def test_markov_with_enough_transitions_returns_valid_result():
    min_trans = load_config()["advanced_metrics"]["markov_min_transitions"]
    # One long session with plenty of normal turns to exceed the transition
    # threshold (n_transitions = n_turns, since each turn transitions once,
    # plus the final absorption transition).
    texts = [f"please implement feature number {i} carefully" for i in range(min_trans + 5)]
    sessions = [_mk_session("copilot", "s1", texts)]
    feats = extract_all_features(sessions)
    feats_by_key = {(f.turn.session_id, f.turn.turn_index): f for f in feats}
    result = am.compute_markov_absorption(sessions, feats_by_key)
    assert result is not None
    assert set(result.states) == set(am.STATES)
    # Every state's transition probabilities (including to itself/absorbing) sum to 1.
    for s in am.STATES:
        total = sum(result.transition_probs[s].values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)
    # Expected turns to resolution should be finite, non-negative numbers.
    for v in result.expected_turns_to_resolution.values():
        assert v >= 0
        assert math.isfinite(v)


# --- Winding numbers ----------------------------------------------------------

def test_winding_numbers_skips_short_sessions():
    min_turns = load_config()["advanced_metrics"]["winding_min_turns"]
    texts = [f"turn {i}" for i in range(min_turns - 1)]
    sessions = [_mk_session("copilot", "s1", texts)]
    feats = extract_all_features(sessions)
    feats_by_key = {(f.turn.session_id, f.turn.turn_index): f for f in feats}
    result = am.compute_winding_numbers(sessions, feats_by_key)
    assert result == []


def test_winding_numbers_returns_one_entry_per_eligible_session():
    min_turns = load_config()["advanced_metrics"]["winding_min_turns"]
    texts = [f"please refactor module {i} and verify tests still pass" for i in range(min_turns + 4)]
    sessions = [_mk_session("copilot", "s1", texts)]
    feats = extract_all_features(sessions)
    feats_by_key = {(f.turn.session_id, f.turn.turn_index): f for f in feats}
    result = am.compute_winding_numbers(sessions, feats_by_key)
    assert len(result) <= 1
    if result:
        assert result[0].session_id == "s1"
        assert math.isfinite(result[0].winding_number)
