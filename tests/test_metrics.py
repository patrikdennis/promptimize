"""Tests for lib/metrics.py: turn-level feature extraction and the 0-100
scoring formulas."""
from datetime import datetime, timedelta, timezone

from backends import Session, Turn
from metrics import build_aggregate, extract_all_features


def _mk_session(agent, session_id, texts, assistant_texts=None):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assistant_texts = assistant_texts or [None] * len(texts)
    turns = [
        Turn(
            agent=agent,
            session_id=session_id,
            turn_index=i,
            timestamp=base + timedelta(minutes=i),
            user_text=text,
            assistant_text=assistant_texts[i],
        )
        for i, text in enumerate(texts)
    ]
    return Session(agent=agent, session_id=session_id, cwd="/tmp/proj", repository="acme/repo", turns=turns)


def test_empty_input_gives_zero_scores():
    agg = build_aggregate([], [])
    assert agg.scores == {"specificity": 0, "context": 0, "structure": 0, "efficiency": 0, "overall": 0}


def test_specific_file_referenced_prompt_scores_high_context():
    sessions = [_mk_session("copilot", "s1", ["fix the bug in `src/app.py` handling null users please make sure tests still pass"])]
    feats = extract_all_features(sessions)
    agg = build_aggregate(feats, sessions)
    assert agg.scores["context"] == 100.0
    assert agg.metric_values["file_ref_rate"] == 1.0


def test_vague_prompt_scores_low_specificity():
    sessions = [_mk_session("copilot", "s1", ["do something, i guess maybe fix stuff, not sure what's wrong"])]
    feats = extract_all_features(sessions)
    agg = build_aggregate(feats, sessions)
    assert agg.metric_values["vague_rate"] == 1.0
    assert agg.scores["specificity"] < 50


def test_correction_turn_lowers_efficiency():
    clean = [_mk_session("copilot", "s1", ["please implement the login flow and make sure existing tests still pass"])]
    with_correction = [_mk_session("copilot", "s2", [
        "please implement the login flow and make sure existing tests still pass",
        "that's wrong, try again, it doesn't work",
    ])]

    feats_clean = extract_all_features(clean)
    agg_clean = build_aggregate(feats_clean, clean)

    feats_bad = extract_all_features(with_correction)
    agg_bad = build_aggregate(feats_bad, with_correction)

    assert agg_bad.correction_rate > agg_clean.correction_rate
    assert agg_bad.scores["efficiency"] < agg_clean.scores["efficiency"]


def test_single_shot_rate_counts_one_turn_sessions():
    sessions = [
        _mk_session("copilot", "s1", ["only turn"]),
        _mk_session("copilot", "s2", ["first turn", "second turn"]),
    ]
    feats = extract_all_features(sessions)
    agg = build_aggregate(feats, sessions)
    assert agg.single_shot_rate == 0.5


def test_multi_ask_detected_with_two_plus_action_verbs():
    sessions = [_mk_session("copilot", "s1", ["please implement the endpoint and also refactor the old client, then test everything"])]
    feats = extract_all_features(sessions)
    agg = build_aggregate(feats, sessions)
    assert agg.metric_values["multi_ask_rate"] == 1.0


def test_context_restatement_detected_across_similar_consecutive_turns():
    similar_text_a = "please refactor the user authentication module carefully handling edge cases properly"
    similar_text_b = "please refactor the user authentication module carefully handling edge cases nicely"
    sessions = [_mk_session("copilot", "s1", [similar_text_a, similar_text_b])]
    feats = extract_all_features(sessions)
    agg = build_aggregate(feats, sessions)
    assert agg.context_restatement_rate > 0.0


def test_clarification_signal_from_short_question_reply():
    sessions = [_mk_session(
        "copilot", "s1", ["update the config"],
        assistant_texts=["Which config file did you mean, the dev one or the prod one?"],
    )]
    feats = extract_all_features(sessions)
    agg = build_aggregate(feats, sessions)
    assert agg.metric_values["clarification_rate"] == 1.0


def test_scores_are_bounded_zero_to_hundred():
    sessions = [_mk_session("copilot", "s1", [
        "something maybe fix stuff i guess kind of not sure whatever",
        "that's wrong, undo, revert, try again, it still doesn't work",
    ])]
    feats = extract_all_features(sessions)
    agg = build_aggregate(feats, sessions)
    for axis, value in agg.scores.items():
        assert 0 <= value <= 100, f"{axis} score {value} out of bounds"
