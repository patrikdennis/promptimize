"""Tests for lib/skills.py: the Leveling tab's XP curve, per-turn XP rules,
and the Total level / Prompt Level composites."""
from datetime import datetime, timedelta, timezone

import pytest

from backends import Session, Turn
from config import DEFAULTS
from metrics import extract_all_features
import skills


def _mk_session(agent, session_id, texts):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns = [
        Turn(agent=agent, session_id=session_id, turn_index=i, timestamp=base + timedelta(minutes=i),
             user_text=text, assistant_text=None)
        for i, text in enumerate(texts)
    ]
    return Session(agent=agent, session_id=session_id, cwd="/tmp/p", repository="acme/repo", turns=turns)


def _feats(texts):
    sessions = [_mk_session("copilot", "s1", texts)]
    return extract_all_features(sessions)


# --- XP curve --------------------------------------------------------------

def test_xp_for_level_1_is_zero():
    assert skills.xp_for_level(1, DEFAULTS) == 0.0


def test_xp_for_level_is_monotonically_increasing():
    xps = [skills.xp_for_level(lv, DEFAULTS) for lv in range(1, skills.MAX_LEVEL + 1)]
    assert all(b > a for a, b in zip(xps, xps[1:]))


def test_xp_curve_is_steeper_at_high_levels_than_low_levels():
    # Classic MMO-shaped curve: the XP gap between levels 90->99 should dwarf
    # the gap between levels 1->10 (this is what "slow grind to max" means).
    early_gap = skills.xp_for_level(10, DEFAULTS) - skills.xp_for_level(1, DEFAULTS)
    late_gap = skills.xp_for_level(99, DEFAULTS) - skills.xp_for_level(90, DEFAULTS)
    assert late_gap > early_gap * 10


def test_level_for_xp_roundtrips_through_xp_for_level():
    for lv in [1, 10, 30, 50, 70, 90, 99]:
        xp = skills.xp_for_level(lv, DEFAULTS)
        assert skills.level_for_xp(xp, DEFAULTS) == lv


def test_level_for_xp_zero_is_level_1():
    assert skills.level_for_xp(0.0, DEFAULTS) == 1


def test_level_for_xp_caps_at_max_level():
    assert skills.level_for_xp(10**9, DEFAULTS) == skills.MAX_LEVEL


# --- Per-turn boolean skill XP ----------------------------------------------

def test_specificity_xp_awarded_for_concrete_turn_no_vague_language():
    feats = _feats(["fix the null pointer bug in the login handler please"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["specificity"] == DEFAULTS["skills"]["xp_per_qualifying_turn"]


def test_specificity_no_xp_for_vague_turn():
    feats = _feats(["fix it, i guess, not sure what's wrong, maybe something"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["specificity"] == 0.0


def test_context_anchoring_xp_for_file_reference():
    feats = _feats(["please update `src/app.py` to handle the edge case"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["context_anchoring"] == DEFAULTS["skills"]["xp_per_qualifying_turn"]


def test_context_anchoring_no_xp_without_file_or_code_ref():
    feats = _feats(["please improve the login flow overall experience"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["context_anchoring"] == 0.0


def test_structure_acceptance_xp_for_explicit_acceptance_criterion():
    feats = _feats(["refactor the parser, make sure existing tests still pass"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["structure_acceptance"] == DEFAULTS["skills"]["xp_per_qualifying_turn"]


def test_efficiency_no_xp_when_turn_is_itself_a_correction():
    feats = _feats(["that's wrong, try again please"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["efficiency"] == 0.0


def test_efficiency_xp_for_non_correction_turn():
    feats = _feats(["add a new endpoint for exporting reports"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["efficiency"] == DEFAULTS["skills"]["xp_per_qualifying_turn"]


def test_context_retention_no_xp_on_first_turn_in_session():
    # First turn has no prior turn to compare against -> not eligible,
    # 0 XP, not a penalty either.
    feats = _feats(["implement the new caching layer for the api"])
    xp = skills.compute_boolean_skill_xp(feats[0], DEFAULTS)
    assert xp["context_retention"] == 0.0


def test_context_retention_xp_when_second_turn_does_not_restate():
    feats = _feats([
        "implement the new caching layer for the api",
        "now add a unit test for the eviction policy",
    ])
    xp = skills.compute_boolean_skill_xp(feats[1], DEFAULTS)
    assert xp["context_retention"] == DEFAULTS["skills"]["xp_per_qualifying_turn"]


# --- Vocabulary XP -----------------------------------------------------------

def test_turn_candidate_words_filters_stopwords_and_short_words():
    words = skills.turn_candidate_words("please fix the login bug in authentication module")
    assert "login" in words
    assert "authentication" in words
    assert "please" not in words  # stopword
    assert "the" not in words  # too short / stopword


def test_compute_vocabulary_xp_scales_with_new_word_count_and_caps():
    cfg = DEFAULTS
    per_word = cfg["skills"]["vocabulary_xp_per_new_word"]
    cap = cfg["skills"]["vocabulary_xp_cap_per_turn"]
    assert skills.compute_vocabulary_xp(3, cfg) == 3 * per_word
    assert skills.compute_vocabulary_xp(cap + 50, cfg) == cap * per_word


# --- Total level / Prompt Level composites ----------------------------------

def test_build_skill_progress_all_zero_xp_gives_level_1_for_every_skill():
    progress = skills.build_skill_progress({}, DEFAULTS)
    assert len(progress) == len(skills.SKILL_KEYS)
    assert all(sp.level == 1 for sp in progress.values())
    assert all(not sp.is_maxed for sp in progress.values())


def test_total_level_is_sum_of_skill_levels():
    progress = skills.build_skill_progress({}, DEFAULTS)
    assert skills.compute_total_level(progress) == len(skills.SKILL_KEYS) * 1


def test_total_level_at_max_xp_is_max_possible():
    xp_totals = {k: 10**9 for k in skills.SKILL_KEYS}
    progress = skills.build_skill_progress(xp_totals, DEFAULTS)
    assert skills.compute_total_level(progress) == len(skills.SKILL_KEYS) * skills.MAX_LEVEL


def test_prompt_level_is_bounded_between_1_and_99():
    progress = skills.build_skill_progress({}, DEFAULTS)
    assert 1 <= skills.compute_prompt_level(progress, DEFAULTS) <= skills.MAX_LEVEL

    xp_totals = {k: 10**9 for k in skills.SKILL_KEYS}
    progress_max = skills.build_skill_progress(xp_totals, DEFAULTS)
    assert skills.compute_prompt_level(progress_max, DEFAULTS) == skills.MAX_LEVEL


def test_prompt_level_weights_sum_to_one():
    weights = DEFAULTS["skills"]["prompt_level_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_skill_progress_reports_xp_to_next_level_correctly():
    xp_for_10 = skills.xp_for_level(10, DEFAULTS)
    progress = skills.build_skill_progress({"specificity": xp_for_10}, DEFAULTS)
    sp = progress["specificity"]
    assert sp.level == 10
    assert sp.xp_into_level == 0.0
    assert sp.xp_for_next_level == skills.xp_for_level(11, DEFAULTS) - xp_for_10
