"""
Skill/XP/level model ("Leveling" tab): a classic MMO-style skilling layer on
top of the same per-turn features the rest of this tool already computes.

This module is intentionally free of any database/IO dependency so its
formulas are unit-testable in isolation: it only consumes `TurnFeatures`
(already extracted by `metrics.py`) and plain XP totals, and returns plain
data. Persistence (the `seen_turns` dedup ledger, the cumulative XP totals,
and the incremental vocabulary set) lives in `history.py`, which calls into
this module for the actual per-turn XP rule and the XP<->level curve.

Design summary (see README "Leveling" section + Appendix B for full
derivations and rationale):

  - 7 skills, each with a simple, inspectable, boolean-or-counted per-turn
    condition. A turn earns XP in a skill exactly when it satisfies that
    skill's condition -- the same underlying signals already used for the
    4 axis scores and the recommendation engine, just re-cut into a
    persistent, cumulative, gamified view instead of a per-run rate.
  - Levels 1-99 on an XP curve shaped like a classic MMO's (slow grind to reach
    the high levels, since `2**(n/growth)` grows geometrically) but rescaled
    by `scale_divisor` so that the early/mid levels are reachable within
    weeks of normal personal usage instead of requiring millions of actions.
  - "Total level" = sum of the 7 skill levels (a classic MMO-style "Total level").
  - "Prompt Level" = a single weighted-average composite across skill
    levels (config-driven weights), analogous to a combat level: one
    number that summarizes overall prompting proficiency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from metrics import TurnFeatures

MAX_LEVEL = 99

# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

SKILL_DEFINITIONS = [
    {
        "key": "specificity",
        "label": "Specificity",
        "description": (
            "Awarded for turns with no vague-language hits and at least the "
            "minimum word count (see Specificity axis, config.scoring.specificity). "
            "Rewards concrete, appropriately-detailed requests; there is no "
            "upper length limit, so useful context and rationale are never penalized."
        ),
    },
    {
        "key": "context_anchoring",
        "label": "Artifact Grounding",
        "description": (
            "Awarded for turns that reference a specific file or include a code span "
            "(the same signal behind the Context axis). Rewards grounding requests in "
            "concrete artifacts instead of unanchored descriptions."
        ),
    },
    {
        "key": "structure_acceptance",
        "label": "Structure & Acceptance Criteria",
        "description": (
            "Awarded for turns that state an explicit acceptance criterion (\"make sure\", "
            "\"without breaking\", \"verify\", etc). Rewards specifying how success will be judged."
        ),
    },
    {
        "key": "efficiency",
        "label": "Efficiency",
        "description": (
            "Awarded for turns that are not themselves a correction of a prior reply. "
            "Rewards a low correction rate -- steady forward progress rather than rework."
        ),
    },
    {
        "key": "clarity",
        "label": "Clarity",
        "description": (
            "Awarded for turns whose reply did not need to ask a clarifying question back. "
            "Rewards requests clear enough to be acted on directly."
        ),
    },
    {
        "key": "context_retention",
        "label": "Context Retention",
        "description": (
            "Awarded for turns that had a prior turn to build on (in the same session) and "
            "did not redundantly restate it (Jaccard similarity below the restatement "
            "threshold). Rewards trusting the model's short-term memory instead of re-explaining."
        ),
    },
    {
        "key": "vocabulary",
        "label": "Vocabulary",
        "description": (
            "Awarded incrementally for genuinely new distinct significant words contributed "
            "across your entire history (an online/incremental analogue of the Heaps' Law "
            "vocabulary-growth curve used in Advanced Analytics -- see Appendix A.5 and "
            "Appendix B). Rewards precise, varied language over repeating the same phrasing."
        ),
    },
]

SKILL_KEYS = [s["key"] for s in SKILL_DEFINITIONS]
BOOLEAN_SKILL_KEYS = [k for k in SKILL_KEYS if k != "vocabulary"]

# ---------------------------------------------------------------------------
# Per-turn XP rules (the 6 boolean skills; vocabulary is handled separately
# by the caller since it requires DB-backed set-membership -- see
# `turn_candidate_words` below and `history.award_skill_xp_for_new_turns`).
# ---------------------------------------------------------------------------


def _condition_specificity(f: TurnFeatures, cfg: dict) -> bool:
    spec = cfg["scoring"]["specificity"]
    return f.vague_hits == 0 and f.word_count >= spec["ideal_length_min"]


def _condition_context_anchoring(f: TurnFeatures, cfg: dict) -> bool:
    return f.has_file_ref or f.has_code_ref


def _condition_structure_acceptance(f: TurnFeatures, cfg: dict) -> bool:
    return f.acceptance_signal


def _condition_efficiency(f: TurnFeatures, cfg: dict) -> bool:
    return not f.correction_signal


def _condition_clarity(f: TurnFeatures, cfg: dict) -> bool:
    return not f.clarification_signal


def _condition_context_retention(f: TurnFeatures, cfg: dict) -> bool:
    return f.restatement_eligible and not f.context_restatement


_CONDITIONS = {
    "specificity": _condition_specificity,
    "context_anchoring": _condition_context_anchoring,
    "structure_acceptance": _condition_structure_acceptance,
    "efficiency": _condition_efficiency,
    "clarity": _condition_clarity,
    "context_retention": _condition_context_retention,
}


def compute_boolean_skill_xp(f: TurnFeatures, cfg: dict) -> dict:
    """Returns {skill_key: xp_awarded} for the 6 non-vocabulary skills, for
    a single (newly-seen) turn. `context_retention` awards nothing (0 XP,
    not a penalty) on turns with no prior turn to compare against."""
    per_turn = cfg["skills"]["xp_per_qualifying_turn"]
    out = {}
    for key in BOOLEAN_SKILL_KEYS:
        condition = _CONDITIONS[key]
        out[key] = float(per_turn) if condition(f, cfg) else 0.0
    return out


_WORD_RE = re.compile(r"[a-zA-Z']{4,}")
_VOCAB_STOPWORDS = {
    "that", "this", "with", "from", "have", "will", "your", "just", "please",
    "also", "into", "then", "than", "does", "should", "could", "would",
    "about", "make", "sure", "when", "what", "which", "want", "need",
}


def turn_candidate_words(text: str) -> set:
    """Tokenizes a turn's text into the distinct, lowercased, stopword-filtered
    candidate words eligible to count as "new vocabulary" for the Vocabulary
    skill. Deliberately simple (same style of filter as `metrics._significant_words`)
    -- this is a fast per-turn heuristic, not a linguistic analysis."""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return words - _VOCAB_STOPWORDS


def compute_vocabulary_xp(new_word_count: int, cfg: dict) -> float:
    """`new_word_count` = number of genuinely-new (never-seen-before, across
    all history) distinct candidate words in one turn, already capped by the
    caller against `duplicate` words within the same run. Caps per-turn XP so
    one unusually long/copy-pasted turn can't dominate the skill."""
    sk = cfg["skills"]
    capped = min(new_word_count, sk["vocabulary_xp_cap_per_turn"])
    return float(capped) * float(sk["vocabulary_xp_per_new_word"])


# ---------------------------------------------------------------------------
# XP <-> level curve
# ---------------------------------------------------------------------------


def _cumulative_xp_table(cfg: dict) -> list:
    """`table[level]` = total XP required to *reach* `level` (1-indexed;
    table[1] == 0). Shape follows a classic MMO XP formula --
    `floor(n + base * 2**(n/growth_divisor))` per level step, summed and
    divided by `scale_divisor` -- but `scale_divisor` is far larger than
    that reference curve's constant 4, compressing total XP-to-99 from ~13.03M down to a
    scale reachable by a real personal-usage volume of qualifying turns.
    See Appendix B for the derivation and the reasoning behind the default
    scale_divisor=300 (~75x compression of that reference curve)."""
    curve = cfg["skills"]["xp_curve"]
    base = curve["base"]
    growth = curve["growth_divisor"]
    divisor = curve["scale_divisor"]

    table = [0.0] * (MAX_LEVEL + 1)
    running = 0.0
    for level in range(2, MAX_LEVEL + 1):
        n = level - 1
        running += int(n + base * (2 ** (n / growth)))
        table[level] = float(int(running / divisor))
    return table


_table_cache: dict = {}


def _table_for(cfg: dict) -> list:
    # Cheap to recompute (99 iterations), but cache per distinct curve config
    # (there is normally exactly one) to avoid redoing it on every call.
    curve = cfg["skills"]["xp_curve"]
    key = (curve["base"], curve["growth_divisor"], curve["scale_divisor"])
    if key not in _table_cache:
        _table_cache[key] = _cumulative_xp_table(cfg)
    return _table_cache[key]


def xp_for_level(level: int, cfg: dict) -> float:
    """Total cumulative XP required to reach `level` (1-99)."""
    level = max(1, min(MAX_LEVEL, level))
    return _table_for(cfg)[level]


def level_for_xp(xp: float, cfg: dict) -> int:
    """Highest level whose XP requirement `xp` has met or exceeded."""
    table = _table_for(cfg)
    level = 1
    for lvl in range(1, MAX_LEVEL + 1):
        if xp >= table[lvl]:
            level = lvl
        else:
            break
    return level


@dataclass
class SkillProgress:
    key: str
    label: str
    description: str
    xp: float
    level: int
    xp_into_level: float
    xp_for_next_level: float | None  # None at max level
    progress_pct: float  # 0-100, progress toward next level (100 if maxed)
    is_maxed: bool


def build_skill_progress(xp_totals: dict, cfg: dict) -> dict:
    """`xp_totals`: {skill_key: cumulative_xp}. Returns {skill_key: SkillProgress}."""
    out = {}
    for s in SKILL_DEFINITIONS:
        key = s["key"]
        xp = float(xp_totals.get(key, 0.0))
        level = level_for_xp(xp, cfg)
        is_maxed = level >= MAX_LEVEL
        floor_xp = xp_for_level(level, cfg)
        if is_maxed:
            xp_into_level = xp - floor_xp
            xp_for_next = None
            progress_pct = 100.0
        else:
            next_xp = xp_for_level(level + 1, cfg)
            xp_into_level = xp - floor_xp
            xp_for_next = next_xp - floor_xp
            progress_pct = 0.0 if xp_for_next <= 0 else min(100.0, 100.0 * xp_into_level / xp_for_next)
        out[key] = SkillProgress(
            key=key,
            label=s["label"],
            description=s["description"],
            xp=xp,
            level=level,
            xp_into_level=xp_into_level,
            xp_for_next_level=xp_for_next,
            progress_pct=progress_pct,
            is_maxed=is_maxed,
        )
    return out


def compute_total_level(skill_progress: dict) -> int:
    """Classic-MMO-style "Total level": the sum of all individual skill levels."""
    return sum(sp.level for sp in skill_progress.values())


def compute_prompt_level(skill_progress: dict, cfg: dict) -> int:
    """Single composite level (1-99): a config-weighted average of skill
    levels, analogous to a combat/overall level. Weights are documented in
    config.yaml -> skills.prompt_level_weights and sum to 1.0."""
    weights = cfg["skills"]["prompt_level_weights"]
    total_weight = sum(weights.values()) or 1.0
    weighted = sum(skill_progress[k].level * w for k, w in weights.items() if k in skill_progress)
    level = round(weighted / total_weight)
    return max(1, min(MAX_LEVEL, level))
