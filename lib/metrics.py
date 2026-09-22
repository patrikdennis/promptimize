"""
Metrics engine.

Every metric here is defined by an explicit, inspectable formula over the
user's own turn history — nothing is looked up from an external benchmark.
`METRIC_DEFINITIONS` documents, for each metric, exactly what is counted and
why it is believed to matter for effective LLM collaboration; the report
renders these definitions verbatim so nothing is a black box.

Two kinds of features are extracted:
  1. Per-turn features (TurnFeatures) — properties of a single user message,
     optionally informed by the assistant's reply to that turn and the
     previous turn in the same session (for context-restatement detection).
  2. Aggregate statistics (Aggregate) — rates, distributions, and
     conditional/segmented statistics computed over the whole set of turns
     in the analyzed period. Conditional statistics (e.g. "correction rate
     when a file is referenced vs. when it isn't") are what let the
     recommendation engine cite real numbers instead of generic advice.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from backends import Session, Turn
from config import load_config

# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

VAGUE_PATTERNS = [
    r"\bsomething\b", r"\bsomehow\b", r"\bmaybe\b", r"\bkind of\b", r"\bi guess\b",
    r"\bfix it\b", r"\bmake it work\b", r"\betc\.?\b", r"\bwhatever\b", r"\bstuff\b",
    r"\bi don'?t know\b", r"\bnot sure\b", r"\bsome sort of\b", r"\bi think\b",
]
CORRECTION_PATTERNS = [
    r"\bthat'?s wrong\b", r"\bnot what i\b", r"\bdoesn'?t work\b", r"\bstill (broken|not working)\b",
    r"\btry again\b", r"\bactually i meant\b", r"\bno,? (that|this|you)\b", r"\brevert\b",
    r"\bundo\b", r"\bwrong\b", r"\bincorrect\b",
]
ACTION_VERBS = [
    "implement", "fix", "add", "create", "refactor", "write", "build", "debug",
    "review", "optimize", "remove", "update", "investigate", "analyze", "analyse",
    "generate", "design", "migrate", "test", "document", "explain", "rename",
    "delete", "move", "clean", "improve", "configure", "deploy", "set up", "setup",
]
FILE_REF_RE = re.compile(
    r"(@[\w./-]+|`[^`\s]+`|\b[\w./-]+\.(py|js|ts|tsx|jsx|go|rb|java|rs|md|json|yaml|yml|sql|html|css)\b)",
    re.I,
)
ACCEPTANCE_PATTERNS = [
    r"\bmake sure\b", r"\bshould (still|not|also)\b", r"\bwithout breaking\b",
    r"\bverify\b", r"\btest(s|ed|ing)?\b", r"\bexpected (result|output|behavior)\b",
    r"\bacceptance\b", r"\bdon'?t change\b", r"\bkeep\b.*\bsame\b",
]
CLARIFICATION_PATTERNS = [
    r"\bcould you clarify\b", r"\bcan you (clarify|confirm|specify)\b",
    r"\bwhich (of|one|approach|option)\b", r"\bdo you want\b", r"\bwould you like\b",
    r"\bshould i\b", r"\bwhat (do you mean|would you like|exactly)\b",
    r"\bto clarify\b", r"\bbefore i proceed\b", r"\bjust to confirm\b",
    r"\ba few (questions|things) (first|before)\b",
]
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "as", "at",
    "by", "from", "i", "you", "we", "they", "he", "she", "not", "so", "if",
    "do", "does", "did", "can", "could", "should", "would", "will", "also",
    "than", "then", "there", "here", "just", "please", "want", "need", "like",
    "have", "has", "had", "into", "your", "my", "our", "their", "its", "about",
}

_vague_re = re.compile("|".join(VAGUE_PATTERNS), re.I)
_correction_re = re.compile("|".join(CORRECTION_PATTERNS), re.I)
_acceptance_re = re.compile("|".join(ACCEPTANCE_PATTERNS), re.I)
_verb_re = re.compile(r"\b(" + "|".join(ACTION_VERBS) + r")\b", re.I)
_clarification_re = re.compile("|".join(CLARIFICATION_PATTERNS), re.I)

CONTEXT_RESTATEMENT_JACCARD_THRESHOLD = 0.38
CLARIFICATION_MAX_WORDS = 60

# ---------------------------------------------------------------------------
# Metric documentation — rendered verbatim in the report's "Methodology" tab
# and used as captions under each chart/table. Keep formulas exact.
# ---------------------------------------------------------------------------

METRIC_DEFINITIONS: dict[str, dict] = {
    "vague_rate": {
        "label": "Vague-language rate",
        "formula": "count(turns matching a vague-language pattern) / total turns",
        "detail": (
            "A turn is flagged if it matches one of 13 patterns associated with "
            "underspecified requests: hedges (\"maybe\", \"I guess\", \"kind of\"), "
            "catch-alls (\"fix it\", \"make it work\", \"etc\"), or explicit "
            "uncertainty (\"not sure\", \"I don't know\"). Lower is better."
        ),
        "why_it_matters": (
            "An LLM resolves ambiguity by sampling the most probable interpretation "
            "of underspecified language from its training distribution, not by asking "
            "you what you meant unless explicitly prompted to. Vague phrasing "
            "increases the chance the resolved interpretation diverges from your "
            "actual intent, which surfaces later as a correction turn."
        ),
    },
    "file_ref_rate": {
        "label": "Context-anchoring rate",
        "formula": "count(turns with a file path, @mention, or inline code/backtick reference) / total turns",
        "detail": (
            "A turn is flagged if it contains an @mention, a backtick-quoted "
            "token, or a bare path/filename with a recognized extension. Higher "
            "is better."
        ),
        "why_it_matters": (
            "Naming an exact file, symbol, or snippet removes an entire search/"
            "disambiguation step from the agent's task: it can jump straight to "
            "the referenced location instead of inferring which of N candidate "
            "files or functions you mean from surrounding conversation. This is "
            "a reduction in required inference steps, not a matter of politeness."
        ),
    },
    "correction_rate": {
        "label": "Correction rate",
        "formula": "count(turns matching a correction/redo pattern) / total turns",
        "detail": (
            "A turn is flagged if it matches patterns such as \"that's wrong\", "
            "\"still broken\", \"try again\", \"revert\", \"undo\", or \"not what "
            "I asked\". This is the closest available proxy for rework caused by "
            "a prior turn being misunderstood or under-specified. Lower is better."
        ),
        "why_it_matters": (
            "Each correction turn is, by definition, a discarded or partially "
            "discarded round of work: it indicates the previous turn did not "
            "carry enough information for the agent to produce an acceptable "
            "result on the first attempt."
        ),
    },
    "acceptance_rate": {
        "label": "Acceptance-criteria rate",
        "formula": "count(turns stating a verification condition or constraint) / total turns",
        "detail": (
            "A turn is flagged if it contains phrasing like \"make sure\", "
            "\"without breaking X\", \"should still\", \"verify\", or references "
            "tests/expected output. Higher is better."
        ),
        "why_it_matters": (
            "Stating what must remain true after a change gives the agent a "
            "concrete self-check to run before responding, shifting error "
            "detection from you (post-hoc) to the agent (pre-response)."
        ),
    },
    "multi_ask_rate": {
        "label": "Bundled multi-ask rate",
        "formula": "count(turns with 2+ distinct action verbs) / total turns",
        "detail": (
            "A turn is flagged if it contains 2 or more of a curated list of 27 "
            "action verbs (implement, fix, add, refactor, etc.), indicating "
            "multiple distinct asks bundled into one message. Lower is better "
            "past a point — some bundling is normal and efficient."
        ),
        "why_it_matters": (
            "Bundling unrelated asks in one turn increases the chance one of them "
            "is dropped, since nothing enforces that a single response addresses "
            "every clause with equal weight — unlike a numbered list, which is "
            "structurally harder to partially skip."
        ),
    },
    "clarification_rate": {
        "label": "Clarification-loop rate",
        "formula": "count(turns whose assistant reply is a short (<60 word) reply matching a clarifying-question pattern) / total turns",
        "detail": (
            "Flags assistant replies that ask the user to choose between options, "
            "confirm intent, or specify missing information, rather than "
            "proceeding with the request. Lower is better — each clarification "
            "loop costs a full round trip."
        ),
        "why_it_matters": (
            "This is a direct, measured cost: a clarification loop means the "
            "agent judged the turn too underspecified to act on safely. It is "
            "the most literal quantification of 'the agent had to ask' available "
            "from transcript data alone."
        ),
    },
    "context_restatement_rate": {
        "label": "Context-restatement rate",
        "formula": (
            f"count(turns where Jaccard(words[t], words[t-1]) >= {CONTEXT_RESTATEMENT_JACCARD_THRESHOLD} "
            "within the same session) / eligible turns (turn_index > 0)"
        ),
        "detail": (
            "Flags turns that re-state a large fraction of the wording of the "
            "immediately preceding turn in the same session (stopwords excluded). "
            "This is a proxy for the user re-explaining context the agent should "
            "already have from the same conversation. Lower is better."
        ),
        "why_it_matters": (
            "Restating context already present in-session is pure overhead: the "
            "agent already has that text in its context window. High restatement "
            "usually signals the user doesn't trust the agent retained earlier "
            "context, or the session has grown long enough that earlier context "
            "was compacted/dropped — both worth noticing."
        ),
    },
    "single_shot_rate": {
        "label": "Single-shot resolution rate",
        "formula": "count(sessions with exactly 1 turn) / count(sessions with >= 1 turn)",
        "detail": (
            "A session counts as single-shot if it consists of exactly one "
            "user/assistant exchange with no follow-up turn recorded. Higher "
            "is generally better as a proxy for first-attempt sufficiency, but "
            "note this also rises trivially for sessions abandoned after one "
            "reply — read alongside correction_rate."
        ),
        "why_it_matters": (
            "A high single-shot rate concentrated with a low correction rate is "
            "the strongest available signal that requests are well-specified "
            "enough to resolve without iteration."
        ),
    },
    "avg_turns_per_session": {
        "label": "Average turns per session",
        "formula": "total turns / total sessions",
        "detail": "Simple mean session length in turns. Reported descriptively; not scored, since long sessions on genuinely complex tasks are expected and desirable.",
        "why_it_matters": (
            "Session length alone is not good or bad — it is contextual signal "
            "used alongside correction and clarification rates to distinguish "
            "'long because complex' from 'long because of rework'."
        ),
    },
}

_verb_re_word_split = re.compile(r"[A-Za-z']+")


def _significant_words(text: str) -> set[str]:
    words = {w.lower() for w in _verb_re_word_split.findall(text)}
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}


def _is_clarification(assistant_text: str | None) -> bool:
    if not assistant_text:
        return False
    if "```" in assistant_text:
        return False
    word_count = len(assistant_text.split())
    if word_count > CLARIFICATION_MAX_WORDS:
        return False
    if _clarification_re.search(assistant_text):
        return True
    return "?" in assistant_text and word_count <= 30


@dataclass
class TurnFeatures:
    turn: Turn
    session: Session
    word_count: int
    has_file_ref: bool
    has_code_ref: bool
    vague_hits: int
    correction_signal: bool
    acceptance_signal: bool
    action_verb_hits: int
    is_question: bool
    starts_with_verb: bool
    context_restatement: bool
    restatement_eligible: bool
    clarification_signal: bool


def extract_turn_features(turn: Turn, session: Session, prev_turn: Turn | None) -> TurnFeatures:
    text = turn.user_text
    words = re.findall(r"\S+", text)
    word_count = len(words)
    file_refs = FILE_REF_RE.findall(text)
    first_word = words[0].lower().strip(".,:;!?") if words else ""

    restatement_eligible = prev_turn is not None
    context_restatement = False
    if restatement_eligible:
        cur_words = _significant_words(text)
        prev_words = _significant_words(prev_turn.user_text)
        if cur_words and prev_words:
            jaccard = len(cur_words & prev_words) / len(cur_words | prev_words)
            context_restatement = jaccard >= CONTEXT_RESTATEMENT_JACCARD_THRESHOLD

    return TurnFeatures(
        turn=turn,
        session=session,
        word_count=word_count,
        has_file_ref=bool(file_refs),
        has_code_ref="`" in text or "```" in text,
        vague_hits=len(_vague_re.findall(text)),
        correction_signal=bool(_correction_re.search(text)),
        acceptance_signal=bool(_acceptance_re.search(text)),
        action_verb_hits=len(_verb_re.findall(text)),
        is_question=text.strip().endswith("?"),
        starts_with_verb=first_word in ACTION_VERBS,
        context_restatement=context_restatement,
        restatement_eligible=restatement_eligible,
        clarification_signal=_is_clarification(turn.assistant_text),
    )


def extract_all_features(sessions: list[Session]) -> list[TurnFeatures]:
    feats: list[TurnFeatures] = []
    for s in sessions:
        ordered = sorted(s.turns, key=lambda t: (t.timestamp, t.turn_index))
        prev = None
        for t in ordered:
            feats.append(extract_turn_features(t, s, prev))
            prev = t
    feats.sort(key=lambda f: f.turn.timestamp)
    return feats


def _rate(feats, pred) -> tuple[float, int]:
    n = len(feats)
    if n == 0:
        return 0.0, 0
    return sum(1 for f in feats if pred(f)) / n, n


def _segment(feats: list[TurnFeatures], pred) -> dict:
    """Split feats into two groups by pred, and report the correction and
    clarification rate within each — the conditional statistics used as
    evidence in recommendations."""
    with_group = [f for f in feats if pred(f)]
    without_group = [f for f in feats if not pred(f)]

    def stats(group):
        n = len(group)
        if n == 0:
            return {"n": 0, "correction_rate": None, "clarification_rate": None}
        return {
            "n": n,
            "correction_rate": sum(1 for f in group if f.correction_signal) / n,
            "clarification_rate": sum(1 for f in group if f.clarification_signal) / n,
        }

    return {"with": stats(with_group), "without": stats(without_group)}


@dataclass
class Aggregate:
    features: list[TurnFeatures]
    total_prompts: int = 0
    total_sessions: int = 0
    by_agent: Counter = field(default_factory=Counter)
    by_day: Counter = field(default_factory=Counter)
    by_hour: Counter = field(default_factory=Counter)
    by_weekday: Counter = field(default_factory=Counter)
    by_repo_cwd: Counter = field(default_factory=Counter)
    avg_word_count: float = 0.0
    vague_rate: float = 0.0
    file_ref_rate: float = 0.0
    correction_rate: float = 0.0
    acceptance_rate: float = 0.0
    multi_ask_rate: float = 0.0
    question_rate: float = 0.0
    clarification_rate: float = 0.0
    context_restatement_rate: float = 0.0
    single_shot_rate: float = 0.0
    avg_turns_per_session: float = 0.0
    word_count_buckets: Counter = field(default_factory=Counter)
    session_turn_buckets: Counter = field(default_factory=Counter)
    conditional_stats: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    metric_values: dict = field(default_factory=dict)  # flat, for history tracking


def _word_bucket(n: int) -> str:
    return (
        "1-5" if n <= 5 else
        "6-15" if n <= 15 else
        "16-40" if n <= 40 else
        "41-80" if n <= 80 else
        "80+"
    )


def _session_turn_bucket(n: int) -> str:
    return "1" if n == 1 else "2-3" if n <= 3 else "4-7" if n <= 7 else "8-15" if n <= 15 else "16+"


def build_aggregate(feats: list[TurnFeatures], sessions: list[Session]) -> Aggregate:
    agg = Aggregate(features=feats)
    n = len(feats)
    agg.total_prompts = n
    agg.total_sessions = len(sessions)

    if n == 0:
        agg.scores = {"specificity": 0, "context": 0, "structure": 0, "efficiency": 0, "overall": 0}
        return agg

    for f in feats:
        t = f.turn
        agg.by_agent[f.session.agent] += 1
        day = t.timestamp.strftime("%Y-%m-%d")
        agg.by_day[day] += 1
        agg.by_hour[t.timestamp.hour] += 1
        agg.by_weekday[t.timestamp.weekday()] += 1
        label = f.session.repository or f.session.cwd or "unknown"
        agg.by_repo_cwd[label] += 1
        agg.word_count_buckets[_word_bucket(f.word_count)] += 1

    for s in sessions:
        agg.session_turn_buckets[_session_turn_bucket(len(s.turns))] += 1

    agg.avg_word_count = sum(f.word_count for f in feats) / n
    agg.vague_rate, _ = _rate(feats, lambda f: f.vague_hits > 0)
    agg.file_ref_rate, _ = _rate(feats, lambda f: f.has_file_ref or f.has_code_ref)
    agg.correction_rate, _ = _rate(feats, lambda f: f.correction_signal)
    agg.acceptance_rate, _ = _rate(feats, lambda f: f.acceptance_signal)
    agg.multi_ask_rate, _ = _rate(feats, lambda f: f.action_verb_hits >= 2)
    agg.question_rate, _ = _rate(feats, lambda f: f.is_question)
    agg.clarification_rate, _ = _rate(feats, lambda f: f.clarification_signal)

    eligible = [f for f in feats if f.restatement_eligible]
    agg.context_restatement_rate, _ = _rate(eligible, lambda f: f.context_restatement) if eligible else (0.0, 0)

    single_shot_sessions = sum(1 for s in sessions if len(s.turns) == 1)
    agg.single_shot_rate = single_shot_sessions / len(sessions) if sessions else 0.0
    agg.avg_turns_per_session = n / len(sessions) if sessions else 0.0

    agg.conditional_stats = {
        "correction_by_file_ref": _segment(feats, lambda f: f.has_file_ref or f.has_code_ref),
        "correction_by_vague": _segment(feats, lambda f: f.vague_hits > 0),
        "correction_by_acceptance": _segment(feats, lambda f: f.acceptance_signal),
        "correction_by_multi_ask": _segment(feats, lambda f: f.action_verb_hits >= 2),
        "clarification_by_file_ref": _segment(feats, lambda f: f.has_file_ref or f.has_code_ref),
        "clarification_by_short": _segment(feats, lambda f: f.word_count <= 5),
    }

    # --- Scoring (0-100 per axis) -------------------------------------------
    cfg = load_config()
    sc = cfg["scoring"]
    spec_cfg, struct_cfg, eff_cfg = sc["specificity"], sc["structure"], sc["efficiency"]

    ideal_len_rate = sum(
        1 for f in feats if spec_cfg["ideal_length_min"] <= f.word_count <= spec_cfg["ideal_length_max"]
    ) / n
    specificity = 100 * (spec_cfg["vague_weight"] * (1 - agg.vague_rate) + spec_cfg["length_weight"] * ideal_len_rate)

    context = 100 * agg.file_ref_rate

    structure = 100 * (
        struct_cfg["verb_weight"] * (sum(1 for f in feats if f.starts_with_verb or f.action_verb_hits >= 1) / n)
        + struct_cfg["acceptance_weight"] * agg.acceptance_rate
        + struct_cfg["multi_ask_weight"] * (1 - agg.multi_ask_rate)
    )

    efficiency = 100 * (
        eff_cfg["correction_weight"] * (1 - agg.correction_rate)
        + eff_cfg["clarification_weight"] * (1 - agg.clarification_rate)
        + eff_cfg["restatement_weight"] * (1 - agg.context_restatement_rate)
    )

    agg.scores = {
        "specificity": round(specificity, 1),
        "context": round(context, 1),
        "structure": round(structure, 1),
        "efficiency": round(efficiency, 1),
    }
    agg.scores["overall"] = round(sum(agg.scores.values()) / 4, 1)

    agg.metric_values = {
        "vague_rate": agg.vague_rate,
        "file_ref_rate": agg.file_ref_rate,
        "correction_rate": agg.correction_rate,
        "acceptance_rate": agg.acceptance_rate,
        "multi_ask_rate": agg.multi_ask_rate,
        "clarification_rate": agg.clarification_rate,
        "context_restatement_rate": agg.context_restatement_rate,
        "single_shot_rate": agg.single_shot_rate,
        "avg_word_count": agg.avg_word_count,
    }
    return agg
