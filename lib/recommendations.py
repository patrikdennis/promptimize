"""
Recommendation engine.

Every recommendation carries:
  - `metric_key`   the Aggregate metric it targets (used for objective tracking
                    across runs via history.py)
  - `current_value` the user's own measured value for that metric right now
                    (becomes the baseline the next run is compared against)
  - `evidence`     a plain-language statement citing the user's *own* segmented
                    statistics (with sample sizes) where available, rather than
                    an unverifiable external citation
  - `mechanism`    a structural explanation of *why* this affects an LLM's
                    ability to process the request correctly the first time
  - `action`       a concrete, checkable behavior change

Two evidentiary standards are used, and each recommendation is explicit about
which one it relies on:
  1. Self-referential statistics: a conditional/segmented rate computed from
     the user's own transcript data (e.g. "your correction rate is 34% on
     turns without a file reference vs. 9% on turns with one, n=142/58").
  2. Structural/mechanistic reasoning about how LLM context processing works
     (token windows, ambiguity resolution, attention over recent turns) —
     labeled as reasoning, not a numeric external citation, since no
     external benchmark data is available to this tool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from metrics import Aggregate

SKILL_DIRS = [
    Path.home() / ".copilot" / "skills",
    Path.home() / ".claude" / "skills",
]

MIN_RELIABLE_N = 20

FEATURE_HINTS = [
    (r"\bplan\b|\bstep by step\b|\bbreak (this|it) down\b",
     "planning-step",
     "Use an explicit planning step before large multi-file changes",
     ("Several of your prompts request multi-step or exploratory work directly "
      "in the same turn as implementation. Separating 'propose an approach' from "
      "'execute the approach' gives you a checkpoint to catch a wrong direction "
      "before it is spent on code, rather than after."),
     "structure"),
    (r"\bsecurity\b|\bvulnerab|\bexploit\b|\bauth\b.*\b(token|password|secret)\b",
     "security-review-pass",
     "Request a dedicated security-review pass for security-sensitive work",
     ("Security-relevant requests appear folded into general implementation "
      "prompts in your history. A dedicated review pass applies a narrower, "
      "more adversarial reading of the same code than an implementation-mode "
      "pass, which is structurally a different task."),
     "structure"),
    (r"\breview my\b|\bcode review\b|\blook over\b",
     "review-pass",
     "Separate review requests from implementation requests",
     ("Mixing 'review this' and 'also fix/change this' in one turn forces a "
      "single response to do two different jobs (critique vs. produce), which "
      "increases the odds one is done superficially."),
     "structure"),
    (r"\bparallel\b|\bat the same time\b|\bmultiple (tasks|things)\b",
     "parallelize-independent-work",
     "Ask for independent tasks to run in parallel/background explicitly",
     ("Your prompts describe multiple independent tasks handled sequentially "
      "within one turn. If the tasks don't depend on each other, saying so "
      "explicitly lets the agent parallelize instead of defaulting to serial "
      "execution."),
     "efficiency"),
    (r"\bresearch\b|\bcompare (options|libraries|approaches)\b",
     "research-pass",
     "Use a dedicated research/investigation pass before committing to an approach",
     ("Open-ended comparison requests benefit from being scoped as research "
      "(gather and weigh options) separately from implementation (commit to "
      "one and build it) — collapsing both into one turn tends to anchor on "
      "the first plausible option."),
     "specificity"),
]


@dataclass
class Recommendation:
    id: str
    title: str
    axis: str
    metric_key: str | None
    current_value: float | None
    target_value: float | None
    impact: float
    priority: int
    evidence: str
    mechanism: str
    action: str
    confidence: str  # 'high' | 'moderate' | 'directional'


def discover_installed_skills() -> list[dict]:
    skills = []
    for base in SKILL_DIRS:
        if not base.exists():
            continue
        for skill_md in base.glob("*/SKILL.md"):
            try:
                text = skill_md.read_text(errors="ignore")
            except OSError:
                continue
            name_match = re.search(r"^name:\s*(.+)$", text, re.M)
            desc_match = re.search(r"^description:\s*(.+)$", text, re.M)
            name = (name_match.group(1).strip().strip("'\"") if name_match else skill_md.parent.name)
            desc = (desc_match.group(1).strip().strip("'\"") if desc_match else "")
            skills.append({"name": name, "description": desc, "path": str(skill_md.parent), "location": base.parent.name})
    return skills


def _confidence(n: int) -> str:
    if n >= 80:
        return "high"
    if n >= MIN_RELIABLE_N:
        return "moderate"
    return "directional (small sample)"


def _segment_sentence(stat: dict, metric_key: str, with_label: str, without_label: str) -> str | None:
    w, wo = stat.get("with", {}), stat.get("without", {})
    wv, wov = w.get(metric_key), wo.get(metric_key)
    if wv is None or wov is None or w.get("n", 0) < 5 or wo.get("n", 0) < 5:
        return None
    conf = _confidence(min(w["n"], wo["n"]))
    return (
        f"{with_label}: {wv:.0%} (n={w['n']}) vs. {without_label}: {wov:.0%} (n={wo['n']}) "
        f"[{conf} confidence]"
    )


def build_recommendations(agg: Aggregate, installed_skills: list[dict]) -> list[Recommendation]:
    recs: list[Recommendation] = []
    if agg.total_prompts == 0:
        return recs

    cs = agg.conditional_stats
    n_total = agg.total_prompts

    # ------------------------------------------------------------------
    # 1. Vague language
    # ------------------------------------------------------------------
    if agg.vague_rate > 0.08:
        ev = _segment_sentence(cs.get("correction_by_vague", {}), "correction_rate",
                                "correction rate on vague turns", "correction rate on non-vague turns")
        recs.append(Recommendation(
            id="reduce-vague-language",
            title="Eliminate vague/hedging language from requests",
            axis="specificity",
            metric_key="vague_rate",
            current_value=agg.vague_rate,
            target_value=0.08,
            impact=min(40, agg.vague_rate * 100 * 0.6),
            priority=1,
            evidence=(
                f"{agg.vague_rate:.0%} of your {n_total} analyzed turns (n={round(agg.vague_rate*n_total)}) "
                "contain at least one of 13 tracked hedge/vague-language patterns "
                "(\"maybe\", \"something\", \"fix it\", \"I guess\", \"not sure\", etc.)."
                + (f" Segmented comparison: {ev}." if ev else "")
            ),
            mechanism=(
                "An LLM resolves an underspecified token sequence by selecting the highest-"
                "probability interpretation given context, not by pausing to ask unless the "
                "phrasing or your explicit instructions cue it to. Hedge words widen the set "
                "of plausible interpretations without narrowing which one you mean, which "
                "increases variance in the output relative to your actual intent."
            ),
            action=(
                "Before sending a request, replace any of 'maybe', 'something', 'kind of', "
                "'I guess', 'not sure', or 'fix it' with the concrete noun/behavior you mean. "
                "If you are genuinely uncertain, say so explicitly and ask the agent to propose "
                "options rather than guessing silently."
            ),
            confidence=_confidence(round(agg.vague_rate * n_total)),
        ))

    # ------------------------------------------------------------------
    # 2. File / context anchoring
    # ------------------------------------------------------------------
    if agg.file_ref_rate < 0.45:
        ev = _segment_sentence(cs.get("correction_by_file_ref", {}), "correction_rate",
                                "correction rate with a file/code reference", "correction rate without one")
        ev2 = _segment_sentence(cs.get("clarification_by_file_ref", {}), "clarification_rate",
                                 "clarification-loop rate with a file/code reference", "without one")
        evidence_parts = [
            f"Only {agg.file_ref_rate:.0%} of your turns reference a specific file, path, "
            "@mention, or inline code/backtick token."
        ]
        if ev:
            evidence_parts.append(f"Segmented correction rate: {ev}.")
        if ev2:
            evidence_parts.append(f"Segmented clarification rate: {ev2}.")
        recs.append(Recommendation(
            id="reference-files",
            title="Anchor requests to exact files, symbols, or code",
            axis="context",
            metric_key="file_ref_rate",
            current_value=agg.file_ref_rate,
            target_value=0.55,
            impact=min(50, (0.6 - agg.file_ref_rate) * 100),
            priority=1,
            evidence=" ".join(evidence_parts),
            mechanism=(
                "Referencing an exact file/symbol converts a search-and-disambiguate problem "
                "(which of N candidates in the codebase/conversation matches this description) "
                "into a direct-lookup problem. This removes an inference step that is a common "
                "source of divergence between what you meant and what the agent locates."
            ),
            action=(
                "Use @file mentions, backtick-quoted symbol names, or paste the exact function/"
                "line before describing the change. Do this even for requests that feel obvious "
                "to you — the agent has no privileged access to which file you're picturing."
            ),
            confidence=_confidence(n_total),
        ))

    # ------------------------------------------------------------------
    # 3. Correction / rework rate
    # ------------------------------------------------------------------
    if agg.correction_rate > 0.10:
        recs.append(Recommendation(
            id="reduce-corrections",
            title="Reduce the correction/redo rate by front-loading constraints",
            axis="efficiency",
            metric_key="correction_rate",
            current_value=agg.correction_rate,
            target_value=0.08,
            impact=min(45, agg.correction_rate * 100 * 0.7),
            priority=1,
            evidence=(
                f"{agg.correction_rate:.0%} of your {n_total} turns match a correction/redo pattern "
                f"(n={round(agg.correction_rate*n_total)}) — phrases like 'that's wrong', 'still "
                "broken', 'try again', 'revert'. Each such turn represents at least one prior "
                "turn whose output did not match intent on the first attempt."
            ),
            mechanism=(
                "A correction turn is definitionally evidence that the previous turn under-"
                "specified the task relative to what was actually wanted. Structurally, this "
                "is the same information as an acceptance-criteria gap: the agent had no way "
                "to verify its own output against your actual bar before responding."
            ),
            action=(
                "State constraints and the definition of 'done' in the first prompt of a task, "
                "not after seeing the first attempt: what must not change, what tests/behavior "
                "must still pass, and what 'complete' looks like."
            ),
            confidence=_confidence(round(agg.correction_rate * n_total)),
        ))

    # ------------------------------------------------------------------
    # 4. Acceptance criteria
    # ------------------------------------------------------------------
    if agg.acceptance_rate < 0.30:
        recs.append(Recommendation(
            id="state-acceptance-criteria",
            title="State verification/acceptance criteria up front",
            axis="structure",
            metric_key="acceptance_rate",
            current_value=agg.acceptance_rate,
            target_value=0.35,
            impact=min(35, (0.4 - agg.acceptance_rate) * 100),
            priority=2,
            evidence=(
                f"Only {agg.acceptance_rate:.0%} of your turns state a verification condition "
                "or constraint (e.g. 'make sure', 'without breaking X', reference to tests/"
                "expected output)."
            ),
            mechanism=(
                "Without a stated acceptance condition, the agent's only self-check is internal "
                "plausibility, not conformance to your actual bar. Stating criteria gives it a "
                "concrete target to validate against before returning control to you."
            ),
            action=(
                "Add one sentence stating what must still be true after the change (e.g. "
                "'existing tests must still pass', 'don't change the public API', 'the output "
                "should match this example')."
            ),
            confidence=_confidence(n_total),
        ))

    # ------------------------------------------------------------------
    # 5. Prompt length — too short
    # ------------------------------------------------------------------
    if agg.avg_word_count < 6:
        recs.append(Recommendation(
            id="add-more-context",
            title="Expand short prompts with intent and constraints",
            axis="specificity",
            metric_key="avg_word_count",
            current_value=agg.avg_word_count,
            target_value=15.0,
            impact=15,
            priority=2,
            evidence=f"Average prompt length across {n_total} turns is {agg.avg_word_count:.1f} words.",
            mechanism=(
                "Very short prompts carry little information beyond a topic label; the agent "
                "fills the remaining specification from prior context and general priors, which "
                "increases the chance of a plausible-but-wrong interpretation, especially in a "
                "long or branching session."
            ),
            action="Add the 'why' and any hard constraints in 1-2 extra sentences, even for quick asks.",
            confidence=_confidence(n_total),
        ))
    elif agg.avg_word_count > 90:
        recs.append(Recommendation(
            id="structure-long-prompts",
            title="Convert long free-form prompts into a numbered checklist",
            axis="structure",
            metric_key=None,
            current_value=agg.avg_word_count,
            target_value=None,
            impact=15,
            priority=3,
            evidence=f"Average prompt length across {n_total} turns is {agg.avg_word_count:.0f} words.",
            mechanism=(
                "Long unstructured prose has no explicit boundary between distinct sub-requests, "
                "so a response can address the salient ones and silently drop others; a numbered "
                "list creates an explicit, checkable structure for both of you."
            ),
            action="Break multi-part requests into a numbered list of distinct, independently-checkable asks.",
            confidence=_confidence(n_total),
        ))

    # ------------------------------------------------------------------
    # 6. Bundled multi-asks
    # ------------------------------------------------------------------
    if agg.multi_ask_rate > 0.30:
        ev = _segment_sentence(cs.get("correction_by_multi_ask", {}), "correction_rate",
                                "correction rate on bundled (2+ verb) turns", "correction rate on single-focus turns")
        recs.append(Recommendation(
            id="split-multi-asks",
            title="Split bundled multi-part requests into separate turns",
            axis="structure",
            metric_key="multi_ask_rate",
            current_value=agg.multi_ask_rate,
            target_value=0.20,
            impact=12,
            priority=3,
            evidence=(
                f"{agg.multi_ask_rate:.0%} of your turns contain 2 or more distinct action verbs "
                "(e.g. 'fix X and also refactor Y and add Z')." + (f" {ev}." if ev else "")
            ),
            mechanism=(
                "Nothing structurally forces equal treatment of every clause in a bundled request; "
                "a numbered list or separate turns make each ask individually trackable and "
                "individually verifiable."
            ),
            action="Split bundled asks into a numbered list, or send them as separate turns when they aren't tightly coupled.",
            confidence=_confidence(round(agg.multi_ask_rate * n_total)),
        ))

    # ------------------------------------------------------------------
    # 7. Clarification loops
    # ------------------------------------------------------------------
    if agg.clarification_rate > 0.08:
        ev = _segment_sentence(cs.get("clarification_by_short", {}), "clarification_rate",
                                "clarification rate on turns of 5 words or fewer", "on longer turns")
        recs.append(Recommendation(
            id="reduce-clarification-loops",
            title="Reduce clarification round-trips by pre-answering likely questions",
            axis="efficiency",
            metric_key="clarification_rate",
            current_value=agg.clarification_rate,
            target_value=0.10,
            impact=min(30, agg.clarification_rate * 100 * 0.8),
            priority=2,
            evidence=(
                f"{agg.clarification_rate:.0%} of your {n_total} turns received a short, "
                "question-shaped reply rather than direct action (detected via reply length "
                "and clarifying-question phrasing)." + (f" {ev}." if ev else "")
            ),
            mechanism=(
                "A clarification reply is a direct, measured signal that the agent judged the "
                "request too ambiguous to act on safely — it is the most literal cost of "
                "under-specification available from transcript data, distinct from the softer "
                "vague-language heuristic."
            ),
            action=(
                "When a request has an obvious branch point (which file, which of two approaches, "
                "destructive vs. non-destructive), answer it preemptively in the same turn instead "
                "of waiting to be asked."
            ),
            confidence=_confidence(round(agg.clarification_rate * n_total)),
        ))

    # ------------------------------------------------------------------
    # 8. Context restatement
    # ------------------------------------------------------------------
    if agg.context_restatement_rate > 0.10:
        recs.append(Recommendation(
            id="reduce-context-restatement",
            title="Avoid re-stating context already established in the same session",
            axis="efficiency",
            metric_key="context_restatement_rate",
            current_value=agg.context_restatement_rate,
            target_value=0.08,
            impact=min(20, agg.context_restatement_rate * 100 * 0.5),
            priority=3,
            evidence=(
                f"{agg.context_restatement_rate:.0%} of eligible turns (those with a prior turn "
                "in the same session) re-use 38% or more of the same significant words as the "
                "immediately preceding turn — a proxy for re-explaining context the agent should "
                "already have."
            ),
            mechanism=(
                "Restating context already inside the model's context window is redundant "
                "input; it doesn't add new information and, if frequent, suggests either "
                "reduced trust in what the agent retained or a session long enough that early "
                "context risks being pushed out or summarized away."
            ),
            action=(
                "If you find yourself re-explaining the same background repeatedly, either "
                "confirm the agent still has it ('do you still have X in context?') instead of "
                "re-typing it, or start a fresh, tightly-scoped session for the new sub-task."
            ),
            confidence=_confidence(round(agg.context_restatement_rate * n_total)),
        ))

    # ------------------------------------------------------------------
    # 9. Single-shot resolution
    # ------------------------------------------------------------------
    if agg.single_shot_rate < 0.35 and agg.total_sessions >= 5:
        recs.append(Recommendation(
            id="improve-single-shot-rate",
            title="Increase the share of tasks resolved in a single exchange",
            axis="efficiency",
            metric_key="single_shot_rate",
            current_value=agg.single_shot_rate,
            target_value=0.5,
            impact=15,
            priority=3,
            evidence=(
                f"{agg.single_shot_rate:.0%} of your {agg.total_sessions} sessions resolved in "
                f"exactly one turn; average session length is {agg.avg_turns_per_session:.1f} turns."
            ),
            mechanism=(
                "This is a composite outcome of the other axes: turns with clear file "
                "references, stated acceptance criteria, and low vague-language content are "
                "structurally more likely to be actionable without a follow-up round."
            ),
            action=(
                "Treat this as a lagging indicator: improving vague-language rate, file-"
                "reference rate, and acceptance-criteria rate should raise this number as a "
                "side effect. Re-check it on your next report."
            ),
            confidence=_confidence(agg.total_sessions),
        ))

    # ------------------------------------------------------------------
    # Feature/workflow hints matched against actual prompt content
    # ------------------------------------------------------------------
    all_text = " \n".join(f.turn.user_text.lower() for f in agg.features)
    for pattern, hint_id, title, detail, axis in FEATURE_HINTS:
        if re.search(pattern, all_text, re.I):
            matches = len(re.findall(pattern, all_text, re.I))
            recs.append(Recommendation(
                id=f"feature-{hint_id}",
                title=title,
                axis=axis,
                metric_key=None,
                current_value=None,
                target_value=None,
                impact=8,
                priority=4,
                evidence=f"Matched this workflow pattern in {matches} of your {n_total} analyzed turns.",
                mechanism=detail,
                action=f"Adopt this as a distinct step in your workflow rather than folding it into a general request.",
                confidence="directional (pattern match)",
            ))

    # ------------------------------------------------------------------
    # Installed-skill reuse / gap
    # ------------------------------------------------------------------
    if installed_skills:
        names = ", ".join(f"{s['name']} ({s['location']})" for s in installed_skills)
        recs.append(Recommendation(
            id="leverage-installed-skills",
            title="Explicitly invoke installed skills instead of re-explaining workflows",
            axis="context",
            metric_key=None,
            current_value=None,
            target_value=None,
            impact=10,
            priority=4,
            evidence=f"You have {len(installed_skills)} personal skill(s) installed: {names}.",
            mechanism=(
                "A skill packages a workflow's fixed instructions once; referencing it by name "
                "in a prompt avoids re-deriving the same procedure from a fresh natural-language "
                "description each time, and reduces the chance a step is omitted."
            ),
            action="Name the skill explicitly in prompts that match its stated purpose.",
            confidence="high",
        ))
    else:
        recs.append(Recommendation(
            id="install-skills",
            title="Install at least one starter skill for a repeated workflow",
            axis="context",
            metric_key=None,
            current_value=None,
            target_value=None,
            impact=10,
            priority=4,
            evidence="No personal skills found in ~/.copilot/skills or ~/.claude/skills.",
            mechanism=(
                "Any workflow you repeat with only minor variation (a report format, a review "
                "checklist, a release process) is a candidate for a skill: encoding it once "
                "removes the need to re-specify structure and constraints from scratch every "
                "time, directly raising both context and structure scores."
            ),
            action="Identify one workflow you repeat at least weekly and write it up as a skill.",
            confidence="high",
        ))

    recs.sort(key=lambda r: (r.priority, -r.impact))
    return recs
