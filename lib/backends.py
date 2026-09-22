"""
Data backends: read local session history from GitHub Copilot CLI and Claude Code,
normalizing both into a single conversational model:

    Session
      -> turns: list[Turn]   (ordered)
           Turn: user_text, assistant_text, timestamp, ...

This turn-level model (rather than a flat list of prompts) is what lets the
metrics layer detect within-session phenomena: clarification loops, context
restatement between consecutive turns, single-shot resolution, etc.

A flattened `Prompt` view is still provided for callers that only care about
individual user messages.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


@dataclass
class Turn:
    agent: str
    session_id: str
    turn_index: int
    timestamp: datetime
    user_text: str
    assistant_text: str | None


@dataclass
class Session:
    agent: str
    session_id: str
    cwd: str | None
    repository: str | None
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Prompt:
    """Flat view of a single user message, for callers that don't need
    session/turn context."""
    agent: str
    session_id: str
    cwd: str | None
    repository: str | None
    timestamp: datetime
    text: str


# ---------------------------------------------------------------------------
# GitHub Copilot CLI backend: ~/.copilot/session-store.db (SQLite)
# ---------------------------------------------------------------------------

def read_copilot_sessions(db_path: Path) -> list[Session]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT t.session_id, t.turn_index, t.user_message, t.assistant_response,
                   t.timestamp, s.cwd, s.repository
            FROM turns t
            LEFT JOIN sessions s ON s.id = t.session_id
            WHERE t.user_message IS NOT NULL AND trim(t.user_message) != ''
            ORDER BY t.session_id, t.turn_index
            """
        ).fetchall()
    finally:
        con.close()

    sessions: dict[str, Session] = {}
    for r in rows:
        ts = _parse_ts(r["timestamp"])
        if ts is None:
            continue
        sid = r["session_id"]
        sess = sessions.get(sid)
        if sess is None:
            sess = Session(agent="copilot", session_id=sid, cwd=r["cwd"], repository=r["repository"])
            sessions[sid] = sess
        sess.turns.append(Turn(
            agent="copilot",
            session_id=sid,
            turn_index=r["turn_index"],
            timestamp=ts,
            user_text=r["user_message"],
            assistant_text=r["assistant_response"],
        ))
    return list(sessions.values())


# ---------------------------------------------------------------------------
# Claude Code backend: ~/.claude/projects/**/*.jsonl
# ---------------------------------------------------------------------------

def _extract_user_text(content) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("tool_result", "tool_use"):
                return None
            if btype == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
        joined = "\n".join(t for t in texts if t).strip()
        return joined or None
    return None


def _extract_assistant_text(content) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
        joined = "\n".join(t for t in texts if t).strip()
        return joined or None
    return None


def read_claude_sessions(projects_dir: Path) -> list[Session]:
    if not projects_dir.exists():
        return []
    sessions: list[Session] = []
    for jsonl_path in projects_dir.glob("*/*.jsonl"):
        session_id = jsonl_path.stem
        cwd = None
        turns: list[Turn] = []
        pending_user: tuple[str, datetime] | None = None
        pending_assistant_chunks: list[str] = []
        turn_index = 0

        def flush():
            nonlocal pending_user, pending_assistant_chunks, turn_index
            if pending_user is None:
                return
            text, ts = pending_user
            assistant_text = "\n".join(pending_assistant_chunks).strip() or None
            turns.append(Turn(
                agent="claude",
                session_id=session_id,
                turn_index=turn_index,
                timestamp=ts,
                user_text=text,
                assistant_text=assistant_text,
            ))
            turn_index += 1
            pending_user = None
            pending_assistant_chunks = []

        try:
            with jsonl_path.open("r", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if cwd is None and entry.get("cwd"):
                        cwd = entry["cwd"]

                    etype = entry.get("type")
                    if etype == "user" and not entry.get("isMeta") and "toolUseResult" not in entry:
                        message = entry.get("message") or {}
                        if message.get("role") != "user":
                            continue
                        text = _extract_user_text(message.get("content"))
                        if not text:
                            continue
                        ts = _parse_ts(entry.get("timestamp", ""))
                        if ts is None:
                            continue
                        flush()
                        pending_user = (text, ts)
                    elif etype == "assistant":
                        message = entry.get("message") or {}
                        text = _extract_assistant_text(message.get("content"))
                        if text and pending_user is not None:
                            pending_assistant_chunks.append(text)
        except OSError:
            continue

        flush()
        if turns:
            sessions.append(Session(agent="claude", session_id=session_id, cwd=cwd, repository=None, turns=turns))
    return sessions


def load_all_sessions(
    agents: Iterable[str],
    copilot_db: Path,
    claude_projects_dir: Path,
) -> list[Session]:
    sessions: list[Session] = []
    if "copilot" in agents:
        sessions.extend(read_copilot_sessions(copilot_db))
    if "claude" in agents:
        sessions.extend(read_claude_sessions(claude_projects_dir))
    return sessions


def flatten_prompts(sessions: list[Session]) -> list[Prompt]:
    prompts: list[Prompt] = []
    for s in sessions:
        for t in s.turns:
            prompts.append(Prompt(
                agent=s.agent,
                session_id=s.session_id,
                cwd=s.cwd,
                repository=s.repository,
                timestamp=t.timestamp,
                text=t.user_text,
            ))
    prompts.sort(key=lambda p: p.timestamp)
    return prompts


def filter_sessions_by_range(sessions: list[Session], start: datetime, end: datetime) -> list[Session]:
    """Return sessions with only the turns falling inside [start, end], dropping
    sessions left with no turns. Turn ordering/adjacency inside a session is
    preserved for within-session metrics (context restatement, clarification)."""
    out: list[Session] = []
    for s in sessions:
        turns = [t for t in s.turns if start <= t.timestamp <= end]
        if turns:
            out.append(Session(agent=s.agent, session_id=s.session_id, cwd=s.cwd, repository=s.repository, turns=turns))
    return out
