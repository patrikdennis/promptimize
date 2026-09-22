"""
Dependency-free configuration loader.

This tool intentionally has zero third-party dependencies (no PyYAML, no
numpy), so `config.yaml` is parsed by a small, deliberately restricted
subset-of-YAML parser (`_parse_yaml_subset`) rather than a real YAML library.
It supports exactly what `config.yaml` uses: nested block mappings via
2-space indentation, scalar values (int/float/bool/str), and `#` comments.
It does NOT support lists, flow-style `{...}`/`[...]`, multi-line strings,
anchors, or any other YAML feature — if you need those, this parser is the
wrong tool and should be swapped for PyYAML.

If `config.yaml` is missing, unreadable, or fails to parse, every accessor
below falls back to `DEFAULTS`, which mirrors the shipped `config.yaml`
exactly. The tool never fails to run because of a broken config file.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

DEFAULTS = {
    "scoring": {
        "specificity": {"vague_weight": 0.55, "length_weight": 0.45, "ideal_length_min": 6, "ideal_length_max": 60},
        "structure": {"verb_weight": 0.40, "acceptance_weight": 0.35, "multi_ask_weight": 0.25},
        "efficiency": {"correction_weight": 0.50, "clarification_weight": 0.25, "restatement_weight": 0.25},
    },
    "objective_thresholds": {
        "vague_rate": {"direction": "lower", "healthy": 0.08},
        "file_ref_rate": {"direction": "higher", "healthy": 0.55},
        "correction_rate": {"direction": "lower", "healthy": 0.08},
        "acceptance_rate": {"direction": "higher", "healthy": 0.35},
        "multi_ask_rate": {"direction": "lower", "healthy": 0.20},
        "clarification_rate": {"direction": "lower", "healthy": 0.10},
        "context_restatement_rate": {"direction": "lower", "healthy": 0.08},
        "single_shot_rate": {"direction": "higher", "healthy": 0.55},
        "avg_word_count": {"direction": "higher", "healthy": 6.0},
    },
    "objective_improvement_band_pct": 5.0,
    "advanced_metrics": {
        "markov_min_transitions": 20,
        "winding_min_turns": 4,
        "winding_circular_threshold": 0.75,
        "burstiness_min_timestamps": 10,
        "compression_min_bytes": 200,
        "heaps_min_tokens": 200,
        "heaps_min_samples": 5,
        "heaps_sample_base": 50,
        "heaps_sample_ratio": 1.35,
        "heaps_sample_points": 24,
    },
    "anomaly_detection": {
        "min_history_runs": 4,
        "z_score_watch_threshold": 1.0,
        "z_score_flag_threshold": 1.75,
    },
    "mahalanobis": {
        "min_history_runs": 5,
        "ridge_epsilon": 0.25,
    },
    "team_mode": {
        "anonymize_labels": True,
        "min_users_for_aggregate": 2,
    },
}


def _coerce_scalar(raw: str):
    raw = raw.strip()
    if raw == "" or raw is None:
        return None
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _strip_comment(line: str) -> str:
    # Only strip '#' that isn't inside a quoted string; good enough for this
    # config's plain scalar values (no '#' appears inside any of them).
    idx = line.find("#")
    return line if idx == -1 else line[:idx]


def _parse_yaml_subset(text: str) -> dict:
    lines = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))

    root: dict = {}
    # Stack of (indent, dict) — the dict active at that indentation level.
    stack = [(-1, root)]

    for indent, content in lines:
        if ":" not in content:
            continue
        key, _, value = content.partition(":")
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            new_dict: dict = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _coerce_scalar(value)

    return root


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


_cached_config: dict | None = None


def load_config(path: Path | None = None) -> dict:
    """Loads config.yaml, deep-merged over DEFAULTS so a partial/edited file
    only needs to specify the keys it wants to override. Cached after first
    load; pass an explicit `path` (e.g. in tests) to bypass the cache."""
    global _cached_config
    target = path or CONFIG_PATH
    if path is None and _cached_config is not None:
        return _cached_config

    parsed = {}
    try:
        text = target.read_text()
        parsed = _parse_yaml_subset(text)
    except (OSError, ValueError):
        parsed = {}

    merged = _deep_merge(DEFAULTS, parsed)
    if path is None:
        _cached_config = merged
    return merged
