"""
Trend-adjusted anomaly detection: flags metrics whose value this run is
statistically unusual relative to your own rolling history, rather than
just "worse than last time." A single bad run is normal variance; a run
that is 2+ standard deviations from your own historical mean is a genuine
outlier worth calling out explicitly.

For each tracked metric with at least `min_history_runs` prior observations,
this computes

    z = (current - mean_history) / std_history

using the sample mean/standard deviation of all *prior* runs (the current
run is never included in its own baseline, otherwise a single extreme value
would partially normalize itself). |z| is then compared against two
thresholds from config.yaml (anomaly_detection.z_score_watch_threshold /
z_score_flag_threshold) to classify the metric as normal / watch / flagged.

This is a standard z-score / "three-sigma-rule"-style control-chart
technique (Shewhart, 1931) applied to a rolling personal baseline instead of
a fixed target -- appropriate here because "normal" prompting behavior
varies a lot between users and there is no universal reference population.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import load_config

# (metric key in metrics_json, display label, direction where "higher region
# of concern" means large values are the bad direction)
TRACKED_METRICS = [
    ("vague_rate", "Vague-language rate", "higher_is_bad"),
    ("file_ref_rate", "File/code reference rate", "lower_is_bad"),
    ("correction_rate", "Correction rate", "higher_is_bad"),
    ("acceptance_rate", "Acceptance-criteria rate", "lower_is_bad"),
    ("multi_ask_rate", "Multi-ask (bundled requests) rate", "higher_is_bad"),
    ("clarification_rate", "Clarification-loop rate", "higher_is_bad"),
    ("context_restatement_rate", "Context-restatement rate", "higher_is_bad"),
    ("single_shot_rate", "Single-shot session rate", "lower_is_bad"),
    ("avg_word_count", "Average prompt length (words)", "lower_is_bad"),
]


@dataclass
class AnomalyFlag:
    metric_key: str
    label: str
    current_value: float
    historical_mean: float
    historical_std: float
    z_score: float
    n_history: int
    status: str  # 'normal' | 'watch' | 'flagged'
    concerning: bool  # True if the anomaly is in the "bad" direction


def compute_anomalies(run_history_metrics: list[dict], current_metric_values: dict) -> list[AnomalyFlag]:
    """run_history_metrics: list of parsed metrics_json dicts from PRIOR runs
    only (oldest first), i.e. excluding the run currently being generated."""
    cfg = load_config()["anomaly_detection"]
    min_runs = cfg["min_history_runs"]
    watch_z = cfg["z_score_watch_threshold"]
    flag_z = cfg["z_score_flag_threshold"]

    if len(run_history_metrics) < min_runs:
        return []

    results = []
    for key, label, bad_direction in TRACKED_METRICS:
        history_vals = [m[key] for m in run_history_metrics if key in m and m[key] is not None]
        if len(history_vals) < min_runs:
            continue
        current = current_metric_values.get(key)
        if current is None:
            continue

        n = len(history_vals)
        mean = sum(history_vals) / n
        var = sum((v - mean) ** 2 for v in history_vals) / n
        std = math.sqrt(var)

        if std < 1e-9:
            z = 0.0 if abs(current - mean) < 1e-9 else (10.0 if current > mean else -10.0)
        else:
            z = (current - mean) / std

        abs_z = abs(z)
        if abs_z >= flag_z:
            status = "flagged"
        elif abs_z >= watch_z:
            status = "watch"
        else:
            status = "normal"

        concerning = (
            (bad_direction == "higher_is_bad" and z > 0)
            or (bad_direction == "lower_is_bad" and z < 0)
        )

        results.append(AnomalyFlag(
            metric_key=key, label=label, current_value=current,
            historical_mean=mean, historical_std=std, z_score=z,
            n_history=n, status=status, concerning=concerning,
        ))

    # Surface the most statistically unusual metrics first.
    results.sort(key=lambda a: -abs(a.z_score))
    return results
