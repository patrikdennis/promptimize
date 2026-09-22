"""
Mahalanobis distance from the current score position to the optimal corner.

The Overview tab's "distance to optimal" was previously a raw Euclidean
distance across the four axis scores, which implicitly assumes the axes are
independent and equally scaled. In reality, specificity and context-anchoring
(for example) tend to move together across your own history — a prompt that
references a concrete file is usually also more specific. The Mahalanobis
distance

    D_M(x) = sqrt( (x - mu)^T * Sigma^-1 * (x - mu) )

instead measures distance in units of the *actual observed covariance*
Sigma of your own historical score vectors, so it does not penalize you
twice for two axes that are naturally correlated, and it weighs an
unusual *combination* of axis values (e.g. very high structure but very low
efficiency, which rarely co-occurs in your own history) as more surprising
than the same total Euclidean gap spread evenly across axes that normally
move together.

Sigma is estimated from your own run history (needs >= min_history_runs
observations to be estimable at all for a 4x4 covariance matrix); with too
little history this module returns `None` and the report falls back to
plain Euclidean distance, which is clearly labeled as an approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import load_config
from linalg import covariance_matrix, invert_matrix, matvec, dot

AXES = ["specificity", "context", "structure", "efficiency"]


@dataclass
class MahalanobisResult:
    distance: float
    euclidean_distance: float
    n_history_runs: int
    covariance: list[list[float]]
    mean: list[float]


def compute_mahalanobis_to_optimal(run_history: list[dict], current_scores: dict) -> MahalanobisResult | None:
    """run_history: list of dicts with score_specificity/score_context/score_structure/score_efficiency
    (as produced by history.get_run_history), OLDEST FIRST. current_scores: agg.scores dict."""
    cfg = load_config()["mahalanobis"]
    min_runs = cfg["min_history_runs"]
    shrinkage = cfg["ridge_epsilon"]

    rows = [
        [r["score_specificity"], r["score_context"], r["score_structure"], r["score_efficiency"]]
        for r in run_history
    ]
    if len(rows) < min_runs:
        return None

    optimal = [100.0, 100.0, 100.0, 100.0]
    current = [current_scores.get(a, 0.0) for a in AXES]
    euclidean = math.sqrt(sum((optimal[i] - current[i]) ** 2 for i in range(4)))

    cov = covariance_matrix(rows)
    # Shrinkage regularization (Ledoit-Wolf-style, simplified): with only a
    # handful of historical runs, the raw 4x4 sample covariance matrix is
    # estimated from very few degrees of freedom and can be nearly singular
    # or badly conditioned, which makes its inverse (and therefore the
    # Mahalanobis distance) numerically explode even though nothing
    # meaningful changed. The standard fix is to shrink the sample estimate
    # toward a well-conditioned target -- here, a diagonal matrix holding the
    # average per-axis variance -- by a shrinkage intensity in [0, 1]:
    #
    #   Sigma_reg = (1 - alpha) * Sigma_sample + alpha * avg_variance * I
    #
    # alpha = 0 recovers the raw sample covariance; alpha = 1 collapses to a
    # scaled identity (equivalent to plain, variance-normalized Euclidean
    # distance). `ridge_epsilon` in config.yaml is this alpha.
    avg_var = sum(cov[i][i] for i in range(4)) / 4
    avg_var = max(avg_var, 1e-6)
    alpha = min(1.0, max(0.0, shrinkage))
    cov_reg = [
        [(1 - alpha) * cov[i][j] + (alpha * avg_var if i == j else 0.0) for j in range(4)]
        for i in range(4)
    ]

    inv_cov = invert_matrix(cov_reg)
    if inv_cov is None:
        return None

    diff = [optimal[i] - current[i] for i in range(4)]
    quad_form = dot(diff, matvec(inv_cov, diff))
    distance = math.sqrt(max(0.0, quad_form))

    mean = [sum(r[j] for r in rows) / len(rows) for j in range(4)]

    return MahalanobisResult(
        distance=distance,
        euclidean_distance=euclidean,
        n_history_runs=len(rows),
        covariance=cov_reg,
        mean=mean,
    )
