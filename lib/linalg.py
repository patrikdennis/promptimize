"""
Tiny pure-Python linear algebra helpers shared by advanced_metrics.py and
mahalanobis.py. No numpy dependency (matches the rest of this tool).
"""
from __future__ import annotations


def invert_matrix(m: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan inversion with partial pivoting. m must be square.
    Returns None if m is singular (or numerically indistinguishable from it)."""
    n = len(m)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            return None
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


def matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def covariance_matrix(rows: list[list[float]]) -> list[list[float]]:
    """Sample covariance matrix (N-1 denominator) of a list of observation
    vectors (each row is one observation, e.g. one historical run's axis
    scores)."""
    n = len(rows)
    d = len(rows[0])
    means = [sum(r[j] for r in rows) / n for j in range(d)]
    cov = [[0.0] * d for _ in range(d)]
    denom = max(1, n - 1)
    for r in rows:
        dev = [r[j] - means[j] for j in range(d)]
        for i in range(d):
            for j in range(d):
                cov[i][j] += dev[i] * dev[j] / denom
    return cov
