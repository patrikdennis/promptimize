"""Tests for lib/linalg.py: pure-Python matrix helpers (no numpy)."""
import math

import linalg


def test_invert_matrix_identity():
    identity = [[1, 0], [0, 1]]
    inv = linalg.invert_matrix(identity)
    assert inv == [[1.0, 0.0], [0.0, 1.0]]


def test_invert_matrix_known_2x2():
    # [[4, 7], [2, 6]] has determinant 10, inverse [[0.6, -0.7], [-0.2, 0.4]]
    m = [[4.0, 7.0], [2.0, 6.0]]
    inv = linalg.invert_matrix(m)
    expected = [[0.6, -0.7], [-0.2, 0.4]]
    for r in range(2):
        for c in range(2):
            assert math.isclose(inv[r][c], expected[r][c], abs_tol=1e-9)


def test_invert_matrix_singular_returns_none():
    singular = [[1.0, 2.0], [2.0, 4.0]]
    assert linalg.invert_matrix(singular) is None


def test_invert_matrix_roundtrip_3x3():
    m = [[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]]
    inv = linalg.invert_matrix(m)
    assert inv is not None
    # m @ inv should be (approximately) the identity matrix.
    product = [[sum(m[i][k] * inv[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert math.isclose(product[i][j], expected, abs_tol=1e-9)


def test_matvec():
    m = [[1.0, 2.0], [3.0, 4.0]]
    v = [5.0, 6.0]
    assert linalg.matvec(m, v) == [1 * 5 + 2 * 6, 3 * 5 + 4 * 6]


def test_dot():
    assert linalg.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0


def test_covariance_matrix_zero_variance_when_constant_rows():
    rows = [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
    cov = linalg.covariance_matrix(rows)
    assert cov == [[0.0, 0.0], [0.0, 0.0]]


def test_covariance_matrix_matches_manual_calculation():
    # Two perfectly-correlated variables: y = 2x.
    rows = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    cov = linalg.covariance_matrix(rows)
    # var(x) with N-1 denominator = 1.0, var(y) = 4.0, cov(x,y) = 2.0
    assert math.isclose(cov[0][0], 1.0, abs_tol=1e-9)
    assert math.isclose(cov[1][1], 4.0, abs_tol=1e-9)
    assert math.isclose(cov[0][1], 2.0, abs_tol=1e-9)
    assert math.isclose(cov[1][0], 2.0, abs_tol=1e-9)
