"""
tests/test_pca.py
-----------------
Smoke tests and shape checks for PCAReducer and the DimensionalityReducer
interface. Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest

from Neuroreduce import DimensionalityReducer, PCAReducer


# ── fixtures ──────────────────────────────────────────────────────────────────

N, T, k = 80, 200, 10   # parcels, timepoints, components
rng = np.random.default_rng(42)


@pytest.fixture
def X():
    """Synthetic BOLD signal, shape (N, T)."""
    return rng.standard_normal((N, T)).astype(np.float32)


@pytest.fixture
def SC():
    """Synthetic symmetric SC matrix, shape (N, N)."""
    A = rng.random((N, N)).astype(np.float32)
    return (A + A.T) / 2


# ── interface contract ────────────────────────────────────────────────────────

def test_is_abstract():
    """DimensionalityReducer cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DimensionalityReducer(k=k)


def test_repr_before_fit():
    r = PCAReducer(k=k)
    assert "not fitted" in repr(r)


def test_repr_after_fit(X):
    r = PCAReducer(k=k).fit(X)
    assert "fitted" in repr(r)


def test_transform_before_fit_raises(X):
    with pytest.raises(RuntimeError, match="not fitted"):
        PCAReducer(k=k).transform(X)


# ── input validation ──────────────────────────────────────────────────────────

def test_wrong_ndim_raises():
    with pytest.raises(ValueError, match="2-D"):
        PCAReducer(k=k).fit(rng.random((N, T, 3)))


def test_transposed_input_hint(X):
    """If user passes (T, N) instead of (N, T) and T < k, a helpful error fires."""
    X_wrong = rng.random((5, N)).astype(np.float32)   # only 5 "parcels"
    with pytest.raises(ValueError, match="shape \\(N, T\\)"):
        PCAReducer(k=k).fit(X_wrong)


def test_SC_ignored_by_pca(X, SC):
    """PCA silently ignores SC; no error should be raised."""
    r = PCAReducer(k=k)
    r.fit(X, SC=SC)                  # should not raise
    assert r._is_fitted


# ── output shapes ─────────────────────────────────────────────────────────────

def test_fit_transform_shape(X):
    Z = PCAReducer(k=k).fit_transform(X)
    assert Z.shape == (k, T), f"Expected ({k}, {T}), got {Z.shape}"


def test_transform_shape(X):
    r = PCAReducer(k=k).fit(X)
    Z = r.transform(X)
    assert Z.shape == (k, T)


def test_basis_shape(X):
    r = PCAReducer(k=k).fit(X)
    W = r.get_basis()
    assert W.shape == (N, k), f"Expected ({N}, {k}), got {W.shape}"


def test_inverse_transform_shape(X):
    r = PCAReducer(k=k).fit(X)
    Z = r.transform(X)
    X_hat = r.inverse_transform(Z)
    assert X_hat.shape == (N, T)


# ── numerical correctness ─────────────────────────────────────────────────────

def test_explained_variance_sum(X):
    """Sum of explained variance ratios must be <= 1."""
    r = PCAReducer(k=k).fit(X)
    total = r.explained_variance_ratio_.sum()
    assert 0.0 < total <= 1.0 + 1e-6


def test_cumulative_variance_monotone(X):
    r = PCAReducer(k=k).fit(X)
    cumvar = r.cumulative_explained_variance_
    assert np.all(np.diff(cumvar) >= 0)


def test_score_in_unit_interval(X):
    r = PCAReducer(k=k).fit(X)
    s = r.score(X)
    assert 0.0 <= s <= 1.0 + 1e-6, f"score={s} out of [0, 1]"


def test_reconstruction_decreases_with_k(X):
    """More components → better reconstruction."""
    scores = [PCAReducer(k=ki).fit(X).score(X) for ki in [2, 5, 10, 20]]
    assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))


def test_basis_orthonormality(X):
    """Principal components must be orthonormal: W.T @ W ≈ I."""
    W = PCAReducer(k=k).fit(X).get_basis()   # (N, k)
    I_approx = W.T @ W                        # (k, k)
    assert np.allclose(I_approx, np.eye(k), atol=1e-5), \
        "Basis vectors are not orthonormal"


def test_whitening_unit_variance(X):
    """With whiten=True, each row of Z should have variance ≈ 1."""
    Z = PCAReducer(k=k, whiten=True).fit_transform(X)
    row_vars = Z.var(axis=1)
    assert np.allclose(row_vars, 1.0, atol=1e-5), \
        f"Row variances after whitening: {row_vars}"


def test_whitening_zero_mean(X):
    """With whiten=True, each row of Z should have mean ≈ 0."""
    Z = PCAReducer(k=k, whiten=True).fit_transform(X)
    row_means = Z.mean(axis=1)
    assert np.allclose(row_means, 0.0, atol=1e-5)
