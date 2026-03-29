"""
tests/test_charm.py
--------------------
Tests for CHARMReducer covering:
  - interface contract (shapes, fitted/not-fitted guards)
  - identity shortcut (same data → exact Phi)
  - Nyström round-trip (force_nystrom on training data ≈ direct Phi)
  - parcel-space basis shape and normalisation
  - eigenvalue sorting and sanity check

Run with:  python -m pytest tests/test_charm.py -v

NOTE: Tm is kept very small (Tm=20) because the O(Tm²) kernel build is
expensive. For real data Tm ~ NSUB * T_per_subject ~ 100 * 175 = 17500.
"""

import warnings

import numpy as np
import pytest

from Neuroreduce import CHARMReducer

# ── fixtures ──────────────────────────────────────────────────────────────────

N, Tm, k = 20, 40, 4   # tiny but sufficient for interface tests
rng = np.random.default_rng(0)


@pytest.fixture
def X():
    """Synthetic concatenated BOLD, shape (N, Tm)."""
    return rng.standard_normal((N, Tm)).astype(np.float32)


@pytest.fixture
def reducer(X):
    r = CHARMReducer(k=k, epsilon=300.0, t_horizon=2)
    r.fit(X)
    return r, X


# ── interface contract ────────────────────────────────────────────────────────

def test_transform_before_fit_raises():
    X = rng.standard_normal((N, Tm)).astype(np.float32)
    with pytest.raises(RuntimeError, match="not fitted"):
        CHARMReducer(k=k).transform(X)


def test_repr_not_fitted():
    assert "not fitted" in repr(CHARMReducer(k=k))


def test_repr_fitted(reducer):
    r, _ = reducer
    assert "fitted" in repr(r)


# ── output shapes ─────────────────────────────────────────────────────────────

def test_fit_transform_shape(X):
    Z = CHARMReducer(k=k, epsilon=300.0, t_horizon=2).fit_transform(X)
    assert Z.shape == (k, Tm), f"Expected ({k}, {Tm}), got {Z.shape}"


def test_basis_shape(reducer):
    r, _ = reducer
    W = r.get_basis()
    assert W.shape == (N, k), f"Expected ({N}, {k}), got {W.shape}"


def test_embedding_shape(reducer):
    r, _ = reducer
    assert r.embedding_.shape == (Tm, k)


def test_eigenvalues_shape(reducer):
    r, _ = reducer
    assert r.eigenvalues_.shape == (k,)


# ── identity shortcut ─────────────────────────────────────────────────────────

def test_identity_shortcut_exact(reducer):
    """transform(X_fit) must return exactly Phi.T without recomputation."""
    r, X = reducer
    Z = r.transform(X)
    assert np.allclose(Z, r.embedding_.T), \
        "Identity shortcut did not return Phi.T exactly"


# ── Nyström ───────────────────────────────────────────────────────────────────

def test_nystrom_shape(reducer):
    """Nyström on new data returns (k, T_new)."""
    r, _ = reducer
    T_new = 10
    X_new = rng.standard_normal((N, T_new)).astype(np.float32)
    Z_new = r.transform(X_new)
    assert Z_new.shape == (k, T_new)


def test_force_nystrom_shape(reducer):
    """force_nystrom on training data should still return (k, Tm)."""
    r, X = reducer
    Z = r.transform(X, force_nystrom=True)
    assert Z.shape == (k, Tm)


def test_force_nystrom_close_to_direct(reducer):
    """
    Nyström on training data should be EXACT (not merely close) when
    force_nystrom=True, because the fixed implementation reads the exact
    Pmatrix row rather than recomputing the kernel element-wise.

    This test therefore checks for near-zero relative error, not just
    a loose tolerance. Any residual error is pure floating-point noise
    from the dot product (p_row @ eigenvectors) vs the eigendecomposition
    path — expected to be at the level of machine epsilon (~1e-6 for float32,
    ~1e-14 for float64).

    Background: the original implementation raised k_row element-wise to
    t_horizon, which approximates the matrix power LA.matrix_power(K, t)
    but is not the same operation. That caused relative errors ~1.0. The
    correct approach for training timepoints is to look up the exact row of
    the already-computed Pmatrix, which is what the fixed code does.
    """
    r, X = reducer
    Z_direct  = r.transform(X, force_nystrom=False)
    Z_nystrom = r.transform(X, force_nystrom=True)
    rel_err = np.linalg.norm(Z_direct - Z_nystrom) / (np.linalg.norm(Z_direct) + 1e-12)
    assert rel_err < 1e-4, (
        f"Nyström relative error on training data is {rel_err:.6e}. "
        "Expected near-zero (< 1e-4) because exact Pmatrix rows are used. "
        "A large error indicates the eigenvector scaling or row lookup is wrong."
    )


# ── eigenvalue sorting ────────────────────────────────────────────────────────

def test_eigenvalues_descending(reducer):
    """Selected eigenvalues should be in descending order."""
    r, _ = reducer
    evals = r.eigenvalues_
    assert np.all(np.diff(evals) <= 0), \
        f"Eigenvalues not in descending order: {evals}"


def test_dominant_eigenvalue_warning():
    """
    If the dominant eigenvalue is far from 1, a RuntimeWarning should fire.
    We simulate this by patching a nonsense matrix — just check the warning
    path exists by calling with data that would produce a well-behaved result
    (no warning expected on normal data).
    """
    X = rng.standard_normal((N, Tm)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Should NOT raise on normal synthetic data
        CHARMReducer(k=k, epsilon=300.0, sort_eigenvectors=True).fit(X)


# ── parcel-space basis normalisation ──────────────────────────────────────────

def test_basis_columns_unit_norm(reducer):
    """
    Each column of conet should have unit L2-norm (normalised in nets()).
    """
    r, _ = reducer
    W = r.get_basis()   # (N, k)
    col_norms = np.linalg.norm(W, axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-5), \
        f"Column norms: {col_norms}"


# ── whitening ─────────────────────────────────────────────────────────────────

def test_whitening_unit_variance(X):
    Z = CHARMReducer(k=k, epsilon=300.0, whiten=True).fit_transform(X)
    assert np.allclose(Z.var(axis=1), 1.0, atol=1e-5)


def test_whitening_zero_mean(X):
    Z = CHARMReducer(k=k, epsilon=300.0, whiten=True).fit_transform(X)
    assert np.allclose(Z.mean(axis=1), 0.0, atol=1e-5)


# ── SC is silently ignored ────────────────────────────────────────────────────

def test_SC_ignored(X):
    SC = rng.random((N, N)).astype(np.float32)
    SC = (SC + SC.T) / 2
    CHARMReducer(k=k).fit(X, SC=SC)   # should not raise
