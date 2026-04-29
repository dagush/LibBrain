"""
tests/test_charm_sc.py
-----------------------
Tests for CHARMSCReducer — geometry-driven CHARM dimensionality reduction.

Uses synthetic parcel coordinates so no real HCP data is needed.
Two modes are tested:
  - Pure geometry (no BOLD provided to fit)
  - Data-enriched (BOLD provided to fit, basis via nets())

Run with:  python -m pytest tests/test_charm_sc.py -v
"""

import numpy as np
import pytest

from Neuroreduce import CHARMSCReducer


# ── shared parameters ─────────────────────────────────────────────────────────

N, T, k = 40, 100, 5    # parcels, timepoints, latent dims
rng = np.random.default_rng(42)


@pytest.fixture
def coords():
    """Synthetic parcel centroids in MNI-like space, shape (N, 3)."""
    return rng.uniform(-100, 100, size=(N, 3)).astype(np.float64)


@pytest.fixture
def X():
    """Synthetic BOLD signal, shape (N, T)."""
    return rng.standard_normal((N, T)).astype(np.float32)


@pytest.fixture
def reducer_geometry(coords):
    """Pure geometry fit — no BOLD."""
    r = CHARMSCReducer(k=k, coords=coords, epsilon=1400.0,
                       t_horizon=2, sort_eigenvectors=True)
    r.fit()
    return r


@pytest.fixture
def reducer_bold(coords, X):
    """Data-enriched fit — BOLD provided."""
    r = CHARMSCReducer(k=k, coords=coords, epsilon=1400.0,
                       t_horizon=2, sort_eigenvectors=True)
    r.fit(X)
    return r


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:

    def test_wrong_coords_ndim_raises(self):
        with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
            CHARMSCReducer(k=k, coords=np.ones((N, 2)))

    def test_repr_not_fitted(self, coords):
        r = CHARMSCReducer(k=k, coords=coords)
        assert "not fitted" in repr(r)

    def test_repr_fitted_geometry(self, reducer_geometry):
        assert "fitted" in repr(reducer_geometry)
        assert "geometry only" in repr(reducer_geometry)

    def test_repr_fitted_bold(self, reducer_bold):
        assert "fitted" in repr(reducer_bold)
        assert "with BOLD" in repr(reducer_bold)

    def test_k_ge_N_raises(self, coords):
        with pytest.raises(ValueError, match="k="):
            CHARMSCReducer(k=N, coords=coords).fit()

    def test_transform_before_fit_raises(self, coords, X):
        with pytest.raises(RuntimeError, match="not fitted"):
            CHARMSCReducer(k=k, coords=coords).transform(X)


# ── Diffusion matrix ──────────────────────────────────────────────────────────

class TestDiffusionMatrix:

    def test_Pmatrix_shape(self, reducer_geometry):
        assert reducer_geometry._Pmatrix.shape == (N, N)

    def test_Pmatrix_row_stochastic(self, reducer_geometry):
        assert np.allclose(reducer_geometry._Pmatrix.sum(axis=1), 1.0, atol=1e-6)

    def test_Pmatrix_non_negative(self, reducer_geometry):
        assert np.all(reducer_geometry._Pmatrix >= 0)

    def test_Pmatrix_real(self, reducer_geometry):
        assert np.isrealobj(reducer_geometry._Pmatrix)

    def test_Ptr_t_symmetric(self, reducer_geometry):
        """
        Q = |K^t|² is symmetric because the coordinate-based kernel
        K[i,j] = exp(i·||c_i-c_j||²/σ) is symmetric (d²_ij = d²_ji).
        """
        Q = reducer_geometry._Ptr_t
        assert np.allclose(Q, Q.T, atol=1e-8)


# ── Basis (get_basis) ─────────────────────────────────────────────────────────

class TestBasis:

    def test_basis_shape_geometry(self, reducer_geometry):
        W = reducer_geometry.get_basis()
        assert W.shape == (N, k)

    def test_basis_shape_bold(self, reducer_bold):
        W = reducer_bold.get_basis()
        assert W.shape == (N, k)

    def test_basis_columns_unit_norm_geometry(self, reducer_geometry):
        """Pure geometry basis columns must be unit-norm (L2-normalised)."""
        W     = reducer_geometry.get_basis()
        norms = np.linalg.norm(W, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_basis_columns_unit_norm_bold(self, reducer_bold):
        """conet columns must be unit-norm (normalised in nets())."""
        W     = reducer_bold.get_basis()
        norms = np.linalg.norm(W, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_bold_flag_false_geometry(self, reducer_geometry):
        assert reducer_geometry.bold_fitted is False

    def test_bold_flag_true_bold(self, reducer_bold):
        assert reducer_bold.bold_fitted is True

    def test_geometry_and_bold_bases_differ(self, coords, X):
        """BOLD enrichment must change the basis."""
        rg = CHARMSCReducer(k=k, coords=coords).fit()
        rb = CHARMSCReducer(k=k, coords=coords).fit(X)
        assert not np.allclose(rg.get_basis(), rb.get_basis(), atol=1e-4)


# ── transform ─────────────────────────────────────────────────────────────────

class TestTransform:

    def test_transform_shape_geometry(self, reducer_geometry, X):
        Z = reducer_geometry.transform(X)
        assert Z.shape == (k, T)

    def test_transform_shape_bold(self, reducer_bold, X):
        Z = reducer_bold.transform(X)
        assert Z.shape == (k, T)

    def test_transform_is_linear_projection(self, reducer_geometry, X):
        """transform(X) must equal W.T @ X exactly."""
        Z = reducer_geometry.transform(X)
        W = reducer_geometry.get_basis()
        assert np.allclose(Z, W.T @ X, atol=1e-5)

    def test_transform_new_data(self, reducer_geometry):
        """Transform on genuinely new data must return (k, T_new)."""
        X_new = rng.standard_normal((N, 30)).astype(np.float32)
        Z_new = reducer_geometry.transform(X_new)
        assert Z_new.shape == (k, 30)

    def test_whitening(self, coords, X):
        Z = CHARMSCReducer(k=k, coords=coords, whiten=True).fit().transform(X)
        assert np.allclose(Z.var(axis=1), 1.0, atol=1e-5)
        assert np.allclose(Z.mean(axis=1), 0.0, atol=1e-5)


# ── inverse_transform ─────────────────────────────────────────────────────────

class TestInverseTransform:

    def test_inverse_transform_shape(self, reducer_geometry, X):
        Z    = reducer_geometry.transform(X)
        X_hat = reducer_geometry.inverse_transform(Z)
        assert X_hat.shape == (N, T)

    def test_inverse_transform_is_W_at_Z(self, reducer_geometry, X):
        """inverse_transform(Z) must equal W @ Z exactly."""
        Z     = reducer_geometry.transform(X)
        X_hat = reducer_geometry.inverse_transform(Z)
        W     = reducer_geometry.get_basis()
        assert np.allclose(X_hat, W @ Z, atol=1e-5)

    def test_score_in_unit_interval(self, reducer_geometry, X):
        s = reducer_geometry.score(X)
        assert -0.1 <= s <= 1.0 + 1e-6


# ── stationary_distribution_ ──────────────────────────────────────────────────

class TestStationaryDistribution:

    def test_shape(self, reducer_geometry):
        p = reducer_geometry.stationary_distribution_
        assert p.shape == (N,)

    def test_non_negative(self, reducer_geometry):
        p = reducer_geometry.stationary_distribution_
        assert np.all(p >= -1e-10)

    def test_sums_to_one(self, reducer_geometry):
        """For a row-stochastic P, all rows of P^n sum to 1."""
        p = reducer_geometry.stationary_distribution_
        assert np.isclose(p.sum(), 1.0, atol=1e-5)


# ── Eigenvalues ───────────────────────────────────────────────────────────────

class TestEigenvalues:

    def test_eigenvalues_shape(self, reducer_geometry):
        assert reducer_geometry.eigenvalues_.shape == (k,)

    def test_eigenvalues_descending(self, reducer_geometry):
        ev = reducer_geometry.eigenvalues_
        assert np.all(np.diff(ev) <= 0)

    def test_embedding_shape(self, reducer_geometry):
        assert reducer_geometry.embedding_.shape == (N, k)


# ── Epsilon and t_horizon sensitivity ────────────────────────────────────────

class TestSensitivity:

    def test_epsilon_affects_basis(self, coords):
        r1 = CHARMSCReducer(k=k, coords=coords, epsilon=500.0).fit()
        r2 = CHARMSCReducer(k=k, coords=coords, epsilon=1400.0).fit()
        assert not np.allclose(r1.get_basis(), r2.get_basis(), atol=1e-4)

    def test_t_horizon_affects_basis(self, coords):
        r1 = CHARMSCReducer(k=k, coords=coords, t_horizon=1).fit()
        r2 = CHARMSCReducer(k=k, coords=coords, t_horizon=2).fit()
        assert not np.allclose(r1.get_basis(), r2.get_basis(), atol=1e-4)

    def test_different_from_charm_bold(self, coords, X):
        """
        CHARM-SC (geometry basis) must differ from CHARM-BOLD (BOLD basis).
        Both use the same kernel math but different input → different P.
        """
        from Neuroreduce import CHARMReducer
        rb = CHARMReducer(k=k, epsilon=300.0, t_horizon=2).fit(X)
        rg = CHARMSCReducer(k=k, coords=coords, epsilon=1400.0).fit()
        # Bases live in same space (N×k) — they should differ
        assert not np.allclose(rb.get_basis(), rg.get_basis(), atol=1e-4)
