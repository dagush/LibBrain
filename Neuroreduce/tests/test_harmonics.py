"""
Neuroreduce/tests/test_harmonics.py
-------------------------------------
Tests for ConnectomeHarmonicsReducer, FunctionalHarmonicsReducer,
and HarmonicAnalysis.

All tests use synthetic data — no real SC/BOLD needed.

Run with:  python -m pytest tests/test_harmonics.py -v

Synthetic setup
---------------
    N = 30  parcels
    T = 80  timepoints
    k = 5   harmonics retained
    n_rsn = 7  RSN binary vectors
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from Neuroreduce import ConnectomeHarmonicsReducer, FunctionalHarmonicsReducer
from Neuroreduce.utils.harmonic_analysis import HarmonicAnalysis


# ── shared parameters ─────────────────────────────────────────────────────────

N, T, k, n_rsn = 30, 80, 5, 7
rng = np.random.default_rng(42)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def SC():
    """Synthetic symmetric non-negative SC matrix, shape (N, N)."""
    A = rng.random((N, N))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    return A.astype(np.float64)


@pytest.fixture
def X():
    """Synthetic BOLD timeseries, shape (N, T)."""
    return rng.standard_normal((N, T)).astype(np.float32)


@pytest.fixture
def RSN_matrix():
    """Synthetic binary RSN matrix, shape (N, n_rsn)."""
    mat = np.zeros((N, n_rsn))
    for r in range(n_rsn):
        indices = rng.choice(N, size=N // n_rsn, replace=False)
        mat[indices, r] = 1.0
    return mat


@pytest.fixture
def ch_reducer(SC):
    r = ConnectomeHarmonicsReducer(k=k, threshold=0.0)
    r.fit(SC=SC)
    return r


@pytest.fixture
def fh_reducer(X):
    r = FunctionalHarmonicsReducer(k=k, threshold=0.0)
    r.fit(X=X)
    return r


@pytest.fixture
def ch_analyser(ch_reducer):
    return HarmonicAnalysis(ch_reducer)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:

    def test_ch_requires_SC(self):
        r = ConnectomeHarmonicsReducer(k=k)
        with pytest.raises(ValueError, match="SC"):
            r.fit()

    def test_fh_requires_X(self):
        r = FunctionalHarmonicsReducer(k=k)
        with pytest.raises(ValueError, match="X"):
            r.fit()

    def test_k_ge_N_raises(self, SC):
        with pytest.raises(ValueError, match="k="):
            ConnectomeHarmonicsReducer(k=N).fit(SC=SC)

    def test_invalid_laplacian_type_raises(self):
        with pytest.raises(ValueError, match="laplacian_type"):
            ConnectomeHarmonicsReducer(k=k, laplacian_type='invalid')

    def test_transform_before_fit_raises(self, X):
        with pytest.raises(RuntimeError, match="not fitted"):
            ConnectomeHarmonicsReducer(k=k).transform(X)

    def test_repr_not_fitted(self):
        assert "not fitted" in repr(ConnectomeHarmonicsReducer(k=k))

    def test_repr_fitted(self, ch_reducer):
        assert "fitted" in repr(ch_reducer)


# ── Laplacian and eigenvectors ────────────────────────────────────────────────

class TestLaplacian:

    def test_eigenvalues_non_negative(self, ch_reducer):
        """
        Graph Laplacian eigenvalues must be >= 0 (positive semi-definite).
        Small numerical noise may give tiny negatives — allow 1e-8 tolerance.
        """
        ev = ch_reducer.eigenvalues_
        assert np.all(ev >= -1e-8), f"Negative eigenvalues: {ev[ev < -1e-8]}"

    def test_eigenvalues_sorted_ascending(self, ch_reducer):
        ev = ch_reducer.eigenvalues_
        assert np.all(np.diff(ev) >= -1e-10)

    def test_first_eigenvalue_near_zero(self, ch_reducer):
        """
        The smallest Laplacian eigenvalue of a connected graph is 0.
        The corresponding eigenvector is the constant vector (DC component).
        """
        assert np.isclose(ch_reducer.eigenvalues_[0], 0.0, atol=1e-6), \
            f"First eigenvalue = {ch_reducer.eigenvalues_[0]:.2e}, expected ≈ 0"

    def test_all_eigenvectors_shape(self, ch_reducer):
        assert ch_reducer.get_all_eigenvectors().shape == (N, N)

    def test_basis_shape(self, ch_reducer):
        assert ch_reducer.get_basis().shape == (N, k)

    def test_basis_is_first_k_of_all(self, ch_reducer):
        """get_basis() must return the first k columns of get_all_eigenvectors()."""
        W_all = ch_reducer.get_all_eigenvectors()
        W_k   = ch_reducer.get_basis()
        assert np.allclose(W_all[:, :k], W_k)

    def test_symmetric_laplacian_variant(self, SC):
        """Symmetric Laplacian should give eigenvalues in [0, 2]."""
        r = ConnectomeHarmonicsReducer(k=k, threshold=0.0,
                                       laplacian_type='symmetric').fit(SC=SC)
        ev = r.eigenvalues_
        assert np.all(ev >= -1e-8)
        assert np.all(ev <= 2.0 + 1e-8)


# ── ConnectomeHarmonicsReducer ────────────────────────────────────────────────

class TestConnectomeHarmonics:

    def test_transform_shape(self, ch_reducer, X):
        Z = ch_reducer.transform(X)
        assert Z.shape == (k, T)

    def test_transform_sign_invariant_non_negative(self, ch_reducer, X):
        """With sign_invariant=True, all projections must be >= 0."""
        Z = ch_reducer.transform(X, sign_invariant=True)
        assert np.all(Z >= 0)

    def test_transform_sign_invariant_equals_abs(self, ch_reducer, X):
        """sign_invariant=True must equal abs(sign_invariant=False)."""
        Z_signed = ch_reducer.transform(X, sign_invariant=False)
        Z_abs    = ch_reducer.transform(X, sign_invariant=True)
        assert np.allclose(Z_abs, np.abs(Z_signed), atol=1e-5)

    def test_transform_is_linear_projection(self, ch_reducer, X):
        """transform(X, sign_invariant=False) must equal W.T @ X."""
        Z = ch_reducer.transform(X, sign_invariant=False)
        W = ch_reducer.get_basis()
        assert np.allclose(Z, W.T @ X, atol=1e-5)

    def test_inverse_transform_shape(self, ch_reducer, X):
        Z    = ch_reducer.transform(X, sign_invariant=False)
        X_hat = ch_reducer.inverse_transform(Z)
        assert X_hat.shape == (N, T)

    def test_score_in_range(self, ch_reducer, X):
        s = ch_reducer.score(X)
        assert -0.1 <= s <= 1.0 + 1e-6

    def test_normalise_input_flag(self, SC):
        """Normalise flag should not change eigenvector directions, only scale."""
        r1 = ConnectomeHarmonicsReducer(k=k, threshold=0.0,
                                        normalise_input=True).fit(SC=SC)
        r2 = ConnectomeHarmonicsReducer(k=k, threshold=0.0,
                                        normalise_input=False).fit(SC=SC)
        # Eigenvectors should be identical (or sign-flipped) — directions same
        W1, W2 = r1.get_basis(), r2.get_basis()
        for d in range(k):
            r = np.abs(np.dot(W1[:, d], W2[:, d]))
            assert np.isclose(r, 1.0, atol=1e-4), \
                f"Harmonic {d} direction changed with normalise_input flag"

    def test_threshold_affects_basis(self, SC):
        """Higher threshold zeros more edges → different Laplacian → different basis."""
        r1 = ConnectomeHarmonicsReducer(k=k, threshold=0.0).fit(SC=SC)
        r2 = ConnectomeHarmonicsReducer(k=k, threshold=0.3).fit(SC=SC)
        assert not np.allclose(r1.get_basis(), r2.get_basis(), atol=1e-4)


# ── FunctionalHarmonicsReducer ────────────────────────────────────────────────

class TestFunctionalHarmonics:

    def test_transform_shape(self, fh_reducer, X):
        assert fh_reducer.transform(X).shape == (k, T)

    def test_eigenvalues_non_negative(self, fh_reducer):
        """FC Laplacian is also positive semi-definite."""
        assert np.all(fh_reducer.eigenvalues_ >= -1e-8)

    def test_basis_shape(self, fh_reducer):
        assert fh_reducer.get_basis().shape == (N, k)

    def test_different_from_connectome(self, ch_reducer, fh_reducer):
        """SC and FC harmonics must differ (different input matrices)."""
        assert not np.allclose(
            ch_reducer.get_basis(), fh_reducer.get_basis(), atol=1e-4
        )

    def test_fit_transform_consistency(self, X):
        """fit(X) then transform(X) must produce (k, T)."""
        r = FunctionalHarmonicsReducer(k=k, threshold=0.0)
        Z = r.fit(X=X).transform(X)
        assert Z.shape == (k, T)


# ── HarmonicAnalysis ──────────────────────────────────────────────────────────

class TestHarmonicAnalysis:

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            HarmonicAnalysis("not_a_reducer")

    def test_unfitted_reducer_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            HarmonicAnalysis(ConnectomeHarmonicsReducer(k=k))

    # ── project_rsn_vectors ───────────────────────────────────────────────────

    def test_project_rsn_shape(self, ch_analyser, RSN_matrix):
        alpha = ch_analyser.project_rsn_vectors(RSN_matrix)
        assert alpha.shape == (k, n_rsn)

    def test_project_rsn_sign_invariant_non_negative(self, ch_analyser, RSN_matrix):
        alpha = ch_analyser.project_rsn_vectors(RSN_matrix, sign_invariant=True)
        assert np.all(alpha >= 0)

    def test_project_rsn_sign_invariant_equals_abs(self, ch_analyser, RSN_matrix):
        alpha_signed = ch_analyser.project_rsn_vectors(RSN_matrix, sign_invariant=False)
        alpha_abs    = ch_analyser.project_rsn_vectors(RSN_matrix, sign_invariant=True)
        assert np.allclose(alpha_abs, np.abs(alpha_signed), atol=1e-4)

    # ── project_timeseries ────────────────────────────────────────────────────

    def test_project_timeseries_shape(self, ch_analyser, X):
        beta = ch_analyser.project_timeseries(X)
        assert beta.shape == (k, T)

    def test_project_timeseries_sign_invariant_non_negative(self, ch_analyser, X):
        beta = ch_analyser.project_timeseries(X, sign_invariant=True)
        assert np.all(beta >= 0)

    def test_project_timeseries_matches_transform(self, ch_reducer, ch_analyser, X):
        """
        project_timeseries should give the same result as reducer.transform()
        when no harmonic_idx is specified.
        Both use per-timepoint sign handling.
        """
        beta_analyser = ch_analyser.project_timeseries(X, sign_invariant=True)
        beta_reducer  = ch_reducer.transform(X, sign_invariant=True)
        assert np.allclose(beta_analyser, beta_reducer, atol=1e-4)

    def test_project_timeseries_with_selected_harmonics(self, ch_analyser, X):
        """Projecting onto a subset of harmonics gives (n_selected, T)."""
        selected = np.array([0, 2, 4])
        beta = ch_analyser.project_timeseries(X, harmonic_idx=selected)
        assert beta.shape == (len(selected), T)

    # ── select_harmonics_by_rsn ───────────────────────────────────────────────

    def test_select_harmonics_shape(self, ch_analyser, RSN_matrix):
        alpha   = ch_analyser.project_rsn_vectors(RSN_matrix)
        indices = ch_analyser.select_harmonics_by_rsn(alpha, n_select=3)
        assert indices.shape == (3,)
        assert len(np.unique(indices)) == 3   # no duplicates

    def test_select_harmonics_valid_indices(self, ch_analyser, RSN_matrix):
        alpha   = ch_analyser.project_rsn_vectors(RSN_matrix)
        indices = ch_analyser.select_harmonics_by_rsn(alpha, n_select=k)
        assert np.all(indices >= 0)
        assert np.all(indices < k)

    # ── reconstruction_error ──────────────────────────────────────────────────

    def test_reconstruction_error_keys(self, ch_analyser, X):
        result = ch_analyser.reconstruction_error(X)
        assert 'mse'       in result
        assert 'rmse'      in result
        assert 'pearson_r' in result
        assert 'X_hat'     in result

    def test_reconstruction_error_X_hat_shape(self, ch_analyser, X):
        result = ch_analyser.reconstruction_error(X)
        assert result['X_hat'].shape == (N, T)

    def test_reconstruction_error_mse_non_negative(self, ch_analyser, X):
        result = ch_analyser.reconstruction_error(X)
        assert result['mse'] >= 0

    def test_reconstruction_improves_with_k(self, SC, X):
        """More harmonics → lower reconstruction error."""
        errors = []
        for ki in [2, 5, 10, 15]:
            r = ConnectomeHarmonicsReducer(k=ki, threshold=0.0).fit(SC=SC)
            a = HarmonicAnalysis(r)
            errors.append(a.reconstruction_error(X)['mse'])
        assert all(errors[i] >= errors[i+1] for i in range(len(errors)-1)), \
            f"MSE did not decrease with more harmonics: {errors}"

    # ── mutual_information ────────────────────────────────────────────────────

    def test_mutual_information_shape(self, ch_analyser, RSN_matrix):
        """MI returns one value per harmonic — shape (k,)."""
        rsn_labels = np.argmax(RSN_matrix, axis=1)   # (N,) int labels
        mi         = ch_analyser.mutual_information(rsn_labels)
        assert mi.shape == (k,)

    def test_mutual_information_non_negative(self, ch_analyser, RSN_matrix):
        """MI values must be >= 0 by definition."""
        rsn_labels = np.argmax(RSN_matrix, axis=1)
        mi         = ch_analyser.mutual_information(rsn_labels)
        assert np.all(mi >= 0)

    def test_mutual_information_wrong_length_raises(self, ch_analyser):
        """rsn_labels must have N entries — wrong length should raise."""
        bad_labels = np.zeros(N + 5, dtype=int)
        with pytest.raises(ValueError, match="parcel"):
            ch_analyser.mutual_information(bad_labels)
