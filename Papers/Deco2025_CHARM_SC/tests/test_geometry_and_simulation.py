"""
CHARMsc/tests/test_geometry_and_simulation.py
----------------------------------------------
Tests for BaseCHARMGeometry, HARM, CHARM_SC, and BOLDGenerator.

All tests use synthetic parcel coordinates — no real HCP data needed.

Run with:  python -m pytest tests/ -v

Notes on test design
---------------------
- Exclusion tests are omitted: excluded parcels are filtered at the
  DataLoader level, so the geometry classes receive clean data.
- Simulation tests that check multi-fire and NaN fraction use a
  uniform_generator (P = 1/N everywhere) rather than the geometry-
  driven P. With N=20 synthetic parcels and epsilon=1400, the HARM/
  CHARM-SC kernels concentrate probability on 1-2 nearest neighbours,
  producing a near-single-walker walk that is geometrically correct
  but makes multi-fire statistically rare. The uniform_generator
  tests the Bernoulli firing mechanism independently of geometry.
"""

import numpy as np
import pytest
from scipy import stats

from geometry import HARM, CHARM_SC, BaseCHARMGeometry
from simulation.bold_generator import BOLDGenerator


# ── shared parameters ─────────────────────────────────────────────────────────

N   = 20
rng = np.random.default_rng(42)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def coords():
    """Synthetic parcel centroids, shape (N, 3), MNI-like range ±100 mm."""
    return rng.uniform(-100, 100, size=(N, 3)).astype(np.float64)


@pytest.fixture
def harm(coords):
    return HARM(epsilon=1400.0, t_horizon=2,
                diffusion_steps=10).fit(coords)


@pytest.fixture
def charm_sc(coords):
    return CHARM_SC(epsilon=1400.0, t_horizon=2,
                    diffusion_steps=10).fit(coords)


@pytest.fixture
def generator(harm):
    """Geometry-driven generator — used for shape/binary/always-active tests."""
    return BOLDGenerator(P=harm.diffusion_matrix, n_timesteps=50, random_state=0)


@pytest.fixture
def uniform_generator():
    """
    Generator with uniform P (each parcel equally likely, row sums = 1).

    Used for multi-fire and NaN-fraction tests. With uniform P, each
    active parcel fires each neighbour with probability 1/N independently,
    giving Binomial(N, 1/N) fires per step — expectation 1, variance
    (N-1)/N ≈ 1, reliably producing multi-fire timesteps.
    """
    P_uniform = np.full((N, N), 1.0 / N)
    return BOLDGenerator(P=P_uniform, n_timesteps=100, random_state=42)


# ── BaseCHARMGeometry ─────────────────────────────────────────────────────────

class TestBaseGeometry:

    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseCHARMGeometry()

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            HARM().stationary_distribution()

    def test_wrong_coords_shape_raises(self):
        with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
            HARM().fit(np.ones((N, 2)))

    def test_repr_not_fitted(self):
        assert "not fitted" in repr(HARM())

    def test_repr_fitted(self, harm):
        assert "fitted" in repr(harm)
        assert f"N={N}" in repr(harm)


# ── HARM ──────────────────────────────────────────────────────────────────────

class TestHARM:

    def test_diffusion_matrix_shape(self, harm):
        assert harm.diffusion_matrix.shape == (N, N)

    def test_diffusion_matrix_real(self, harm):
        assert harm.diffusion_matrix.dtype in (np.float32, np.float64)

    def test_diffusion_matrix_row_stochastic(self, harm):
        assert np.allclose(harm.diffusion_matrix.sum(axis=1), 1.0, atol=1e-6)

    def test_diffusion_matrix_non_negative(self, harm):
        assert np.all(harm.diffusion_matrix >= 0)

    def test_stationary_shape(self, coords):
        h = HARM(diffusion_steps=5).fit(coords)
        assert h.stationary_distribution().shape == (N,)

    def test_stationary_non_negative(self, harm):
        assert np.all(harm.stationary_distribution() >= -1e-10)

    def test_kernel_symmetry_Q(self, coords):
        """
        The HARM kernel K[i,j] = exp(-d²/σ) is symmetric (d²_ij = d²_ji),
        so Q = |K^t|² must be symmetric.

        Note: P = D⁻¹Q is NOT symmetric in general — row-normalisation
        by the diagonal degree matrix D breaks symmetry unless all row
        sums of Q are equal. We test Q (_Ptr_t) not P (diffusion_matrix).
        """
        h = HARM(t_horizon=2, diffusion_steps=5).fit(coords)
        assert np.allclose(h._Ptr_t, h._Ptr_t.T, atol=1e-8)


# ── CHARM-SC ──────────────────────────────────────────────────────────────────

class TestCHARMSC:

    def test_diffusion_matrix_shape(self, charm_sc):
        assert charm_sc.diffusion_matrix.shape == (N, N)

    def test_diffusion_matrix_real(self, charm_sc):
        """After |K^t|² and row-normalisation, P must be real."""
        assert np.isrealobj(charm_sc.diffusion_matrix)

    def test_diffusion_matrix_row_stochastic(self, charm_sc):
        assert np.allclose(charm_sc.diffusion_matrix.sum(axis=1), 1.0, atol=1e-6)

    def test_diffusion_matrix_non_negative(self, charm_sc):
        assert np.all(charm_sc.diffusion_matrix >= 0)

    def test_stationary_shape(self, charm_sc):
        assert charm_sc.stationary_distribution().shape == (N,)

    def test_stationary_non_negative(self, charm_sc):
        assert np.all(charm_sc.stationary_distribution() >= -1e-10)

    def test_different_from_harm(self, harm, charm_sc):
        """Complex and real kernels must produce different diffusion matrices."""
        assert not np.allclose(harm.diffusion_matrix,
                               charm_sc.diffusion_matrix, atol=1e-4)

    def test_t_horizon_affects_result(self, coords):
        c1 = CHARM_SC(t_horizon=1, diffusion_steps=5).fit(coords)
        c2 = CHARM_SC(t_horizon=2, diffusion_steps=5).fit(coords)
        assert not np.allclose(c1.diffusion_matrix, c2.diffusion_matrix)

    def test_epsilon_affects_result(self, coords):
        c1 = CHARM_SC(epsilon=500.0,  diffusion_steps=5).fit(coords)
        c2 = CHARM_SC(epsilon=1400.0, diffusion_steps=5).fit(coords)
        assert not np.allclose(c1.diffusion_matrix, c2.diffusion_matrix)


# ── BOLDGenerator ─────────────────────────────────────────────────────────────

class TestBOLDGenerator:

    def test_wrong_P_shape_raises(self):
        with pytest.raises(ValueError, match="square"):
            BOLDGenerator(P=np.ones((N, N + 1)))

    # ── loop version ──────────────────────────────────────────────────────────

    def test_loop_shape(self, generator):
        assert generator._run_single_trial_loop().shape == (N, 50)

    def test_loop_binary(self, generator):
        tssim = generator._run_single_trial_loop()
        assert np.all((tssim == 0) | (tssim == 1))

    def test_loop_always_active(self, generator):
        """At least one parcel must be active at every timestep."""
        tssim = generator._run_single_trial_loop()
        assert np.all(tssim.sum(axis=0) >= 1)

    def test_loop_multi_fire(self, uniform_generator):
        """
        The independent Bernoulli model must allow multiple parcels to fire
        simultaneously. Uses uniform_generator so multi-fire is reliable
        regardless of parcel geometry (see module docstring).
        """
        tssim = uniform_generator._run_single_trial_loop()
        assert tssim.sum(axis=0).max() > 1, \
            "All timesteps had exactly 1 active parcel — single-walker bug"

    # ── vectorised version ────────────────────────────────────────────────────

    def test_vectorised_shape(self, generator):
        assert generator._run_single_trial_vectorised().shape == (N, 50)

    def test_vectorised_binary(self, generator):
        tssim = generator._run_single_trial_vectorised()
        assert np.all((tssim == 0) | (tssim == 1))

    def test_vectorised_always_active(self, generator):
        tssim = generator._run_single_trial_vectorised()
        assert np.all(tssim.sum(axis=0) >= 1)

    def test_vectorised_multi_fire(self, uniform_generator):
        """Uses uniform_generator — see test_loop_multi_fire."""
        tssim = uniform_generator._run_single_trial_vectorised()
        assert tssim.sum(axis=0).max() > 1, \
            "All timesteps had exactly 1 active parcel — single-walker bug"

    # ── FC output ─────────────────────────────────────────────────────────────

    def test_simulate_trials_shape(self, generator):
        assert generator.simulate_trials(n_trials=3).shape == (N, N)

    def test_simulate_trials_symmetric(self, generator):
        FC = generator.simulate_trials(n_trials=5)
        assert np.allclose(FC, FC.T, atol=1e-5, equal_nan=True)

    def test_simulate_trials_diagonal_one(self, generator):
        """Valid (non-NaN) diagonal entries must be 1."""
        FC    = generator.simulate_trials(n_trials=5)
        valid = np.diag(FC)
        valid = valid[~np.isnan(valid)]
        assert np.allclose(valid, 1.0, atol=1e-5)

    def test_simulate_trials_low_nan_fraction(self, uniform_generator):
        """
        NaN fraction check using uniform P.

        With uniform P, T=100 and N=20, each parcel fires ~T/N = 5 times
        per trial on average — enough for Pearson correlation to be defined
        for most parcel pairs. NaN fraction should be well below 20%.
        """
        FC       = uniform_generator.simulate_trials(n_trials=10)
        nan_frac = np.isnan(FC).mean()
        assert nan_frac < 0.2, \
            f"NaN fraction {nan_frac:.1%} too high — check Bernoulli simulation"

    def test_simulate_fc_shape(self, generator):
        FC_reps = generator.simulate_fc(n_trials=3, n_repetitions=2)
        assert FC_reps.shape == (2, N, N)

    # ── loop vs vectorised consistency ────────────────────────────────────────

    def test_loop_and_vectorised_both_produce_finite_fc(self, uniform_generator):
        """
        Both simulators must produce FC matrices with a low NaN fraction
        and finite values — verifying the Bernoulli multi-fire model works
        in both implementations.

        We do NOT test that the two FC matrices are correlated with each
        other: they are independent stochastic realisations of the same
        process. With N=20 and 15 trials the sampling variance is large
        enough that two independent runs can easily anti-correlate by
        chance, making a correlation test unreliable at this scale.
        The meaningful check is that both produce valid (non-degenerate) FC.
        """
        FC_loop = uniform_generator.simulate_trials(n_trials=15,
                                                    use_vectorised=False)
        FC_vec  = uniform_generator.simulate_trials(n_trials=15,
                                                    use_vectorised=True)
        # Both must have low NaN fraction
        assert np.isnan(FC_loop).mean() < 0.2, "Loop FC has too many NaNs"
        assert np.isnan(FC_vec).mean()  < 0.2, "Vectorised FC has too many NaNs"
        # Both must contain at least some non-trivial (non-NaN, not all-zero)
        # off-diagonal entries
        i_lt, j_lt = np.tril_indices(N, k=-1)
        assert np.any(np.isfinite(FC_loop[i_lt, j_lt])), "Loop FC is all NaN"
        assert np.any(np.isfinite(FC_vec[i_lt,  j_lt])), "Vectorised FC is all NaN"
