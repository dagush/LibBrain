"""
tests/test_charm_analysis.py
-----------------------------
Tests for CHARMAnalysis and its sub-analyses.

All tests use synthetic data: a tiny CHARMReducer (N=15, Tm=60, k=4)
fit on random BOLD so that the full analysis pipeline can be exercised
without real HCP data.

Parameters chosen so that:
    n_subjects=3, t_per_subject=20 → Tm=60 per condition group
    Two conditions → reducer fit on Tm=120 total timepoints

Run with:  python -m pytest tests/test_charm_analysis.py -v
"""

import numpy as np
import pytest

from Neuroreduce import CHARMReducer
from Neuroreduce.utils.charm_analysis import (
    CHARMAnalysis,
    SubjectIndex,
    GroupAnalysisResult,
    ClassificationResult,
)

# ── shared test parameters ────────────────────────────────────────────────────

N            = 15    # parcels
T_PER_SUB    = 20    # timepoints per subject
N_SUB        = 3     # subjects per condition
K            = 4     # latent dimensions
TAU          = 2     # lag (must be < T_PER_SUB)
TM           = T_PER_SUB * N_SUB * 2   # two conditions concatenated

rng = np.random.default_rng(42)


@pytest.fixture(scope="module")
def reducer():
    """Fit a CHARMReducer on synthetic two-condition concatenated BOLD."""
    X = rng.standard_normal((N, TM)).astype(np.float32)
    r = CHARMReducer(k=K, epsilon=50.0, t_horizon=2, sort_eigenvectors=True)
    r.fit(X)
    return r


@pytest.fixture(scope="module")
def analysis(reducer):
    """Build a CHARMAnalysis object on the fitted reducer."""
    return CHARMAnalysis(
        reducer       = reducer,
        t_per_subject = T_PER_SUB,
        n_subjects    = N_SUB,
        tau           = TAU,
    )


@pytest.fixture(scope="module")
def rest_result(analysis):
    return analysis.analyze_group(group_offset=0)


@pytest.fixture(scope="module")
def task_result(analysis):
    return analysis.analyze_group(group_offset=N_SUB * T_PER_SUB)


# ── SubjectIndex ─────────────────────────────────────────────────────────────

class TestSubjectIndex:
    def test_start_zero(self):
        idx = SubjectIndex(n_subjects=3, t_per_subject=20, group_offset=0)
        assert idx.start(0) == 0

    def test_start_offset(self):
        idx = SubjectIndex(n_subjects=3, t_per_subject=20, group_offset=60)
        assert idx.start(0) == 60

    def test_end(self):
        idx = SubjectIndex(n_subjects=3, t_per_subject=20, group_offset=0)
        assert idx.end(0) == 20
        assert idx.end(1) == 40
        assert idx.end(2) == 60

    def test_slice(self):
        idx = SubjectIndex(n_subjects=3, t_per_subject=20, group_offset=0)
        s = idx.slice(1)
        assert s.start == 20 and s.stop == 40


# ── CHARMAnalysis construction ────────────────────────────────────────────────

class TestCHARMAnalysisConstruction:
    def test_wrong_type_raises(self, reducer):
        with pytest.raises(TypeError):
            CHARMAnalysis("not_a_reducer", t_per_subject=20, n_subjects=3)

    def test_unfitted_reducer_raises(self):
        r = CHARMReducer(k=K)
        with pytest.raises(RuntimeError, match="not fitted"):
            CHARMAnalysis(r, t_per_subject=T_PER_SUB, n_subjects=N_SUB)

    def test_too_large_layout_raises(self, reducer):
        """If declared layout exceeds Phi rows, should raise ValueError."""
        with pytest.raises(ValueError, match="timepoints"):
            CHARMAnalysis(reducer, t_per_subject=T_PER_SUB, n_subjects=999)

    def test_single_condition_warning(self, reducer):
        """One condition only (half the data) should emit a UserWarning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CHARMAnalysis(
                reducer,
                t_per_subject = TM,       # all timepoints = one "subject"
                n_subjects    = 1,
            )
            # Should have warned about single condition
            assert any("ONE condition" in str(x.message) for x in w)


# ── analyze_group output shapes ───────────────────────────────────────────────

class TestAnalyzeGroupShapes:
    def test_fc_sub_shape(self, rest_result):
        assert rest_result.fc_sub.shape == (N_SUB, K, K)

    def test_metastability_shape(self, rest_result):
        assert rest_result.metastability.shape == (N_SUB,)

    def test_hierarchical_levels_shape(self, rest_result):
        assert rest_result.hierarchical_levels.shape == (N_SUB, K)

    def test_trophic_coherence_shape(self, rest_result):
        assert rest_result.trophic_coherence.shape == (N_SUB,)

    def test_patterns_length(self, rest_result):
        assert len(rest_result.patterns) == N_SUB

    def test_patterns_element_shape(self, rest_result):
        n_edges = K * (K - 1) // 2
        for p in rest_result.patterns:
            assert p.shape == (1, n_edges), f"Expected (1, {n_edges}), got {p.shape}"


# ── metastability ─────────────────────────────────────────────────────────────

class TestMetastability:
    def test_is_finite(self, rest_result):
        """Metastability should be a finite real number for synthetic data."""
        assert np.all(np.isfinite(rest_result.metastability))

    def test_is_scalar_per_subject(self, rest_result):
        assert rest_result.metastability.ndim == 1
        assert len(rest_result.metastability) == N_SUB


# ── lagged FC ─────────────────────────────────────────────────────────────────

class TestLaggedFC:
    def test_values_in_range(self, rest_result):
        """Pearson correlations must lie in [-1, 1]."""
        fc = rest_result.fc_sub
        assert np.all(fc >= -1.0 - 1e-6)
        assert np.all(fc <= +1.0 + 1e-6)

    def test_asymmetric(self, rest_result):
        """
        Lagged FC should generally be asymmetric (FC[i,j] ≠ FC[j,i]).
        This is the key property exploited by trophic coherence.
        For random data this will virtually always be true.
        """
        for s in range(N_SUB):
            fc = rest_result.fc_sub[s]
            assert not np.allclose(fc, fc.T), \
                f"Subject {s} lagged FC is symmetric — check tau or data."


# ── trophic coherence ─────────────────────────────────────────────────────────

class TestTrophicCoherence:
    def test_coherence_in_range(self, rest_result):
        """
        Trophic coherence Q = 1 - F0 is theoretically in (-∞, 1].
        For realistic data it is usually in [0, 1].
        We only test finiteness here since synthetic data may be unusual.
        """
        tc = rest_result.trophic_coherence
        finite_mask = np.isfinite(tc)
        # At least some subjects should have finite coherence
        assert np.any(finite_mask), "All trophic coherence values are NaN"

    def test_hierarchical_levels_non_negative(self, rest_result):
        """After shifting by min, all levels should be >= 0."""
        hl = rest_result.hierarchical_levels
        finite = hl[np.isfinite(hl)]
        assert np.all(finite >= -1e-9), \
            "Trophic levels contain negative values after min-shift"


# ── classification ────────────────────────────────────────────────────────────

class TestClassification:
    def test_output_type(self, analysis, rest_result, task_result):
        result = analysis.classification(
            patterns_rest = rest_result.patterns,
            patterns_task = task_result.patterns,
            n_train       = N_SUB - 1,
            k_fold        = 5,             # very small for speed
            random_state  = 0,
        )
        assert isinstance(result, ClassificationResult)

    def test_confusion_matrix_shape(self, analysis, rest_result, task_result):
        result = analysis.classification(
            rest_result.patterns, task_result.patterns,
            n_train=N_SUB - 1, k_fold=5, random_state=0,
        )
        assert result.confusion_matrix.shape == (2, 2)

    def test_accuracy_in_range(self, analysis, rest_result, task_result):
        result = analysis.classification(
            rest_result.patterns, task_result.patterns,
            n_train=N_SUB - 1, k_fold=5, random_state=0,
        )
        assert 0.0 <= result.accuracy <= 1.0

    def test_per_fold_accuracy_length(self, analysis, rest_result, task_result):
        k_fold = 7
        result = analysis.classification(
            rest_result.patterns, task_result.patterns,
            n_train=N_SUB - 1, k_fold=k_fold, random_state=0,
        )
        assert len(result.per_fold_accuracy) == k_fold

    def test_confusion_rows_sum_to_one(self, analysis, rest_result, task_result):
        """Each row of the averaged confusion matrix should sum to ≈ 1."""
        result = analysis.classification(
            rest_result.patterns, task_result.patterns,
            n_train=N_SUB - 1, k_fold=10, random_state=0,
        )
        row_sums = result.confusion_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"Confusion matrix rows do not sum to 1: {row_sums}"


# ── calc_pfctau ───────────────────────────────────────────────────────────────

class TestCalcPfctau:
    def test_output_shapes(self, analysis, rest_result, task_result):
        nsig, pfctau = analysis.calc_pfctau(
            rest_result.fc_sub,
            task_result.fc_sub,
            n_permutations=99,   # minimal for speed
        )
        assert nsig.shape   == (K * K,)
        assert pfctau.shape == (K * K,)

    def test_pvalues_in_range(self, analysis, rest_result, task_result):
        _, pfctau = analysis.calc_pfctau(
            rest_result.fc_sub,
            task_result.fc_sub,
            n_permutations=99,
        )
        assert np.all(pfctau >= 0.0)
        assert np.all(pfctau <= 1.0)

    def test_nsig_is_boolean(self, analysis, rest_result, task_result):
        nsig, _ = analysis.calc_pfctau(
            rest_result.fc_sub,
            task_result.fc_sub,
            n_permutations=99,
        )
        assert nsig.dtype == bool
