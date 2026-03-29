"""
Neuroreduce/utils/charm_analysis.py
-------------------------------------
Post-hoc analyses that consume the output of a fitted CHARMReducer.

Implements the analysis pipeline from:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

Original MATLAB code by Gustavo Deco.
Python translation and Neuroreduce integration by Gustavo Patow.

Design notes
------------
- This module operates on the TIMEPOINT-SPACE embedding Φ (shape Tm×k),
  accessed via reducer.embedding_, NOT on the parcel-space basis conet.
- All analyses assume that Φ was computed on CONCATENATED timeseries
  from multiple subjects and (optionally) multiple conditions. The caller
  is responsible for tracking which rows of Φ belong to which subject
  and condition — this is done via SubjectIndex objects (see below).
- Functions are grouped into a CHARMAnalysis class that holds shared
  state (reducer reference, subject layout), plus a GroupAnalysisResult
  dataclass that bundles per-group outputs cleanly.

Dependencies
------------
    numpy, scipy, sklearn, statsmodels
    statsmodels is used for Benjamini-Hochberg FDR correction.
    Install with:  pip install statsmodels
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy import linalg as LA
from scipy import stats
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from statsmodels.stats.multitest import multipletests

from Neuroreduce.methods.charm import CHARMReducer
# ECM computation delegates to the NeuroNumba ECM observable via these wrappers
from Neuroreduce.utils.ecm import compute_ecm, compute_ecm_per_subject


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class SubjectIndex:
    """
    Tracks which rows of the concatenated Φ matrix belong to each subject
    within a single condition group.

    In the original paper, all subjects from all conditions are concatenated
    into a single Tm×k matrix. To recover per-subject analyses we need to
    know where each subject starts and ends.

    Parameters
    ----------
    n_subjects : int
        Number of subjects in this group.
    t_per_subject : int
        Number of timepoints per subject (must be identical for all subjects
        in the group — the original code assumes this).
    group_offset : int
        Row index in Φ where this condition group starts.
        - REST group : group_offset = 0
        - TASK group : group_offset = n_subjects * t_per_subject

    Attributes
    ----------
    start(sub) : int   — first Φ row for subject sub (0-indexed)
    end(sub)   : int   — one-past-last Φ row for subject sub
    """
    n_subjects:    int
    t_per_subject: int
    group_offset:  int = 0

    def start(self, sub: int) -> int:
        """First row of Φ belonging to subject sub in this group."""
        return self.group_offset + sub * self.t_per_subject

    def end(self, sub: int) -> int:
        """One-past-last row of Φ belonging to subject sub in this group."""
        return self.group_offset + (sub + 1) * self.t_per_subject

    def slice(self, sub: int) -> slice:
        """Convenience: return a slice object for subject sub."""
        return slice(self.start(sub), self.end(sub))


@dataclass
class GroupAnalysisResult:
    """
    Bundles all per-group outputs from CHARMAnalysis.analyze_group().

    Attributes
    ----------
    fc_sub : np.ndarray, shape (n_subjects, k, k)
        Per-subject time-lagged FC matrix in latent space.
        fc_sub[s, i, j] = corr(Φ_s[:-τ, i], Φ_s[τ:, j])
        Asymmetric by construction (τ > 0 breaks time symmetry).

    metastability : np.ndarray, shape (n_subjects,)
        Per-subject metastability index: differential entropy of the
        FCD matrix variance. Higher values → more dynamic, less stable.
        Eq. from paper: H = 0.5 * log(2πe * Var(FCD)) ≈ 0.5*log(2π*Var) + 0.5

    hierarchical_levels : np.ndarray, shape (n_subjects, k)
        Per-subject trophic level γ of each latent dimension in the
        directed FC graph. Reflects the hierarchical position of each
        mode in the latent-space functional hierarchy.

    trophic_coherence : np.ndarray, shape (n_subjects,)
        Per-subject trophic coherence Q ∈ [0, 1]. Q ≈ 1 means edges
        connect adjacent trophic levels (coherent hierarchy);
        Q ≈ 0 means random mixing across levels.

    patterns : list of np.ndarray
        Per-subject FC fingerprints (lower-triangle of fc_sub[s]),
        used as features for classification. Length = n_subjects.
        Each element has shape (n_patterns, n_edges) where n_edges =
        k*(k-1)//2.
    """
    fc_sub:              np.ndarray          # (n_subjects, k, k)
    metastability:       np.ndarray          # (n_subjects,)
    hierarchical_levels: np.ndarray          # (n_subjects, k)
    trophic_coherence:   np.ndarray          # (n_subjects,)
    patterns:            list[np.ndarray]    # list of (n_patterns, n_edges)


@dataclass
class ClassificationResult:
    """
    Output of CHARMAnalysis.classification().

    Attributes
    ----------
    confusion_matrix : np.ndarray, shape (2, 2)
        Average confusion matrix across k-fold runs, normalised by
        the number of validation samples per class.
        confusion_matrix[true_class, predicted_class]

    accuracy : float
        Mean balanced accuracy across folds:
        acc = sum(diag(confusion_matrix)) / 2

    per_fold_accuracy : np.ndarray, shape (k_fold,)
        Balanced accuracy for each individual fold. Useful for
        computing confidence intervals or permutation tests.
    """
    confusion_matrix:   np.ndarray   # (2, 2)
    accuracy:           float
    per_fold_accuracy:  np.ndarray   # (k_fold,)


# =============================================================================
# Main analysis class
# =============================================================================

class CHARMAnalysis:
    """
    Post-hoc analyses on the CHARM latent-space embedding.

    Takes a fitted CHARMReducer and the subject layout, then exposes
    methods for metastability, lagged FC, trophic coherence, FDR-corrected
    condition comparison, and SVM classification.

    Parameters
    ----------
    reducer : CHARMReducer
        A fitted CHARMReducer. reducer.embedding_ (Tm×k) is the primary
        input to all analyses.
    t_per_subject : int
        Number of timepoints per subject after preprocessing (Tmsub in the
        original code: Tmax + 1 - 2*CUT).
    n_subjects : int
        Number of subjects per condition group.
    tau : int
        Time lag (in timepoints) used for the lagged FC computation.
        Default: 3, matching the original paper.

    Notes
    -----
    The concatenated Φ matrix is assumed to have the layout:
        rows 0            .. n_subjects*t_per_subject - 1  → condition A (REST)
        rows n_subjects*t_per_subject .. 2*n_subjects*t_per_subject - 1 → condition B (TASK)
    This matches filterAndConcatSubj() called twice in the original run().
    """

    def __init__(
        self,
        reducer:       CHARMReducer,
        t_per_subject: int,
        n_subjects:    int,
        tau:           int = 3,
    ):
        # -- validate inputs --------------------------------------------------
        if not isinstance(reducer, CHARMReducer):
            raise TypeError(
                f"Expected a CHARMReducer, got {type(reducer).__name__}."
            )
        reducer._check_is_fitted()

        self._reducer       = reducer
        self._Phi           = reducer.embedding_   # (Tm, k) — timepoint embedding
        self._k             = reducer.k
        self.t_per_subject  = t_per_subject
        self.n_subjects     = n_subjects
        self.tau            = tau

        # Validate that Phi is large enough for the declared layout
        Tm_expected = t_per_subject * n_subjects
        Tm_actual   = self._Phi.shape[0]
        if Tm_actual < Tm_expected:
            raise ValueError(
                f"Phi has {Tm_actual} timepoints but n_subjects={n_subjects} × "
                f"t_per_subject={t_per_subject} = {Tm_expected} were declared. "
                "Check that the reducer was fit on the full concatenated data."
            )
        if Tm_actual < 2 * Tm_expected:
            warnings.warn(
                f"Phi has {Tm_actual} timepoints, enough for only ONE condition "
                f"group ({Tm_expected} timepoints). If you have two conditions, "
                "make sure the reducer was fit on both groups concatenated.",
                UserWarning,
                stacklevel=2,
            )

    # -------------------------------------------------------------------------
    # Public: build subject index for a condition group
    # -------------------------------------------------------------------------

    def subject_index(self, group_offset: int = 0) -> SubjectIndex:
        """
        Build a SubjectIndex for a condition group starting at group_offset.

        Parameters
        ----------
        group_offset : int
            Row in Φ where this group starts.
            - First group (e.g. REST) : group_offset = 0
            - Second group (e.g. TASK): group_offset = n_subjects * t_per_subject

        Returns
        -------
        SubjectIndex
        """
        return SubjectIndex(
            n_subjects    = self.n_subjects,
            t_per_subject = self.t_per_subject,
            group_offset  = group_offset,
        )

    # -------------------------------------------------------------------------
    # Public: full per-group analysis (wraps the three sub-analyses)
    # -------------------------------------------------------------------------

    def analyze_group(
        self,
        group_offset: int = 0,
    ) -> GroupAnalysisResult:
        """
        Run all per-subject analyses for one condition group.

        Corresponds to the ``analyze(Phi, offset)`` function in the original
        code, but split into clearly named sub-methods and returned as a
        structured result object rather than a bare tuple.

        Parameters
        ----------
        group_offset : int
            Row in Φ where this condition group starts.

        Returns
        -------
        GroupAnalysisResult
        """
        idx = self.subject_index(group_offset)

        # Pre-allocate outputs
        fc_sub              = np.zeros((self.n_subjects, self._k, self._k))
        metastability       = np.zeros(self.n_subjects)
        hierarchical_levels = np.zeros((self.n_subjects, self._k))
        trophic_coherence   = np.zeros(self.n_subjects)
        patterns            = []

        for sub in range(self.n_subjects):
            # Extract this subject's rows from Φ, then z-score across time.
            # Assumption: z-scoring is per-latent-dimension (axis=0), matching
            # scipy.stats.zscore default and the original code's usage.
            Phi_sub = stats.zscore(self._Phi[idx.slice(sub), :])  # (t_per_subject, k)

            # -- metastability ------------------------------------------------
            metastability[sub] = self._metastability(Phi_sub)

            # -- lagged FC ----------------------------------------------------
            fc_sub[sub] = self._lagged_fc(Phi_sub)

            # -- trophic coherence + hierarchical levels -----------------------
            hier, troph = self._trophic_analysis(fc_sub[sub])
            hierarchical_levels[sub] = hier
            trophic_coherence[sub]   = troph

            # -- FC fingerprint (for classification) --------------------------
            patterns.append(self._fc_fingerprint(Phi_sub))

        return GroupAnalysisResult(
            fc_sub              = fc_sub,
            metastability       = metastability,
            hierarchical_levels = hierarchical_levels,
            trophic_coherence   = trophic_coherence,
            patterns            = patterns,
        )

    # -------------------------------------------------------------------------
    # Public: condition comparison with FDR correction
    # -------------------------------------------------------------------------

    def calc_pfctau(
        self,
        fc_rest: np.ndarray,
        fc_task: np.ndarray,
        n_permutations: int = 10_000,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compare lagged FC matrices between two conditions using permutation
        tests with Benjamini-Hochberg FDR correction.

        Corresponds to ``calc_pfctau()`` in the original code, completed with:
        - explicit fc_task parameter (was an undeclared global)
        - scipy permutation test replacing the original MATLAB call to
          ``permutation_htest2_np``
        - statsmodels Benjamini-Hochberg replacing ``FDR_benjHoch``

        Parameters
        ----------
        fc_rest : np.ndarray, shape (n_subjects, k, k)
            Per-subject lagged FC for condition A (e.g. REST).
            Typically GroupAnalysisResult.fc_sub from analyze_group(offset=0).
        fc_task : np.ndarray, shape (n_subjects, k, k)
            Per-subject lagged FC for condition B (e.g. TASK).
        n_permutations : int
            Number of permutations for the permutation t-test. Default: 10000.
        alpha : float
            FDR threshold for Benjamini-Hochberg correction. Default: 0.05.

        Returns
        -------
        nsig : np.ndarray of bool, shape (k*k,)
            Boolean mask of FC matrix entries that survive FDR correction.
            Reshape to (k, k) to identify which latent dimension pairs differ
            significantly between conditions.
        pfctau : np.ndarray of float, shape (k*k,)
            Raw p-values for each (i, j) entry of the FC matrix, before FDR.

        Notes
        -----
        Assumption: two-sample permutation test with t-statistic, matching
        the original MATLAB ``permutation_htest2_np(..., 'ttest')``.
        The test is two-tailed.
        """
        k = self._k
        pfctau = np.ones(k * k)   # initialise to 1 (no evidence against H0)

        idx = 0
        for i in range(k):
            for j in range(k):
                # Extract the (i,j) FC entry across all subjects
                # Assumption: same indexing as original — fc_sub[:, j, i]
                # (column-major order inherited from MATLAB)
                a = fc_rest[:, j, i]   # (n_subjects,)  REST
                b = fc_task[:, j, i]   # (n_subjects,)  TASK

                # Two-sample permutation t-test
                # scipy.stats.permutation_test pools [a, b] and permutes labels
                result = stats.permutation_test(
                    (a, b),
                    statistic   = lambda x, y: np.mean(x) - np.mean(y),
                    permutation_type = 'samples',
                    n_resamples = n_permutations,
                    alternative = 'two-sided',
                    random_state = 42,    # reproducibility
                )
                pfctau[idx] = result.pvalue
                idx += 1

        # Benjamini-Hochberg FDR correction across all k*k comparisons.
        # multipletests returns: (reject, p_corrected, alpha_sidak, alpha_bf)
        # We only need the boolean reject mask.
        reject, _, _, _ = multipletests(pfctau, alpha=alpha, method='fdr_bh')
        nsig = reject   # bool array, shape (k*k,)

        return nsig, pfctau

    # -------------------------------------------------------------------------
    # Public: SVM classification (rest vs task)
    # -------------------------------------------------------------------------

    def classification(
        self,
        patterns_rest: list[np.ndarray],
        patterns_task: list[np.ndarray],
        n_train:       int  = 90,
        k_fold:        int  = 1000,
        random_state:  Optional[int] = None,
    ) -> ClassificationResult:
        """
        k-fold SVM classification distinguishing two conditions (e.g. REST vs
        TASK) using the lower-triangle FC fingerprints as features.

        Completes the ``classification()`` function from the original code,
        translating MATLAB's ``fitcecoc`` + ``templateSVM('KernelFunction','rbf')``
        to sklearn's SVC(kernel='rbf'). For a 2-class problem these are
        equivalent.

        Parameters
        ----------
        patterns_rest : list of np.ndarray, length n_subjects
            FC fingerprints for condition A. Each element has shape
            (n_patterns, n_edges). Typically GroupAnalysisResult.patterns
            from analyze_group(offset=0).
        patterns_task : list of np.ndarray, length n_subjects
            FC fingerprints for condition B.
        n_train : int
            Number of subjects used for training in each fold. The remaining
            n_subjects - n_train subjects form the validation set.
            Default: 90 (matching the original MATLAB code's NTRAIN=90).
        k_fold : int
            Number of random train/validation splits. Default: 1000.
        random_state : int or None
            Seed for the random shuffling. None → non-reproducible.

        Returns
        -------
        ClassificationResult

        Notes
        -----
        Assumption: balanced accuracy is used (average of per-class accuracies),
        matching the original:  acc = sum(diag(conf)) / 2
        This is robust to class imbalance in the validation set.

        Assumption: the SVM is trained on ALL patterns from the training
        subjects stacked vertically (multiple patterns per subject if
        n_patterns > 1), but the confusion matrix is accumulated per
        validation SUBJECT, not per pattern. This matches the MATLAB code.
        """
        n_sub  = self.n_subjects
        n_val  = n_sub - n_train
        rng    = np.random.default_rng(random_state)

        # Accumulate confusion matrix and per-fold accuracy
        confusion   = np.zeros((2, 2))
        fold_acc    = np.zeros(k_fold)

        for fold in range(k_fold):
            shuffling = rng.permutation(n_sub)   # random subject order
            train_idx = shuffling[:n_train]
            val_idx   = shuffling[n_train:]

            # ── build training set ────────────────────────────────────────
            # Stack patterns from all training subjects for each condition.
            # Label 0 = REST, label 1 = TASK.
            # Assumption: np.vstack matches MATLAB's vertcat.
            TrainData1 = np.vstack([patterns_rest[s] for s in train_idx])
            TrainData2 = np.vstack([patterns_task[s] for s in train_idx])
            TrainData  = np.vstack([TrainData1, TrainData2])
            Labels     = np.concatenate([
                np.zeros(len(TrainData1)),   # REST = class 0
                np.ones(len(TrainData2)),    # TASK = class 1
            ])

            # ── train SVM ────────────────────────────────────────────────
            # SVC(kernel='rbf') ≡ MATLAB fitcecoc + templateSVM('rbf')
            # for 2-class problems. C=1.0 is sklearn default.
            # Assumption: no explicit C or gamma tuning, matching the
            # original code which uses MATLAB defaults.
            clf = SVC(kernel='rbf')
            clf.fit(TrainData, Labels)

            # ── validate (per-subject, not per-pattern) ───────────────────
            # Assumption: each validation subject contributes ONE vote,
            # regardless of n_patterns. We take the majority prediction
            # across all their patterns (mode vote).
            con = np.zeros((2, 2))
            for s in val_idx:
                # REST subject
                preds_rest = clf.predict(patterns_rest[s])
                pred_rest  = int(np.round(np.mean(preds_rest)))  # majority vote
                con[0, pred_rest] += 1

                # TASK subject
                preds_task = clf.predict(patterns_task[s])
                pred_task  = int(np.round(np.mean(preds_task)))
                con[1, pred_task] += 1

            # Normalise each row by the number of validation subjects
            # (one per class → each row sums to 1 when n_val is equal per class)
            con[0] /= n_val
            con[1] /= n_val

            confusion        += con
            fold_acc[fold]    = np.sum(np.diag(con)) / 2   # balanced accuracy

        # Average confusion matrix across all folds
        confusion /= k_fold
        accuracy   = float(np.sum(np.diag(confusion)) / 2)

        return ClassificationResult(
            confusion_matrix  = confusion,
            accuracy          = accuracy,
            per_fold_accuracy = fold_acc,
        )

    # =========================================================================
    # Private sub-analyses (one per conceptual quantity)
    # =========================================================================

    def _metastability(self, Phi_sub: np.ndarray) -> float:
        """
        Compute ECM for one subject's latent embedding.

        Delegates to ``Neuroreduce.utils.ecm.compute_ecm()``, which is the
        shared implementation used by both CHARMAnalysis and PCAAnalysis.

        Parameters
        ----------
        Phi_sub : np.ndarray, shape (t_per_subject, k)
            z-scored latent embedding for one subject.

        Returns
        -------
        float
            ECM value H.
        """
        # Signal is already z-scored by analyze_group() before this call
        return compute_ecm(Phi_sub)

    def compute_source_ecm(
        self,
        X:            np.ndarray,
        group_offset: int = 0,
    ) -> np.ndarray:
        """
        Compute ECM in SOURCE space (full BOLD) for each subject in a group.

        This is the counterpart to the manifold-space metastability already
        computed in analyze_group(). Both are needed to compute the
        per-subject source/manifold ECM correlation used in Figure 2(b)
        of the paper.

        Parameters
        ----------
        X : np.ndarray, shape (N, Tm)
            Full concatenated BOLD signal — the same array passed to
            reducer.fit(). NOT the reduced embedding.
        group_offset : int
            Starting column in X for this condition group. Default: 0.

        Returns
        -------
        ecm_source : np.ndarray, shape (n_subjects,)
            Per-subject ECM computed in the N-dimensional source space.
        """
        # X has shape (N, Tm) — our standard convention.
        # compute_ecm_per_subject handles the (N, Tm) → (Tm, N) transpose.
        return compute_ecm_per_subject(
            signal        = X,
            n_subjects    = self.n_subjects,
            t_per_subject = self.t_per_subject,
            group_offset  = group_offset,
        )

    def _lagged_fc(self, Phi_sub: np.ndarray) -> np.ndarray:
        """
        Compute the time-lagged functional connectivity matrix in latent space.

        FC[i, j] = Pearson correlation between Φ[:,i] at time t and Φ[:,j]
                   at time t + τ, over all valid timepoints.

        The lag τ (self.tau) breaks time-reversal symmetry, making FC
        asymmetric: FC[i,j] ≠ FC[j,i] in general. This asymmetry is
        exploited by the trophic coherence analysis to define a directed
        functional hierarchy.

        Assumption: uses np.corrcoef on the full valid window [0:-τ] vs [τ:],
        matching tricks.corr() in the original code (Pearson, ddof=1).

        Parameters
        ----------
        Phi_sub : np.ndarray, shape (t_per_subject, k)
            z-scored latent embedding for one subject.

        Returns
        -------
        fc : np.ndarray, shape (k, k)
            Asymmetric lagged FC matrix.
        """
        # Assumption: tau < t_per_subject (checked implicitly by array slicing)
        A = Phi_sub[:-self.tau, :]   # timepoints 0 .. T-τ-1  (k columns)
        B = Phi_sub[self.tau:,  :]   # timepoints τ .. T-1    (k columns)

        # np.corrcoef stacks rows → shape (2k, 2k); we want the (k×k) cross-block
        # corrcoef([A.T, B.T])[0:k, k:2k] = corr(A_cols, B_cols)
        combined = np.corrcoef(A.T, B.T)   # (2k, 2k)
        fc = combined[:self._k, self._k:]  # (k, k)  upper-right block
        return fc

    def _trophic_analysis(
        self, fc: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Compute trophic levels and trophic coherence of the directed FC graph.

        Theory (Johnson et al., 2014; used in Deco et al. 2025):
        Given a directed weighted graph with adjacency A (non-negative,
        no self-loops), define:
            - in-degree  d_i  = sum_j A_{ji}
            - out-degree δ_i  = sum_j A_{ij}
            - u_i = d_i + δ_i   (total degree)
            - v_i = d_i - δ_i   (degree imbalance)

        Trophic levels γ solve the linear system:
            Λ γ = v,   where Λ_{ij} = u_i δ_{ij} - A_{ij} - A_{ji}
        with Λ[0,0] = 0 to fix the gauge (one level pinned to 0).

        Trophic coherence:
            F0 = sum_{ij} A_{ij} (γ_j - γ_i - 1)² / sum_{ij} A_{ij}
            Q  = 1 - F0   ∈ [0, 1]
        Q = 1: perfectly coherent (all edges span exactly one trophic level)
        Q = 0: incoherent (random mixing)

        Parameters
        ----------
        fc : np.ndarray, shape (k, k)
            Lagged FC matrix for one subject (possibly asymmetric).

        Returns
        -------
        gamma : np.ndarray, shape (k,)
            Trophic levels, shifted so min(gamma) = 0.
        trophic_coherence : float
            Q = 1 - F0.

        Notes
        -----
        Assumption: negative FC values are thresholded to 0 before computing
        the graph, and self-loops are removed. This matches the original code:
            A[A < 0] = 0
            A = A - diag(diag(A))
        If the Laplacian Λ is singular (degenerate graph), gamma is set to NaN
        and trophic coherence to NaN. This is rare but can occur for small k.
        """
        k = self._k

        # Threshold: keep only positive FC edges, remove self-loops
        # Assumption: negative lagged FC is not interpreted as inhibition here;
        # it is simply discarded for the graph-theoretic analysis.
        A = np.copy(fc)
        A[A < 0] = 0.0
        A -= np.diag(np.diag(A))   # zero out diagonal

        # Degree vectors
        d     = np.sum(A, axis=0)   # in-degree  (column sums)
        delta = np.sum(A, axis=1)   # out-degree (row sums)
        u     = d + delta           # total degree
        v     = d - delta           # degree imbalance

        # Graph Laplacian for trophic level system
        # Λ = diag(u) - A - A.T
        Lambda_ = np.diag(u) - A - A.T

        # Fix gauge: pin first node's level by zeroing its equation.
        # This makes Λ invertible when the graph is connected.
        # Assumption: node 0 is used as the reference, matching original code.
        Lambda_[0, 0] = 0.0

        # Solve Λ γ = v for trophic levels
        if self._is_invertible(Lambda_):
            gamma = LA.solve(Lambda_, v)
        else:
            warnings.warn(
                "Trophic level Laplacian is singular — graph may be disconnected "
                "or have isolated nodes. Setting gamma and trophic coherence to NaN.",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.full(k, np.nan), np.nan

        # Shift so minimum level = 0
        gamma = gamma - np.min(gamma)

        # Trophic coherence: F0 = sum_ij A_ij (γ_j - γ_i - 1)² / sum_ij A_ij
        # Build (γ_j - γ_i) matrix using broadcasting
        # meshgrid_gamma[i,j] = gamma[j]  (matches MATLAB meshgrid 'xy' indexing)
        gamma_j = gamma[np.newaxis, :]   # (1, k) — broadcast as columns
        gamma_i = gamma[:, np.newaxis]   # (k, 1) — broadcast as rows
        H  = (gamma_j - gamma_i - 1) ** 2   # (k, k)

        sum_A = np.sum(A)
        if sum_A == 0:
            # Fully disconnected graph — coherence undefined
            warnings.warn(
                "All FC edges are zero after thresholding. "
                "Trophic coherence set to NaN.",
                RuntimeWarning,
                stacklevel=3,
            )
            return gamma, np.nan

        F0                = np.sum(A * H) / sum_A
        trophic_coherence = float(1.0 - F0)

        return gamma, trophic_coherence

    def _fc_fingerprint(self, Phi_sub: np.ndarray) -> np.ndarray:
        """
        Compute the FC fingerprint of one subject for use in classification.

        The fingerprint is the lower triangle of the Pearson FC matrix
        computed over the full session (no lag, no windowing in FULLWIN mode).
        This matches the MATLAB code's ``corrcoef(Phi(..., :))`` with FULLWIN=1.

        Assumption: FULLWIN=1 (one pattern per subject, full session window).
        Windowed mode (FULLWIN=0) is not implemented here; the result shape
        would differ and is left for a future extension.

        Parameters
        ----------
        Phi_sub : np.ndarray, shape (t_per_subject, k)
            z-scored latent embedding for one subject.

        Returns
        -------
        patterns : np.ndarray, shape (1, n_edges)
            Lower-triangle FC values as a feature vector.
            n_edges = k*(k-1)//2.
        """
        # Full-session FC matrix: (k, k) Pearson correlation across timepoints
        FC = np.corrcoef(Phi_sub.T)   # (k, k)

        # Extract lower triangle (excluding diagonal)
        i_lower, j_lower = np.tril_indices(self._k, k=-1)
        fingerprint = FC[i_lower, j_lower]   # (n_edges,)

        # Return as (1, n_edges) to match the MATLAB patterns(np, :) layout,
        # where np indexes the pattern within a subject (=1 in FULLWIN mode)
        return fingerprint[np.newaxis, :]    # (1, n_edges)

    @staticmethod
    def _is_invertible(A: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if a square matrix is invertible via its condition number.

        Uses the ratio of largest to smallest singular value as a proxy
        for numerical invertibility. A matrix is considered singular if
        its condition number exceeds 1/tol.

        Parameters
        ----------
        A : np.ndarray, shape (n, n)
        tol : float
            Numerical tolerance. Default: 1e-10.

        Returns
        -------
        bool
        """
        # np.linalg.cond uses SVD; cheaper than computing det for large matrices
        return np.linalg.cond(A) < (1.0 / tol)
