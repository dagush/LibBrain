"""
Neuroreduce/utils/harmonic_analysis.py
----------------------------------------
Analysis utilities for Connectome and Functional Harmonics.

Provides HarmonicAnalysis — a class that wraps a fitted
BaseLaplacianReducer and exposes the RSN-projection and dynamic
analysis pipelines from the student's original code.

Corresponds to (and refactors):
    Projecter.projectVectorRegion()  →  project_rsn_vectors()
    Projecter.projectVectorTime()    →  project_timeseries()  (= transform())
    mutualInfo module                →  mutual_information()
    reconstructionError module       →  reconstruction_error()

Convention note
---------------
Neuroreduce uses (N, T) for BOLD and (N, k) for basis matrices.
NeuroNumba observables use (T, N). All transpositions are handled
internally — callers always use the Neuroreduce convention.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
from numpy import linalg as LA
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif

from Neuroreduce.methods.base_laplacian import BaseLaplacianReducer


class HarmonicAnalysis:
    """
    Post-hoc analyses on a fitted harmonic reducer.

    Wraps ConnectomeHarmonicsReducer or FunctionalHarmonicsReducer and
    exposes the full static + dynamic analysis pipeline.

    Parameters
    ----------
    reducer : BaseLaplacianReducer
        A fitted ConnectomeHarmonicsReducer or FunctionalHarmonicsReducer.
    """

    def __init__(self, reducer: BaseLaplacianReducer):
        if not isinstance(reducer, BaseLaplacianReducer):
            raise TypeError(
                f"Expected a BaseLaplacianReducer subclass, "
                f"got {type(reducer).__name__}."
            )
        reducer._check_is_fitted()
        self._reducer = reducer

    # ------------------------------------------------------------------
    # Static analysis: RSN projection
    # ------------------------------------------------------------------

    def project_rsn_vectors(
        self,
        rsn_matrix:     np.ndarray,
        sign_invariant: bool = True,
        n_harmonics:    Optional[int] = None,
    ) -> np.ndarray:
        """
        Project RSN binary vectors onto the harmonic basis.

        For each RSN r and each harmonic d, computes the projection
        coefficient alpha[d, r] = dot(phi_d, rsn_r).

        Corresponds to Projecter.projectVectorRegion() in the student's code.

        Parameters
        ----------
        rsn_matrix : np.ndarray, shape (N, n_rsn)
            RSN binary (or continuous) vectors. Each column is one RSN.
            Must have N rows matching the reducer's N parcels.
        sign_invariant : bool
            If True, take max(dot(phi,rsn), dot(-phi,rsn)) to handle
            eigenvector sign ambiguity. Default: True.
        n_harmonics : int or None
            Number of harmonics to use. If None, uses reducer.k.
            Can be set larger than reducer.k if get_all_eigenvectors()
            is available and more harmonics are needed.

        Returns
        -------
        alpha : np.ndarray, shape (n_harmonics, n_rsn)
            Projection coefficients. alpha[d, r] is the alignment of
            RSN r with harmonic d.
        """
        rsn_matrix = np.atleast_2d(rsn_matrix)
        if rsn_matrix.shape[0] == 1:
            rsn_matrix = rsn_matrix.T   # ensure (N, n_rsn)

        N_rsn, n_rsn = rsn_matrix.shape

        # Choose basis: top-k or all eigenvectors
        if n_harmonics is None or n_harmonics <= self._reducer.k:
            W = self._reducer.get_basis()            # (N, k)
        else:
            W = self._reducer.get_all_eigenvectors() # (N, N)
        W = W[:, :n_harmonics] if n_harmonics else W
        n_h = W.shape[1]

        # Align dimensions — trim to the smaller of N_rsn and N_basis
        N = min(N_rsn, W.shape[0])
        W         = W[:N, :]
        rsn_matrix = rsn_matrix[:N, :]

        alpha = np.zeros((n_h, n_rsn))
        for d in range(n_h):
            phi = W[:, d]                           # (N,)
            for r in range(n_rsn):
                rsn = rsn_matrix[:, r]              # (N,)
                if sign_invariant:
                    # Matches Projecter.projectVectorRegion(invert=True)
                    alpha[d, r] = max(
                        np.dot(phi, rsn),
                        np.dot(-phi, rsn),
                    )
                else:
                    alpha[d, r] = np.dot(phi, rsn)

        return np.round(alpha, 5)

    # ------------------------------------------------------------------
    # Static analysis: select harmonics by RSN importance
    # ------------------------------------------------------------------

    def select_harmonics_by_rsn(
        self,
        alpha:     np.ndarray,
        n_select:  int,
        method:    str = 'max_projection',
    ) -> np.ndarray:
        """
        Select the n_select most RSN-relevant harmonics.

        Used in the dynamic pipeline to choose which harmonics to project
        the timeseries onto — matching the student's sorting step.

        Parameters
        ----------
        alpha : np.ndarray, shape (n_harmonics, n_rsn)
            RSN projection matrix from project_rsn_vectors().
        n_select : int
            Number of harmonics to select.
        method : str
            'max_projection' : select harmonics with the highest maximum
                               projection across all RSNs (default).
            'sum_projection' : select harmonics with the highest sum of
                               projections across all RSNs.

        Returns
        -------
        selected_indices : np.ndarray, shape (n_select,)
            Indices into the harmonic basis of the selected harmonics,
            sorted by descending importance.
        """
        if method == 'max_projection':
            importance = alpha.max(axis=1)
        elif method == 'sum_projection':
            importance = alpha.sum(axis=1)
        else:
            raise ValueError(f"Unknown method '{method}'.")

        return np.argsort(importance)[::-1][:n_select]

    # ------------------------------------------------------------------
    # Dynamic analysis: project timeseries
    # ------------------------------------------------------------------

    def project_timeseries(
        self,
        X:              np.ndarray,
        sign_invariant: bool = True,
        harmonic_idx:   Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Project BOLD timeseries onto selected harmonics.

        Corresponds to Projecter.projectVectorTime() in the student's code.

        Per-timepoint sign handling: for each timepoint t and harmonic d,
            beta[d, t] = max(dot(phi_d, X[:,t]), dot(-phi_d, X[:,t]))
        This preserves projection magnitude regardless of eigenvector sign.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)
            BOLD timeseries.
        sign_invariant : bool
            Use per-timepoint sign-invariant projection. Default: True.
        harmonic_idx : np.ndarray of int, or None
            If provided, project only onto these harmonics (e.g. from
            select_harmonics_by_rsn()). If None, use all k basis harmonics.

        Returns
        -------
        beta : np.ndarray, shape (n_selected, T)
            Harmonic coefficients over time.
        """
        X = self._reducer._validate_input(X)   # (N, T)

        if harmonic_idx is not None:
            # Use selected harmonics from all eigenvectors
            W = self._reducer.get_all_eigenvectors()[:, harmonic_idx]
        else:
            W = self._reducer.get_basis()       # (N, k)

        N, T  = X.shape
        n_h   = W.shape[1]
        beta  = np.zeros((n_h, T))

        for t in range(T):
            x_t = X[:, t]                       # (N,)
            for d in range(n_h):
                phi = W[:, d]                   # (N,)
                if sign_invariant:
                    # Faithful to Projecter.projectVectorRegion(invert=True):
                    # max(dot(phi, x), dot(-phi, x)) = |dot(phi, x)|
                    beta[d, t] = max(
                        np.dot(phi, x_t),
                        np.dot(-phi, x_t),
                    )
                else:
                    beta[d, t] = np.dot(phi, x_t)

        return np.round(beta, 5)

    # ------------------------------------------------------------------
    # Reconstruction error
    # ------------------------------------------------------------------

    def reconstruction_error(
        self,
        X:            np.ndarray,
        harmonic_idx: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute reconstruction error metrics for the harmonic basis.

        Reconstructs X from its harmonic projections and computes MSE,
        RMSE, and Pearson r between original and reconstructed BOLD.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)
        harmonic_idx : np.ndarray or None
            Selected harmonic indices. If None, uses all k basis harmonics.

        Returns
        -------
        dict with keys:
            'mse'    : float — mean squared error
            'rmse'   : float — root mean squared error
            'pearson_r' : float — correlation between X and X_hat (flattened)
            'X_hat'  : np.ndarray (N, T) — reconstructed BOLD
        """
        X    = self._reducer._validate_input(X)
        beta = self.project_timeseries(X, sign_invariant=False,
                                       harmonic_idx=harmonic_idx)

        if harmonic_idx is not None:
            W = self._reducer.get_all_eigenvectors()[:, harmonic_idx]
        else:
            W = self._reducer.get_basis()

        X_hat  = W @ beta                        # (N, T)
        mse    = float(np.mean((X - X_hat) ** 2))
        rmse   = float(np.sqrt(mse))
        r, _   = pearsonr(X.ravel(), X_hat.ravel())

        return {
            'mse':       mse,
            'rmse':      rmse,
            'pearson_r': float(r),
            'X_hat':     X_hat,
        }

    # ------------------------------------------------------------------
    # Mutual information between RSN and harmonic projections
    # ------------------------------------------------------------------

    def mutual_information(
        self,
        phi,
        rsn
    ) -> np.ndarray:
        """
        Computes the mutual information between RSNs and each eigenvector.

        Returns
        -------
        mi : np.ndarray, shape (RSNs x harmonics)
             matrix of mutual information
        """
        # np.random.seed(42)  # just for debug
        mutual_info = []
        for i in range(rsn.shape[1]):  # iterate through RSNs
            vector = rsn[:, i]
            mi_rsn_i = mutual_info_classif(phi, vector, discrete_features=False)
            mutual_info.append(mi_rsn_i)

        mi = np.array(mutual_info)
        return mi
