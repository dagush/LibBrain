"""
Neuroreduce/methods/base_charm.py
-----------------------------------
BaseCHARMKernel: shared kernel mathematics for all CHARM variants.

Both CHARM-BOLD and CHARM-SC share an identical three-step kernel pipeline:

    Step 1 — Build complex kernel matrix K from pairwise squared distances:
               K[i,j] = exp( i · d²_ij / σ )

    Step 2 — Raise to diffusion horizon t, take |·|², row-normalise:
               Q = |K^t|²
               P = D⁻¹ Q    (row-stochastic, real)

    Step 3 — Eigendecompose P, skip trivial eigenvector, scale by |λ|:
               Φ = Re(V[:, 1:k+1]) @ diag(|λ|)   (Tm×k or N×k)

The variants differ only in what feeds into d²_ij:
    CHARM-BOLD:  d²_ij = ||x_i - x_j||²   (BOLD timepoint columns, x ∈ ℝᴺ)
    CHARM-SC:    d²_ij = ||c_i - c_j||²   (parcel centroids, c ∈ ℝ³)

And in what is extracted from P:
    CHARM-BOLD:  Φ ∈ ℝ^(Tm×k)  then mapped to parcel space via nets()
    CHARM-SC:    Φ ∈ ℝ^(N×k)   directly in parcel space (or via nets() if
                                 BOLD is provided)

This mixin provides the shared methods so neither subclass duplicates code.
It is not a DimensionalityReducer itself — it is mixed into subclasses that
already inherit from DimensionalityReducer.

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy import linalg as LA
from scipy import stats


class BaseCHARMKernel:
    """
    Mixin providing shared CHARM kernel mathematics.

    Intended to be used via multiple inheritance alongside
    DimensionalityReducer:

        class CHARMReducer(DimensionalityReducer, BaseCHARMKernel): ...
        class CHARMSCReducer(DimensionalityReducer, BaseCHARMKernel): ...

    Subclasses must set self.epsilon, self.t_horizon, self.k, and
    self.sort_eigenvectors before calling any method here.
    """

    # ------------------------------------------------------------------
    # Shared: kernel + diffusion matrix
    # ------------------------------------------------------------------

    def _build_diffusion_matrix(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the complex CHARM diffusion matrix from a set of points.

        Computes pairwise squared Euclidean distances, applies the complex
        exponential kernel, raises to t_horizon, takes |·|², row-normalises.

        Parameters
        ----------
        points : np.ndarray, shape (M, D)
            M points in D-dimensional space.
            For CHARM-BOLD: (Tm, N) — transposed BOLD, timepoints as rows.
            For CHARM-SC:   (N, 3)  — parcel centroids.

        Returns
        -------
        Pmatrix : np.ndarray, shape (M, M)
            Row-stochastic diffusion matrix P = D⁻¹ |K^t|².
        Ptr_t : np.ndarray, shape (M, M)
            Intermediate |K^t|² before row-normalisation (needed for Nyström).
        Kmatrix : np.ndarray, shape (M, M), dtype complex
            Raw complex kernel matrix K (stored for Nyström in CHARM-BOLD).
        """
        M = points.shape[0]

        # ── Eq. (10): K[i,j] = exp( i · ||p_i - p_j||² / σ ) ────────────────
        # Vectorised pairwise squared distances — replaces the MATLAB double loop.
        # diff[i,j,:] = p_i - p_j  →  d2[i,j] = ||p_i - p_j||²
        diff    = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (M,M,D)
        d2      = np.sum(diff ** 2, axis=2)                             # (M,M)
        Kmatrix = np.exp(1j * d2 / self.epsilon)                        # (M,M) complex

        # ── Eq. (11): Q = |K^t|² ──────────────────────────────────────────────
        # Matrix power (not element-wise) — mixes all rows and columns.
        Ktr_t = LA.matrix_power(Kmatrix, self.t_horizon)
        Ptr_t = np.abs(Ktr_t) ** 2                                      # (M,M) real

        # ── Eq. (12-13): P = D⁻¹ Q  (row-stochastic) ─────────────────────────
        row_sums = np.sum(Ptr_t, axis=1)
        if np.any(row_sums == 0):
            warnings.warn(
                "Zero row-sum in CHARM diffusion matrix. "
                "Check for duplicate points or very large epsilon.",
                RuntimeWarning, stacklevel=3,
            )
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
        D       = np.diag(row_sums)
        Pmatrix = LA.inv(D) @ Ptr_t                                     # (M,M) real

        return Pmatrix, Ptr_t, Kmatrix

    # ------------------------------------------------------------------
    # Shared: eigendecomposition and eigenvector selection
    # ------------------------------------------------------------------

    def _eigendecompose(
        self,
        Pmatrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Eigendecompose the diffusion matrix and extract the top-k modes.

        Skips the trivial first eigenvector (constant, eigenvalue ≈ 1)
        and returns eigenvectors 2..k+1, scaled by their eigenvalue magnitudes
        as in Eq. (14) of the paper.

        Parameters
        ----------
        Pmatrix : np.ndarray, shape (M, M)
            Row-stochastic diffusion matrix.

        Returns
        -------
        Phi : np.ndarray, shape (M, k)
            Eigenvalue-scaled embedding  Φ = Re(V[:,1:k+1]) @ diag(|λ|).
        eigenvectors_k : np.ndarray, shape (M, k)
            Raw (unscaled) real parts of the selected eigenvectors.
            Stored separately for the Nyström formula.
        eigenvalues_k : np.ndarray, shape (k,)
            Absolute eigenvalue magnitudes |λ| for the selected modes.
        eigenvalues_k_signed : np.ndarray, shape (k,)
            Signed real eigenvalues λ. Used in the Nyström formula —
            |λ| gives the wrong sign for negative eigenvalues.
        """
        LL, VV = LA.eig(Pmatrix)

        if self.sort_eigenvectors:
            order = np.argsort(np.abs(LL))[::-1]
            LL    = LL[order]
            VV    = VV[:, order]
            if not np.isclose(np.abs(LL[0]), 1.0, atol=1e-3):
                warnings.warn(
                    f"Dominant eigenvalue magnitude is {np.abs(LL[0]):.4f}, "
                    "expected ≈ 1.0 for a row-stochastic matrix. "
                    "Check your input data and epsilon parameter.",
                    RuntimeWarning, stacklevel=3,
                )

        # Skip trivial first eigenvector (index 0), take indices 1..k
        selected_idx         = slice(1, self.k + 1)
        eigenvalues_k        = np.abs(LL[selected_idx])          # (k,)
        eigenvalues_k_signed = np.real(LL[selected_idx])         # (k,) signed
        LLMatr_k             = np.diag(eigenvalues_k)            # (k,k)

        # Eq. (14): Φ = Re(V) @ |Λ|
        eigenvectors_k = np.real(VV[:, selected_idx])            # (M,k) unscaled
        Phi            = eigenvectors_k @ LLMatr_k               # (M,k) scaled

        return Phi, eigenvectors_k, eigenvalues_k, eigenvalues_k_signed

    # ------------------------------------------------------------------
    # Shared: nets() — correlation-based parcel-space basis
    # ------------------------------------------------------------------

    def _nets(self, Phi: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Map the timepoint/parcel embedding Φ into a parcel-space basis via
        Pearson correlation with the BOLD signal, then L2-normalise columns.

        Corresponds to the nets() function in the original MATLAB/Python code.

        Parameters
        ----------
        Phi : np.ndarray, shape (M, k)
            Latent embedding — rows are either timepoints (CHARM-BOLD)
            or parcels (CHARM-SC when M==N).
        X : np.ndarray, shape (N, T)
            BOLD timeseries in Neuroreduce (N, T) convention.

        Returns
        -------
        conet : np.ndarray, shape (N, k)
            L2-normalised correlation basis. Each column is a spatial map
            over parcels for one latent dimension.
        """
        # z-score both signals across their time/sample axis
        zPhiA = stats.zscore(X.T,  ddof=1)   # (T, N)  z-scored BOLD
        zPhi  = stats.zscore(Phi,  ddof=1)   # (M, k)  z-scored embedding

        # Pearson correlation: (k, N) then transpose → (N, k)
        conet2 = (zPhi.T @ zPhiA) / (zPhi.shape[0] - 1)   # (k, N)
        conet2 = conet2.T                                   # (N, k)

        # L2-normalise each column
        norms  = LA.norm(conet2, axis=0, keepdims=True)
        norms  = np.where(norms == 0, 1.0, norms)
        return conet2 / norms                               # (N, k)

    # ------------------------------------------------------------------
    # Shared: Nyström out-of-sample extension
    # ------------------------------------------------------------------

    def _nystrom_transform_shared(
        self,
        X_new:         np.ndarray,
        X_fit:         np.ndarray,
        Phi:           np.ndarray,
        eigenvalues_signed: np.ndarray,
        Pmatrix:       np.ndarray,
        is_same_data:  bool,
        use_exact_rows: bool,
    ) -> np.ndarray:
        """
        Nyström out-of-sample extension for CHARM-BOLD.

        Shared implementation — called by CHARMReducer.transform().
        CHARM-SC does not use Nyström (its basis is in parcel space,
        and new BOLD is projected directly via conet.T @ X).

        For mathematical derivation see CHARMReducer._nystrom_transform.

        Parameters
        ----------
        X_new          : (N, T_new)  new BOLD, already validated
        X_fit          : (N, Tm)     training BOLD
        Phi            : (Tm, k)     eigenvalue-scaled embedding
        eigenvalues_signed : (k,)   signed eigenvalues for Nyström formula
        Pmatrix        : (Tm, Tm)   stored diffusion matrix
        is_same_data   : bool        True if X_new is the training data
        use_exact_rows : bool        True to use exact Pmatrix rows

        Returns
        -------
        Z : np.ndarray, shape (k, T_new)
        """
        T_new = X_new.shape[1]
        Z     = np.zeros((self.k, T_new))

        for t in range(T_new):
            if use_exact_rows:
                # Exact path: read the pre-computed row of P directly
                p_row = Pmatrix[t, :]
            else:
                # Approximate path: element-wise kernel (see CHARMReducer docs)
                x      = X_new[:, t]
                diffs2 = np.sum((X_fit - x[:, None]) ** 2, axis=0)
                k_row  = np.exp(1j * diffs2 / self.epsilon)
                k_row_t = k_row ** self.t_horizon
                q_row   = np.abs(k_row_t) ** 2
                d       = q_row.sum()
                if d == 0:
                    warnings.warn(
                        f"Zero row-sum at t={t} in Nyström kernel. "
                        "Timepoint may be outside the training manifold.",
                        RuntimeWarning, stacklevel=2,
                    )
                    d = 1.0
                p_row = q_row / d

            # Nyström formula: Z[:,t] = (p_row @ Phi) / λ_signed
            Z[:, t] = (p_row @ Phi) / eigenvalues_signed

        return Z
