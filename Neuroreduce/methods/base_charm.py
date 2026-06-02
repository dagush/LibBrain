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
        kernel_type: str = 'quantum',
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the CHARM diffusion matrix from a set of points.

        Supports two kernel variants that differ in how the kernel matrix is
        constructed and whether a matrix power is applied.

        Kernel variants
        ---------------
        ``'quantum'`` (default, Eq. 10-13 of the 2025 PRE paper):
            K[i,j] = exp( i · d²_ij / σ )          complex kernel
            Q      = |K^t|²                          matrix power then |·|²
            P      = D⁻¹ Q                           row-stochastic

            ``Ptr_t`` stores Q = |K^t|² (real).
            ``Kmatrix`` stores K (complex).

        ``'classical'`` (Compare_Analysis_singleTh.m, real Gaussian kernel):
            K[i,j] = exp( -d²_ij / σ )              real symmetric kernel
            P      = D⁻¹ K                           row-stochastic directly
                                                      (no matrix power step)

            ``Ptr_t`` stores K (real) — the raw kernel before normalisation.
            This is the cross-block used in the Nyström CV reconstruction:
                 ts_est = K_cv @ Φ_tr @ Λ^{-τ} @ Φ_tr.T @ ts_train.T
            ``Kmatrix`` = K (same as Ptr_t for this variant).

            The diffusion horizon τ (self.t_horizon) enters only via the
            eigenvalue scaling and Nyström denominator, NOT the kernel itself.

        Parameters
        ----------
        points : np.ndarray, shape (M, D)
            M points in D-dimensional space.
            For CHARM-BOLD: (Tm, N) — transposed BOLD, timepoints as rows.
            For CHARM-SC:   (N, 3)  — parcel centroids.
        kernel_type : {'quantum', 'classical'}
            Which kernel variant to build. Default: 'quantum'.

        Returns
        -------
        Pmatrix : np.ndarray, shape (M, M)
            Row-stochastic diffusion matrix.
        Ptr_t : np.ndarray, shape (M, M)
            Raw kernel (classical) or |K^t|² (quantum) before row-normalisation.
            Stored for Nyström CV: cross-block Ptr_t[T_tr:, :T_tr] is the
            correct left-multiplier in both reconstruction formulas.
        Kmatrix : np.ndarray, shape (M, M)
            Raw kernel matrix K — complex for quantum, real for classical.
        """
        if kernel_type not in ('quantum', 'classical'):
            raise ValueError(
                f"kernel_type must be 'quantum' or 'classical', got {kernel_type!r}"
            )

        M = points.shape[0]

        # ── Pairwise squared distances (shared by both variants) ───────────────
        # Vectorised via ||a-b||² = ||a||² + ||b||² - 2aᵀb.
        # diff approach kept for clarity; einsum is equivalent for small D.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (M,M,D)
        d2   = np.sum(diff ** 2, axis=2)                             # (M,M) real

        if kernel_type == 'quantum':
            # ── Eq. (10): K[i,j] = exp( i · d²_ij / σ ) ─────────────────────
            Kmatrix = np.exp(1j * d2 / self.epsilon)                 # (M,M) complex

            # ── Eq. (11): Q = |K^t|² ─────────────────────────────────────────
            # Matrix power (not element-wise) — mixes all rows and columns.
            Ktr_t = LA.matrix_power(Kmatrix, self.t_horizon)
            Ptr_t = np.abs(Ktr_t) ** 2                               # (M,M) real

        else:  # 'classical'
            # ── Real Gaussian kernel: K[i,j] = exp( -d²_ij / σ ) ─────────────
            # No matrix power — τ only appears later in eigenvalue scaling and
            # in the Nyström denominator (Λ^{-τ} instead of Λ^{-1}).
            Kmatrix = np.exp(-d2 / self.epsilon)                     # (M,M) real
            Ptr_t   = Kmatrix                                        # alias: no copy

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
        Pmatrix = LA.inv(D) @ Ptr_t                                  # (M,M) real

        return Pmatrix, Ptr_t, Kmatrix

    # ------------------------------------------------------------------
    # Shared: eigendecomposition and eigenvector selection
    # ------------------------------------------------------------------

    def _eigendecompose(
        self,
        Pmatrix: np.ndarray,
        eigenvalue_scale: str = 'abs',
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Eigendecompose the diffusion matrix and extract the top-k modes.

        Skips the trivial first eigenvector (constant, eigenvalue ≈ 1)
        and returns eigenvectors 2..k+1, scaled by their eigenvalues according
        to ``eigenvalue_scale``.

        Scaling variants
        ----------------
        ``'abs'`` (default, quantum CHARM, Eq. 14 of the 2025 PRE paper):
            Φ = Re(V[:,1:k+1]) @ diag(|λ|)
            The magnitude |λ| is used, matching the quantum kernel convention.

        ``'power'`` (classical CHARM, Compare_Analysis_singleTh.m):
            Φ = Re(V[:,1:k+1]) @ diag(λ^τ)    where τ = self.t_horizon
            The real eigenvalue raised to the diffusion horizon τ is used,
            matching the MATLAB line: ``Phi = Phi * (LL(...).^Thorizont)``.
            λ^τ and |λ|^τ are identical for real positive eigenvalues (which
            is guaranteed for the real Gaussian kernel by Perron-Frobenius).

        Both variants always return the UNSCALED eigenvectors separately
        (``eigenvectors_k``), which is what the Nyström formula requires.

        Parameters
        ----------
        Pmatrix : np.ndarray, shape (M, M)
            Row-stochastic diffusion matrix.
        eigenvalue_scale : {'abs', 'power'}
            How to scale Φ columns. Default: 'abs'.

        Returns
        -------
        Phi : np.ndarray, shape (M, k)
            Eigenvalue-scaled embedding.
        eigenvectors_k : np.ndarray, shape (M, k)
            Raw (unscaled) real parts of the selected eigenvectors.
            Stored separately for the Nyström formula.
        eigenvalues_k : np.ndarray, shape (k,)
            Absolute eigenvalue magnitudes |λ| for the selected modes.
        eigenvalues_k_signed : np.ndarray, shape (k,)
            Signed real eigenvalues λ. Used to build the Nyström denominator —
            callers raise to the appropriate power (1 for quantum, τ for
            classical) and store the result as _eigenvalues_nystrom.
        """
        if eigenvalue_scale not in ('abs', 'power'):
            raise ValueError(
                f"eigenvalue_scale must be 'abs' or 'power', got {eigenvalue_scale!r}"
            )

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
        eigenvalues_k        = np.abs(LL[selected_idx])          # (k,) |λ|
        eigenvalues_k_signed = np.real(LL[selected_idx])         # (k,) signed λ
        eigenvectors_k       = np.real(VV[:, selected_idx])      # (M,k) unscaled

        # ── Scale Φ columns according to the chosen variant ───────────────────
        if eigenvalue_scale == 'abs':
            # Eq. (14): Φ = Re(V) @ |Λ|  — quantum default
            scale = eigenvalues_k                                  # |λ|
        else:
            # Classical: Φ[:,d] *= λ_d^τ  (MATLAB: Phi * LL.^Thorizont)
            scale = eigenvalues_k_signed ** self.t_horizon         # λ^τ

        LLMatr_k = np.diag(scale)                                 # (k,k)
        Phi      = eigenvectors_k @ LLMatr_k                      # (M,k) scaled

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
