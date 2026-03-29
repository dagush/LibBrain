"""
Neuroreduce/methods/charm.py
-----------------------------
CHARM: Complex HARMonics dimensionality reduction for fMRI BOLD signals.

Implements the framework from:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

Original MATLAB code by Gustavo Deco.
Python translation and Neuroreduce integration by Gustavo Patow.

Convention recap (Neuroreduce-wide):
    N  : number of brain parcels / ROIs
    T  : number of fMRI timepoints (per subject)
    Tm : total concatenated timepoints (= NSUB * T_per_subject after trimming)
    k  : number of reduced dimensions (LATDIM in the paper)

    BOLD input  : np.ndarray, shape (N, T)   or (N, Tm) when multi-subject
    Output      : np.ndarray, shape (k, T)
    Basis       : np.ndarray, shape (N, k)   — parcel-space, via nets()
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy import linalg as LA
from scipy import stats

from Neuroreduce.base import DimensionalityReducer


class CHARMReducer(DimensionalityReducer):
    """
    CHARM (Complex HARMonics) dimensionality reduction for fMRI.

    CHARM builds a complex diffusion kernel over the *timepoint* space of the
    concatenated BOLD signal, computes its spectral decomposition, and recovers
    a parcel-space basis by correlating the latent coordinates back against the
    original BOLD signal (the ``nets()`` step in the paper).

    Key difference from classical diffusion maps: the kernel uses a complex
    exponential ``exp(i·d²/σ)`` instead of the real ``exp(-d²/σ)``, which
    captures oscillatory structure in the data manifold.

    Parameters
    ----------
    k : int
        Number of latent dimensions to retain (``LATDIM`` in the paper).
    epsilon : float
        Kernel bandwidth σ (Eq. 10). Controls the spatial scale of the
        diffusion. Default: 300 (value used in the 2025 PRE paper).
    t_horizon : int
        Diffusion horizon t (Eq. 11). The kernel matrix is raised to this
        power before extracting eigenvectors. Default: 2.
    whiten : bool
        If True, z-score each row of the projected output across time.
        Default: False.
    sort_eigenvectors : bool
        If True (recommended), sort eigenpairs by descending eigenvalue
        magnitude before selecting the top-k. The paper assumes this order
        but neither numpy nor MATLAB's eig() guarantee it. Default: True.

    Notes on the fit / transform split
    ------------------------------------
    CHARM is fit on the *concatenated* BOLD of all subjects. The latent
    embedding Φ (shape Tm×k) is defined over those exact Tm timepoints.

    ``transform(X)`` therefore has two modes:

    1. **Same data** (default): if ``X`` is the same array used in ``fit``
       (checked by identity ``X is self._X_fit``), Φ is returned directly —
       no recomputation, guaranteed exact match with the paper's results.

    2. **Out-of-sample / Nyström** (new data): the Nyström extension embeds
       new timepoints into the existing manifold via:

           φ̃ᵢ(x_new) = (1/λᵢ) · P(x_new, X_train) · φᵢ

       where P(x_new, X_train) is the row of the row-normalised kernel between
       the new point and all training timepoints.

    Set ``force_nystrom=True`` in ``transform()`` to force the Nyström path
    even on the training data — useful for numerical verification.

    Examples
    --------
    >>> reducer = CHARMReducer(k=7, epsilon=300, t_horizon=2)
    >>> Z = reducer.fit_transform(X_concat)   # X_concat : (N, Tm)
    >>> W = reducer.get_basis()               # W : (N, 7)  parcel-space basis
    >>> Z_new = reducer.transform(X_new)      # X_new : (N, T') — Nyström
    """

    def __init__(
        self,
        k: int = 7,
        epsilon: float = 300.0,
        t_horizon: int = 2,
        whiten: bool = False,
        sort_eigenvectors: bool = True,
    ):
        super().__init__(k=k, whiten=whiten)
        self.epsilon = epsilon
        self.t_horizon = t_horizon
        self.sort_eigenvectors = sort_eigenvectors

        # Set during fit
        self._X_fit_original: Optional[np.ndarray] = None  # pre-validation ref for identity check
        self._X_fit: Optional[np.ndarray] = None           # validated (float32) copy
        self._Phi: Optional[np.ndarray] = None          # (Tm, k)  eigenvalue-scaled embedding
        self._eigenvectors: Optional[np.ndarray] = None # (Tm, k)  raw (unscaled) eigenvectors
        self._eigenvalues: Optional[np.ndarray] = None        # (k,) |λ| magnitudes, for Phi scaling
        self._eigenvalues_signed: Optional[np.ndarray] = None # (k,) signed λ, for Nyström formula
        self._conet: Optional[np.ndarray] = None        # (N, k)   parcel-space basis
        self._Pmatrix: Optional[np.ndarray] = None      # (Tm, Tm) row-normalised diffusion matrix
        self._Ptr_t: Optional[np.ndarray] = None        # (Tm, Tm) |K^t|^2 before normalisation
        # Both _Pmatrix and _Ptr_t are stored because:
        #   - _Pmatrix rows give EXACT Nyström rows for training timepoints
        #   - _Ptr_t is needed for out-of-sample normalisation (Eq. 12-13)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, SC: Optional[np.ndarray] = None) -> "CHARMReducer":
        """
        Learn the CHARM latent space from the (concatenated) BOLD signal.

        Parameters
        ----------
        X  : np.ndarray, shape (N, Tm)
            BOLD fMRI timeseries. For multi-subject analysis, pass the
            concatenated timeseries (N × Tm) where Tm = sum of per-subject T.
            Preprocessing (filtering, z-scoring) should be done before calling
            fit(); see ``filterAndConcatSubj`` in the original pipeline.
        SC : ignored — this variant operates on BOLD alone.

        Returns
        -------
        self
        """
        # Store the PRE-validation reference for the identity check in transform().
        # _validate_input() always produces a new array (dtype coercion),
        # so we must compare against the original object the caller passed,
        # not the coerced copy.
        self._X_fit_original = X             # pre-validation reference
        X = self._validate_input(X)
        self._X_fit = X                      # validated (float32) copy

        # Compute latent embedding Φ ∈ ℝ^(Tm × k).
        # _latent() now returns 5 values; the extra two (raw eigenvectors and
        # Ptr_t) are needed for a numerically correct Nyström extension.
        (self._Phi,
         self._eigenvectors,
         self._eigenvalues,
         self._eigenvalues_signed,
         self._Pmatrix,
         self._Ptr_t) = self._latent(X)

        # Recover parcel-space basis via correlation (the nets() step)
        self._conet = self._nets(self._Phi, X)

        self._is_fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        force_nystrom: bool = False,
    ) -> np.ndarray:
        """
        Project BOLD data into the CHARM latent space.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)
            BOLD fMRI timeseries. May be the training data or new data.
        force_nystrom : bool
            If True, always use the Nyström extension even when X is the
            training data. Useful for numerical verification. Default: False.

        Returns
        -------
        Z : np.ndarray, shape (k, T)
            Projected signal: each row is one latent dimension over time.
        """
        self._check_is_fitted()

        # Identity check BEFORE _validate_input(), because _validate_input()
        # always returns a new array (dtype coercion to float32), so the
        # post-validation array can never be identical to self._X_fit.
        # Check identity BEFORE _validate_input() — independent of force_nystrom.
        # is_same_data: True whenever the caller passed the exact same object
        #   used in fit(), regardless of whether force_nystrom is set.
        # is_training_data: True only when we should take the fast Phi.T path
        #   (same data AND caller did NOT ask for Nyström).
        is_same_data     = (X is self._X_fit_original)
        is_training_data = is_same_data and (not force_nystrom)

        X = self._validate_input(X)

        # --- fast path: same data as training ---
        if is_training_data:
            # Φ is (Tm, k); transpose to (k, Tm) for our (k, T) convention
            Z = self._Phi.T
            return self._apply_whitening(Z)

        # --- out-of-sample / forced Nyström ---
        # When force_nystrom=True on training data, pass the stored validated
        # _X_fit so that _nystrom_transform can use exact Pmatrix rows.
        # When genuinely new data, pass X as-is.
        # is_training_data was computed before _validate_input() above,
        # so it correctly reflects the original caller's intent.
        # When force_nystrom=True on training data, pass self._X_fit (the
        # validated copy) so _nystrom_transform reads exact Pmatrix rows.
        # use_exact_rows: True when force_nystrom=True on the training data.
        # In that case we pass self._X_fit (validated copy) and read exact
        # Pmatrix rows — giving zero approximation error on training data.
        use_exact     = force_nystrom and is_same_data
        nystrom_input = self._X_fit if use_exact else X
        Z = self._nystrom_transform(nystrom_input, use_exact_rows=use_exact)
        return self._apply_whitening(Z)

    def get_basis(self) -> np.ndarray:
        """
        Return the parcel-space basis matrix (the ``conet`` matrix).

        This is the result of correlating the latent coordinates Φ back
        against the original BOLD signal, then normalising each column.
        Analogous to PCA's principal components in parcel space.

        Returns
        -------
        W : np.ndarray, shape (N, k)
            Each column is a spatial map over parcels for one latent dimension.
        """
        self._check_is_fitted()
        return self._conet

    # ------------------------------------------------------------------
    # CHARM core: latent() — preserved from original, with paper equations
    # ------------------------------------------------------------------

    def _latent(
        self, ts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the CHARM latent space embedding.

        Corresponds to the ``latent()`` function in the original Python code.
        Equations reference the 2025 Physical Review E paper.

        Parameters
        ----------
        ts : np.ndarray, shape (N, Tm)
            Concatenated, preprocessed BOLD timeseries.

        Returns
        -------
        Phi : np.ndarray, shape (Tm, k)
            Latent embedding: first k non-trivial eigenvectors scaled by
            their eigenvalues (Eq. 14).
        eigenvalues : np.ndarray, shape (k,)
            Eigenvalues corresponding to the selected eigenvectors.
        Pmatrix : np.ndarray, shape (Tm, Tm)
            Row-normalised diffusion matrix (stored for Nyström extension).
        """
        Tm = ts.shape[1]  # total concatenated timepoints

        # ------------------------------------------------------------------
        # Eq. (10):
        # \hat{W}_{ij} = exp( i * ||x_i - x_j||^2 / sigma )  ∈ ℂ^{Tm × Tm}
        # ------------------------------------------------------------------
        Kmatrix = np.zeros((Tm, Tm), dtype=complex)
        for i in range(Tm):
            for j in range(Tm):
                dij2 = np.sum((ts[:, i] - ts[:, j]) ** 2)
                Kmatrix[i, j] = np.exp((0 + 1j) * dij2 / self.epsilon)

        # ------------------------------------------------------------------
        # Eq. (11):
        # \hat{Q}(t) = | \hat{W}^t |^2  ∈ ℝ^{Tm × Tm}
        # ------------------------------------------------------------------
        Ktr_t = LA.matrix_power(Kmatrix, self.t_horizon)
        Ptr_t = np.square(np.abs(Ktr_t))

        # ------------------------------------------------------------------
        # Eq. (12):
        # \hat{D}_{ii} = sum_j \hat{Q}_{ij}
        # ------------------------------------------------------------------
        Dmatrix = np.diag(np.sum(Ptr_t, axis=1))

        # ------------------------------------------------------------------
        # Eq. (13):
        # \hat{P}(t) = \hat{D}^{-1} \hat{Q}(t)  ∈ ℝ^{Tm × Tm}
        # ------------------------------------------------------------------
        Pmatrix = LA.inv(Dmatrix) @ Ptr_t

        # ------------------------------------------------------------------
        # Eigendecomposition
        # \hat{Ψ} = [\hat{φ}_1, ..., \hat{φ}_N]  ∈ ℝ^{Tm × Tm}
        # ------------------------------------------------------------------
        LL, VV = LA.eig(Pmatrix)  # eigenvalues, eigenvectors

        if self.sort_eigenvectors:
            # Sort by descending eigenvalue magnitude (not guaranteed by eig)
            order = np.argsort(np.abs(LL))[::-1]
            LL = LL[order]
            VV = VV[:, order]
            # Sanity check: the dominant eigenvalue should be ≈ 1 for a
            # row-stochastic matrix
            if not np.isclose(np.abs(LL[0]), 1.0, atol=1e-3):
                warnings.warn(
                    f"Dominant eigenvalue magnitude is {np.abs(LL[0]):.4f}, "
                    "expected ≈ 1.0 for a row-stochastic matrix. "
                    "Check your input data and epsilon parameter.",
                    RuntimeWarning,
                    stacklevel=3,
                )

        # Skip the trivial first eigenvector (constant / stationary dist.)
        # Select eigenvectors 2 .. k+1  →  indices 1 .. k
        # \hat{Ψ}_reduced = [\hat{φ}_2, ..., \hat{φ}_{k+1}]  ∈ ℝ^{Tm × k}
        selected_idx = slice(1, self.k + 1)
        eigenvalues_k = np.abs(LL[selected_idx])          # (k,)  real, positive
        LLMatr_k = np.diag(eigenvalues_k)                 # (k, k)

        # ------------------------------------------------------------------
        # Eq. (14):
        # \hat{P}(t) = \hat{Ψ} \hat{Λ} \hat{Ψ}^T
        # Phi = Ψ_k * |Λ_k|  ∈ ℝ^{Tm × k}
        # ------------------------------------------------------------------
        # Raw (unscaled) eigenvectors — stored separately for Nyström.
        # The standard Nyström formula requires unscaled eigenvectors;
        # Phi below is the eigenvalue-scaled version used everywhere else.
        eigenvectors_k = np.real(VV[:, selected_idx])      # (Tm, k)  unscaled

        # Signed eigenvalues: used in the Nyström formula p_row @ V / λ.
        # IMPORTANT: eigenvalues can be negative real — np.abs() would
        # break the sign in the Nyström formula. We store both:
        #   eigenvalues_k        = |λ|  for scaling Phi (Eq. 14 uses abs)
        #   eigenvalues_k_signed = λ    for Nyström (must preserve sign)
        eigenvalues_k_signed = np.real(LL[selected_idx])   # (k,) signed, real part

        Phi = eigenvectors_k @ LLMatr_k                    # (Tm, k)  scaled (Eq. 14)

        return Phi, eigenvectors_k, eigenvalues_k, eigenvalues_k_signed, Pmatrix, Ptr_t

    # ------------------------------------------------------------------
    # CHARM core: nets() — parcel-space basis recovery
    # ------------------------------------------------------------------

    def _nets(self, Phi: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """
        Convert the timepoint-space embedding Φ into a parcel-space basis.

        Correlates each latent dimension (column of Φ) with each parcel's
        BOLD timeseries, then L2-normalises each column.

        Parameters
        ----------
        Phi : np.ndarray, shape (Tm, k)
            Latent embedding from ``_latent()``.
        ts  : np.ndarray, shape (N, Tm)
            The same BOLD signal used to compute Phi.

        Returns
        -------
        conet : np.ndarray, shape (N, k)
            Parcel-space basis. Equivalent to PCA's components_ transposed.
        """
        N = ts.shape[0]

        zPhiA = stats.zscore(ts.T, ddof=1)    # (Tm, N)  z-scored BOLD
        zPhi  = stats.zscore(Phi,  ddof=1)    # (Tm, k)  z-scored latent coords

        # Correlation between each latent dim and each parcel BOLD signal
        # conet[seed, red] = corr( zPhi[:, red], zPhiA[:, seed] )
        conet2 = zPhi.T @ zPhiA / (zPhi.shape[0] - 1)  # (k, N),  vectorised corrcoef
        conet2 = conet2.T                                # (N, k)

        # L2-normalise each column (latent dimension)
        conet = conet2 / LA.norm(conet2, axis=0, keepdims=True)  # (N, k)
        return conet

    # ------------------------------------------------------------------
    # Nyström out-of-sample extension
    # ------------------------------------------------------------------

    def _nystrom_transform(
        self,
        X_new: np.ndarray,
        use_exact_rows: bool = False,
    ) -> np.ndarray:
        """
        Embed timepoints into the fitted CHARM manifold via Nyström.

        The correct Nyström formula for a non-symmetric row-stochastic matrix P
        with eigenpairs (λᵢ, φᵢ) is:

            φ̃ᵢ(x) = (1/λᵢ) · p_row(x) · φᵢ

        where p_row is a ROW of the normalised diffusion matrix P, and φᵢ is
        the RAW (unscaled) eigenvector — NOT the eigenvalue-scaled self._Phi.

        Critical implementation notes
        ------------------------------
        1. **Matrix power vs element-wise power.**
           LA.matrix_power(K, t) mixes all rows and columns — there is no
           cheap single-row equivalent. For the EXACT path (use_exact_rows=True),
           we read rows directly from the stored _Pmatrix, avoiding approximation.
           For the APPROXIMATE path (new data), we use element-wise k_row**t.
           The approximation error grows with t_horizon; at t_horizon=1 it is
           exact because K^1 = K and the row of K is trivially computable.

        2. **Unscaled eigenvectors.**
           self._Phi = eigenvectors * |Λ| (Eq. 14, eigenvalue-scaled).
           The Nyström formula requires self._eigenvectors (unscaled).
           Using self._Phi would introduce a spurious double-scaling by λᵢ².

        Parameters
        ----------
        X_new : np.ndarray, shape (N, T_new)
            Timepoints to embed. Must already be validated (float32).
        use_exact_rows : bool
            If True, read p_row directly from self._Pmatrix for each
            timepoint (assumes X_new IS the validated training data).
            If False, compute p_row via the approximate kernel formula.

        Returns
        -------
        Z : np.ndarray, shape (k, T_new)
        """
        X_train = self._X_fit        # (N, Tm)  validated training data
        T_new   = X_new.shape[1]
        Z       = np.zeros((self.k, T_new))

        for t in range(T_new):

            # ── EXACT PATH: use stored Pmatrix row ────────────────────────
            # Valid when X_new is the training data (force_nystrom=True).
            # Reads the exact normalised diffusion row computed during fit(),
            # which is what LA.matrix_power + D^{-1} produced — no approximation.
            if use_exact_rows:
                p_row = self._Pmatrix[t, :]                 # (Tm,) exact row

            # ── APPROXIMATE PATH: genuinely new timepoint ─────────────────
            else:
                x = X_new[:, t]                             # (N,)

                # Eq. (10): kernel row  K[j] = exp(i * ||x - x_j||^2 / sigma)
                diffs2 = np.sum((X_train - x[:, None]) ** 2, axis=0)  # (Tm,)
                k_row  = np.exp(1j * diffs2 / self.epsilon)            # (Tm,) complex

                # Eq. (11) approximation: element-wise power.
                # Exact would require O(Tm^2 * N) to recompute the full matrix.
                # Error grows with t_horizon; exact when t_horizon=1.
                k_row_t = k_row ** self.t_horizon                      # (Tm,) complex

                # Eq. (11): |K^t|^2
                q_row = np.abs(k_row_t) ** 2                           # (Tm,) real

                # Eq. (12-13): row-normalise  P = D^{-1} Q
                d = q_row.sum()
                if d == 0:
                    warnings.warn(
                        f"Zero row-sum at timepoint t={t} in Nyström kernel. "
                        "This timepoint may lie far outside the training manifold. "
                        "Consider increasing epsilon or checking the input data.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    d = 1.0
                p_row = q_row / d                                      # (Tm,) real

            # Nyström formula (correct, using raw unscaled eigenvectors):
            #   φ̃ᵢ(x) = (1/λᵢ) · p_row · φᵢ   for each latent dim i
            # Vectorised: (Tm,) @ (Tm, k) → (k,)  then / (k,) → (k,)
            # Nyström formula — correct derivation:
            # Phi = Re(V) * |λ|  (Eq. 14), and P @ Re(V) = Re(V) * λ_signed
            # Therefore: P[t,:] @ Phi = (P @ Re(V))[t,:] * |λ|
            #                         = Re(V)[t,:] * λ_signed * |λ|
            # And:        Phi[t,:]    = Re(V)[t,:] * |λ|
            # So:    (P[t,:] @ Phi) / λ_signed = Re(V)[t,:] * |λ| = Phi[t,:]
            # i.e. the correct formula is (p_row @ Phi) / eigenvalues_signed.
            # Using raw eigenvectors or |λ| would both give wrong results.
            Z[:, t] = (p_row @ self._Phi) / self._eigenvalues_signed

        return Z

    # ------------------------------------------------------------------
    # Override score: explained variance is not the natural metric for CHARM
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray) -> float:
        """
        Reconstruction quality via the parcel-space basis (conet).

        Uses the base-class explained variance formula:
            1 - SS_res / SS_tot

        where reconstruction is X_hat = conet @ Z = conet @ conet.T @ X.

        Note: unlike PCA, CHARM does not guarantee that conet columns are
        orthonormal, so this is an approximation of explained variance.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        float
        """
        return super().score(X)

    # ------------------------------------------------------------------
    # Extra: expose the raw timepoint embedding for downstream analysis
    # ------------------------------------------------------------------

    @property
    def embedding_(self) -> np.ndarray:
        """
        Raw timepoint embedding Φ, shape (Tm, k).

        This is the direct output of the eigendecomposition, before the
        parcel-space projection via nets(). Useful for FCD analysis,
        metastability computation, and other timepoint-level analyses
        (``analyze()`` in the original pipeline).
        """
        self._check_is_fitted()
        return self._Phi

    @property
    def eigenvalues_(self) -> np.ndarray:
        """Selected eigenvalues, shape (k,), in descending magnitude order."""
        self._check_is_fitted()
        return self._eigenvalues
