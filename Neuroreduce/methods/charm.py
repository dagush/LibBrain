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

Note: the code refers as "Classical CHARM" to the traditional Harmonics, and
      "Quantum" or "Quantum CHARM" to the actual CHARM technique.

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
from Neuroreduce.methods.base_charm import BaseCHARMKernel


class CHARMReducer(DimensionalityReducer, BaseCHARMKernel):
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
        kernel_type: str = 'quantum',
    ):
        """
        Parameters
        ----------
        k : int
            Number of latent dimensions (LATDIM in the paper).
        epsilon : float
            Kernel bandwidth σ. Paper defaults: 300 (quantum), 400 (classical).
        t_horizon : int
            Diffusion horizon τ. Paper defaults: 2 (quantum), 1 (classical).
            For quantum: K is raised to this matrix power before |·|².
            For classical: τ enters only via eigenvalue scaling (λ^τ) and
            the Nyström denominator (Λ^{-τ}) — not the kernel itself.
        whiten : bool
            If True, z-score each row of the projected output across time.
        sort_eigenvectors : bool
            Sort eigenpairs by descending |λ| before selecting top-k.
        kernel_type : {'quantum', 'classical'}
            Which kernel variant to use:

            ``'quantum'`` (default):
                K[i,j] = exp(i · d²/σ), then Ptr = |K^τ|².
                Full embedding scales Φ by |λ|.
                Nyström denominator: λ  (no τ power).
                Paper params: ε=300, τ=2.

            ``'classical'``:
                K[i,j] = exp(-d²/σ), real and symmetric.
                No matrix power — τ only scales eigenvalues.
                Full embedding scales Φ by λ^τ.
                Nyström denominator: λ^τ.
                Paper params: ε=400, τ=1.
        """
        super().__init__(k=k, whiten=whiten)
        self.epsilon = epsilon
        self.t_horizon = t_horizon
        self.sort_eigenvectors = sort_eigenvectors

        if kernel_type not in ('quantum', 'classical'):
            raise ValueError(
                f"kernel_type must be 'quantum' or 'classical', got {kernel_type!r}"
            )
        self.kernel_type = kernel_type

        # Set during fit
        self._X_fit_original: Optional[np.ndarray] = None  # pre-validation ref for identity check
        self._X_fit: Optional[np.ndarray] = None           # validated (float32) copy
        self._Phi: Optional[np.ndarray] = None             # (Tm, k) eigenvalue-scaled embedding
        self._eigenvectors: Optional[np.ndarray] = None    # (Tm, k) raw (unscaled) eigenvectors
        self._eigenvalues: Optional[np.ndarray] = None     # (k,) |λ| magnitudes
        self._eigenvalues_signed: Optional[np.ndarray] = None   # (k,) signed λ
        self._eigenvalues_nystrom: Optional[np.ndarray] = None  # (k,) Nyström denominator:
        # _eigenvalues_nystrom = λ^τ (classical) or λ (quantum).
        # Separating this from _eigenvalues_signed avoids confusion:
        # the Nyström formula always divides by _eigenvalues_nystrom,
        # but the exponent differs between kernel types.
        self._conet: Optional[np.ndarray] = None           # (N, k) parcel-space basis
        self._Pmatrix: Optional[np.ndarray] = None         # (Tm, Tm) row-normalised diffusion matrix
        self._Ptr_t: Optional[np.ndarray] = None           # (Tm, Tm) raw kernel before normalisation:
        # For quantum: |K^τ|²    For classical: K (real Gaussian)
        # Either way, _Ptr_t[T_tr:, :T_tr] is the correct cross-block for
        # evaluate_fc_cv() — the left-multiplier in the Nyström reconstruction.

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
        # _latent() returns 7 values; eigenvalues_nystrom is the correct
        # Nyström denominator for this kernel type (λ^τ classical, λ quantum).
        (self._Phi,
         self._eigenvectors,
         self._eigenvalues,
         self._eigenvalues_signed,
         self._eigenvalues_nystrom,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            their eigenvalues (Eq. 14 for quantum; λ^τ scaling for classical).
        eigenvectors_k : np.ndarray, shape (Tm, k)
            Raw (unscaled) eigenvectors — used by Nyström.
        eigenvalues_k : np.ndarray, shape (k,)
            Absolute eigenvalue magnitudes |λ|.
        eigenvalues_k_signed : np.ndarray, shape (k,)
            Signed real eigenvalues λ.
        eigenvalues_nystrom : np.ndarray, shape (k,)
            Correct Nyström denominator for this kernel type:
            λ^τ (classical) or λ (quantum).
        Pmatrix : np.ndarray, shape (Tm, Tm)
            Row-normalised diffusion matrix (stored for Nyström extension).
        Ptr_t : np.ndarray, shape (Tm, Tm)
            Raw kernel before normalisation: K (classical) or |K^τ|² (quantum).
            Cross-block Ptr_t[T_tr:, :T_tr] is the left-multiplier in
            evaluate_fc_cv() for both kernel types.
        """
        # Delegates to BaseCHARMKernel shared methods.
        # Input: ts.T gives (Tm, N) — rows = timepoints, matching
        # the convention d²_ij = ||x_i - x_j||² between BOLD columns.
        Pmatrix, Ptr_t, _ = self._build_diffusion_matrix(
            ts.T, kernel_type=self.kernel_type,
        )

        # Eigenvalue scaling differs between kernel types:
        #   quantum  → Φ[:,d] *= |λ_d|      ('abs')
        #   classical → Φ[:,d] *= λ_d^τ     ('power')
        eigenvalue_scale = 'power' if self.kernel_type == 'classical' else 'abs'

        Phi, eigenvectors_k, eigenvalues_k, eigenvalues_k_signed = \
            self._eigendecompose(Pmatrix, eigenvalue_scale=eigenvalue_scale)

        # Nyström denominator — differs between kernel types:
        #   quantum  → divide by λ    (no extra power)
        #   classical → divide by λ^τ
        if self.kernel_type == 'classical':
            eigenvalues_nystrom = eigenvalues_k_signed ** self.t_horizon
        else:
            eigenvalues_nystrom = eigenvalues_k_signed  # quantum: just λ

        return Phi, eigenvectors_k, eigenvalues_k, eigenvalues_k_signed, \
               eigenvalues_nystrom, Pmatrix, Ptr_t


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
        # Delegates to BaseCHARMKernel._nystrom_transform_shared()
        # _eigenvalues_nystrom is the correct Nyström denominator:
        #   classical: λ^τ    quantum: λ
        return self._nystrom_transform_shared(
            X_new              = X_new,
            X_fit              = self._X_fit,
            Phi                = self._Phi,
            eigenvalues_signed = self._eigenvalues_nystrom,
            Pmatrix            = self._Pmatrix,
            is_same_data       = False,
            use_exact_rows     = use_exact_rows,
        )

    # ------------------------------------------------------------------
    # CV BOLD reconstruction and FC quality (MATLAB CV block)
    # ------------------------------------------------------------------

    def evaluate_fc_cv(
        self,
        X:       np.ndarray,
        t_train: int,
    ) -> dict:
        """
        Reconstruct held-out BOLD and measure FC quality (the MATLAB CV block).

        Must be called AFTER ``fit(X)`` — this method reuses ``_Ptr_t``
        (the full Tm×Tm kernel computed during fit) to extract the cross-block
        cheaply, avoiding a second O(Tm³) kernel build.

        Algorithm
        ---------
        1. Extract training block of the stored kernel:
               ``block_tr = _Ptr_t[:t_train, :t_train]``
           Row-normalise to get ``P_tr``.

        2. Eigendecompose ``P_tr`` with the same ``k`` and ``eigenvalue_scale``
           as the full fit.  This gives ``Phi_tr`` (T_tr × k) and ``lambda_k``.

        3. Build the Nyström reconstruction matrix:
               ``A = Phi_tr @ Λ_inv @ Phi_tr.T``    (T_tr × T_tr)
           where:
               Classical:  ``Λ_inv = diag(1 / λ^τ)``
               Quantum:    ``Λ_inv = diag(1 / λ)``

        4. Reconstruct held-out BOLD for all N parcels simultaneously:
               ``X_est = (cross_block @ (A @ X_train.T)).T``  (N × T_test)
           where ``cross_block = _Ptr_t[t_train:, :t_train]``  (T_test × T_tr).
           For classical, this is the raw K cross-block (matching MATLAB's
           ``Pcv = Kmatrix(Ttrain+1:end, 1:Ttrain)``).
           For quantum, this is the |K^τ|² cross-block.

        5. Compute FC on held-out data (corrcoef over parcels):
               ``FC_true = corrcoef(X[:, t_train:])``
               ``FC_est  = corrcoef(X_est)``
           and report lower-triangular Pearson r and MSE.

        Why not use transform() + inverse_transform()?
        ------------------------------------------------
        ``transform()`` projects into embedding space via Nyström and
        ``inverse_transform()`` maps back via ``conet @ Z``.  That path uses
        the full-data eigenvectors (Tm modes) and the correlation-based ``conet``
        bridge.  This method instead fits eigenvectors on the training block only
        and directly reconstructs parcel-space BOLD — matching the MATLAB CV
        block exactly.  The two approaches measure different things:
          - ``inverse_transform``: embedding quality of the full-data manifold
          - ``evaluate_fc_cv``:    out-of-sample BOLD prediction quality

        Parameters
        ----------
        X       : np.ndarray, shape (N, Tm)
            The same BOLD passed to ``fit()``.  Used to split train/test and
            to compute FC.  Must NOT be re-preprocessed.
        t_train : int
            Number of training timepoints.  Must satisfy 0 < t_train < Tm.

        Returns
        -------
        dict with keys:
            'corr_fit'  : float — Pearson r between lower-tri of FC_true and FC_est
            'err_fit'   : float — MSE between lower-tri of FC_true and FC_est
            'fc_true'   : np.ndarray, shape (N, N) — FC on held-out timepoints
            'fc_est'    : np.ndarray, shape (N, N) — FC from reconstructed BOLD
        """
        self._check_is_fitted()
        X = self._validate_input(X)
        N, Tm = X.shape

        if not (0 < t_train < Tm):
            raise ValueError(
                f"t_train={t_train} must be strictly between 0 and Tm={Tm}."
            )

        # ------------------------------------------------------------------
        # 1. Training block of the stored kernel → P_tr
        #    _Ptr_t is (Tm, Tm): K for classical, |K^τ|² for quantum.
        #    Subblock extraction is O(T_tr²), no new kernel build needed.
        # ------------------------------------------------------------------
        block_tr = self._Ptr_t[:t_train, :t_train]          # (T_tr, T_tr)
        row_sums = block_tr.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        P_tr     = block_tr / row_sums[:, None]              # (T_tr, T_tr)

        # ------------------------------------------------------------------
        # 2. Eigendecompose P_tr — same k and scale as full fit.
        #    We want UNSCALED eigenvectors for the Nyström formula, but
        #    eigenvalues need the correct denominator power.
        # ------------------------------------------------------------------
        eigenvalue_scale = 'power' if self.kernel_type == 'classical' else 'abs'
        _, eigvecs_tr, _, evals_signed_tr = self._eigendecompose(
            P_tr, eigenvalue_scale=eigenvalue_scale,
        )
        # eigvecs_tr      : (T_tr, k)  unscaled eigenvectors of P_tr
        # evals_signed_tr : (k,)       signed real eigenvalues of P_tr

        # ------------------------------------------------------------------
        # 3. Nyström reconstruction matrix A = Φ_tr @ Λ_inv @ Φ_tr.T
        #
        #    Classical:  Λ_inv = diag(1 / λ^τ)   MATLAB: inv(LL.^Thorizont)
        #    Quantum:    Λ_inv = diag(1 / λ)      MATLAB: inv(LL)  (no τ)
        # ------------------------------------------------------------------
        if self.kernel_type == 'classical':
            lambda_denom = evals_signed_tr ** self.t_horizon   # λ^τ
        else:
            lambda_denom = evals_signed_tr                     # λ

        Lambda_inv = np.diag(1.0 / lambda_denom)              # (k, k)
        A          = eigvecs_tr @ Lambda_inv @ eigvecs_tr.T    # (T_tr, T_tr)

        # ------------------------------------------------------------------
        # 4. Vectorised reconstruction over all N parcels:
        #        X_est = cross_block @ (A @ X_train.T)
        #
        #    cross_block : (T_test, T_tr)  — right cross-block of _Ptr_t
        #    A @ X_train.T : (T_tr, N)    — precomputed once, not per-parcel
        #    Result before transpose : (T_test, N) → X_est : (N, T_test)
        #
        #    MATLAB (per-parcel loop, r=1..N):
        #        tscvestimated = Pcv * Phi * inv(LAMBDA) * Phi' * ts(r,1:Ttrain)'
        #    Python (vectorised):
        #        X_est.T = cross_block @ A @ X_train.T
        # ------------------------------------------------------------------
        cross_block = self._Ptr_t[t_train:, :t_train]         # (T_test, T_tr)
        X_train     = X[:, :t_train]                           # (N, T_tr)
        X_est       = (cross_block @ (A @ X_train.T)).T        # (N, T_test)

        # ------------------------------------------------------------------
        # 5. FC on held-out data and reconstruction quality
        # ------------------------------------------------------------------
        FC_true = np.corrcoef(X[:, t_train:])                  # (N, N)
        FC_est  = np.corrcoef(X_est)                           # (N, N)

        r_idx, c_idx = np.tril_indices(N, k=-1)
        fc_true_lt   = FC_true[r_idx, c_idx]
        fc_est_lt    = FC_est [r_idx, c_idx]
        corr_fit     = float(np.corrcoef(fc_true_lt, fc_est_lt)[0, 1])
        err_fit      = float(np.mean((fc_true_lt - fc_est_lt) ** 2))

        return {
            'corr_fit' : corr_fit,
            'err_fit'  : err_fit,
            'fc_true'  : FC_true,
            'fc_est'   : FC_est,
        }

    # ------------------------------------------------------------------
    # Override inverse_transform: explicit caveats vs PCA
    # ------------------------------------------------------------------

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Approximate reconstruction of BOLD from CHARM latent coordinates.

        Computes:  X_hat = conet @ Z   (N×k) @ (k×T) → (N×T)

        Important caveats — read before using
        ----------------------------------------
        1. **conet is not orthonormal.**
           conet columns are unit-norm (L2-normalised) but not mutually
           orthogonal. Therefore conet.T @ conet ≠ I, and this is NOT
           a proper orthogonal projection. PCA's inverse_transform is
           exact in the least-squares sense; CHARM's is not.

        2. **Nyström compounds the approximation.**
           If Z was produced by transform(X_new) on genuinely new data,
           it carries Nyström approximation error. Feeding it into this
           function compounds that error with the non-orthogonal basis.
           Always call score(X_new) afterwards to quantify the quality.

        3. **Semantic difference from PCA.**
           In PCA, Z lives in the same geometric space as X (just
           lower-dimensional). In CHARM, Z = Phi is a diffusion manifold
           embedding — conet is a correlation-based bridge back to parcel
           space, not a geometric inverse. Think of X_hat as "the BOLD
           pattern associated with these manifold coordinates" rather than
           "the original signal".

        Use check_reconstruction_quality() to get an explicit quality
        report before trusting X_hat for downstream analysis.

        Parameters
        ----------
        Z : np.ndarray, shape (k, T)
            Latent coordinates — output of transform().

        Returns
        -------
        X_hat : np.ndarray, shape (N, T)
            Approximate BOLD reconstruction in parcel space.
        """
        self._check_is_fitted()
        # Delegate to base class: W @ Z where W = conet (N, k)
        return self._conet @ Z

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
        Typical values are lower than PCA for the same k — this is expected
        and does not indicate a bug.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        float
            Explained variance ratio. Values below ~0.1 suggest the
            reconstruction is poor and should not be used for analysis
            that depends on X_hat amplitude (e.g. ECM comparison).
        """
        return super().score(X)

    def check_reconstruction_quality(
        self,
        X:                np.ndarray,
        warn_threshold:   float = 0.1,
    ) -> dict:
        """
        Compute and report reconstruction quality metrics for X.

        Runs both the reconstruction and several quality metrics, prints
        a human-readable summary, and warns if quality is below threshold.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)
            BOLD signal to reconstruct and compare against.
        warn_threshold : float
            Emit a RuntimeWarning if explained variance falls below this.
            Default: 0.1 (10%).

        Returns
        -------
        dict with keys:
            'explained_variance' : float   — 1 - SS_res/SS_tot
            'pearson_r'          : float   — correlation(X_flat, X_hat_flat)
            'conet_orthogonality': float   — ||conet.T @ conet - I||_F
                                             (0 = orthonormal, like PCA)
        """
        from scipy import stats as _stats

        self._check_is_fitted()
        X       = self._validate_input(X)
        Z       = self.transform(X)
        X_hat   = self.inverse_transform(Z)

        # Explained variance
        ev = self.score(X)

        # Pearson r between original and reconstructed (flattened)
        r, _ = _stats.pearsonr(X.ravel(), X_hat.ravel())

        # How far conet is from orthonormal
        # Perfect orthonormality (PCA): ||W.T @ W - I||_F = 0
        W   = self._conet   # (N, k)
        WtW = W.T @ W       # (k, k)
        orth_err = float(np.linalg.norm(WtW - np.eye(self.k), 'fro'))

        print(f"\nCHARM reconstruction quality  (k={self.k})")
        print(f"  Explained variance : {ev:.4f}  "
              f"({'good' if ev > 0.3 else 'moderate' if ev > 0.1 else 'poor'})")
        print(f"  Pearson r          : {r:.4f}")
        print(f"  conet orthogonality error : {orth_err:.4f}  "
              f"(PCA = 0.0, CHARM typically > 0)")

        if ev < warn_threshold:
            warnings.warn(
                f"CHARM reconstruction quality is poor: explained variance = "
                f"{ev:.4f} < threshold {warn_threshold:.4f}. "
                "X_hat may not be suitable for analyses that depend on "
                "amplitude fidelity (e.g. ECM comparison). "
                "Consider increasing k or checking your epsilon/t_horizon.",
                RuntimeWarning,
                stacklevel=2,
            )

        return {
            'explained_variance':  ev,
            'pearson_r':           float(r),
            'conet_orthogonality': orth_err,
        }

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
