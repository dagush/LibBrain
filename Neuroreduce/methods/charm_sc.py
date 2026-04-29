"""
Neuroreduce/methods/charm_sc.py
---------------------------------
CHARM-SC: Complex HARMonics Structural Connectivity dimensionality reduction.

Uses parcel centroid coordinates (structural geometry) rather than BOLD
timeseries columns to build the complex diffusion kernel. The resulting
eigenvectors form a geometry-driven basis for projecting BOLD signals.

Key difference from CHARMReducer (CHARM-BOLD)
----------------------------------------------
                CHARM-BOLD              CHARM-SC
Kernel input    BOLD columns x_i ∈ ℝᴺ  Parcel centroids c_i ∈ ℝ³
Matrix size     (Tm × Tm)               (N × N)
Needs BOLD      At fit() time           Optional at fit() time
Basis           conet via nets()        Eigenvectors of geometry P,
                                        or conet if BOLD provided
Coords needed   No                      Yes — at construction time

The geometry is fixed (coordinates never change), so coords are passed
at construction time and fit() can be called with no BOLD argument.

If BOLD is provided to fit(X), the parcel-space basis is enriched via
the nets() correlation step (identical to CHARM-BOLD), giving a data-
driven parcel weighting on top of the geometry-driven eigenvectors.

If BOLD is NOT provided, get_basis() returns the raw eigenvectors of
the geometry diffusion matrix — a pure structural basis.

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

    Model_subjects.m / FCmodel.m — original MATLAB code by Gustavo Deco.

Convention recap (Neuroreduce-wide):
    N  : number of brain parcels / ROIs
    T  : number of fMRI timepoints
    k  : number of reduced dimensions

    Coords input : np.ndarray, shape (N, 3)   — parcel centroid coordinates
    BOLD input   : np.ndarray, shape (N, T)   — optional, for conet basis
    Output       : np.ndarray, shape (k, T)
    Basis        : np.ndarray, shape (N, k)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy import linalg as LA

from Neuroreduce.base import DimensionalityReducer
from Neuroreduce.methods.base_charm import BaseCHARMKernel


class CHARMSCReducer(DimensionalityReducer, BaseCHARMKernel):
    """
    CHARM-SC dimensionality reduction using structural parcel geometry.

    Builds a complex diffusion kernel from parcel centroid coordinates
    (rather than BOLD timeseries), extracts the top-k eigenvectors as
    a structural basis, and projects BOLD signals onto that basis.

    Parameters
    ----------
    k : int
        Number of latent dimensions to retain.
    coords : np.ndarray, shape (N, 3)
        Parcel centroid coordinates in 3D space (e.g. MNI/RAS mm).
        Obtained via ``parcellation.get_CoGs()``.
        Fixed at construction — the geometry does not change between calls.
    epsilon : float
        Kernel bandwidth σ. Controls spatial scale of the diffusion.
        Default: 1400.0 (paper value for Schaefer1000 in mm coordinates).
    t_horizon : int
        Diffusion horizon t. Default: 2.
    whiten : bool
        If True, z-score each row of transform() output. Default: False.
    sort_eigenvectors : bool
        If True, sort eigenpairs by descending eigenvalue magnitude.
        Default: True.

    Examples
    --------
    >>> # Pure geometry basis — no BOLD needed
    >>> reducer = CHARMSCReducer(k=7, coords=cog, epsilon=1400)
    >>> reducer.fit()
    >>> Z = reducer.transform(X)          # (7, T)
    >>> W = reducer.get_basis()           # (N, 7)  eigenvectors

    >>> # Data-enriched basis — nets() correlation with BOLD
    >>> reducer = CHARMSCReducer(k=7, coords=cog, epsilon=1400)
    >>> reducer.fit(X)
    >>> Z = reducer.transform(X)          # (7, T)
    >>> W = reducer.get_basis()           # (N, 7)  conet (like CHARM-BOLD)

    Notes
    -----
    transform() always uses the linear projection W.T @ X regardless of
    whether the basis came from eigenvectors or conet. No Nyström extension
    is needed because the basis lives in parcel space (N×k), not timepoint
    space — new BOLD is projected directly.

    stationary_distribution() exposes P^diffusion_steps[0,:] — the
    long-run probability of the geometry random walk occupying each parcel.
    """

    def __init__(
        self,
        k:                 int,
        coords:            np.ndarray,
        epsilon:           float = 1400.0,
        t_horizon:         int   = 2,
        whiten:            bool  = False,
        sort_eigenvectors: bool  = True,
        diffusion_steps:   int   = 50,
    ):
        super().__init__(k=k, whiten=whiten)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"coords must have shape (N, 3), got {coords.shape}. "
                "Pass the output of parcellation.get_CoGs()."
            )

        self.coords            = coords.astype(np.float64)
        self.epsilon           = epsilon
        self.t_horizon         = t_horizon
        self.sort_eigenvectors = sort_eigenvectors
        self.diffusion_steps   = diffusion_steps

        # Set during fit()
        self._Phi:                Optional[np.ndarray] = None  # (N, k) scaled
        self._eigenvectors:       Optional[np.ndarray] = None  # (N, k) unscaled
        self._eigenvalues:        Optional[np.ndarray] = None  # (k,)
        self._eigenvalues_signed: Optional[np.ndarray] = None  # (k,)
        self._Pmatrix:            Optional[np.ndarray] = None  # (N, N)
        self._Ptr_t:              Optional[np.ndarray] = None  # (N, N)
        self._conet:              Optional[np.ndarray] = None  # (N, k) or None
        self._bold_fitted:        bool = False  # True if BOLD provided to fit()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X:  Optional[np.ndarray] = None,
        SC: Optional[np.ndarray] = None,
    ) -> "CHARMSCReducer":
        """
        Build the geometry-driven basis from parcel centroid coordinates.

        Parameters
        ----------
        X : np.ndarray, shape (N, T), optional
            BOLD timeseries. If provided, the basis is enriched via the
            nets() correlation step (same as CHARM-BOLD), giving a data-
            driven parcel weighting on top of the geometry eigenvectors.
            If None, the basis is the raw geometry eigenvectors.
        SC : ignored.
            Present only for base-class interface compatibility.
            Coordinates were provided at construction time.

        Returns
        -------
        self
        """
        N = self.coords.shape[0]
        if self.k >= N:
            raise ValueError(
                f"k={self.k} must be < N={N} (number of parcels)."
            )

        # ── Step 1: build geometry diffusion matrix ────────────────────────
        # Input to kernel: coords directly — rows = parcels ∈ ℝ³
        # d²_ij = ||c_i - c_j||²  (Euclidean distance between centroids)
        self._Pmatrix, self._Ptr_t, _ = self._build_diffusion_matrix(
            self.coords
        )

        # ── Step 2: eigendecompose → geometry-driven basis ─────────────────
        # Phi ∈ ℝ^(N×k): each row is a parcel, each column a latent mode.
        # This is directly in parcel space — no nets() needed unless
        # BOLD is provided for enrichment.
        (self._Phi,
         self._eigenvectors,
         self._eigenvalues,
         self._eigenvalues_signed) = self._eigendecompose(self._Pmatrix)

        # ── Step 3: optional BOLD enrichment via nets() ────────────────────
        if X is not None:
            X = self._validate_input(X)
            if X.shape[0] != N:
                raise ValueError(
                    f"X has {X.shape[0]} parcels but coords has {N}."
                )
            # For CHARM-SC, Phi is (N×k) — rows are PARCELS, not timepoints.
            # The standard _nets() mixin assumes rows = timepoints, so we
            # use a dedicated correlation: for each latent dim d and parcel i,
            # conet[i,d] = corr( Phi[:,d], X[i,:] ) — correlation of the
            # geometry mode across parcels with the BOLD timeseries of parcel i.
            # Phi[:,d] (N,) acts as a spatial weight vector; X[i,:] (T,) is
            # the BOLD for parcel i. The result (N×k) is then L2-normalised.
            from scipy import stats as _stats
            zPhi  = _stats.zscore(self._Phi, axis=0, ddof=1)   # (N, k) z-score across parcels
            zBOLD = _stats.zscore(X,         axis=1, ddof=1)   # (N, T) z-score across time
            # corr[d, i] = dot(zPhi[:,d], zBOLD[i,:]) — but dimensions mismatch
            # We want conet[i,d] = sum_j zPhi[j,d] * (spatial weight on parcel i)
            # The meaningful correlation: conet2[i,d] = corr(Phi[:,d], X[i,:])
            # i.e. how well does the geometry mode d predict the BOLD of parcel i?
            # zPhi is (N,k), zBOLD.T is (T,N) — we need (N,k).T @ (N,N)... not right.
            # Correct: for each d, compute corr between Phi[:,d] (length N vector)
            # and X[i,:] (length T vector) — these have different lengths.
            # The right interpretation: correlate geometry eigenvalue d with BOLD
            # across parcels at each timepoint, then average across time.
            # conet[i,d] = corr_over_time( sum_j Phi[j,d]*X[j,t], X[i,t] )
            # Simplest correct implementation: project BOLD onto geometry modes
            # to get Z=(k,T), then correlate Z[d,:] with X[i,:] across time.
            Z_geom   = self._Phi.T @ X                         # (k, T)
            zZ       = _stats.zscore(Z_geom, axis=1, ddof=1)   # (k, T)
            zX       = _stats.zscore(X,      axis=1, ddof=1)   # (N, T)
            # conet2[i,d] = corr(zZ[d,:], zX[i,:]) across T timepoints
            conet2   = (zX @ zZ.T) / (X.shape[1] - 1)          # (N, k)
            norms    = np.linalg.norm(conet2, axis=0, keepdims=True)
            norms    = np.where(norms == 0, 1.0, norms)
            self._conet     = conet2 / norms
            self._bold_fitted = True
        else:
            # Pure geometry basis: normalise eigenvector columns to unit norm
            # so get_basis() always returns unit-norm columns (N×k).
            norms           = LA.norm(self._Phi, axis=0, keepdims=True)
            norms           = np.where(norms == 0, 1.0, norms)
            self._conet     = self._Phi / norms
            self._bold_fitted = False

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project BOLD into the geometry-driven latent space.

        Uses direct linear projection: Z = W.T @ X
        where W = get_basis() (N×k).

        No Nyström extension is needed — unlike CHARM-BOLD, the basis
        lives in parcel space (N×k) and new BOLD is projected directly.
        This makes CHARM-SC transform() exact and O(N·k·T), not O(Tm²).

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        Z : np.ndarray, shape (k, T)
        """
        self._check_is_fitted()
        X = self._validate_input(X)
        # W.T @ X  where W = (N, k)  →  Z = (k, T)
        Z = self._conet.T @ X
        return self._apply_whitening(Z)

    def get_basis(self) -> np.ndarray:
        """
        Return the parcel-space basis matrix.

        If BOLD was provided to fit(): returns conet (N×k), the
        correlation-enriched basis (same as CHARM-BOLD).

        If BOLD was NOT provided: returns unit-norm eigenvectors of the
        geometry diffusion matrix P (N×k) — a pure structural basis.

        Returns
        -------
        W : np.ndarray, shape (N, k)
        """
        self._check_is_fitted()
        return self._conet

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Approximate reconstruction of BOLD from geometry latent coordinates.

        X_hat = W @ Z   where W = get_basis() (N×k)

        Caveats (same as CHARMReducer.inverse_transform)
        -------------------------------------------------
        - W columns are unit-norm but NOT orthogonal → not a proper
          orthogonal projection. Use check_reconstruction_quality()
          to verify X_hat is usable for your downstream analysis.
        - If BOLD was used in fit(), W = conet has the same non-orthogonal
          structure as CHARM-BOLD. If BOLD was NOT used, W = eigenvectors
          which are closer to orthonormal but still not guaranteed.

        Parameters
        ----------
        Z : np.ndarray, shape (k, T)

        Returns
        -------
        X_hat : np.ndarray, shape (N, T)
        """
        self._check_is_fitted()
        return self._conet @ Z

    def score(self, X: np.ndarray) -> float:
        """
        Explained variance of the geometry-driven reconstruction.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        float
            Explained variance ratio in [0, 1] (higher is better).
        """
        return super().score(X)

    # ------------------------------------------------------------------
    # Extra: stationary distribution (useful for Model_subjects comparison)
    # ------------------------------------------------------------------

    @property
    def stationary_distribution_(self) -> np.ndarray:
        """
        Stationary distribution of the geometry random walk.

        Computes P^diffusion_steps and returns the first row — the
        long-run probability of the random walker occupying each parcel.

        This is the quantity compared against empirical parcel activation
        distributions in Model_subjects.m and FCmodel.m.

        Returns
        -------
        p_states : np.ndarray, shape (N,)
            Probability of each parcel in the stationary distribution.
            Sums to approximately 1.
        """
        self._check_is_fitted()
        P_n = LA.matrix_power(self._Pmatrix, self.diffusion_steps)
        return P_n[0, :]

    @property
    def bold_fitted(self) -> bool:
        """True if BOLD was provided to fit(), enriching the basis via nets()."""
        return self._bold_fitted

    @property
    def embedding_(self) -> np.ndarray:
        """
        Geometry eigenvector embedding Φ, shape (N, k).

        Each row is a parcel, each column is one latent dimension.
        Unlike CHARM-BOLD where Φ is (Tm×k) in timepoint space,
        here Φ is already in parcel space.
        """
        self._check_is_fitted()
        return self._Phi

    @property
    def eigenvalues_(self) -> np.ndarray:
        """Selected eigenvalues |λ|, shape (k,), descending order."""
        self._check_is_fitted()
        return self._eigenvalues

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        bold   = "with BOLD" if self._bold_fitted else "geometry only"
        N      = self.coords.shape[0]
        return (f"CHARMSCReducer(k={self.k}, N={N}, "
                f"epsilon={self.epsilon}, t_horizon={self.t_horizon}) "
                f"[{status}, {bold}]")
