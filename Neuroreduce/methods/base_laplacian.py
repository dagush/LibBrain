"""
Neuroreduce/methods/base_laplacian.py
---------------------------------------
BaseLaplacianReducer: shared Laplacian and harmonic mathematics for
Connectome Harmonics and Functional Harmonics.

Both methods share an identical pipeline:
    Step 1 — Threshold and symmetrise the input matrix → adjacency A
    Step 2 — Compute degree matrix D
    Step 3 — Compute Laplacian L (unnormalised or symmetric)
    Step 4 — Eigendecompose L via eigh (real, symmetric → sorted automatically)
    Step 5 — Store the k lowest-frequency eigenvectors as the harmonic basis

The variants differ only in what feeds into the pipeline:
    Connectome Harmonics : SC matrix  (structural connectivity)
    Functional Harmonics : FC matrix  (derived from BOLD timeseries)

References
----------
Atasoy, S., Donnelly, I., & Pearce, J. (2016). Human brain networks function
in connectome-specific harmonic waves. Nature Communications, 7, 10340.

Glomb, K., et al. (2021). Functional harmonics reveal multi-dimensional basis
functions underlying cortical organization. Cell Reports, 36(8).

Vohryzek, J., et al. (2024). Harmonic modes of neural activity in the resting
state. NeuroImage.

Design note
-----------
This is NOT a mixin — it is a full abstract DimensionalityReducer subclass.
Subclasses only need to implement ``_get_input_matrix(X, SC)`` which extracts
the matrix to feed into the Laplacian pipeline.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
from numpy import linalg as LA

from Neuroreduce.base import DimensionalityReducer


class BaseLaplacianReducer(DimensionalityReducer):
    """
    Abstract base class for graph-harmonic dimensionality reduction.

    Computes the k lowest-frequency eigenvectors of the graph Laplacian
    of a connectivity matrix (SC or FC) and uses them as the harmonic basis
    for projecting BOLD timeseries.

    Parameters
    ----------
    k : int
        Number of harmonics (eigenvectors) to retain.
        k=1 gives the DC component (constant); k>1 adds progressively
        higher-frequency spatial modes.
    threshold : float
        Values in the connectivity matrix at or below this threshold are
        set to zero before computing the Laplacian. This removes spurious
        weak connections.
        Default: 0.00065 (value used in the student's original code).
    laplacian_type : str
        Which Laplacian to compute:
        - 'unnormalised' : L = D - A  (default, matches student's code)
        - 'symmetric'    : L_sym = D^{-1/2} L D^{-1/2}
        Default: 'unnormalised'.
    normalise_input : bool
        If True, divide the input matrix by its maximum value before
        computing the Laplacian. This ensures edge weights are in [0,1].
        Default: True.
    remove_self_connections : bool
        If True, zero out the diagonal of the input matrix before
        computing the Laplacian (removes self-loops).
        Default: True.
    whiten : bool
        If True, z-score each row of transform() output. Default: False.

    Notes on eigenvector sign convention
    -------------------------------------
    The Laplacian eigenvectors are defined only up to sign — if v is an
    eigenvector, so is -v. The student's code handles this in transform()
    by taking max(dot(x, phi), dot(-x, phi)) per timepoint, which preserves
    the magnitude of the projection regardless of sign. We preserve this
    behaviour via the ``sign_invariant`` parameter in transform().
    """

    def __init__(
        self,
        k:                        int,
        threshold:                float = 0.00065,
        laplacian_type:           str   = 'unnormalised',
        normalise_input:          bool  = True,
        remove_self_connections:  bool  = True,
        whiten:                   bool  = False,
    ):
        super().__init__(k=k, whiten=whiten)

        if laplacian_type not in ('unnormalised', 'symmetric'):
            raise ValueError(
                f"laplacian_type must be 'unnormalised' or 'symmetric', "
                f"got '{laplacian_type}'."
            )

        self.threshold               = threshold
        self.laplacian_type          = laplacian_type
        self.normalise_input         = normalise_input
        self.remove_self_connections = remove_self_connections

        # Set during fit()
        self._eigenvectors: Optional[np.ndarray] = None  # (N, N_all)
        self._eigenvalues:  Optional[np.ndarray] = None  # (N_all,)
        self._basis:        Optional[np.ndarray] = None  # (N, k) selected

    # ------------------------------------------------------------------
    # Abstract: subclasses define which matrix enters the pipeline
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_input_matrix(
        self,
        X:  Optional[np.ndarray],
        SC: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Extract the (N×N) connectivity matrix to feed into the Laplacian.

        Parameters
        ----------
        X  : np.ndarray, shape (N, T) — BOLD timeseries (may be None)
        SC : np.ndarray, shape (N, N) — structural connectivity (may be None)

        Returns
        -------
        M : np.ndarray, shape (N, N)
            The connectivity matrix to use. Must be square and symmetric.
        """
        ...

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X:  Optional[np.ndarray] = None,
        SC: Optional[np.ndarray] = None,
    ) -> "BaseLaplacianReducer":
        """
        Compute the harmonic basis from the connectivity matrix.

        Parameters
        ----------
        X  : np.ndarray, shape (N, T), optional
            BOLD timeseries. Required by FunctionalHarmonicsReducer (to
            compute FC); ignored by ConnectomeHarmonicsReducer.
        SC : np.ndarray, shape (N, N), optional
            Structural connectivity. Required by ConnectomeHarmonicsReducer;
            ignored by FunctionalHarmonicsReducer.

        Returns
        -------
        self
        """
        # Get the input matrix from the subclass
        M = self._get_input_matrix(X, SC)

        # Validate: must be square
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(
                f"Input matrix must be square (N×N), got {M.shape}."
            )
        N = M.shape[0]
        if self.k > N:
            raise ValueError(f"k={self.k} must be <= N={N}.")

        M = M.astype(np.float64, copy=True)

        # ── Preprocessing ──────────────────────────────────────────────────
        # Normalise to [0, 1]
        if self.normalise_input:
            max_val = np.max(np.abs(M))
            if max_val > 0:
                M /= max_val

        # Remove self-connections (diagonal = self-loops)
        if self.remove_self_connections:
            np.fill_diagonal(M, 0.0)

        # ── Step 1: Adjacency matrix ───────────────────────────────────────
        # Threshold: zero out weak connections
        # Matching the student's LaplacianCalculator.get_adj():
        #   A = copy(M); A[M <= th] = 0; A = max(A, A.T)
        A = np.copy(M)
        A[A <= self.threshold] = 0.0
        A = np.maximum(A, A.T)   # ensure symmetry after thresholding

        # ── Step 2: Degree matrix ──────────────────────────────────────────
        # D_ii = sum_j A_ij  (weighted degree)
        # Matching the student's LaplacianCalculator.get_deg()
        deg = np.sum(A, axis=0)
        D   = np.diag(deg)

        # ── Step 3: Laplacian ──────────────────────────────────────────────
        if self.laplacian_type == 'unnormalised':
            # L = D - A
            # Original paper convention; preserves edge weight magnitude.
            L = D - A

        else:  # 'symmetric'
            # L_sym = D^{-1/2} L D^{-1/2}
            # Makes eigenvalues in [0, 2]; useful for comparing graphs of
            # different sizes / densities.
            # Matching the student's SymmetricLaplacian.get_laplacian()
            L   = D - A
            D2  = np.copy(D)
            for i in range(N):
                if D2[i, i] > 0:
                    D2[i, i] = 1.0 / np.sqrt(D2[i, i])
                else:
                    D2[i, i] = 0.0
            L = D2 @ L @ D2

        # ── Step 4: Eigendecomposition ─────────────────────────────────────
        # eigh assumes real symmetric matrix → returns sorted eigenvalues
        # in ascending order (lowest frequency first).
        # The student sorts again "just to be sure" — we keep that habit.
        e_val, e_vec = np.linalg.eigh(L)
        idx          = np.argsort(e_val)
        e_val        = e_val[idx]
        e_vec        = e_vec[:, idx]

        # Store all eigenpairs (useful for selecting top-k later)
        self._eigenvalues  = e_val              # (N,)
        self._eigenvectors = np.real(e_vec)     # (N, N)

        # ── Step 5: Select k lowest-frequency harmonics ────────────────────
        # Convention: the first eigenvector (eigenvalue ≈ 0) is the DC
        # component (constant vector). The next k-1 give progressively
        # higher-frequency spatial modes.
        self._basis    = self._eigenvectors[:, :self.k]   # (N, k)
        self._is_fitted = True
        return self

    def transform(
        self,
        X:              np.ndarray,
        sign_invariant: bool = True,
    ) -> np.ndarray:
        """
        Project BOLD timeseries onto the harmonic basis.

        For each timepoint t and each harmonic d, computes:

            sign_invariant=True  (default, student's convention):
                beta[d, t] = max( dot(phi_d, X[:,t]),
                                  dot(-phi_d, X[:,t]) )
                           = |dot(phi_d, X[:,t])|   (absolute projection)

                Rationale: eigenvectors are defined up to sign. Taking the
                absolute value gives a sign-invariant projection magnitude
                that is consistent across subjects and sessions.

            sign_invariant=False:
                beta[d, t] = dot(phi_d, X[:,t])

                Use when the sign of the projection matters (e.g. for
                reconstruction via inverse_transform).

        Parameters
        ----------
        X : np.ndarray, shape (N, T)
            BOLD timeseries.
        sign_invariant : bool
            Whether to take the absolute value of each projection.
            Default: True (matching the student's projectVectorTime).

        Returns
        -------
        Z : np.ndarray, shape (k, T)
            Harmonic coefficients over time.
        """
        self._check_is_fitted()
        X = self._validate_input(X)     # (N, T)
        W = self._basis                  # (N, k)

        # Z[d, t] = dot(W[:,d], X[:,t])  →  vectorised: W.T @ X
        Z = W.T @ X                      # (k, T)

        if sign_invariant:
            # Take absolute value to handle eigenvector sign ambiguity.
            # Equivalent to the student's max(dot(x,phi), dot(-x,phi))
            # since dot(-x,phi) = -dot(x,phi), so the max of the two
            # is always the absolute value.
            Z = np.abs(Z)

        return self._apply_whitening(Z)

    def get_basis(self) -> np.ndarray:
        """
        Return the harmonic basis vectors (eigenvectors of the Laplacian).

        Returns
        -------
        W : np.ndarray, shape (N, k)
            Columns are the k lowest-frequency eigenvectors, ordered by
            ascending eigenvalue (ascending spatial frequency).
        """
        self._check_is_fitted()
        return self._basis

    def get_all_eigenvectors(self) -> np.ndarray:
        """
        Return ALL eigenvectors of the Laplacian (not just the top k).

        Useful for the dynamic analysis pipeline where the most relevant
        harmonics are selected AFTER projecting RSN vectors, rather than
        taking the k lowest-frequency ones.

        Returns
        -------
        e_vec : np.ndarray, shape (N, N)
            All eigenvectors, sorted by ascending eigenvalue.
        """
        self._check_is_fitted()
        return self._eigenvectors

    @property
    def eigenvalues_(self) -> np.ndarray:
        """All eigenvalues in ascending order, shape (N,)."""
        self._check_is_fitted()
        return self._eigenvalues

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct BOLD from harmonic coefficients.

        X_hat = W @ Z   where W = get_basis() (N, k)

        Note: only meaningful when transform() was called with
        sign_invariant=False. With sign_invariant=True, the sign
        information is lost and reconstruction will be inaccurate.

        Parameters
        ----------
        Z : np.ndarray, shape (k, T)

        Returns
        -------
        X_hat : np.ndarray, shape (N, T)
        """
        self._check_is_fitted()
        return self._basis @ Z

    def score(self, X: np.ndarray) -> float:
        """
        Explained variance of the k-harmonic reconstruction.

        Uses sign_invariant=False for reconstruction quality assessment.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        float
        """
        self._check_is_fitted()
        X     = self._validate_input(X)
        Z     = self.transform(X, sign_invariant=False)
        X_hat = self.inverse_transform(Z)
        ss_res = np.sum((X - X_hat) ** 2)
        ss_tot = np.sum((X - X.mean(axis=1, keepdims=True)) ** 2)
        return float(1.0 - ss_res / ss_tot)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        N = self._basis.shape[0] if self._is_fitted else "?"
        return (f"{self.__class__.__name__}("
                f"k={self.k}, N={N}, "
                f"threshold={self.threshold}, "
                f"laplacian='{self.laplacian_type}') [{status}]")
