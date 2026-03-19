"""
neuroreduce/base.py
-------------------
Abstract base class for all dimensionality reduction methods.

Notation (enforced at every interface boundary):
    N : number of brain parcels / ROIs
    T : number of fMRI timepoints
    k : number of reduced dimensions

    BOLD input  : np.ndarray, shape (N, T)  — float32 or float64
    SC input    : np.ndarray, shape (N, N)  — symmetric, float32 or float64
    Output      : np.ndarray, shape (k, T)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DimensionalityReducer(ABC):
    """
    Abstract base class for structural connectivity and fMRI dimensionality
    reduction methods.

    All subclasses share the same external contract:

        reducer = SomeReducer(k=10, **method_specific_kwargs)
        reducer.fit(X)           # X : (N, T)  — learns the basis
        Z = reducer.transform(X) # Z : (k, T)  — projects into reduced space
        Z = reducer.fit_transform(X)             # convenience: fit then transform

    Optionally, if the method supports it:
        X_hat = reducer.inverse_transform(Z)     # (N, T) reconstruction
        W     = reducer.get_basis()              # (N, k) basis / dictionary
        score = reducer.score(X)                 # scalar goodness-of-fit

    Parameters
    ----------
    k : int
        Number of dimensions to retain.
    whiten : bool
        If True, each component of the projected signal is z-scored across
        time after projection (zero mean, unit variance per row).
        Applied inside transform(). Default: False.
    """

    def __init__(self, k: int, whiten: bool = False):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.whiten = whiten
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core interface — every subclass MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray, SC: Optional[np.ndarray] = None) -> "DimensionalityReducer":
        """
        Learn the low-dimensional basis from data.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)
            BOLD fMRI timeseries. Required by all methods.
        SC : np.ndarray, shape (N, N), optional
            Structural connectivity matrix. Required by graph-based methods
            (e.g. Connectome Harmonics); ignored by purely signal-based ones.

        Returns
        -------
        self
            Returns the fitted instance to allow method chaining.
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project BOLD data into the learned low-dimensional space.

        Parameters
        ----------
        X : np.ndarray, shape (N, T)
            BOLD fMRI timeseries.

        Returns
        -------
        Z : np.ndarray, shape (k, T)
            Projected signal in the reduced space.
        """
        ...

    @abstractmethod
    def get_basis(self) -> np.ndarray:
        """
        Return the learned basis vectors (columns) in the original ROI space.

        Returns
        -------
        W : np.ndarray, shape (N, k)
            Each column is one basis vector (principal component, harmonic,
            centroid, decoder direction, …). Interpretation is
            method-specific; see subclass docstrings.
        """
        ...

    # ------------------------------------------------------------------
    # Optional interface — subclasses may override these
    # ------------------------------------------------------------------

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct BOLD data from the reduced representation.

        Default implementation: linear back-projection via the basis matrix.
        Override in subclasses where reconstruction is non-linear (e.g.
        autoencoder decoder) or not well-defined (e.g. LEiDA cluster labels).

        Parameters
        ----------
        Z : np.ndarray, shape (k, T)
            Projected signal.

        Returns
        -------
        X_hat : np.ndarray, shape (N, T)
            Reconstructed BOLD signal.
        """
        self._check_is_fitted()
        W = self.get_basis()           # (N, k)
        return W @ Z                   # (N, T)

    def score(self, X: np.ndarray) -> float:
        """
        Compute a goodness-of-fit scalar for the fitted model on X.

        Default: fraction of variance explained by the reconstruction.
        Override in subclasses where a more natural metric exists (e.g.
        cluster stability for LEiDA, log-likelihood for probabilistic methods).

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        float
            Explained variance ratio in [0, 1] (higher is better).
        """
        self._check_is_fitted()
        X = self._validate_input(X)
        X_hat = self.inverse_transform(self.transform(X))
        ss_res = np.sum((X - X_hat) ** 2)
        ss_tot = np.sum((X - X.mean(axis=1, keepdims=True)) ** 2)
        return float(1.0 - ss_res / ss_tot)

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        SC: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit to X (and optionally SC), then return the projected signal.

        Parameters
        ----------
        X  : np.ndarray, shape (N, T)
        SC : np.ndarray, shape (N, N), optional

        Returns
        -------
        Z : np.ndarray, shape (k, T)
        """
        return self.fit(X, SC=SC).transform(X)

    # ------------------------------------------------------------------
    # Shared helpers — used by subclasses, not part of the public API
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        X: np.ndarray,
        SC: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Validate and coerce inputs to float32.

        Checks
        ------
        - X must be 2-D with shape (N, T), N >= k
        - SC, if provided, must be 2-D, square, and have same N as X
        """
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D with shape (N, T), got shape {X.shape}"
            )
        N, T = X.shape
        if N < self.k:
            raise ValueError(
                f"Number of parcels N={N} is smaller than k={self.k}. "
                "Reduce k or check that X has shape (N, T), not (T, N)."
            )
        if SC is not None:
            if SC.ndim != 2 or SC.shape[0] != SC.shape[1]:
                raise ValueError(
                    f"SC must be a square 2-D matrix, got shape {SC.shape}"
                )
            if SC.shape[0] != N:
                raise ValueError(
                    f"SC has {SC.shape[0]} parcels but X has {N} parcels."
                )
        return X.astype(np.float32, copy=False)

    def _validate_SC(self, SC: np.ndarray, N: int) -> np.ndarray:
        """Standalone SC validator for graph-based subclasses."""
        if SC is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a structural connectivity "
                "matrix SC with shape (N, N). Pass SC= to fit() or fit_transform()."
            )
        if SC.ndim != 2 or SC.shape != (N, N):
            raise ValueError(
                f"SC must have shape ({N}, {N}), got {SC.shape}"
            )
        return SC.astype(np.float32, copy=False)

    def _check_is_fitted(self) -> None:
        """Raise a clean error if transform/score is called before fit."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call fit() or fit_transform() before transform()."
            )

    def _apply_whitening(self, Z: np.ndarray) -> np.ndarray:
        """
        Z-score each row of Z across time (zero mean, unit variance).
        Only applied when self.whiten is True.

        Parameters
        ----------
        Z : np.ndarray, shape (k, T)

        Returns
        -------
        Z_w : np.ndarray, shape (k, T)
        """
        if not self.whiten:
            return Z
        mu = Z.mean(axis=1, keepdims=True)
        sigma = Z.std(axis=1, keepdims=True)
        # Avoid division by zero for constant components
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (Z - mu) / sigma

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(k={self.k}, whiten={self.whiten}) [{status}]"
