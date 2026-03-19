"""
neuroreduce/methods/pca.py
--------------------------
PCA-based dimensionality reduction of fMRI BOLD signals.

The basis learned here is the set of principal components of the *spatial*
covariance — i.e. we decompose the (N, T) matrix so that each component
captures a spatial mode of variation across parcels.

Convention recap:
    X : (N, T)  — input BOLD
    W : (N, k)  — principal components (columns), i.e. spatial modes
    Z : (k, T)  — PC scores, i.e. temporal expression of each mode
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from Neuroreduce.base import DimensionalityReducer


class PCAReducer(DimensionalityReducer):
    """
    Principal Component Analysis dimensionality reduction for fMRI.

    Decomposes X (N × T) via PCA on the timepoint covariance, retaining the
    k components that explain the most variance.

    Parameters
    ----------
    k : int
        Number of principal components to retain.
    whiten : bool
        If True, z-score each row of Z across time after projection.
        Default: False.
    svd_solver : str
        Solver passed to sklearn.decomposition.PCA. Use 'randomized' for
        large N or large T. Default: 'full'.
    random_state : int or None
        Random seed (only relevant for svd_solver='randomized').

    Notes
    -----
    - sklearn's PCA operates on (samples × features). We pass X.T so that
      timepoints are samples and parcels are features, giving us spatial modes.
    - The returned basis W = components_.T has shape (N, k): each column is a
      spatial map (loadings vector) in parcel space.
    - Z = W.T @ X  —  this recovers the PC scores with shape (k, T).
    - Whitening (if enabled) is applied to Z after projection, not during PCA,
      so the basis W is always the unwhitened eigenvectors.

    Examples
    --------
    >>> reducer = PCAReducer(k=10)
    >>> Z = reducer.fit_transform(X)          # X : (N, T) → Z : (10, T)
    >>> W = reducer.get_basis()               # W : (N, 10)
    >>> evr = reducer.explained_variance_ratio_  # array of length 10
    >>> X_hat = reducer.inverse_transform(Z)  # X_hat : (N, T)
    """

    def __init__(
        self,
        k: int = 30,
        whiten: bool = False,
        svd_solver: str = "full",
        random_state: Optional[int] = None,
    ):
        super().__init__(k=k, whiten=whiten)
        self.svd_solver = svd_solver
        self.random_state = random_state
        self._pca: Optional[PCA] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, SC: Optional[np.ndarray] = None) -> "PCAReducer":
        """
        Fit PCA to X and store the spatial basis.

        Parameters
        ----------
        X  : np.ndarray, shape (N, T)
        SC : ignored — PCA does not use structural connectivity.

        Returns
        -------
        self
        """
        X = self._validate_input(X)   # ensures (N, T) and float32

        self._pca = PCA(
            n_components=self.k,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        # sklearn PCA expects (samples, features) = (T, N)
        self._pca.fit(X.T)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into the PC space.

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
        Z = self.get_basis().T @ X
        return self._apply_whitening(Z)

    def get_basis(self) -> np.ndarray:
        """
        Return the principal component vectors in parcel space.

        Returns
        -------
        W : np.ndarray, shape (N, k)
            Columns are principal components (spatial modes), ordered by
            descending explained variance.
        """
        self._check_is_fitted()
        # sklearn stores components_ as (k, N); transpose to (N, k)
        return self._pca.components_.T

    # ------------------------------------------------------------------
    # PCA-specific extras
    # ------------------------------------------------------------------

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """
        Fraction of total variance explained by each component.

        Returns
        -------
        np.ndarray, shape (k,)
        """
        self._check_is_fitted()
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_explained_variance_(self) -> np.ndarray:
        """
        Cumulative explained variance ratio across components.

        Returns
        -------
        np.ndarray, shape (k,)
        """
        return np.cumsum(self.explained_variance_ratio_)

    def score(self, X: np.ndarray) -> float:
        """
        Fraction of variance explained by the k-component reconstruction.

        Overrides the base class default to use sklearn's analytic result
        (cheaper than computing reconstruction error explicitly).

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        float
            Sum of explained variance ratios for the retained components.
        """
        self._check_is_fitted()
        _ = self._validate_input(X)   # validate shape; result unused here
        return float(self.explained_variance_ratio_.sum())
