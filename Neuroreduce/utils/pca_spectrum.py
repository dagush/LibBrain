"""
Neuroreduce/utils/pca_spectrum.py
----------------------------------
Diagnostic tools for inspecting the PCA explained variance spectrum.

Designed to work with a fitted PCAReducer, so PCA is never computed twice.

Example
-------
>>> from Neuroreduce import PCAReducer
>>> from Neuroreduce.utils.pca_spectrum import PCASpectrumAnalyzer
>>>
>>> reducer = PCAReducer(k=100, svd_solver="randomized")
>>> reducer.fit(X)
>>>
>>> analyzer = PCASpectrumAnalyzer(reducer)
>>> analyzer.report()
>>> analyzer.plot()
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from Neuroreduce.methods.pca import PCAReducer


class PCASpectrumAnalyzer:
    """
    Compute and visualize the PCA explained variance spectrum.

    Operates on a *fitted* PCAReducer, so no redundant PCA computation
    is performed.

    Parameters
    ----------
    reducer : PCAReducer
        A fitted PCAReducer instance. Must have been fitted before passing
        to this class.

    Raises
    ------
    TypeError
        If reducer is not a PCAReducer.
    RuntimeError
        If reducer has not been fitted yet.
    """

    def __init__(self, reducer: PCAReducer):
        if not isinstance(reducer, PCAReducer):
            raise TypeError(
                f"Expected a PCAReducer, got {type(reducer).__name__}."
            )
        reducer._check_is_fitted()
        self._reducer = reducer

    # ------------------------------------------------------------------
    # Properties — thin wrappers around the reducer
    # ------------------------------------------------------------------

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fraction of variance explained by each component. Shape: (k,)."""
        return self._reducer.explained_variance_ratio_

    @property
    def cumulative_variance_(self) -> np.ndarray:
        """Cumulative explained variance ratio. Shape: (k,)."""
        return self._reducer.cumulative_explained_variance_

    @property
    def k(self) -> int:
        """Number of components in the fitted reducer."""
        return self._reducer.k

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def plot(self, ax: plt.Axes | None = None) -> plt.Axes:
        """
        Plot the per-component and cumulative explained variance spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot, for further customisation.
        """
        explained  = self.explained_variance_ratio_
        cumulative = self.cumulative_variance_
        components = np.arange(1, len(explained) + 1)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(components, explained,  label="Per-component variance")
        ax.plot(components, cumulative, label="Cumulative variance")

        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance explained")
        ax.set_title(f"PCA Variance Spectrum  (k={self.k})")
        ax.legend()
        ax.grid(True)

        return ax

    def report(
        self,
        thresholds: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
        n_top: int = 10,
    ) -> None:
        """
        Print a textual summary useful for logbooks.

        Parameters
        ----------
        thresholds : sequence of float
            Cumulative variance thresholds to report the component count for.
        n_top : int
            Number of leading components to list individually.
        """
        explained  = self.explained_variance_ratio_
        cumulative = self.cumulative_variance_

        print(f"\nPCA spectrum summary  (k={self.k})\n")
        print("Cumulative variance thresholds:")
        for t in thresholds:
            if cumulative[-1] < t:
                print(f"  {int(t * 100)}%  — not reached within k={self.k} components")
            else:
                d = int(np.searchsorted(cumulative, t)) + 1
                print(f"  {int(t * 100)}% variance explained at d = {d}")

        print(f"\nTop {n_top} components:")
        for i, v in enumerate(explained[:n_top]):
            print(f"  PC{i + 1:>3d}: {v:.4f}  (cumulative: {cumulative[i]:.4f})")
