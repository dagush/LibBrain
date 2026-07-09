"""
Neuroreduce/methods/functional_harmonics.py
---------------------------------------------
Functional Harmonics: graph harmonics of the functional connectivity matrix.

Uses an FC or computes it from BOLD timeseries (via NeuroNumba FC observable), then
computes eigenvectors of the graph Laplacian of that FC matrix.

References
----------
Glomb, K., et al. (2021). Functional harmonics reveal multi-dimensional basis
functions underlying cortical organization. Cell Reports, 36(8).

Vohryzek, J., et al. (2024). Harmonic modes of neural activity in the resting
state. NeuroImage.

based on the code by
Agatha Aguilar Calvache, June 2025
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from Neuroreduce.methods.base_laplacian import BaseLaplacianReducer

from neuronumba.observables.fc import FC as _FCObservable


class FunctionalHarmonicsReducer(BaseLaplacianReducer):
    """
    Functional Harmonics dimensionality reduction.

    Uses the FC or computes it from BOLD (via NeuroNumba FC observable), then computes
    the k lowest-frequency eigenvectors of the graph Laplacian of that FC matrix.

    Parameters
    ----------
    k : int
        Number of harmonics to retain.
    threshold : float
        FC values at or below this threshold are zeroed. Default: 0.00065.
    laplacian_type : str
        'unnormalised' or 'symmetric'. Default: 'unnormalised'.
    normalise_input : bool
        Divide FC by its max before processing. Default: True.
    remove_self_connections : bool
        Zero out the diagonal before computing. Default: True.
    whiten : bool
        Z-score each row of transform() output. Default: False.

    Examples
    --------
    >>> reducer = FunctionalHarmonicsReducer(k=10)
    >>> reducer.fit(X=X_bold)              # X: (N, T) — FC computed internally
    >>> Z = reducer.transform(X_bold)      # Z: (10, T)
    >>> W = reducer.get_basis()            # W: (N, 10)
    """

    def _get_input_matrix(
        self,
        X:  Optional[np.ndarray],
        FC: Optional[np.ndarray],
        SC: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute FC from BOLD and use it as input to the Laplacian pipeline.
        If the FC is provided, return it directly.
        Otherwise, compute it from the BOLD timeseries X:
            FC is computed via the NeuroNumba FC observable which expects
            (T, N) input — we transpose X (N, T) before passing it.

        Parameters
        ----------
        X  : np.ndarray, shape (N, T) — BOLD timeseries
        FC : np.ndarray, shape (N, N) — functional connectivity matrix
        SC : ignored

        Returns
        -------
        FC : np.ndarray, shape (N, N)
        """
        if X is None and FC is None:
            raise ValueError(
                "FunctionalHarmonicsReducer requires either the BOLD timeseries X of the FC itself. "
                "Call fit(X=your_bold_signal)."
                "Call fit(FC=your_fc_matrix)."
            )
        if X is not None:
            # NeuroNumba FC observable expects (T, N) — transpose from (N, T)
            obs    = _FCObservable()
            result = obs.from_fmri(X.T)
            FC     = result['FC']                    # (N, N) Pearson correlation
            # Replace any NaN (from constant parcels) with 0
            FC     = np.nan_to_num(FC, nan=0.0)
        return FC
