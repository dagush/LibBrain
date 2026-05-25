"""
Neuroreduce/methods/connectome_harmonics.py
---------------------------------------------
Connectome Harmonics: graph harmonics of the structural connectivity matrix.

Computes the eigenvectors of the graph Laplacian of the SC matrix.
These form a brain-wide set of spatially organised oscillatory modes —
the "natural frequencies" of the structural connectome.

Reference
---------
Atasoy, S., Donnelly, I., & Pearce, J. (2016). Human brain networks function
in connectome-specific harmonic waves. Nature Communications, 7, 10340.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from Neuroreduce.methods.base_laplacian import BaseLaplacianReducer


class ConnectomeHarmonicsReducer(BaseLaplacianReducer):
    """
    Connectome Harmonics dimensionality reduction.

    Computes the k lowest-frequency eigenvectors of the graph Laplacian
    of the structural connectivity (SC) matrix and uses them to project
    BOLD timeseries into a structure-informed harmonic basis.

    Parameters
    ----------
    k : int
        Number of harmonics to retain.
    threshold : float
        SC values at or below this threshold are zeroed before computing
        the Laplacian. Default: 0.00065.
    laplacian_type : str
        'unnormalised' (L = D - A) or 'symmetric' (L_sym = D^{-1/2}LD^{-1/2}).
        Default: 'unnormalised' (matches the student's original code).
    normalise_input : bool
        Divide SC by its maximum value before processing. Default: True.
    remove_self_connections : bool
        Zero out the SC diagonal before computing. Default: True.
    whiten : bool
        Z-score each row of transform() output. Default: False.

    Examples
    --------
    >>> reducer = ConnectomeHarmonicsReducer(k=10)
    >>> reducer.fit(SC=SC)                    # SC: (N, N)
    >>> Z = reducer.transform(X_bold)         # Z:  (10, T)
    >>> W = reducer.get_basis()               # W:  (N, 10)  harmonic modes
    """

    def _get_input_matrix(
        self,
        X:  Optional[np.ndarray],
        SC: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Use the SC matrix as input to the Laplacian pipeline.

        Parameters
        ----------
        X  : ignored
        SC : np.ndarray, shape (N, N) — structural connectivity matrix

        Returns
        -------
        SC : np.ndarray, shape (N, N)
        """
        if SC is None:
            raise ValueError(
                "ConnectomeHarmonicsReducer requires a structural connectivity "
                "matrix SC. Call fit(SC=your_sc_matrix)."
            )
        return SC
