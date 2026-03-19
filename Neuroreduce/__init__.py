"""
neuroreduce
-----------
Structural connectivity and fMRI dimensionality reduction library.

Notation (enforced everywhere):
    N : number of brain parcels / ROIs
    T : number of fMRI timepoints
    k : number of reduced dimensions

    BOLD input  : np.ndarray, shape (N, T)
    SC input    : np.ndarray, shape (N, N)
    Output      : np.ndarray, shape (k, T)
"""

from Neuroreduce.base import DimensionalityReducer
from Neuroreduce.methods.pca import PCAReducer

__all__ = [
    "DimensionalityReducer",
    "PCAReducer",
]
