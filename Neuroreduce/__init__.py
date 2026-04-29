"""
Neuroreduce
-----------
Structural connectivity and fMRI dimensionality reduction library.

Notation (enforced everywhere):
    N  : number of brain parcels / ROIs
    T  : number of fMRI timepoints
    Tm : concatenated timepoints across subjects (method-dependent)
    k  : number of reduced dimensions

    BOLD input  : np.ndarray, shape (N, T)
    SC input    : np.ndarray, shape (N, N)
    Output      : np.ndarray, shape (k, T)
"""

from Neuroreduce.base import DimensionalityReducer
from Neuroreduce.methods import CHARMSCReducer
from Neuroreduce.methods.pca import PCAReducer
from Neuroreduce.methods.charm import CHARMReducer

__all__ = [
    "DimensionalityReducer",
    "PCAReducer",
    "CHARMReducer",
    "CHARMSCReducer",
]
