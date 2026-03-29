"""
Deco2025_CHARM_SC/geometry/harm.py
--------------------------
HARM (Harmonic) geometry model.

Uses the REAL exponential kernel:
    K[i,j] = exp( -d²_ij / σ )

This is the classical diffusion map kernel. It defines a standard
Gaussian random walk on the parcel geometry — parcels that are spatially
close are strongly connected, distant parcels weakly connected.

HARM is the baseline against which CHARM-SC is compared in the paper.
The claim is that the COMPLEX kernel in CHARM-SC captures brain geometry
better than this real kernel.

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

    Model_subjects.m section "%% HARM SC" — original MATLAB code.
"""

import numpy as np
from geometry.base_geometry import BaseCHARMGeometry


class HARM(BaseCHARMGeometry):
    """
    HARM: real Gaussian diffusion on parcel geometry.

    Kernel: K[i,j] = exp( -d²_ij / σ )

    For HARM, t_horizon has no special meaning because the kernel is
    already real and symmetric, so |K^t|² = K^(2t) — raising a
    non-negative real matrix to any power leaves it non-negative and real.
    The t_horizon parameter is kept for API consistency with CHARM-SC.

    Parameters
    ----------
    epsilon : float
        Kernel bandwidth σ. Default: 1400.0.
    t_horizon : int
        Diffusion horizon. Default: 2 (matches CHARM-SC for fair comparison).
    diffusion_steps : int
        Steps for stationary distribution. Default: 50.
    exclude_parcels : list of int or None
        0-indexed parcels to exclude. Default: [554, 907].

    Examples
    --------
    >>> harm = HARM(epsilon=1400.0)
    >>> harm.fit(parcellation.get_CoGs())
    >>> p_states = harm.stationary_distribution()   # (N_valid,)
    """

    def _kernel_value(self, d2: np.ndarray) -> np.ndarray:
        """
        Real Gaussian kernel.

        K[i,j] = exp( -d²_ij / σ )

        Parameters
        ----------
        d2 : np.ndarray, shape (N, N)
            Squared inter-parcel distances.

        Returns
        -------
        K : np.ndarray, shape (N, N), dtype float64
            Real, symmetric, non-negative kernel matrix.
        """
        # Negative sign: nearby parcels (small d²) get weight ≈ 1,
        # distant parcels (large d²) get weight ≈ 0.
        return np.exp(-d2 / self.epsilon)
