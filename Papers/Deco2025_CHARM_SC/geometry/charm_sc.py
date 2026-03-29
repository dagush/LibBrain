"""
Deco2025_CHARM_SC/geometry/charm_sc.py
------------------------------
CHARM-SC (Complex HARMonic Structural Connectivity) geometry model.

Uses the COMPLEX exponential kernel:
    K[i,j] = exp( i·d²_ij / σ )

The complex exponential introduces oscillatory structure: the phase of
K[i,j] rotates as a function of distance. After taking |K^t|², this
produces interference patterns between diffusion paths — parcels that
are close in straight-line distance but "phase-distant" can end up
weakly connected, reshaping the connectivity landscape beyond simple
proximity.

This is the key innovation over classical HARM: the complex kernel
captures brain geometry better because the brain's functional organisation
is not purely distance-dependent — it has a topological structure that
the oscillatory phase encodes.

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

    Model_subjects.m section "%% CHARM SC" — original MATLAB code.
"""

import numpy as np
from geometry.base_geometry import BaseCHARMGeometry


class CHARM_SC(BaseCHARMGeometry):
    """
    CHARM-SC: complex diffusion on parcel geometry.

    Kernel: K[i,j] = exp( i·d²_ij / σ )

    The imaginary unit i in the exponent means that the kernel value
    is a complex number of modulus 1 (a phase). After raising to power
    t and taking |·|², the interference between paths determines the
    effective transition probabilities.

    Parameters
    ----------
    epsilon : float
        Kernel bandwidth σ. Default: 1400.0.
    t_horizon : int
        Diffusion horizon t. The interference pattern depends critically
        on this value. Default: 2 (value from Model_subjects.m).
    diffusion_steps : int
        Steps for stationary distribution. Default: 50.
    exclude_parcels : list of int or None
        0-indexed parcels to exclude. Default: [554, 907].

    Examples
    --------
    >>> charm_sc = CHARM_SC(epsilon=1400.0, t_horizon=2)
    >>> charm_sc.fit(parcellation.get_CoGs())
    >>> p_states = charm_sc.stationary_distribution()   # (N_valid,)
    >>> P = charm_sc.diffusion_matrix                    # (N, N) for simulation
    """

    def _kernel_value(self, d2: np.ndarray) -> np.ndarray:
        """
        Complex exponential kernel.

        K[i,j] = exp( i·d²_ij / σ )

        Parameters
        ----------
        d2 : np.ndarray, shape (N, N)
            Squared inter-parcel distances.

        Returns
        -------
        K : np.ndarray, shape (N, N), dtype complex128
            Complex, symmetric kernel matrix. Each entry has modulus 1.
        """
        # +i sign: phase rotates with distance.
        # The imaginary unit 1j matches MATLAB's complex(0,1).
        # d2 / epsilon gives the phase angle in radians.
        return np.exp(1j * d2 / self.epsilon)
