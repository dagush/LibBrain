"""
Deco2025_CHARM_SC/geometry/base_geometry.py
-----------------------------------
Abstract base class for geometry-based diffusion models on brain parcellations.

Both HARM and CHARM-SC share the same three-step pipeline:

    Step 1 — Build kernel matrix K (N×N) from parcel distances.
             Subclasses differ ONLY in the kernel function:
               HARM:     K[i,j] = exp(  -d²_ij / σ )   real, symmetric
               CHARM-SC: K[i,j] = exp( i·d²_ij / σ )   complex, symmetric

    Step 2 — Raise K to power t_horizon, take |·|², row-normalise:
               Q = |K^t|²           (real, non-negative)
               P = D⁻¹ Q            (row-stochastic, real)

    Step 3 — Extract quantity of interest from P.
             Subclasses differ in what they extract:
               HARM:     P raised to diffusion_steps → stationary distribution
               CHARM-SC: same (but complex kernel gives different P)

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

    Model_subjects.m / FCmodel.m — original MATLAB code by Gustavo Deco.

Design note
-----------
This class is NOT a Neuroreduce DimensionalityReducer. HARM and CHARM-SC
do not reduce dimensionality — they define a probability distribution over
parcels, used as a generative model of brain state occupancy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy import linalg as LA


class BaseCHARMGeometry(ABC):
    """
    Abstract base class for geometry-based brain diffusion models.

    Subclasses implement ``_kernel_value(d2)`` to define the specific
    kernel function applied to squared inter-parcel distances.

    Parameters
    ----------
    epsilon : float
        Kernel bandwidth σ. Controls the spatial scale of the diffusion.
        Larger σ → smoother, more global connectivity.
        Default: 1400.0 (value used in the 2025 PRE paper with Schaefer1000).
    t_horizon : int
        Diffusion horizon t. The kernel matrix K is raised to this power
        before extracting the real diffusion matrix Q = |K^t|².
        Default: 2.
    diffusion_steps : int
        Number of steps for computing the stationary distribution.
        P^diffusion_steps[0,:] approximates the stationary distribution.
        The larger this is, the closer to the true stationary distribution.
        Default: 50 (value used in Model_subjects.m as PmatrixC^50).

    Notes
    -----
    About P^diffusion_steps[0,:]:
        The full matrix P^n is computed, and then row 0 is extracted.
        For a row-stochastic matrix, after enough steps ALL rows converge
        to the same stationary distribution, so any row gives the same
        answer. Row 0 is chosen by convention, matching the MATLAB code:
            PmatrixC100 = PmatrixC^50;
            PstatesC    = PmatrixC100(1,:);   % MATLAB 1-indexed → row 0
        The variable name 'PmatrixC100' is historical (originally ^100);
        the actual exponent used in the paper is 50.
    """

    def __init__(
        self,
        epsilon:         float = 1400.0,
        t_horizon:       int   = 2,
        diffusion_steps: int   = 50,
    ):
        self.epsilon         = epsilon
        self.t_horizon       = t_horizon
        self.diffusion_steps = diffusion_steps

        # Set during fit()
        self._coords:      Optional[np.ndarray] = None   # (N, 3) parcel centroids
        self._Pmatrix:     Optional[np.ndarray] = None   # (N, N) row-stochastic
        self._Ptr_t:       Optional[np.ndarray] = None   # (N, N) |K^t|^2
        self._is_fitted:   bool                 = False

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses implement only this
    # -------------------------------------------------------------------------

    @abstractmethod
    def _kernel_value(self, d2: np.ndarray) -> np.ndarray:
        """
        Compute kernel values from squared distances.

        Parameters
        ----------
        d2 : np.ndarray, shape (N, N)
            Matrix of squared Euclidean distances between parcel centroids:
            d2[i,j] = ||c_i - c_j||²

        Returns
        -------
        K : np.ndarray, shape (N, N)
            Kernel matrix. May be complex (CHARM-SC) or real (HARM).
        """
        ...

    # -------------------------------------------------------------------------
    # Core interface
    # -------------------------------------------------------------------------

    def fit(self, coords: np.ndarray) -> "BaseCHARMGeometry":
        """
        Build the diffusion matrix P from parcel centroid coordinates.

        Parameters
        ----------
        coords : np.ndarray, shape (N, 3)
            Parcel centroid coordinates in 3D space (e.g. MNI/RAS).
            This is SchaeferCOG in the original MATLAB code, obtained via
            parcellation.get_CoGs().

        Returns
        -------
        self
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"coords must have shape (N, 3), got {coords.shape}. "
                "Pass the output of parcellation.get_CoGs()."
            )
        self._coords = coords.astype(np.float64)
        N = coords.shape[0]

        # ── Step 1: Build kernel matrix K (N×N) ──────────────────────────────
        # Compute all pairwise squared Euclidean distances vectorised.
        # d2[i,j] = ||c_i - c_j||² = sum_k (c_i[k] - c_j[k])²
        # This replaces the double for-loop in the MATLAB code:
        #   for i=1:N; for j=1:N; dij2=sum((COG(i,:)-COG(j,:)).^2); end; end
        diff     = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N,N,3)
        d2       = np.sum(diff ** 2, axis=2)                             # (N,N)
        Kmatrix  = self._kernel_value(d2)                                # (N,N) complex or real

        # ── Step 2: Raise to t_horizon, take |·|², row-normalise ─────────────
        # Eq. (11): Q = |K^t|²   (real, non-negative)
        # Assumption: matrix power LA.matrix_power mixes all rows/columns,
        # unlike element-wise power. For t_horizon=1 they are equivalent.
        Ktr_t        = LA.matrix_power(Kmatrix, self.t_horizon)
        Ptr_t        = np.abs(Ktr_t) ** 2                                # (N,N) real

        # Eq. (12): D_ii = sum_j Q_ij
        # Eq. (13): P = D^{-1} Q   (row-stochastic)
        D            = np.diag(np.sum(Ptr_t, axis=1))
        self._Pmatrix = LA.inv(D) @ Ptr_t                                # (N,N) real
        self._Ptr_t   = Ptr_t
        self._is_fitted = True
        return self

    def stationary_distribution(self) -> np.ndarray:
        """
        Compute the stationary distribution of the diffusion process.

        Raises P to ``diffusion_steps`` and returns the first row.
        For a row-stochastic matrix, all rows of P^n converge to the
        same stationary distribution as n → ∞. Row 0 is taken by
        convention, matching the MATLAB code.

        Returns
        -------
        p_states : np.ndarray, shape (N_valid,)
            Stationary probability of each parcel.
        """
        self._check_is_fitted()

        # Compute P^diffusion_steps (full matrix power)
        P_n      = LA.matrix_power(self._Pmatrix, self.diffusion_steps)
        p_states = P_n[0, :]                                # (N,) — first row
        return p_states

    @property
    def diffusion_matrix(self) -> np.ndarray:
        """
        The row-stochastic diffusion matrix P, shape (N, N).
        Available after fit() has been called.
        Used by BOLDGenerator for random-walk simulation.
        """
        self._check_is_fitted()
        return self._Pmatrix

    @property
    def n_parcels(self) -> int:
        """Number of parcels N."""
        self._check_is_fitted()
        return self._coords.shape[0]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. "
                "Call fit(coords) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        N = self._coords.shape[0] if self._is_fitted else "?"
        return (f"{self.__class__.__name__}("
                f"epsilon={self.epsilon}, "
                f"t_horizon={self.t_horizon}, "
                f"diffusion_steps={self.diffusion_steps}, "
                f"N={N}) [{status}]")
