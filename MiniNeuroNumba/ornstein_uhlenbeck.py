# ==========================================================================
# ==========================================================================
# Multivariate Ornstein-Uhlenbeck Process
#
# A continuous-time stochastic process describing mean-reverting dynamics
# with Gaussian noise. The multivariate form couples N nodes through a
# structural connectivity matrix, making it suitable for whole-brain modeling.
#
# The SDE for each node i is:
#   dX_i = theta * (mu - X_i) * dt + g * sum_j(W_ij * X_j) * dt + sigma * dW_i
#
# where theta is the mean-reversion rate, mu is the long-term mean,
# g is the global coupling strength, W is the structural connectivity,
# and sigma is the noise amplitude (handled by the integrator).
#
# The resulting full drift matrix is:
#   A = -theta * I + g * W
#
# For stability, all eigenvalues of A must have negative real parts.
#
# References:
#   [Uhlenbeck_Ornstein_1930] G.E. Uhlenbeck, L.S. Ornstein
#       "On the Theory of the Brownian Motion"
#       Phys. Rev. 36, 823 (1930)
#
#   [Risken_1996] H. Risken
#       "The Fokker-Planck Equation: Methods of Solution and Applications"
#       Springer, 2nd ed., 1996
#
# ==========================================================================
# ==========================================================================
import numpy as np
import numba as nb
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model, LinearCouplingModel
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class OrnsteinUhlenbeck(LinearCouplingModel):
    """
    Multivariate Ornstein-Uhlenbeck (OU) Process.

    Implements the multivariate OU process as a neural mass model coupled
    through structural connectivity. Each node follows mean-reverting
    dynamics driven by Gaussian noise, with inter-regional coupling
    provided by the structural connectivity matrix.

    The deterministic drift for node i is:
        dX_i/dt = theta * (mu - X_i) + g * sum_j(W_ij * X_j)

    State Variables:
        x: State variable for each ROI

    Observable Variables:
        None

    Coupling Variables:
        x: The state variable couples between regions via linear coupling
    """

    # State variables: x (one per ROI)
    state_vars = Model._build_var_dict(['x'])
    n_state_vars = len(state_vars)
    c_vars = [0]  # x couples between regions

    # No observable variables
    observable_vars = Model._build_var_dict([])
    n_observable_vars = len(observable_vars)

    # ==========================================================================
    # Model Parameters
    # ==========================================================================

    theta = Attr(default=1.0, attributes=Model.Type.Model,
                 doc="Mean-reversion rate (must be positive for stable local dynamics)")
    mu = Attr(default=0.0, attributes=Model.Type.Model,
              doc="Long-term mean of the process")

    @property
    def get_state_vars(self):
        """Get dictionary mapping state variable names to their indices."""
        return OrnsteinUhlenbeck.state_vars

    @property
    def get_observablevars(self):
        """Get dictionary mapping observable variable names to their indices."""
        return OrnsteinUhlenbeck.observable_vars

    @property
    def get_c_vars(self):
        """Get list of coupling variable indices."""
        return OrnsteinUhlenbeck.c_vars

    def initial_state(self, n_rois):
        """
        Initialize state variables.

        Args:
            n_rois: Number of regions of interest

        Returns:
            Initial state array with shape (1, n_rois), initialized to mu
        """
        state = np.empty((OrnsteinUhlenbeck.n_state_vars, n_rois))
        state[0] = 0.0
        return state

    def initial_observed(self, n_rois):
        """
        Initialize observable variables (none for OU process).

        Args:
            n_rois: Number of regions of interest

        Returns:
            Empty (1,1) array (no observables)
        """
        observed = np.empty((1, 1))
        return observed

    def get_numba_dfun(self):
        """
        Generate the Numba-compiled differential function for the OU process.

        The drift term is:
            dx = theta * (mu - x) + coupling

        Returns:
            Compiled function computing state derivatives
        """
        m = self.m.copy()
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                 cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def OrnsteinUhlenbeck_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Compute derivatives of state variables.

            Args:
                state: Current state (1, n_rois) containing [x]
                coupling: Coupling input (1, n_rois) = g * W @ x

            Returns:
                Tuple of (state_derivatives, observables)
                - state_derivatives: (1, n_rois) with [dx/dt]
                - observables: empty (1,1) array
            """
            x = state[0, :]

            theta = m[np.intp(P.theta)]
            mu = m[np.intp(P.mu)]

            # OU drift: theta * (mu - x) + inter-regional coupling
            dx = theta * (mu - x) + coupling[0, :]

            # Pack derivatives; no observables
            d_state = np.empty((1, x.shape[0]))
            d_state[0] = dx
            return d_state, np.empty((1, 1))

        return OrnsteinUhlenbeck_dfun

    @overrides
    def get_jacobian(self, sc):
        """
        Compute the analytical Jacobian of the multivariate OU process.

        The Jacobian of dX/dt = theta*(mu - X) + g*W@X is:
            J = -theta * I + g * W

        Args:
            sc: Structural connectivity matrix (N, N).
                If using global coupling g, it should be pre-multiplied.

        Returns:
            Jacobian matrix (N, N)
        """
        N = len(sc)
        jacobian = -self.theta * np.eye(N) + self.g * sc
        return jacobian
