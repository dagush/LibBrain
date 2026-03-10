# ==========================================================================
# ==========================================================================
# Multivariate Ornstein–Uhlenbeck (OU) Process
#
# The multivariate OU process is a linear stochastic model defined by:
#
#   dX = -A (X - mu) dt + B dW
#
# where:
#   X   : state vector (n_rois,)
#   A   : drift/mean-reversion matrix — in the whole-brain context this is
#         built as A = (1/tau) * I  -  g * W_SC, so that the structural
#         connectivity drives inter-regional coupling
#   mu  : long-term mean vector (n_rois,)
#   B   : diffusion coefficient (scalar * I, i.e. isotropic noise)
#   dW  : vector Wiener increment
#
# In the whole-brain neuronal mass formulation the linearised drift matrix is
#
#   A_ij = -g * W_SC_ij   (i ≠ j)
#   A_ii =  1 / tau
#
# so that the deterministic skeleton is:
#
#   dX_i/dt = -(1/tau) * (X_i - mu_i)  +  g * sum_j [W_SC_ij * X_j]
#
# This is the minimal analytically tractable baseline model and is widely
# used to derive the linearised covariance structure of BOLD-like signals.
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

from typing import Dict, List

import numba as nb
import numpy as np
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model, LinearCouplingModel
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class OrnsteinUhlenbeck(LinearCouplingModel):
    """
    Multivariate Ornstein–Uhlenbeck (OU) whole-brain model.

    Each region i evolves as:

        dX_i/dt = -(1/tau) * (X_i - mu_i)  +  g * coupling_i

    where coupling_i = sum_j [W_SC_ij * X_j] is computed externally by the
    LinearCouplingModel base class (weighted by the global coupling g).

    This is the deterministic skeleton; the simulator injects Gaussian noise
    of amplitude sigma at each time step to realise the full SDE.

    State Variables:
        x : local activity variable (a.u.) — one per ROI

    Observable Variables:
        x : same as the state (the activity itself is the observable)

    Coupling Variables:
        x : linear weighted sum via the structural connectivity matrix
    """

    # ------------------------------------------------------------------
    # Variable bookkeeping
    # ------------------------------------------------------------------
    _state_var_names = ['x']
    _coupling_var_names = ['x']
    _observable_var_names = ['x']

    # ------------------------------------------------------------------
    # Model parameters  (attributes tagged Model.Type.Model are
    # automatically packed into the m-array used inside numba kernels)
    # ------------------------------------------------------------------

    tau = Attr(
        default=20.0,
        attributes=Model.Type.Model,
        doc="Mean-reversion (decay) time constant (ms). Controls how fast "
            "activity returns to the long-term mean mu.",
    )

    mu = Attr(
        default=0.0,
        attributes=Model.Type.Model,
        doc="Long-term mean of the process (a.u.). "
            "Can be a scalar (broadcast to all ROIs) or an array of shape (n_rois,).",
    )

    # NOTE: the global coupling strength g is inherited from LinearCouplingModel.
    #       It pre-multiplies the weights matrix, so the effective drift matrix is
    #           A = diag(1/tau) - g * W_SC
    #       which matches the standard multivariate OU parameterisation.

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @overrides
    def _init_dependant(self):
        super()._init_dependant()
        self._init_dependant_automatic()

    @property
    def get_state_vars(self) -> Dict[str, int]:
        return OrnsteinUhlenbeck.state_vars

    @property
    def get_observablevars(self) -> Dict[str, int]:
        return OrnsteinUhlenbeck.observable_vars

    @property
    def get_c_vars(self) -> List[int]:
        return OrnsteinUhlenbeck.c_vars

    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Return initial state array of shape (n_state_vars, n_rois).

        Initialised to the long-term mean plus small Gaussian noise so that
        different ROIs start from slightly different positions.
        """
        state = np.empty((OrnsteinUhlenbeck.n_state_vars, n_rois))
        mu_val = np.asarray(self.mu)
        if mu_val.ndim == 0:
            state[0] = float(mu_val) + 0.01 * np.random.randn(n_rois)
        else:
            state[0] = mu_val + 0.01 * np.random.randn(n_rois)
        return state

    def initial_observed(self, n_rois: int) -> np.ndarray:
        """
        Return initial observable array of shape (n_observable_vars, n_rois).
        """
        observed = np.empty((OrnsteinUhlenbeck.n_observable_vars, n_rois))
        observed[0] = 0.0
        return observed

    # ------------------------------------------------------------------
    # Numba differential function
    # ------------------------------------------------------------------

    def get_numba_dfun(self):
        """
        Return the Numba-compiled differential function for the OU model.

        The returned callable has signature::

            dfun(state, coupling) -> (derivatives, observables)

        Both return arrays have shape (n_vars, n_rois).

        Deterministic update per region:

            dx_i/dt = -(1/tau) * (x_i - mu_i)  +  g * coupling_i

        The global coupling factor g is already baked into `coupling` by the
        LinearCouplingModel, so the dfun only needs the local decay term.
        """
        m = self.m.copy()
        P = self.P

        @nb.njit(
            nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
            cache=NUMBA_CACHE,
            fastmath=NUMBA_FASTMATH,
            nogil=NUMBA_NOGIL,
        )
        def OU_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Compute the deterministic drift and observable for the OU model.

            Parameters
            ----------
            state    : (1, n_rois) — current activity x
            coupling : (1, n_rois) — g * W_SC @ x  (linear coupling, already
                       scaled by g inside LinearCouplingModel)

            Returns
            -------
            derivatives : (1, n_rois) — dx/dt
            observables : (1, n_rois) — x  (activity is its own observable)
            """
            # Unpack parameters from the pre-built parameter matrix
            tau = m[np.intp(P.tau)]        # shape (n_rois,)
            mu  = m[np.intp(P.mu)]         # shape (n_rois,)

            x = state[0, :]                # current activity (n_rois,)

            # Deterministic OU drift:
            #   dx/dt = -(x - mu) / tau  +  coupling
            #
            # coupling already contains  g * (W_SC @ x),  i.e. the full
            # inter-regional drive scaled by the global coupling constant g.
            dx = -(x - mu) / tau + coupling[0, :]

            # Stack into 2-D arrays expected by the simulator
            derivatives = np.empty((1, x.shape[0]))
            observables = np.empty((1, x.shape[0]))
            derivatives[0, :] = dx
            observables[0, :] = x

            return derivatives, observables

        return OU_dfun

    # ------------------------------------------------------------------
    # Analytical Jacobian
    # ------------------------------------------------------------------

    @overrides
    def get_jacobian(self, sc: np.ndarray) -> np.ndarray:
        """
        Return the analytical Jacobian of the OU model.

        For the multivariate OU process the Jacobian evaluated at any point
        equals the drift matrix A itself (the dynamics are linear):

            J = -diag(1/tau)  +  g * W_SC

        i.e.  J_ij = g * W_SC_ij          (i ≠ j)
              J_ii = -1/tau_i  +  g * W_SC_ii

        Parameters
        ----------
        sc : (n_rois, n_rois) structural connectivity matrix.
             If global coupling g has already been applied, pass g * W_SC.

        Returns
        -------
        jacobian : (n_rois, n_rois) Jacobian matrix.
        """
        N = sc.shape[0]
        tau = np.asarray(self.tau)
        if tau.ndim == 0:
            inv_tau = np.ones(N) / float(tau)
        else:
            inv_tau = 1.0 / tau

        # J = -diag(1/tau) + g * W_SC
        jacobian = self.g * sc - np.diag(inv_tau)
        return jacobian

