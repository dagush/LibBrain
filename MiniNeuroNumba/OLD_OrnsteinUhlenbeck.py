# ==========================================================================
# ==========================================================================
# Multivariate Ornstein–Uhlenbeck (OU) Process
#
# The multivariate OU process is a linear stochastic model defined by:
#
#   dX = -A X dt + B dW
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
#   dX_i/dt = - sum_j [A_ij * X_j]
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

        dX_i/dt = - sum_j [A_ij * X_j]
                = - coupling_i

    where coupling_i = sum_j [A_ij * X_j] is computed externally by the
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
    # Model parameters  (attributes tagged Model.Tag.REGIONAL are
    # automatically packed into the m-array used inside numba kernels)
    # ------------------------------------------------------------------

    tau = Attr(
        default=20.0,
        attributes=Model.Tag.REGIONAL,
        doc="Mean-reversion (decay) time constant (ms). Controls how fast "
            "activity returns to the long-term mean mu.",
    )

    # NOTE: the global coupling strength g is inherited from LinearCouplingModel.
    #       It pre-multiplies the weights matrix, so the effective drift matrix is
    #           A = diag(1/tau) * I - g * W_SC
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
        state = np.ones((OrnsteinUhlenbeck.n_state_vars, n_rois))
        state[0] = 0.01 * np.random.randn(n_rois)
        return state

    def initial_observed(self, n_rois: int) -> np.ndarray:
        """
        Return initial observable array of shape (n_observable_vars, n_rois).
        """
        observed = np.empty((OrnsteinUhlenbeck.n_observable_vars, n_rois))
        observed[0] = 0.0
        return observed

    # ----------------------------------------------------------------------------------------------------
    # Normalization criteria:
    # We need A to be Hurwitz, i.e. Re(λᵢ(A)) < 0 for all i.
    # There are several ways to enforce this, each with different trade-offs.
    #
    # Comparison at a Glance (asked Claude IA)
    # ----------------------------------------------------------------------------------------------------
    # Method                 Hurwitz        Preserves     Preserves        Best used when...
    #                        on W?          asymmetry?    off-diagonal
    #                                                     structure?
    # ----------------------------------------------------------------------------------------------------
    # Spectral radius norm   ❌            ✅             ✅               We tune g and τ explicitly
    #                        (depends on g, τ)
    # Spectral projection    ✅            ✅             ⚠️ partial       We need exact spectral control
    # Diagonal dominance     ✅            ✅             ✅ off-diagonal  We want to preserve topology
    # Symmetrization + shift ✅            ❌             ⚠️ averaged      The model assumes undirected SC
    # ----------------------------------------------------------------------------------------------------

    def verify_stability(self, W_norm: np.ndarray) -> bool:
        """
        Verify that the actual system matrix A = -(1/tau)*I + g*W is Hurwitz.
        This is what actually governs stability, NOT W alone.
        """
        N = W_norm.shape[0]
        tau = np.asarray(self.tau)
        inv_tau = np.ones(N) / float(tau) if tau.ndim == 0 else 1.0 / tau

        A_system = -np.diag(inv_tau) + self.g * W_norm  # the TRUE system matrix

        eigvals = np.linalg.eigvals(A_system)
        lambda_max = np.max(np.real(eigvals))
        is_stable = lambda_max < 0

        print(f"  Max Re(λ) of A_system = -(1/τ)I + g·W : {lambda_max:.6f}")
        print(f"  Hurwitz stable: {is_stable}  (need < 0)")
        if not is_stable:
            print(f"  WARNING: reduce g below {np.min(inv_tau) / np.max(np.real(np.linalg.eigvals(W_norm))):.4f}")

        return is_stable

    def normalize_spectral_radius(self, W_raw: np.ndarray,
                                  zero_diagonal: bool = True,
                                  verify_stability: bool = True) -> np.ndarray:
        """
        Spectral Radius Normalization.

        Divides the matrix by its spectral radius (largest real part of eigenvalues),
        so that rho(W) = 1. The system is then stable iff g < 1/tau.

        - Preserves: connectivity structure, relative weights, asymmetry
        - Destroys:  absolute weight scale
        - Does NOT guarantee Hurwitz on W itself; stability depends on g and tau.
        """
        W = W_raw.copy().astype(float)
        if zero_diagonal:
            np.fill_diagonal(W, 0)

        spectral_radius = np.max(np.real(np.linalg.eigvals(W)))
        W_norm = W / spectral_radius

        # Verify stability margin: g * lambda_max < 1/tau
        if verify_stability:
            lambda_max = np.max(np.real(np.linalg.eigvals(W_norm)))
            stability_margin = 1 / self.tau - self.g * lambda_max
            print(f"Spectral radius of raw W: {spectral_radius:.4f}")
            print(f"lambda_max after normalization: {lambda_max:.4f}")
            print(f"Stability margin (should be > 0): {stability_margin:.4f}")
            if stability_margin <= 0:
                print(f"WARNING: System is unstable! Reduce g below {1 / (self.tau * lambda_max):.4f}")

        return W_norm

    def hurwitz_spectral_projection(self, W_raw: np.ndarray, epsilon: float = 0.01,
                                    zero_diagonal: bool = True) -> np.ndarray:
        """
        Hurwitz Stabilization via Spectral Projection (Eigendecomposition).

        Decomposes W into its eigenvalues, reflects any eigenvalue with a
        non-negative real part to -epsilon, then reconstructs. This directly
        enforces Hurwitz stability on W itself.

        - Preserves: dimensionality, general spectral shape
        - Destroys:  connectivity structure (non-local operation), symmetry,
                     and can introduce complex-valued entries (take real part)
        - Guarantees Hurwitz on W itself (all Re(lambda) <= -epsilon)
        """
        W = W_raw.copy().astype(float)
        if zero_diagonal:
            np.fill_diagonal(W, 0)

        eigenvalues, eigenvectors = np.linalg.eig(W)

        # Reflect unstable eigenvalues: if Re(lambda) >= 0, replace with -epsilon
        stabilized = np.where(np.real(eigenvalues) >= 0, -epsilon, eigenvalues)

        W_stable = eigenvectors @ np.diag(stabilized) @ np.linalg.inv(eigenvectors)
        return np.real(W_stable)  # discard residual imaginary parts from numerics

    def hurwitz_diagonal_dominance(self, W_raw: np.ndarray, epsilon: float = 0.01,
                                   zero_diagonal: bool = True) -> np.ndarray:
        """
        Hurwitz Stabilization via Strict Diagonal Dominance (Gershgorin-based).

        Sets each diagonal entry to -(row_sum + epsilon), enforcing strict diagonal
        dominance. By the Gershgorin circle theorem, all eigenvalues then lie in
        discs centered at the diagonal values, guaranteeing Hurwitz stability.
        This is your friend's second method.

        - Preserves: off-diagonal connectivity pattern, relative off-diagonal weights
        - Destroys:  absolute weight scale of diagonals; diagonal is fully overwritten
        - Guarantees Hurwitz on W itself (strictly diagonally dominant, neg diagonal)
        """
        W = W_raw.copy().astype(float)
        if zero_diagonal:
            np.fill_diagonal(W, 0)

        row_sums = np.sum(np.abs(W), axis=1)  # sum of off-diagonal absolute weights
        np.fill_diagonal(W, -(row_sums + epsilon))
        return W

    def hurwitz_symmetrized_negative_definite(self, W_raw: np.ndarray, epsilon: float = 0.01,
                                              zero_diagonal: bool = True) -> np.ndarray:
        """
        Hurwitz Stabilization via Symmetrization + Negative Definite Projection.

        Symmetrizes W as W_sym = (W + W^T) / 2, then shifts the spectrum so all
        eigenvalues are <= -epsilon. For symmetric matrices, real eigenvalues are
        guaranteed, and Hurwitz == negative definite.

        - Preserves: average bidirectional connectivity strength
        - Destroys:  asymmetry (directionality of connections is lost)
        - Guarantees Hurwitz on W itself (symmetric negative definite)
        """
        W = W_raw.copy().astype(float)
        if zero_diagonal:
            np.fill_diagonal(W, 0)

        W_sym = (W + W.T) / 2.0

        # Shift so that lambda_max = -epsilon
        lambda_max = np.max(np.linalg.eigvalsh(W_sym))  # eigvalsh: exact for symmetric
        shift = lambda_max + epsilon
        W_stable = W_sym - shift * np.eye(W_sym.shape[0])
        return W_stable

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

            dx_i/dt = - coupling_i

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
            coupling : (1, n_rois) — W_SC @ x  (linear coupling, already
                       scaled by g inside LinearCouplingModel)

            Returns
            -------
            derivatives : (1, n_rois) — dx/dt
            observables : (1, n_rois) — x  (activity is its own observable)
            """
            # Unpack parameters from the pre-built parameter matrix
            tau = m[np.intp(P.tau)]        # shape (n_rois,)

            x = state[0, :]                # current activity (n_rois,)

            # Deterministic OU drift:
            # dX = -A*X = -(1/tau)*x + g*W@x
            #          coupling already holds g * W @ x
            dx = -(1.0 / tau) * x + coupling[0, :]

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
