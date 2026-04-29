# ==========================================================================
# ==========================================================================
# Exact Mean-Field Model (EMFM), from:
#
# [Sanz-Perl_2023] Y. Sanz Perl, G. Zamora-Lopez, E. Montbrió, M. Monge-Asensio,
#                  J. Vohryzek, S. Fittipaldi, C. González Campo, S. Moguilner,
#                  A. Ibañez, E. Tagliazucchi, B.T.T. Yeo, M.L. Kringelbach, G. Deco
#                  "The impact of regional heterogeneity in whole-brain dynamics
#                  in the presence of oscillations"
#                  Network Neuroscience, 7(2), 632–660, 2023.
#                  https://doi.org/10.1162/netn_a_00299
#
# The local dynamics of each brain region are described by the Firing Rate
# Equations (FRE) exactly derived from populations of QIF neurons in:
#
# [Montbrió_2015] E. Montbrió, D. Pazó, A. Roxin
#                 "Macroscopic description for networks of spiking neurons"
#                 Physical Review X, 5(2), 021028, 2015.
#
# [Devalle_2017]  F. Devalle, A. Roxin, E. Montbrió
#                 "Firing rate equations require a spike synchrony mechanism
#                 to correctly describe fast oscillations in inhibitory networks"
#                 PLOS Computational Biology, 13(12), e1005881, 2017.
#
# --------------------------------------------------------------------------
# *** IMPORTANT – DIMENSIONAL vs. NONDIMENSIONAL ***
# --------------------------------------------------------------------------
# The paper (§ Methods) presents the model in NONDIMENSIONAL form (Eq. 9-11).
# However, the reference MATLAB code (slurm_sbatch_EMFM_exploration_SC68.m)
# integrates the model in its original DIMENSIONAL form, which has more
# explicit parameters (Delta_e, Delta_i, eta_e, eta_i are all separate).
#
# This implementation follows the MATLAB reference code exactly – it uses
# DIMENSIONAL parameters throughout, so that numerical results can be
# compared directly with the original simulations.
#
# --------------------------------------------------------------------------
# MODEL EQUATIONS (dimensional form, per region n)
# --------------------------------------------------------------------------
# Each brain region contains two neural populations, excitatory (e) and
# inhibitory (i), each described by two variables:
#   r : mean firing rate of the population   [spikes / (tau_m · s)]
#   v : mean membrane potential              [mV (up to tau_m scaling)]
#
# Excitatory population (region n):
#
#   dr_e^n/dt = Delta_e / pi  +  2 · r_e^n · v_e^n
#
#   dv_e^n/dt = (v_e^n)^2  +  eta_e  -  (pi · r_e^n)^2
#               +  J_ee · r_e^n                          [local exc→exc]
#               +  J_ei · r_i^n                          [local inh→exc]
#               +  J_ee · G · sum_p(C_np · r_e^p)        [long-range exc]
#
# Inhibitory population (region n):
#
#   dr_i^n/dt = Delta_i / pi  +  2 · r_i^n · v_i^n
#
#   dv_i^n/dt = (v_i^n)^2  +  eta_i  -  (pi · r_i^n)^2
#               +  J_ii · r_i^n                          [local inh→inh]
#               +  J_ie · r_e^n                          [local exc→inh]
#
# Long-range coupling is exclusively E-to-E (inhibitory populations receive
# no inter-region input), scaled by G · J_glob where J_glob = J_ee.
#
# --------------------------------------------------------------------------
# HOW THE MATLAB A-MATRIX MAPS TO THIS CODE
# --------------------------------------------------------------------------
# The MATLAB code builds a (2N × 2N) connectivity matrix A that encodes
# ALL synaptic interactions (local and long-range) in one product A @ r.
# Here we decompose that product into explicit terms:
#
#   MATLAB exc row for region m:
#     A[2m,   2m  ] = J_ee               → J_ee * r_e[m]   (local exc→exc)
#     A[2m,   2m+1] = J_ei               → J_ei * r_i[m]   (local inh→exc)
#     A[2m,   2n  ] += G·J_glob·C[m,n]  → J_ee * coupling[0][m]  (long-range)
#       where coupling[0] = G * C^T @ r_e  (from LinearCouplingModel, g = G)
#       and J_glob = J_ee in the MATLAB code.
#
#   MATLAB inh row for region m:
#     A[2m+1, 2m+1] = J_ii               → J_ii * r_i[m]   (local inh→inh)
#     A[2m+1, 2m  ] = J_ie               → J_ie * r_e[m]   (local exc→inh)
#     (no long-range entries in inhibitory rows)
#
# --------------------------------------------------------------------------
# PARAMETER DEFAULTS (from MATLAB code)
# --------------------------------------------------------------------------
#   J_ee    = 50     (excitatory-to-excitatory synaptic strength; = J_glob)
#   J_ie    = 20     (excitatory-to-inhibitory synaptic strength)
#   J_ii    = -20    (inhibitory-to-inhibitory synaptic strength)
#   J_ei    = -20    (inhibitory-to-excitatory synaptic strength)
#   eta_e   = 30     (Ie1 in MATLAB – centre of exc. Lorentzian current)
#   eta_i   = 40     (Ii1 in MATLAB – centre of inh. Lorentzian current)
#   Delta_e = 40     (fixed half-width of exc. Lorentzian distribution)
#   Delta_i = swept  (40–50 in MATLAB exploration; ~47 at optimal point)
#   G = g   = swept  (global coupling; ~1.0 at reported optimal point)
#   sigma   = 0.01   (noise amplitude; handled by the integrator, not dfun)
#   gamma   = 1      (dt scale; absorbed into the integrator step)
# ==========================================================================
# ==========================================================================

import numpy as np
import numba as nb
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.simulator.models import Model, LinearCouplingModel
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class EMFM(LinearCouplingModel):
    """
    Exact Mean-Field Model (EMFM) whole-brain model.

    Implements the DIMENSIONAL Firing Rate Equations (FRE) for coupled
    excitatory-inhibitory populations of quadratic integrate-and-fire (QIF)
    neurons, as used in Sanz-Perl et al. (2023).

    Parameters are kept in their original dimensional form to match the
    MATLAB reference code (slurm_sbatch_EMFM_exploration_SC68.m) exactly.
    See the module docstring for the mapping between the MATLAB A-matrix
    formulation and the decomposed terms used here.

    State Variables (4 per region)
    --------------------------------
    r_e : Mean firing rate of the excitatory population.
    v_e : Mean membrane potential of the excitatory population.
    r_i : Mean firing rate of the inhibitory population.
    v_i : Mean membrane potential of the inhibitory population.

    Coupling Variables
    ------------------
    Only r_e participates in long-range coupling (E-to-E structural
    connections).  LinearCouplingModel.get_numba_coupling() computes
    coupling[0] = g * C^T @ r_e  (with g = G), which is then multiplied
    by J_ee inside dfun to give the full long-range term.

    Observable Variables
    --------------------
    None declared beyond state variables.  r_e (row 0 of state) is the
    signal passed to the Balloon–Windkessel BOLD model externally.

    References
    ----------
    [Sanz-Perl_2023] Sanz Perl et al., Network Neuroscience, 7(2), 632-660, 2023.
    [Montbrió_2015]  Montbrió et al., Physical Review X, 5(2), 021028, 2015.
    [Devalle_2017]   Devalle et al., PLOS Comput. Biol., 13(12), e1005881, 2017.
    """

    # ------------------------------------------------------------------
    # Variable declarations (row order in the state array)
    # ------------------------------------------------------------------

    _state_var_names = ['r_e', 'v_e', 'r_i', 'v_i']
    """
    State variables, one row per variable across all n_rois columns:
      row 0 -> r_e : excitatory mean firing rate
      row 1 -> v_e : excitatory mean membrane potential
      row 2 -> r_i : inhibitory mean firing rate
      row 3 -> v_i : inhibitory mean membrane potential
    """

    _coupling_var_names = ['r_e']
    """
    Only r_e is transmitted between regions.  Long-range connections are
    purely E-to-E; inhibitory populations receive no inter-region input.
    LinearCouplingModel computes coupling[0] = g * C^T @ r_e.
    """

    _observable_var_names = []
    """
    No additional observable variables.  r_e is already a state variable
    and is the quantity passed to the BOLD model.
    """

    # ==================================================================
    # Model Parameters – DIMENSIONAL, matching the MATLAB reference code
    # ==================================================================

    # ------------------------------------------------------------------
    # Half-widths of the Lorentzian input-current distributions.
    # These are the primary bifurcation parameters of the model.
    # The Hopf bifurcation exists as a curve in the (Delta_e, Delta_i)
    # plane (equivalent to the (d_e, d_i) plane of the nondim. paper).
    # ------------------------------------------------------------------

    Delta_e = Attr(
        default=40.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Half-width of the Lorentzian distribution of input currents for "
            "the EXCITATORY population (dimensional). "
            "Called 'Deltae' in the MATLAB code; fixed at 40 in the parameter "
            "sweep.  In the nondimensional paper notation: d_e = Delta_e / eta_i. "
            "In the heterogeneous model this is set per-region via "
            "compute_heterogeneous_delta(); see that method for details."
        )
    )

    Delta_i = Attr(
        default=47.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Half-width of the Lorentzian distribution of input currents for "
            "the INHIBITORY population (dimensional). "
            "Called 'Deltai' in the MATLAB code; swept in the range 40:50. "
            "In the nondimensional paper notation: d_i = Delta_i / eta_i. "
            "The optimal homogeneous value d_i ≈ 1.175 (Fig. 3A) corresponds "
            "to Delta_i = 1.175 * eta_i = 1.175 * 40 = 47 (dimensional)."
        )
    )

    # ------------------------------------------------------------------
    # Centre input currents (mean of the Lorentzian distributions).
    # Called Ie1 (eta_e) and Ii1 (eta_i) in the MATLAB code.
    # ------------------------------------------------------------------

    eta_e = Attr(
        default=30.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Centre of the Lorentzian input-current distribution for the "
            "EXCITATORY population (dimensional). "
            "Called 'Ie1' in the MATLAB code. Default: 30."
        )
    )

    eta_i = Attr(
        default=40.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Centre of the Lorentzian input-current distribution for the "
            "INHIBITORY population (dimensional). "
            "Called 'Ii1' in the MATLAB code. Default: 40.  Also used as "
            "the normalisation constant for the nondimensionalisation."
        )
    )

    # ------------------------------------------------------------------
    # Synaptic strengths (dimensional).
    # In the MATLAB code J_glob = J_ee, so the long-range coupling is:
    #   G * J_glob * C @ r_e  =  G * J_ee * C @ r_e
    # which we write as  J_ee * coupling[0]  in dfun (coupling[0] = G*C^T@r_e).
    # ------------------------------------------------------------------

    J_ee = Attr(
        default=50.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Excitatory-to-excitatory mean synaptic strength (dimensional). "
            "Also equals J_glob in the MATLAB code, so it multiplies the "
            "long-range coupling term:  J_ee * G * C^T @ r_e. "
            "Default: 50."
        )
    )

    J_ei = Attr(
        default=-20.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Inhibitory-to-excitatory mean synaptic strength (dimensional, "
            "negative). Called 'Jei' in the MATLAB code (-20 for every region). "
            "Default: –20."
        )
    )

    J_ie = Attr(
        default=20.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Excitatory-to-inhibitory mean synaptic strength (dimensional). "
            "Called 'Jie' in the MATLAB code. Default: 20."
        )
    )

    J_ii = Attr(
        default=-20.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Inhibitory-to-inhibitory mean synaptic strength (dimensional, "
            "negative). Called 'Jii' in the MATLAB code. Default: –20."
        )
    )

    # ------------------------------------------------------------------
    # g (inherited from LinearCouplingModel) = global coupling factor G.
    #
    # LinearCouplingModel.get_numba_coupling() returns:
    #   coupling[0] = g * C^T @ r_e  =  G * C^T @ r_e
    #
    # In dfun we multiply by J_ee to obtain the full long-range term:
    #   J_ee * G * C^T @ r_e  =  G * J_glob * C @ r_e   (MATLAB notation)
    #
    # MATLAB parameter sweep: GG = 10:0.2:12, then G = GG/Jee*5.
    # For GG=10 → G = 10/50*5 = 1.0; for GG=12 → G = 12/50*5 = 1.2.
    # ------------------------------------------------------------------
    g = Attr(
        default=1.0,
        attributes=Model.Tag.REGIONAL,
        doc=(
            "Global structural coupling scaling factor G (= 'g' in the "
            "LinearCouplingModel base class). "
            "LinearCouplingModel computes coupling[0] = g * C^T @ r_e. "
            "In dfun this is multiplied by J_ee to form the full long-range "
            "E-to-E term. "
            "MATLAB sweep: G = GG/Jee*5 with GG in [10,12] → G in [1.0,1.2]."
        )
    )

    # ==================================================================
    # Initialisation
    # ==================================================================

    @overrides
    def _init_dependant(self):
        super()._init_dependant()

    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Return the initial state used in the MATLAB reference code.

        The MATLAB code sets:
            re = 0.7813;  ve = -0.4196;
            ri = 0.7813;  vi = -0.4196;
        and repeats them for all N regions via repmat.

        Args:
            n_rois: Number of brain regions (ROIs).

        Returns:
            state : np.ndarray, shape (n_state_vars, n_rois)
                Rows ordered as [r_e, v_e, r_i, v_i].
        """
        state = np.empty((EMFM.n_state_vars, n_rois))

        # Taken verbatim from the MATLAB reference code
        state[EMFM.state_vars['r_e']] = 0.7813   # excitatory firing rate
        state[EMFM.state_vars['v_e']] = -0.4196  # excitatory membrane potential
        state[EMFM.state_vars['r_i']] = 0.7813   # inhibitory firing rate
        state[EMFM.state_vars['v_i']] = -0.4196  # inhibitory membrane potential

        return state

    # ==================================================================
    # Core dynamics
    # ==================================================================

    def get_numba_dfun(self):
        """
        Return a Numba-compiled function evaluating the FRE time derivatives.

        This replicates the MATLAB inner-loop body (with gamma = 1):

            kr = Deltavec / pi  +  2 * rold * vold
            kv = vold^2  +  A * rold  +  etavec  -  pi^2 * rold^2

        where A * rold is decomposed into local synaptic terms plus the
        long-range coupling (see module docstring and inline comments).

        The factor gamma = 1 and the noise term sigma * randn(...) are
        both handled by the simulator's integrator, not here.

        Returns:
            emfm_dfun : Numba @njit compiled function with signature
                UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:])
                (state, coupling) -> (d_state, observed)
        """
        # Capture the parameter matrix as a plain ndarray; Numba cannot
        # capture 'self' in a closure, but it can capture a numpy array.
        m = self.m.copy()
        P = self.P

        _PI  = np.float64(np.pi)
        _PI2 = np.float64(np.pi ** 2)

        @nb.njit(
            nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
            cache=NUMBA_CACHE,
            fastmath=NUMBA_FASTMATH,
            nogil=NUMBA_NOGIL
        )
        def emfm_dfun(state, coupling):
            """
            Compute dimensional FRE derivatives for all brain regions.

            Replicates the MATLAB Euler-step right-hand side:

              Excitatory (maps to MATLAB kr/kv for odd indices):
                dr_e = Delta_e / pi  +  2 * r_e * v_e
                dv_e = v_e^2  +  eta_e  -  pi^2 * r_e^2
                       + J_ee * r_e  +  J_ei * r_i          (local, from A)
                       + J_ee * coupling[0]                  (long-range, from A)

              Inhibitory (maps to MATLAB kr/kv for even indices):
                dr_i = Delta_i / pi  +  2 * r_i * v_i
                dv_i = v_i^2  +  eta_i  -  pi^2 * r_i^2
                       + J_ii * r_i  +  J_ie * r_e          (local, from A)

            Args:
                state    : (4, n_rois) – rows are [r_e, v_e, r_i, v_i].
                coupling : (1, n_rois) – G * C^T @ r_e, i.e. the weighted
                           sum of excitatory firing rates from all other
                           regions, scaled by the global coupling G.

            Returns:
                d_state  : (4, n_rois) – time derivatives.
                observed : (1, 1)      – empty placeholder.
            """
            # ----------------------------------------------------------
            # Unpack regional parameter vectors from m
            # ----------------------------------------------------------
            Delta_e = m[np.intp(P.Delta_e)]   # exc. Lorentzian half-width
            Delta_i = m[np.intp(P.Delta_i)]   # inh. Lorentzian half-width
            eta_e   = m[np.intp(P.eta_e)]     # exc. Lorentzian centre current
            eta_i   = m[np.intp(P.eta_i)]     # inh. Lorentzian centre current
            J_ee    = m[np.intp(P.J_ee)]      # exc→exc synaptic strength
            J_ei    = m[np.intp(P.J_ei)]      # inh→exc synaptic strength
            J_ie    = m[np.intp(P.J_ie)]      # exc→inh synaptic strength
            J_ii    = m[np.intp(P.J_ii)]      # inh→inh synaptic strength

            # ----------------------------------------------------------
            # Unpack state variables
            # ----------------------------------------------------------
            r_e = state[0, :]   # excitatory mean firing rate
            v_e = state[1, :]   # excitatory mean membrane potential
            r_i = state[2, :]   # inhibitory mean firing rate
            v_i = state[3, :]   # inhibitory mean membrane potential

            # ----------------------------------------------------------
            # Long-range E-to-E coupling term
            #
            # LinearCouplingModel has computed:
            #   coupling[0] = g * C^T @ r_e  =  G * C^T @ r_e
            #
            # Multiplying by J_ee recovers the MATLAB expression:
            #   G * J_glob * C @ r_e   (J_glob = J_ee = 50)
            #
            # This corresponds to the off-diagonal block in A:
            #   A[exc_m, exc_n] += G * J_glob * C[m,n]  for all n≠m,
            #   summed over n gives: sum_n A[exc_m, exc_n]*r_e[n]
            #                      = G*J_glob * (C @ r_e)[m]
            #                      = J_ee * coupling[0][m]
            # ----------------------------------------------------------
            long_range_exc = J_ee * coupling[0, :]

            # ----------------------------------------------------------
            # Excitatory population  (MATLAB: odd-indexed rows/cols of A)
            #
            # MATLAB:  kr(1:2:end) = Deltae/pi + 2*r(1:2:end)*v(1:2:end)
            #          kv(1:2:end) = v(1:2:end)^2
            #                        + A(exc,:)*r   <-- all terms below
            #                        + etavec(exc)  <-- eta_e
            #                        - pi^2*r(exc)^2
            #
            # A(exc_m, exc_m)  = J_ee  → J_ee * r_e
            # A(exc_m, inh_m)  = J_ei  → J_ei * r_i
            # A(exc_m, exc_n) += G*Jglob*C[m,n] → J_ee * coupling[0]
            # ----------------------------------------------------------
            dr_e = Delta_e / _PI  +  2.0 * r_e * v_e

            dv_e = (v_e ** 2
                    + eta_e
                    - _PI2 * r_e ** 2   # -(pi*r_e)^2 from the QIF reset rule
                    + J_ee * r_e        # diagonal A[exc,exc]: local exc→exc
                    + J_ei * r_i        # off-block A[exc,inh]: local inh→exc
                    + long_range_exc)   # long-range E-to-E coupling

            # ----------------------------------------------------------
            # Inhibitory population  (MATLAB: even-indexed rows/cols of A)
            #
            # MATLAB:  kr(2:2:end) = Deltai/pi + 2*r(2:2:end)*v(2:2:end)
            #          kv(2:2:end) = v(2:2:end)^2
            #                        + A(inh,:)*r   <-- terms below
            #                        + etavec(inh)  <-- eta_i
            #                        - pi^2*r(inh)^2
            #
            # A(inh_m, inh_m) = J_ii  → J_ii * r_i
            # A(inh_m, exc_m) = J_ie  → J_ie * r_e
            # (no long-range entries in inhibitory rows of A)
            # ----------------------------------------------------------
            dr_i = Delta_i / _PI  +  2.0 * r_i * v_i

            dv_i = (v_i ** 2
                    + eta_i
                    - _PI2 * r_i ** 2   # -(pi*r_i)^2 from the QIF reset rule
                    + J_ii * r_i        # diagonal A[inh,inh]: local inh→inh
                    + J_ie * r_e)       # off-block A[inh,exc]: local exc→inh

            # ----------------------------------------------------------
            # Pack derivatives, ordered as [r_e, v_e, r_i, v_i]
            # to match _state_var_names.
            # ----------------------------------------------------------
            d_state = np.stack((dr_e, dv_e, dr_i, dv_i))

            # No observable variables; return a (1,1) placeholder as
            # required by the Model interface.
            observed = np.empty((1, 1))

            return d_state, observed

        return emfm_dfun

    # ==================================================================
    # Heterogeneity helpers
    # ==================================================================

    @staticmethod
    def compute_heterogeneous_delta(beta: np.ndarray,
                                    delta_1: float,
                                    delta_2: float,
                                    eta_i: float = 40.0) -> np.ndarray:
        """
        Compute per-region Delta values from a heterogeneity map.

        The paper (Eq. 12 in [Sanz-Perl_2023]) defines the modulation in
        nondimensional form:

            d^n = 1 + delta_1 + delta_2 * beta_n       (d = Delta / eta_i)

        Converting back to dimensional Delta (multiply by eta_i):

            Delta^n = eta_i * (1 + delta_1 + delta_2 * beta_n)

        Assign the returned array to both Delta_e and Delta_i when
        configuring the heterogeneous model, as in [Sanz-Perl_2023] where
        both populations are modulated identically.

        Args:
            beta    : np.ndarray, shape (n_rois,)
                      Regional heterogeneity map (e.g. T1w/T2w ratio or
                      node-level GBC), typically normalised to [0, 1].
            delta_1 : float
                      Additive bias (uniform offset across all regions).
            delta_2 : float
                      Scale multiplier (amplifies regional variation).
            eta_i   : float
                      Dimensional centre of the inhibitory Lorentzian,
                      used to convert the nondimensional modulation back
                      to dimensional Delta.  Default: 40 (MATLAB 'Ii1').

        Returns:
            Delta_hetero : np.ndarray, shape (n_rois,)
                           Per-region dimensional Delta values, suitable for
                           direct assignment to model.Delta_e and model.Delta_i.

        Example usage:
            delta = EMFM.compute_heterogeneous_delta(beta, delta_1=-0.7,
                                                     delta_2=2.1)
            model = EMFM(weights=SC, Delta_e=delta, Delta_i=delta, g=G)
            model.configure()
        """
        return eta_i * (1.0 + delta_1 + delta_2 * beta)

    def get_noise_template(self):
        return np.r_[1., 1., 1., 1.]  # ['r_e', 'v_e', 'r_i', 'v_i']
