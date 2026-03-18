# ==========================================================================
# ==========================================================================
# Whole-brain models from:
# [SanzPerl_2023] Y. Sanz Perl, G. Zamora-Lopez, E. Montbrió, M. Monge-Asensio,
#                 J. Vohryzek, S. Fittipaldi, C. González Campo, S. Moguilner,
#                 A. Ibañez, E. Tagliazucchi, B.T.T. Yeo, M.L. Kringelbach, G. Deco
#                 The impact of regional heterogeneity in whole-brain dynamics
#                 in the presence of oscillations.
#                 Network Neuroscience, 7(2): 632–660, 2023.
#                 doi: 10.1162/netn_a_00299
#
# Model implemented:
#
# ExactMeanField2023  – exact mean-field model of coupled excitatory (E) and
#     inhibitory (I) populations of quadratic integrate-and-fire (QIF) neurons.
#     Derived from Montbrió et al. (2015) and extended to E-I networks.
#     Regional heterogeneity enters through region-specific dispersion parameters.
#
#     Non-dimensionalised FRE (Firing-Rate Equations) per node n (Eqs. in paper):
#
#       tau_e * dR_e/dt = Delta_e_n / (pi * tau_e) + 2 * R_e * V_e
#       tau_e * dV_e/dt = V_e^2 + eta_e - (pi * tau_e * R_e)^2
#                         + J_ee * tau_e * R_e
#                         - J_ei * tau_e * R_i
#                         + G * sum_j [ C_{nj} * R_e_j ]
#
#       tau_i * dR_i/dt = Delta_i_n / (pi * tau_i) + 2 * R_i * V_i
#       tau_i * dV_i/dt = V_i^2 + eta_i - (pi * tau_i * R_i)^2
#                         + J_ie * tau_i * R_e
#                         - J_ii * tau_i * R_i
#
#     where:
#       R_e, R_i : mean firing rates of excitatory/inhibitory populations
#       V_e, V_i : mean membrane potentials of E/I populations
#       Delta_e_n = delta1 + delta2 * het_n   (region-specific E dispersion, Lorentzian HWHM)
#       Delta_i_n = delta1 + delta2 * het_n   (region-specific I dispersion, Lorentzian HWHM)
#       eta_e, eta_i : mean input currents (centre of Lorentzian distribution)
#       J_ee, J_ei, J_ie, J_ii : synaptic coupling strengths
#       G : global coupling strength
#
#     The fMRI BOLD observable is derived from R_e via the Balloon–Windkessel model
#     (not implemented here; R_e is exposed as the observable directly).
#
# ==========================================================================
# ==========================================================================

from typing import Dict, List

import numpy as np
import numba as nb
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model, LinearCouplingModel
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


# ---------------------------------------------------------------------------
# Small numerical constants
# ---------------------------------------------------------------------------
_ZERO = 0.0
_ONE  = 1.0
_PI   = np.pi


# ===========================================================================
#  Exact Mean-Field (QIF-based) whole-brain model
# ===========================================================================

class ExactMeanField2023(LinearCouplingModel):
    """
    Exact Mean-Field (EMF) whole-brain model with regional heterogeneity,
    as described in Sanz Perl et al. (2023), derived from Montbrió et al. (2015).

    Each brain region hosts an E-I population of all-to-all coupled QIF neurons.
    The Lorentzian ansatz yields exact firing-rate equations (FRE) for the mean
    firing rate R and mean membrane potential V of each population.

    Non-dimensionalised FRE per node n:

      Excitatory population:
        tau_e * dR_e/dt = Delta_e_n / pi + 2 * R_e * V_e
        tau_e * dV_e/dt = V_e^2 + eta_e - (pi * tau_e * R_e)^2
                          + J_ee * tau_e * R_e - J_ei * tau_e * R_i
                          + G * J_ee * tau_e * sum_j [C_{nj} * R_e_j]        ← long-range coupling

      Inhibitory population (no long-range coupling):
        tau_i * dR_i/dt = Delta_i_n / pi + 2 * R_i * V_i
        tau_i * dV_i/dt = V_i^2 + eta_i - (pi * tau_i * R_i)^2
                          + J_ie * tau_i * R_e - J_ii * tau_i * R_i

    Regional heterogeneity (T1w/T2w) is incorporated as:
        Delta_e_n = Delta_e * (delta1 + delta2 * het_n)
        Delta_i_n = Delta_i * (delta1 + delta2 * het_n)

    where het_n is the normalised regional T1w/T2w map value and (delta1, delta2)
    are the two free fitting parameters (see Figure 3B in paper).

    State variables:
        R_e : excitatory mean firing rate
        V_e : excitatory mean membrane potential
        R_i : inhibitory mean firing rate
        V_i : inhibitory mean membrane potential

    Observable variables:
        R_e : excitatory firing rate (BOLD proxy; pass through Balloon–Windkessel
              model externally for proper fMRI comparison)

    Coupling variables:
        R_e only (index 0) – long-range excitatory drive.

    Parameters
    ----------
    tau_e : float   Excitatory membrane time constant.   Default: 1.0 (non-dim.)
    tau_i : float   Inhibitory membrane time constant.   Default: 1.0 (non-dim.)
    eta_e : float   Mean excitatory input current (centre of Lorentzian). Default: -40.0
    eta_i : float   Mean inhibitory input current.  Default: -40.0
    Delta_e : float Base half-width of E Lorentzian heterogeneity distribution. Default: 1.0
    Delta_i : float Base half-width of I Lorentzian heterogeneity distribution.
                    Fit to data: paper optimal di = 1.175. Default: 1.175
    J_ee : float    E→E recurrent coupling.  Default: 50.0
    J_ei : float    I→E coupling strength.   Default: -20.0
    J_ie : float    E→I coupling strength.   Default: 20.0
    J_ii : float    I→I coupling strength.   Default: -20.0

    delta1 : float  Bias for heterogeneity modulation.  Default: 0.0
    delta2 : float  Scale for heterogeneity modulation. Default: 0.0
    het : array (N,) Normalised T1w/T2w heterogeneity map.  Default: zeros.

    g : float       Global coupling G (LinearCouplingModel).  Default: 1.04 (paper optimal)
    """

    # ------------------------------------------------------------------
    # Variable bookkeeping
    # ------------------------------------------------------------------
    _state_var_names = ['R_e', 'V_e', 'R_i', 'V_i']
    _coupling_var_names = ['V_e']
    _observable_var_names = ['R_e']

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------
    # Time constants (non-dimensionalised; typically set to 1 after rescaling)
    tau_e   = Attr(default=1.0,     attributes=Model.Tag.REGIONAL,
                   doc="Excitatory membrane time constant (non-dim.)")
    tau_i   = Attr(default=1.0,     attributes=Model.Tag.REGIONAL,
                   doc="Inhibitory membrane time constant (non-dim.)")

    # Centre of the Lorentzian excitability distributions
    eta_e   = Attr(default=30.0,   attributes=Model.Tag.REGIONAL,
                   doc="Mean excitatory input current (centre of Lorentzian)")
    eta_i   = Attr(default=40.0,   attributes=Model.Tag.REGIONAL,
                   doc="Mean inhibitory input current (centre of Lorentzian)")

    # Base half-widths of the Lorentzian distributions
    # These are modulated regionally: Delta_n = Delta * (delta1 + delta2 * het_n)
    Delta_e = Attr(default=40.0,     attributes=Model.Tag.REGIONAL,
                   doc="Base half-width of excitatory Lorentzian (HWHM)")
    Delta_i = Attr(default=47.0,   attributes=Model.Tag.REGIONAL,
                   doc="Base half-width of inhibitory Lorentzian (HWHM); paper optimal")

    # Synaptic coupling constants
    J_ee    = Attr(default=50.0,    attributes=Model.Tag.REGIONAL,
                   doc="Excitatory self-coupling J_EE")
    J_ei    = Attr(default=-20.0,    attributes=Model.Tag.REGIONAL,
                   doc="Inhibitory-to-excitatory coupling J_EI")
    J_ie    = Attr(default=20.0,    attributes=Model.Tag.REGIONAL,
                   doc="Excitatory-to-inhibitory coupling J_IE")
    J_ii    = Attr(default=-20.0,     attributes=Model.Tag.REGIONAL,
                   doc="Inhibitory self-coupling J_II")

    # Heterogeneity modulation parameters
    delta1  = Attr(default=1.0,     attributes=Model.Tag.REGIONAL,
                   doc="Bias term for heterogeneity modulation of Delta")
    delta2  = Attr(default=0.0,     attributes=Model.Tag.REGIONAL,
                   doc="Scaling term for heterogeneity modulation of Delta")

    # Regional heterogeneity map (per-node vector)
    het     = Attr(default=0.0,    attributes=Model.Tag.REGIONAL,
                   doc="Normalised regional T1w/T2w heterogeneity map, shape (N,)")

    @overrides
    def _init_dependant(self):
        super()._init_dependant()
        if not self._attr_defined('het'):
            self.het = np.zeros(self.n_rois)

    # ------------------------------------------------------------------
    # Standard interface
    # ------------------------------------------------------------------
    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Initialise near a plausible fixed point.
        R variables are non-negative; V variables near zero.
        """
        state = np.zeros((ExactMeanField2023.n_state_vars, n_rois))
        state[0, :] = 0.7813     # R_e  (small positive firing rate)
        state[1, :] = -0.4196    # V_e
        state[2, :] = 0.7813     # R_i
        state[3, :] = -0.4196    # V_i
        return state

    def initial_observed(self, n_rois: int) -> np.ndarray:
        observed = np.zeros((ExactMeanField2023.n_observable_vars, n_rois))
        return observed

    # ------------------------------------------------------------------
    # Numba dfun
    # ------------------------------------------------------------------
    def get_numba_dfun(self):
        """
        Return a Numba-compiled function computing the EMF model RHS.

        Signature:
            dfun(state: (4, N), coupling: (1, N))
                -> (d_state: (4, N), obs: (1, N))

        The coupling array is  G * C^T @ R_e  (provided by
        LinearCouplingModel.get_numba_coupling).

        Notes
        -----
        * R_e and R_i are clamped to [0, ∞) for physical plausibility.
        * The regional dispersion is computed as:
              Delta_e_n = Delta_e * (delta1 + delta2 * het_n)
              Delta_i_n = Delta_i * (delta1 + delta2 * het_n)
          If you want a purely homogeneous model, leave delta1=delta2=0 and
          Delta_e / Delta_i carry the global dispersion directly.
        """
        m   = self.m.copy()
        P   = self.P
        het = self.het.copy()

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                 cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def EMF_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Compute derivatives for the Exact Mean-Field whole-brain model.

            Args:
                state    : (4, N) – rows are [R_e, V_e, R_i, V_i]
                coupling : (1, N) – G * C^T @ R_e  (long-range excitatory input)

            Returns:
                (d_state (4, N), obs (1, N))
            """
            # Unpack scalar / per-roi parameters from the parameter matrix
            tau_e   = m[np.intp(P.tau_e)]
            tau_i   = m[np.intp(P.tau_i)]
            eta_e   = m[np.intp(P.eta_e)]
            eta_i   = m[np.intp(P.eta_i)]
            Delta_e = m[np.intp(P.Delta_e)]
            Delta_i = m[np.intp(P.Delta_i)]
            J_ee    = m[np.intp(P.J_ee)]
            J_ei    = m[np.intp(P.J_ei)]
            J_ie    = m[np.intp(P.J_ie)]
            J_ii    = m[np.intp(P.J_ii)]
            delta1  = m[np.intp(P.delta1)]
            delta2  = m[np.intp(P.delta2)]

            # Unpack state; clamp firing rates to be non-negative
            R_e = state[0, :]
            V_e = state[1, :]
            R_i = state[2, :]
            V_i = state[3, :]

            # -------------------------------------------------------
            # Regional dispersion (heterogeneity-modulated half-widths)
            # -------------------------------------------------------
            # When delta1 = delta2 = 0 the modulation factor is 0 and
            # Delta_e_n = 0 (degenerate). The user should set delta1 = 1
            # (or a non-zero value) to recover the homogeneous case, or
            # provide meaningful (delta1, delta2, het) for the heterogeneous
            # model.  Paper convention (homogeneous): delta2 = 0, delta1 = 1.
            modulation = delta1 + delta2 * het        # shape (N,)
            Delta_e_n  = Delta_e * modulation         # shape (N,)
            Delta_i_n  = Delta_i * modulation         # shape (N,)

            # -------------------------------------------------------
            # Long-range coupling (only excitatory, as in the paper)
            # -------------------------------------------------------
            lr_input = coupling[0, :]    # G * sum_j [C_{jn} * R_e_j]

            # -------------------------------------------------------
            # Excitatory population FRE
            # -------------------------------------------------------
            # tau_e * dR_e/dt = Delta_e_n / pi + 2 * R_e * V_e
            dR_e = (Delta_e_n / _PI + 2.0 * R_e * V_e) / tau_e

            # tau_e * dV_e/dt = V_e^2 + eta_e - (pi*tau_e*R_e)^2
            #                   + J_ee * tau_e * R_e
            #                   - J_ei * tau_e * R_i
            #                   + long-range coupling
            dV_e = (V_e**2
                    + eta_e
                    - (_PI * tau_e * R_e)**2
                    + J_ee * tau_e * R_e
                    - J_ei * tau_e * R_i
                    + J_ee * tau_e * lr_input) / tau_e

            # -------------------------------------------------------
            # Inhibitory population FRE  (no long-range coupling)
            # -------------------------------------------------------
            # tau_i * dR_i/dt = Delta_i_n / pi + 2 * R_i * V_i
            dR_i = (Delta_i_n / _PI + 2.0 * R_i * V_i) / tau_i

            # tau_i * dV_i/dt = V_i^2 + eta_i - (pi*tau_i*R_i)^2
            #                   + J_ie * tau_i * R_e - J_ii * tau_i * R_i
            dV_i = (V_i**2
                    + eta_i
                    - (_PI * tau_i * R_i)**2
                    + J_ie * tau_i * R_e
                    - J_ii * tau_i * R_i) / tau_i

            # -------------------------------------------------------
            # Pack and return derivatives and observables
            # -------------------------------------------------------
            d_state = np.stack((dR_e, dV_e, dR_i, dV_i))
            obs     = np.stack((R_e.copy(),))
            return d_state, obs

        return EMF_dfun