# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#             How local excitation-inhibition ratio impacts the whole brain dynamics
#             J. Neurosci., 34 (2014), pp. 7886-7898
#
# For the linear version, we use [Deco_2014] and
# [Demirtaş_2019] M. Demirtaş, J.B. Burt, M. Helmer, J.L. Ji, B.D. Adkinson, M.F. Glasser,
#                 D.C. Van Essen, S.N. Sotiropoulos, A. Anticevic, J.D. Murray
#                 Hierarchical Heterogeneity across Human Cortex Shapes Large-Scale Neural Dynamics
#                 Volume 101, Issue 6, p1181-1194.e13, March 20, 2019
#
# ==========================================================================
# ==========================================================================
import numpy as np
# from numba import jit

from numpy.random import randn as randn  # normal randn, comment for debug
# from Utils.randn2 import randn2 as randn  # uncomment for debug

import neuronumba.bold.stephan_2007 as BOLD

from Utils import numTricks as nT


# Set General Model Parameters
taon    = 100
taog    = 10
gamma   = 0.641
JN      = 0.15
I0      = 0.382
Jexte   = 1.
Jexti   = 0.7
w       = 1.4


# @jit(nopython=True)
def phi_e(x, Receptor, wgaine):
    de = 0.16
    I = 125.
    c = 310
    y = (c * x - I) * (1 + Receptor * wgaine)
    # if y != 0:
    result = y / (1 - np.exp(-de * y))
    # else:
    #     result = 0
    return result


# @jit(nopython=True)
def phi_i(x, Receptor, wgaini):
    di = 0.087
    I = 177.
    c = 615
    y = (c * x - I) * (1 + Receptor * wgaini)
    # if y != 0:
    result = y / (1 - np.exp(-di * y))
    # else:
    #     result=0
    return result


# @jit(nopython=True)
def dfun(Se, Si, J, G, SC, Receptor, wgaine, wgaini):
    coup = G * SC @ Se
    Ie = I0 * Jexte + w * JN * Se + JN * coup - J * Si
    Ii = I0 * Jexti + JN * Se - Si
    re = phi_e(Ie, Receptor, wgaine)
    ri = phi_i(Ii, Receptor, wgaini)
    dSe = -Se / taon + (1 - Se) * gamma * re / 1000.
    dSi = -Si / taog + ri / 1000.

    # Debug info (very useful!)
    debug = {
        "coup": coup,
        "He": re,
        "Hi": ri,
    }

    return dSe, dSi, re, debug


# @jit(nopython=True)
def simulate(
    # -------- model parms
    SC,
    G,
    J,
    # -------- Specific Deco 2018 parms
    Receptor,
    wgaine,
    wgaini,
    # -------- Simulation parms
    TR: float,
    dt: float,
    dtt: float,
    sigma: float,
    Tmax: float,
    burn_in: int = 0,
):
    """
    Minimal Hopf simulation with explicit integration.

    Returns
    -------
    xs : (Tmax, N)
    debug : dict
    """

    N = SC.shape[0]

    # Initial condition
    Se = 0.001 * np.ones(N)
    Si = 0.001 * np.ones(N)

    # -------- Burn-in --------
    t = 0.0
    while t < burn_in:
        dSe, dSi, _, dbg = dfun(Se, Si, J, G, SC, Receptor, wgaine, wgaini)
        Se = Se + dt * dSe + np.sqrt(dt) * sigma * randn(N)
        Si = Si + dt * dSi + np.sqrt(dt) * sigma * randn(N)
        Se[Se > 1] = 1
        Se[Se < 0] = 0
        Si[Si > 1] = 1
        Si[Si < 0] = 0

        t += dt

    # -------- Main simulation --------
    debug = {
        "Se_samples": [],
        "Si_samples": [],
    }

    Tmaxneuronal = int((Tmax + 10) * (TR / dtt))  # Number of simulated time points (milliseconds)
    sim_signal = np.zeros((N, Tmaxneuronal+1))
    t = 0.0
    while t < Tmaxneuronal:  # loop in milliseconds
        dSe, dSi, He, dbg = dfun(Se, Si, J, G, SC, Receptor, wgaine, wgaini)
        Se = Se + dt * dSe + np.sqrt(dt) * sigma * randn(N,1).flatten()
        Si = Si + dt * dSi + np.sqrt(dt) * sigma * randn(N,1).flatten()
        Se[Se > 1] = 1
        Se[Se < 0] = 0
        Si[Si > 1] = 1
        Si[Si < 0] = 0

        # Store debug samples (few only)
        if len(debug["Se_samples"]) < 5:
            debug["Se_samples"].append(Se.copy())
            debug["Si_samples"].append(Si.copy())

        # Sampling condition (same logic as MATLAB)
        if nT.isInt(t/1.):  # keep one sample every millisecond
            sim_signal[:, round(t)] = He  # He component

        t += dt

    sim_signal = sim_signal[:,:-1]  # remove the last pesky entry
    sim_signal = sim_signal.T  # time x regions

    # %%%%BOLD
    # Friston BALLOON - WINDKESSEL MODEL
    # T = nn * dtt # Total time in seconds
    # B = BOLD(T, neuro_act(:, 1)') # B=BOLD activity, bf=Foutrier transform, f=frequency range)
    bold_sim = BOLD.BoldStephan2007Alt(tr=TR*1000.).configure()  # TR in milliseconds
    bold_signal = bold_sim.compute_bold(sim_signal, dt=1.)  # dt in milliseconds
    #
    # for nnew = 2:N
    #   B = BOLD(T, neuro_act(:, nnew));
    #   BOLD_act(:, nnew) = B;
    #
    # bds = BOLD_act(TR / dtt:TR / dtt: end,:);

    return sim_signal, bold_signal, debug
