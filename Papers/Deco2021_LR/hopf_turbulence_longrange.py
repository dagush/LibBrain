# =======================================================================
# hopf_turbulence_longrange.py
#
# Python translation of slurm_sbatch_hopf_turbu_longrange.m
# (Deco et al., Current Biology, 2021)
#
# Sweeps G from 0 to 3 in steps of 0.01 and, for each G value, runs
# NSUBSIM independent Hopf simulations under four connectivity conditions:
#
#   'EDR'     -- plain exponential distance rule
#   'EDR_LR'  -- EDR + empirical long-range connections (Clong)
#   'EDR_RND' -- EDR + spatially shuffled long-range connections (null)
#   'SC'      -- empirical structural connectivity
#
# Results are saved to a long-format CSV:  one row per (G, sub, condition).
#
# Gustavo Patow, 2025
# =======================================================================

import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from compact_bold_simulator import CompactHopfSimulator

# -----------------------------------------------------------------------
# Observables — replace placeholders with real imports when available
# -----------------------------------------------------------------------
from neuronumba.observables.turbulence import Turbulence
from neuronumba.observables.distance_rule import EDR_LR_distance_rule

# PLACEHOLDER: Functional Connectivity observable
# Replace with your real FC class, e.g.:
#   from neuronumba.observables.fc import FC
class FC:
    """Placeholder — replace with the real FC observable from your library."""
    def from_fmri(self, bold_signal):
        # bold_signal: (T, N) → returns (N, N) Pearson FC matrix
        return np.corrcoef(bold_signal.T)

# PLACEHOLDER: Long-range FC observable
# Receives CoGs in the constructor; returns mean |FC| over long-range pairs.
class LR_FC:
    """
    Placeholder — computes mean absolute FC restricted to node pairs whose
    Euclidean distance exceeds dist_threshold_mm (default 40 mm, matching
    the MATLAB script's IndLong criterion).

    Replace or extend with a proper observable subclass when ready.
    """
    def __init__(self, cog_dist: np.ndarray, dist_threshold_mm: float = 40.0):
        N = cog_dist.shape[0]
        rr = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                rr[i, j] = np.linalg.norm(cog_dist[i] - cog_dist[j])
        self._long_range_mask = rr > dist_threshold_mm

    def from_fmri(self, fc_matrix: np.ndarray) -> float:
        """Return mean |FC| over long-range pairs."""
        return float(np.nanmean(np.abs(fc_matrix[self._long_range_mask])))

# PLACEHOLDER: Information cascade / mutual information observable
# Replace with your real implementation when available.
class InfoCascade:
    """
    Placeholder — computes the cross-scale information cascade from the
    enstrophy matrices at multiple lambda scales.

    Returns
    -------
    mutinf1 : np.ndarray, shape (n_lambdas,)
        Mean significant correlation between enstrophy at scale ilam and
        ilam-1 (first element is NaN, matching the MATLAB convention).
    mutinfo_p : np.ndarray, shape (NFUTURE,)
        Mean significant temporal correlation of enstrophy at future lags.
    """
    def __init__(self, n_future: int = 10):
        self.n_future = n_future

    def compute(self,
                lam_mean_spatime_enstrophy: np.ndarray,
                indsca: int) -> dict:
        """
        Parameters
        ----------
        lam_mean_spatime_enstrophy : np.ndarray, shape (n_lambdas, N, T)
        indsca : int
            Index of the lambda closest to the model lambda.

        Returns
        -------
        dict with keys 'mutinf1' and 'mutinfo_p'.
        """
        n_lambdas, N, T = lam_mean_spatime_enstrophy.shape

        # Cross-scale cascade
        mutinf1 = np.full(n_lambdas, np.nan)
        for ilam in range(1, n_lambdas):
            A = lam_mean_spatime_enstrophy[ilam, :, 1:].T     # (T-1, N)
            B = lam_mean_spatime_enstrophy[ilam-1, :, :-1].T  # (T-1, N)
            cc_vals = []
            for ni in range(N):
                r, p = pearsonr(A[:, ni], B[:, ni])
                if p < 0.05:
                    cc_vals.append(r)
            mutinf1[ilam] = np.nanmean(cc_vals) if cc_vals else np.nan

        # Temporal future-lag correlations at the model lambda
        E = lam_mean_spatime_enstrophy[indsca]   # (N, T)
        mutinfo_p = np.full(self.n_future, np.nan)
        for ifut in range(self.n_future):
            lag = ifut + 1
            A = E[:, lag:].T      # (T-lag, N)
            B = E[:, :T-lag].T    # (T-lag, N)
            cc_vals = []
            for ni in range(N):
                r, p = pearsonr(A[:, ni], B[:, ni])
                if p < 0.05:
                    cc_vals.append(r)
            mutinfo_p[ifut] = np.nanmean(cc_vals) if cc_vals else np.nan

        return {'mutinf1': mutinf1, 'mutinfo_p': mutinfo_p}


# -----------------------------------------------------------------------
# Helper: spatial FC-vs-distance profile
# -----------------------------------------------------------------------

def compute_spatial_fc_profile(fc_matrix: np.ndarray,
                                rr: np.ndarray,
                                NR: int) -> np.ndarray:
    """
    Bin the FC matrix by distance and return the per-node FC-vs-distance
    profile, exactly as in the MATLAB script.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (N, N)
    rr : np.ndarray, shape (N, N)   pairwise distances
    NR : int                         number of distance bins

    Returns
    -------
    corrfcn : np.ndarray, shape (N, NR)
        Mean FC per node per distance bin.
    """
    N = fc_matrix.shape[0]
    dist_range = rr.max()
    delta = dist_range / NR
    corrfcn = np.zeros((N, NR))

    for i in range(N):
        num = np.zeros(NR)
        acc = np.zeros(NR)
        for j in range(N):
            idx = int(rr[i, j] / delta)
            if idx >= NR:
                idx = NR - 1
            v = fc_matrix[i, j]
            if not np.isnan(v):
                acc[idx] += v
                num[idx] += 1
        mask = num > 0
        corrfcn[i, mask] = acc[mask] / num[mask]
        corrfcn[i, ~mask] = np.nan

    return corrfcn


# -----------------------------------------------------------------------
# Helper: spatial heterogeneity error
# -----------------------------------------------------------------------

def spatial_heterogeneity_error(corrfcn_sim: np.ndarray,
                                corrfcn_emp: np.ndarray,
                                NRini: int,
                                NRfin: int) -> float:
    """
    RMSE between simulated and empirical spatial FC profiles, averaged
    over nodes and over the bin range [NRini, NRfin] (0-based, inclusive).
    Matches the MATLAB err_hete computation.
    """
    # Slice the relevant bin range
    sim = corrfcn_sim[:, NRini:NRfin+1]
    emp = corrfcn_emp[:, NRini:NRfin+1]
    err_per_node = np.nanmean((sim - emp) ** 2, axis=1)
    return float(np.sqrt(np.nanmean(err_per_node)))


# -----------------------------------------------------------------------
# Helper: RSN (Yeo-7) FC error
# -----------------------------------------------------------------------

def rsn_fc_errors(fc_matrix: np.ndarray,
                  fc_emp: np.ndarray,
                  yeo7_vector: np.ndarray) -> np.ndarray:
    """
    Compute per-network RMSE between simulated and empirical FC,
    restricted to within-network lower-triangle pairs.

    Parameters
    ----------
    fc_matrix : (N, N)
    fc_emp    : (N, N)
    yeo7_vector : (N,) integer array with values 1..7

    Returns
    -------
    errors : np.ndarray, shape (7,)
    """
    errors = np.full(7, np.nan)
    for net in range(1, 8):
        ind = np.where(yeo7_vector == net)[0]
        if len(ind) < 2:
            continue
        tril_idx = np.tril_indices(len(ind), k=-1)
        sim_vals = fc_matrix[np.ix_(ind, ind)][tril_idx]
        emp_vals = fc_emp[np.ix_(ind, ind)][tril_idx]
        errors[net - 1] = float(np.sqrt(np.nanmean((emp_vals - sim_vals) ** 2)))
    return errors


# -----------------------------------------------------------------------
# Helper: Ctotrnd — spatially shuffled null model
# -----------------------------------------------------------------------

def make_Ctotrnd(C_edr: np.ndarray, Clong: np.ndarray) -> np.ndarray:
    """
    Build the null-model connectivity matrix by permuting the long-range
    connection weights within each hemisphere separately, then replacing
    the corresponding entries in the plain EDR matrix.

    Matches the MATLAB construction of Ctotrnd exactly:
    - hemisphere 1: rows/cols 0..499
    - hemisphere 2: rows/cols 500..999
    - cross-hemisphere connections (Cxhemi) are kept unchanged from Ctot

    Parameters
    ----------
    C_edr  : (N, N) plain EDR matrix (after factor normalisation)
    Clong  : (N, N) long-range connection matrix

    Returns
    -------
    Ctotrnd : (N, N)
    """
    N = C_edr.shape[0]
    half = N // 2
    Ctotrnd = C_edr.copy()

    # Keep cross-hemisphere long-range connections as in Ctot
    Clong_cross_1 = Clong[:half, half:]
    Clong_cross_2 = Clong[half:, :half]
    Ctotrnd[:half, half:] = np.where(Clong_cross_1 > 0,
                                      Clong_cross_1,
                                      C_edr[:half, half:])
    Ctotrnd[half:, :half] = np.where(Clong_cross_2 > 0,
                                      Clong_cross_2,
                                      C_edr[half:, :half])

    # Permute within-hemisphere long-range weights for each hemisphere
    for h_slice in [slice(0, half), slice(half, N)]:
        Clong_hemi = Clong[h_slice, h_slice]
        C_edr_hemi = C_edr[h_slice, h_slice]

        tril_idx = np.tril_indices(half, k=-1)
        long_vec = Clong_hemi[tril_idx]

        # Permute with the same random permutation for both hemispheres
        # (matches MATLAB's single indexpermuted reused for hemi2)
        perm = np.random.permutation(len(long_vec))
        long_vec_rnd = long_vec[perm]

        Clong_rnd = np.zeros((half, half))
        Clong_rnd[tril_idx] = long_vec_rnd
        Clong_rnd = Clong_rnd + Clong_rnd.T   # symmetrise

        hemi_rnd = C_edr_hemi.copy()
        mask = Clong_rnd > 0
        hemi_rnd[mask] = Clong_rnd[mask]
        Ctotrnd[h_slice, h_slice] = hemi_rnd

    return Ctotrnd


# -----------------------------------------------------------------------
# Helper: enstrophy across multiple lambda scales
# -----------------------------------------------------------------------

def compute_multi_lambda_enstrophy(phases: np.ndarray,
                                   cog_dist: np.ndarray,
                                   lambdas: np.ndarray) -> np.ndarray:
    """
    Compute the Kuramoto local order parameter (enstrophy) for each
    lambda in `lambdas`, using a fresh EDR kernel per lambda.

    Parameters
    ----------
    phases   : (N, T)  instantaneous phases from Hilbert transform
    cog_dist : (N, 3)  CoG coordinates
    lambdas  : (L,)    array of lambda values

    Returns
    -------
    lam_enstrophy : (L, N, T)
    """
    N, T = phases.shape
    L = len(lambdas)
    lam_enstrophy = np.zeros((L, N, T))

    # Build distance matrix once
    rr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            rr[i, j] = np.linalg.norm(cog_dist[i] - cog_dist[j])

    exp_phases = np.exp(1j * phases)   # (N, T)

    for ilam, lam in enumerate(lambdas):
        C1 = np.exp(-lam * rr)          # (N, N)
        row_sums = C1.sum(axis=1)        # (N,)
        for i in range(N):
            weighted = (C1[i, :, np.newaxis] * exp_phases).sum(axis=0)
            lam_enstrophy[ilam, i, :] = np.abs(weighted / row_sums[i])

    return lam_enstrophy


# -----------------------------------------------------------------------
# Single-condition simulation
# -----------------------------------------------------------------------

def run_condition(weights: np.ndarray,
                  omega: np.ndarray,
                  g: float,
                  a_val,           # float or np.ndarray (N,)
                  TR_ms: float,
                  dt_ms: float,
                  T_max: int,
                  warmup_s: float,
                  sigma: float,
                  bpf) -> np.ndarray:
    """
    Run one Hopf simulation and return the filtered BOLD signal.

    Parameters
    ----------
    weights  : (N, N) connectivity matrix (already scaled by factor/0.2)
    omega    : (N,)   intrinsic frequencies in Hz
    g        : float  global coupling
    a_val    : float or (N,) bifurcation parameter
    TR_ms    : float  repetition time in milliseconds
    dt_ms    : float  integration step in milliseconds
    T_max    : int    number of BOLD timepoints to generate
    warmup_s : float  warmup duration in seconds
    sigma    : float  noise amplitude
    bpf      : band-pass filter object with a .filter(ts) method

    Returns
    -------
    bold_filt : (T_max, N)  filtered BOLD signal
    """
    warmup_samples = int(warmup_s * 1000 / dt_ms)   # seconds → ms → steps
    simulated_samples = int(T_max * TR_ms / dt_ms)

    simulator = CompactHopfSimulator(
        weights=weights,
        use_temporal_avg_monitor=False,
        a=a_val,
        omega=omega,
        g=g,
        sigma=sigma,
        tr=TR_ms,
        dt=dt_ms,
    )

    bold_raw = simulator.generate_bold(
        warmup_samples=warmup_samples,
        simulated_samples=simulated_samples,
    )
    # bold_raw: (T_max, N)
    bold_filt = bpf.filter(bold_raw.T).T   # filter expects (N, T), returns (N, T)
    return bold_filt


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def run():
    # ===================================================================
    # PARAMETERS (matching the MATLAB script)
    # ===================================================================
    NPARCELLS = 1000
    NR        = 400          # distance bins for spatial FC profile
    NRini     = 20           # first bin for err_hete (0-based)
    NRfin     = 380          # last  bin for err_hete (0-based, inclusive)
    NFUTURE   = 10
    NSUBSIM   = 100
    LAMBDAS   = np.array([0.26, 0.22, 0.18, 0.14, 0.10, 0.06, 0.02])

    TR_s      = 0.72         # seconds
    TR_ms     = TR_s * 1000  # milliseconds (for CompactHopfSimulator)
    dt_s      = 0.1 * TR_s / 2.0
    dt_ms     = dt_s * 1000  # milliseconds
    T_max     = 1200         # BOLD timepoints
    warmup_s  = 2000.0       # warmup duration in seconds (MATLAB: t=0:dt:2000)
    sigma     = 0.01         # noise amplitude
    a_fixed   = -0.02        # bifurcation parameter (main runs)

    G_range   = np.round(np.arange(0.0, 3.01, 0.01), 2)   # 0.00 … 3.00

    # ===================================================================
    # DATA LOADING — replace placeholders with real DataLoader calls
    # ===================================================================

    # PLACEHOLDER: load CoG coordinates, shape (NPARCELLS, 3)
    cog_dist = np.zeros((NPARCELLS, 3))       # <-- replace

    # PLACEHOLDER: load normalised SC matrix, shape (NPARCELLS, NPARCELLS)
    SC = np.zeros((NPARCELLS, NPARCELLS))     # <-- replace
    SC /= SC.max() if SC.max() > 0 else 1.0

    # PLACEHOLDER: load Clong (long-range connectivity), shape (NPARCELLS, NPARCELLS)
    # and the lambda value used to produce it
    Clong  = np.zeros((NPARCELLS, NPARCELLS)) # <-- replace
    lambda_val = 0.18                          # <-- replace with value from file

    # PLACEHOLDER: load intrinsic node frequencies, shape (NPARCELLS,) in Hz
    f_diff = np.full(NPARCELLS, 0.05)         # <-- replace
    omega  = 2 * np.pi * f_diff               # angular frequencies (rad/s)

    # PLACEHOLDER: load empirical FC matrix, shape (NPARCELLS, NPARCELLS)
    fce = np.zeros((NPARCELLS, NPARCELLS))    # <-- replace

    # PLACEHOLDER: load empirical spatial FC profile, shape (NPARCELLS, NR)
    corrfcn_emp = np.zeros((NPARCELLS, NR))   # <-- replace

    # PLACEHOLDER: load empirical enstrophy cascade quantities
    Rmeta_emp  = 0.0                          # <-- replace (scalar)
    Inflam_emp = np.zeros(len(LAMBDAS) - 1)  # <-- replace (shape: n_lambdas-1)

    # PLACEHOLDER: Yeo-7 network assignment vector, shape (NPARCELLS,), values 1..7
    yeo7_vector = np.ones(NPARCELLS, dtype=int)  # <-- replace

    # PLACEHOLDER: band-pass filter — replace with your BandPassFilter instance
    # e.g.: bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=TR_s)
    class _BPFPlaceholder:
        def filter(self, ts):   # ts: (N, T) → returns (N, T)
            return ts
    bpf = _BPFPlaceholder()    # <-- replace

    # ===================================================================
    # PRE-COMPUTE EDR KERNEL AND DISTANCE MATRIX
    # ===================================================================
    lr_rule = EDR_LR_distance_rule(
        sc=SC,
        lambda_val=lambda_val,
        NR=144,     # bins used internally for Clong fitting
        NRini=7,
        NRfin=30,
        NSTD=5,
    )
    rr, C_edr = lr_rule.compute(cog_dist)    # plain EDR at fitted lambda

    # Long-range index mask (pairs > 40 mm apart)
    long_range_mask = rr > 40.0

    # Build EDR+LR matrix: replace EDR entries where Clong > 0
    IClong = Clong > 0
    Ctot = C_edr.copy()
    Ctot[IClong] = Clong[IClong]

    # Normalise all matrices to the same scale as MATLAB (factor / 0.2)
    factor = C_edr.max()
    C_edr_norm  = C_edr  / factor * 0.2
    Ctot_norm   = Ctot   / factor * 0.2
    SC_norm     = SC     / factor * 0.2
    # Ctotrnd is built fresh each sub-iteration (random permutation)

    # Index of the lambda closest to lambda_val
    indsca = int(np.argmin(np.abs(LAMBDAS - lambda_val)))

    # ===================================================================
    # OBSERVABLES SETUP
    # ===================================================================
    fc_obs    = FC()
    lr_fc_obs = LR_FC(cog_dist, dist_threshold_mm=40.0)
    info_obs  = InfoCascade(n_future=NFUTURE)

    # Turbulence observable for Kuramoto enstrophy (uses EDR kernel at lambda_val)
    turbu = Turbulence()
    turbu.cog_dist   = cog_dist
    turbu.lambda_val = lambda_val

    # ===================================================================
    # RESULT ACCUMULATION
    # ===================================================================
    records = []

    # ===================================================================
    # G-SWEEP
    # ===================================================================
    for G in G_range:
        print(f"G = {G:.2f}")

        # Per-sub accumulators for info capacity / susceptibility
        # (these are computed across subs at the end of each G, per condition)
        ensspasub      = np.zeros((NSUBSIM, NPARCELLS))
        ensspasub1     = np.zeros((NSUBSIM, NPARCELLS))
        ensspasubtot   = np.zeros((NSUBSIM, NPARCELLS))
        ensspasub1tot  = np.zeros((NSUBSIM, NPARCELLS))
        ensspasubtotrnd  = np.zeros((NSUBSIM, NPARCELLS))
        ensspasub1totrnd = np.zeros((NSUBSIM, NPARCELLS))
        ensspasubSC    = np.zeros((NSUBSIM, NPARCELLS))
        ensspasub1SC   = np.zeros((NSUBSIM, NPARCELLS))

        # ==============================================================
        for sub in range(NSUBSIM):
            print(f"  sub {sub+1}/{NSUBSIM}")

            # ----------------------------------------------------------
            # Build Ctotrnd (fresh random permutation each sub)
            # ----------------------------------------------------------
            Ctotrnd_norm = make_Ctotrnd(C_edr_norm, Clong / factor * 0.2)

            # ----------------------------------------------------------
            # Condition loop — avoids code repetition
            # ----------------------------------------------------------
            conditions = {
                'EDR':     C_edr_norm,
                'EDR_LR':  Ctot_norm,
                'EDR_RND': Ctotrnd_norm,
                'SC':      SC_norm,
            }

            for cond_name, W in conditions.items():

                # ---- Main simulation (fixed a) -----------------------
                bold_filt = run_condition(
                    weights=W, omega=omega, g=G,
                    a_val=a_fixed,
                    TR_ms=TR_ms, dt_ms=dt_ms, T_max=T_max,
                    warmup_s=warmup_s, sigma=sigma, bpf=bpf,
                )
                # bold_filt: (T_max, N)

                # Phases from Hilbert transform
                from scipy import signal as sp_signal
                phases = np.zeros((NPARCELLS, T_max))
                for seed in range(NPARCELLS):
                    x = bold_filt[:, seed]
                    x = x - x.mean()
                    xa = sp_signal.hilbert(x)
                    phases[seed, :] = np.angle(xa)

                # FC
                fc_matrix = fc_obs.from_fmri(bold_filt)   # (N, N)

                # Long-range FC
                fc_long_val = lr_fc_obs.from_fmri(fc_matrix)

                # Spatial FC-vs-distance profile
                corrfcn_sim = compute_spatial_fc_profile(fc_matrix, rr, NR)

                # Spatial heterogeneity error vs empirical
                err_hete_val = spatial_heterogeneity_error(
                    corrfcn_sim, corrfcn_emp, NRini, NRfin
                )

                # Enstrophy across lambda scales
                lam_enstrophy = compute_multi_lambda_enstrophy(
                    phases, cog_dist, LAMBDAS
                )   # (L, N, T)

                Rspatime = lam_enstrophy[indsca]           # (N, T)
                R_meta   = float(np.nanstd(Rspatime))
                enssp    = np.nanmean(Rspatime, axis=1)    # (N,)

                # Store for info_capacity / susceptibility (across subs)
                if cond_name == 'EDR':
                    ensspasub[sub]    = enssp
                elif cond_name == 'EDR_LR':
                    ensspasubtot[sub] = enssp
                elif cond_name == 'EDR_RND':
                    ensspasubtotrnd[sub] = enssp
                elif cond_name == 'SC':
                    ensspasubSC[sub]  = enssp

                # Information cascade
                info_result = info_obs.compute(lam_enstrophy, indsca)
                mutinf1    = info_result['mutinf1']      # (L,)
                mutinfo_p  = info_result['mutinfo_p']    # (NFUTURE,)

                Inflam_val     = float(np.nanmean(mutinf1[1:]))
                Err_Rlam_val   = float(np.sqrt((R_meta - Rmeta_emp) ** 2))
                Err_Inflam_val = float(np.sqrt(np.nanmean(
                    (mutinf1[1:] - Inflam_emp) ** 2
                )))

                # RSN (Yeo-7) FC errors
                rsn_errors = rsn_fc_errors(fc_matrix, fce, yeo7_vector)  # (7,)

                # fcfittlong — RMSE of FC over long-range pairs vs empirical
                fc_long_rmse = float(np.sqrt(np.nanmean(
                    (fce[long_range_mask] - fc_matrix[long_range_mask]) ** 2
                )))

                # ---- Perturbation simulation (heterogeneous a) -------
                a_pert = a_fixed + 0.02 * np.random.rand(NPARCELLS)

                bold_filt_pert = run_condition(
                    weights=W, omega=omega, g=G,
                    a_val=a_pert,
                    TR_ms=TR_ms, dt_ms=dt_ms, T_max=T_max,
                    warmup_s=0.0,   # MATLAB perturbation runs have no warmup
                    sigma=sigma, bpf=bpf,
                )

                phases_pert = np.zeros((NPARCELLS, T_max))
                for seed in range(NPARCELLS):
                    x = bold_filt_pert[:, seed]
                    x = x - x.mean()
                    xa = sp_signal.hilbert(x)
                    phases_pert[seed, :] = np.angle(xa)

                lam_enstrophy_pert = compute_multi_lambda_enstrophy(
                    phases_pert, cog_dist, LAMBDAS
                )
                Rspatime_pert = lam_enstrophy_pert[indsca]  # (N, T)
                enssp_pert    = np.nanmean(Rspatime_pert, axis=1)  # (N,)

                if cond_name == 'EDR':
                    ensspasub1[sub]      = enssp_pert
                elif cond_name == 'EDR_LR':
                    ensspasub1tot[sub]   = enssp_pert
                elif cond_name == 'EDR_RND':
                    ensspasub1totrnd[sub] = enssp_pert
                elif cond_name == 'SC':
                    ensspasub1SC[sub]    = enssp_pert

                # ---- Collect scalar results for this row -------------
                record = {
                    'G':              G,
                    'sub':            sub,
                    'condition':      cond_name,
                    # Metastability
                    'R_meta':         R_meta,
                    'Err_Rlam':       Err_Rlam_val,
                    # Spatial heterogeneity
                    'err_hete':       err_hete_val,
                    # Long-range FC
                    'fc_long':        fc_long_val,
                    'fc_long_rmse':   fc_long_rmse,
                    # Information cascade
                    'Inflam':         Inflam_val,
                    'Err_Inflam':     Err_Inflam_val,
                }

                # mutinfo_p lags as individual columns
                for k in range(NFUTURE):
                    record[f'mutinfo_p_{k+1}'] = float(mutinfo_p[k])

                # InfoCascade per lambda-pair as individual columns
                for k in range(1, len(LAMBDAS)):
                    record[f'InfoCascade_{k}'] = float(mutinf1[k])

                # RSN errors per network
                rsn_names = ['Vis', 'SomMot', 'DorsAttn',
                             'SalVentAttn', 'Limbic', 'Cont', 'Default']
                for net_idx, net_name in enumerate(rsn_names):
                    record[f'CorrRSN_{net_name}'] = float(rsn_errors[net_idx])

                records.append(record)

        # ==============================================================
        # Info capacity and susceptibility (aggregated across subs, per G)
        # Stored as extra rows with sub=-1 to keep the long format
        # ==============================================================
        agg_data = {
            'EDR':     (ensspasub,       ensspasub1),
            'EDR_LR':  (ensspasubtot,    ensspasub1tot),
            'EDR_RND': (ensspasubtotrnd, ensspasub1totrnd),
            'SC':      (ensspasubSC,     ensspasub1SC),
        }
        for cond_name, (ens0, ens1) in agg_data.items():
            mean0 = np.nanmean(ens0, axis=0)   # (N,)
            diff  = ens1 - mean0[np.newaxis, :]

            info_capacity  = float(np.nanmean(np.nanstd(diff, axis=0)))
            susceptibility = float(np.nanmean(np.nanmean(diff, axis=0)))

            records.append({
                'G':              G,
                'sub':            -1,    # sentinel: aggregate row
                'condition':      cond_name,
                'info_capacity':  info_capacity,
                'susceptibility': susceptibility,
            })

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    df = pd.DataFrame(records)
    out_path = 'hopf_turbulence_longrange_results.csv'
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}  ({len(df)} rows)")


if __name__ == '__main__':
    run()
