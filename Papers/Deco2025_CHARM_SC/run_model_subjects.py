"""
Deco2025_CHARM_SC/examples/run_model_subjects.py
----------------------------------------
Reproduces Model_subjects.m from Deco et al. (2025).

Compares the STATIONARY DISTRIBUTION of HARM and CHARM-SC against the
EMPIRICAL parcel activation distribution computed from BOLD data.

Original MATLAB               →   This file
------------------------------    --------------------------------
load schaefercog.mat          →   DL.get_parcellation().get_CoGs()
load hcp_REST_LR_schaefer1000 →   DL (HCP DataLoader)
filtfilt + events pipeline    →   EmpiricalTransitionMatrix.compute()
%% HARM SC block               →   HARM.fit(coords).stationary_distribution()
%% CHARM SC block              →   CHARM_SC.fit(coords).stationary_distribution()
-log(nansum(sqrt(p.*q)))      →   bhattacharyya_distance(p, q)
corrcoef(p, q)                →   scipy.stats.pearsonr(p, q)
boxplot + ranksum              →   matplotlib violinplot + scipy.stats.ranksums

Demo parameters (reduced from paper for speed):
    NSUB      = 1003  → n_subjects = 20
    NGroup    = 20    → group_size = 5   (subjects per group)
    N         = 1000  (unchanged — determined by parcellation)
    epsilon   = 1400  (unchanged)
    Thorizont = 2     (unchanged)
    P^50      (unchanged)

Usage
-----
    python examples/run_model_subjects.py

Output
------
    _Data_Produced/results_model_subjects.mat
    _Data_Produced/model_subjects_comparison.pdf
"""

import os
import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CHARMsc library ───────────────────────────────────────────────────────────
from geometry import HARM, CHARM_SC
from empirical.transition_matrix import EmpiricalTransitionMatrix

# ── Our existing data infrastructure ────────────────────────────────────────
from DataLoaders.HCP_Schaefer2018 import HCP


# =============================================================================
# Parameters
# =============================================================================

# Demo values (reduced from paper for speed)
# To reproduce paper results exactly: n_subjects=1000, group_size=20
N_SUBJECTS  = 20    # paper: 1000
GROUP_SIZE  = 5     # paper: 20  (subjects per group for Pstatesemp)
EPSILON     = 1400.0
T_HORIZON   = 2
DIFF_STEPS  = 50
OUTPUT_DIR  = '_Data_Produced'

# Parcels to exclude: 0-indexed [554, 907] = MATLAB 1-indexed [555, 908]
# These two parcels have NaN BOLD in the Schaefer 1000 atlas.
EXCLUDE_PARCELS = [554, 907]


# =============================================================================
# Helper: Bhattacharyya distance
# Corresponds to MATLAB: -log(nansum(sqrt(p .* q)))
# =============================================================================

def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Bhattacharyya distance between two probability distributions.

    BD = -log( sum_i sqrt(p_i * q_i) )

    Lower = more similar distributions. 0 = identical.
    Matches the MATLAB KLfitt computation.

    Parameters
    ----------
    p, q : np.ndarray, shape (N,)
        Non-negative arrays (need not sum to 1, but should).

    Returns
    -------
    float
    """
    bc = np.nansum(np.sqrt(p * q))
    return float(-np.log(bc)) if bc > 0 else np.inf


# =============================================================================
# Data loading
# =============================================================================

def load_data(n_subjects: int) -> tuple[HCP, np.ndarray]:
    """
    Load HCP data and parcel centroid coordinates.

    Returns
    -------
    DL : HCP
        Fitted data loader.
    coords : np.ndarray, shape (N, 3)
        Parcel centroid coordinates (SchaeferCOG in MATLAB).
    """
    print("Loading HCP Schaefer1000 data...")
    DL     = HCP(SchaeferSize=1000)
    parc   = DL.get_parcellation()
    coords = parc.get_CoGs()              # (N, 3)  — SchaeferCOG in MATLAB
    print(f"  N parcels : {coords.shape[0]}")
    print(f"  Subjects  : {len(DL.get_groupSubjects('REST1'))}")
    return DL, coords


# =============================================================================
# Geometry: fit HARM and CHARM-SC on parcel coordinates
# =============================================================================

def fit_geometry(coords: np.ndarray) -> tuple[HARM, CHARM_SC]:
    """
    Fit both HARM and CHARM-SC geometry models on parcel centroids.

    Returns
    -------
    harm : HARM      (fitted)
    charm : CHARM_SC (fitted)
    """
    print("\nFitting HARM geometry model...")
    harm = HARM(
        epsilon         = EPSILON,
        t_horizon       = T_HORIZON,
        diffusion_steps = DIFF_STEPS,
        exclude_parcels = EXCLUDE_PARCELS,
    )
    harm.fit(coords)
    print(f"  {harm}")

    print("Fitting CHARM-SC geometry model...")
    charm = CHARM_SC(
        epsilon         = EPSILON,
        t_horizon       = T_HORIZON,
        diffusion_steps = DIFF_STEPS,
        exclude_parcels = EXCLUDE_PARCELS,
    )
    charm.fit(coords)
    print(f"  {charm}")

    return harm, charm


# =============================================================================
# Empirical: compute Pstatesemp per group of subjects
# =============================================================================

def compute_empirical_distributions(
    DL:         HCP,
    n_subjects: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical parcel state distributions for each subject group.

    Divides the first n_subjects into groups of group_size, computes
    Pstatesemp for each group, and returns arrays of per-group metrics.

    Returns
    -------
    pstates_emp : np.ndarray, shape (n_groups, N_valid)
        Per-group empirical distributions.
    kl_harm  : np.ndarray, shape (n_groups,)   — placeholder, filled later
    kl_charm : np.ndarray, shape (n_groups,)   — placeholder, filled later
    """
    emp_computer = EmpiricalTransitionMatrix(
        tr_seconds      = DL.TR(),
        flp             = 0.008,
        fhi             = 0.08,
        cut             = 50,
        diffusion_steps = DIFF_STEPS,
        exclude_parcels = EXCLUDE_PARCELS,
    )

    subjects   = DL.get_groupSubjects('REST1')[:n_subjects]
    n_groups   = max(1, len(subjects) // group_size)
    print(f"\nComputing empirical distributions for {n_groups} groups "
          f"of {group_size} subjects each...")

    pstates_emp = []
    for g in range(n_groups):
        group_subjs = subjects[g * group_size: (g + 1) * group_size]
        print(f"  Group {g + 1}/{n_groups} ({len(group_subjs)} subjects)...")

        # Load timeseries for this group: each is (N, T) in Neuroreduce convention
        ts_list = [
            DL.get_subjectData(subj)[subj]['timeseries']
            for subj in group_subjs
        ]
        p_emp = emp_computer.compute(ts_list)   # (N_valid,)
        pstates_emp.append(p_emp)

    return np.array(pstates_emp)   # (n_groups, N_valid)


# =============================================================================
# Comparison: Bhattacharyya distance and Pearson r
# =============================================================================

def compare_distributions(
    pstates_emp:   np.ndarray,
    p_harm:        np.ndarray,
    p_charm:       np.ndarray,
) -> dict:
    """
    Compute per-group Bhattacharyya distance and Pearson correlation
    between empirical and geometric stationary distributions.

    Matches the MATLAB computation:
        KLfitt(nsub)  = -log(nansum(sqrt(Pstatesemp .* Pstates)))
        Corrfitt(nsub) = corrcoef(Pstates, Pstatesemp)

    Parameters
    ----------
    pstates_emp : np.ndarray, shape (n_groups, N_valid)
    p_harm      : np.ndarray, shape (N_valid,)
    p_charm     : np.ndarray, shape (N_valid,)

    Returns
    -------
    dict with keys:
        'kl_harm', 'kl_charm'   : np.ndarray, shape (n_groups,)
        'corr_harm', 'corr_charm': np.ndarray, shape (n_groups,)
    """
    n_groups = len(pstates_emp)
    kl_harm   = np.zeros(n_groups)
    kl_charm  = np.zeros(n_groups)
    corr_harm = np.zeros(n_groups)
    corr_charm = np.zeros(n_groups)

    for g, p_emp in enumerate(pstates_emp):
        kl_harm[g]    = bhattacharyya_distance(p_emp, p_harm)
        kl_charm[g]   = bhattacharyya_distance(p_emp, p_charm)
        corr_harm[g], _  = stats.pearsonr(p_harm,  p_emp)
        corr_charm[g], _ = stats.pearsonr(p_charm, p_emp)

    return {
        'kl_harm':    kl_harm,
        'kl_charm':   kl_charm,
        'corr_harm':  corr_harm,
        'corr_charm': corr_charm,
    }


# =============================================================================
# Plotting: violin plots + Wilcoxon rank-sum test
# =============================================================================

def plot_results(metrics: dict, output_dir: str) -> None:
    """
    Produce violin plots comparing HARM vs CHARM-SC.
    Corresponds to the boxplot + ranksum figures in Model_subjects.m.

    Figure 1: Bhattacharyya distance (lower = better)
    Figure 2: Pearson correlation    (higher = better)
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_key, ylabel, title, lower_is_better in [
        ('kl',   'Bhattacharyya distance', 'State Distribution Fit',    True),
        ('corr', 'Pearson r',              'Correlation with Empirical', False),
    ]:
        harm_vals  = metrics[f'{metric_key}_harm']
        charm_vals = metrics[f'{metric_key}_charm']

        stat, p = stats.ranksums(harm_vals, charm_vals)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        fig, ax = plt.subplots(figsize=(5, 5))

        vp = ax.violinplot([harm_vals, charm_vals],
                           positions=[1, 2], widths=0.6,
                           showmedians=True, showextrema=True)
        vp['bodies'][0].set_facecolor('#5B9BD5')
        vp['bodies'][1].set_facecolor('#F4A261')
        for body in vp['bodies']:
            body.set_alpha(0.7)
            body.set_edgecolor('white')
        for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            vp[part].set_color('0.3')

        # Overlay individual points
        rng = np.random.default_rng(0)
        for i, vals in enumerate([harm_vals, charm_vals], start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(i + jitter, vals, s=30, alpha=0.8,
                       color=['#1F5FAC', '#C1440E'][i - 1],
                       edgecolors='white', linewidths=0.4, zorder=3)

        # Significance annotation
        y_max = max(np.max(harm_vals), np.max(charm_vals))
        y_sig = y_max + 0.05 * abs(y_max)
        ax.plot([1, 2], [y_sig, y_sig], 'k-', linewidth=1.0)
        ax.text(1.5, y_sig, stars, ha='center', va='bottom', fontsize=11)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['HARM', 'CHARM-SC'], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(
            f'{title}\n(n={len(harm_vals)} groups, '
            f'Wilcoxon p={p:.3f})',
            fontsize=10,
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2, linestyle=':')

        path = f'{output_dir}/model_subjects_{metric_key}.pdf'
        fig.savefig(path, bbox_inches='tight', dpi=150)
        fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Model Subjects: HARM vs CHARM-SC stationary distributions")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    DL, coords = load_data(N_SUBJECTS)

    # 2. Fit geometry models (no BOLD needed — pure geometry)
    harm, charm = fit_geometry(coords)
    p_harm  = harm.stationary_distribution()    # (N_valid,)
    p_charm = charm.stationary_distribution()   # (N_valid,)
    print(f"\n  HARM  stationary distribution: "
          f"min={p_harm.min():.4e}  max={p_harm.max():.4e}")
    print(f"  CHARM stationary distribution: "
          f"min={p_charm.min():.4e}  max={p_charm.max():.4e}")

    # 3. Compute empirical distributions from BOLD
    pstates_emp = compute_empirical_distributions(DL, N_SUBJECTS, GROUP_SIZE)

    # 4. Compare
    print("\nComputing Bhattacharyya distances and correlations...")
    metrics = compare_distributions(pstates_emp, p_harm, p_charm)

    print(f"\n  HARM   Bhattacharyya : {metrics['kl_harm'].mean():.4f} ± {metrics['kl_harm'].std():.4f}")
    print(f"  CHARM  Bhattacharyya : {metrics['kl_charm'].mean():.4f} ± {metrics['kl_charm'].std():.4f}")
    print(f"  HARM   Pearson r     : {metrics['corr_harm'].mean():.4f} ± {metrics['corr_harm'].std():.4f}")
    print(f"  CHARM  Pearson r     : {metrics['corr_charm'].mean():.4f} ± {metrics['corr_charm'].std():.4f}")

    # 5. Plot
    print("\nGenerating plots...")
    plot_results(metrics, OUTPUT_DIR)

    # 6. Save
    out_path = f'{OUTPUT_DIR}/results_model_subjects.mat'
    sio.savemat(out_path, {
        'KLfitt':       metrics['kl_harm'],
        'KLfittC':      metrics['kl_charm'],
        'Corrfitt':     metrics['corr_harm'],
        'CorrfittC':    metrics['corr_charm'],
        'Pstatesempm':  np.nanmean(pstates_emp, axis=0),
        'Pstatesm':     p_harm,
        'PstatesCm':    p_charm,
    })
    print(f"  Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
