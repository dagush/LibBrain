"""
examples/plot_ecm.py
---------------------
Produces ECM violin and scatter plots comparing PCA and CHARM.

Reproduces the right panel of Figure 2(b) from Deco et al. (2025):
    violin plots showing the correlation between source-space ECM and
    reconstructed-space ECM, across subjects and conditions.

Pipeline per subject, per method:
    1. Slice X_sub from the concatenated BOLD      (N × Tmsub)
    2. Reduce:   Z_sub  = reducer.transform(X_sub) (k × Tmsub)
    3. Invert:   X_hat  = reducer.inverse_transform(Z_sub)  (N × Tmsub)
    4. ECM_source[sub]       = compute_ecm(X_sub)
    5. ECM_reconstructed[sub] = compute_ecm(X_hat)
    6. Pearson r(ECM_source, ECM_reconstructed) across subjects → violin annotation

Usage
-----
    python examples/plot_ecm.py

Output
------
    _Results/ecm_violins.pdf
    _Results/ecm_violins.png
    _Results/ecm_scatter.pdf
"""

import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Neuroreduce library ───────────────────────────────────────────────────────
from Neuroreduce import PCAReducer, CHARMReducer
from Neuroreduce.utils.charm_analysis import CHARMAnalysis
from Neuroreduce.utils.ecm import compute_reconstructed_ecm_per_subject
from Neuroreduce.utils.ecm_plot import ECMPlotter

# ── your existing data infrastructure ────────────────────────────────────────
import DataLoaders.HCP_dbs80 as HCP
from neuronumba.tools.filters import BandPassFilter


# =============================================================================
# Parameters — must match run_charm.py exactly
# =============================================================================

N_regions  = 80
LATDIM     = 7
Tmax       = 175
CUT        = 10
Tmsub      = Tmax + 1 - 2 * CUT   # = 156 timepoints per subject
NSUB       = 10
Tau        = 3
epsilon    = 300
Thorizont  = 2
OUTPUT_DIR = '_Results'


# =============================================================================
# Data loading
# =============================================================================

def load_data() -> tuple[np.ndarray, int, int]:
    """
    Load, filter, z-score and concatenate BOLD for both conditions.

    Returns
    -------
    ts : np.ndarray, shape (N_regions, Tm_total)
        Concatenated z-scored BOLD. REST subjects first, then TASK.
    rest_offset : int
        Column in ts where REST group starts (always 0).
    task_offset : int
        Column in ts where TASK (EMOTION) group starts.
    """
    print("Loading and preprocessing data...")
    DL  = HCP.HCP(chosenDatasets=['REST1', 'EMOTION'])
    DL.discardSubject((553, 'EMOTION'))
    bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08,
                         tr=DL.TR(), remove_artifacts=False)

    ts      = np.empty((N_regions, 0))
    offsets = {}
    for group in ['REST1', 'EMOTION']:
        offsets[group] = ts.shape[1]
        subjects = DL.get_groupSubjects(group)[:NSUB]
        for subj in subjects:
            ts2         = DL.get_subjectData(subj)[subj]['timeseries'][:, :Tmax]
            signal_filt = bpf.filter(ts2.T).T
            tss         = signal_filt[:, CUT - 1:-CUT]
            ts          = np.concatenate((ts, tss), axis=1)

    ts = stats.zscore(ts, axis=1)
    print(f"  BOLD shape: {ts.shape}  "
          f"(REST offset={offsets['REST1']}, TASK offset={offsets['EMOTION']})")
    return ts, offsets['REST1'], offsets['EMOTION']


# =============================================================================
# ECM computation helpers
# =============================================================================

def compute_pca_ecm(
    ts:          np.ndarray,
    rest_offset: int,
    task_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCAReducer and compute source + reconstructed ECM per subject.

    The PCA reconstruction is:
        X_hat = W @ W.T @ X   where W = principal components (N × k)

    Returns
    -------
    ecm_src_rest, ecm_src_task : source ECM (method-independent)
    ecm_recon_rest, ecm_recon_task : ECM of PCA-reconstructed BOLD
    """
    print("\nFitting PCAReducer...")
    reducer_pca = PCAReducer(k=LATDIM, whiten=False)
    reducer_pca.fit(ts)

    print("  Computing PCA ECM (source + reconstructed) for REST...")
    ecm_src_rest, ecm_recon_pca_rest = compute_reconstructed_ecm_per_subject(
        X             = ts,
        reducer       = reducer_pca,
        n_subjects    = NSUB,
        t_per_subject = Tmsub,
        group_offset  = rest_offset,
    )

    print("  Computing PCA ECM (source + reconstructed) for TASK...")
    ecm_src_task, ecm_recon_pca_task = compute_reconstructed_ecm_per_subject(
        X             = ts,
        reducer       = reducer_pca,
        n_subjects    = NSUB,
        t_per_subject = Tmsub,
        group_offset  = task_offset,
    )

    _report_ecm('PCA', 'REST', ecm_src_rest, ecm_recon_pca_rest)
    _report_ecm('PCA', 'TASK', ecm_src_task, ecm_recon_pca_task)
    return ecm_src_rest, ecm_src_task, ecm_recon_pca_rest, ecm_recon_pca_task


def compute_charm_ecm(
    ts:          np.ndarray,
    rest_offset: int,
    task_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit CHARMReducer and compute source + reconstructed ECM per subject.

    The CHARM reconstruction is:
        X_hat = conet @ Phi.T   (approximately — conet maps latent → parcel)

    Note: source ECM from PCA and CHARM will be identical because it is
    computed on the same raw BOLD slices. We return it here for completeness
    and as a cross-check; callers should verify both match.

    Returns
    -------
    ecm_src_rest, ecm_src_task : source ECM
    ecm_recon_rest, ecm_recon_task : ECM of CHARM-reconstructed BOLD
    """
    print("\nFitting CHARMReducer...")
    reducer_charm = CHARMReducer(
        k=LATDIM, epsilon=epsilon, t_horizon=Thorizont,
        whiten=False, sort_eigenvectors=True,
    )
    reducer_charm.fit(ts)

    print("  Computing CHARM ECM (source + reconstructed) for REST...")
    ecm_src_rest, ecm_recon_charm_rest = compute_reconstructed_ecm_per_subject(
        X             = ts,
        reducer       = reducer_charm,
        n_subjects    = NSUB,
        t_per_subject = Tmsub,
        group_offset  = rest_offset,
    )

    print("  Computing CHARM ECM (source + reconstructed) for TASK...")
    ecm_src_task, ecm_recon_charm_task = compute_reconstructed_ecm_per_subject(
        X             = ts,
        reducer       = reducer_charm,
        n_subjects    = NSUB,
        t_per_subject = Tmsub,
        group_offset  = task_offset,
    )

    _report_ecm('CHARM', 'REST', ecm_src_rest, ecm_recon_charm_rest)
    _report_ecm('CHARM', 'TASK', ecm_src_task, ecm_recon_charm_task)
    return ecm_src_rest, ecm_src_task, ecm_recon_charm_rest, ecm_recon_charm_task


def _report_ecm(
    method: str, condition: str,
    ecm_src: np.ndarray, ecm_recon: np.ndarray,
) -> None:
    """Print a brief ECM summary and Pearson r."""
    r, p = stats.pearsonr(ecm_src, ecm_recon) if len(ecm_src) >= 3 else (np.nan, np.nan)
    print(f"  {method} {condition}:")
    print(f"    Source ECM      mean ± std : {ecm_src.mean():.3f} ± {ecm_src.std():.3f}")
    print(f"    Recon  ECM      mean ± std : {ecm_recon.mean():.3f} ± {ecm_recon.std():.3f}")
    print(f"    Pearson r(source, recon)   : {r:.3f}  (p={p:.3f})")


# =============================================================================
# Plotting
# =============================================================================

def make_ecm_plots(
    ecm_source_rest:       np.ndarray,
    ecm_source_task:       np.ndarray,
    ecm_recon_charm_rest:  np.ndarray,
    ecm_recon_charm_task:  np.ndarray,
    ecm_recon_pca_rest:    np.ndarray,
    ecm_recon_pca_task:    np.ndarray,
    group_labels:          tuple[str, str] = ('REST', 'EMOTION'),
    output_dir:            str = OUTPUT_DIR,
) -> None:
    """
    Produce and save ECM violin and scatter plots.

    Can be called independently with pre-computed arrays.
    """
    os.makedirs(output_dir, exist_ok=True)

    plotter = ECMPlotter(
        ecm_source_rest      = ecm_source_rest,
        ecm_source_task      = ecm_source_task,
        ecm_recon_charm_rest = ecm_recon_charm_rest,
        ecm_recon_charm_task = ecm_recon_charm_task,
        ecm_recon_pca_rest   = ecm_recon_pca_rest,
        ecm_recon_pca_task   = ecm_recon_pca_task,
        group_labels         = group_labels,
    )

    # ── Violin plot ───────────────────────────────────────────────────────────
    print("\nGenerating ECM violin plot...")
    fig_v = plotter.plot_ecm_violins(
        title = (f'ECM: Original vs Reconstructed BOLD — PCA vs CHARM\n'
                 f'{group_labels[0]} vs {group_labels[1]}  '
                 f'(n={len(ecm_source_rest)} subjects, k={LATDIM} dims)'),
    )
    for ext in ('pdf', 'png'):
        path = f'{output_dir}/ecm_violins.{ext}'
        fig_v.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig_v)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    print("Generating ECM scatter plot...")
    fig_s = plotter.plot_ecm_scatter(
        title = (f'ECM: Source vs Reconstructed — PCA vs CHARM\n'
                 f'(n={len(ecm_source_rest)} subjects per condition)'),
    )
    path = f'{output_dir}/ecm_scatter.pdf'
    fig_s.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig_s)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ECM Analysis: PCA vs CHARM (source vs reconstructed)")
    print("=" * 60)

    # 1. Load data
    ts, rest_offset, task_offset = load_data()

    # 2. PCA: fit, reconstruct, compute ECM
    ecm_src_rest_pca, ecm_src_task_pca, \
        ecm_recon_pca_rest, ecm_recon_pca_task = compute_pca_ecm(
            ts, rest_offset, task_offset)

    # 3. CHARM: fit, reconstruct, compute ECM
    ecm_src_rest_charm, ecm_src_task_charm, \
        ecm_recon_charm_rest, ecm_recon_charm_task = compute_charm_ecm(
            ts, rest_offset, task_offset)

    # Cross-check: source ECM must be identical between methods
    # (it is computed on the same raw X_sub slices)
    assert np.allclose(ecm_src_rest_pca, ecm_src_rest_charm, atol=1e-6), \
        "Source ECM differs between PCA and CHARM — check slicing!"
    assert np.allclose(ecm_src_task_pca, ecm_src_task_charm, atol=1e-6), \
        "Source ECM differs between PCA and CHARM — check slicing!"
    print("\n  Cross-check passed: source ECM is identical for both methods.")

    # 4. Plot and save
    make_ecm_plots(
        ecm_source_rest      = ecm_src_rest_pca,    # same as charm version
        ecm_source_task      = ecm_src_task_pca,
        ecm_recon_charm_rest = ecm_recon_charm_rest,
        ecm_recon_charm_task = ecm_recon_charm_task,
        ecm_recon_pca_rest   = ecm_recon_pca_rest,
        ecm_recon_pca_task   = ecm_recon_pca_task,
        group_labels         = ('REST', 'EMOTION'),
        output_dir           = OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print(f"Done. Figures saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
