"""
examples/run_charm.py
----------------------
Full CHARM analysis pipeline using Neuroreduce, reproducing the original
``run()`` function from main_CHARM.py step by step.

This file is the canonical example of how to use CHARMReducer +
CHARMAnalysis together. It is intentionally verbose and commented so
that every library call can be traced back to its original counterpart.

Original pipeline (main_CHARM.py)       →   This file
----------------------------------------    --------------------------------
filterAndConcatSubj(ts, 'REST1', NSUB)  →   load_and_concat()
filterAndConcatSubj(ts, 'EMOTION', ...)  →   load_and_concat()
stats.zscore(ts, 1)                      →   stats.zscore(ts, axis=1)
latent(ts)                               →   reducer.fit(ts)
nets(Phi, ts)                            →   reducer.get_basis()
analyze(Phi, offset=0)                   →   analysis.analyze_group(offset=0)
analyze(Phi, offset=Tmsub*NSUB)         →   analysis.analyze_group(offset=...)
classification()                         →   analysis.classification(...)
sio.savemat(...)                         →   sio.savemat(...)  (unchanged)

Usage
-----
    python examples/run_charm.py

Dependencies
------------
    Neuroreduce, numpy, scipy, sklearn, statsmodels
    Your project's DataLoaders and neuronumba packages (unchanged).
"""

import numpy as np
import scipy.io as sio
from scipy import stats

# ── Neuroreduce library ───────────────────────────────────────────────────────
from Neuroreduce import CHARMReducer
from Neuroreduce.utils.charm_analysis import CHARMAnalysis

# ── your existing data infrastructure (unchanged) ─────────────────────────────
import DataLoaders.HCP_dbs80 as HCP
from neuronumba.tools.filters import BandPassFilter


# =============================================================================
# Parameters — identical to the original main_CHARM.py
# These are kept as module-level constants so they can be adjusted in one
# place, just as in the original script.
# =============================================================================

N_regions  = 80    # number of brain parcels (DBS80 atlas)
LATDIM     = 7     # k: number of latent dimensions to retain
Tmax       = 175   # maximum timepoints to load per subject
CUT        = 10    # timepoints to cut from each end after filtering

# Tmsub: effective timepoints per subject after cutting both ends.
# Original formula: Tmsub = Tmax + 1 - 2*CUT
# Assumption: this must match the actual length of each subject's timeseries
# after filterAndConcatSubj; used by CHARMAnalysis to index into Phi.
Tmsub      = Tmax + 1 - 2 * CUT   # = 156 for Tmax=175, CUT=10

NSUB       = 10     # number of subjects per condition group
Tau        = 3     # time lag for lagged FC (τ in the paper)

epsilon    = 300   # kernel bandwidth σ  (Eq. 10 in the paper)
Thorizont  = 2     # diffusion horizon t (Eq. 11 in the paper)

# Classification parameters
N_TRAIN    = NSUB - 1   # subjects used for training in each fold
KFOLD      = 1000       # number of cross-validation folds


# =============================================================================
# Data loading and preprocessing
# Corresponds to filterAndConcatSubj() in the original code.
# =============================================================================

class DataLoader:
    """
    Loads, filters, and concatenates HCP BOLD timeseries for a set of
    condition groups and subjects.

    Wraps the original filterAndConcatSubj() logic into a reusable class
    that also tracks per-subject time boundaries — information that is
    needed by CHARMAnalysis to slice Phi correctly.

    Parameters
    ----------
    chosen_datasets : list of str
        Condition group names passed to HCP() (e.g. ['REST1', 'EMOTION']).
    discard : tuple or None
        (subject_id, group) to discard, e.g. (553, 'EMOTION').
        Matches the original DL.discardSubject() call.
    n_subjects : int
        Number of subjects to use per group.
    tmax : int
        Maximum timepoints to load per subject.
    cut : int
        Timepoints to remove from each end after bandpass filtering.
    """

    def __init__(
        self,
        chosen_datasets: list[str],
        discard: tuple | None = None,
        n_subjects: int = NSUB,
        tmax: int = Tmax,
        cut: int = CUT,
    ):
        self.n_subjects = n_subjects
        self.tmax       = tmax
        self.cut        = cut

        # Initialise the HCP data loader (your existing infrastructure)
        self._DL = HCP.HCP(chosenDatasets=chosen_datasets)
        if discard is not None:
            self._DL.discardSubject(discard)

        # Bandpass filter — same parameters as original
        # TR is read from the dataloader to stay consistent with HCP metadata
        self._bpf = BandPassFilter(
            k=2, flp=0.008, fhi=0.08,
            tr=self._DL.TR(),
            remove_artifacts=False,
        )

        self.chosen_datasets = chosen_datasets

    def load_and_concat(self) -> tuple[np.ndarray, dict]:
        """
        Load, filter, and concatenate BOLD timeseries for all groups.

        Returns
        -------
        ts : np.ndarray, shape (N_regions, Tm)
            Concatenated, z-scored BOLD timeseries across all groups and
            subjects. Ready to pass directly to CHARMReducer.fit().
        subject_boundaries : dict
            Maps group name → list of (start_col, end_col) tuples, one per
            subject. Used to build CHARMAnalysis group offsets.
            Example:
                {'REST1':   [(0, 156), (156, 312), (312, 468)],
                 'EMOTION': [(468, 624), ...]}
        """
        ts                 = np.empty((N_regions, 0))
        subject_boundaries = {}

        for group in self.chosen_datasets:
            boundaries = []
            subjects   = self._DL.get_groupSubjects(group)[:self.n_subjects]

            for subj in subjects:
                # Load timeseries: shape (N_regions, Tmax)
                ts2 = self._DL.get_subjectData(subj)[subj]['timeseries'][:, :self.tmax]

                # Bandpass filter (bpf expects (T, N), returns (T, N))
                signal_filt = self._bpf.filter(ts2.T).T   # back to (N, T)

                # Remove CUT timepoints from each end
                # Original: tss = signal_filt[:, cutPots-1:-cutPots]
                tss = signal_filt[:, self.cut - 1:-self.cut]   # (N, Tmsub)

                # Track where this subject's data will land in the concat array
                col_start = ts.shape[1]
                ts        = np.concatenate((ts, tss), axis=1)
                col_end   = ts.shape[1]
                boundaries.append((col_start, col_end))

            subject_boundaries[group] = boundaries

        # Z-score across time for each parcel (axis=1 = time axis for N×T)
        # Assumption: z-scoring is applied AFTER concatenation across all
        # subjects and groups, matching the original:
        #   ts = stats.zscore(ts, 1)   ← axis=1 in MATLAB convention = rows
        ts = stats.zscore(ts, axis=1)

        return ts, subject_boundaries


# =============================================================================
# Main pipeline
# =============================================================================

def run():
    """
    Full CHARM pipeline, reproducing the original run() from main_CHARM.py
    using Neuroreduce classes throughout.
    """

    # ── 1. Load and preprocess data ──────────────────────────────────────────
    # Corresponds to the two filterAndConcatSubj() calls + zscore in run()
    print("=" * 60)
    print("Step 1: Loading and preprocessing data")
    print("=" * 60)

    loader = DataLoader(
        chosen_datasets = ['REST1', 'EMOTION'],
        discard         = (553, 'EMOTION'),   # subject with fewer timepoints
        n_subjects      = NSUB,
        tmax            = Tmax,
        cut             = CUT,
    )
    ts, boundaries = loader.load_and_concat()
    print(f"  Concatenated BOLD shape : {ts.shape}  (N={N_regions}, Tm={ts.shape[1]})")
    print(f"  REST1   boundaries      : {boundaries['REST1']}")
    print(f"  EMOTION boundaries      : {boundaries['EMOTION']}")

    # Derive group offsets for CHARMAnalysis from the boundaries dict.
    # The first subject of each group tells us where that group starts in Phi.
    # Assumption: subjects within a group are contiguous in the concat array.
    rest_offset = boundaries['REST1'][0][0]               # = 0
    task_offset = boundaries['EMOTION'][0][0]             # = NSUB * Tmsub

    # ── 2. Fit CHARMReducer ───────────────────────────────────────────────────
    # Corresponds to: Phi = latent(ts)
    #                 conet = nets(Phi, ts)
    print("\n" + "=" * 60)
    print("Step 2: Fitting CHARMReducer (latent + nets)")
    print("=" * 60)
    print(f"  epsilon    = {epsilon}  (kernel bandwidth σ)")
    print(f"  t_horizon  = {Thorizont}  (diffusion horizon t)")
    print(f"  k          = {LATDIM}  (latent dimensions)")

    reducer = CHARMReducer(
        k                 = LATDIM,
        epsilon           = epsilon,
        t_horizon         = Thorizont,
        whiten            = False,       # no whitening — matches original
        sort_eigenvectors = True,        # recommended for production use
    )
    reducer.fit(ts)

    # Phi  (Tm × k): timepoint embedding — equivalent to original Phi
    # conet (N × k): parcel-space basis — equivalent to original conet
    Phi   = reducer.embedding_   # (Tm, k)
    conet = reducer.get_basis()  # (N, k)

    # main_projections: for each parcel, which latent dimension dominates?
    # Corresponds to: main_projections = np.argmax(np.abs(conet), axis=1)
    main_projections = np.argmax(np.abs(conet), axis=1)   # (N_regions,)
    print(f"  Phi shape           : {Phi.shape}")
    print(f"  conet shape         : {conet.shape}")
    print(f"  main_projections    : {main_projections}")

    # ── 3. Per-subject analysis for both condition groups ─────────────────────
    # Corresponds to the two analyze() calls in the original run():
    #   fc_rest_sub, Meta_rest, ... = analyze(Phi, offset=0)
    #   fc_task_sub, Meta_task, ... = analyze(Phi, offset=Tmsub*NSUB)
    print("\n" + "=" * 60)
    print("Step 3: Per-subject analysis (metastability, lagged FC, trophic)")
    print("=" * 60)

    analysis = CHARMAnalysis(
        reducer       = reducer,
        t_per_subject = Tmsub,
        n_subjects    = NSUB,
        tau           = Tau,
    )

    print(f"  Analyzing REST1   (offset={rest_offset})...")
    rest = analysis.analyze_group(group_offset=rest_offset)

    print(f"  Analyzing EMOTION (offset={task_offset})...")
    task = analysis.analyze_group(group_offset=task_offset)

    # Group-level averages — corresponds to:
    #   fc_rest = np.squeeze(np.nanmean(fc_rest_sub, axis=0))
    #   fc_task = np.squeeze(np.nanmean(fc_task_sub, axis=0))
    fc_rest = np.squeeze(np.nanmean(rest.fc_sub, axis=0))   # (k, k)
    fc_task = np.squeeze(np.nanmean(task.fc_sub, axis=0))   # (k, k)

    print(f"  REST metastability (mean ± std) : "
          f"{rest.metastability.mean():.4f} ± {rest.metastability.std():.4f}")
    print(f"  TASK metastability (mean ± std) : "
          f"{task.metastability.mean():.4f} ± {task.metastability.std():.4f}")
    print(f"  REST trophic coherence (mean)   : "
          f"{np.nanmean(rest.trophic_coherence):.4f}")
    print(f"  TASK trophic coherence (mean)   : "
          f"{np.nanmean(task.trophic_coherence):.4f}")

    # ── 4. FDR-corrected condition comparison ─────────────────────────────────
    # Corresponds to: nsig, pfctau = calc_pfctau(fc_rest_sub)
    # (fc_task_sub was an undeclared global in the original — now explicit)
    print("\n" + "=" * 60)
    print("Step 4: FDR-corrected lagged FC comparison (REST vs EMOTION)")
    print("=" * 60)

    nsig, pfctau = analysis.calc_pfctau(
        fc_rest        = rest.fc_sub,
        fc_task        = task.fc_sub,
        n_permutations = 10_000,
        alpha          = 0.05,
    )
    n_significant = nsig.sum()
    print(f"  Significant FC pairs (FDR q<0.05) : {n_significant} / {LATDIM * LATDIM}")
    print(f"  Significant indices (flat)         : {np.where(nsig)[0].tolist()}")

    # ── 5. SVM classification ─────────────────────────────────────────────────
    # Corresponds to: pc, acc = classification()
    print("\n" + "=" * 60)
    print("Step 5: SVM classification (REST vs EMOTION)")
    print("=" * 60)
    print(f"  k-fold = {KFOLD}, n_train = {N_TRAIN} / {NSUB}")

    clf_result = analysis.classification(
        patterns_rest = rest.patterns,
        patterns_task = task.patterns,
        n_train       = N_TRAIN,
        k_fold        = KFOLD,
        random_state  = 42,          # reproducible; not in original
    )
    print(f"  Balanced accuracy : {clf_result.accuracy:.4f}")
    print(f"  Confusion matrix  :\n{clf_result.confusion_matrix}")

    # ── 6. Save results ───────────────────────────────────────────────────────
    # Identical keys to the original sio.savemat() call so downstream
    # MATLAB scripts that read this file continue to work unchanged.
    print("\n" + "=" * 60)
    print("Step 6: Saving results")
    print("=" * 60)

    output_path = '_Data_Produced/results_CHARMem_vs_rest.mat'
    sio.savemat(output_path, {
        # Per-subject group-level quantities
        'Meta_rest':               rest.metastability,
        'Meta_task':               task.metastability,
        'hierarchicallevels_rest': rest.hierarchical_levels,
        'hierarchicallevels_task': task.hierarchical_levels,
        'trophiccoherence_rest':   rest.trophic_coherence,
        'trophiccoherence_task':   task.trophic_coherence,
        'fc_rest_sub':             rest.fc_sub,
        'fc_task_sub':             task.fc_sub,
        # Group averages
        'fc_rest':                 fc_rest,
        'fc_task':                 fc_task,
        # Classification
        'pc':                      clf_result.confusion_matrix,
        'acc':                     clf_result.accuracy,
        # Parcel-space basis
        'conet':                   conet,
        # Condition comparison
        'pfctau':                  pfctau,
        'nsig':                    nsig.astype(np.uint8),  # MATLAB-friendly bool
    })
    print(f"  Saved to: {output_path}")
    print("\nDone.")


# =============================================================================

if __name__ == "__main__":
    run()
