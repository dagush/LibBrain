"""
Papers/Deco2025_CHARM/compare_analysis_single_th.py

Python reproduction of Compare_Analysis_singleTh.m (Deco et al., 2025, PRE)
using the Neuroreduce library.

Benchmarks three dimensionality-reduction methods on the HCP REST-1 dataset
(80-parcel DBS atlas, N=62 parcels selected).  Every subject is embedded
INDEPENDENTLY — no cross-subject concatenation.

Usage
-----
    python compare_analysis_single_th.py --mat hcp1003_REST1_LR_dbs80.mat
    python compare_analysis_single_th.py --mat hcp1003_REST1_LR_dbs80.mat --nsub 5

Methods and their CHARMReducer parameters
------------------------------------------
Classical CHARM:  CHARMReducer(k=LATDIM, epsilon=400, t_horizon=1, kernel_type='classical')
Quantum CHARM:    CHARMReducer(k=LATDIM, epsilon=300, t_horizon=2, kernel_type='quantum')
PCA:              PCAReducer(k=LATDIM)   (handled inline — 5 lines of sklearn)

Two-step pattern per subject
-----------------------------
    reducer.fit(ts_z)                      # full-data embedding → embedding_
    fc_result = reducer.evaluate_fc_cv(ts_z, T_TRAIN)  # CV reconstruction

fit() builds the Tm×Tm kernel on ALL timepoints (for metastability),
evaluate_fc_cv() reuses the stored _Ptr_t to split train/test without
rebuilding the kernel.

Reference
---------
Deco, Sanz Perl, Kringelbach (2025). Physical Review E 111(1).
https://doi.org/10.1103/physreve.111.014410

MATLAB original: Compare_Analysis_singleTh.m
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.io
import scipy.signal as sig
from scipy.stats import zscore, ranksums
from sklearn.decomposition import PCA

from Neuroreduce.methods.charm import CHARMReducer


# ---------------------------------------------------------------------------
# Global constants (mirror MATLAB exactly)
# ---------------------------------------------------------------------------
NSUB    = 1003   # number of HCP subjects
N_FULL  = 80     # parcels in the DBS-80 atlas
N       = 62     # parcels used (indexregion below)
T_TRAIN = 800    # training timepoints (Ttrain in MATLAB)
LATDIM  = 7      # number of latent dimensions (k)
TR      = 0.72   # repetition time (seconds)
TRIM    = 50     # timepoints to drop from each edge after filtering

# MATLAB: indexregion = [1:31, 50:80]  (1-indexed)
# Python (0-indexed):  [0:31, 49:80]
INDEX_REGION = np.r_[0:31, 49:80]   # shape (62,)
assert len(INDEX_REGION) == N

# Lower-triangular index arrays (reused across all FCD computations).
# _rL/_cL index the k×k outer product for FCD in latent space.
# _rN/_cN index the N×N FC matrix for quality metrics.
_rN, _cN = np.tril_indices(N, k=-1)
_rL, _cL = np.tril_indices(LATDIM, k=-1)


# ---------------------------------------------------------------------------
# Bandpass filter construction
# ---------------------------------------------------------------------------
def build_bandpass_filter(
    tr: float = TR,
    flp: float = 0.008,
    fhi: float = 0.08,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 2nd-order zero-phase Butterworth bandpass filter.

    Mirrors MATLAB exactly:
        fnq = 1/(2*TR);
        Wn  = [flp/fnq, fhi/fnq];
        [bfilt, afilt] = butter(k, Wn);
    """
    fnq = 1.0 / (2.0 * tr)
    Wn  = [flp / fnq, fhi / fnq]
    b, a = sig.butter(order, Wn, btype='bandpass')
    return b, a


BFILT, AFILT = build_bandpass_filter()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_subject(
    ts_raw: np.ndarray,
    bfilt: np.ndarray = BFILT,
    afilt: np.ndarray = AFILT,
    trim: int = TRIM,
) -> np.ndarray:
    """
    Preprocess one subject's BOLD timeseries (parcel selection already done).

    Steps (verbatim from MATLAB):
      1. Remove parcel mean + linear detrend
      2. Zero-phase bandpass filter (filtfilt)
      3. Trim `trim` timepoints from each edge

    Trimming note: MATLAB ``signal_filt(:, 50:end-50)`` (1-indexed) keeps
    columns 50..T-50 inclusive → length T-99.  For HCP T=1200: Tm=1101.
    Python: ``signal_filt[:, trim-1 : T-trim]`` → length T-2*trim+1 = 1101 ✓

    Parameters
    ----------
    ts_raw : (N, T_orig)  BOLD, NOT yet detrended

    Returns
    -------
    ts : (N, Tm)  preprocessed BOLD, NOT yet z-scored
    """
    N_parcels, T_orig = ts_raw.shape
    signal_filt = np.empty_like(ts_raw, dtype=float)
    for seed in range(N_parcels):
        x = ts_raw[seed].astype(float)
        x -= np.nanmean(x)
        x  = sig.detrend(x)
        signal_filt[seed] = sig.filtfilt(bfilt, afilt, x)
    return signal_filt[:, trim - 1 : T_orig - trim]   # (N, Tm)


# ---------------------------------------------------------------------------
# Metastability helpers  (pipeline code — not part of the reducer)
# ---------------------------------------------------------------------------
def _fcd_variance(zPhi: np.ndarray, edge_rows: np.ndarray, edge_cols: np.ndarray) -> float:
    """
    Variance of the lower-triangular FCD cosine-similarity matrix.

    FCD[s,t] = cos_sim(edges_s, edges_t)
    where Edges[:,t] = lower-tri of (zPhi[t,:] ⊗ zPhi[t,:]).

    Parameters
    ----------
    zPhi      : (T, D)  z-scored trajectory (rows = timepoints)
    edge_rows : lower-tri row    indices for the D×D outer product
    edge_cols : lower-tri column indices for the D×D outer product
    """
    # Outer product at every timepoint: (T, D, D)
    outer  = zPhi[:, :, None] * zPhi[:, None, :]
    # Extract lower-tri edges: (n_edges, T)
    Edges  = outer[:, edge_rows, edge_cols].T
    # Cosine-similarity: normalise columns, Gram matrix
    norms  = np.linalg.norm(Edges, axis=0, keepdims=True)
    Edges_n = Edges / (norms + 1e-12)
    FCD    = Edges_n.T @ Edges_n                    # (T, T)
    r_t, c_t = np.tril_indices(FCD.shape[0], k=-1)
    return float(np.var(FCD[r_t, c_t]))


def compute_metastability(
    zPhi: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
) -> float:
    """
    Differential entropy of the FCD distribution (Gaussian approximation).

    MATLAB: Meta = 0.5*(log(2*pi*var(FCD(IsubdiagT)))) + 0.5;
    This is the standard Gaussian entropy h(X) = 0.5*log(2πe·σ²).

    Parameters
    ----------
    zPhi      : (T, D)  z-scored latent trajectory
    edge_rows : lower-tri row    indices for D×D outer products
    edge_cols : lower-tri column indices for D×D outer products
    """
    return 0.5 * np.log(2.0 * np.pi * _fcd_variance(zPhi, edge_rows, edge_cols)) + 0.5


def compute_raw_bold_metastability(ts: np.ndarray) -> float:
    """
    MetaA — metastability of the raw (uncompressed) BOLD.

    Called BEFORE z-scoring ts row-wise. MATLAB: ``zscore(ts')`` applied
    to the preprocessed ts, using N-dimensional lower-tri edges.

    Parameters
    ----------
    ts : (N, Tm)  preprocessed BOLD, NOT yet row-z-scored
    """
    zPhi = zscore(ts.T, axis=0)          # (Tm, N) — z-score each parcel (column)
    r, c = np.tril_indices(N, k=-1)
    return compute_metastability(zPhi, r, c)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_hcp_dbs80(mat_file: str) -> list[np.ndarray]:
    """
    Load HCP REST-1 timeseries from the MATLAB .mat file.

    PLACEHOLDER — the LibBrain DataLoader for this dataset (HCP_DBS80) has
    not yet been implemented.  This function provides a direct scipy.io.loadmat
    loading path as a stopgap.

    TODO: replace body with a call to DataLoaders.HCP_DBS80 once implemented:
          dl = HCP_DBS80(mat_file)
          return [dl.get_ts(i) for i in range(dl.n_subjects())]

    Expected .mat structure: subject{sub}.dbs80ts → (N_FULL=80, T_orig)
    """
    print(f"Loading {mat_file} …")
    mat = scipy.io.loadmat(mat_file, squeeze_me=False)
    raw = mat['subject']                  # numpy object array (1, NSUB)
    subjects_ts = []
    for i in range(raw.shape[1]):
        try:
            ts = raw[0, i]['dbs80ts'][0, 0].astype(float)
        except (IndexError, KeyError):
            ts = raw[0, i][0, 0]['dbs80ts'].astype(float)
        subjects_ts.append(ts)
    print(f"  Loaded {len(subjects_ts)} subjects, shape = {subjects_ts[0].shape}")
    return subjects_ts


# ---------------------------------------------------------------------------
# PCA CV helper  (inline — too short to justify a PCAReducer method)
# ---------------------------------------------------------------------------
def pca_cv_fc(ts_z: np.ndarray, latdim: int, t_train: int) -> dict:
    """
    PCA cross-validation: fit on training timepoints, reconstruct test BOLD.

    Mirrors the MATLAB block:
        [CoePCA,PhiPCA,...,mu] = pca(ts(:,1:Ttrain)')
        PhiPCAcv = ts(:,1+Ttrain:end)' * CoePCA
        tscve    = PhiPCAcv(:,1:k) * CoePCA(:,1:k)' + mu

    PCA metastability correction vs MATLAB
    ----------------------------------------
    The MATLAB metastability block applies IsubdiagL (7×7 lower-tri indices)
    to the outer product of ALL 62 PCA scores — a linear-indexing bug that
    picks wrong elements from the 62×62 outer product.  We correct this by
    restricting the FCD computation to the first `latdim` components.

    Parameters
    ----------
    ts_z    : (N, Tm)  z-scored BOLD
    latdim  : number of PCA components k
    t_train : number of training timepoints

    Returns
    -------
    dict with 'corr_fit', 'err_fit', 'fc_true', 'fc_est', 'metastability'
    """
    N_p, Tm = ts_z.shape

    # ── Training PCA ──────────────────────────────────────────────────
    X_train = ts_z[:, :t_train].T                               # (T_train, N)
    pca_tr  = PCA(n_components=N_p)
    pca_tr.fit(X_train)
    mu = pca_tr.mean_                                           # (N,)

    # ── CV reconstruction ─────────────────────────────────────────────
    X_test    = ts_z[:, t_train:].T                             # (T_test, N)
    Phi_cv    = X_test @ pca_tr.components_.T                  # (T_test, N) all scores
    ts_cv_est = Phi_cv[:, :latdim] @ pca_tr.components_[:latdim] + mu  # (T_test, N)

    # ── FC quality ────────────────────────────────────────────────────
    FC_true    = np.corrcoef(ts_z[:, t_train:])                 # (N, N)
    FC_est     = np.corrcoef(ts_cv_est.T)                       # (N, N)
    fc_true_lt = FC_true[_rN, _cN]
    fc_est_lt  = FC_est [_rN, _cN]
    corr_fit   = float(np.corrcoef(fc_true_lt, fc_est_lt)[0, 1])
    err_fit    = float(np.mean((fc_true_lt - fc_est_lt) ** 2))

    # ── Metastability (corrected: latdim components only) ─────────────
    pca_full  = PCA(n_components=latdim)
    Phi_full  = pca_full.fit_transform(ts_z.T)                  # (Tm, latdim)
    zPhi_full = zscore(Phi_full, axis=0)
    meta = compute_metastability(zPhi_full, _rL, _cL)

    return dict(corr_fit=corr_fit, err_fit=err_fit,
                fc_true=FC_true, fc_est=FC_est, metastability=meta)


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------
def run_comparison(
    subjects_ts       : list[np.ndarray],
    epsilon_classic   : float = 400.0,
    epsilon_quantum   : float = 300.0,
    thorizont_classic : int   = 1,
    thorizont_quantum : int   = 2,
    latdim            : int   = LATDIM,
    t_train           : int   = T_TRAIN,
    save_path         : Optional[str] = None,
) -> dict:
    """
    Process all subjects through Classical CHARM, Quantum CHARM, and PCA.

    Per-subject pattern
    -------------------
    For each CHARM variant:

        reducer = CHARMReducer(k=latdim, epsilon=..., t_horizon=...,
                               kernel_type='classical'/'quantum')
        reducer.fit(ts_z)
        # Metastability from full-data embedding:
        zPhi = zscore(reducer.embedding_, axis=0)   # (Tm, k)
        meta = compute_metastability(zPhi, _rL, _cL)
        # FC reconstruction quality on held-out timepoints:
        fc   = reducer.evaluate_fc_cv(ts_z, t_train)

    fit() builds the full Tm×Tm kernel once and stores _Ptr_t.
    evaluate_fc_cv() reuses _Ptr_t without rebuilding the kernel.

    Computational cost note
    -----------------------
    Quantum CHARM requires a Tm×Tm complex matrix power (O(Tm³) per subject).
    For Tm≈1101 this is ~30–60 s/subject on CPU.  For NSUB=1003 subjects,
    consider joblib.Parallel or GPU (cupy) for production runs.

    Parameters
    ----------
    subjects_ts        : list of NSUB arrays, each (N_FULL, T_orig)
    epsilon_classic    : bandwidth for classical CHARM kernel (default 400)
    epsilon_quantum    : bandwidth for quantum  CHARM kernel (default 300)
    thorizont_classic  : τ for classical CHARM — affects Φ scaling and Nyström
                         denominator, NOT the kernel itself (default 1)
    thorizont_quantum  : τ for quantum  CHARM — matrix power K^τ (default 2)
    latdim             : latent dimensions k (default 7)
    t_train            : training timepoints for CV (default 800)
    save_path          : if given, save results as .npz

    Returns
    -------
    dict with keys matching the MATLAB save block:
        MetaA, Meta, MetaQ, MetaPCA          — per-subject metastability (nsub,)
        Corrfitt, CorrfittQ, CorrfittPCA     — per-subject FC Pearson r (nsub,)
        ERRfitt,  ERRfittQ,  ERRfittPCA      — per-subject FC MSE (nsub,)
        FCtrueG, FCestG, FCestQ, FCestPCA    — group-mean FC matrices (N, N)
    """
    nsub = len(subjects_ts)

    # Preallocate
    MetaA       = np.zeros(nsub)
    Meta        = np.zeros(nsub)
    MetaQ       = np.zeros(nsub)
    MetaPCA     = np.zeros(nsub)
    Corrfitt    = np.zeros(nsub)
    CorrfittQ   = np.zeros(nsub)
    CorrfittPCA = np.zeros(nsub)
    ERRfitt     = np.zeros(nsub)
    ERRfittQ    = np.zeros(nsub)
    ERRfittPCA  = np.zeros(nsub)
    FCtrue_acc  = np.zeros((nsub, N, N))
    FCestC_acc  = np.zeros((nsub, N, N))
    FCestQ_acc  = np.zeros((nsub, N, N))
    FCestP_acc  = np.zeros((nsub, N, N))

    # Build reducers once — they are stateless until fit() is called,
    # so the same instance can be refit per subject.
    reducer_c = CHARMReducer(
        k=latdim, epsilon=epsilon_classic, t_horizon=thorizont_classic,
        kernel_type='classical',
    )
    reducer_q = CHARMReducer(
        k=latdim, epsilon=epsilon_quantum, t_horizon=thorizont_quantum,
        kernel_type='quantum',
    )

    for sub_idx, ts_raw in enumerate(subjects_ts):
        print(f"  Subject {sub_idx + 1:4d}/{nsub}", end='\r', flush=True)

        # ------------------------------------------------------------------
        # Preprocess (once, shared across all methods)
        # ------------------------------------------------------------------
        ts_sel  = ts_raw[INDEX_REGION, :]           # (N=62, T_orig) parcel selection
        ts_filt = preprocess_subject(ts_sel)        # (N, Tm) — not yet z-scored

        # ------------------------------------------------------------------
        # MetaA — raw BOLD metastability, computed BEFORE row-z-scoring.
        # MATLAB: zscore(ts') inside the first loop, before zscore(ts,[],2).
        # Both are per-parcel z-scores; MetaA is measured in N-dim edge space.
        # ------------------------------------------------------------------
        MetaA[sub_idx] = compute_raw_bold_metastability(ts_filt)

        # ------------------------------------------------------------------
        # Per-parcel z-score (row-wise).  MATLAB: ts = zscore(ts, [], 2)
        # ------------------------------------------------------------------
        mu_p = ts_filt.mean(axis=1, keepdims=True)
        sd_p = ts_filt.std(axis=1, keepdims=True)
        ts_z = (ts_filt - mu_p) / (sd_p + 1e-12)   # (N, Tm)

        # ------------------------------------------------------------------
        # Classical CHARM
        #
        # fit()            → builds full Tm×Tm real kernel, stores _Ptr_t=K
        # embedding_       → (Tm, k) scaled by λ^τ  (MATLAB: Phi*LL.^Thorizont)
        # evaluate_fc_cv() → splits _Ptr_t into train/CV blocks, no rebuild
        # ------------------------------------------------------------------
        reducer_c.fit(ts_z)
        zPhi_c       = zscore(reducer_c.embedding_, axis=0)   # (Tm, k)
        Meta[sub_idx] = compute_metastability(zPhi_c, _rL, _cL)

        fc_c                   = reducer_c.evaluate_fc_cv(ts_z, t_train)
        Corrfitt[sub_idx]      = fc_c['corr_fit']
        ERRfitt[sub_idx]       = fc_c['err_fit']
        FCtrue_acc[sub_idx]    = fc_c['fc_true']
        FCestC_acc[sub_idx]    = fc_c['fc_est']

        # ------------------------------------------------------------------
        # Quantum CHARM
        #
        # fit()            → builds full Tm×Tm complex kernel K, computes
        #                    |K^τ|², stores _Ptr_t = |K^τ|²
        # embedding_       → (Tm, k) scaled by |λ|  (MATLAB: Phi*abs(LL))
        # evaluate_fc_cv() → uses |K^τ|² cross-block, Λ^{-1} denominator
        # ------------------------------------------------------------------
        reducer_q.fit(ts_z)
        zPhi_q        = zscore(reducer_q.embedding_, axis=0)  # (Tm, k)
        MetaQ[sub_idx] = compute_metastability(zPhi_q, _rL, _cL)

        fc_q                   = reducer_q.evaluate_fc_cv(ts_z, t_train)
        CorrfittQ[sub_idx]     = fc_q['corr_fit']
        ERRfittQ[sub_idx]      = fc_q['err_fit']
        FCestQ_acc[sub_idx]    = fc_q['fc_est']

        # ------------------------------------------------------------------
        # PCA (handled inline — simpler than adding a method to PCAReducer)
        # See pca_cv_fc() docstring for the MATLAB equivalence and the
        # metastability correction vs the original MATLAB code.
        # ------------------------------------------------------------------
        pca_result             = pca_cv_fc(ts_z, latdim=latdim, t_train=t_train)
        MetaPCA[sub_idx]       = pca_result['metastability']
        CorrfittPCA[sub_idx]   = pca_result['corr_fit']
        ERRfittPCA[sub_idx]    = pca_result['err_fit']
        FCestP_acc[sub_idx]    = pca_result['fc_est']

    print()  # newline after progress bar

    # Group-mean FC (MATLAB: FCtrueG = squeeze(mean(FCtrue2)))
    FCtrueG  = FCtrue_acc.mean(axis=0)
    FCestG   = FCestC_acc.mean(axis=0)
    FCestQ   = FCestQ_acc.mean(axis=0)
    FCestPCA = FCestP_acc.mean(axis=0)

    results = dict(
        MetaA=MetaA, Meta=Meta, MetaQ=MetaQ, MetaPCA=MetaPCA,
        Corrfitt=Corrfitt,   CorrfittQ=CorrfittQ,   CorrfittPCA=CorrfittPCA,
        ERRfitt=ERRfitt,     ERRfittQ=ERRfittQ,     ERRfittPCA=ERRfittPCA,
        FCtrueG=FCtrueG, FCestG=FCestG, FCestQ=FCestQ, FCestPCA=FCestPCA,
    )

    if save_path is not None:
        np.savez(save_path, **results)
        print(f"Results saved → {save_path}")

    return results


# ---------------------------------------------------------------------------
# Statistical analysis and plotting
# ---------------------------------------------------------------------------
def analyse_and_plot(
    results  : dict,
    n_trials : int = 100,
    n_sample : int = 500,
    rng_seed : int = 42,
) -> dict:
    """
    Reproduce MATLAB statistical analysis and figures 1, 2, 4, 5.

    Bootstrap meta-correlation (MATLAB):
        for trial=1:100
            indsub = randperm(NSUB);
            CorrMeta(trial) = corr2(Meta(indsub(1:500)), MetaA(indsub(1:500)));
        end

    MATLAB corr2() on vectors = np.corrcoef()[0,1].
    MATLAB ranksum()           = scipy.stats.ranksums().

    Parameters
    ----------
    results  : dict returned by run_comparison()
    n_trials : bootstrap iterations (100 in MATLAB)
    n_sample : subjects per bootstrap sample (500 in MATLAB)
    rng_seed : for reproducibility
    """
    import matplotlib.pyplot as plt

    MetaA       = results['MetaA']
    Meta        = results['Meta']
    MetaQ       = results['MetaQ']
    MetaPCA     = results['MetaPCA']
    Corrfitt    = results['Corrfitt']
    CorrfittQ   = results['CorrfittQ']
    CorrfittPCA = results['CorrfittPCA']
    ERRfitt     = results['ERRfitt']
    ERRfittQ    = results['ERRfittQ']
    ERRfittPCA  = results['ERRfittPCA']

    nsub = len(MetaA)
    rng  = np.random.default_rng(rng_seed)

    # Bootstrap meta-correlation (Figure 2 source)
    CorrMeta    = np.empty(n_trials)
    CorrMetaQ   = np.empty(n_trials)
    CorrMetaPCA = np.empty(n_trials)
    for trial in range(n_trials):
        idx = rng.permutation(nsub)[:n_sample]
        CorrMeta   [trial] = np.corrcoef(Meta   [idx], MetaA[idx])[0, 1]
        CorrMetaQ  [trial] = np.corrcoef(MetaQ  [idx], MetaA[idx])[0, 1]
        CorrMetaPCA[trial] = np.corrcoef(MetaPCA[idx], MetaA[idx])[0, 1]

    # Whole-cohort Pearson r
    r_classic = np.corrcoef(MetaA, Meta  )[0, 1]
    r_quantum = np.corrcoef(MetaA, MetaQ )[0, 1]
    r_pca     = np.corrcoef(MetaA, MetaPCA)[0, 1]
    print("\n── Whole-cohort metastability correlation with raw BOLD ──")
    print(f"  PCA:             r = {r_pca    :.4f}")
    print(f"  Classical CHARM: r = {r_classic:.4f}")
    print(f"  Quantum  CHARM:  r = {r_quantum:.4f}")

    # Rank-sum tests
    _, p_cq  = ranksums(CorrMeta,  CorrMetaQ)
    _, p_cp  = ranksums(CorrMeta,  CorrMetaPCA)
    _, p_qp  = ranksums(CorrMetaQ, CorrMetaPCA)
    print("\n── Rank-sum: bootstrap meta-r ──")
    print(f"  Classic vs Quantum: p = {p_cq:.4g}")
    print(f"  Classic vs PCA:     p = {p_cp:.4g}")
    print(f"  Quantum  vs PCA:    p = {p_qp:.4g}")

    _, p_fcq = ranksums(Corrfitt,  CorrfittQ)
    _, p_fcp = ranksums(Corrfitt,  CorrfittPCA)
    _, p_fqp = ranksums(CorrfittQ, CorrfittPCA)
    print("\n── Rank-sum: per-subject FC reconstruction r ──")
    print(f"  Classic vs Quantum: p = {p_fcq:.4g}")
    print(f"  Classic vs PCA:     p = {p_fcp:.4g}")
    print(f"  Quantum  vs PCA:    p = {p_fqp:.4g}")

    def _violin_or_box(ax, data_list, labels, title):
        try:
            import seaborn as sns, pandas as pd
            sns.violinplot(data=pd.DataFrame(dict(zip(labels, data_list))),
                           ax=ax, inner='box', palette='muted')
        except ImportError:
            ax.boxplot(data_list, labels=labels)
        ax.set_title(title)

    # Figure 1: scatter MetaA vs each method
    fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4))
    fig1.suptitle('Metastability: raw BOLD vs latent space', fontsize=12)
    for ax, meta_lat, label, color in zip(
        axes1,
        [MetaPCA, Meta, MetaQ],
        ['PCA', 'Classical CHARM', 'Quantum CHARM'],
        ['k', 'b', 'r'],
    ):
        r = np.corrcoef(MetaA, meta_lat)[0, 1]
        ax.scatter(MetaA, meta_lat, marker='x', color=color, s=8, alpha=0.4)
        ax.set_title(f'{label}\nr = {r:.3f}')
        ax.set_xlabel('MetaA (raw BOLD)')
        ax.set_ylabel('Meta (latent space)')
        ax.set_aspect('equal', adjustable='box')
    fig1.tight_layout()

    # Figure 2: violin bootstrap meta-correlation
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    _violin_or_box(ax2, [CorrMetaPCA, CorrMeta, CorrMetaQ],
                   ['PCA', 'Classic', 'Quantum'],
                   'Bootstrap meta-correlation with raw BOLD')
    ax2.set_ylabel('Pearson r')

    # Figure 4: violin FC reconstruction r
    fig4, ax4 = plt.subplots(figsize=(5, 5))
    _violin_or_box(ax4, [CorrfittPCA, Corrfitt, CorrfittQ],
                   ['PCA', 'Classic', 'Quantum'],
                   'FC reconstruction Pearson r (held-out)')
    ax4.set_ylabel('Pearson r')

    # Figure 5: violin FC MSE
    fig5, ax5 = plt.subplots(figsize=(5, 5))
    _violin_or_box(ax5, [ERRfittPCA, ERRfitt, ERRfittQ],
                   ['PCA', 'Classic', 'Quantum'],
                   'FC reconstruction MSE (held-out)')
    ax5.set_ylabel('MSE')

    plt.show()

    return dict(
        CorrMeta=CorrMeta, CorrMetaQ=CorrMetaQ, CorrMetaPCA=CorrMetaPCA,
        r_classic=r_classic, r_quantum=r_quantum, r_pca=r_pca,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Reproduce Compare_Analysis_singleTh.m using Neuroreduce')
    parser.add_argument('--mat',  default='hcp1003_REST1_LR_dbs80.mat')
    parser.add_argument('--out',  default='results_analysis_single_LD7_Th2.npz')
    parser.add_argument('--nsub', type=int, default=None,
                        help='Limit subjects (default: all 1003)')
    args = parser.parse_args()

    subjects_ts = load_hcp_dbs80(args.mat)
    if args.nsub is not None:
        subjects_ts = subjects_ts[:args.nsub]
        print(f"Processing first {args.nsub} subjects only.")

    results = run_comparison(
        subjects_ts,
        epsilon_classic   = 400.0,
        epsilon_quantum   = 300.0,
        thorizont_classic = 1,
        thorizont_quantum = 2,
        latdim            = LATDIM,
        t_train           = T_TRAIN,
        save_path         = args.out,
    )
    analyse_and_plot(results)
