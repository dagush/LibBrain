"""
examples/validate_charm.py
---------------------------
Validates that CHARMReducer reproduces the original latent() + nets() pipeline
to numerical precision.

Runs both the original functions and the new CHARMReducer on the same data,
then compares results. Any discrepancy beyond floating-point noise will be
flagged explicitly.

Usage
-----
    python examples/validate_charm.py

Expected output (if everything is correct)
-------------------------------------------
    Loading and preprocessing data...
    Running ORIGINAL pipeline...
    Running CHARMReducer pipeline...

    === Phi (timepoint embedding) ===
    Max absolute difference : 0.000000e+00  (should be 0 — identical path)
    All close (atol=1e-5)   : True

    === conet / get_basis() ===
    Max absolute difference : 0.000000e+00
    All close (atol=1e-5)   : True

    === transform() identity shortcut ===
    Z shape                 : (7, 525)     [k x Tm]
    Z == Phi.T exactly      : True

    All checks passed. CHARMReducer reproduces the original pipeline.
"""

import numpy as np
from numpy import linalg as LA
from scipy import stats

# ── new library ───────────────────────────────────────────────────────────────
from Neuroreduce import CHARMReducer

# ── your original data loading infrastructure ─────────────────────────────────
import DataLoaders.HCP_dbs80 as HCP
from neuronumba.tools.filters import BandPassFilter

# ── parameters (must match your original run() exactly) ──────────────────────
N_regions = 80
LATDIM    = 7
Tmax      = 175
CUT       = 10
NSUB      = 10   # 100 in the original code
epsilon   = 300
Thorizont = 2


# =============================================================================
# Original functions — copied verbatim from main_CHARM.py so we have a
# guaranteed reference. Do NOT modify these.
# =============================================================================

def _original_latent(ts, latdim, eps, thorizont):
    """Verbatim copy of latent() from main_CHARM.py."""
    Tm = ts.shape[1]
    Kmatrix = np.zeros((Tm, Tm), dtype=complex)
    for i in range(Tm):
        for j in range(Tm):
            dij2 = np.sum((ts[:, i] - ts[:, j]) ** 2)
            Kmatrix[i, j] = np.exp((0 + 1j) * dij2 / eps)
    Ktr_t = LA.matrix_power(Kmatrix, thorizont)
    Ptr_t = np.square(np.abs(Ktr_t))
    Dmatrix = np.diag(np.sum(Ptr_t, axis=1))
    Pmatrix = LA.inv(Dmatrix) @ Ptr_t
    LL, VV = LA.eig(Pmatrix)
    Phi = np.real(VV[:, 1:latdim + 1])
    LLMatr = np.diag(LL)
    Phi = Phi @ np.abs(LLMatr[1:latdim + 1, 1:latdim + 1])
    return Phi


def _original_nets(Phi, ts, n_regions, latdim):
    """Verbatim copy of nets() from main_CHARM.py."""
    import neuronumba.tools.matlab_tricks as tricks
    zPhiA = stats.zscore(ts.T, ddof=1)
    zPhi  = stats.zscore(Phi,  ddof=1)
    conet  = np.zeros((n_regions, latdim))
    conet2 = np.zeros((n_regions, latdim))
    for red in range(latdim):
        for seed in range(n_regions):
            conet2[seed, red] = tricks.corr(zPhi[:, red], zPhiA[:, seed])
        conet[:, red] = conet2[:, red] / LA.norm(conet2[:, red])
    return conet


# =============================================================================
# Data loading — same as filterAndConcatSubj() in your original code
# =============================================================================

def load_data():
    print("Loading and preprocessing data...")
    DL = HCP.HCP(chosenDatasets=['REST1', 'EMOTION'])
    DL.discardSubject((553, 'EMOTION'))
    bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=DL.TR(),
                         remove_artifacts=False)

    ts = np.empty((N_regions, 0))
    for group in ['REST1', 'EMOTION']:
        subjects = DL.get_groupSubjects(group)[:NSUB]
        for subj in subjects:
            ts2 = DL.get_subjectData(subj)[subj]['timeseries'][:, :Tmax]
            signal_filt = bpf.filter(ts2.T).T
            tss = signal_filt[:, CUT - 1:-CUT]
            ts = np.concatenate((ts, tss), axis=1)

    ts = stats.zscore(ts, axis=1)
    return ts


# =============================================================================
# Comparison helpers
# =============================================================================

def _compare(name, A, B, atol=1e-5):
    """Print a comparison block for two arrays A and B."""
    diff = np.max(np.abs(A - B))
    close = np.allclose(A, B, atol=atol)
    print(f"\n=== {name} ===")
    print(f"  Shape (original)  : {A.shape}")
    print(f"  Shape (new)       : {B.shape}")
    print(f"  Max abs difference: {diff:.6e}  (should be < {atol})")
    print(f"  All close         : {close}")
    if not close:
        print(f"  *** MISMATCH — check implementation! ***")
    return close


# =============================================================================
# Main validation
# =============================================================================

def main():
    ts = load_data()
    print(f"Concatenated BOLD shape: {ts.shape}  (N={N_regions}, Tm={ts.shape[1]})")

    # ── original pipeline ─────────────────────────────────────────────────────
    print("\nRunning ORIGINAL pipeline...")
    Phi_orig   = _original_latent(ts, LATDIM, epsilon, Thorizont)
    conet_orig = _original_nets(Phi_orig, ts, N_regions, LATDIM)

    # ── new library pipeline ──────────────────────────────────────────────────
    print("Running CHARMReducer pipeline...")
    reducer = CHARMReducer(
        k           = LATDIM,
        epsilon     = epsilon,
        t_horizon   = Thorizont,
        whiten      = False,          # keep False to match original exactly
        sort_eigenvectors = False,    # original code does NOT sort — match it
    )
    reducer.fit(ts)

    Phi_new   = reducer.embedding_    # (Tm, k)  — direct Phi from _latent()
    conet_new = reducer.get_basis()   # (N, k)   — from _nets()

    # ── comparisons ───────────────────────────────────────────────────────────
    ok_phi   = _compare("Phi (timepoint embedding)", Phi_orig,   Phi_new)
    ok_conet = _compare("conet / get_basis()",       conet_orig, conet_new)

    # ── transform() identity shortcut ─────────────────────────────────────────
    print("\n=== transform() identity shortcut ===")
    Z = reducer.transform(ts)
    z_is_phi_t = np.array_equal(Z, Phi_new.T)
    print(f"  Z shape            : {Z.shape}  [k x Tm]")
    print(f"  Z == Phi.T exactly : {z_is_phi_t}")

    # ── Nyström self-consistency (optional, slow) ─────────────────────────────
    run_nystrom_check = True   # set False to skip if Tm is large
    if run_nystrom_check:
        print("\n=== Nyström self-consistency (force_nystrom=True on training data) ===")
        Z_nystrom = reducer.transform(ts, force_nystrom=True)
        diff_nystrom = np.max(np.abs(Z - Z_nystrom))
        print(f"  Max abs diff vs direct Phi.T : {diff_nystrom:.6e}")
        print(f"  (small but non-zero is expected — Nyström is an approximation)")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ok_phi and ok_conet and z_is_phi_t:
        print("All checks passed. CHARMReducer reproduces the original pipeline.")
    else:
        print("SOME CHECKS FAILED. Review the mismatches above.")
    print("=" * 60)


if __name__ == "__main__":
    main()