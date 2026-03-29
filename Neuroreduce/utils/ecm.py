"""
Neuroreduce/utils/ecm.py
--------------------------
Edge-centric metastability utilities for Neuroreduce.

The core ECM computation now lives in NeuroNumba as a proper Observable:

    neuronumba/observables/ecm.py  →  ECM._compute_from_fmri()

This module provides three thin wrappers that bridge the NeuroNumba
observable into the Neuroreduce per-subject pipeline:

    compute_ecm(signal)
        One-shot scalar ECM for a single (T, D) signal.
        Delegates directly to the ECM observable.

    compute_ecm_per_subject(signal, ...)
        Slice a concatenated (D, Tm) signal into per-subject windows
        and return an array of per-subject ECM scalars.

    compute_reconstructed_ecm_per_subject(X, reducer, ...)
        The key function for Figure 2(b): per subject, reduce X then
        invert, compute ECM on both original and reconstructed BOLD,
        and return both arrays for downstream correlation / plotting.

Convention note
---------------
NeuroNumba observables expect (T, N) — rows = timepoints, columns = ROIs.
Neuroreduce uses (N, T) — rows = ROIs/dims, columns = timepoints.
All transpositions are handled inside these functions; callers always
pass arrays in the Neuroreduce (D, T) or (N, T) convention.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy import stats

# NeuroNumba ECM observable — the single source of truth for the core maths.
# This import will fail if neuronumba is not installed; the error message
# is intentionally left as-is so the user knows exactly what to install.
from neuronumba.observables.ecm import ECM as _ECMObservable


# =============================================================================
# Public API
# =============================================================================

def compute_ecm(signal: np.ndarray) -> float:
    """
    Compute edge-centric metastability (ECM) for a single signal matrix.

    Thin wrapper around the NeuroNumba ECM observable. Handles the
    Neuroreduce → NeuroNumba convention transpose internally.

    Parameters
    ----------
    signal : np.ndarray, shape (T, D)
        Signal in NeuroNumba convention — rows = timepoints, cols = dims.
        D can be N parcels (source space) or k latents (manifold space).
        Should be z-scored across time (axis=0) before calling.

    Returns
    -------
    float
        ECM value H. Typically negative (log of a small variance).
        Higher = more metastable / dynamic.
    """
    # signal is already (T, D) — matches NeuroNumba convention directly
    obs    = _ECMObservable()
    result = obs.from_fmri(signal)
    return float(result['ECM'])


def compute_ecm_per_subject(
    signal:        np.ndarray,
    n_subjects:    int,
    t_per_subject: int,
    group_offset:  int = 0,
) -> np.ndarray:
    """
    Compute ECM for each subject in a concatenated signal matrix.

    Parameters
    ----------
    signal : np.ndarray, shape (D, Tm)
        Concatenated signal in Neuroreduce convention (D, Tm).
        D can be N parcels or k latent dimensions.
        If the array appears to already be (Tm, D) (first dim > second),
        it is used as-is without transposing.
    n_subjects : int
        Number of subjects in this group.
    t_per_subject : int
        Timepoints per subject.
    group_offset : int
        Starting timepoint index for this group in the concatenated array.
        Default: 0.

    Returns
    -------
    ecm_values : np.ndarray, shape (n_subjects,)
        Per-subject ECM scalars.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D, got shape {signal.shape}")

    # Transpose from Neuroreduce (D, Tm) → NeuroNumba (Tm, D).
    # Heuristic: if first axis ≤ second axis, assume (D, Tm) and transpose.
    if signal.shape[0] <= signal.shape[1]:
        signal = signal.T   # now (Tm, D)

    # Instantiate once and reuse — avoids repeated object construction
    obs        = _ECMObservable()
    ecm_values = np.zeros(n_subjects)

    for sub in range(n_subjects):
        start   = group_offset + sub * t_per_subject
        end     = group_offset + (sub + 1) * t_per_subject
        sig_sub = signal[start:end, :]                    # (t_per_subject, D)
        sig_sub = stats.zscore(sig_sub, axis=0, ddof=1)  # z-score across time

        result          = obs.from_fmri(sig_sub)
        ecm_values[sub] = result['ECM']

    return ecm_values


def compute_reconstructed_ecm_per_subject(
    X:             np.ndarray,
    reducer,
    n_subjects:    int,
    t_per_subject: int,
    group_offset:  int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ECM in SOURCE and RECONSTRUCTED space per subject.

    For each subject:
        1. Slice X_sub from the concatenated BOLD              (N, t_per_subject)
        2. Z-score X_sub across time                           (N, t_per_subject)
        3. Reduce:   Z_sub  = reducer.transform(X_sub_z)       (k, t_per_subject)
        4. Invert:   X_hat  = reducer.inverse_transform(Z_sub) (N, t_per_subject)
        5. ecm_source[sub]        = ECM observable on X_sub_z.T
        6. ecm_reconstructed[sub] = ECM observable on X_hat.T

    Both X_sub_z and X_hat are in N-dimensional parcel space, so their
    ECM values are directly comparable. The per-subject Pearson correlation
    r(ecm_source, ecm_reconstructed) across subjects is the key quality
    metric from Figure 2(b) of Deco et al. (2025).

    Parameters
    ----------
    X : np.ndarray, shape (N, Tm)
        Full concatenated BOLD in Neuroreduce convention (N, Tm).
        This is the same array passed to reducer.fit().
    reducer : DimensionalityReducer
        A fitted Neuroreduce reducer. Must implement transform() and
        inverse_transform().
    n_subjects : int
        Number of subjects in this group.
    t_per_subject : int
        Timepoints per subject (= Tmsub).
    group_offset : int
        Starting column in X for this group. Default: 0.

    Returns
    -------
    ecm_source : np.ndarray, shape (n_subjects,)
        Per-subject ECM of the ORIGINAL BOLD.
    ecm_reconstructed : np.ndarray, shape (n_subjects,)
        Per-subject ECM of the RECONSTRUCTED BOLD.

    Notes
    -----
    Assumption: X_sub is z-scored before ECM and before reduction so
    both signals are on the same scale going into the observable.
    X_hat is NOT re-z-scored after inversion — re-scaling would destroy
    the amplitude information the reconstruction provides and make the
    ECM comparison meaningless.
    """
    # Instantiate observable once and reuse across subjects
    obs               = _ECMObservable()
    ecm_source        = np.zeros(n_subjects)
    ecm_reconstructed = np.zeros(n_subjects)

    for sub in range(n_subjects):
        start = group_offset + sub * t_per_subject
        end   = group_offset + (sub + 1) * t_per_subject

        # Original BOLD slice in Neuroreduce convention: (N, t_per_subject)
        X_sub = X[:, start:end]

        # Z-score across time — axis=1 because shape is (N, T) here
        X_sub_z = stats.zscore(X_sub, axis=1, ddof=1)   # (N, t_per_subject)

        # ECM of original BOLD.
        # Transpose X_sub_z.T → (T, N) for NeuroNumba observable convention.
        result_src      = obs.from_fmri(X_sub_z.T)
        ecm_source[sub] = result_src['ECM']

        # Reconstruct: reduce → invert, all in Neuroreduce (N, T) convention
        Z_sub = reducer.transform(X_sub_z)           # (k, t_per_subject)
        X_hat = reducer.inverse_transform(Z_sub)     # (N, t_per_subject)

        # ECM of reconstructed BOLD.
        # Transpose X_hat.T → (T, N) for NeuroNumba observable convention.
        # Assumption: NOT re-z-scored — see Notes above.
        result_rec             = obs.from_fmri(X_hat.T)
        ecm_reconstructed[sub] = result_rec['ECM']

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if np.any(~np.isfinite(ecm_source)) or np.any(~np.isfinite(ecm_reconstructed)):
        warnings.warn(
            "Non-finite ECM values detected. Check that "
            "reducer.inverse_transform() produces a valid (N, T) signal.",
            RuntimeWarning,
            stacklevel=2,
        )
    print(f"    [ECM diagnostic]  source       : "
          f"min={ecm_source.min():.3f}  max={ecm_source.max():.3f}")
    print(f"    [ECM diagnostic]  reconstructed: "
          f"min={ecm_reconstructed.min():.3f}  max={ecm_reconstructed.max():.3f}")

    return ecm_source, ecm_reconstructed
