"""
check_EDR.py
=====================
Integration tests for EDR distance rule and its use inside Turbulence.

Each function is self-contained and can be called directly during development
or collected automatically by pytest.

Originally written by Giuseppe Pau (Dec 2025).
Refactored into importable test functions by Gustavo Patow, 2025.

Original code by Giuseppe Pau, December 2025.
Refactored by Gustavo Patow, April 6, 2026.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import DataLoaders.HCP_Schaefer2018 as HCP
from neuronumba.observables.distance_rule import EDR_distance_rule, EDR_LR_distance_rule


# ---------------------------------------------------------------------------
# Test 1 — EDR_LR_distance_rule with real data
# ---------------------------------------------------------------------------

def check_distance_rule(DL, resultsPath):
    """
    Verify the EDR and EDR_LR distance rules against real data.

    Checks
    ------
    - rr and c_exp shapes match the number of parcels.
    - Exponential fit converges and returns a positive lambda.

    Parameters
    ----------
    DL : DataLoader object
    """

    # ------------------------------------------------------------------
    # 1) Data loading
    # ------------------------------------------------------------------
    coords = DL.get_parcellation().get_CoGs()
    print("CoG shape:", coords.shape)

    SC_matrix = np.array(DL.get_AvgSC_ctrl(), dtype=float)
    SC_max = SC_matrix.max()
    if SC_max > 0:
        SC_matrix /= SC_max
    print("SC matrix shape:", SC_matrix.shape)

    # ------------------------------------------------------------------
    # 2) Plain EDR
    # ------------------------------------------------------------------
    num_bins      = 400  # 144
    lambda_val    = 0.18
    edr_rule      = EDR_distance_rule(lambda_val=lambda_val)
    rr, c_exp     = edr_rule.compute(coords)
    print("rr shape:", rr.shape)
    print("c_exp shape:", c_exp.shape)

    assert rr.shape    == (coords.shape[0], coords.shape[0]), "rr shape mismatch"
    assert c_exp.shape == rr.shape,                           "c_exp shape mismatch"

    # ------------------------------------------------------------------
    # 3) Histogram + exponential fit
    # ------------------------------------------------------------------
    means, stds, bin_edges, maxs = edr_rule.compute_hist(c_exp, rr, num_bins)
    centers  = (bin_edges[:-1] + bin_edges[1:]) / 2
    A1_fit, lambda_fit = edr_rule.fit_exponential(centers, means)
    print(f"\nEstimated A1: {A1_fit:.4f}  lambda: {lambda_fit:.4f}")

    assert lambda_fit > 0, "Fitted lambda should be positive"

    plt.figure(figsize=(8, 6))
    plt.errorbar(centers, means, yerr=stds, fmt='o', markersize=4,
                 capsize=3, label='Histogram')
    plt.plot(centers, A1_fit * np.exp(-lambda_fit * centers),
             'r-', label=f'Exp fit: λ={lambda_fit:.3f}')
    plt.plot(centers, maxs, 'green', label='Max')
    plt.xlabel("Distance")
    plt.ylabel("Mean c_exp (± std)")
    plt.title("Histogram: EDR decay with distance")
    plt.legend()
    plt.grid(True)
    plt.savefig(resultsPath)
    # plt.show()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    resultsPath = './_Results/Figure_3A.pdf'

    DL = HCP.HCP()

    check_distance_rule(DL, resultsPath)
