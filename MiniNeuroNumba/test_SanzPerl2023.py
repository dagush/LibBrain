# =======================================================================
# Convenience simplification layer for NeuroNumba:
#     https://github.com/neich/neuronumba
#
# By Albert Juncà
# adapted by Gustavo Patow
# =======================================================================
import argparse
import os
import time
import math

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

# If need to debug numba code, uncomment this
# from numba import config
# config.DISABLE_JIT = True

import SanzPerl2023
from compact_generic_bold_model import Compact_Simulator


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tmax", help="Simulation time (milliseconds)", type=float, default=1000.0)
    parser.add_argument("--tr", help="Temporal resolution (TR) for the BOLD signal (milliseconds)", type=float, default=2000.0)
    parser.add_argument("--dt", help=("Simulation delta-time (milliseconds)."), type=float, default=0.1)
    parser.add_argument("--g", help="Global scaling for SC matrix normalization", type=float, default=1.0)

    args = parser.parse_args()
    return args  # returns something like: Namespace(model='Hopf', tmax=10000.0, tr=2000.0, dt=100, g=1.0)


def run():
    args = parse_arguments()

    # The number of nodes
    n_rois = 90

    # We generate a Mock-up structural connectivity (SC) matrix for the purpose of the example. In a real-world scenario
    # you should use the real one.
    # sc_norm = np.random.uniform(0.05, 0.2, size=(n_rois, n_rois))
    # np.npfill_diagonal(sc_norm, 0.0)
    sc_norm = sio.loadmat('./_Data_Raw/CNT_S01_structure.mat')['CNT_S01_structure']
    plt.matshow(sc_norm)
    plt.show()


    compact_simulator = Compact_Simulator(
        model = SanzPerl2023.ExactMeanField2023(),
        obs_var = 'R_e',
        weights = sc_norm,
        use_temporal_avg_monitor = False,
        g = args.g,
        sigma = 1e-03,
        tr = args.tr,
        dt = args.dt
    )

    simulated_bold = compact_simulator.generate_bold(
        warmup_samples = 100, # This samples will be discarded
        simulated_samples = math.ceil(args.tmax / args.dt)  # Number of useful samples to generate, this will be the size of the generated bold
    )

    fig, axs = plt.subplots(1)
    fig.suptitle(f'Result for model SanzPerl2023 (g={args.g})')
    axs.plot(np.arange(simulated_bold.shape[0]), simulated_bold)
    plt.show()


if __name__ == '__main__':
    run()
