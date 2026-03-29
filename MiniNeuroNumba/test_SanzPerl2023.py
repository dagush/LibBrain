# =======================================================================
# Convenience simplification layer for NeuroNumba:
#     https://github.com/neich/neuronumba
#
# By Albert Juncà
# adapted by Gustavo Patow
# =======================================================================
import argparse
import math

import numpy as np
import scipy.io as sio
from scipy.signal import detrend
from matplotlib import pyplot as plt

# If need to debug numba code, uncomment this
from numba import config
config.DISABLE_JIT = True

from neuronumba.tools.filters import BandPassFilter

import SanzPerl2023
from compact_generic_bold_model import Compact_Simulator


def filer_fMRI(fMRI):  # fMRI in (time, RoIs) format
    # ========================================================================
    # We create the bandpass filter we will use for the signals
    # 3 Filters(Bandpass 0.008 - 0.08 Hz)
    flp = 0.008
    fhi = 0.08
    k = 2
    TR = 2.0

    bpf = BandPassFilter(
        k=k,
        flp=flp,
        fhi=fhi,
        tr=TR * 1000.,
        apply_detrend=True,
        apply_demean=True,
        remove_artifacts=True
    )
    return bpf.filter(fMRI)


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

    # We generate a Mock-up structural connectivity (SC) matrix for the purpose of the example. In a real-world scenario
    # you should use the real one.
    # sc_norm = np.random.uniform(0.05, 0.2, size=(n_rois, n_rois))
    # np.npfill_diagonal(sc_norm, 0.0)
    sc_norm = sio.loadmat('./_Data_Raw/CNT_S01_structure.mat')['CNT_S01_structure']
    sc_norm = sc_norm / np.max(sc_norm) * 0.2  # Normalize
    # plt.matshow(sc_norm)
    # plt.show()

    # ts = sio.loadmat('./_Data_Raw/CNT.mat')['ts_emp_raw']
    # # Reorder AAL to Deco
    # left_idx = list(range(0, 90, 2))
    # right_idx = list(range(89,0,-2))
    # order_deco = left_idx + right_idx
    # ts_emp = ts[order_deco,:]
    # ts_emp = detrend(ts_emp)
    # ts_emp_filt = filer_fMRI(ts_emp.T).T
    # FC_emp = np.corrcoef(ts_emp_filt)

    tr = 2.0
    dt = 0.01  # milliseconds (1e-5 seconds)
    Tmax_vol = 295
    T_sim_seconds = (Tmax_vol * tr)
    T_warm_seconds = 20

    compact_simulator = Compact_Simulator(
        model = SanzPerl2023.EMFM(),
        obs_var = 'r_e',
        weights = sc_norm,
        use_temporal_avg_monitor = False,
        g = 5.30,
        sigma = 1e-03,
        tr = tr*1000,  # milliseconds
        dt = dt   # milliseconds
    )

    simulated_bold = compact_simulator.generate_bold(
        warmup_time = T_warm_seconds*1000, # This samples will be discarded
        simulated_time = T_sim_seconds*1000  # Number of useful samples to generate, this will be the size of the generated bold
    )

    fig, axs = plt.subplots(1)
    fig.suptitle(f'Result for model SanzPerl2023 (g={args.g})')
    axs.plot(np.arange(simulated_bold.shape[0]), simulated_bold)
    plt.show()


if __name__ == '__main__':
    run()
