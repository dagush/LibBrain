# ==========================================================================
# ==========================================================================
#  Computes SIMULATIONS of Placebo and LSD with the Dynamic Mean Field Model (DMF) using
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#  - the optimal coupling (G=2.1, we in the original) for fitting the placebo condition
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#  Before this, needs the results computed in
#   - pipeline_fgain_PlaceboLSD.py to get the G=2.1 value...
#
#  Taken from the code (FCD_LSD_model.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C., Logothetis,N.K.
#       & Kringelbach,M.L.; Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear
#       functional effects of LSD (2018) Current Biology
#       https://www.cell.com/current-biology/fulltext/S0960-9822(18)31045-5
#
#  The FIC optimization was simplified to the one described in:
#  [HerzogEtAl_2024] Rubén Herzog, Pedro A. M. Mediano, Fernando E. Rosas, Andrea I. Luppi, Yonatan Sanz-Perl, Enzo
#       Tagliazucchi, Morten L. Kringelbach, Rodrigo Cofré, Gustavo Deco; Neural mass modeling for the masses:
#       Democratizing access to whole-brain biophysical modeling with FastDMF. Network Neuroscience 2024;
#       8 (4): 1590–1612.
#       doi: https://doi.org/10.1162/netn_a_00410
#
#  Code written by Gustavo Deco gustavo.deco@upf.edu 2017:
#  Reviewed by Josephine Cruzat and Joana Cabral
#
#  Translated to Python by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio

# from DataLoaders.HCP_dbs80 import HCP
from deco2018 import *
from signal_processing import *
import observables as obs


# base_path = '/Users/dagush/Dpt. IMAE Dropbox/Gustavo Patow/'  # for Mac
base_path = 'L:/Dpt. IMAE Dropbox/Gustavo Patow/'  # for Win
input_path = base_path + 'SRC/Neuro-Data/Papers-Data/Deco2018/'
output_path = './_Data_Produced/results_Deco2018_{}.npz'


def run_sim(wgaine, wgaini, suffix):
    # ============================================================
    # --- Load data ----------------------------------------------
    # ============================================================
    data = sio.loadmat(input_path + 'all_SC_FC_TC_76_90_116.mat')
    sc90 = data['sc90']
    SC = sc90 / np.max(sc90) * 0.2
    N = SC.shape[0]

    mean5HT2A_aalsymm = sio.loadmat(input_path + 'mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
    receptor = mean5HT2A_aalsymm[:,0] / np.max(mean5HT2A_aalsymm[:,0])

    # ============================================================
    # --- PARAMETERS (from MATLAB header) -------------------------
    # ============================================================
    burn_in = 0       # transient time (seconds)

    sigma = 0.01      # noise amplitude (example, adjust)
    dt = 0.1          # integration step (0.1 * dtt = 0.1 milliseconds)
    dtt = 1e-3        # Sampling rate of simulated neuronal activity (seconds)

    TR = 2      # Sampling rate of saved simulated BOLD (seconds)
    NSUB = 15   # Number of Subjects in empirical fMRI dataset
    Tmax = 220  # Number of timepoints in each fMRI session (seconds)


    N_windows = int(190./3.) + 1
    cotsampling_pla_s = np.zeros((NSUB, int(N_windows * (N_windows - 1) / 2)))

    # ----------------------------------------------------
    # --- Model parameters -------------------------------
    # ----------------------------------------------------
    G = 2.1  # Global Coupling parameter
    J = 1 + 0.75 * G * np.sum(SC, 0)  # Herzog FIC mechanism [HerzogEtAl_2024]

    # ============================================================
    # --- MAIN STRUCTURE -----------------------------------------
    # ============================================================
    res = {}
    for subject_id in range(NSUB):
        # --- Loop over simulated subjects ---
        print(f'Simulating subject {subject_id} at G={G}')
        # ----------------------------------------------------
        # --- HAND-OFF TO THE SIMULATION  --------------------
        # ----------------------------------------------------
        ts, bold, debug = simulate(
            # -------- model parms
            SC=SC,
            G=G,
            J=J,
            # -------- Specific Deco 2018 parms
            Receptor=receptor,
            wgaine=wgaine,
            wgaini = wgaini,
            # -------- Simulation parms
            TR=TR,
            dt=dt,
            dtt=dtt,
            sigma=sigma,
            Tmax=Tmax,
            burn_in=burn_in,
        )
        print(f'   simulation (G={G}) done!')

        # ----------------------------------------------------
        # --- analysis ---------------------------------------
        # ----------------------------------------------------
        signal_filt = filer_fMRI(bold, TR)
        res_obs = obs.compute_swFCD(signal_filt)

        res['S_' + str(subject_id)] = res_obs
    print('Simulation done!')
    np.savez(output_path.format(suffix), **res)
    print(f'Saved {output_path.format(suffix)}')


def run(seed=None):
    if seed is not None:
        np.random.seed(seed)
    # SIMULATION OF OPTIMAL PLACEBO FIT
    wge = 0 # 0 for placebo, 0.2 for LSD
    wgaini = 0
    wgaine = wge
    run_sim(wgaine, wgaini, suffix='PLA')

    # SIMULATION OF OPTIMAL LSD fit
    wge = 0.2; # 0 for placebo, 0.2 for LSD
    wgaini = 0
    wgaine = wge
    run_sim(wgaine, wgaini, suffix='LSD')


if __name__ == '__main__':
    run(42)
