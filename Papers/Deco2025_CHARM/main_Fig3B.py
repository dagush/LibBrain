import numpy as np
import scipy.io as sio

from DataLoaders.HCP_dbs80 import HCP
from hopf import *
from signal_processing import *
import observables as obs


# dataPath = './_Data_Produced/hopf_parms_HCP_dbs80_REST1.npz'
diff_path = './_Data_Produced/results_f_diff_REST_dk62.mat'
SC_path = './_Data_Produced/SC_dbs80HARDIFULL.mat'
output_path = './_Data_Produced/results_hopf_fitt_KoP_fineG0203.npz'


def run(seed=None):
    if seed is not None:
        np.random.seed(seed)

    # ============================================================
    # --- Load data -------------------- -------------------------
    # ============================================================
    # DL = HCP()
    # SC = DL.get_AvgSC_ctrl()
    # data = np.load(dataPath)
    diff_mat = sio.loadmat(diff_path)
    SC_mat = sio.loadmat(SC_path)

    # ============================================================
    # --- PARAMETERS (from MATLAB header) -------------------------
    # ============================================================
    num_subjects = 30
    N = 62

    TR = 0.72         # sampling interval
    Tmax = 1200       # number of time points
    burn_in = 2000    # transient time

    sigma = 0.01      # noise amplitude (example, adjust)
    dt = 0.1 * TR/2.  # integration step

    # ----------------------------------------------------
    # --- Model parameters (vectorized form) -------------
    # ----------------------------------------------------
    a = -0.02
    a_vec = a * np.ones(N)

    # ============================================================
    # --- MAIN STRUCTURE -----------------------------------------
    # ============================================================
    index = list(range(0,31)) + list(range(49,80))

    # SC = data['SC']
    SC = SC_mat['SC_dbs80HARDI']
    SC = SC / np.max(SC)
    SC = SC[np.ix_(index,index)]

    # fc_emp = data['fc_emp']
    fc_emp = diff_mat['FCemp']
    # fc_emp = fc_emp[np.ix_(index,index)]

    # omega = data['omega'][np.ix_(index)]
    f_diff = diff_mat['f_diff'].flatten()
    omega = f_diff * 2 * np.pi
    omega_vec = np.zeros((N,2))
    omega_vec[:,0] = omega
    omega_vec[:,1] = omega

    subj_range = range(num_subjects)
    G_range = np.arange(0.0, 0.7, 0.01)

    # --- Loop over coupling strength ---
    res = {}
    for subject_id in subj_range:  #
        # --- Loop over simulated subjects ---
        subj_res = {}
        for G in G_range:
            print(f'Simulating subject {subject_id} at G={G}')
            # ----------------------------------------------------
            # --- HAND-OFF TO THE SIMULATION  --------------------
            # ----------------------------------------------------
            ts, debug = simulate(
                SC=SC,
                a=a_vec,
                omega=omega_vec,
                G=G,
                dt=dt,
                sigma=sigma,
                Tmax=Tmax,
                TR=TR,
                burn_in=burn_in,
            )
            print(f'   simulation (G={G}) done!')

            # ----------------------------------------------------
            # --- analysis ---------------------------------------
            # ----------------------------------------------------
            signal_filt = filer_fMRI(ts, TR)
            fc_sim = obs.compute_fc(ts[50:1150,:])
            comp = obs.compare_fc(fc_sim, fc_emp)
            fitt = comp['corr']
            err = comp['mse']

            metastability = obs.compute_metastability(ts)
            phases = compute_phases(signal_filt)
            kop, KoPMeta = obs.compute_kuramoto(phases)
            subj_res[G] = {'corr': fitt, 'mse': err, 'KoP': kop, 'KoPMeta': KoPMeta, 'metastability': metastability}
        res['S_' + str(subject_id)] = subj_res
    print('Simulation done!')
    np.savez(output_path, **res)
    print(f'Saved {output_path}')


if __name__ == '__main__':
    run(42)
