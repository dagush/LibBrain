# =======================================================================
#
# by Gustavo Patow, June 9, 2024
# =======================================================================
import numpy as np
import pandas as pd

from neuronumba.tools.filters import BandPassFilter
import neuronumba.observables.turbulence2 as turbu2
from MiniNeuroNumba.compact_bold_simulator import CompactHopfSimulator

import Utils.decorators as decorators
from fitting.EDR.exponential_distance_rule import EDR_distance_rule, EDR_LR_distance_rule

decorators.forceCompute = False  # Use this to force re-computations.

## ------------------------------ HCP Data Loader
import DataLoaders.HCP_Schaefer2018 as HPC
DL = HPC.HCP()
# ------------------------------

dataPath = './_Data_Produced/'


def clean_data(ts, id, CoGs=None):
    bad_regions = np.where(np.all(np.isnan(ts), axis=1))[0]
    ts_clean = np.delete(ts, bad_regions, axis=0)
    CoGs_clean = np.delete(CoGs, bad_regions, axis=0) if CoGs is not None else None
    print(f"Subject {id}: {len(bad_regions)} parcels removed due to missing time series")
    return ts_clean, CoGs_clean


def save_results(all_results, path):
    rows = []
    for subj in all_results:
        for obs_lam in all_results[subj]:
            value = all_results[subj][obs_lam].flatten()
            obs = obs_lam.split('-')[0]
            lam = float(obs_lam.split('-')[1]) if len(obs_lam.split('-')) > 1 else None
            rows.append({'id': subj, 'obs': obs, 'lambda': lam, 'value': value})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Results saved tp {path}")


# =======================================================================
# =======================================================================
@decorators.loadOrCompute
def from_fMRI(Turbu, ts):
    res = Turbu.from_fmri(ts)
    return res


# @decorators.loadOrCompute
# def from_fMRI_surrogate(Turbu, ts):
#     return Turbu.from_surrogate(ts)


def compute_information_cascade_subject(subj, bpf, turbu, fullDataPath, lambdas):
    print(f'Processing subj: {subj} @ lambdas: {lambdas}')
    subjData = DL.get_subjectData(subj)
    timeseries = subjData[subj]['timeseries']
    ts, CoGs = clean_data(timeseries, subj)
    # VERY IMPORTANT: signal from the simulator si returned with shape (n_time_samples, n_rois),
    # but, for performance reasons, the filter expects it in the transposed form (n_rois, n_time_samples).
    # We have to transpose it before passing it
    bold_filt = bpf.filter(ts.T)
    # Load and compute the observable
    subjPath = fullDataPath + f'turbu_{subj}.mat'
    turbuRes = from_fMRI(turbu, bold_filt, subjPath)
    return turbuRes


def compute_information_cascade(lambdas, dist_rule, dist_rule_name):
    fullDataPath = dataPath + dist_rule_name + '/'
    coords = DL.get_parcellation().get_CoGs()

    bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=DL.TR())   # Define a band pass filter

    sujes = list(DL.get_classification().keys())[0:20]
    first = DL.get_subjectData(sujes[0])[sujes[0]]['timeseries']
    first, CoGs = clean_data(first, sujes[0], CoGs=coords)
    # =======================================================================
    # Define the turbulence object
    # =======================================================================
    # Turbu = Turbulence(cog_dist=coords, lambda_val=lambda_v, ignore_nans=True)
    Turbu = turbu2.Information_cascade(cog_dist=CoGs, lambda_values=lambdas,
                                       distance_rule=dist_rule)
    # Turbu = turbu2.Information_transfer(cog_dist=coords,
    #                                     distance_rule=DR.EDR_LR_distance_rule(lambda_val=lambdas[0]))
    Turbu.configure()

    # =======================================================================
    # dictionaries for each subject
    # =======================================================================
    # bpfs = [bpf for _ in sujes]
    # fullPaths = [fullDataPath for _ in sujes]
    # pool = ProcessPool(nodes=8)
    # results = pool.map(compute_information_cascade_subject,
    #                    sujes, bpfs, turbus, fullPaths)
    # for subj_, bpf_, fullPath_ in zip(sujes, bpfs, fullPaths):
    turbuRes = {}
    for subj in sujes:
        turbuRes[subj] = compute_information_cascade_subject(subj, bpf, Turbu, fullDataPath, lambdas)

    save_results(turbuRes, dataPath + f'turbu_{dist_rule_name}_res.csv')


# =======================================================================
# ==========================================================================
if __name__=="__main__":
    lambdas = [0.01, 0.03, 0.06, 0.09, 0.12,
               0.15, 0.18, 0.21, 0.24, 0.27]
    rev_lambdas = list(reversed(lambdas))  # To have the same order as in Matlab
    dr = EDR_distance_rule()
    compute_information_cascade(rev_lambdas, dr, 'EDR')
    dr_lr = EDR_LR_distance_rule()
    compute_information_cascade(rev_lambdas, dr_lr, 'EDR_LR')
    print("done")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF