# =======================================================================
# Turbulence framework, main part. From:
# Gustavo Deco, Morten L. Kringelbach, Turbulent-like Dynamics in the Human Brain,
# Cell Reports, Volume 33, Issue 10, 2020, 108471, ISSN 2211-1247,
# https://doi.org/10.1016/j.celrep.2020.108471.
# (https://www.sciencedirect.com/science/article/pii/S2211124720314601)
#
# Part of the Thermodynamics of Mind framework:
# Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
# Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
# https://doi.org/10.1016/j.tics.2024.03.009
#
# Code by Gustavo Patow, June 9, 2024
# =======================================================================
import numpy as np
import pandas as pd

from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import Turbulence
# from neuronumba.observables import Information_transfer
from Plotting.p_values import printStats

import Utils.decorators as decorators

# ------------------------------ HCP Data Loader
import DataLoaders.HCP_Schaefer2018 as HPC
DL = HPC.HCP()
# ------------------------------

dataPath = './_Data_Produced/' + DL.name() + '/'


def clean_data(ts, CoGs, id):
    bad_regions = np.where(np.all(np.isnan(ts), axis=1))[0]
    ts_clean = np.delete(ts, bad_regions, axis=0)
    CoGs_clean = np.delete(CoGs, bad_regions, axis=0)
    print(f"Subject {id}: {len(bad_regions)} parcels removed due to missing time series")
    return ts_clean, CoGs_clean


def save_results(all_results, path):
    rows = []
    for subj in all_results:
        for lam in all_results[subj]:
            turbu = all_results[subj][lam]['turbu']
            for obs in turbu:
                rows.append({'id': subj, 'lambda': lam, 'type': 'turbu', 'obs': obs, 'value': turbu[obs]})
            surr = all_results[subj][lam]['surrogate']
            for obs in surr:
                rows.append({'id': subj, 'lambda': lam, 'type': 'surrogate', 'obs': obs, 'value': surr[obs]})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Results saved tp {path}")

@decorators.loadOrCompute
def from_fMRI(Turbu, ts):
    return Turbu.from_fmri(ts)


@decorators.loadOrCompute
def from_fMRI_surrogate(Turbu, ts):
    return Turbu.from_surrogate(ts)


def computeTurbu_subj(subj, timeseries, range, DL):
    coords = DL.get_parcellation().get_CoGs()
    timeseries, coords = clean_data(timeseries, coords, subj)

    bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=DL.TR()*1000.)   # Define a band pass filter

    # fullDataPath = dataPath + f'subj_{subj}/'
    # if not os.path.exists(fullDataPath):
    #     os.makedirs(fullDataPath)

    # =======================================================================
    # dictionaries for each subject
    # =======================================================================
    all_results = {}
    for lambda_v in range:
        print(f'Processing subj: {subj} @ lambda: {lambda_v}')

        # =======================================================================
        # Define the turbulence object
        # =======================================================================
        Turbu = Turbulence(cog_dist=coords, lambda_val=lambda_v, ignore_nans=True)
        # Turbu = Information_transfer(cog_dist=coords, lambda_val=lambda_v, ignore_nans=True)
        Turbu.configure()

        # VERY IMPORTANT: For performance reasons, the filter expects the signal to be in the
        # transposed form (n_time_samples, n_rois). We have to transpose it before passing it
        bold_filt = bpf.filter(timeseries.T)  # we keep it transposed...
        # ======================= main analysis
        # Compute the observable
        subjPath = dataPath + f'turbu_{subj}_{lambda_v}_main.mat'
        turbuRes = from_fMRI(Turbu, bold_filt, subjPath)
        # ======================= Surrogate analysis
        # Compute the surrogate
        subjPath = dataPath + f'turbu_{subj}_{lambda_v}_surrogate.mat'
        surrogateRes = from_fMRI_surrogate(Turbu, bold_filt, subjPath)
        # ======================= Done analysis
        all_results[lambda_v] = {"turbu": turbuRes, "surrogate": surrogateRes}
    print(f"done {subj} !!")
    return all_results


def computeTurbu(range, DL, path):
    all_results = {}
    classific = DL.get_classification()
    c = list(classific.keys())  #[0:2]
    for subj in c:
        print(f'Computing Turbu, subj: {subj}')
        subjData = DL.get_subjectData(subj)
        timeseries = subjData[subj]['timeseries']
        all_results[subj] = computeTurbu_subj(subj, timeseries, range, DL)

    save_results(all_results, path)

# =======================================================================
# ==========================================================================
if __name__=="__main__":
    # decorators.forceCompute = True  # Use this to force re-computations.
    # lambdas = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27]
    lambdas = [0.18]
    rev_lambdas = list(reversed(lambdas))  # To have the same order as in Matlab
    path = f'./_Data_Produced/turbu.csv'
    computeTurbu(rev_lambdas, DL, path)
    print("done")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF