# =======================================================================
# Turbulence framework, plotting part. From:
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
# Code by Gustavo Deco, 2020.
# Translated by Marc Gregoris, May 21, 2024
# Refactored by Gustavo Patow, June 9, 2024
# =======================================================================
import os
import numpy as np
import neuronumba.tools.hdf as sio
import matplotlib.pyplot as plt
import Plotting.p_values as pValues
# import Utils.numTricks as nTricks  # my numerical tricks
import neuronumba.tools.matlab_tricks as mTricks  # matlab compatibility tricks

# ------------------------------ Data Loader
# ----------- ADNI_A
import DataLoaders.ADNI_A as ADNI
DL = ADNI.ADNI_A(use360=False, cutTimeSeries=True)
# ----------- ADNI_B
# ADNI_version = 'N238rev' # ADNI3 / IRFSPGR
# import DataLoaders.ADNI_B_N238rev as ADNI_B
# DL = ADNI_B.ADNI_B_N238rev()
# ------------------------------

dataPath = './_Data_Produced/' + DL.name() + '/'

# def calculate_stats(datas):
#     means = np.nanmean(datas, axis=0)
#     stds = np.nanstd(datas, axis=0)
#     return means, stds


def plotTurbu_lambda(ax, turbuRes, observ_name, lambda_val, range):
    index = range.index(lambda_val)
    # --------------------------------------------------------------------------------------------
    # Comparisons of Amplitude Turbulence (D) across groups
    # --------------------------------------------------------------------------------------------
    classific = DL.get_classification()
    subjects = DL.get_allStudySubjects()
    groups = DL.get_groupLabels()
    BOX_R_SPA = {group: [] for group in groups}
    observ = observ_name if observ_name in turbuRes[subjects[0]] else observ_name + f'-{lambda_val}'
    for subj in subjects:
        obs_lista = np.atleast_1d(np.squeeze(turbuRes[subj][observ]))
        elem = (obs_lista[index]
                if isinstance(obs_lista, list) or isinstance(obs_lista, np.ndarray)
                else obs_lista)
        if observ_name == 'Transfer': elem = 1-elem
        BOX_R_SPA[classific[subj]].append(elem)
    # for group in groups:
    #     BOX_R_SPA[group] = mTricks.reject_outliers(BOX_R_SPA[group])
    pValues.plotComparisonAcrossLabels2Ax(ax, BOX_R_SPA,
                                          graphLabel=fr'D ($\lambda={lambda_val}$)',
                                          test='Mann-Whitney',
                                          comparisons_correction=None)  # 'BH'/None


def plotTurbuAttr(range, data_emp, observ, title):
    if len(range) == 1:
        fig, axs = plt.subplots(1, 1)
        axs = [axs]
    else:
        fig, axs = plt.subplots(2, int(len(range)/2))
        axs = axs.reshape(-1)
    for ax, lambda_val in zip(axs, range):
        print(f'\n\nPlotting Turbu lambda: {lambda_val}')
        plotTurbu_lambda(ax, data_emp, observ, lambda_val, range)
    plt.suptitle(title)
    plt.show()


# =======================================================================
# main plot organization routines
# =======================================================================
def plotTurbu(rev_lambdas, turbus, observations):
    for obs in observations:
        print('\n\n############################################')
        print(f'#    Turbulence: {observations[obs]} #')
        print('############################################')
        plotTurbuAttr(rev_lambdas, turbus, obs, observations[obs])


def plotMeta(turbus, lambdas):
    print('\n\n############################################')
    print('#    Turbulence: Metastability             #')
    print('############################################')
    fig, ax = plt.subplots()
    plotTurbu_lambda(ax, turbus, 'Meta', lambdas[0])
    fig.suptitle('Metastability')
    plt.show()


def plotInfoCascadeFlow(range, turbus):
    print('\n\n############################################')
    print('#    Turbulence: Information Cascade Flow  #')
    print('############################################')
    plotTurbuAttr(range, turbus, 'TransferLambda', 'Information Cascade Flow')


def plotInfoCascade(turbus, lambdas):
    print('\n\n############################################')
    print('#    Turbulence: Information Cascade       #')
    print('############################################')
    fig, ax = plt.subplots()
    plotTurbu_lambda(ax, turbus, 'InformationCascade', lambdas[0])
    fig.suptitle('Information Cascade')
    plt.show()

# =======================================================================
# load results
# =======================================================================
def load_turbu(dataPath, lambdas, observ):
    turbus = {}
    for subj in DL.get_classification():
        turbus[subj] = {o: np.zeros(len(lambdas)) for o in observ}
        for pos, lambda_v in enumerate(lambdas):
            subjPath = dataPath + f'subj_{subj}/turbu_{subj}_{lambda_v}.mat'
            turubu_data = sio.loadmat(subjPath)
            for o in observ:
                turbus[subj][o][pos] = turubu_data[o]
    return turbus

# def load_turbu2(datapath):
#     turbus = {}
#     for subj in DL.get_classification():
#         turbus[subj] = sio.loadmat(datapath + f'subj_{subj}/turbu_{subj}.mat')
#     return turbus


def run(modality):
    if modality == 'Turbu':
        _observations = {'R_spa_time': 'amplitude turbulence (D)',}
                         # 'Transfer': 'Information Transfer'}
    # lambdas = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27]
    lambdas = [0.18]
    rev_lambdas = list(reversed(lambdas))  # To have the same order as in Matlab

    # ------------- Information Cascade and Information Cascade Flow
    if os.path.exists(dataPath):
        if modality == 'Turbu':
            turbus_ = load_turbu(dataPath, rev_lambdas, _observations)
            plotTurbu(rev_lambdas, turbus_, _observations)
        # turbus_ = load_turbu2(fullDataPath)
        # plotMeta(turbus_, rev_lambdas)
        # plotInfoCascadeFlow(rev_lambdas, turbus_)
        # plotInfoCascade(turbus_)


# =======================================================================
# ==========================================================================
if __name__=="__main__":
    run(modality='Turbu')  # Turbu / Turbu2
    print("done")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF