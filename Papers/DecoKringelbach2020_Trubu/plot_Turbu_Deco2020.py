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
import pandas as pd

import matplotlib.pyplot as plt
import Plotting.p_values as pValues

# =======================================================================
# paths
# =======================================================================

dataPath = './_Data_Produced/turbu.csv'
resultsPath = './_Results/Figure_2A.pdf'


def plotTurbu_lambda(ax, turbuRes, observ_name, lambda_val):
    # --------------------------------------------------------------------------------------------
    # Comparisons of Amplitude Turbulence (D) across groups
    # --------------------------------------------------------------------------------------------
    df_feat = turbuRes[(turbuRes['type'] == 'turbu') & (turbuRes['obs'] == observ_name)].dropna()
    df_surr = turbuRes[(turbuRes['type'] == 'surrogate') & (turbuRes['obs'] == observ_name)].dropna()
    # Prepare data grouped by task
    feat_vals = [float(s.replace('[','').replace(']','')) for s in df_feat['value'].values]
    surr_vals = [float(s.replace('[','').replace(']','')) for s in df_surr['value'].values]
    data = {'Empirical': feat_vals, 'Surrogate': surr_vals}
    pValues.plotComparisonAcrossLabels2Ax(ax, data,
                                          graphLabel=fr'D ($\lambda={lambda_val}$)',
                                          test='Mann-Whitney',
                                          comparisons_correction=None)  # 'BH'/None


def plotTurbuAttr(lambdas, data_emp, observ, title):
    if len(lambdas) == 1:
        fig, axs = plt.subplots(1, 1)
        axs = [axs]
    else:
        fig, axs = plt.subplots(2, int(len(lambdas)/2))
        axs = axs.reshape(-1)
    for ax, lambda_val in zip(axs, lambdas):
        print(f'\n\nPlotting Turbu lambda: {lambda_val}')
        plotTurbu_lambda(ax, data_emp, observ, lambda_val)
    plt.suptitle(title)
    plt.savefig(resultsPath)
    # plt.show()


# =======================================================================
# main plot organization routines
# =======================================================================
def plotTurbu(rev_lambdas, turbus, observations):
    for obs in observations:
        print('\n\n############################################')
        print(f'#    Turbulence: {observations[obs]} #')
        print('############################################')
        plotTurbuAttr(rev_lambdas, turbus, obs, observations[obs])


# =======================================================================
# load results
# =======================================================================
def load_turbu(dataPath):
    df = pd.read_csv(dataPath, dtype={'value': object})
    return df

def run():
    _observations = {'R_spa_time': 'amplitude turbulence (D)',} # 'Transfer': 'Information Transfer'}
    lambdas = [0.18]

    # ------------- Information Cascade and Information Cascade Flow
    if os.path.exists(dataPath):
        turbus_ = load_turbu(dataPath)
        plotTurbu(lambdas, turbus_, _observations)
    else:
        print(f'No turbulence data ({dataPath}))')


# =======================================================================
# ==========================================================================
if __name__=="__main__":
    run()
    print("done")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF