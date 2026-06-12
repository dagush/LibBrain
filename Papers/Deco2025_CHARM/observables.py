import numpy as np
from scipy.spatial.distance import pdist
from scipy import stats

import neuronumba.tools.matlab_tricks as mt


def compute_fc(ts):
    cc = np.corrcoef(ts, rowvar=False)  # Pearson correlation coefficients
    return cc


def compare_fc(fc_sim, fc_emp):
    fitt2 = mt.corr2(fc_sim, fc_emp)
    N = fc_emp.shape[0]
    Isubdiag = np.tril_indices(N, k=-1)
    err2 = np.mean(np.square(  fc_emp[Isubdiag]-fc_sim[Isubdiag]  ))
    return {'corr': fitt2, 'mse': err2}


def compute_metastability(ts):
    # Edges
    N = ts.shape[1]
    Isubdiag = np.tril_indices(N, k=-1)
    zPhi = stats.zscore(ts, axis=0)
    edges = np.zeros((len(Isubdiag[0]), zPhi.shape[0]))
    for t in range(zPhi.shape[0]):
        fcd = np.outer(zPhi[t], zPhi[t])
        edges[:, t] = fcd[Isubdiag].T
    FCD = pdist(edges)
    Metastability2 = 0.5 * np.log(2 * np.pi * np.var(FCD)) + 0.5
    return Metastability2


def compute_kuramoto(phases):
    T, N = phases.shape
    KoP = np.abs(np.nansum(np.cos(phases) + 1j * np.sin(phases), 1)) / N
    KoP = KoP[19:-20]
    KuramotoOrderParameter = np.average(KoP)
    KuramotoMetastability = np.std(KoP)
    return KuramotoOrderParameter, KuramotoMetastability
