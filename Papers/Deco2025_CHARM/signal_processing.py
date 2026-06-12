import numpy as np
from scipy.signal import hilbert

from neuronumba.tools.filters import BandPassFilter


def filer_fMRI(fMRI, TR):  # fMRI in (time, RoIs) format
    # ========================================================================
    # We create the bandpass filter we will use for the signals
    # 3 Filters(Bandpass 0.008 - 0.08 Hz)
    flp = 0.008
    fhi = 0.08
    k = 2

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


def compute_phases(fMRI_signal):
    fMRI_det = fMRI_signal - fMRI_signal.mean(axis=0)  # not really needed, kept for backwards compatibility
    analytic_signal = hilbert(fMRI_det, axis=0)
    angle = np.angle(analytic_signal)
    return angle