import numpy as np
from scipy.signal import periodogram, welch
import matplotlib.pyplot as plt


# ------------------------------------------------
def plot_means_and_std(fMRI_signal):
    means = np.mean(fMRI_signal, axis=1)
    global_mean = np.mean(fMRI_signal)
    stdevs = np.std(fMRI_signal, axis=1)
    global_std = np.std(fMRI_signal)
    plt.plot(range(DL.N()), means, label='mean')
    plt.plot(range(DL.N()), stdevs, label='stdev')
    plt.axhline(global_mean, color='red', ls=':', label=f'mean={global_mean:.2f}')
    plt.axhline(global_std, color='blue', ls=':', label=f'std={global_std:.2f}')
    plt.title('Plot of the means and std')
    plt.xlabel('Amplitude')
    plt.ylabel('RoI')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# Plot the power law of fMRI
def plot_power_law(fMRI_signal, type):  # periodogram / Welch
    N = len(fMRI_signal)
    fs = 1. / DL.TR()
    for i in range(N):
        signal = fMRI_signal[i]
        if type == 'periodogram':
            # Calculating the periodogram
            frequencies, psd_values = periodogram(signal, fs)
        else:
            # Apply Welch's method to estimate the PSD
            frequencies, psd_values = welch(signal, fs, nperseg=200)
        plt.plot(frequencies, psd_values)

    plt.xscale('log')
    plt.yscale('log')
    if type == 'periodogram':
        plt.title(f'Periodogram of fMRI signal at RoI {RoI} for {first}')
    else:
        plt.title('Power Spectral Density (PSD) Estimate using Welch\'s Method')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.show()


# ------------------------------------------------
# Plot FFT
def plot_FFT(fMRI_signal, t, low_lim=1, first_subject=''):
    N = len(fMRI_signal)
    locations = None
    for i in range(N):
        signal = fMRI_signal[i]

        # Applying FFT
        fft_result = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(t.shape[-1], d=DL.TR())
        abs_fft = np.abs(fft_result)

        maxs = np.where(abs_fft > np.mean(abs_fft))[0]
        locations = freq[[maxs[0], maxs[-1]]]

        # ----- Plotting the spectrum
        plt.plot(freq[low_lim:], abs_fft[low_lim:])

    print(f'min freq={locations[0]}\nmax freq={locations[1]}')
    nyquist = 1. / (2 * DL.TR())
    print(f'Nyquist freq={nyquist}')
    plt.axvline(locations[0], color='red', ls=':', label=f'{locations[0]}' )
    plt.axvline(locations[1], color='purple', ls=':', label=f'{locations[1]}' )
    plt.axvline(nyquist, color='green', ls='--', label=f'nyquist ({nyquist})' )  # Registration limit of the sampling, Nyquist theorem!!!
    plt.title(f'FFT of fMRI signal for {first_subject}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


def analyze_signals(DL):
    first_subject = list(DL.get_classification().keys())[0]
    fMRI = DL.get_subjectData(first_subject)[first_subject]['timeseries']
    t_max = fMRI.shape[1]
    t = np.linspace(0, t_max*DL.TR(), t_max)

    plot_FFT(fMRI, t, first_subject=first_subject)
    plot_power_law(fMRI, type='Welch')  # periodogram / Welch
    plot_means_and_std(fMRI)


# =======================================================================
# ==========================================================================
if __name__=="__main__":
    # ADNI_version = 'matching'  # ADNI3 / IRFSPGR / matching
    # DL = ADNI_B.ADNI_B(ADNI_version=ADNI_version, SchaeferSize=400)
    # DL = ADNI_A.ADNI_A(cutTimeSeries=True);
    # import DataLoaders.MAS_W4 as MAS
    # DL = MAS.MAS_W4(AALSize=512)  # 88 / 512
    import DataLoaders.HCP_dbs80 as HCP_dbs80
    DL = HCP_dbs80.HCP()
    # import DataLoaders.Wakefulness as Wake
    # DL = Wake.Wakefulness()

    analyze_signals(DL)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF

