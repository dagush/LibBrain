# ==========================================================================
# ==========================================================================
#  Plotting results of simulations of Placebo and LSD with the Dynamic Mean Field Model (DMF) using
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#  - the optimal coupling (G=2.1, we in the original) for fitting the placebo condition
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#  Before this, needs the results computed in
#   - pipeline_fgain_PlaceboLSD.py to get the G=2.1 value...
#
#  Taken from the code (FCD_LSD_model.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/fulltext/S0960-9822(18)31045-5
#
#  Code written by Gustavo Deco gustavo.deco@upf.edu 2017:
#  Reviewed by Josephine Cruzat and Joana Cabral
#
#  Translated to Python by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt


# SC_path = './_Data_Produced/SC_dbs80HARDIFULL.mat'
base_path = '/Users/dagush/Dpt. IMAE Dropbox/Gustavo Patow/'
# base_path = 'L:/Dpt. IMAE Dropbox/Gustavo Patow/'
input_path = base_path + 'SRC/Neuro-Data/Papers-Data/Deco2018/'
output_path = './_Data_Produced/results_Deco2018_{}.npz'

NSUB = 15  # Number of Subjects in empirical fMRI dataset


def my_hist(x, bin_centers):
    bin_edges = np.r_[-np.inf, 0.5 * (bin_centers[:-1] + bin_centers[1:]), np.inf]
    counts, edges = np.histogram(x, bin_edges)
    return [counts, bin_centers]


def load_data(modality):
    data = np.load(output_path.format(modality), allow_pickle=True)
    res = []
    for s in range(NSUB):
        res.append(data['S_' + str(s)].item()['swFCD'])
    return np.array(res)

def run():

    res = {}
    res['PLA'] = load_data(modality='PLA')
    res['LSD'] = load_data(modality='LSD')

    # ============================================================================
    # Plot
    # ============================================================================
    [h_pla, x1] = my_hist(res['PLA'].T.flatten(), np.arange(-.1, 1.025, .025))
    [h_lsd, x] = my_hist(res['LSD'].T.flatten(), np.arange(-.1, 1.025, .025))

    import matplotlib.pyplot as plt

    width = 0.01
    plaBar = plt.bar(x, h_pla, width=width, color="red", label="Placebo")
    lsdBar = plt.bar(x + width, h_lsd, width=width, color="blue", label="LSD")
    plt.xlabel('FCD values')
    plt.ylabel('Count')
    plt.legend(handles=[plaBar, lsdBar], loc='upper right')
    plt.title('Simulated data')
    plt.show()

if __name__ == '__main__':
    run()
