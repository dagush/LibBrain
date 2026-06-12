import numpy as np
import matplotlib.pyplot as plt

import Plotting.p_values_raincloud as plot


plot_data = './_Data_Produced/results_hopf_fitt_KoP_fineG0203.npz'


def load_data():
    res = {}
    data = np.load(plot_data, allow_pickle=True)
    for subj in data.keys():
        s_data = data[subj].item()
        res[subj] = s_data
    return res


def run():
    data = load_data()
    first = list(data.keys())[0]
    Gs = np.array([g for g in data[first]])
    metas = [np.average([data[s][g]['KoPMeta'] for s in data]) for g in Gs]
    errors = [np.average([data[s][g]['mse'] for s in data]) for g in Gs]

    fig, ax1 = plt.subplots()
    fig.set_figheight(4.5)
    fig.set_figwidth(8)
    ax1.set_xlabel('G')
    color = 'tab:orange'
    ax1.plot(Gs, metas, label='KoP', color=color)
    ax1.set_ylabel('KoP', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'yellowgreen'
    ax2.set_ylabel('MSE', color=color)
    ax2.plot(Gs, errors, label='Error', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    print('stop')


if __name__ == '__main__':
    run()