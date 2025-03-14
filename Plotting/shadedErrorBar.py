# =====================================================================================
# Methods to plot a few properties FCD matrices
# =====================================================================================
import matplotlib.pyplot as plt
import numpy as np


def shadeErrorBarAx(ax, x, y, errBar, color='red', title=None):
    ax.plot(x, y, color=color)
    ax.fill_between(x, y - errBar, y + errBar, color=color, alpha=0.2)
    if title is not None:
        plt.suptitle(title)


def plotShadedVarianceAx(ax, x, values, color='red', title=None):
    y = np.average(values, axis=1)
    std = np.std(values, axis=1)
    shadeErrorBarAx(ax, x, y, std, color=color, title=title)


def shadedErrorBar(x, y, errBar, color='red', title=None):
    fig, ax = plt.subplots()
    shadeErrorBarAx(ax, x, y, errBar, color=color, title=title)
    plt.show()


def plotShadedVariance(x, values, color='red', title=None):
    fig, ax = plt.subplots()
    plotShadedVarianceAx(ax, x, values, color=color, title=title)
    plt.show()


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================eof