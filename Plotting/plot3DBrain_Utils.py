# =================================================================
# =================================================================
# Utility WholeBrain to compute multi-views of cortex data
# =================================================================
# =================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from pandas.compat.numpy.function import validate_min

from Plotting.plot3DBrain import plotColorView


# set the colormap and centre the colorbar
# from https://chris35wills.github.io/matplotlib_diverging_colorbar/
class MidpointNormalize(Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# =================================================================
# plots the 6-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
#           L-flat,         R-flat
# =================================================================
def multiview6(cortex, data, numRegions, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(3, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 3)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Rh-lateral', cmap=rightCmap, **kwds)
    ax = plt.subplot(3, 2, 4)
    plotColorView(ax, cortex, data, numRegions, 'Rh-medial', cmap=rightCmap, **kwds)

    # ================== flatmaps
    ax = fig.add_subplot(3, 2, 5)  # left hemisphere flat
    plotColorView(ax, cortex, data, numRegions, 'L-flat', cmap=leftCmap, **kwds)
    ax = fig.add_subplot(3, 2, 6)  # right hemisphere flat
    plotColorView(ax, cortex, data, numRegions, 'R-flat', cmap=rightCmap, **kwds)

    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()

# =================================================================
# Functions to finish a 3D plot
# =================================================================
def finish_multiview(fig, cmap, data, display=True, savePath=None, **kwds):
    # ============= Adjust the sizes
    plt.subplots_adjust(left=0.0, right=0.8, bottom=0.0, top=1.0, wspace=0, hspace=0)
    # ============= now, let's add a colorbar...
    if 'norm' not in kwds:
        if 'vmin' in data:
            vmin = data['vmin']
            vmax = data['vmax']
        else:
            vmin = np.min(data['func_L']) if 'vmin' not in kwds else kwds['vmin']
            vmax = np.max(data['func_L']) if 'vmax' not in kwds else kwds['vmax']
        if vmin < 0. < vmax:
            norm = MidpointNormalize(vmin=vmin, midpoint=0., vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = kwds['norm']
    PCM = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])  # This parameter is the dimensions [left, bottom, width, height] of the new axes.
    fig.colorbar(PCM, cax=cbar_ax)
    # ============ and show!!!
    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()

# =================================================================
# Functions to plot a 5-view plot, either "star"-shaped or linear
# =================================================================

# =================================================================
# multiview5_linear:
#     lh-lateral, rh-lateral, l/r-superior, lh-medial, rh-medial
# =================================================================
def multiview5_linear(cortex, data, numRegions, cmap=plt.cm.coolwarm,
                      suptitle='', figsize=(12, 3), display=True, savePath=None, **kwds):
    fig, axs = plt.subplots(1, 5, figsize=figsize)
    plotColorView(axs[0], cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds)
    plotColorView(axs[1], cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds)
    plotColorView(axs[2], cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(axs[2], cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(axs[3], cortex, data, numRegions, 'Rh-medial', cmap=cmap, **kwds)
    plotColorView(axs[4], cortex, data, numRegions, 'Rh-lateral', cmap=cmap, **kwds)
    return fig


# =================================================================
# plots a 5-view star-shaped plot:
#           lh-lateral,               rh-lateral,
#                       l/r-superior,
#           lh-medial,                rh-medial
# =================================================================
def multiview5_star(cortex, data, numRegions, cmap=plt.cm.coolwarm,
                    suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    plotColorView(axs[0,0], cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds)
    plotColorView(axs[1,0], cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds)
    plotColorView(axs[0,2], cortex, data, numRegions, 'Rh-lateral', cmap=cmap, **kwds)
    plotColorView(axs[1,2], cortex, data, numRegions, 'Rh-medial', cmap=cmap, **kwds)
    # === L/R-superior
    gs = axs[0, 1].get_gridspec()
    # remove the underlying axes
    for ax in axs[:,1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:,1])
    plotColorView(axbig, cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(axbig, cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    return fig


def multiview5(cortex, data, numRegions, cmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, linear=False, **kwds):
    if not linear:
        fig = multiview5_star(cortex, data, numRegions, cmap=cmap, suptitle=suptitle, **kwds)
    else:
        fig = multiview5_linear(cortex, data, numRegions, cmap=cmap, suptitle=suptitle, **kwds)
    finish_multiview(fig, cmap, data, display=display, savePath=savePath, **kwds)


# =================================================================
# plots the 4-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
# =================================================================
def multiview4(cortex, data, numRegions, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(2, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 3)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Rh-medial', cmap=rightCmap, **kwds)
    ax = plt.subplot(2, 2, 4)
    plotColorView(ax, cortex, data, numRegions, 'Rh-lateral', cmap=rightCmap, **kwds)

    finish_multiview(fig, rightCmap, data, display=display, savePath=savePath, **kwds)


# =================================================================
# plots a left/Right-view plot:
#                       l-medial, l-lateral
# =================================================================
def leftRightView(cortex, data, numRegions, cmap=plt.cm.coolwarm,
                  suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds)
    ax = plt.subplot(1, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds)

    finish_multiview(fig, cmap, data, display=display, savePath=savePath, **kwds)


# =================================================================
# plots a superior + inferior plot:
# =================================================================
def supInfView(cortex, data, numRegions, cmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(ax, cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    ax = plt.subplot(1, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'L-inferior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(ax, cortex, data, numRegions, 'R-inferior', suptitle=suptitle, cmap=cmap, **kwds)

    finish_multiview(fig, cmap, data, display=display, savePath=savePath, **kwds)


# =================================================================
# plots a top-view plot:
#                       l/r-superior, and all the others!!!
# =================================================================
def plot_ViewAx(ax, cortex, data, numRegions, view,
                cmap=plt.cm.coolwarm,
                suptitle='', **kwds):
    if view == 'superior':  # this is 'L-superior' + 'R-superior'
        plotColorView(ax, cortex, data, numRegions, 'L-superior',
                      suptitle=suptitle, cmap=cmap, **kwds)
        plotColorView(ax, cortex, data, numRegions, 'R-superior',
                      suptitle=suptitle, cmap=cmap, **kwds)
    else:  # this is 'Lh-medial' / 'Lh-lateral' / 'Rh-medial' / 'Rh-lateral' / 'L-flat' / 'R-flat'
        plotColorView(ax, cortex, data, numRegions, view,
                      suptitle=suptitle, cmap=cmap, **kwds)
    if suptitle == '':
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)


def plot_View(cortex, data, numRegions, view, cmap=plt.cm.coolwarm,
              figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    plot_ViewAx(ax, cortex, data, numRegions, view,  **kwds)

    finish_multiview(fig, cmap, data, display=display, savePath=savePath, **kwds)


# =================================================================
# functions to plot multiple mutiview5 plots for different sets,
# but all with a common normalization and a common colorbar
# =================================================================
# This one plots a single multiview5 plot
def plot_multiview5Values(obs, crtx, title, fileName, display, cmap, norm):
    # crtx = setUpGlasser360_cortex()
    # =============== Plot!!! =============================
    data = {'func_L': obs, 'func_R': obs}
    multiview5(crtx, data, 360, cmap, suptitle=title, lightingBias=0.1, mode='flatWire', shadowed=True,
               display=display, savePath=fileName+'.png', norm=norm)


# plots multiple multiview5 plots
def plot_multiview5ValuesForEachChort(burdens, crtx, title, metaName, display, cmap, path):
    vmin = np.min([np.min(burdens[c]) for c in burdens])
    vmax = np.max([np.max(burdens[c]) for c in burdens])
    norm = Normalize(vmin=vmin, vmax=vmax)
    for cohort in burdens:
        fullFileName = path + cohort + metaName
        plot_multiview5Values(burdens[cohort], crtx, title, fullFileName, display, cmap, norm)


# =================================================================
# ================================= module test code
if __name__ == '__main__':
    from matplotlib import cm

    from Plotting.project3DBrain import set_up_Glasser360_cortex
    import DataLoaders.WorkBrainFolder as WBF
    crtx = set_up_Glasser360_cortex(WBF.WorkBrainDataFolder + '_Parcellations/')

    # =============== Plot!!! =============================
    testData = np.arange(0, 360)
    data = {'func_L': testData, 'func_R': testData}
    # testColors = cm.cividis
    testColors = cm.YlOrBr

    multiview5(crtx, data, 360, testColors,
               linear=True,
               lightingBias=0.1, mode='flatWire', shadowed=True)  # flatWire / flat / gouraud

# ======================================================
# ======================================================
# ======================================================EOF
