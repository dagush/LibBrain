# =================================================================================
# =================================================================================
# Utility WholeBrain to compute views of the cortex data across DIFFERENT groups
# =================================================================================
# =================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from Plotting.plot3DBrain_Utils import plot_ViewAx


# =================================================================
# plots rows of views for multiple different vectors (groups)...
# Now, only one view. All plots have the same limits and use the
# same colorbar
#
# Here, burdens is a list of rows, each rows is a dict of the name
# of the burden and the N values. Views is a list as long as
# burdens, specifying the different views to render
# =================================================================
def plot_ValuesForAllGroups_multiline(burdens, views, crtx, cmap, orientation='h'):
    if orientation == 'h':
        sizeX, sizeY = len(burdens), len(burdens[0])+1
    else:
        sizeX, sizeY = len(burdens[0]), len(burdens)+1
    fig, axs = plt.subplots(sizeX, sizeY,
                            # we add an extra row to solve a strange bug I found, where the last panel does not show the ticks, and do not have the patience to fix
                            gridspec_kw={'wspace': 0.2, 'hspace': 0.2},
                            figsize=(sizeY+1, sizeX+1))
    axs = np.atleast_2d(axs)
    num_regions = len(burdens[0][list(burdens[0].keys())[0]])
    vmin = np.min([np.min(row[c]) for row in burdens for c in row])
    vmax = np.max([np.max(row[c]) for row in burdens for c in row])
    for r, (view, row) in enumerate(zip(views, burdens)):
        for c, cohort in enumerate(row):
            vect = row[cohort]
            data = {'func_L': vect, 'func_R': vect}
            plot_ViewAx(axs[r,c], crtx, data, num_regions, view,
                        vmin=vmin, vmax=vmax,
                        cmap=cmap, suptitle=str(cohort) if r==0 else '', fontSize=15)
    for r in range(sizeX): fig.delaxes(axs[r,sizeY-1])
    norm = Normalize(vmin=vmin, vmax=vmax)
    PCM = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])  # This parameter is the dimensions [left, bottom, width, height] of the new axes.
    fig.colorbar(PCM, cax=cbar_ax)
    fig.tight_layout()
    plt.show()


# =================================================================
# plots views for multiple different vectors (groups)... Now, only one view.
# All plots have the same limits and use the same colorbar
# =================================================================
def plot_ValuesForAllGroups(burdens, view, crtx, cmap, orientation='h'):
    plot_ValuesForAllGroups_multiline([burdens], [view], crtx, cmap, orientation)


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF