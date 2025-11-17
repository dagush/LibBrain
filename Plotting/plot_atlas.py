# ==================================================================
# plotting code for atlases
# ==================================================================
import numpy as np
import matplotlib.pyplot as plt
import Plotting.plot2DSliced_Brain as plot2D


def plot_3D_region(atlas, id, full_size=True):
    # select voxels
    region = atlas == id
    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(region, edgecolor='k')
    if full_size:
        ax.set_xlim3d(0, atlas.shape[0])
        ax.set_ylim3d(0, atlas.shape[1])
        ax.set_zlim3d(0, atlas.shape[2])
    plt.show()


def plot_3D(atlas, full_size=True):
    max_val = int(np.max(atlas))
    # Plot
    colors = np.zeros(atlas.shape + (3,))
    for id in range(1, max_val +1):
        colors[atlas == id, :] = np.random.rand(3)
    colors = colors / np.max(colors)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.voxels(atlas != 0,
              facecolors=colors,
              edgecolors=np.clip(2 * colors - 0.5, 0, 1),  # brighter
              edgecolor='k')
    if full_size:
        ax.set_xlim3d(0, atlas.shape[0])
        ax.set_ylim3d(0, atlas.shape[1])
        ax.set_zlim3d(0, atlas.shape[2])
    plt.show()


def plot_slices(atlas, rows=4, cols=4):
    plot2D.plot_slices(atlas, rows=rows, cols=cols)
