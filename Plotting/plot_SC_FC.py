# =====================================================================================
# Methods to plot a few properties SC/FC matrices
# =====================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_histogram(ax, matr, subjectName):
    # plt.rcParams["figure.figsize"] = (7,5)
    # plt.rcParams["figure.dpi"] = 300
    # plt.figure()  #num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    bins = 50 #'auto'
    n, bins, patches = ax.hist(matr.flatten(), bins=bins, color='#0504aa', alpha=0.7, histtype='step')  #, rwidth=0.85)
    ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel('SC weights')
    ax.set_ylabel('Counts')
    ax.set_title("SC histogram ({}: {})".format(subjectName, matr.shape), fontweight="bold", fontsize="18")
    # plt.savefig("./_Results/Abeta/"+subject+".png", dpi=200)
    # plt.close()


def plot_matr(ax, matr, title, labelAxis='both', colormap='viridis', fontSize=24):
    ax.imshow(np.asarray(matr), cmap=colormap)
    if labelAxis == 'both' or labelAxis == 'xlabels':
        ax.set_xlabel("Regions")
    if labelAxis == 'both' or labelAxis == 'ylabels':
            ax.set_ylabel("Regions")
    ax.set_title(title, fontsize=fontSize)
    # ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
    print(f"Scale({title}): Max={np.max(matr)}, Min={np.min(matr)}")


def plot_matr_and_Histogram(subjectName, SCnorm, plotColorBar = True):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 2)
    ax1 = fig.add_subplot(grid[0,0])
    plot_matr(ax1, SCnorm, subjectName)
    if plotColorBar:
        img = ax1.get_images()[0]
        fig.colorbar(img)
    ax2 = fig.add_subplot(grid[0,1])
    plot_histogram(ax2, SCnorm, subjectName)
    plt.suptitle("Structural Connectivity ({})".format(subjectName), fontweight="bold", fontsize="18", y=1.05)
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
    # img = ax1.get_images()[0]
    # fig.colorbar(img, cax=cbar_ax)
    plt.show()


def just_plot_matrix(subjectName, matr, plottingFunction, colormap='viridis'):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0,0])
    plottingFunction(ax, matr, subjectName, colormap=colormap)
    plt.show()


# =====================================================================================================================
# get indices of n maximum values in a numpy array
# Code taken from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
# =====================================================================================================================
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# =====================================================================================================================
# Plot SC as a graph
# =====================================================================================================================
def plot_matrix_as_fancy_graph(M):
    plt.rcParams.update({'font.size': 25})
    import networkx as nx

    # Using a figure to use it as a parameter when calling nx.draw_networkx
    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)

    # Keep only the 5% of the largest values
    M2 = largest_indices(M, int(M.size * .05))
    print(f"selected {M2[0].size} nodes")
    compl = M==M
    compl[M2] = False
    M[compl] = 0.0

    color_map = ['red']*180 + ['blue']*180 + ['green']*9 + ['orange']*9 + ['black']
    legend_elements = [Line2D([0], [0], marker='o', color='red', label='Right Cortex', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='blue', label='Left Cortex', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='green', label='Left Subcortical', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='orange', label='Right Subcortical', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='black', label='Brainstem', linewidth=0, markersize=15)]

    # Now, plot it!!!
    G = nx.from_numpy_array(M)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"resulting edges => {G.number_of_edges()}")
    d = nx.degree(G)
    d = [(d[node]+1) * 20 for node in G.nodes()]
    pos=nx.fruchterman_reingold_layout(G, k=1/np.sqrt(M2[0].size))
    nx.draw(G, with_labels=False, node_size=d, node_color=color_map, pos=pos, ax=ax)
    plt.title("Connectivity Graph")

    plt.legend(handles=legend_elements)
    plt.show()


def plot_fancy_matrix_ax(ax, M,
                      axisName="Regions",
                      matrixName="Connectivity Matrix",
                      showAxis='on', cmap='viridis'):
    im = ax.matshow(M, cmap=cmap)
    ax.set_title(matrixName)
    ax.set_xlabel(axisName)
    ax.set_ylabel(axisName)
    ax.axis(showAxis)
    return im


def plot_fancy_matrix(M,
                      axisName="Regions",
                      matrixName="Connectivity Matrix",
                      showAxis='on', fontSize=25, cmap='viridis'):
    plt.rcParams.update({'font.size': fontSize})
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = plot_fancy_matrix_ax(ax, M,
                         axisName=axisName, matrixName=matrixName,
                         showAxis=showAxis, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

# =====================================================================================================
# =====================================================================================================
# =====================================================================================================eof
