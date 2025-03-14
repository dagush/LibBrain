# =================================================================
#  Plotting sliced brain WholeBrain with
#    NiBabel for the gifti files, and
#    matplotlib for plotting
# =================================================================
# =================================================================
import matplotlib.pyplot as plt
from matplotlib import cm


def plotColorView(ax, brain_img, slice_number, frame=None, cmap=plt.cm.coolwarm):
    # print(brain_img.shape)
    if (len(brain_img.shape) == 3):
        ax.imshow(brain_img[:, :, slice_number], cmap=cmap)
        title = f'slice {slice_number}'
    if (len(brain_img.shape) == 4):
        ax.imshow(brain_img[:, :, slice_number, frame], cmap=cmap)
        title = f'slice {slice_number} ({frame})'
    ax.set_xlabel(title)


def plotBrain(brain_img, title='', rows = 3, cols = 4, frame=None, cmap=plt.cm.coolwarm):
    num = brain_img.shape[2]
    fig, axs = plt.subplots(rows,cols,sharex=True, sharey=True)
    for x in range(rows):
        for y in range(cols):
            pos = int(num * (x*cols + y) / (cols*rows))
            plotColorView(axs[x,y], brain_img, pos, frame=frame, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def plot_timeseries(brain_img, slice, title='', rows = 3, cols = 4, cmap=plt.cm.coolwarm):
    num = brain_img.shape[3]
    fig, axs = plt.subplots(rows,cols,sharex=True, sharey=True)
    for x in range(rows):
        for y in range(cols):
            frame = int(num * (x*cols + y) / (cols*rows))
            plotColorView(axs[x,y], brain_img, slice, frame=frame, cmap=cmap)
    plt.suptitle(title)
    plt.show()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF