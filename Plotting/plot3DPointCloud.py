# from sympy.physics.control.control_plots import matplotlib

import matplotlib.pyplot as plt


def plot_point_cloud(parcellationData, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for r in parcellationData:
        ax.scatter(r[0], r[1], r[2], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)
    plt.show()


def plot_2x3D_brains(c_coords, p_coords):
    def on_move(event):
        ax = axs[0];
        ax2 = axs[1]
        if event.inaxes == ax:
            if ax.button_pressed in ax._rotate_btn:
                ax2.view_init(elev=ax.elev, azim=ax.azim)
            elif ax.button_pressed in ax._zoom_btn:
                ax2.set_xlim3d(ax.get_xlim3d())
                ax2.set_ylim3d(ax.get_ylim3d())
                ax2.set_zlim3d(ax.get_zlim3d())
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                ax.view_init(elev=ax2.elev, azim=ax2.azim)
            elif ax2.button_pressed in ax2._zoom_btn:
                ax.set_xlim3d(ax2.get_xlim3d())
                ax.set_ylim3d(ax2.get_ylim3d())
                ax.set_zlim3d(ax2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()

    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    axs[0].scatter(c_coords[:, 0], c_coords[:, 1], c_coords[:, 2])
    axs[0].title.set_text('Budapest Reference Connectome')
    axs[1].scatter(p_coords[:, 0], p_coords[:, 1], p_coords[:, 2])
    axs[1].title.set_text('Glasser 360')
    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()