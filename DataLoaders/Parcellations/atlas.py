# =====================================================================================
# Methods to input Parcellation scheme data
#
# =====================================================================================
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

import nibabel as nib

from DataLoaders.WorkBrainFolder import *
normalization = 2
parcellation_folder = WorkBrainDataFolder + '_Parcellations/'
# parcellation_file = parcellation_folder + f"ParcelsMNI{normalization}mm.mat"

class Atlas:
    def __init__(self, parcellation, N=None, normalization=None, RSN=7):
        # ----- select parcellation
        if parcellation == 'dbs80':
            file = parcellation_folder + 'dbs80/dbs80symm_2mm.nii.gz'
        elif parcellation == 'Glasser360':
            file = parcellation_folder + 'Glasser360/glasser360MNI.nii.gz'
        elif parcellation == 'Schaefer2018':
            file = parcellation_folder + f'Schaefer2018/MNI/Schaefer2018_{N}Parcels_{RSN}Networks_order_FSLMNI152_{normalization}mm.nii.gz'
        else:  # none of the above, generic parcellation
            file = parcellation
        # ----- load parcellation
        self.brain_vol = nib.load(file)
        # ----- extract some useful info!
        # What is the type of this object?
        # print(f'type: {type(self.brain_vol)}')
        # print(self.brain_vol.header)
        self.data = self.brain_vol.get_fdata()
        self.affine = self.brain_vol.affine
        self.max = int(np.max(self.data))
        # plt.imshow(self.data[96], cmap='bone')
        # plt.axis('off')
        # plt.show()
        # print('loaded parcellation')

    def get_data(self):
        return self.data

    def get_mask(self, id):
        return self.data == id

    def voxel_stats(self):
        shape = self.data.shape
        total_size = shape[0] * shape[1] * shape[2]
        sizes = self.size_regions()
        check = np.sum(list(sizes.values()))
        return {'total': total_size,
                'empty': int(sizes[0]),
                'nonempty': int(total_size - sizes[0]),
                'check total size': int(check),
                'region sizes': sizes
                }

    def size_regions(self):
        unique, counts = np.unique(self.data, return_counts=True)
        return dict(zip(unique, counts))

    # compute the subset masked data of a given dataset for a given id
    # based on the code in https://git.fmrib.ox.ac.uk/fsl/avwutils/-/blob/master/fslmeants.cc
    def get_masked_data(self, id, dataset):
        mask = self.data == id
        data = dataset[mask]
        return data

    # compute the masked average of a given dataset for a given id
    # based on the code in https://git.fmrib.ox.ac.uk/fsl/avwutils/-/blob/master/fslmeants.cc
    def get_masked_avg(self, id, dataset):
        masked_data = self.get_masked_data(id, dataset)
        count = masked_data.shape[0]
        acc = np.sum(masked_data, axis=0)
        avg = acc / count
        return avg

    # compute the masked average of a given dataset
    def get_avg(self, dataset):
        avg = np.zeros((self.max, dataset.shape[-1]))
        for id in range(self.max):
            avg[id] = self.get_masked_avg(id+1, dataset)
        return avg

    def populate(self, dataset):
        length = 1 if len(dataset.shape) == 1 else dataset.shape[-1]
        vols = np.zeros(self.data.shape + (length,))
        for id in range(self.max):
            mask = self.data == id+1  # regions go from 1 to max (inclusive), because 0 is air
            vols[mask] = dataset[id]
        return vols

    def snap_coords(self, all_coords, id, coord):
        mask = self.data == id
        m_coords = all_coords[mask]
        distances = np.linalg.norm(m_coords - coord, axis=1)
        min_index = np.argmin(distances)
        return m_coords[min_index]

    # computes the centroids in space (mm)
    # could have used measure from scikit-image, but this way our code is more consistent
    def get_centroids(self, snap=True):
        x, y, z = self.data.shape
        # from https://stackoverflow.com/questions/24436063/numpy-matrix-of-coordinates
        coords = np.stack([x for x in np.ndindex(x, y, z)]).reshape(x,y,z,3)
        # Compute the centroids for each RoI
        centers_of_mass = [self.get_masked_avg(id=i, dataset=coords) for i in range(1,self.max+1)]
        # Snap to the nearest point
        # from https://github.com/alfnie/conn/blob/master/conn_roicenters.m
        if snap:
            centroids = np.array([self.snap_coords(coords, id=i+1, coord=centers_of_mass[i]) for i in range(0,self.max)])
        else:
            centroids = np.array(centers_of_mass)
        # add W=1 coord for homogeneous coords
        centroids = np.append(centroids, np.ones(centroids.shape[0])[..., None], 1)
        # transform from indices to space
        coords_space = self.convert_coordinates_idx2spc(centroids)
        return coords_space[:,0:3]  # (in MNI space, mm units)

    def get_centers_of_gravity(self):
        return self.get_centroids(snap=False)

    def get_label(self, coord):
        idx = self.convert_coordinates_spc2idx(coord)
        raise NotImplementedError

    # ==================================================================
    # Coordinate transformation routines
    # ==================================================================
    # TRANSFORMS SPACE COORDINATES (mm) TO INDEXES IN A VOLUME
    # from https://github.com/alfnie/conn/blob/master/conn_convertcoordinates.m
    def convert_coordinates_spc2idx(self, coord):
        tr = np.dot(coord, pinv(self.affine.T))
        return tr

    # TRANSFORMS INDEXES IN A VOLUME TO SPACE COORDINATES (mm)
    # from https://github.com/alfnie/conn/blob/master/conn_convertcoordinates.m
    def convert_coordinates_idx2spc(self, idxs):
        tr = np.dot(idxs, self.affine.T)
        return tr

    # ==================================================================
    # plotting code
    # ==================================================================
    def plot_region(self, id, full_size=True):
        # select voxels
        region = self.data == id
        # and plot everything
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(region, edgecolor='k')
        if full_size:
            ax.set_xlim3d(0, self.data.shape[0])
            ax.set_ylim3d(0, self.data.shape[1])
            ax.set_zlim3d(0, self.data.shape[2])
        plt.show()

    def plot_brain(self, full_size=True):
        # Plot
        colors = np.zeros(self.data.shape + (3,))
        for id in range(1, self.max+1):
            colors[self.data == id, :] = np.random.rand(3)
        colors = colors / np.max(colors)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.voxels(self.data != 0,
                  facecolors=colors,
                  edgecolors=np.clip(2 * colors - 0.5, 0, 1),  # brighter
                  edgecolor='k')
        if full_size:
            ax.set_xlim3d(0, self.data.shape[0])
            ax.set_ylim3d(0, self.data.shape[1])
            ax.set_zlim3d(0, self.data.shape[2])
        plt.show()

    # plot brain slices, from Working with NIfTI images
    # https://neuraldatascience.io/8-mri/nifti.html
    def plot_slices(self):
        import scipy.ndimage as ndi

        fig_rows = 4
        fig_cols = 4
        n_subplots = fig_rows * fig_cols
        n_slice = self.data.shape[0]
        step_size = n_slice // n_subplots
        plot_range = n_subplots * step_size
        start_stop = int((n_slice - plot_range) / 2)

        fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
        for idx, img in enumerate(range(start_stop, plot_range, step_size)):
            axs.flat[idx].imshow(ndi.rotate(self.data[img, :, :], 90), cmap='bone')
            axs.flat[idx].axis('off')
        plt.tight_layout()
        plt.show()


# ==================================================================
# debug code: transfer data between parcellations
# ==================================================================
if __name__ == '__main__':
    p = Atlas('Schaefer2018', N=100, normalization=2, RSN=7)
    # p = Atlas('dbs80')
    # p.plot_brain(full_size=True)
    # p.plot_slices()
    sizes = p.size_regions()
    for key, value in sizes.items():
        print(f"{int(key)}: {value}", end=', ')
    print(f'\n\nvoxel stats: {p.voxel_stats()}')
    c = p.get_centroids()
    # for pos in range(1, c.shape[0]):
    #     print(f"{int(pos)}: {c[pos]}")

    print(f'Done !!!')

# ======================================================
# ======================================================
# ======================================================EOF