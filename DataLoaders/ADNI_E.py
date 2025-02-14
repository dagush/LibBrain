# ================================================================================================================
# ================================================================================================================
# Full code for loading the ADNI_E dataset
# This dataset is in raw voxel space
#
# Dataset graciously provided by Frithjof Kruggel
#
# Code by Gustavo Patow
# ================================================================================================================
# ================================================================================================================
import pandas as pd
import numpy as np
import nibabel as nib

# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *
from DataLoaders.baseDataLoader import DataLoader


class ADNI_E(DataLoader):
    def __init__(self, path=None,
                 ):
        self.groups = ['all']
        if path is not None:
            self.set_basePath(path)
        else:
            self.set_basePath(WorkBrainDataFolder)
        self.__load_general_data()

    def __load_general_data(self):
        self.data = pd.read_csv(self.base_folder + 'demo.csv')

    def name(self):
        return 'ADNI_E (Voxels)'

    def set_basePath(self, path):
        self.base_folder = path + "ADNI-E/"
        self.fMRI_path = self.base_folder + 'excite/{}/{}_rs_reg.nii.gz'
        self.avg_path = self.base_folder + 'excite/{}/{}_rs_avg.nii.gz'
        self.t1_path = self.base_folder + 'excite/{}/{}_t1_mni.nii.gz'

    def TR(self):  # Returns a float with the TR of the dataset
        return 3  # Repetition Time (seconds)

    def N(self):  # returns an integer with the number of RoIs in the parcellation
        raise Exception('We are using VOXELS here!')

    def get_classification(self):  # Returns a dict with {subjID: groupLabel}
        return {s: 'all' for s in self.data['ID']}

    def get_subjectData(self, subjectID):
        """
        Returns the data corresponding to the given subject ID.
        :param subjectID:
        :return: a dict of
        {subjectID:
            {'timeseries': timeseries,  # N x T
             'SC': SCnorm,  # N x N
             # other information
             }}
        """
        file = self.fMRI_path.format(subjectID,subjectID)
        brain_vols = nib.load(file)
        # shape = brain_vols.header.get_data_shape()
        # dtype = brain_vols.header.get_data_dtype()
        fMRI = brain_vols.get_fdata()

        file = self.avg_path.format(subjectID,subjectID)
        brain_avg = nib.load(file)
        avg = brain_avg.get_fdata()

        file = self.avg_path.format(subjectID,subjectID)
        brain_t1 = nib.load(file)
        t1 = brain_avg.get_fdata()

        if not (brain_vols.affine == brain_avg.affine).any() or \
           not (brain_avg.affine == brain_t1.affine).any() or \
           not (brain_t1.affine == brain_vols.affine).any():  # OK,  I know, no need for the 3 checks...
           raise Exception('Affine matrices are different!')

        print(f'Loaded {subjectID}')
        return {
            'timeseries': fMRI,
            'avg': avg,
            't1': t1,
            'affine': brain_t1.affine,
        }

    def get_parcellation(self):
        return NotImplementedError('This should have been implemented by a subclass')

    # -------------------------- Convenience methods -----------------------------------
    # get_fullGroup_data: convenience method to load all data for a given group
    def get_fullGroup_data(self, group):
        raise Exception('Do not try to load full group data')


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    def check_first_volumes(subj):
        test = {}
        for n in range(5):
            test[n] = not np.any(subj['timeseries'][:, :, :, n] - subj['avg'])
        return test

    DL = ADNI_E()
    sujes = DL.get_classification()
    print(f'Classification: {sujes}')
    print(f'Group labels: {DL.get_groupLabels()}')
    gMCI = DL.get_groupSubjects('all')
    s1 = DL.get_subjectData('002_S_0413')
    s2 = DL.get_subjectData('002_S_1155')

    print(check_first_volumes(s1))
    print(check_first_volumes(s2))

    import Utils.Plotting.plot2DSliced_Brain as plot
    # plot.plotBrain(s1['t1'], title='002_S_0413', cmap='gray')
    # plot.plotBrain(s1['timeseries'], title='002_S_0413 (0)', frame=0, cmap='gray')
    # plot.plotBrain(s1['timeseries'], title='002_S_0413 (100)', frame=100, cmap='gray')
    plot.plot_timeseries(s1['timeseries'], title='002_S_0413 (100)', slice=22, cmap='gray')


    # avgSC = DL.get_AvgSC_ctrl()
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF