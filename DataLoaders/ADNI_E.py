# ================================================================================================================
# ================================================================================================================
# Full code for loading the ADNI_E dataset
# This dataset is in raw voxel space
# TR = 3 - timepoints: 197
# Subjects: HC 187, MCI 107, AD 38
#
# Dataset graciously provided by Frithjof Kruggel
#
# Code by Gustavo Patow
# ================================================================================================================
# ================================================================================================================
import os.path
import pandas as pd
import numpy as np
import nibabel as nib

# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *
from DataLoaders.baseDataLoader import DataLoader
from DataLoaders.Parcellations.aal import aal
from DataLoaders.Parcellations.atlas import Atlas


class ADNI_E(DataLoader):
    def __init__(self, path=None,
                 full_dataset=False,
                 ):
        self.groups = ['all']
        if path is not None:
            self.set_basePath(path)
        else:
            if not full_dataset:
                self.set_basePath(WorkBrainDataFolder)
            else:
                self.set_basePath('D:/Neuro/')
        self.__load_general_data()

    def __load_general_data(self):
        self.data = pd.read_csv(self.base_folder + 'demo.csv')

    def name(self):
        return 'ADNI_E (Voxels)'

    def set_basePath(self, path):
        self.base_folder = path + "ADNI-E/"
        excite_base_folder = self.base_folder + "excite2/"
        self.fMRI_path = excite_base_folder + '{}/{}_rs_reg.nii.gz'
        self.avg_path = excite_base_folder + '{}/{}_rs_avg.nii.gz'
        self.t1_path = excite_base_folder + '{}/{}_t1_mni.nii.gz'
        self.aal_path = excite_base_folder + '{}/{}_rs_aal.nii.gz'

    def TR(self):  # Returns a float with the TR of the dataset
        return 3  # Repetition Time (seconds)

    def N(self):  # returns an integer with the number of RoIs in the parcellation
        raise Exception('We are using VOXELS here!')

    def get_groupLabels(self):  # Returns a list with all group labels
        return ['CTRL', 'MCI', 'AD']  # use my own sorting

    def get_classification(self):  # Returns a dict with {subjID: groupLabel}
        res = {s: d for s, d in zip(self.data['ID'], self.data['DIAG'])}
        return res

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
        if os.path.isfile(file):
            brain_vols = nib.load(file)
            # shape = brain_vols.header.get_data_shape()
            # dtype = brain_vols.header.get_data_dtype()
            fMRI = brain_vols.get_fdata()

            file = self.avg_path.format(subjectID, subjectID)
            brain_avg = nib.load(file)
            avg = brain_avg.get_fdata()

            file = self.avg_path.format(subjectID, subjectID)  # in this case they are the same as avg!
            brain_t1 = nib.load(file)
            t1 = brain_avg.get_fdata()

            file = self.aal_path.format(subjectID,subjectID)
            aal_atlas = Atlas(file)

            if  not (brain_vols.affine == brain_avg.affine).any() or \
                not (brain_avg.affine == brain_t1.affine).any() or \
                not (brain_t1.affine == brain_vols.affine).any():  # OK,  I know, no need for the 3 checks...
                raise Exception('Affine matrices are different!')
            affine = brain_t1.affine
        else:  # at least it will not burn when doing local tests...
            fMRI = None
            avg = None
            t1 = None
            aal_atlas = None
            affine = None

        # -------- done, now create the return subject data
        print(f'Loaded {subjectID}')
        s_metadata = self.data[self.data['ID'] == subjectID]
        s_data = {
            'timeseries': fMRI,
            'avg': avg,
            't1': t1,
            'affine': affine,
            'atlas': aal_atlas,
            'meta': {'site': subjectID[:3],
                     'diag': s_metadata['DIAG'].iloc[0],
                     'sex': s_metadata['SEX'].iloc[0],
                     'age': s_metadata['AGE'].iloc[0],
                     },
        }
        return s_data

    def get_parcellation(self):
        return aal(version=1)

    # -------------------------- Convenience methods -----------------------------------
    # get_fullGroup_data: convenience method to load all data for a given group
    def get_fullGroup_data(self, group):
        raise Exception('Do not try to load full group data (too much!)')


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
    s2 = DL.get_subjectData('003_S_1122')
    s3 = DL.get_subjectData('018_S_2133')

    print(check_first_volumes(s1))
    print(check_first_volumes(s2))

    # import Utils.Plotting.plot2DSliced_Brain as plot
    # plot.plotBrain(s1['t1'], title='002_S_0413', cmap='gray')
    # plot.plotBrain(s1['timeseries'], title='002_S_0413 (0)', frame=0, cmap='gray')
    # plot.plotBrain(s1['timeseries'], title='002_S_0413 (100)', frame=100, cmap='gray')
    # plot.plot_timeseries(s1['timeseries'], title='002_S_0413 (100)', slice=22, cmap='gray')


    # avgSC = DL.get_AvgSC_ctrl()
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF