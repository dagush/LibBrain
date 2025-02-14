# --------------------------------------------------------------------------------------
# Full code for loading the ADNI data in the Schaefer2018 parcellation 400
# RoIs: 400 - TR = 3 - timepoints: 197
# Subjects: N238rev - HC 109, MCI 90, AD 39
# Info for each subject: timeseries
#
# Parcellated by NOELIA MARTINEZ MOLINA
#
# Code by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
import tools.hdf as hdf
from DataLoaders.baseDataLoader import DataLoader
import DataLoaders.Parcellations.Schaefer2018 as Schaefer2018


# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class ADNI_B_N238rev(DataLoader):
    def __init__(self, path=None,
                 # ADNI_version='N238rev',  # N238rev
                 # SchaeferSize=400,  # by default, let's use the Schaefer2018 400 parcellation
                 ):
        # self.SchaeferSize = SchaeferSize
        # self.ADNI_version = ADNI_version
        self.groups = ['HC','MCI', 'AD']
        if path is not None:
            self.set_basePath(path)
        else:
            self.set_basePath(WorkBrainDataFolder)
        self.timeseries = {}
        self.__loadAllData()  # SchaeferSize)

    # ---------------- load data
    def __loadSubjectsData(self, ID_path, fMRI_path):
        print(f'Loading {fMRI_path}')
        PTIDs = hdf.loadmat(ID_path)
        PTIDs = PTIDs['PTID']
        IDs = [id[0] for id in np.squeeze(PTIDs).tolist()]
        fMRIs = hdf.loadmat(fMRI_path)
        fMRIs = fMRIs['tseries'][:, 0]
        res = {IDs[i].tolist(): fMRIs[i] for i in range(len(IDs))}
        return res

    def __loadAllData(self, # SchaeferSize,
                      chosenDatasets=None):
        if chosenDatasets is None:
            chosenDatasets = self.groups
        for task in chosenDatasets:
            print(f'----------- Checking: {task} --------------')
            taskRealName = task
            ID_path = self.ID_path.format(taskRealName)
            fMRI_task_path = self.fMRI_path.format(taskRealName)
            self.timeseries[task] = self.__loadSubjectsData(ID_path, fMRI_task_path)
            print(f'----------- done {task} --------------')

    def name(self):
        return 'ADNI_B_N238rev'

    def set_basePath(self, path):
        base_folder = path + "ADNI-B/N238rev/tseries/sch400/"
        self.fMRI_path = base_folder + 'tseries_ADNI3_{}_MPRAGE_IRFSPGR_sch400_N238rev.mat'
        self.ID_path = base_folder + 'PTID_ADNI3_{}_MPRAGE_IRFSPGR_all.mat'

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return 400  # self.SchaeferSize

    def get_AvgSC_ctrl(self, normalized=None):
        # SC = hdf.loadmat(base_folder + 'sc_schaefer_MK.mat')['sc_schaefer']
        # if normalized:
        #     return self._correctSC(SC)
        # else:
        #     return SC
        raise NotImplemented('We do not have the SC!')

    def get_groupLabels(self):
        return self.groups

    def get_classification(self):
        classi = {}
        for group in self.groups:
            ts_group = self.timeseries[group]
            for subj in ts_group:
                classi[subj] = group
        return classi

    def discardSubject(self, subjectID):
        self.timeseries[subjectID[1]] = np.delete(self.timeseries[subjectID[1]], subjectID[0])

    def get_subjectData(self, subjectID):
        group = self.get_classification()[subjectID]
        ts = self.timeseries[group][subjectID]
        return {subjectID: {'timeseries': ts}}

    def get_parcellation(self):
        return Schaefer2018.Schaefer2018(N=400, normalization=2, RSN=7)  # use normalization of 2mm, 7 RSNs


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = ADNI_B_N238rev()
    sujes = DL.get_classification()
    gCtrl = DL.get_groupSubjects('HC')
    s1 = DL.get_subjectData('002_S_6007')
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF