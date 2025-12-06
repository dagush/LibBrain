# --------------------------------------------------------------------------------------
# Full code for loading the ADNI data in the Schaefer2018 parcellation 1000, 400 and 100
# RoIs: 100, 400 or 1000 - TR = 3 - timepoints: 140
# Subjects: ADNI2 - HC 43, EMCI 28, AD 25
#           ADNI3 - HC 156, MCI 64, AD 21
#           IRFSPGR - HC 156, MCI 26, AD 11
#           matching - HC 156, MCI 64, AD 21
# Info for each subject: timeseries
#
# Parcellated by NOELIA MARTINEZ MOLINA
#
# Code by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
from neuronumba.tools import hdf

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
class ADNI_B(DataLoader):
    def __init__(self, path=None,
                 ADNI_version='ADNI2',  # ADNI2 / ADNI3 / IRFSPGR / matching
                 SchaeferSize=1000,  # by default, let's use the Schaefer2018 1000 parcellation
                 ):
        self.SchaeferSize = SchaeferSize
        self.ADNI_version = ADNI_version
        if ADNI_version == 'ADNI2':
            self.groups = ['HC', 'EMCI', 'AD']
        elif ADNI_version in ['ADNI3', 'IRFSPGR', 'matching']:
            self.groups = ['HC','MCI', 'AD']
        else:
            raise Exception('ADNI version not supported')
        if path is not None:
            self.set_basePath(path)
        else:
            self.set_basePath(WorkBrainDataFolder)
        self.timeseries = {}
        self.__loadAllData(SchaeferSize)

    # ---------------- load data
    def __loadSubjectsData(self, fMRI_path):
        print(f'Loading {fMRI_path}')
        fMRIs = hdf.loadmat(fMRI_path)
        return fMRIs['tseries'][:, 0]

    def __loadAllData(self, SchaeferSize, chosenDatasets=None):
        if chosenDatasets is None:
            chosenDatasets = self.groups
        for task in chosenDatasets:
            print(f'----------- Checking: {task} --------------')
            if self.ADNI_version != 'IRFSPGR':
                taskRealName = task if task != 'EMCI' else 'EMCI_v02'
            elif self.ADNI_version == 'matching':
                taskRealName = task + '_matching'
            else:
                taskRealName = task + '_IRFSPGR' if task != 'HC' else task
            fMRI_task_path = self.fMRI_path.format(taskRealName, SchaeferSize)
            self.timeseries[task] = self.__loadSubjectsData(fMRI_task_path)
            print(f'------ done {task}------')

    def name(self):
        return 'ADNI_B'

    def set_basePath(self, path):
        base_folder = path + "ADNI-B/"
        self.ID_path = None
        if self.ADNI_version == 'ADNI2':
            self.fMRI_path = base_folder + 'ADNI2/tseries_ADNI2_{}_batch1_sch{}.mat'
        elif self.ADNI_version in ['ADNI3', 'IRFSPGR', 'matching']:
            self.fMRI_path = base_folder + 'ADNI3/tseries_ADNI3_{}_sch{}_QC.mat'

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return self.SchaeferSize

    # get_fullGroup_data: convenience method to load all data for a given subject group
    def get_fullGroup_data(self, group):
        group_fMRI = {(s, group): {'timeseries': self.timeseries[group][s]} for s in range(len(self.timeseries[group]))}
        return group_fMRI

    # def _correctSC(self, SC):
    #     return SC/np.max(SC)

    def get_AvgSC_ctrl(self, normalized=None):
        # SC = hdf.loadmat(base_folder + 'sc_schaefer_MK.mat')['sc_schaefer']
        # if normalized:
        #     return self._correctSC(SC)
        # else:
        #     return SC
        raise NotImplemented('We do not have the SC!')

    def get_groupSubjects(self, group):
        classi = {}
        numsubj = len(self.timeseries[group])
        for subj in range(numsubj):
            classi[(subj, group)] = group
        return list(classi)

    def get_groupLabels(self):
        return self.groups

    def get_classification(self):
        classi = {}
        for group in self.groups:
            numsubj = len(self.timeseries[group])
            for subj in range(numsubj):
                classi[(subj, group)] = group
        return classi

    def discardSubject(self, subjectID):
        self.timeseries[subjectID[1]] = np.delete(self.timeseries[subjectID[1]], subjectID[0])

    def get_subjectData(self, subjectID):
        ts = self.timeseries[subjectID[1]][subjectID[0]]
        return {subjectID: {'timeseries': ts}}

    def get_parcellation(self):
        return Schaefer2018.Schaefer2018(N=self.SchaeferSize, normalization=2, RSN=7)  # use normalization of 2mm, 7 RSNs


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = ADNI_B(ADNI_version='matching', SchaeferSize=400)
    sujes = DL.get_classification()
    gCtrl = DL.get_groupSubjects('HC')
    s1 = DL.get_subjectData((0,'HC'))
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF