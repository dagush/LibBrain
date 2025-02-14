# --------------------------------------------------------------------------------------
# Full code for loading the HCB data in the Schaefer2018 parcellation 1000 and 100
# Subjects: HC 122, MCI 72 - RoIs: 52 - TR = 3 - timepoints: 197
# Info for each subject: timeseries
#
# Parcellated by Xenia Kobeleva
#
# Code by Gustavo Patow
# Note: We have 72 subjects and 52 areas
# --------------------------------------------------------------------------------------
import numpy as np
from neuronumba.tools import hdf

from DataLoaders.baseDataLoader import DataLoader
# import DataLoaders.Parcellations.Schaefer2018 as Schaefer2018


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
class ADNI_D(DataLoader):
    def __init__(self, path=None,
                 ):
        self.groups = ['hc', 'mci']
        if path is not None:
            self.set_basePath(path)
        else:
            self.set_basePath(WorkBrainDataFolder)
        self.timeseries = {}
        self.__loadAllData()

    # ---------------- load data
    def __load_fMRI_task_data(self, fMRI_path):
        print(f'Loading {fMRI_path}')
        fMRIs = hdf.loadmat(fMRI_path)['ts_emp']
        res = {s: fMRIs[s] for s in range(fMRIs.shape[0])}
        return res

    def __loadAllData(self):
        for group in self.groups:
            print(f'----------- Checking: {group} --------------')
            fMRI_task_path = self.fMRI_path.format(group)
            self.timeseries[group] = self.__load_fMRI_task_data(fMRI_task_path)
            print(f'------ done {group}------')
        self.taus = hdf.loadmat(self.tau_path)['tau']

    def name(self):
        return 'ADNI_D'

    def set_basePath(self, path):
        base_folder = path + "ADNI-D/"
        self.fMRI_path = base_folder + 'ts_{}.mat'
        self.SC_path = base_folder + 'sc_new.mat'
        self.tau_path = base_folder + 'tau_mci.mat'

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return 52

    def _correctSC(self, SC):
        return SC/np.max(SC)

    def get_AvgSC_ctrl(self, normalized=None):
        SC = hdf.loadmat(self.SC_path)['sc_new']
        if normalized:
            return self._correctSC(SC)
        else:
            return SC

    def get_classification(self):
        classi = {}
        for group in self.groups:
            numsubj = len(self.timeseries[group])
            for subj in range(numsubj):
                classi[(subj, group)] = group
        return classi

    def get_subjectData(self, subjectID):
        ts = self.timeseries[subjectID[1]][subjectID[0]]
        res = {subjectID: {'timeseries': ts}}
        if subjectID[1] == 'mci':  # we only have tau for mci
            res[subjectID]['tau'] = np.concatenate(self.taus[subjectID[0]]).ravel()
        return res

# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = ADNI_D()
    sujes = DL.get_classification()
    print(f'Classification: {sujes}')
    print(f'Group labels: {DL.get_groupLabels()}')
    gMCI = DL.get_groupSubjects('mci')
    s1 = DL.get_subjectData((1, 'mci'))
    s2 = DL.get_subjectData((1, 'hc'))
    avgSC = DL.get_AvgSC_ctrl()
    print('done! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF