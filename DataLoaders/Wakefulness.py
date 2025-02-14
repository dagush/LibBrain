# --------------------------------------------------------------------------------------
# Full pipeline for loading Wakefulness data
# Subjects: N3 15, W 15 - RoIs: 90 - TR = 2 - timepoints: 224
#
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
import os
import csv
import random
import numpy as np
import scipy.io as sio
import h5py

from DataLoaders.baseDataLoader import DataLoader


# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *
base_folder = WorkBrainDataFolder + "Wakefulness/"
# ==========================================================================
# ==========================================================================
# ==========================================================================


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class Wakefulness(DataLoader):
    def __init__(self, path=None,
                 cutTimeSeries=True
                 ):
        if path is not None:
            self.set_basePath(self, path)
        data = sio.loadmat(base_folder + 'DataSleepW_N3.mat')
        self.SC = data['SC']
        self.Num = self.SC.shape[0]  # 90
        self.NumSubj = data['TS_N3'].size  # 15
        self.TS = {}
        self.TS['N3'] = {}
        self.TS['W'] = {}
        for s in range(self.NumSubj):
            self.TS['N3'][(s, 'N3')] = np.squeeze(data['TS_N3'])[s]
            self.TS['W'][(s, 'W')] = np.squeeze(data['TS_W'])[s]
        minT_N3 = np.min([self.TS['N3'][(s, 'N3')].shape[1] for s in range(self.NumSubj)])
        minT_W = np.min([self.TS['W'][(s, 'W')].shape[1] for s in range(self.NumSubj)])
        if cutTimeSeries:
            for s in range(self.NumSubj):
                self.TS['N3'][(s, 'N3')] = self.TS['N3'][(s, 'N3')][:,:minT_N3]
                self.TS['W'][(s, 'W')] = self.TS['W'][(s, 'W')][:,:minT_W]
        print(f'loaded, {self.NumSubj} subjects, N={self.N}, minimum length: N3={minT_N3} W={minT_W}')

    def name(self):
        return 'Wakefulness'

    def set_basePath(self, path):
        global WholeBrainFolder, base_folder
        # WholeBrainFolder = path
        base_folder = path

    def TR(self):
        return 2  # Repetition Time (seconds)

    def N(self):
        return self.Num  # 90

    # get_fullGroup_fMRI: convenience method to load all fMRIs for a given subject group
    def get_fullGroup_fMRI(self, group):
        return self.TS[group]

    def get_AvgSC_ctrl(self, **kwargs):
        normSC = self._normalize_SC(self.SC, kwargs)
        return normSC

    def get_groupSubjects(self, group):
        test = self.TS[group].keys()
        return list(test)

    def get_groupLabels(self):
        return ['N3', 'W']

    def get_classification(self):
        classi = {}
        for task in self.get_groupLabels():
            test = self.TS[task].keys()
            for subj in test:
                classi[subj] = subj[0]
        return classi

    def discardSubject(self, subjectID):
        self.TS[subjectID[0]].pop(subjectID)

    def get_subjectData(self, subjectID):
        ts = self.TS[subjectID[1]][subjectID]
        return {subjectID: {'timeseries': ts}}

    # def get_GlobalData(self):
    #     cog = sio.loadmat(base_folder + 'schaefercog.mat')['SchaeferCOG']
    #     return {'coords': cog} | super().get_GlobalData()


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = Wakefulness()
    sujes = DL.get_classification()
    gCtrl = DL.get_groupSubjects('N3')
    s1 = DL.get_subjectData((0, 'N3'))
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF