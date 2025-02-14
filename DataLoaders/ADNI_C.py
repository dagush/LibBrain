# =====================================================================================
# Methods to input AD data
# Subjects: CN 3, MCI 2 - in 2 longitudinal sessions each
# RoIs: 84 - TR = 3 - timepoints: 193
# Info for each subject: timeseries, SC
#
# The ADNI-3 dataset was graciously provided by Riccardo Leone and Xenia Kobeleva
#
# =====================================================================================
from sys import platform
import numpy as np
import pandas as pd
import os, csv

from DataLoaders.baseDataLoader import DataLoader
import DataLoaders.Parcellations.Glasser379 as Glasser379

# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *
base_folder = WorkBrainDataFolder + "ADNI-C/"
derivatives_folder = base_folder + "bids/derivatives/"
fMRI_folder = derivatives_folder + 'xcp_d/'
SC_folder = derivatives_folder + 'qsirecon/'

subj_info_file = base_folder + 'utils/df_adni_tau_amy_mri_longitud.csv'
# ==========================================================================
# ==========================================================================
# ==========================================================================


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class ADNI_C(DataLoader):
    def __loadAllData(self):
        # ----------- load general info
        self.info = pd.read_csv(subj_info_file, sep=',')
        classif = self.info[['PTID', 'TADAD_SES', 'DX']]
        classif = classif.reset_index()
        self.classification = {}
        for index, row in classif.iterrows():
            self.classification[(row['PTID'], 'ses-' + str(row['TADAD_SES']).zfill(2))] = row['DX']
        patients = list(dict.fromkeys(classif['PTID']))
        print(f"loaded: {patients}")

    def __init__(self, path=None):
        if path is not None:
            self.set_basePath(self, path)
        self.__loadAllData()

    def name(self):
        return 'ADNI_C'

    def set_basePath(self, path):
        global base_folder
        base_folder = path

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return 84

    def get_classification(self):
        return self.classification

    def get_subjectData(self, subjectID):
        # ----------- load fMRI data
        patient_fMRI_folder = fMRI_folder + f'sub-{subjectID[0]}/{subjectID[1]}/func/sub-{subjectID[0]}_{subjectID[1]}_task-rest_run-1_space-MNI152NLin6Asym_res-01_desc-timeseries_desikan_killiany.tsv'
        timeseries = pd.read_csv(patient_fMRI_folder, sep='\t')
        print(f"loaded: {patient_fMRI_folder}")
        # ----------- load SC
        patient_SC_folder = SC_folder + f'sub-{subjectID[0]}/{subjectID[1]}/dwi/sub-{subjectID[0]}_desikan_sift_invnodevol_radius2_count_connectivity.csv'
        SC = pd.read_csv(patient_SC_folder, sep=',')
        SC = SC.drop('Unnamed: 0', axis=1)
        return {subjectID: {'timeseries': timeseries.to_numpy().T,
                            'SC': SC.to_numpy()}}

    def get_parcellation(self):
        raise NotImplemented('We do not have this data yet!')


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = ADNI_C()
    sujes = DL.get_classification()
    print(f'Classification: {sujes}')
    print(f'Group labels: {DL.get_groupLabels()}')
    gMCI = DL.get_groupSubjects('AD')
    s1 = DL.get_subjectData(('ADNI003S6067', 'ses-01'))
    avgSC = DL.get_AvgSC_ctrl(ctrl_label='CN')
    print('done! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF