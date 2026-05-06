# --------------------------------------------------------------------------------------
# Full code for loading the ADNI data in the BDS80 parcellation
# RoIs: 80 - TR = 3 - timepoints: 197 (but some have more)
# Subjects: {'HC': 165, 'MCI': 97, 'AD': 38}
# Info for each subject: timeseries, ABeta, Tau (whenever available)
#
# fMRI Parcellated by GUILLERMO MONTAÑA VALVERDE,
# ABeta and Tau by David Aquilué Llorens
#
# Code by Gustavo Patow
# --------------------------------------------------------------------------------------
import os
import glob
import re
import numpy as np
import pandas as pd
import tools.hdf as hdf  # assuming you already use this wrapper for .mat
from DataLoaders.baseDataLoader import DataLoader

# ================================================================================================================
# ADNI_G Loader (DBS80 parcellation, subject-wise .mat files)
# ================================================================================================================
class ADNI_G(DataLoader):
    def __init__(self, path=None,
                 discard_AD_ABminus=False,
                 use_pvc=True):

        self.use_pvc = use_pvc
        self.groups = ['HC', 'MCI', 'AD']

        if path is not None:
            self.set_basePath(path)
        else:
            from DataLoaders.WorkBrainFolder import WorkBrainDataFolder
            self.set_basePath(WorkBrainDataFolder)

        self.timeseries = {}
        self.burdens = {}
        self.meta_information = None
        self.classification = {}

        self._load_metadata()
        self._loadTimeseries()
        self._loadSC()
        self._loadBurdenData()

        if discard_AD_ABminus:
            # Keep same exclusion logic as ADNI_B if needed
            pass

        print(f'loaded {self.get_subject_count()} subjects')

    # ------------------------------------------------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------------------------------------------------
    def set_basePath(self, path):
        self.base_folder = path + "ADNI-G/"
        self.timeseries_folder = self.base_folder + "DBS80/tseries/"
        self.Avg_SC_file = self.base_folder + "DBS80/SC/HCn_connectome_dbs80.mat"

        # Reuse ADNI_B structure for metadata and PET
        self.meta_path = self.base_folder + "demographics.csv"
        self.meta_path2 = self.base_folder + "ADNI3_sites.csv"
        self.ABeta_path = self.base_folder + ('DBS80/abeta_wc_pvc/' if self.use_pvc else 'abeta_wc/')
        self.tau_path = self.base_folder + ('DBS80/tau_igm_pvc/' if self.use_pvc else 'tau_igm/')

    # ------------------------------------------------------------------------------------------------------------
    # Core dataset properties
    # ------------------------------------------------------------------------------------------------------------
    def name(self):
        return "ADNI_G"

    def TR(self):
        return 3  # assuming same as ADNI_B (update if needed)

    def N(self):
        return 80  # DBS80 parcellation

    def get_groupLabels(self):
        return self.groups

    # ------------------------------------------------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------------------------------------------------
    def _load_metadata(self):
        self.meta_information = pd.read_csv(self.meta_path)
        self.meta_information2 = pd.read_csv(self.meta_path2)

    # ------------------------------------------------------------------------------------------------------------
    # Load timeseries
    # ------------------------------------------------------------------------------------------------------------
    def _loadTimeseries(self):
        files = glob.glob(self.timeseries_folder + "sub-*_dbs80_timeseries.mat")

        for f in files:
            filename = os.path.basename(f)

            # Extract subject ID: sub-0010337 → 0010337
            match = re.match(r"sub-(\d+)_dbs80_timeseries.mat", filename)
            if match is None:
                continue

            subj_id = match.group(1)

            # Load MATLAB file
            data = hdf.loadmat(f)
            ts = data['ts']  # <-- key difference

            BIDS_id = 'sub-' + subj_id[0:3] + '-' + subj_id[3:]
            id = int(subj_id[3:])
            # locator = self.meta_information['BIDS_ID'] == BIDS_id  # locate by BIDS_ID
            # locator = self.meta_information['PTID_short'] == id  # locate by PTID_short
            locator = self.meta_information['BIDS_ID'].str[-4:] == subj_id[3:]  # locate by the short PTID in BIDS_ID
            if (locator).any():
                full_subj_id = self.meta_information[locator]['PTID'].values[0]
                self.timeseries[full_subj_id] = ts
                self.classification[full_subj_id] = self.meta_information[locator]['GROUP'].values[0]
            else:  # repêchage, let's see what we can get from the other meta file!
                locator2 = self.meta_information2['ID'] == id
                if (locator2).any():
                    full_subj_id = self.meta_information2[locator2]['Subject ID'].values[0]
                    self.timeseries[full_subj_id] = ts
                    self.classification[full_subj_id] = self.meta_information2[locator2]['Group'].values[0]
                else:
                    print(f'no PTID for {BIDS_id}')

    # ------------------------------------------------------------------------------------------------------------
    # Load average SC
    # ------------------------------------------------------------------------------------------------------------
    def _loadSC(self):
        SCf = hdf.loadmat(self.Avg_SC_file)
        self.SC = SCf['SC']

    # ------------------------------------------------------------------------------------------------------------
    # Load Amyloid / Tau (same philosophy as ADNI_B)
    # ------------------------------------------------------------------------------------------------------------
    def _loadBurdenData(self):
        for subjectID in self.timeseries:
            id_compressed = re.sub('_', '', subjectID)
            abeta = tau = None

            for file in glob.glob(self.ABeta_path + f'*{id_compressed}*.npy'):
                abeta = np.load(file)

            for file in glob.glob(self.tau_path + f'*{id_compressed}*.npy'):
                tau = np.load(file)

            self.burdens[subjectID] = {
                'ABeta': abeta,
                'Tau': tau
            }

    # ------------------------------------------------------------------------------------------------------------
    # Public API (consistent with ADNI_B)
    # ------------------------------------------------------------------------------------------------------------
    def get_classification(self):
        return self.classification

    def get_subjectData(self, subjectID):
        ts = self.timeseries[subjectID]
        meta = self.meta_information[
            self.meta_information['PTID'] == subjectID
        ].to_dict('records')[0]

        res = {
            subjectID: {
                'timeseries': ts,
                'meta': meta,
            }
        }

        if subjectID in self.burdens:
            res[subjectID] |= {
                'ABeta': self.burdens[subjectID]['ABeta'],
                'Tau': self.burdens[subjectID]['Tau']
            }

        return res

    def get_AvgSC_ctrl(self, normalized='maxSC', normalizationFactor=0.2):  # returns a SINGLE SC matrix (average over control subjects)
        normSC = self._normalize_SC(self.SC, normalizationMethod=normalized, normalizationFactor=normalizationFactor)
        return normSC


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    # ---- test DBS 80
    baseDL = ADNI_G()
    sujes = baseDL.get_classification()
    gCtrl = baseDL.get_groupSubjects('HC')
    s1 = baseDL.get_subjectData(gCtrl[0])
    avg_SC = baseDL.get_AvgSC_ctrl()
    print('done DBS80! ;-)')
    # -- just a quick test:
    # # ---- test alternative classification
    # DL = ADNI_B_Alt(baseDL, ['HC(AB-)', 'HC(AB+)', 'MCI(AB+)', 'AD(AB+)'])  # all subjects, irregardly if they have burden or not
    # sujes_alt = DL.get_classification()
    # gCtrl_alt = DL.get_groupSubjects('HC(AB-)')
    # s1_alt = DL.get_subjectData(gCtrl_alt[0])
    # print('done ALT! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF