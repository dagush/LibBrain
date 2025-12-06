# --------------------------------------------------------------------------------------
# Full code for loading the ADNI data in the Schaefer2018 parcellation 400/1000
# RoIs: 400/1000 - TR = 3 - timepoints: 197
# Subjects: N238rev - HC 109, MCI 90, AD 39
#           N193 No Filt - HC 105, MCI 64, AD 24
#                          'HC(AB-)': 53, 'HC(AB+)': 38, 'MCI(AB+)': 24, 'AD(AB+)': 13
# Info for each subject: timeseries
# Note: not all subjects have ABeta and Tau for 400,
#       no subjects have burden for 1000
#
# fMRI Parcellated by NOELIA MARTINEZ MOLINA
# ABeta and Tau by David Aquilué Llorens
#
# Code by Gustavo Patow
# --------------------------------------------------------------------------------------
import glob
import re
import numpy as np
import pandas as pd
import tools.hdf as hdf
from DataLoaders.baseDataLoader import DataLoader
import DataLoaders.Parcellations.Schaefer2018 as Schaefer2018


# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *

# ================================================================================================================
# ================================================================================================================
# ADNI_B base class Loading
# ================================================================================================================
# ================================================================================================================
class ADNI_B(DataLoader):
    def __init__(self, path=None,
                 # prefiltered_fMRI=False,
                 discard_AD_ABminus=True,
                 SchaeferSize=400,  # by default, let's use the Schaefer2018 400 parcellation / 1000
                 use_pvc = True,
                 ):
        self.SchaeferSize = SchaeferSize
        self.use_pvc = use_pvc
        self.groups = ['HC','MCI', 'AD']
        if path is not None:
            self.set_basePath(path)  #, prefiltered_fMRI)
        else:
            self.set_basePath(WorkBrainDataFolder)  #, prefiltered_fMRI)
        self.timeseries = {}
        self.burdens = {}
        self.meta_information = None
        self._loadAllData()
        if discard_AD_ABminus:
            # ---------- discard all subjects with AD and ABeta-, because they are not subjects usually
            #            classified as having with dementia by AD...
            self.discardSubjects(['116_S_6543','168_S_6754','022_S_6013','126_S_6721'])
        print(self.get_subject_count())

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return self.SchaeferSize

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
        classi = self.get_classification()
        if subjectID in classi:
            group = classi[subjectID]
            del self.timeseries[group][subjectID]

    def get_subjectData(self, subjectID):
        group = self.get_classification()[subjectID]
        ts = self.timeseries[group][subjectID]
        meta = self.meta_information[self.meta_information['PTID'] == subjectID].to_dict('records')[0]
        res= {subjectID: {'timeseries': ts,
                          'meta': meta,}}
        if self.SchaeferSize == 400:  # add the burden info
            res |= {'ABeta': self.burdens[group][subjectID]['ABeta'],
                    'Tau': self.burdens[group][subjectID]['Tau'],}
        return res

    def get_parcellation(self):
        return Schaefer2018.Schaefer2018(N=400, normalization=2, RSN=7)  # use normalization of 2mm, 7 RSNs

# ================================================================================================================
# ================================================================================================================
# ADNI_B_N193_no_filt Loading
# ================================================================================================================
# ================================================================================================================
class ADNI_B_N193_no_filt(ADNI_B):
    def __init__(self, path=None,
                 # prefiltered_fMRI=False,
                 discard_AD_ABminus=True,
                 SchaeferSize=400,  # by default, let's use the Schaefer2018 400 parcellation / 1000
                 use_pvc = True,
                 ):
        super().__init__(path=None,
                         discard_AD_ABminus=True,
                         SchaeferSize=400,  # by default, let's use the Schaefer2018 400 parcellation / 1000
                         use_pvc = True,)
        # self.SchaeferSize = SchaeferSize
        # self.use_pvc = use_pvc
        # self.groups = ['HC','MCI', 'AD']
        # if path is not None:
        #     self.set_basePath(path)  #, prefiltered_fMRI)
        # else:
        #     self.set_basePath(WorkBrainDataFolder)  #, prefiltered_fMRI)
        # self.timeseries = {}
        # self.burdens = {}
        # self.meta_information = None
        # self._loadAllData()
        # if discard_AD_ABminus:
        #     # ---------- discard all subjects with AD and ABeta-, because they are not subjects usually
        #     #            classified as having with dementia by AD...
        #     self.discardSubjects(['116_S_6543','168_S_6754','022_S_6013'])  #,'126_S_6721'])

    def set_basePath(self, path):  #, prefiltered_fMRI):
        # --------- timeseries
        self.base_193_folder = path + "ADNI-B/N193_no_filt/"
        fMRI_folder = self.base_193_folder + f'sch{self.SchaeferSize}/'
        self.fMRI_path = fMRI_folder + 'tseries_ADNI3_{}_MPRAGE_batches{}_' + f'sch{self.SchaeferSize}_matching_QC_COMBINED.mat'
        self.ID_path = fMRI_folder + 'combined_PTIDS_ADNI3_{}_MPRAGE.mat'
        self.base_238_folder = path + "ADNI-B/N238rev/"
        if self.SchaeferSize == 400:
            # --------- ABeta and Tau
            self.ABeta_path = self.base_238_folder + 'abeta_wc' + '_pvc/' if self.use_pvc else '/'
            self.tau_path = self.base_238_folder + 'tau_igm' + '_pvc/' if self.use_pvc else '/'

    # ---------------- load fMRI data
    def _loadSubjects_fMRI(self, IDs, fMRI_path, task):
        print(f'Loading {fMRI_path}')
        fMRIs = hdf.loadmat(fMRI_path)
        fMRIs = fMRIs[f'combined_tseries_ADNI3_{task}_MPRAGE'][:, 0]
        res = {IDs[i].tolist(): fMRIs[i] for i in range(len(IDs))}
        return res

    # ---------------- load burden data
    def _loadSubjects_burden(self, IDs):
        if self.SchaeferSize == 1000:  # we do not have this info!
            return
        abeta_fails = []
        tau_fails = []
        res = {}
        for id in IDs:
            print(f'Loading burdens for {id}')
            id_compressed = re.sub('_','',id)
            abeta = tau = None
            for file in glob.glob(self.ABeta_path + f'*ADNI{id_compressed}*_CL.npy'):
                abeta = np.load(file)
            for file in glob.glob(self.tau_path + f'*ADNI{id_compressed}*.npy'):
                tau = np.load(file)
            if abeta is None: abeta_fails.append(str(id))
            if tau is None: tau_fails.append(str(id))
            res[id] = {'ABeta': abeta, 'Tau': tau}
        # print(f'Failed loads:\n     ABeta ({len(abeta_fails)}): {abeta_fails}')
        # print(f'     Tau ({len(tau_fails)}): {tau_fails}')
        # intersection = set(abeta_fails).intersection(set(tau_fails))
        # print(f'     Intersection ({len(intersection)}): {intersection}')
        # union = set(abeta_fails).union(set(tau_fails))
        # print(f'     Union ({len(union)}): {union}')
        return res

    def _loadAllData(self):
        for task in self.groups:
            print(f'----------- Checking: {task} --------------')
            taskRealName = task
            taskBatch = '123' if task == 'AD' or task == 'HC' else '1'
            ID_path = self.ID_path.format(taskRealName, taskBatch)
            fMRI_task_path = self.fMRI_path.format(taskRealName, taskBatch)
            PTIDs = hdf.loadmat(ID_path)
            PTIDs_name = f'combined_PTIDS_ADNI3_{task}_MPRAGE' if task == 'HC' or task == 'AD' \
                else f'PTID_BIDS_MPRAGE_60_89_batch_1_{task}'
            PTIDs = PTIDs[PTIDs_name]
            IDs = [id[0] for id in np.squeeze(PTIDs).tolist()]
            self.timeseries[task] = self._loadSubjects_fMRI(IDs, fMRI_task_path, task)
            self.burdens[task] = self._loadSubjects_burden(IDs)
            print(f'----------- done {task} --------------')
        meta_data_path = self.base_238_folder + 'ADNI3_N238rev_with_ABETA_Status.xlsx'
        self.meta_information = pd.read_excel(meta_data_path)
        print(f'----------- done loading All --------------')

    def name(self):
        return 'ADNI_B_N193_no_filt'


# ================================================================================================================
# ================================================================================================================
# ADNI_B_N238rev Loading
# ================================================================================================================
# ================================================================================================================
class ADNI_B_N238rev(ADNI_B):
    def __init__(self, path=None,
                 # prefiltered_fMRI=False,
                 discard_AD_ABminus=True,
                 # ADNI_version='N238rev',  # N238rev
                 # SchaeferSize=400,  # by default, let's use the Schaefer2018 400 parcellation
                 use_pvc=True,
                 ):
        super().__init__(path=None,
                         discard_AD_ABminus=True,
                         SchaeferSize=400,  # by default, let's use the Schaefer2018 400 parcellation / 1000
                         use_pvc = True,)
        # # self.SchaeferSize = SchaeferSize
        # # self.ADNI_version = ADNI_version
        # self.use_pvc = use_pvc
        # self.groups = ['HC','MCI', 'AD']
        # if path is not None:
        #     self.set_basePath(path)  #, prefiltered_fMRI)
        # else:
        #     self.set_basePath(WorkBrainDataFolder)  #, prefiltered_fMRI)
        # self.timeseries = {}
        # self.burdens = {}
        # self.meta_information = None
        # self._loadAllData()
        # if discard_AD_ABminus:
        #     # ---------- discard all subjects with AD and ABeta-, because they are not subjects usually
        #     #            classified as having with dementia by AD...
        #     self.discardSubjects(['116_S_6543','168_S_6754','022_S_6013','126_S_6721'])

    def set_basePath(self, path):  #, prefiltered_fMRI):
        self.base_folder = path + "ADNI-B/N238rev/"
        fMRI_folder = self.base_folder + 'tseries/sch400/'
        # if prefiltered_fMRI:
        #     self.fMRI_path = fMRI_folder + 'tseries_ADNI3_{}_MPRAGE_IRFSPGR_sch400_N238rev.mat'
        # else:
        self.fMRI_path = fMRI_folder + 'tseries_ADNI3_{}_MPRAGE_IRFSPGR_sch400_N238rev_nofilt.mat'
        self.ID_path = fMRI_folder + 'PTID_ADNI3_{}_MPRAGE_IRFSPGR_all.mat'
        self.ABeta_path = self.base_folder + 'abeta_wc' + ('_pvc/' if self.use_pvc else '/')
        self.tau_path = self.base_folder + 'tau_igm' + ('_pvc/' if self.use_pvc else '/')

    # ---------------- load fMRI data
    def _loadSubjects_fMRI(self, IDs, fMRI_path):
        print(f'Loading {fMRI_path}')
        fMRIs = hdf.loadmat(fMRI_path)
        fMRIs = fMRIs['tseries'][:, 0]
        res = {IDs[i].tolist(): fMRIs[i] for i in range(len(IDs))}
        return res

    # ---------------- load burden data
    def _loadSubjects_burden(self, IDs):
        abeta_fails = []
        tau_fails = []
        res = {}
        for id in IDs:
            print(f'Loading burdens for {id}')
            id_compressed = re.sub('_','',id)
            abeta = tau = None
            for file in glob.glob(self.ABeta_path + f'*ADNI{id_compressed}*_CL.npy'):
                abeta = np.load(file)
            for file in glob.glob(self.tau_path + f'*ADNI{id_compressed}*.npy'):
                tau = np.load(file)
            if abeta is None: abeta_fails.append(str(id))
            if tau is None: tau_fails.append(str(id))
            res[id] = {'ABeta': abeta, 'Tau': tau}
        # print(f'Failed loads:\n     ABeta ({len(abeta_fails)}): {abeta_fails}')
        # print(f'     Tau ({len(tau_fails)}): {tau_fails}')
        # intersection = set(abeta_fails).intersection(set(tau_fails))
        # print(f'     Intersection ({len(intersection)}): {intersection}')
        # union = set(abeta_fails).union(set(tau_fails))
        # print(f'     Union ({len(union)}): {union}')
        return res

    def _loadAllData(self):
        for task in self.groups:
            print(f'----------- Checking: {task} --------------')
            taskRealName = task
            ID_path = self.ID_path.format(taskRealName)
            fMRI_task_path = self.fMRI_path.format(taskRealName)
            PTIDs = hdf.loadmat(ID_path)
            PTIDs = PTIDs['PTID']
            IDs = [id[0] for id in np.squeeze(PTIDs).tolist()]
            self.timeseries[task] = self._loadSubjects_fMRI(IDs, fMRI_task_path)
            self.burdens[task] = self._loadSubjects_burden(IDs)
            print(f'----------- done {task} --------------')
        meta_data_path = self.base_folder + 'ADNI3_N238rev_with_ABETA_Status.xlsx'
        self.meta_information = pd.read_excel(meta_data_path)
        print(f'----------- done loading All --------------')

    def name(self):
        return 'ADNI_B_N238rev'


# ================================================================================================================
# ================================================================================================================
# Alternate Classification DataLoader
# This allows different classification schemes, such as
#       ['HC', 'AD']  -> all subjects with labels HC and AD, irrespectively of their ABeta status
#       ['HC', 'MCI(AB-)', 'MCI(AB+)', 'AD']  -> Same as before, with the MCI subjects that are either AB- or AB+
#       ['HC(AB-)', 'HC(AB+)', 'MCI(AB-)', 'MCI(AB+)', 'AD']
#       ['HC(AB-)', 'HC(AB+)', 'MCI(AB-)', 'MCI(AB+)', 'AD(AB+)']
# Observe the last two should be the same, but with all the AD or only those with an AB+ status
# Note: at this moment, this class is intimately related to the ADNI_B_N193_no_filt DataLoader
# ================================================================================================================
# ================================================================================================================
class ADNI_B_Alt(DataLoader):
    def __init__(self, OrigDataLoader, new_classification):
        self.DL = OrigDataLoader
        # self.DL = ADNI_B_N193_no_filt(path, #prefiltered_fMRI=prefiltered_fMRI,
        #                               discard_AD_ABminus=discard_AD_ABminus,
        #                               SchaeferSize=SchaeferSize,
        #                               use_pvc=use_pvc)
        # self.DL = ADNI_B_N238rev(path, #prefiltered_fMRI=prefiltered_fMRI,
        #                          discard_AD_ABminus=discard_AD_ABminus,
        #                          use_pvc=use_pvc)
        self.groups = new_classification
        self.classification = {}
        orig_classification = self.DL.get_classification()

        # Regex: group + optional (BURDEN with optional + or -)
        pattern = re.compile(r"^([A-Z]+)(?:\(([A-Z]+)([+-]?)\))?$")

        for subject in orig_classification:
            subject_group = orig_classification[subject]
            for group in new_classification:
                if subject_group in group:  # we discard the subject if it is NOT in any set
                    m = pattern.match(group)
                    if m:
                        group_id, burden, sign = m.groups()
                        if burden is None:
                            self.classification[subject] = group
                        else:
                            data = self.get_subjectData(subject)[subject]
                            # labels are in the Abeta_pvc column, where pvc = partial volume correction.
                            # 1=Abeta+, 0=Abeta-;   threshold is >24CL.
                            # Subjects without ABeta classification are discarded
                            if ((data['meta']['ABeta_pvc'] == 0 and sign=='-')  # ABeta-
                                    or
                                (data['meta']['ABeta_pvc'] == 1 and sign=='+')):  # ABeta+
                                self.classification[subject] = group

        print(self.get_subject_count())

    def name(self):
        return self.DL.name() + '_alt'

    def get_groupLabels(self):
        return self.groups

    def get_classification(self):
        return self.classification

    def get_subjectData(self, subjectID):
        return self.DL.get_subjectData(subjectID)

    def get_parcellation(self):
        return self.DL.get_parcellation()

    def TR(self):
        return self.DL.TR()

    def N(self):
        return self.DL.N()


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    # ---- test Schaefer 400
    baseDL = ADNI_B_N193_no_filt()  # ADNI_B_N238rev / ADNI_B_N193_no_filt
    sujes = baseDL.get_classification()
    gCtrl = baseDL.get_groupSubjects('HC')
    s1 = baseDL.get_subjectData(gCtrl[0])
    print('done sch400! ;-)')
    # ---- test Schaefer 1000
    DL = ADNI_B_Alt(baseDL, ['HC(AB-)', 'HC(AB+)', 'MCI(AB+)', 'AD(AB+)'])  # all subjects, irregardly if they have burden or not
    sujes_alt = DL.get_classification()
    gCtrl_alt = DL.get_groupSubjects('HC(AB-)')
    s1_alt = DL.get_subjectData(gCtrl_alt[0])
    print('done sch! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF