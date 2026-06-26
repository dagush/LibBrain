# --------------------------------------------------------------------------------------
# Full code for loading the ADNI data in the different parcellations
# RoIs: check parcellation - TR = 3 - timepoints: 197
# Subjects:
# Info for each subject: timeseries for all parcellations, and ABeta and Tau for some
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
import neuronumba.tools.hdf as hdf
import h5py
import scipy.io as sio
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
class ADNI_B2(DataLoader):
    def __init__(self,
                 parcellation,
                 path=None,
                 use_pvc = True,
                 ):
        supported = {'dbs80': 80, 'Glasser360': 360, 'Schaefer100': 100, 'Schaefer400': 400, 'Schaefer1000':1000}
        if parcellation not in supported:
            raise ValueError(f'Invalid parcellation: {parcellation}. Supported parcellations: {supported}')
        self.parcellations_with_burden = ['dbs80', 'Glasser360', 'Schaefer400']
        self.parcellations_with_GMV = ['Schaefer400']
        self.parcellation = parcellation
        self.use_pvc = use_pvc
        self.groups = ['HC-', 'HC+', 'MCI+', 'AD+']
        if path is not None:
            self.set_basePath(path)  #, prefiltered_fMRI)
        else:
            self.set_basePath(WorkBrainDataFolder)  #, prefiltered_fMRI)
        self.timeseries = {}
        self.burdens = {}
        self.meta_information = None
        self._loadAll_MetaData()
        self._loadAllData()
        print(self.get_subject_count())

    def set_basePath(self, path):
        self.base_238_folder = path + "ADNI-B/N238rev/"
        self.base_parcellation_folder = self.base_238_folder + self.parcellation + '/'
        # --------- IDs
        self.ID_path = self.base_238_folder + 'PTIDs/PTID_ADNI3_{}_MPRAGE_IRFSPGR_all.mat'
        # --------- fMRI
        real_parc_names = {'dbs80': 'dbs80',
                           'Glasser360': 'glasser360',
                           'Schaefer100': 'sch100',
                           'Schaefer400': 'sch400',
                           'Schaefer1000': 'sch1000',}
        fMRI_folder = self.base_parcellation_folder + 'tseries/'
        self.fMRI_path = fMRI_folder + 'CONN_denoised_pipeline/'\
                'tseries_ADNI3_{}_MPRAGE_IRFSPGR_' + f'{real_parc_names[self.parcellation]}_N238rev.mat'
        # --------- ABeta and Tau
        if self.parcellation in self.parcellations_with_burden:
            self.ABeta_path = self.base_parcellation_folder + 'abeta_wc' + '_pvc/' if self.use_pvc else '/'
            self.tau_path = self.base_parcellation_folder + 'tau_igm' + '_pvc/' if self.use_pvc else '/'

    def _loadAll_MetaData(self):
        # ---- load common metadata
        meta_data_path = self.base_238_folder + 'ADNI3_N238rev_with_ABETA_Status.xlsx'
        self.meta_information = pd.read_excel(meta_data_path)
        # ---- load site information
        site_path = '/'.join(self.base_238_folder.split('/')[:-2])+'/'
        self.site_information = pd.read_csv(site_path + 'sites_ADNI3_ABeta_N304.csv')
        # ---- load GMV info
        if self.parcellation in self.parcellations_with_GMV:
            site_GMV_path = self.base_parcellation_folder + 'GMV/ADNI_GMV_schaefer400_wide.csv'
            self.GMV = pd.read_csv(site_GMV_path)

    # ---------------- load fMRI data
    def __load_HDF5(self, filename, file_name_detail):
        entry_name = 'tseries_' + file_name_detail
        try:
            f = h5py.File(filename, 'r')
            r = []
            ds = f[entry_name]
            for subj in ds[0]:
                r.append(np.array(f[subj]))
            return r
        except Exception as e:
            f = sio.loadmat(filename)
            return f[entry_name]


    def _loadSubjects_fMRI(self, IDs, fMRI_path, file_name_detail):
        print(f'Loading {fMRI_path}')
        fMRIs = self.__load_HDF5(fMRI_path, file_name_detail)
        res = {str(IDs[i]): fMRIs[i] for i in range(len(IDs))}
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
        real_task_names = {'HC-': ('HC_ABetaNeg','HC_ABneg'),
                           'HC+': ('HC_ABetaPos', 'HC_ABpos'),
                           'MCI+': ('MCI_ABetaPos', 'MCI_ABpos'),
                           'AD+': ('AD_ABetaPos','AD_ABpos')}
        for task in self.groups:
            file_name_detail, field_name = real_task_names[task]
            print(f'----------- Checking: {task} --------------')
            ID_path = self.ID_path.format(file_name_detail)
            PTIDs = hdf.loadmat(ID_path)
            PTIDs = PTIDs['PTID_' + field_name]
            IDs = [id[0] for id in np.squeeze(PTIDs).tolist()]
            fMRI_task_path = self.fMRI_path.format(file_name_detail)
            self.timeseries[task] = self._loadSubjects_fMRI(IDs, fMRI_task_path, field_name)
            if self.parcellation in self.parcellations_with_burden:
                self.burdens[task] = self._loadSubjects_burden(IDs)
            print(f'----------- done {task} --------------')
        print(f'----------- done loading All --------------')

    def name(self):
        return 'ADNI_B_N238rev'

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return self.SchaeferSize

    def get_AvgSC_ctrl(self, normalized=None):
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
        # ---- ABeta and Tau, when available
        if self.parcellation in self.parcellations_with_burden:  # add the burden info
            res[subjectID] |= {'ABeta': self.burdens[group][subjectID]['ABeta'],
                               'Tau': self.burdens[group][subjectID]['Tau'],}
        # ---- GMV information
        if self.parcellation in self.parcellations_with_GMV:
            GMV_subj_id = 'sub-' + subjectID.split('_')[0] + '-' + subjectID.split('_')[2]
            if (self.GMV['subject'] == GMV_subj_id).any():
                subj_GMV = self.GMV.loc[self.GMV['subject'] == GMV_subj_id]
                # Data seems to be in the "canonical" order as the rest of the files here...
                res_GMV = subj_GMV.to_numpy()[0,2:]
            else:
                res_GMV = None  # Always add GMV information...
            res[subjectID] |= {'GMV': res_GMV}
        # ---- site information
        # Quick confirmation checks:
        # print(self.site_information['site'].unique())  # Print all sites
        # print(self.site_information[self.site_information['Measure']=='MRI']['site'].unique())  # Print all MRI sites
        # print(self.site_information[self.site_information['Measure']=='Beta']['site'].unique())  # Print all ABeta sites
        # print(self.site_information[self.site_information['Measure']=='Tau']['site'].unique())  # Print all Tau sites
        subj_site = self.site_information[self.site_information['PTID'] == subjectID]
        site_info = {}
        for _, row in subj_site.iterrows():
            site_info[row['Measure']+'_site'] = row['site']
        res[subjectID]['meta'] |= site_info
        return res

    def get_parcellation(self):
        raise NotImplementedError

    # -------------------------- Modality methods -----------------------------------
    def list_modalities(self):
        return ["fmri", "amyloid", "tau"]

    def get_modality(self, subject, modality, session=None, **kwargs):
        group = self.get_classification()[subject]
        modality = modality.lower()

        if modality == "fmri":
            return self.timeseries[group][subject]

        if modality == "amyloid":
            return self.burdens[group][subject]["ABeta"]

        if modality == "tau":
            return self.burdens[group][subject]["Tau"]

        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Available modalities are: {self.list_modalities()}"
        )


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    # ---- test Schaefer 400
    baseDL = ADNI_B2(parcellation='dbs80')  # Glasser360 / dbs80 / Schaefer100 / Schaefer400 / Schaefer1000
    sujes = baseDL.get_classification()
    gCtrl = baseDL.get_groupSubjects('HC+')
    s1 = baseDL.get_subjectData(gCtrl[0])
    print('done ADNI-B2! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF