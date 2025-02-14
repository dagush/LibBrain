# --------------------------------------------------------------------------------------
# Full pipeline for loading the HCB data in the Schaefer2018 parcellation 1000
# Subjects: Rest 1003 (989 without NaNs) - RoIs: 1000 - TR = 0.72 - timepoints: 1200
#
# The HCP dataset was graciously provided by Morten Kringelbach
#
# By Gustavo Patow
# --------------------------------------------------------------------------------------
import csv
import numpy as np
import neuronumba.tools.hdf as hdf
import h5py

from DataLoaders.baseDataLoader import DataLoader
import DataLoaders.Parcellations.Schaefer2018 as Schaefer2018


# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *
# ==========================================================================
# ==========================================================================
# ==========================================================================

maxSubjects = 1003
tasks = ['REST1']  # ['REST1', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
minLength = 175


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class HCP(DataLoader):
    def __init__(self, path=None,
                 SchaeferSize=1000,  # by default, let's use the Schaefer2018 1000 parcellation
                 ):
        self.SchaeferSize = SchaeferSize
        if path is not None:
            self.set_basePath(self, path)
        else:
            self.set_basePath(WorkBrainDataFolder)
        self.timeseries = {}
        self.excluded = {}
        self.__loadFilteredData()  # chosenDatasets=chosenDatasets, forceUniqueSet=forceUniqueSet)

    def name(self):
        return 'HCP_schaefer1000'

    def set_basePath(self, path):
        self.base_folder = path + "HCP/Schaefer2018/"
        if self.SchaeferSize == 1000:
            self.fMRI_path = self.base_folder + str(self.SchaeferSize) + '/hcp_{}_LR_schaefer1000.mat'
        else:
            self.fMRI_path = self.base_folder + str(self.SchaeferSize) + '/ALL_HCP_100_unrelated_Schaefer2018_100Parcels_17Networks_order_{}_LR.mat'
        # self.parcellations_folder = WorkBrainDataFolder + "Data_Raw/_Parcellations/Schaefer2018/"

    def TR(self):
        return 0.72  # Repetition Time (seconds)

    def N(self):
        return self.SchaeferSize

    # # get_fullGroup_fMRI: convenience method to load all fMRIs for a given subject group
    # def get_fullGroup_fMRI(self, group):
    #     return timeseries[group]

    def _correctSC(self, SC):
        return SC/np.max(SC)

    def get_AvgSC_ctrl(self, normalized=None):
        if self.SchaeferSize == 1000:
            SC = hdf.loadmat(self.base_folder + '1000/sc_schaefer_MK.mat')['sc_schaefer']
        else:
            SC = hdf.loadmat(self.base_folder + '100/SC_schaefer100_17Networks_32fold_groupconnectome_2mm_symm.mat')['SC']
        if normalized is not None:
            return self._correctSC(SC)
        else:
            return SC

    def get_groupSubjects(self, group):
        test = self.timeseries[group].keys()
        return list(test)

    def get_groupLabels(self):
        return tasks

    def get_classification(self):
        classi = {}
        for task in tasks:
            test = self.timeseries[task].keys()
            for subj in test:
                classi[subj] = subj[1]
        return classi

    def discardSubject(self, subjectID):
        self.timeseries[subjectID[1]].pop(subjectID)

    def get_subjectData(self, subjectID):
        ts = self.timeseries[subjectID[1]][subjectID]
        return {subjectID: {'timeseries': ts}}

    def get_parcellation(self):
        return Schaefer2018.Schaefer2018(N=self.SchaeferSize, normalization=1, RSN=17)  # USe normalization 1mm, 17 RSNs

    # --------------------------------------------------------------------------
    # functions to load fMRI data for certain subjects
    # --------------------------------------------------------------------------
    def __read_matlab_h5py(self, filename, task, selectedIDs):
        with h5py.File(filename, "r") as h5File:
            # Print all root level object names (aka keys)
            # these can be group or dataset names
            # print("Keys: %s" % f.keys())
            # get first object name/key; may or may NOT be a group
            # a_group_key = list(f.keys())[0]
            # get the object type for a_group_key: usually group or dataset
            # print(type(f['subjects_idxs']))
            # If a_group_key is a dataset name,
            # this gets the dataset values and returns as a list
            # data = list(f[a_group_key])
            # preferred methods to get dataset values:
            # ds_obj = f[a_group_key]  # returns as a h5py dataset object
            # ds_arr = f[a_group_key][()]  # returns as a numpy array

            all_fMRI = {}
            excluded = []
            subjects = list(h5File['subject'])
            for pos, subj in enumerate(subjects):
                # print(f'reading subject {pos}')
                group = h5File[subj[0]]
                try:
                    dbs80ts = np.array(group['schaeferts'])
                    if dbs80ts.shape[0] < minLength:  # Some individuals have too short time series
                        print(f'should ignore register {subj} at {(pos, task)}: length {dbs80ts.shape[0]}')
                    all_fMRI[(pos, task)] = dbs80ts.T
                except:
                    print(f'ignoring register {subj} at {pos}')
                    excluded.append(pos)
        return all_fMRI, excluded

    def __loadSubjectsData(self, fMRI_path, task, selectedIDs):
        print(f'Loading {fMRI_path}')
        fMRIs, excluded = self.__read_matlab_h5py(fMRI_path, task, selectedIDs)  # ignore the excluded list
        return fMRIs, excluded

    # --------------------------------------------------------------------------
    # ---------------- load a previously saved list
    # --------------------------------------------------------------------------
    def loadSubjectList(self, path):
        subjects = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                subjects.append(int(row[0]))
        return subjects

    # save a freshly created list
    def saveSelectedSubjects(self, path, subj):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for s in subj:
                writer.writerow([s])

    # # fix subset of subjects to sample
    # def selectSubjectSubset(selectedSubjectsF, numSampleSubj, forceRecompute=False):
    #     if not os.path.isfile(selectedSubjectsF) or forceRecompute:  # if we did not already select a list...
    #         allExcl = set()
    #         for task in tasks:
    #             allExcl |= set(excluded[task])
    #         listIDs = random.sample(range(0, maxSubjects), numSampleSubj)
    #         while allExcl & set(listIDs):
    #             listIDs = random.sample(range(0, maxSubjects), numSampleSubj)
    #         saveSelectedSubjects(selectedSubjectsF, listIDs)
    #     else:  # if we did, load it!
    #         listIDs = loadSubjectList(selectedSubjectsF)
    #     # ---------------- OK, let's proceed
    #     return listIDs

    # --------------------------------------------------------------------------
    # ---------------- load and filter data (some entries are "broken")
    # --------------------------------------------------------------------------
    def __loadFilteredData(self, chosenDatasets=tasks,  # forceUniqueSet=False
                         ):
        allSubj = set([s for s in range(maxSubjects)])

        for task in chosenDatasets:
            print(f'----------- Checking: {task} --------------')
            fMRI_task_path = self.fMRI_path.format(task)
            self.timeseries[task], self.excluded[task] = self.__loadSubjectsData(fMRI_task_path, task, allSubj)
            print(f'------ Excluded: {len(self.excluded[task])} for {task}------')


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = HCP(SchaeferSize=100)
    sujes = DL.get_classification()
    gCtrl = DL.get_groupSubjects('REST1')
    s1 = DL.get_subjectData((0,'REST1'))
    sc = DL.get_AvgSC_ctrl(normalized=True)
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
