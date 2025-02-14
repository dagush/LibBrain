import tarfile
import os
import numpy as np

from DataLoaders.WorkBrainFolder import *
from DataLoaders.baseDataLoader import DataLoader

# ==========================================================================
# Important config options: filenames
# ==========================================================================


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class HCP_YA_PTN1200(DataLoader):
    def __init__(self, size=15, path=None):
        self.size = size
        if size not in [15, 25, 50, 100, 200, 300]:
            raise ValueError("size must be 15 25 50 100 200 300")
        base_folder = WorkBrainDataFolder + "HCP/HCP-YA_PTN1200/"
        self.fMRI_path = base_folder + f'NodeTimeseries_3T_HCP1200_MSMAll_ICAd{size}_ts2.tar.gz'
        self.netmats_path = base_folder + f'netmats_3T_HCP1200_MSMAll_ICAd{size}_ts2.tar.gz'
        if path is not None:
            self.set_basePath(path)
        self.tasks = ['rest']
        self.subjs = {}
        self._load_all_data()  # chosenDatasets=chosenDatasets, forceUniqueSet=forceUniqueSet)

    # Node timeseries (individual subjects)
    def _load_fMRI(self):
        with tarfile.open(self.fMRI_path, "r:gz") as tar:
            for member in tar.getmembers():
                id = os.path.basename(member.name).replace('.txt', '')
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read().decode('utf-8', errors='ignore')
                    ts = np.fromstring(content, dtype=float, sep=' ').reshape((-1, self.size))
                    self.subjs[id] = {'timeseries': ts.T}

    # Network-matrices (also referred to as "netmats" or "parcellated connectomes")
    # 1. netmats1: Using "full" normalized temporal correlation between every node timeseries
    #    and every other. This is a common approach and is very simple, but it has various
    #    practical and interpretational disadvantages [Smith 2012].
    # 2. netmats2: Using partial temporal correlation between nodes' timeseries. This aims
    #    to estimate direct connection strengths better than achieved by full correlation. To
    #    slightly improve the estimates of partial correlation coefficients, a small amount
    #    of L2 regularization is applied (setting rho=0.01 in the Ridge Regression netmats
    #    option in FSLNets) [Smith OHBM 2014, FSLNets].
    def _load_netmats(self, type):
        with tarfile.open(self.netmats_path, "r:gz") as tar:
            member = tar.getmember(f'netmats/3T_HCP1200_MSMAll_d{self.size}_ts2/netmats{type}.txt')
            f = tar.extractfile(member)
            if f is not None:
                content = f.read().decode('utf-8', errors='ignore')
                SCs = np.fromstring(content, dtype=float, sep=' ').reshape((-1, self.size, self.size))
                for pos, s in enumerate(self.subjs.keys()):
                    self.subjs[s][f'netmat{type}'] = SCs[pos]
            print('done')

    def _load_all_data(self):
        self._load_fMRI()
        self._load_netmats(1)
        self._load_netmats(2)

    def name(self):
        return 'HCP_YA_PTN1200'

    def set_basePath(self, path):
        global base_folder
        base_folder = path + "_Data_Raw/HCP/"

    def TR(self):
        return 0.72  # Repetition Time (seconds)

    def N(self):
        return self.size

    def get_groupSubjects(self, group):
        test = self.subjs.keys()
        return list(test)

    def get_groupLabels(self):
        return self.tasks

    def get_classification(self):
        classi = {}
        for task in self.tasks:
            classi = {subj: task for subj in self.subjs}
        return classi

    def discardSubject(self, subjectID):
        self.subjs.pop(subjectID)

    def get_subjectData(self, subjectID):
        ts =  self.subjs[subjectID]
        return {subjectID: ts}

    def get_parcellation(self):
        raise ValueError('Available, not implemented yet')


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    import numpy.random as random
    DL = HCP_YA_PTN1200(15)
    sujes = DL.get_classification()
    gCtrl = DL.get_groupSubjects('rest')
    id = '117930'
    s1 = DL.get_subjectData(id)
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF