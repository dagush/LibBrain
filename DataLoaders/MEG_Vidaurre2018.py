# ================================================================================================================
# ================================================================================================================
# Loader for Vidaurre NatComms2018 resting state HCP MEG data
# Subjects: Rest 55 - RoIs: 42 - TR = 1/250 (250 Hz) - timepoints: 72500
#
# Described at
#   Vidaurre, D., Hunt, L.T., Quinn, A.J. et al.
#   Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks.
#   Nat Commun 9, 2987 (2018). https://doi.org/10.1038/s41467-018-05316-z
#
# Dataset from https://ora.ox.ac.uk/objects/uuid:2770bfd4-6ab8-4f1e-b5e7-06185e8e2ae1
# ================================================================================================================
# ================================================================================================================
import numpy as np
import pandas as pd
from DataLoaders.baseDataLoader import DataLoader
from neuronumba.tools import hdf

# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *
base_folder = WorkBrainDataFolder + "HCP/MEG/"
data_folder = base_folder + "Vidaurre_NatComms2018_resting_state/"


# ==========================================================================
# As this seems to be a non-standard parcellation, for the moment we put it here...
# ==========================================================================
class MEG_Parcellation:
    def get_coords(self):
        values = pd.read_excel(base_folder + 'parcellationInfo.xlsx')['MNI Coordinates (X,Y,Z)'].tolist()
        coords = np.array([np.array(v.translate({ord(i): None for i in '())'}).split(',')).astype(int) for v in values])
        return coords

    def get_region_labels(self):
        return self.get_region_short_labels()

    def get_region_short_labels(self):
        values = pd.read_excel(base_folder + 'parcellationInfo.xlsx')['Area']
        names = values.tolist()
        return names


# ==========================================================================
# Vidaurre's 2010 dataset loader
# ==========================================================================
class MEG_Vidaurre2018(DataLoader):
    def name(self):
        return 'MEG_Vidaurre2018'

    def set_basePath(self, path):
        global base_folder
        base_folder = path

    def TR(self):  # Returns a float with the TR of the dataset
        return 1./250.  # Data subsamples at 250 Hz

    def N(self):  # returns an integer with the number of RoIs in the parcellation
        return 42

    def get_classification(self):  # Returns a dict with {subjID: groupLabel}
        return {'subj'+str(id+1): 'REST1' for id in range(55)}

    def get_subjectData(self, subjectID):
        subj = hdf.loadmat(data_folder + subjectID + '.mat')['X'].T
        return {subjectID: {'timeseries': subj}}

    def get_parcellation(self):  # Should return a class!
        return MEG_Parcellation()


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = MEG_Vidaurre2018()
    sujes = DL.get_classification()
    print(f'Classification: {sujes}')
    print(f'Group labels: {DL.get_groupLabels()}')
    gCtrl = DL.get_groupSubjects('REST1')
    s1 = DL.get_subjectData('subj10')
    # ================= Parcellation
    p = DL.get_parcellation()
    c = p.get_region_short_labels()
    print('done! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF