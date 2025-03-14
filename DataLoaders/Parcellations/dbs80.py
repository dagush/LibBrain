# =====================================================================================
# Methods to input dbs80 Parcellation data
#
# =====================================================================================
import numpy as np
import csv
import pandas as pd

from DataLoaders.Parcellations.parcellation import Parcellation
from DataLoaders.Parcellations.atlas import Atlas
from neuronumba.tools import hdf

from DataLoaders.WorkBrainFolder import *
dbs80ParcellationFolder = WorkBrainDataFolder + "_Parcellations/dbs80/"


class dbs80(Parcellation):
    names = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default', 'Subcortical']

    # def get_coords(self):
    #     # ----------------- coordinates, but only for the 80 version...
    #     cog = np.loadtxt(dbs80ParcellationFolder + 'Glasser360_coords.txt')
    #     return cog

    def get_region_labels(self):
        # ----------------- node long labels
        df = pd.read_csv(dbs80ParcellationFolder + 'dbs80_labels.csv', sep=';')
        nlist = df['Rois'].tolist()
        return nlist

    # def get_region_short_labels(self):
    #     # ----------------- node labels
    #     with open(dbs80ParcellationFolder + 'dbs80.csv', 'r') as file:
    #         node_names = [line.strip() for line in file]
    #     return node_names

    # def get_cortices(self):
    #     # ----------------- Cortices
    #     df = pd.read_csv(dbs80ParcellationFolder + 'HCP-MMP1_UniqueRegionList.csv')
    #     clist = df['cortex'][0:180].tolist()
    #     cortex = clist + clist + ['Subcortical'] * 18 + ['Brainstem']
    #     return cortex

    def get_RSN(self, useLR=False):
        with open(dbs80ParcellationFolder + 'dbs80Yeo7.csv') as csvfile:
            csvReader = csv.reader(csvfile)
            data = list(csvReader)[0]
        data.extend([8] * 18)  # subcortical areas
        RSN_labels = [self.names[int(r)-1] for r in data]
        return RSN_labels

    def get_atlas_MNI(self):
        return Atlas('dbs80')

# ================================================================================================================
# =========================  debug
if __name__ == '__main__':
    Parc = dbs80()
    labels = Parc.get_region_labels()
    RSNs = Parc.get_RSN()
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
