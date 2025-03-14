# =====================================================================================
# Methods to input Schaeffer2018 Parcellation data
# https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal
#
# Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ, Eickhoff SB, Yeo BTT. Local-Global parcellation
# of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral Cortex, 29:3095-3114, 2018
#
# Code by Gustavo Patow
# =====================================================================================
import numpy as np
import pandas as pd

from DataLoaders.Parcellations.parcellation import Parcellation
from DataLoaders.Parcellations.atlas import Atlas
from tools import hdf

from DataLoaders.WorkBrainFolder import *
parcellation_folder = WorkBrainDataFolder + "_Parcellations/Schaefer2018/"
centroids_folder = parcellation_folder + 'MNI/Centroid_coordinates/'


class Schaefer2018(Parcellation):
    def __init__(self, N, normalization, RSN):
        self.N = N  # 100..1000
        self.RSN = RSN  # 7/17 we always load one, just in case
        self.normalization = normalization  # 1/2
        self.load()

    def load(self):
        filename = f"Schaefer2018_{self.N}Parcels_{self.RSN}Networks_order_FSLMNI152_{self.normalization}mm.Centroid_RAS.csv"
        self.data = pd.read_csv(centroids_folder + filename, delimiter=',')

    def get_coords(self):
        cog = self.data[['R','A','S']].to_numpy()
        return cog

    def get_region_labels(self):
        nodeInfo = self.data['ROI Name'].tolist()
        return nodeInfo

    def get_RSN(self, useLR=False):
        names = self.get_region_labels()
        RSNs = [n.split('_')[2] if not useLR else n.split('_')[2]+'_'+n.split('_')[1] for n in names]
        return RSNs

    def get_atlas_MNI(self):
        return Atlas('Schaefer2018',
                     N=self.N, normalization=self.normalization, RSN=self.RSN)

# =====================================================================================
# =====================================================================================
# =====================================================================================EOF