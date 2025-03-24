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
parcellation_folder = WorkBrainDataFolder + "_Parcellations/"


class aal(Parcellation):
    def __init__(self, version=1, sampling_size=1):
        super().__init__()
        self.version = version  # 1 / 2 / 3
        self.sampling_size = sampling_size
        self.aal_folder = parcellation_folder + f'AAL{version}/'
        if version == 1:
            self.atlas_file = self.aal_folder + 'atlas/AAL.nii'
        elif version==2:
            self.atlas_file = self.aal_folder + 'atlas/AAL2.nii'
        else:
            if sampling_size == 1:
                self.atlas_file = self.aal_folder + 'AAL3v1_1mm.nii'
            else:
                self.atlas_file = self.aal_folder + 'AAL3v1.nii'
        self.names_data = self._load_names_file()

    def _load_names_file(self):
        if self.version == 1 or self.version == 2:
            labels_file = f'ROI_MNI_V{self.version+3}.txt'
            df = pd.read_csv(self.aal_folder + labels_file, sep='\\t', header=None,
                             names=['nom_c', 'nom_l', 'vol_vox'])
        else:
            labels_file = f'ROI_MNI_V7_1mm_vol.txt' if self.sampling_size == 1 else f'ROI_MNI_V7_vol.txt'
            df = pd.read_csv(self.aal_folder + labels_file, sep='\\t')
        return df

    # def get_coords(self):
    #     # cog = self.data[['R','A','S']].to_numpy()
    #     # return cog
    #     pass

    def get_region_labels(self):
        nlist = self.names_data['nom_l'].tolist()
        return nlist

    def get_IDs(self):
        nlist = self.names_data['vol_vox'].tolist()
        return nlist


    # def get_RSN(self, useLR=False):
    #     # names = self.get_region_labels()
    #     # RSNs = [n.split('_')[2] if not useLR else n.split('_')[2]+'_'+n.split('_')[1] for n in names]
    #     # return RSNs
    #     pass

    def get_atlas_MNI(self):
        return Atlas(self.atlas_file)


# ================================================================================================================
# =========================  debug
if __name__ == '__main__':
    Parc = aal(version=3, sampling_size=None)
    labels = Parc.get_region_labels()
    # RSNs = Parc.get_RSN()
    print('done! ;-)')
# =====================================================================================
# =====================================================================================
# =====================================================================================EOF