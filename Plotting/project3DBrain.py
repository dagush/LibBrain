import nibabel as nib

# ==========================================================================
# Important config options: filenames
# ==========================================================================
import DataLoaders.WorkBrainFolder as WBF
brain_3D_directory = WBF.WorkBrainDataFolder + 'brain_map/'

import Utils.geometric as geometric


# ===========================
#  Convenience function for the Glasser parcellation, for debug purposes only...
# ===========================
def set_up_Glasser360_cortex(base_folder):
    Glasser360_baseFolder = base_folder + "Glasser360/"

    # =============== Load the geometry ==================
    glassers_L = nib.load(Glasser360_baseFolder + 'Glasser360.L.mid.32k_fs_LR.surf.gii')
    # glassers_L = nib.load(Glasser360_baseFolder + 'Glasser360.L.inflated.32k_fs_LR.surf.gii')
    # glassers_L = nib.load(Glasser360_baseFolder + 'Glasser360.L.very_inflated.32k_fs_LR.surf.gii')

    glassers_R = nib.load(Glasser360_baseFolder + 'Glasser360.R.mid.32k_fs_LR.surf.gii')
    # glassers_R = nib.load(Glasser360_baseFolder + 'Glasser360.R.inflated.32k_fs_LR.surf.gii')
    # glassers_R = nib.load(Glasser360_baseFolder + 'Glasser360.R.very_inflated.32k_fs_LR.surf.gii')

    flat_L = nib.load(Glasser360_baseFolder + 'Glasser360.L.flat.32k_fs_LR.surf.gii')
    flat_R = nib.load(Glasser360_baseFolder + 'Glasser360.R.flat.32k_fs_LR.surf.gii')
    mapL = nib.load(Glasser360_baseFolder + 'fsaverage.L.glasser360_fs_LR.func.gii').agg_data()
    mapR = nib.load(Glasser360_baseFolder + 'fsaverage.R.glasser360_fs_LR.func.gii').agg_data()

    cortex = {'model_L': glassers_L, 'model_R': glassers_R,
              'flat_L': flat_L, 'flat_R': flat_R,
              'map_L': mapL, 'map_R': mapR}
    return cortex


def set_up_cortex(coordinates):
    flat_L = nib.load(f'{brain_3D_directory}/L.flat.32k_fs_LR.surf.gii')
    flat_R = nib.load(f'{brain_3D_directory}/R.flat.32k_fs_LR.surf.gii')
    model_L = nib.load(f'{brain_3D_directory}/L.mid.32k_fs_LR.surf.gii')
    model_R = nib.load(f'{brain_3D_directory}/R.mid.32k_fs_LR.surf.gii')
    crtx = {
        'map_L': geometric.findClosestPoints(coordinates, model_L.darrays[0].data)[0].flatten(),  # indexes
        'map_R': geometric.findClosestPoints(coordinates, model_R.darrays[0].data)[0].flatten(),  # indexes
        'flat_L': flat_L, 'flat_R': flat_R,
        'model_L': model_L, 'model_R': model_R
    }
    return crtx
