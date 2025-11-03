# ======================================================
# Resting State Networks transfer between parcellations
# ----------------------
# This code uses the Schaefer 1000 Centers of Gravity
# coords (CoG) to compute the RSN labels for any other
# parcellation that has, at least, its CoG defined.
#
# Code by Gustavo Patow
# ======================================================
import csv
import numpy as np
from scipy import spatial

import DataLoaders.WorkBrainFolder as WBF
from Plotting.plot3DPointCloud import plot_point_cloud


# def readDestinationParcellation(filePath):
#     res = []
#     with open(filePath) as f:
#         lines = f.readlines()
#         for line in lines:
#             res.append(('', '', np.fromstring(line, dtype=float, sep=' ')))
#     return res


# Taken from
#   https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
# The approach is generally to first use the point data to build up a k-d tree. The computational complexity
# of that is on the order of N log N, where N is the number of data points. Range queries and nearest neighbour
# searches can then be done with log N complexity. This is much more efficient than simply cycling through all
# points (complexity N).
#   Thus, if you have repeated range or nearest neighbor queries, a k-d tree is highly recommended.
def findClosestPoints(reference, target):
    tree = spatial.cKDTree(reference)
    dist, indexes = tree.query(target)
    return indexes


def assignRSNLabels(referenceSet, target_coords):
    # Shoe-horn existing data for entry into KDTree routines
    reference_coords = np.array([r[2] for r in referenceSet])
    res = findClosestPoints(reference_coords, target_coords)

    targetSetLabelled = [('', referenceSet[res[pos]][1], r) for pos, r in enumerate(target_coords)]

    return targetSetLabelled


# Simple function to extract the RSN name from a description string in Yeo's format. For instance, from
# '7Networks_LH_Default_Temp_8' we extract 'Default'. If an detailedRSNs llist is added, these will be
# added to the output. If no detailes wanted, just pass []
def extractRSNName(name, useLR, detailedRSNs):
    rsnName = name.split('_')[2]
    if rsnName in detailedRSNs:
        subregionsTest = [sub in name.split('_')[3] for sub in detailedRSNs[rsnName]]  # check whether the subarea name is in the list
        if any(subregionsTest):
            rsnName += '_' + detailedRSNs[rsnName][subregionsTest.index(True)]
        else:
            if len(detailedRSNs[rsnName]) > 0:  # if we were given a list, but this particular area is missing...
                rsnName += '_OTHER'
            # If we weren't given a llist, nothing to do!
    if useLR:
        rsnName += '_' + name.split('_')[1]  # and clean them! (left/right separated)
    return rsnName


# detailedRSNs is a dictionary of {'RoI': subareas}, where
# subareas is a list of all subarea names to be considered. If empty, the default RoI value
# will be used. All nodes not in any of these subareas will be added to the 'OTHER' default area.
def collectNamesRSN(rsn, useLR=True, detailedRSNs={}):
    names = [(roi[1], int(roi[0])-1) for roi in rsn]  # extract names
    cleanNames = [extractRSNName(n[0], useLR, detailedRSNs) for n in names]
    return cleanNames


def indices4RSNs(parcellation):
    names = list(set(parcellation))
    res = {}
    for rsn in names:
        idx = [pos for pos,roi in enumerate(parcellation) if roi == rsn]
        res[rsn] = idx
    return res


# ================================================================
# Load and save parcellation data
# ================================================================
def readReferenceRSN(parcellation, roundCoords=True):
    parc_name = parcellation.get_name() + '-' + str(parcellation.get_N())
    file_RSN = WBF.WorkBrainProducedDataFolder + '_Parcellations/' + parc_name + '_RSN.csv'
    res = []
    with open(file_RSN, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if roundCoords:
                res.append((row['ROI Label'], row['ROI Name'], np.array([int(float(row['R'])), int(float(row['A'])), int(float(row['S']))])))
            else:
                res.append((row['ROI Label'], row['ROI Name'], np.array([float(row['R']), float(row['A']), float(row['S'])])))
    return res


def read_RSN_data(target_parcellation, useLR=False):
    parc_name = target_parcellation.get_name() + '-' + str(target_parcellation.get_N())
    file_RSN = WBF.WorkBrainProducedDataFolder + '_Parcellations/' + parc_name + '_RSN.csv'
    file_indices = WBF.WorkBrainProducedDataFolder + '_Parcellations/' + parc_name + f'_RSN_{"14" if useLR else "7"}_indices.csv'  # if we do NOT use detailed regions
    file_labels = WBF.WorkBrainProducedDataFolder + '_Parcellations/' + parc_name + f'_RSN_{"14" if useLR else "7"}_RSN_labels.csv'  # if we do NOT use detailed regions

    res_indices = {}
    with open(file_indices, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            res_indices |= {row['RSN Label']: row['Indices']}

    return res_indices


def save_parcellation_info(target_parcellation, parcellation_info):
    # this keeps all the original names, no distinctions whether we use detailed regions or not...
    parc_name = target_parcellation.get_name() + '-' + str(target_parcellation.get_N())
    filename = WBF.WorkBrainProducedDataFolder + '_Parcellations/' + \
               parc_name + '_RSN.csv'
    header = ["ROI Label", "ROI Name", "R", "A", "S"]
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for r in parcellation_info:
            writer.writerow(r)
    print(f'saved parcellation to: {filename}')


def save_RSN_Indices(target_parcellation, idxs, useLR=False):
    # The outfile should change according to whether we use or not detailed regions...
    parc_name = target_parcellation.get_name() + '-' + str(target_parcellation.get_N())
    outFile = WBF.WorkBrainProducedDataFolder + '_Parcellations/' + \
              parc_name + f'_RSN_{"14" if useLR else "7"}_indices.csv'  # if we do NOT use detailed regions
    header = ["RSN Label", "Indices"]
    with open(outFile, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rsn in idxs:
            writer.writerow([rsn, idxs[rsn]])
    print(f'saved indices to: {outFile}')


def save_RSN_labels(target_parcellation, idxs, useLR=False):
    parc_name = target_parcellation.get_name() + '-' + str(target_parcellation.get_N())
    outFile = WBF.WorkBrainProducedDataFolder + '_Parcellations/' + \
              parc_name + f'_RSN_{"14" if useLR else "7"}_RSN_labels.csv'  # if we do NOT use detailed regions
    # header = ["RSN Label"]
    with open(outFile, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(header)
        for rsn in idxs:
            writer.writerow([rsn])


# -----------------------------------------------------------------
# Convenience method to transfer from a parcellation
# -----------------------------------------------------------------
def build_RSN_for_parcellation(
        target_parcellation,
        plotNodes=False,
        useLR=False,
        detailNetworks={},  # If a mode detailed region is NOT needed, use an empty detailNetworks
                            # detailNetworks = {'Default': ['PFC', 'Par', 'Temp', 'pCunPCC', 'PHC']}
                            # If a more detailed region is needed, especify it here (see comment for collectNamesAndIDsRSN)
        ):
    numNodes = 1000
    # -------- As input, we are going to use Yeo's 1000 roi RSN info on Schaefer's 2018 parcellation
    inPath = WBF.WorkBrainDataFolder + '_Parcellations/'
    inFileNameRef = f'Schaefer2018/MNI/Centroid_coordinates/Schaefer2018_{numNodes}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
    # inFileNameTarget = 'Glasser360/Glasser360_coords.txt'


    # Read all reference values, i.e., from Yeo's Schaefer2018 RSN labels
    dataRef = readReferenceRSN(inPath+inFileNameRef)
    print(f'we have {len(dataRef)} elements')
    if plotNodes:
        coords_ref = [reg[2] for reg in dataRef]
        plot_point_cloud(coords_ref, title='Ref')

    # read all coords for the target parcellation (here, Glasser360)
    # dataTarget = readDestinationParcellation(inPath+inFileNameTarget)
    coords_target = target_parcellation.get_CoGs()
    print(f'we have {len(coords_target)} elements')
    if plotNodes:
        plot_point_cloud(coords_target, title='Target')

    # First, transfer the closest RSN label to each node of the target parcellation
    labelledTarget = assignRSNLabels(dataRef, coords_target)
    # Second, extract the parcellation info in a formatted list
    formatted_parc = [[pos+1, roi[1], roi[2][0], roi[2][1], roi[2][2]] for pos, roi in enumerate(labelledTarget)]
    # Third, group RSN labels into fewer sets, grouping by RSN name and, if needed, subregion name
    names = collectNamesRSN(formatted_parc, useLR=useLR, detailedRSNs=detailNetworks)
    print(f'Names collected: {list(set(names))}')
    # Fourth, generate the indices
    i = indices4RSNs(names)

    # ------- save!!!
    save_parcellation_info(target_parcellation, formatted_parc)
    save_RSN_labels(target_parcellation, names)
    save_RSN_Indices(target_parcellation, i)


# ==================================================================
# test code: transfer RSNs to the Glasser360 parcellation
# ==================================================================
if __name__ == '__main__':
    import DataLoaders.Parcellations.Glasser379 as Glasser379
    parc = Glasser379.Glasser379(N=360)
    build_RSN_for_parcellation(parc, plotNodes=False)

# ======================================================
# ======================================================
# ======================================================EOF
