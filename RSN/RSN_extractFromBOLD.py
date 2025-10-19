# -------------------------------------------------------------------------
#  Extract RSN-seppartaed BOLD signals from a general BOLD file
#
#  by Gustavo Patow
# -------------------------------------------------------------------------
import numpy as np
import scipy.io as sio

import RSN.RSN_transfer as transfer


# -------------------------------------------------------------------------
# Convenience method to save results in MATLAB format
def saveRSN_Matlab(rsnBOLDs, fileName, saveSufix=''):
    data = {}
    for subject in rsnBOLDs:
        networks = rsnBOLDs[subject]
        proc_rsn = np.array([(n, networks[n]) for n in networks], dtype=object)
        data[saveSufix+subject] = proc_rsn
    sio.savemat(fileName, data)


# -------------------------------------------------------------------------
# RSN grouping code
# -------------------------------------------------------------------------
def simplifyIndicesFile(rsnIndices, subset):
    return {k: rsnIndices[k] for k in subset}


def extract_subjectfMRI_RSN(BOLD, namesAndIDs):
    allNames = list(set(namesAndIDs))  # no repeated entries
    res = {}
    for name in allNames:
        ids = eval(namesAndIDs[name])
        print(f'for {name} we have {len(ids)} regions')
        res[name] = BOLD[ids]
    print(f'We have {sum([res[reg].shape[0] for reg in res])} regions in total.')
    return res


def extract_GroupfMRI_RSN(BOLDs, rsn):
    res = {}
    for s in BOLDs:
        res[s] = extract_subjectfMRI_RSN(BOLDs[s], rsn)
    return res


def process_group_RSN(data_loader, group, rsn, useLR,
                      save_result, sufix, RSN_save_folder):
    # -------- get all fMRIs for a given group
    all_fMRI = data_loader.get_fullGroup_data(group)
    fMRIs = {s: all_fMRI[s]['timeseries'] for s in all_fMRI}
    RSNBOLDs = extract_GroupfMRI_RSN(fMRIs, rsn)
    if save_result:
        fileName = RSN_save_folder + f'/RSN-{"14" if useLR else "7"}_{group}{sufix}.mat'
        saveRSN_Matlab(RSNBOLDs, fileName, saveSufix=sufix)
        print(f'\nSaved to: {fileName}\n')
    return RSNBOLDs


# -------------------------------------------------------------------------
#  Takes a DataLoader and some optional parameters, and group all regional
#  BOLD signals for all the subjects in the groups into their RSNs.
# -------------------------------------------------------------------------
def fromBOLD(data_loader,
             useLR=False,  # True: differentiate between Left and Right hemispheres. False, don't
             save_result=False,  # True: save results to a MATLAB file
             fileSufix='',  # saving sufix (usually, something like 'subj_'
             RSN_save_folder=''  # if we save, where to?
             ):
    # ------ first, make sure the basic RSN information is built for the chosen target parcellation
    target_parc = data_loader.get_parcellation()
    transfer.build_RSN_for_parcellation(target_parc, plotNodes=False)

    # ------- For the full RSN test
    print('-------------- Processing full RSN set')
    rsn = transfer.read_RSN_data(target_parc)
    # --------------------------------------------------
    # Process fMRI for all subjects, in all cohorts
    # --------------------------------------------------
    res = {}
    for group in data_loader.get_groupLabels():
        res[group] = process_group_RSN(data_loader, group, rsn, useLR=useLR,
                                       save_result=save_result, sufix=fileSufix, RSN_save_folder=RSN_save_folder)

    # ------- For the detailed Default Mode Network test
    # do_detailed_DMN_test = False
    # if do_detailed_DMN_test:
    #     print(f'-------------- Processing detail network for {[k for k in detailNetworks.keys()]}')
    #     # ------------ For the RSN analysis, but keeping only the detail on the Default Mode network
    #     rsn = extractRSNFromBOLD.readIndicesFile(indicesFileParcellationRSNDetail)
    #     rsn = extractRSNFromBOLD.simplifyIndicesFile(rsn, detailNetworksFullNames)  # if want to process ONLY the detail info.
    #     fileSufix = '-Detail-Default'
    #     for group in DL.get_groupLabels():
    #         process_group(group, rsn, useLR=useLR, sufix=fileSufix)

    return res


# --------------------------------------------------
# ------------------- Debug code -------------------
# --------------------------------------------------
if __name__ == '__main__':
    import DataLoaders.ADNI_A as ADNI_A
    # Groups: HC, MCI, AD
    # Glasser360 -> DorsAttn: 47, SalVentAttn: 47, Cont: 46, Limbic: 23, Default: 85, SomMot: 55, Vis: 57
    DL = ADNI_A.ADNI_A()
    res = fromBOLD(DL)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------EOF