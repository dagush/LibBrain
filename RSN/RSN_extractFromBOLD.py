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


def extract_subjectfMRI_RSN(data_loader, subject, namesAndIDs):
    all_data = data_loader.get_subjectData(subject)
    fMRI_signal = all_data[subject]['timeseries']
    allNames = list(set(namesAndIDs))  # no repeated entries
    res = {}
    for name in allNames:
        ids = eval(namesAndIDs[name])
        print(f'for {name} we have {len(ids)} regions')
        res[name] = fMRI_signal[ids]
    print(f'We have {sum([res[reg].shape[0] for reg in res])} regions in total.')
    return res


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
    RSNs = transfer.read_RSN_data(target_parc)
    # --------------------------------------------------
    # Process fMRI for all subjects
    # --------------------------------------------------
    res = {}
    for subject in data_loader.get_classification():
        res[subject] = extract_subjectfMRI_RSN(data_loader, subject, RSNs)

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

    if save_result:
        fileName = RSN_save_folder + f'/RSN-{"14" if useLR else "7"}_{group}{fileSufix}.mat'
        saveRSN_Matlab(res, fileName, saveSufix=fileSufix)
        print(f'\nSaved to: {fileName}\n')

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