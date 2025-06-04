# =======================================================================
# Simple example of how a DataLoader should be used...
# =======================================================================
import DataLoaders.ADNI_A as ADNI_A

# =======================================================================
# iterates over ALL subjects, displaying their data
# =======================================================================
def list_subject_data(DL):
    print('listing subject data')
    sujes = DL.get_classification()  # Classification is a dict {subjID: groupLabel}
    for s in sujes:
        print(f'  {s} -> {sujes[s]}')
        data = DL.get_subjectData(s)[s]
        for d in data:
            print(f'    {d} -> {data[d].shape}')

# =======================================================================
# Iterated over the cohort groups, and for each group, lists the subjects
# that belong to that cohort
# =======================================================================
def list_by_group(DL):
    print('listing subjects by group')
    groups = DL.get_groupLabels()
    for g in groups:
        subjectIDs = DL.get_groupSubjects(g)
        print(f'  {g} -> {subjectIDs}')

# =======================================================================
# main method
# =======================================================================
def run():
    DL = ADNI_A.ADNI_A(cutTimeSeries=True)  # some subjects have VERY LONG timeseries!
    list_subject_data(DL)
    list_by_group(DL)


# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    run()
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF