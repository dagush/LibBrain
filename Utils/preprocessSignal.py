# ================================================================================================================
# Function to preprocess empirical signals for an optimization stage
# ================================================================================================================
import Utils.decorators as decorators
# import time


def processBOLDSignals(bold_signals, observables, bpf, verbose=True):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    num_subjects = len(bold_signals)
    # get the first key to retrieve the value of N = number of areas
    n_rois = bold_signals[next(iter(bold_signals))].shape[0]

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    for ds, (_, accumulator, _) in observables.items():
        measureValues[ds] = accumulator.init(num_subjects, n_rois)

    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        if verbose:
            print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos + 1, num_subjects, s, bold_signals[s].shape[0],
                                                                    bold_signals[s].shape[1]), flush=True)
        # BOLD signals from file have inverse shape
        signal = bold_signals[s].T  # LR_version_symm(tc[s])

        signal_filt = bpf.filter(signal)
        for ds, (observable, accumulator, _) in observables.items():
            procSignal = observable.from_fmri(signal_filt)
            measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, procSignal[ds])

    for ds, (observable, accumulator, _) in observables.items():
        measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


# ============== a practical way to save recomputing necessary (but lengthy) results ==========
@decorators.loadOrCompute
def processEmpiricalSubjects(bold_signals, observables, bpf, verbose=True):
    return processBOLDSignals(bold_signals, observables, bpf, verbose=verbose)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
