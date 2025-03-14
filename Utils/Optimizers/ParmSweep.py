# ==========================================================================
# ==========================================================================
#  Computes a Parameter Sweep
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Inspired from the code (fgain_Neuro.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear
#       functional effects of LSD
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import Utils.decorators as decorators
import time

from Utils.preprocessSignal import processBOLDSignals, processEmpiricalSubjects
verbose = True

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
integrator = None
simulateBOLD = None
sim_inf = 100
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# ==========================================================================
# ==========================================================================
# ==========================================================================
# ---- convenience method, to parallelize the code (someday)
@decorators.loadOrCompute
def evaluateForOneParm(currValue, modelParms, NumSimSubjects,
                       observablesToUse, label):  # observablesToUse is a dictionary of {name: (observable module, apply filters bool)}
    integrator.neuronalModel.setParms(modelParms)
    # integrator.neuronalModel.recompileSignatures()  # just in case the integrator.neuronalModel != neuronalModel here... ;-)
    # integrator.recompileSignatures()

    print(f"   --- BEGIN TIME @ {label}={currValue} ---")
    simulatedBOLDs = {}
    start_time = time.perf_counter()
    for nsub in range(NumSimSubjects):  # trials. Originally it was 20.
        print(f"   Simulating {label}={currValue} -> subject {nsub}/{NumSimSubjects}!!!")
        bds = simulateBOLD.simulateSingleSubject().T
        repetitionsCounter = 0
        while np.isnan(bds).any() or (np.abs(bds) > sim_inf).any():  # This is certainly dangerous, we can have an infinite loop... let's hope not! ;-)
            repetitionsCounter += 1
            print(f"      REPEATING simulation: NaN or inf ({sim_inf}) found!!! (trial: {repetitionsCounter})")
            bds = simulateBOLD.simulateSingleSubject().T
        simulatedBOLDs[nsub] = bds

    dist = processBOLDSignals(simulatedBOLDs, observablesToUse)
    # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
    dist[label] = currValue
    print("   --- TOTAL TIME: {} seconds ---".format(time.perf_counter() - start_time))
    return dist


def distanceForAll_Parms(tc,
                         modelParms, NumSimSubjects,
                         observablesToUse,  # This is a dictionary of {name: (observable module, apply filters bool)}
                         doPreprocessing=True,
                         parmLabel='',
                         outFilePath=None,
                         fileNameSuffix=''):
    print("\n\n*************** Starting: parmSweep.distanceForAll_Parms *****************\n\n")
    NumSubjects = len(tc)
    N = tc[next(iter(tc))].shape[0]  # get the first key to retrieve the value of N = number of areas
    print('tc({} subjects): each entry has N={} regions'.format(NumSubjects, N))

    if doPreprocessing:
        outEmpFileName = outFilePath + '/fNeuro_emp' + fileNameSuffix + '.mat'
        processed = processEmpiricalSubjects(tc, observablesToUse, outEmpFileName)
    else:
        processed = tc
    numParms = len(modelParms)  # len(Parms)

    fitting = {}
    for ds in observablesToUse:
        fitting[ds] = np.zeros((numParms))

    # Model Simulations
    # -----------------
    print('\n\n ====================== Model Simulations ======================\n\n')
    for pos, parm in enumerate(modelParms):  # iteration over the values for G (we in this code)
        # ---- Perform the simulation of NumSimSubjects ----
        outFileNamePattern = outFilePath + '/fitting_'+parmLabel+'{}'+fileNameSuffix+'.mat'
        simMeasures = evaluateForOneParm(parm, modelParms[pos], NumSimSubjects,
                                         observablesToUse, parmLabel,
                                         outFileNamePattern.format(np.round(parm, decimals=3)))

        # ---- and now compute the final FC, FCD, ... distances for this G (we)!!! ----
        print(f"@{parm}:", end='', flush=True)
        for ds in observablesToUse:
            fitting[ds][pos] = observablesToUse[ds][0].distance(simMeasures[ds], processed[ds])
            print(f" {ds}: {fitting[ds][pos]};", end='', flush=True)
        print("\n")

    print("\n\n#####################################################################################################")
    print(f"# Results (in ({Parms[0]}, {Parms[-1]})):")
    for ds in observablesToUse:
        optimValDist = observablesToUse[ds][0].findMinMax(fitting[ds])
        parmPos = [a for a in np.nditer(Parms)][optimValDist[1]]
        print(f"# Optimal {ds} =     {optimValDist[0]} @ {np.round(parmPos, decimals=3)}")
    print("#####################################################################################################\n\n")

    print("DONE!!!")
    return fitting


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
