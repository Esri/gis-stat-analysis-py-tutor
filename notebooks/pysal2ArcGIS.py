"""
Author(s): Luc Anselin, Sergio Rey, Xun Li and Mark Janikas
"""

import os as OS
import numpy as NUM
import ErrorUtils as ERROR
import arcpy as ARCPY
import SSDataObject as SSDO
import SSUtilities as UTILS
import WeightsUtilities as WU
import locale as LOCALE
import pysal as PYSAL
import pysal.spreg as SPREG

def swm2Weights(ssdo, swmfile):
    """Converts ArcGIS Sparse Spatial Weights Matrix (*.swm) file to 
    PySAL Sparse Spatial Weights Class.
    
    INPUTS:
    ssdo (class): instance of SSDataObject [1,2]
    swmFile (str): full path to swm file
    
    NOTES:
    (1) Data must already be obtained using ssdo.obtainData()
    (2) The masterField for the swm file and the ssdo object must be
        the same and may NOT be the OID/FID/ObjectID
    """
    neighbors = {}
    weights = {}
    
    #### Create SWM Reader Object ####
    swm = WU.SWMReader(swmfile)
    
    #### SWM May NOT be a Subset of the Data ####
    if ssdo.numObs > swm.numObs:
        ARCPY.AddIDMessage("ERROR", 842, ssdo.numObs, swm.numObs)
        raise SystemExit()

    if swm.masterField != ssdo.masterField:
        ARCPY.AddWarning("ERROR", 938)
        raise SystemExit()
        
    #### Parse All SWM Records ####
    for r in UTILS.ssRange(swm.numObs):
        info = swm.swm.readEntry()
        masterID, nn, nhs, w, sumUnstandard = info
        
        #### Must Have at Least One Neighbor ####
        if nn:
            #### Must be in Selection Set (If Exists) ####
            if masterID in ssdo.master2Order:
                outNHS = []
                outW = []
                
                #### Transform Master ID to Order ID ####
                orderID = ssdo.master2Order[masterID]
                
                #### Neighbors and Weights Adjusted for Selection ####
                for nhInd, nhVal in enumerate(nhs):
                    try:
                        nhOrder = ssdo.master2Order[nhVal]
                        outNHS.append(nhOrder)
                        weightVal = w[nhInd]
                        if swm.rowStandard:
                            weightVal = weightVal * sumUnstandard[0]
                        outW.append(weightVal)
                    except KeyError:
                        pass
                
                #### Add Selected Neighbors/Weights ####
                if len(outNHS):
                    neighbors[orderID] = outNHS
                    weights[orderID] = outW
    swm.close()
    
    #### Construct PySAL Spatial Weights and Standardize as per SWM ####
    w = PYSAL.W(neighbors, weights)
    if swm.rowStandard:
        w.transform = 'R'
        
    return w

def poly2Weights(ssdo, contiguityType = "ROOK", rowStandard = True):
    """Uses GP Polygon Neighbor Tool to construct contiguity relationships
    and stores them in PySAL Sparse Spatial Weights class.
    
    INPUTS:
    ssdo (class): instance of SSDataObject [1]
    contiguityType {str, ROOK}: ROOK or QUEEN contiguity
    rowStandard {bool, True}: whether to row standardize the spatial weights
    
    NOTES:
    (1) Data must already be obtained using ssdo.obtainData() or ssdo.obtainDataGA ()
    """
    
    neighbors = {}
    weights = {}
    polyNeighDict = WU.polygonNeighborDict(ssdo.inputFC, ssdo.masterField,
                                           contiguityType = contiguityType)
    
    for masterID, neighIDs in UTILS.iteritems(polyNeighDict):
        orderID = ssdo.master2Order[masterID]
        neighbors[orderID] = [ssdo.master2Order[i] for i in neighIDs]

    w = PYSAL.W(neighbors)
    if rowStandard:
        w.transform = 'R'
    return w

def distance2Weights(ssdo, neighborType = 1, distanceBand = 0.0, numNeighs = 0,
                     distanceType = "euclidean", exponent = 1.0, rowStandard = True,
                     includeSelf = False):
    """Uses ArcGIS Neighborhood Searching Structure to create a PySAL Sparse Spatial 
    Weights Matrix.
    
    INPUTS:
    ssdo (class): instance of SSDataObject [1]
    neighborType {int, 1}: 0 = inverse distance, 1 = fixed distance, 
                           2 = k-nearest-neighbors, 3 = delaunay
    distanceBand {float, 0.0}: return all neighbors within this distance for 
                               inverse/fixed distance
    numNeighs {int, 0}: number of neighbors for k-nearest-neighbor, can also 
                        be used to set a minimum number of neighbors for 
                        inverse/fixed distance
    distanceType {str, euclidean}: manhattan or euclidean distance [2]  
    exponent {float, 1.0}: distance decay factor for inverse distance
    rowStandard {bool, True}: whether to row standardize the spatial weights
    includeSelf {bool, False}: whether to return self as a neighbor
    
    NOTES:
    (1) Data must already be obtained using ssdo.obtainDataGA()
    (2) Chordal Distance is used for GCS Data
    """
    
    neighbors = {}
    weights = {}
    gaSearch = GAPY.ga_nsearch(ssdo.gaTable)
    if neighborType == 3:
        gaSearch.init_delaunay()
        neighSearch = ARC._ss.NeighborWeights(ssdo.gaTable, gaSearch, weight_type = 1)
    else:
        if neighborType == 2:
            distanceBand = 0.0
            weightType = 1
        else:
            weightType = neighborType
        
        concept, gaConcept = WU.validateDistanceMethod(distanceType.upper(), ssdo.spatialRef)
        gaSearch.init_nearest(distanceBand, numNeighs, gaConcept)
        neighSearch = ARC._ss.NeighborWeights(ssdo.gaTable, gaSearch, weight_type = weightType, 
                                              exponent = exponent, include_self = includeSelf)
        
    for i in range(len(neighSearch)):
        neighOrderIDs, neighWeights = neighSearch[i]
        neighbors[i] = neighOrderIDs
        weights[i] = neighWeights
        
    w = PYSAL.W(neighbors, weights)
    if rowStandard:
        w.transform = 'R'
    return w 

def lmChoice(result, criticalValue):
    """Makes choice of aspatial/spatial model based on LeGrange Multiplier
    stats from an OLS result.

    INPUTS:
    result (object): instance of PySAL OLS Model with spatial weights given.
    criticalValue (float): significance value

    RETURN:
    category (str): ['MIXED', 'LAG', 'ERROR', 'OLS']
    """

    sigError = result.lm_error[1] < criticalValue
    sigLag = result.lm_lag[1] < criticalValue
    sigBoth = sigError and sigLag
    if sigLag or sigError:
        sigErrorRob = result.rlm_error[1] < criticalValue
        sigLagRob = result.rlm_lag[1] < criticalValue
        sigBothRob = sigErrorRob and sigLagRob
        if sigBothRob:
            return "MIXED"
        else:
            if sigLagRob:
                return "LAG"
            if sigErrorRob:
                return "ERROR"
            if sigBoth:
                return "MIXED"
            else:
                if sigLag:
                    return "LAG"
                return "ERROR"
    else:
        return "OLS"

def autospace(y,x,w,gwk,opvalue=0.01,combo=False,name_y=None,name_x=None,
              name_w=None,name_gwk=None,name_ds=None):
    """
    Runs automatic spatial regression using decision tree
    
    Accounts for both heteroskedasticity and spatial autocorrelation
    
    No endogenous variables
    
    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object 
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    opvalue      : real
                   p-value to be used in tests; default: opvalue = 0.01
    combo        : boolean
                   flag for use of combo model rather than HAC for lag-error
                   model; default: combo = False
                   
    Returns
    -------
    results      : a dictionary with
                   results['final model']: one of
                        No Space - Homoskedastic
                        No Space - Heteroskedastic
                        Spatial Lag - Homoskedastic
                        Spatial Lag - Heteroskedastic
                        Spatial Error - Homoskedastic
                        Spatial Error - Heteroskedastic
                        Spatial Lag with Spatial Error - HAC
                        Spatial Lag with Spatial Error - Homoskedastic
                        Spatial Lag with Spatial Error - Heteroskedastic
                        Robust Tests not Significant - Check Model
                   results['heteroskedasticity']: True or False
                   results['spatial lag']: True or False
                   results['spatial error']: True or False
                   results['regression1']: regression object with base model (OLS)
                   results['regression2']: regression object with final model
    """
    results = {}
    results['spatial error']=False
    results['spatial lag']=False
    r1 = SPREG.ols.OLS(y,x,w=w,gwk=gwk,spat_diag=True,
                       name_y=name_y,name_x=name_x,
                       name_w=name_w,name_gwk=name_gwk,
                       name_ds=name_ds)
    results['regression1'] = r1
    Het = r1.koenker_bassett['pvalue']
    if Het < opvalue:
        Hetflag = True
    else:
        Hetflag = False
    results['heteroskedasticity'] = Hetflag
    model = lmChoice(r1, opvalue)
    if model == "MIXED":
        if not combo:
            r2 = SPREG.twosls_sp.GM_Lag(y,x,w=w,gwk=gwk,robust='hac',name_y=name_y,
                                        name_x=name_x,name_w=name_w,name_gwk=name_gwk,
                                        name_ds=name_ds)
            results['final model']="Spatial Lag with Spatial Error - HAC"
        elif Hetflag:
            r2 = SPREG.error_sp_het.GM_Combo_Het(y,x,w=w,name_y=name_y,name_x=name_x,
                                                 name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Lag with Spatial Error - Heteroskedastic"
        else:
            r2 = SPREG.error_sp_hom.GM_Combo_Hom(y,x,w=w,name_y=name_y,name_x=name_x,
                                                 name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Lag with Spatial Error - Homoskedastic"
    elif model == "ERROR":
        results['spatial error']=True
        if Hetflag:
            r2 = SPREG.error_sp_het.GM_Error_Het(y,x,w,name_y=name_y,name_x=name_x,
                                                 name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Error - Heteroskedastic"
        else:
            r2 =SPREG.error_sp_hom.GM_Error_Hom(y,x,w,name_y=name_y,name_x=name_x,
                                                name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Error - Homoskedastic"
    elif model == "LAG":
        results['spatial lag']=True
        if Hetflag:
            r2 = SPREG.twosls_sp.GM_Lag(y,x,w=w,robust='white',
                                        name_y=name_y,name_x=name_x,
                                        name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Lag - Heteroskedastic"
        else:
            r2 = SPREG.twosls_sp.GM_Lag(y,x,w=w,name_y=name_y,name_x=name_x,
                                        name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Lag - Homoskedastic"
    else:
        if Hetflag:
            r2 = SPREG.ols.OLS(y,x,robust='white',name_y=name_y,name_x=name_x,
                               name_ds=name_ds)
            results['final model']="No Space - Heteroskedastic"
        else:
            r2 = r1
            results['final model']="No Space - Homoskedastic"
    results['regression2'] = r2

    return results
