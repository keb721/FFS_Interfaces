################################
#                              #
#   FLUX ANALYSIS FUNCTIONS    #
#                              #
################################

# Import packages
# ---------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter

# Get the points of crossing in a positive direction
# --------------------------------------------------

def get_ptv_crossing(cluster, boundary, dt, lA=None):
    ''' Get the times of occurence of positive crossings of a cluster across a boundary with a timestep dt between outputs.'''

    # Determine crossings
    # -------------------
    
    if lA is None:
        # Find all occurences of clusters larger than critical
        wh = np.where(cluster > boundary)[0]
        
        # If the previous index is not in wh, then it has crossed the boundary in the positive direction
        crossings = [i for i in wh if (i-1) not in wh]   
        
    else:
        # Need to separate into crossings of lA and l0
        wh_l0 = np.where(cluster > boundary)[0]
        wh_lA = np.where(cluster < lA)[0]
        
        crossings_l0 = [i for i in wh_l0 if (i-1) not in wh_l0]        # Entering l0
        crossings_lA = [i for i in wh_lA if (i+1) not in wh_lA]        # Leaving  lA

        crossings = []
        
        if len(crossings_l0)!=0:
            crossings = [crossings_l0[0]]

            for i in range(1, len(crossings_l0)):
                last_out = np.where(crossings_lA <= crossings_l0[i])[0]
                if len(last_out)!=0:
                    if crossings_lA[last_out[-1]] > crossings[-1]:
                        crossings.append(crossings_l0[i])
                        
    times = np.asarray(crossings)*dt
    
    return times


def get_ntv_crossing(cluster, boundary, dt):
    ''' Get the times of occurence of negative crossings of a cluster across a boundary with a timestep dt between outputs.'''

    # Determine crossings
    # -------------------
    
    # Find all occurences of clusters larger than critical
    wh = np.where(cluster > boundary)[0]
        
    # If the subsequent index is not in wh, then it has crossed the boundary in the negative direction
    crossings = [i for i in wh if (i+1) not in wh]   
    times = np.asarray(crossings)*dt

    return times



# Get fluxes from a single run
# ----------------------------

def get_fluxes(clus, vol, dt, boundary, lA=None):
    ''' Take in the time series of the size(s) of the cluster (clus), simulation box volume (vol), the time separation between successive
        snapshots (dt) and the cluster size(s) of interest (boundary) and determine the fluxs.'''

    bnds     = len(boundary)
    try:
        clusters = len(clus[0])
        fluxes   = np.zeros((clusters, bnds))
    
        for cl in range(clusters):
            for b in range(bnds):

                c      = clus[:, cl]
            
                times  = get_ptv_crossing(c, boundary[b], dt, lA)
                
                flux = len(times) 
                flux = flux/(len(c)*dt)
                flux = flux/vol  # per unit volume
        
                fluxes[cl][b] = flux

    except TypeError:
        # Single cluster
        fluxes   = np.zeros((1, bnds))
    
        for b in range(bnds):

            c      = clus
            
            times  = get_ptv_crossing(c, boundary[b], dt, lA)
                
            flux = len(times) 
            flux = flux/(len(c)*dt)
            flux = flux/vol  # per unit volume
        
            fluxes[0][b] = flux


    return fluxes

# Get average fluxes from all runs
# --------------------------------

def get_average_fluxes(fluxes):
    ''' Take in the fluxes of multiple simulations and return averages and errors.'''

    runs, clusters, bnds = np.shape(fluxes)

    av_fluxes = np.zeros((clusters, bnds))
    eb_fluxes = np.zeros((clusters, bnds))

    tot_av = np.zeros(bnds)
    tot_eb = np.zeros(bnds)

    for b in range(bnds):
        tot_fluxes = np.sum(fluxes[:, :, b], axis=1)
        tot_av[b]  = np.mean(tot_fluxes)
        tot_eb[b]  = np.std(tot_fluxes)/np.sqrt(runs)

        for c in range(clusters):
            av_fluxes[c, b] = np.mean(fluxes[:,c,b])
            eb_fluxes[c, b] = np.std(fluxes[:,c,b])/np.sqrt(runs)      
        
    return av_fluxes, eb_fluxes, tot_av, tot_eb


def get_grs_data(bn, Delta=False):
    '''Generate data from a series of csv files with basename bn, wherein several cluster sizes may be considered. Cluster sizes are given in the header. Delta is
       whether there is a Delta cluster size.
       NOTE: Depends heavily on using consistent keywords and naming with the g(r) analysis script.'''
    
    files = glob(bn+"*csv")
    bn    = bn.split("/")[-1]
    size  = bn.split("_")[0]
    if not Delta:
        mins  = bn.split("_")[2]
    
        if "min" not in mins:
            # This is from a fluctuating, not static minimum run
            # where only sizes above cluster size are considered.
            # NOTE: If min is above the cluster size, the cluster
            #       size is the de facto min
        
            mins = "minNn"
    else:
        dlta = bn.split("_")[2]

    data0   = pd.read_csv(files[0])
    headers = data0.columns.values

    cluster_sizes = []
    results       = pd.DataFrame()
    sqrtN         = 1.0/np.sqrt(len(files))

    for h in headers:
        if "Edge" in h:
            if not Delta:
                results[size+h+"_"+mins] = data0[h]
            else:
                results[size+h+"_"+dlta] = data0[h]
        else:
            if "SizedCom" in h:
                if not Delta:
                    cluster_sizes.append(int(h.split("Com")[-1]))
                    hr = h
                else:
                    cluster_sizes.append(int((h.split("Com")[-1]).split("+")[0]))
                    hr = h.split("+")[0]
                    
            else:
                hr = h.split("+")[0] if Delta else h
                
            value = np.zeros(len(data0[h]))

            for f in files:
                data  = pd.read_csv(f)
                value = np.vstack((value, data[h]))

            value = value[1:]     # Remove initial 0s 
            mean  = np.mean(value, axis=0)
            error = np.std(value, axis=0)*sqrtN
            
            if Delta:
                results[size+"Mean"+hr+"_"+dlta]  = mean
                results[size+"Error"+hr+"_"+dlta] = error
            else:
                results[size+"Mean"+hr+"_"+mins]  = mean
                results[size+"Error"+hr+"_"+mins] = error

    return results, cluster_sizes


def plottable_grs(results, keyword, pick_diff, Delta=False, pick_sims=None, pick_clusters=None):
    ''' Return a list of the results to plot. Keyword determimes ['Sized'/'Normed', 'Com'/'Min']. Diff gives EITHER
        the minimum cluster size to be considered (Delta=False, default), or the delta value to consider (Delta=True).
        Pick_sims and pick_clusters are optional tags to select only certain cluster sizes and/or simulation sizes.
        NOTE: If min is "Nn" then minimum cluster size is pick_clusters.'''

    headers = results.columns.values

    plt_data = []
    
    pick_mins = []
    for i in pick_diff:
        x = "_D"+str(i) if Delta else "_min"+str(i) 
        pick_mins.append(x)

    if pick_sims is None and pick_clusters is None:
        Data = [col for col in results.columns if keyword[0] in col]
        Data = [col for col in Data if keyword[1] in col]
        for mn in pick_mins:
            # Want all data fulfilling keywords and minimum sizes
            mn_Data = [col for col in Data if mn in col]
            plt_data.append(mn_Data)

    elif pick_clusters is None:
        # Data fulfilling simulation sizes and keywords
        for sm in pick_sims:
            sm_Data = [col for col in results.columns if str(sm) in col]
            sm_Data = [col for col in sm_Data if keyword[0] in col]
            sm_Data = [col for col in sm_Data if keyword[1] in col]
            for mn in pick_mins:
                mn_Data = [col for col in sm_Data if mn in col]
                plt_data.append(mn_Data)

    elif pick_sims is None:
        # Data fulfilling cluster sizes and keywords
        for sz in pick_clusters:
            sz_Data = [col for col in results.columns if keyword[1]+str(sz)+"_" in col]
            sz_Data = [col for col in sz_Data if keyword[0] in col]
            for mn in pick_mins:
                mn_Data = [col for col in sz_Data if mn in col]
                plt_data.append(mn_Data)
    else:
        # Data fulfilling cluster sizes and keywords
        for sz in pick_clusters:
            sz_Data = [col for col in results.columns if keyword[1]+str(sz)+"_" in col]
            sz_Data = [col for col in sz_Data if keyword[0] in col]
            for sm in pick_sims:
                sm_Data = [col for col in sz_Data if sm in col]   # Ignore issues with repeated digits
                for mn in pick_mins:
                    mn_Data = [col for col in sm_Data if mn in col]
                    plt_data.append(mn_Data)

    plt_data = np.asarray(plt_data).flatten()

    return plt_data


def sizes_data(sizes_file):
    ''' Return the count of sizes of solid clusters in a simulation.'''
    
    data = np.genfromtxt(sizes_file, delimiter=',', skip_header=1)
    data = data.tolist()
    nd   = []

    for item in data:
        item[0] = int(item[0])
        item[1] = int(item[1])
        nd.append(tuple(item))
        
    c = Counter(dict(nd))
    
    return c
    
def get_hist(clusters, volume, norm=True):
    ''' Return the weights of the number of (a counter of) clusters, at each relevant cluster size. Norm=True normalises the weights by simulation volume'''
    
    top       = int(max(clusters))
    histogram = np.zeros(top)
       
    # Construct histogram
    # -------------------

    for q in range(top):            
        histogram[q] += clusters[q+1]

    if norm:
        histogram = histogram*1.0/volume    # Normalise

    return histogram
