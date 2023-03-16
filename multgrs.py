####################################################
#                                                  #
#    Centre of Mass Analysis - Multiple Cutoffs    #
#                                                  #
####################################################

# Last Edited: 14/12/22
# Author     : K. Blow

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
# This code is written to find the g(r) of a system of finite sized clusters. To use the code, ensure  #
# that a copy is placed into a directory including a ".dcd" file, and one "clusterinf" file for every  #
# frame of the dcd file. It also requires size information                                             #
#                                                                                                      #
# Currently, the code produced the minimum-minimum g(r) and the CoM-CoM g(r) in LJ and normalised      #
# coordinates.                                                                                         #
#                                                                                                      #
# Analysis parameters, and conditions:                                                                 #
#                                                                                                      #
# * GR_CLUSTER is a list containing the minimum size/s a cluster must be to compute its g(r). REQUIRED #
# * MIN_EDGE_SIZE is a decimal stating the minimum edge to consider for the sized g(r). REQUIRED       #
#                                                                                                      # 
# * SPACING, TYP="bins" is an integer of the number of bins the g(r) is discretised to. ALTERNATIVELY  #
# * SPACING, TYP="dist" is the distance between the histogram bins (in real space, same number of      #
#                       bins are used for normalisation. REQUIRED                                      #
#                                                                                                      #
# * DELTA is None, or an integer stating the difference from gr_cluster to consider. If None, all      #
#            clusters over gr_cluster are considered, else gr_cluster to gr_cluster+Delta. OPTIONAL    #
# * MIN_CLUSTER is None, in which case the g(r) is the g(r) of all clusters over gr_cluster.           #
#               Alternatively this can be a list of integers, , or an integer stating that the g(r)    #
#               of gr_cluster clusters should be taken, but all clusters over min_clusters count       #
#               towards this g(r). OPTIONAL                                                            #
#                                                                                                      #
#                                                                                                      #
# Keyword/parameters DELTA and MIN_CLUSTER are incompatible with one another                           #
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#


# Import packages
# ---------------

import numpy as np
from read_dcd import read_dcd 
from collections import Counter
import FuncGrs
from glob import glob

# Analysis parameters
# -------------------

gr_cluster    = [8, 20, 35, 60, 80]   # Count clusters of this size or larger 
min_edge_size = 47.4                  # Give minimum size of simulation cell

#spacing       = 50                    
#typ           = 'bins'                # Number of histogram bins for spatial resolution

spacing       = 0.1
typ           = 'dist'                # Distance between histogram bins for spatial resolution

neigh_dist     = 1.432                # Maximum distance between connected solid atoms

Delta          = None                 # gr_cluster+Delta is allowed
min_cluster    = [1, 8, 20, 35, 60]

top_cluster    = 30                    # Number of clusters to output for analysis

# Get files
# ---------

dcd_file = glob("*.dcd")[0]

files   = glob("*clusterinf*")
num     = [h.split(".")[1] for h in files]
num.sort(key=lambda y: int(y))

split = files[0].split(".")
names = [split[0]+"."+n+"."+split[-1] for n in num]


# Give simulation data
# --------------------

assert((Delta and min_cluster) is None), "Use of Delta and min_cluster is unsupported"

print("Performing g(r) analyisis on clusters in {file}.".format(file=dcd_file))
if Delta is not None:
    print("INFO: Considering g(r) of all clusters of sizes {sizes}  -> {over}.".format(sizes=gr_cluster, over=(np.asarray(gr_cluster)+Delta))) 
else:
    if min_cluster is None:
        print("INFO: Considering g(r) of all clusters of sizes {sizes} or over.".format(sizes=gr_cluster))
        mn_cls = [0]
    else:
        print("INFO: Considering g(r) of all clusters of sizes {sizes} or over with all clusters over {minc}.".format(sizes=gr_cluster, minc=min_cluster))
        mn_cls = sorted(min_cluster)
if typ=='dist':
    print("INFO: Using {bins} bin spacing to bin g(r), and absolute coordinates to a maximum edge size of {mlen}.".format(bins=spacing, mlen=min_edge_size))
else:
    print("INFO: Using {bins} to bin g(r), and absolute coordinates to a maximum edge size of {mlen}.".format(bins=spacing, mlen=min_edge_size))

# CoM analysis tools
# -------------------

N, snapshots, uc_present = read_dcd.read_dcd_header(dcd_file)

edges = []

gr_cluster = sorted(gr_cluster) 

norm_edges, inv_norm_vol, size_edges, inv_size_vol, h_bins = FuncGrs.make_vol(spacing, min_edge_size, typ)

norm_hist_com = np.zeros((h_bins, len(gr_cluster), len(mn_cls)))
norm_hist_min = np.zeros((h_bins, len(gr_cluster), len(mn_cls)))
size_hist_com = np.zeros((h_bins, len(gr_cluster), len(mn_cls)))
size_hist_min = np.zeros((h_bins, len(gr_cluster), len(mn_cls)))

doubles = np.zeros((len(gr_cluster), len(mn_cls)))    

top_len = np.zeros((len(names), top_cluster))
all_szs = Counter() 

max_cluster = 0  # Counter for the largest cluster

# Read files and snapshots
# ------------------------

for snap, filename in enumerate(names):    
    
    read_in = False

    dat  = np.genfromtxt(filename, skip_header=9, dtype=int)
    data = dat[:, 1] # Just cluster number
    
    clus_num = [i for i in data if i!=0] # Count only solid chunks

    sizes = []
    
    for p in range(1, int(np.max(clus_num))+1):
        sizes.append(clus_num.count(p))
        
    sizes = sorted(sizes, reverse=True)
    sizes = np.asarray(sizes)

    all_szs += Counter(sizes)

    max_cluster = sizes[0] if sizes[0] > max_cluster else max_cluster

    try:
        top_len[snap,:] = sizes[:top_cluster] 
    except ValueError:
        top_len[snap,:] = 0
        top_len[snap,:len(sizes)] = sizes


    for counter, cs in enumerate(gr_cluster):
        if sizes[0] < (cs):
            # As cs increases, if there are no clusters over one cs then the
            # frame can be discounted for the rest of the (larger) cluster sizes
            break 
        else:
            test_sz = sizes[:np.where(sizes<cs)[0][0]]
            if Delta is not None:
                if sizes[0] > (cs+Delta):
                    test_sz = test_sz[np.where(sizes>(cs+Delta))[0][-1]+1:]
            if len(test_sz)==0:
                # No clusters of critical nuclear size, or none within Delta limit
                continue
            if len(test_sz)==1 and min_cluster==None:
                # Only one cluster, and only considering clusters both over gr_cluster
                continue
        # Want histograms of CoM-CoM distance, and minimum separation
    
        if min_cluster==None:
            mn_cls = [cs]

        pos = []
        frame_pos = []
        
        for al in range(len(mn_cls)):
            #Avoid pointer creation
            frame_pos.append([])  

        if not read_in:
    
            f = int(num[snap])
            snapshot = f/100 - 130
            snapshot += 1 # Avoid off by one error as FORTRAN indexes from 1 and python from 0

            # Now want the atomic IDs of the atoms in the relevant clusters

            positions,uc_vecs = read_dcd.read_dcd_snapshot(dcd_file,N,snapshot,uc_present)

            # Isotropic pressure, cubic box

            uc      = uc_vecs[0][0]
            half_uc = 0.5*uc
            inv_uc  = 1/uc

            edges.append(uc)
            
            rpositions = positions.transpose()
            read_in    = True

        for q in range(1, int(np.max(clus_num))+1):
            sz = clus_num.count(q)
            if ((Delta is not None) and sz > (cs+Delta)):
                # Outside Delta limit - no other considerations needed
                continue
            if sz < cs:
                # Smaller than value in gr_cluster. Now need to consider if min_cluster counts here.
                # Python doesn't allow a loop of a single variable
                for cc, mc in enumerate(mn_cls):
                    if sz >= mc:
                        # Within the min_cluster range to consider as part of g(r), but don't want to calculate its g(r)
                        loc_pos = FuncGrs.get_cluster_pos(data, dat, q, rpositions)
                        per_pos = FuncGrs.create_cluster(loc_pos, uc, inv_uc, neigh_dist)    
                        frame_pos[cc].append(per_pos)
                continue

                
            loc_pos = FuncGrs.get_cluster_pos(data, dat, q, rpositions)

            per_pos = FuncGrs.create_cluster(loc_pos, uc, inv_uc, neigh_dist)
            pos.append(per_pos)

        # Now have list of positions to compute g(r)s for.
        # Also, for minimum_cluster we have an array called frame_pos which are to be taken into 
        # account, but not computed.
 

        for cc in range(len(mn_cls)):
            if (len(frame_pos[cc])+len(pos)==1):
                continue

            doubles[counter,cc] +=1

            # Compute the grs for this frame
            norm_frame_min, norm_frame_com, \
            size_frame_min, size_frame_com = FuncGrs.frame_grs(pos, frame_pos[cc], norm_edges, size_edges, 
                                                               inv_norm_vol, inv_size_vol, uc, half_uc, 
                                                               inv_uc)
            

            size_hist_com[:,counter,cc] += size_frame_com
            size_hist_min[:,counter,cc] += size_frame_min
            norm_hist_com[:,counter,cc] += norm_frame_com
            norm_hist_min[:,counter,cc] += norm_frame_min                



# Output results
# ---------------

print("INFO: The simulation had a minimum edge length of {medge}, which is {croc} than the specified value of {spec_edge}.".format(medge=min(edges), spec_edge=min_edge_size, croc = ("less" if (min(edges) < min_edge_size) else "greater")))

basename = dcd_file.split(".")[0]
basename = basename.split("_")[1]
basename = "_"+basename+".csv"

for mcc in range(len(mn_cls)):
    if min_cluster is not None:
        bname = "_min"+str(mn_cls[mcc])+basename
    else:
        bname = basename

    data  = size_edges[:-1]
    label = ['SizedEdges']

    for i in range(len(gr_cluster)):
        if doubles[i][mcc] == 0:
            size_tot_com = np.zeros(h_bins)
            size_tot_min = np.zeros(h_bins)
        else:
            size_tot_com = size_hist_com[:,i,mcc]/doubles[i,mcc]
            size_tot_min = size_hist_min[:,i,mcc]/doubles[i,mcc]
    
        data = np.vstack((data, size_tot_com, size_tot_min))

        if Delta is None:
            label.append("SizedCom"+str(gr_cluster[i]))
            label.append("SizedMin"+str(gr_cluster[i]))
        else:
            label.append("SizedCom"+str(gr_cluster[i])+"+"+str(Delta))
            label.append("SizedMin"+str(gr_cluster[i])+"+"+str(Delta))

    data = np.vstack((data, norm_edges[:-1]))
    label.append('NormedEdges')

    for i in range(len(gr_cluster)):
        if doubles[i][mcc] == 0 :
            norm_tot_com = np.zeros(h_bins)
            norm_tot_min = np.zeros(h_bins)
        else:
            norm_tot_com = norm_hist_com[:,i,mcc]/doubles[i,mcc]
            norm_tot_min = norm_hist_min[:,i,mcc]/doubles[i,mcc]

        data = np.vstack((data, norm_tot_com, norm_tot_min))

        if Delta is None:
            label.append("NormedCom"+str(gr_cluster[i]))
            label.append("NormedMin"+str(gr_cluster[i]))
        else:
            label.append("NormedCom"+str(gr_cluster[i])+"+"+str(Delta))
            label.append("NormedMin"+str(gr_cluster[i])+"+"+str(Delta))

    data = np.transpose(data)
    
    outfile = open("grs"+bname, 'w')

    for q in range(len(label)):
        outfile.write(str(label[q]))
        if q == len(label)-1:
            outfile.write('\n')
        else:
            outfile.write(',')
    for line in data:
        for i in range(len(line)):
            outfile.write(str(line[i]))
            if i == len(line)-1:
                outfile.write("\n")
            else:
                outfile.write(", ")
    outfile.close()
    print("INFO: The number of frames with more than one cluster to consider present was {d}.".format(d=doubles))


np.savetxt("top"+str(top_cluster)+basename, top_len, fmt="%i")

sizefile = open("sizes"+basename, 'w')
sizefile.write("Size,Number\n")

for i in range(1, max_cluster+1):
    sizefile.write(str(i)+","+str(all_szs[i])+"\n")
sizefile.close()

