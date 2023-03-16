###########################################
#                                         #
#    Functions to Streamline g(r) Code    #
#                                         #
###########################################

import numpy as np

# This code has the following modules:
#
# make_vol(spacing, min_edge_size, typ='bins') = norm_edge, inv_norm_vol, size_edge, inv_size_vol, h_bins
# get_cluster_pos(indexes, val, positions) = cpos
# create_cluster(cpos, uc, inv_uc) = new_cpos
# closest_cluster(cpos_a, cpos_b, uc, half_uc, inv_uc) = [min_dist]/
# all_mins(cpos_a, cpos_b, uc, half_uc, inv_uc) = min_dist_list/[]
# get_com_dist(cpos_a, cpos_b, half_ic, inv_uc) = [com_dist]
# frame_grs(pos, frame_pos, norm_edges, size_edges, inv_norm_vol, inv_size_vol, uc, half_uc, inv_uc, 
#           normalise = True, minimum_func='all') = gr_norm_min, gr_norm_com, gr_size_min, gr_size_com


# all_mins computes all minimum image distances between clusters, whereas closest_cluster only 
# computes the minimum of all minimum image distances


def make_vol(spacing, min_edge_size, typ='bins'):

    ''' Takes in the number of histogram bins (default, type="bins") or the
        temporal spacing between bins (type="dist") and the minimum
        edge size to consider, and returns arrays of bin edges and 
        inverse bin volumes for both normalised and absolute coordinates.
        Format: norm_edge, inv_norm_vol, size_edge, inv_size_edge = make_vol'''

    ftp = 4.0*np.pi/3

    if typ=="bins":
        h_bins     = spacing
        size_edges = np.linspace(0, 0.5*min_edge_size, h_bins+1)
    elif typ=="dist":
        size_edges = np.arange(0, 0.5*min_edge_size, spacing)
        if (0.5*min_edge_size > size_edges[-1]):
            size_edges = np.append(size_edges, 0.5*min_edge_size)
        h_bins     = len(size_edges)-1
    else:
        print("ERROR: Incorrect bin creation specified.")
        exit

    size_vol   = np.zeros(h_bins)
    norm_edges = np.linspace(0, 0.5, h_bins+1)
    norm_vol   = np.zeros(h_bins)
        
    for i in range(h_bins):

        norm_out = ftp*(norm_edges[i+1]**3)
        norm_in  = ftp*(norm_edges[i]**3)
        norm_vol[i] = norm_out-norm_in

        size_out = ftp*(size_edges[i+1]**3)
        size_in  = ftp*(size_edges[i]**3)
        size_vol[i] = size_out-size_in
    

    inv_n_vol = 1.0/norm_vol
    inv_s_vol = 1.0/size_vol

    return norm_edges, inv_n_vol, size_edges, inv_s_vol, h_bins


def get_cluster_pos(cluster_indexes, cluster_data, val, positions):

    dat_ids = np.where(cluster_indexes == val)[0]
    dat_ids = sorted(dat_ids)
    dat_ids = np.asarray(dat_ids)
                    
    # dat_ids gives a list of the positions where the atomic indices are
    
    # LAMMPS indexes from 1, python would index from 0
    loc_ids = cluster_data[dat_ids, 0]-1
    loc_ids = np.asarray(loc_ids)
            
    # Create lists of lists
    cpos = positions[loc_ids, :]
    
    # Periodic images dealt with in create_cluster

    return cpos


def create_cluster(cpos, uc, inv_uc, neigh_dist):

    ''' Places all atoms in a cluster, cpos, into a cluster. For ease of calculation atom 
        1 is placed in the image  -7.5, 7.5. Returns these positions'''

    cpos     = np.asarray(cpos)
    new_cpos = np.zeros((np.shape(cpos)))

    allocated = []
    i         = 0

    for j in range(3):
        new_cpos[i, j] = cpos[0, j] - uc*round(cpos[0, j]*inv_uc)
        
    allocated.append(i)
    
    i    += 1
    opos  = 0
    cl_sz = np.shape(cpos)[0]
    
    while (len(allocated)!= cl_sz):
        for a in range(0, cl_sz):
            if (allocated.count(a)==1):
                continue
                
            tmp = cpos[a] - new_cpos[opos]
                
            for j in range(3):
                tmp[j]  = tmp[j] - uc*round(tmp[j]*inv_uc)

            if (np.sqrt(np.dot(tmp, tmp)) <= neigh_dist):
                new_cpos[i] = new_cpos[opos] + tmp
                i          += 1 
                allocated.append(a)
 
        opos += 1 

    return new_cpos


def all_mins(cpos_a, cpos_b, uc, half_uc, inv_uc):

    ''' Find the minimum distances between each image of cluster B, cpos_b,  an cluster A, cpos_a,
        within the arbitrary cut-off of L/2. Returns a list of minimum distances, or None if no
        images are within sufficient range for use of the minimum image convention. '''

    distances = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):

                image_dist = []

                tmp_clus_b = np.zeros(np.shape(cpos_b))
                # Avoid creating pointer
                tmp_clus_b[:,0] = cpos_b[:,0] + i*uc
                tmp_clus_b[:,1] = cpos_b[:,1] + j*uc
                tmp_clus_b[:,2] = cpos_b[:,2] + k*uc

                for a in cpos_a:
                    for b in tmp_clus_b:
                        tmpab = a - b

                        image_dist.append(np.sqrt(np.dot(tmpab, tmpab)))

                if min(image_dist) <= half_uc:
                    # Inside the minimum image range we can work in
                    distances.append(min(image_dist))
            
    return distances


def closest_cluster(cpos_a, cpos_b, uc, half_uc, inv_uc):

    ''' Find the image of cluster B, cpos_b,  which is closest to cluster A, cpos_a.
        Returns the minimum distance, or None if the two clusters are not within
        sufficient range for use of the minimum image convention. '''

    distances = []

    for a in cpos_a:
        for b in cpos_b:
            tmpab = a - b

            for j in range(3):
               # Get minimum image distance vector
               tmpab[j] = tmpab[j] - uc*round(tmpab[j]*inv_uc)

            distances.append(np.sqrt(np.dot(tmpab, tmpab)))

    if min(distances) > half_uc:
        # Outside the minimum image range we can work in
        return []
    else:
        return [min(distances)]


def get_com_dist(cpos_a, cpos_b, uc, half_uc, inv_uc):
    
    ''' Find the CoM-CoM distance between two clusters. '''
    
    coma = np.mean(cpos_a, axis = 0)
    comb = np.mean(cpos_b, axis = 0)

    tmpcom = coma - comb
            
    for j in range(3):
        tmpcom[j] = tmpcom[j] - uc*round(tmpcom[j]*inv_uc)
        
    comdist = np.sqrt(np.dot(tmpcom, tmpcom))

    if comdist < half_uc:
        return [comdist]
    else:
        return []


def frame_grs(pos, frame_pos, norm_edges, size_edges, inv_norm_vol, inv_size_vol, uc, half_uc, inv_uc, normalise = True, minimum_func='all'):
    
    ''' Takes an array of the positions of atoms in each cluster in the form 
        [cluster[atom[coord]]] and finds frame grs for CoM distances and minimum
        distances, for normalised and absolute coordinates. Normalise = False gives PDFs.
        minimum_func controls if all minimum or just minimum_minimum distances are counted
        Format: norm_min, norm_com, size_min, size_com = create_frame_grs '''

    n_bins = len(norm_edges)-1

    frame_norm_com  = np.zeros(n_bins)
    frame_norm_min  = np.zeros(n_bins)
    frame_size_com  = np.zeros(n_bins)
    frame_size_min  = np.zeros(n_bins)
        
    
    # Loop over all clusters
    # ----------------------

    for l in range(len(pos)):
        mins = []
        coms = []
        for k in range(len(pos)):
            if l==k:
                for q in range(len(frame_pos)):
                    # Find the minimum distance
                    if minimum_func is "minimum":
                        mins.extend(closest_cluster(pos[l], frame_pos[q], uc, half_uc, inv_uc))
                    else:
                        mins.extend(all_mins(pos[l], frame_pos[q], uc, half_uc, inv_uc))
                    
                    coms.extend(get_com_dist(pos[l], frame_pos[q], uc, half_uc, inv_uc))
                continue
                    
            # Find the minimum distance
            if minimum_func is "minimum":
                mins.extend(closest_cluster(pos[l], pos[k], uc, half_uc, inv_uc))
            else:
                mins.extend(all_mins(pos[l], pos[k], uc, half_uc, inv_uc))
                    
            coms.extend(get_com_dist(pos[l], pos[k], uc, half_uc, inv_uc))
            
        # Construct histograms
        # --------------------

        size_com, size_edges = np.histogram(coms, bins=size_edges)
        size_min, size_edges = np.histogram(mins, bins=size_edges)

        # Normalise distances

        coms = np.asarray(coms)*1.0/uc
        mins = np.asarray(mins)*1.0/uc

        norm_com, norm_edges = np.histogram(coms, bins=norm_edges)
        norm_min, norm_edges = np.histogram(mins, bins=norm_edges)
        
        
        invlen  = 1/(len(pos)+len(frame_pos)-1)
        density = invlen*uc*uc*uc

        size_com = np.asarray(size_com, dtype=float)
        size_min = np.asarray(size_min, dtype=float)
        norm_com = np.asarray(norm_com, dtype=float)
        norm_min = np.asarray(norm_min, dtype=float)

    
        # Normalise
        # --------
        
        if normalise:
            for i in range(n_bins):
                norm_com[i] = norm_com[i]*inv_norm_vol[i]*invlen
                norm_min[i] = norm_min[i]*inv_norm_vol[i]*invlen

                size_com[i] = size_com[i]*inv_size_vol[i]*density
                size_min[i] = size_min[i]*inv_size_vol[i]*density
    
            
        frame_size_com = np.vstack((frame_size_com, size_com))
        frame_size_min = np.vstack((frame_size_min, size_min))
        frame_norm_com = np.vstack((frame_norm_com, norm_com))
        frame_norm_min = np.vstack((frame_norm_min, norm_min))
    
    # For construction, 0 is 0

    tot_frame_norm_com = np.mean(frame_norm_com[1:], axis=0)
    tot_frame_norm_min = np.mean(frame_norm_min[1:], axis=0)
    tot_frame_size_com = np.mean(frame_size_com[1:], axis=0)
    tot_frame_size_min = np.mean(frame_size_min[1:], axis=0)

    return tot_frame_norm_min, tot_frame_norm_com, tot_frame_size_min, tot_frame_size_com
