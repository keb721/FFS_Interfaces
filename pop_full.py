#########################
#                       #
#   CLUSTER ANALYSIS    #
#                       #
#########################

# Import packages
# ---------------

import numpy as np
import pandas as pd
import FuncPop
import warnings
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from matplotlib import rc
from collections import Counter


rc('text', usetex=True)
parser = argparse.ArgumentParser()

parser.add_argument('-r', '--runs',  help = "Number of statistically independent runs. Default = 24", type = int)
parser.add_argument('-A', '--lA',    help = "Clusters must return to this cluster size before a new boundary crossing. Default = value of boundary", type = float)
parser.add_argument('-d', '--bound', help = "Discard clusters under this size for PDFs. Default = 1", type = int)
parser.add_argument('-l', '--lbins', help = "Number of bins for lag time distribution plot. Default = 50", type=int)
parser.add_argument('-p', '--pbins', help = "Number of bins for PDFs. Default = 20", type=int)

args = parser.parse_args()

# Analysis parameters
# -------------------

discard = 1  if args.bound is None else args.bound
runs    = 24 if args.runs  is None else args.runs
lbins   = 50 if args.lbins is None else args.lbins
pbins   = 20 if args.pbins is None else args.pbins

lA = args.lA

simulation_sizes = ["4k", "5k", "7k", "11k", "16k", "32k", "1e5"]


bounds = np.linspace(20.5, 60.5, 21)

temporal_bounds = np.array([20.5, 34.5, 48.5, 60.5])    # Subplots at each of these 4 l0 values

gr_mins  = [1, 8, 20, 35, 60]    # Load data at each of these minimum solid sizes


# Spatial analysis parameters
# ---------------------------

spt_keys = ["Sized", "Min"]  # Keywords for analysis - Sized/Normed, Com/Min
pick_mns = [1]               # Minimum cluster sizes to consider
pick_sms = None              # OPTIONAL: None, or list of simulations to consider
pick_szs = None              # OPTIONAL: None, or list of cluster sizes to consider
liquid   = False             # Plot liquid g(r) 
solid    = False              # Plot solid g(r)
 

volume_file = "_vol.txt"
sizes_file  = "_sizes_"
liquid_grs  = "liquid_grs.txt"
solid_grs   = "solid_grs.txt"

timestep  = 0.002
dump_step = 100

dt = timestep*dump_step

# Dictionary of simulation sizes and properties
# ---------------------------------------------

simulation_atoms   = {"4k"  : 4000,
                      "5k"  : 5324,
                      "7k"  : 6912,
                      "11k" : 10967,
                      "16k" : 16384,
                      "32k" : 32000,
                      "1e5" : 108000}

simulation_clusfs  = {"4k"  : "data/4k_top6_",
                      "5k"  : "data/5k_top8_",
                      "7k"  : "data/7k_top10_",
                      "11k" : "data/11k_top12_",
                      "16k" : "data/16k_top16_",
                      "32k" : "data/32k_top20_",
                      "1e5" : "data/1e5_top30_"}



simulation_cols    = {"4k"  : (1, 0, 0),        
                      "5k"  : (0.84, 0, 0.16),  
                      "7k"  : (0.68, 0, 0.32),  
                      "11k" : (0.5, 0, 0.5),    
                      "16k" : (0.32, 0, 0.68),  
                      "32k" : (0.16, 0, 0.84),  
                      "1e5" : (0, 0, 1)}        



simulation_labs    = {"4k"  : "4,000 atoms",
                      "5k"  : "5,324 atoms",
                      "7k"  : "6,912 atoms",
                      "11k" : "10,967 atoms",
                      "16k" : "16,384 atoms",
                      "32k" : "32,000 atoms",
                      "1e5" : "108,000 atoms"}

simulation_data    = {} 


# =============== #
# Generate graphs # 
# =============== #

lAs = [11.5, 12.5, 14.5, 16.5, 19.5, 24.5, 34.5]

# Crossings of l0
# ---------------

figA, axA = plt.subplots(2, 2, figsize=(12.8, 9.8), sharex='all', sharey='all')
figA.subplots_adjust(left=0.115, right=0.9975, bottom=0.06, top=0.9975, wspace=0, hspace=0)  #0.06 works well for 12.8, 12.8

for j, s in enumerate(simulation_sizes):
    pre = "data/"+s
    vols = np.genfromtxt(pre+volume_file)

    all_fluxes = []
    lA_fluxes  = []

    lAbounds = bounds
    if j==len(simulation_sizes)-2:
        lAbounds = np.linspace(26.5, 60.5, 18)
    if j==len(simulation_sizes)-1:
        lAbounds = np.linspace(36.5, 60.5, 13)


    for i in range(runs):
        try:
            clusters = np.genfromtxt(simulation_clusfs[s]+str(i+1)+".csv", dtype=int)
            flux     = FuncPop.get_fluxes(clusters, vols[i], dt, bounds)
            all_fluxes.append(flux)
            flux     = FuncPop.get_fluxes(clusters, vols[i], dt, lAbounds, lA=lAs[j])
            lA_fluxes.append(flux)

        except OSError:
            print("#INFO: File ", simulation_clusfs[s], str(i+1), ".csv not found.")

        if (np.max(clusters[:, -1]) >= bounds[0]):
            print("#INFO: File ", simulation_clusfs[s], str(i+1), ".csv has a minimum cluster size over a boundary.")

    av_fluxes, eb_fluxes, tot_av, tot_eb = FuncPop.get_average_fluxes(np.array(all_fluxes))

    simulation_data[s+"_avfluxes"] = tot_av
    simulation_data[s+"_avflxerr"] = tot_eb
    simulation_data[s+"_1fluxes"]  = av_fluxes[0]
    simulation_data[s+"_1flxerr"]  = eb_fluxes[0]


    av_fluxlA, eb_fluxlA, tot_avlA, tot_eblA = FuncPop.get_average_fluxes(np.array(lA_fluxes))

    simulation_data[s+"_avfluxlA"] = tot_avlA
    simulation_data[s+"_avflxerrlA"] = tot_eblA
    simulation_data[s+"_1fluxlA"]  = av_fluxlA[0]
    simulation_data[s+"_1flxerrlA"]  = eb_fluxlA[0]

    # Plot total and primary fluxes

    axA[0][0].plot(bounds, tot_av, color=simulation_cols[s], linewidth=3.5)
    axA[0][0].fill_between(bounds, tot_av-tot_eb, tot_av+tot_eb, color=simulation_cols[s], alpha=0.5)   
    axA[0][1].plot(bounds, av_fluxes[0], color=simulation_cols[s], linewidth=3.5)
    axA[0][1].fill_between(bounds, av_fluxes[0]-eb_fluxes[0], av_fluxes[0]+eb_fluxes[0], color=simulation_cols[s], alpha=0.5)
    axA[1][0].plot(lAbounds, tot_avlA, color=simulation_cols[s], linewidth=3.5)
    axA[1][0].fill_between(lAbounds, tot_avlA-tot_eblA, tot_avlA+tot_eblA, color=simulation_cols[s], alpha=0.5)   
    axA[1][1].plot(lAbounds, av_fluxlA[0], color=simulation_cols[s], linewidth=3.5, label=simulation_labs[s])
    axA[1][1].fill_between(lAbounds, av_fluxlA[0]-eb_fluxlA[0], av_fluxlA[0]+eb_fluxlA[0], color=simulation_cols[s], alpha=0.5)

axA[0][0].set_ylabel(r'$\Phi_{0}$', size=20)      
axA[0][0].set_yticks([0.0, 0.5e-4, 1e-4, 1.5e-4])
axA[0][0].set_yticklabels(['$0.0$', r'$0.5 \times 10^{-4}$', r'$1 \times 10^{-4}$', r'$1.5 \times 10^{-4}$'])
axA[0][0].tick_params(labelsize=15)
axA[0][0].tick_params(axis='x', which='both', direction='in')
axA[0][0].text(0.925, 0.925, '(a)', size=20, transform=axA[0][0].transAxes)
axA[0][1].tick_params(axis='x', which='both', direction='in')
axA[0][1].tick_params(axis='y', which='both', direction='in')
axA[0][1].text(0.925, 0.925, '(b)', size=20, transform=axA[0][1].transAxes)
axA[1][0].set_ylabel(r'$\Phi_{0|\lambda_A}$', size=20)
axA[1][0].tick_params(labelsize=15)
axA[1][0].set_xlabel(r'$\lambda_{0}$', size=20)      
axA[1][0].text(0.925, 0.925, '(c)', size=20, transform=axA[1][0].transAxes)
axA[1][1].set_xlabel(r'$\lambda_0$', size=20)      
axA[1][1].tick_params(axis='y', which='both', direction='in')
axA[1][1].tick_params(labelsize=15)
axA[1][1].text(0.61, 0.925, '(d)', size=20, transform=axA[1][1].transAxes)

axA[1][1].legend(prop={'size':13})

#figA.savefig("Flux_lAvl0_allvpri.png", dpi=300)

plt.show()

# Crossings as f(time)
# --------------------

fig, ax = plt.subplots(1, 1, figsize=(6.3,4.8))
fig.subplots_adjust(left=0.15, right=0.995, bottom=0.15, top=0.995, wspace=0, hspace=0)

boundary = 36.5
lAs = [11.5, 12.5, 14.5, 16.5, 19.5, 24.5, 34.5]

for c, s in enumerate(simulation_sizes):
    pre = "data/"+s

    for i in range(runs):
        times = []
        try:
            clusters = np.genfromtxt(simulation_clusfs[s]+str(i+1)+".csv", dtype=int)
            for cl in range(np.shape(clusters)[1]):
                times.extend(FuncPop.get_ptv_crossing(clusters[:, cl], boundary, dt, lA=lAs[c]))
            
            count = np.linspace(0, len(times), len(times)+1)
            count = np.append(count, len(times))

            times.append(0)
            times.append(dt*(len(clusters)-1))
            times = sorted(times)
            
            if i==0:
                ax.plot(times, count, label=simulation_labs[s], color=simulation_cols[s], alpha=0.25, linewidth=3.5)
            else:
                ax.plot(times, count, color=simulation_cols[s], alpha=0.25, linewidth=3.5)
        except OSError:
            print("#INFO: File ", simulation_clusfs[s], str(i+1), ".csv not found.")


ax.set_xlabel(r'$t*$', size=20)
ax.set_ylabel('Crossings', size=20)
ax.tick_params(labelsize=15)
ax.legend(prop={'size':10})

#fig.savefig("Num_crossings_365.png", dpi=300)

plt.show()


# Spatial correlations
# --------------------

results = pd.DataFrame()

print("#INFO: Assuming the use of the same cluster sizes for all simulations")

pick_mns = [20, 35, 60, 80]               # Minimum cluster sizes to consider
pick_sms = None
Delta    = False

for s in simulation_sizes:
    for mn in pick_mns:
        curr, cs = FuncPop.get_grs_data("grs_data/"+s+"_grs_min"+str(mn)+"_")  # Trailing underscore needed for e.g. 1 vs 10
        results = pd.concat([results, curr], axis=1, sort=True)



fig, axs = plt.subplots(2, 2, figsize=(12.8,9.6), sharex='all')
fig.subplots_adjust(left=0.07, right=0.9975, bottom=0.07, top=0.995, wspace=0, hspace=0)
ax = axs.flatten()

for j, mn in enumerate(pick_mns):

    plot_data = FuncPop.plottable_grs(results, spt_keys, [mn], Delta, pick_sms, [mn])

    for ln in plot_data:
        if "Mean" in ln:
            size = ln.split("Mean")[0]
            mns  = ln.split("_")[-1]
            clst = ln.split("_")[0].split(spt_keys[1])[-1]
            err  = results[size+"Error"+spt_keys[0]+spt_keys[1]+clst+"_"+mns]
            try:
                edge = results[size+spt_keys[0]+"Edges"+str(clst)+"_"+mns]
            except KeyError:
                edge = results[size+spt_keys[0]+"Edges_"+mns]

            edge = edge[~np.isnan(edge)]
            data = results[ln]
            data = data[~np.isnan(data)]
            err  = err[~np.isnan(err)]

            ax[j].plot(edge, data, label=simulation_labs[size], color=simulation_cols[size], linewidth=3.5)#label=spt_keys[1]+size+clst+" "+mns)
            ax[j].fill_between(edge, data-err, data+err, alpha=0.25, color=simulation_cols[size])



axs[0][0].set_ylabel(r'$g(r)$', size=20)
axs[0][0].tick_params(labelsize=15)
axs[0][0].tick_params(axis='x', which='both', direction='in')
axs[0][0].text(0.925, 0.925, '(a)', size=20, transform=axs[0][0].transAxes)
axs[0][1].tick_params(axis='x', which='both', direction='in')
axs[0][1].tick_params(axis='y', which='both', direction='in')
axs[0][1].text(0.925, 0.925, '(b)', size=20, transform=axs[0][1].transAxes)
axs[1][0].set_ylabel(r'$g(r)$', size=20)
axs[1][0].tick_params(labelsize=15)
axs[1][0].set_xlabel(r'$r/\sigma$', size=20)
axs[1][0].text(0.925, 0.925, '(c)', size=20, transform=axs[1][0].transAxes)
axs[1][1].set_xlabel(r'$r/\sigma$', size=20)
axs[1][1].tick_params(axis='y', which='both', direction='in')
axs[1][1].tick_params(labelsize=15)
axs[1][1].text(0.625, 0.925, '(d)', size=20, transform=axs[1][1].transAxes)
axs[1][1].legend(prop={'size':13})


axs[0][1].tick_params(labelsize=15)
axs[0][1].tick_params(axis="y",direction="in", pad=-7.5)
axs[1][1].tick_params(axis="y",direction="in", pad=-7.5)

plt.setp(axs[1][1].yaxis.get_majorticklabels(), ha="left" )
plt.setp(axs[0][1].yaxis.get_majorticklabels(), ha="left" )

#plt.savefig("grs_mn_w_mn.png", dpi=300)

plt.show()

# Total cluster populations
# -------------------------

largest = 0

for s in simulation_sizes:
    pre = "data/"+s
    vols = np.genfromtxt(pre+volume_file)

    for i in range(runs):
        all_sizes = FuncPop.sizes_data(pre+"_sizes_"+str(i+1)+".csv")
        clusters  = np.genfromtxt(simulation_clusfs[s]+str(i+1)+".csv", dtype=int)
        largest   = np.max(clusters[:, 0]) if np.max(clusters[:, 0]) > largest else largest
        pri_sizes = Counter(clusters[:, 0])

        norm_all  = FuncPop.get_hist(all_sizes, vols[i])
        norm_pri  = FuncPop.get_hist(pri_sizes, vols[i], norm=False)
        
        simulation_data[s+"AllHist"+str(i)] = norm_all
        simulation_data[s+"PriHist"+str(i)] = norm_pri
        

# Form histogram

figA, axA = plt.subplots(1, 2, figsize=(12.8,4.8))       
figA.subplots_adjust(left=0.065, right=0.995, bottom=0.125, top=0.99, wspace=0.175, hspace=0)

bins   = np.linspace(discard, largest, pbins+1)
bins   = bins - 1    # histogram[0] corresponds to clusters of size 1, NOT 0. It has a LENGTH of largest but indexes to largest-1


d_bnd     = bins[1]-bins[0]

for c, s in enumerate(simulation_sizes):
    all_hist = []
    pri_hist = []
    
    for i in range(runs):
        data = simulation_data[s+"AllHist"+str(i)]
        data = np.concatenate((data, np.zeros(largest-len(data))))

        all_hist.append(data)
        
        data = simulation_data[s+"PriHist"+str(i)]
        data = np.concatenate((data, np.zeros(largest-len(data))))

        pri_hist.append(data)

    # Plot
    all_hist = np.array(all_hist)
    pri_hist = np.array(pri_hist)

    x   = np.arange(discard, largest+1, 1)
    mns = np.mean(all_hist, axis=0)
    err = np.std(all_hist, axis=0)/np.sqrt(runs)

    axA[0].plot(x, mns, label=simulation_labs[s], color=simulation_cols[s], linewidth=3.5)
    axA[0].fill_between(x, mns-err, mns+err, color=simulation_cols[s], alpha=0.5)
    
    x   = np.arange(discard, largest+1, 1)
    mns = np.mean(pri_hist, axis=0)
    err = np.std(pri_hist, axis=0)/np.sqrt(runs)
    axA[1].plot(x, mns, label=simulation_labs[s], color=simulation_cols[s], linewidth=3.5)
    axA[1].fill_between(x, mns-err, mns+err, color=simulation_cols[s], alpha=0.5)

axA[0].set_xlabel('Cluster size/atoms', size=20)
axA[0].set_ylabel('PDF', size=20)
axA[0].tick_params(labelsize=15)
axA[0].set_yscale('log')
axA[0].text(0.935, 0.925, '(a)', size=20, transform=axA[0].transAxes)

axA[1].set_xlabel('Cluster size/atoms', size=20)
axA[1].set_ylabel('PDF', size=20)
axA[1].tick_params(labelsize=15)
axA[1].set_yscale('log')
axA[1].text(0.6, 0.925, '(b)', size=20, transform=axA[1].transAxes)

axA[1].legend(prop={'size':13})

#figA.savefig("PDF_allvpri.png", dpi=300)

plt.show()
