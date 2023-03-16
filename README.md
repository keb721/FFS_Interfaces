# FFS_Interfaces
Repository for the Blow, Tribello, Sosso and Quigley paper "Interplay of multiple clusters and initial interface positioning for forward flux sampling of crystal nucleation"

"in.population" is a LAMMPS input file used to generate data for popoulation and initial flux data.
"in.FFS_initial" is a LAMMPS input file to generate points "at" the $\lambda_0$ interfaces of interest.
"in.FFS_l0s" is a LAMMPS input file to generate the probability of crossing the $\lambda_1$ interface.

"FuncPop.py" and "pop_full.py" give analysis and plotting codes for initial fluxes and population distributions.
"FuncGrs.py" and "multgrs.py" give the analysis scripts used for spatial correlations in the paper.
