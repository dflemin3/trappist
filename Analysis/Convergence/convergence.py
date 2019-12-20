#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot how the relative error between the approxposterior and fiducial MCMC
posterior medians and standard deviations evolve as a function of approxposterior
iteration.

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

script output:

emcee Monte Carlo error:
[3.41105036e-06 1.23212229e-03 2.00180716e-02 1.29851870e-02
 2.12420922e-03]

Average percent error between medians, standard deviations for each parameter
[ 0.03997395  0.2186239  -0.15428029 -2.70650118 -0.29857911]
[-15.91414918   7.73208078   5.31705512  12.92854318  16.78230614]

Average percent error between means, standard deviations
-0.5801525471260128
5.369167209932392

Final percent error between medians, standard deviations for each parameter
[ 0.02695217 -0.00481336 -1.71035567 -0.94876542 -0.63936197]
[-0.23136617  3.62777257  0.83304075  9.57526654  8.95143481]

Final average percent error between medians, standard deviations for each parameter
-0.6552688483086629
4.551229702452913

Final approxposterior Monte Carlo error:
[3.41105036e-06 1.23212229e-03 2.00180716e-02 1.29851870e-02
 2.12420922e-03]

Final ratio of mean diffs / mcseEMCEE
[ 7.90142818e+03 -3.90655703e+00 -8.54405810e+01 -7.30652102e+01
 -3.00988228e+02]

"""

import numpy as np
import os
import sys
import corner
import emcee
import matplotlib as mpl
from scipy.linalg import norm
import matplotlib.pyplot as plt
from trappist.mcmcUtils import extractMCMCResults
from approxposterior import mcmcUtils as mcu

#Typical plot parameters that make for pretty plots
mpl.rcParams['font.size'] = 15

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1FiducialRatio.h5"
nIters = 15

# Extract true MCMC results
trueChain = extractMCMCResults(filename, blobsExist=False, burn=500,
                               verbose=False)

# Estimate Monte Carlo error
mcseEMCEE = mcu.batchMeansMCSE(trueChain)
print("emcee Monte Carlo error:")
print(mcseEMCEE)
print()

# Define labels
labels = [r"$m_{\star}$", r"$f_{sat}$", r"$t_{sat}$", r"Age", r"$\beta_{XUV}$"]

# Containers
medDiffs = np.zeros((nIters, 5))
uwidthDiffs = np.zeros_like(medDiffs)

for ii in range(nIters):
    # Load iith approxposterior output
    approxFilename = "../../Data/apRun%d.h5" % ii
    approxChain = extractMCMCResults(approxFilename, blobsExist=False, burn=500,
                                     verbose=False)
    # Loop over variables
    for jj in range(len(labels)):

        # Compute relative difference between marginal distribution medians
        medDiffs[ii,jj] = 100 * (np.median(trueChain[:,jj]) - np.median(approxChain[:,jj])) / np.median(trueChain[:,jj])

        # Relative difference between marginal std
        true = np.std(trueChain[:,jj])
        approx = np.std(approxChain[:,jj])
        uwidthDiffs[ii,jj] = 100 * (true - approx) / true

# Output statistics
print("Average percent error between medians, standard deviations for each parameter")
print(np.mean(medDiffs, axis=0))
print(np.mean(uwidthDiffs, axis=0))
print()
print("Average percent error between means, standard deviations")
print(np.mean(medDiffs))
print(np.mean(uwidthDiffs))
print()
print("Final percent error between medians, standard deviations for each parameter")
print(medDiffs[-1,:])
print(uwidthDiffs[-1, :])
print()
print("Final average percent error between medians, standard deviations for each parameter")
print(np.mean(medDiffs[-1,:]))
print(np.mean(uwidthDiffs[-1, :]))
print()
mcseAP = mcu.batchMeansMCSE(approxChain)
print("Final AP Monte Carlo error:")
print(mcseAP)
print()
print("Final ratio of mean diffs / mcseEMCEE")
print(medDiffs[-1,:]/mcseEMCEE)

# Plot
fig, axes = plt.subplots(ncols=2, figsize=(14, 6))

iters = [ii+1 for ii in range(nIters)]
colors = ["C0", "C1", "C2", "C4", "C5"]
for ii in range(len(labels)):

    # Left: percenter difference between median
    axes[0].plot(iters, np.fabs(medDiffs[:,ii]), "o-", lw=2.5, color=colors[ii],
                 label=labels[ii], zorder=11)

    # Format
    axes[0].set_xlim(0.8, nIters + 0.2)
    axes[0].set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    axes[0].set_xticklabels(["1", "3", "5", "7", "9", "11", "13", "15"])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$|$Median Error$|$ [$\%$]")
    axes[0].set_ylim(5.0e-3, 1.0e2)
    axes[0].set_yscale("log")

    # Right: percenter difference between uncertainty width
    axes[1].plot(iters, np.fabs(uwidthDiffs[:,ii]), "o-", lw=2.5,
                 color=colors[ii], label=labels[ii], zorder=ii)

    # Format
    axes[1].set_xlim(0.8, nIters + 0.2)
    axes[1].set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    axes[1].set_xticklabels(["1", "3", "5", "7", "9", "11", "13", "15"])
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$|$Standard Deviation Error$|$ [$\%$]")
    axes[1].set_ylim(5.0e-3, 1.0e2)
    axes[1].set_yscale("log")
    axes[1].legend(loc="lower left", framealpha=0, fontsize=16)

# Plot lines at 10%, 1% error
axes[0].axhline(1, lw=2, ls="--", color="k", zorder=0)
axes[0].axhline(10, lw=2, ls="--", color="k", zorder=0)
axes[1].axhline(1, lw=2, ls="--", color="k", zorder=0)
axes[1].axhline(10, lw=2, ls="--", color="k", zorder=0)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("convergence.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("convergence.png", bbox_inches="tight", dpi=200)

# Done!
