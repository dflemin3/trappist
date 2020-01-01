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
[ 0.0278958   0.12321058 -0.48396682 -2.50206929  0.4631157 ]
[ 6.66959603 10.7688684   3.71396325 13.78800122 15.42533837]

Average percent error between means, standard deviations
-0.4743628061793054
10.073153453034298

Final percent error between medians, standard deviations for each parameter
[-0.0030652   0.00535184 -1.77538659 -1.47336565  0.18588125]
[5.62768255 3.25762233 0.46249447 9.93277425 8.08109564]

Final average percent error between medians, standard deviations for each parameter
-0.612116871124775
5.472333849608273

Final AP Monte Carlo error:
[3.79469263e-06 1.26285916e-03 2.02074534e-02 1.10407025e-02
 1.72846464e-03]

Final ratio of mean diffs / mcseEMCEE
[-898.60870316    4.34359174  -88.68919135 -113.4651086    87.50609112]

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
nIters = 7

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
    axes[0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axes[0].set_xticklabels(["1", "2", "3", "4", "5", "6", "7"])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$|$Median Error$|$ [$\%$]")
    axes[0].set_ylim(5.0e-3, 1.0e2)
    axes[0].set_yscale("log")

    # Right: percenter difference between uncertainty width
    axes[1].plot(iters, np.fabs(uwidthDiffs[:,ii]), "o-", lw=2.5,
                 color=colors[ii], label=labels[ii], zorder=ii)

    # Format
    axes[1].set_xlim(0.8, nIters + 0.2)
    axes[1].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axes[1].set_xticklabels(["1", "2", "3", "4", "5", "6", "7"])
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
    fig.savefig("medConvergence.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("medConvergence.png", bbox_inches="tight", dpi=200)

# Done!
