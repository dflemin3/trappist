#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XXX

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

script output:

XXX

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

#Typical plot parameters that make for pretty plots
mpl.rcParams['font.size'] = 15

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1FiducialRatio.h5"
nIters = 10

# Extract true MCMC results
trueChain = extractMCMCResults(filename, blobsExist=False, burn=500,
                               verbose=False)

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

        # Compute relative difference between width of IQR
        true = np.percentile(trueChain[:,jj], 86) - np.percentile(trueChain[:,jj], 14)
        approx = np.percentile(approxChain[:,jj], 86) - np.percentile(approxChain[:,jj], 14)
        uwidthDiffs[ii,jj] = 100 * (true - approx) / true

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
    axes[0].set_xticks([2, 4, 6, 8, 10])
    axes[0].set_xticklabels(["2", "4", "6", "8", "10"])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$|$Median Error$|$ [$\%$]")
    axes[0].set_ylim(5.0e-3, 4.0e1)
    axes[0].set_yscale("log")

    # Right: percenter difference between uncertainty width
    axes[1].plot(iters, np.fabs(uwidthDiffs[:,ii]), "o-", lw=2.5,
                 color=colors[ii], label=labels[ii], zorder=ii)

    # Format
    axes[1].set_xlim(0.8, nIters + 0.2)
    axes[1].set_xticks([2, 4, 6, 8, 10])
    axes[1].set_xticklabels(["2", "4", "6", "8", "10"])
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$|$$\Delta$ Error$|$ [$\%$]")
    axes[1].set_ylim(5.0e-3, 4.0e1)
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
