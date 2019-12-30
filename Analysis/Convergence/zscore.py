#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot how the relative error between the approxposterior and fiducial MCMC
posterior means and standard deviations evolve as a function of approxposterior
iteration.

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

script output:

zscores:
[[0.07350508 0.19064462 0.09149129 0.00101502 0.22113886]
 [0.02925934 0.06807738 0.03525095 0.03183426 0.06449193]
 [0.02717987 0.02246936 0.02043032 0.01504691 0.00968476]
 [0.00065738 0.02966166 0.00651577 0.00075434 0.0033683 ]
 [0.02709885 0.05604613 0.02920786 0.0251325  0.00297203]
 [0.00166558 0.01235034 0.00073106 0.01775269 0.0099493 ]]

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

# Define labels
labels = [r"$m_{\star}$", r"$f_{sat}$", r"$t_{sat}$", r"age", r"$\beta_{XUV}$"]

# Containers
nIters = 7
zscores = np.zeros((nIters-1, 5))

for ii in range(1,nIters):
    # Load ii-1th approxposterior output
    approxFilenamePrev = "../../Data/apRun%d.h5" % (ii-1)
    approxChainPrev = extractMCMCResults(approxFilenamePrev, blobsExist=False,
                                         burn=500, verbose=False)

    # Load iith approxposterior output
    approxFilename = "../../Data/apRun%d.h5" % ii
    approxChain = extractMCMCResults(approxFilename, blobsExist=False, burn=500,
                                     verbose=False)
    # Loop over variables
    for jj in range(len(labels)):

        # Compute relative difference between marginal distribution medians
        zscores[ii-1,jj] = np.fabs((np.mean(approxChain[:,jj]) - np.mean(approxChainPrev[:,jj]))) / np.std(approxChainPrev[:,jj])

print("zscores:")
print(zscores)

# Plot
fig, ax = plt.subplots(figsize=(7, 6))

iters = [ii+1 for ii in range(nIters-1)]
colors = ["C0", "C1", "C2", "C4", "C5"]
for ii in range(len(labels)):

    # Left: percenter difference between median
    ax.plot(iters, zscores[:,ii], "o-", lw=2.5, color=colors[ii],
            label=labels[ii], zorder=11)

    # Format
    ax.set_xlim(0.8, nIters - 0.8)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|\mu_i - \mu_{i-1}| / \sigma_{i-1}$")
    ax.set_ylim(4.0e-4, 0.25)
    #ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0, ncol=2, fontsize=15)

# Plot lines convergence threshold, eps
ax.axhline(0.1, lw=2, ls="--", color="k", zorder=0)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("convergence.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("convergence.png", bbox_inches="tight", dpi=200)

# Done!
