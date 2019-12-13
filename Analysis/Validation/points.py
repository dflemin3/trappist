#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the approxposterior-derived posterior distributions
overplotted with approxposterior's training set

@author: David P. Fleming, 2019

"""

import numpy as np
import os
import sys
import corner
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
from trappist.mcmcUtils import extractMCMCResults
from trappist import trappist1 as t1
from scipy.stats import norm

#Typical plot parameters that make for pretty plots
mpl.rcParams['font.size'] = 17

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/convergedAP.h5"

# Extract data
samples = extractMCMCResults(filename, blobsExist=False, burn=500, verbose=False,
                             thinChains=True, removeRogueChains=False)

labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
          r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# Scale Mass for readability
samples[:,0] = samples[:,0] * 1.0e2

# Plot!
range = [[8.7, 9.06], [-3.4, -2.2], [0, 12], [0, 12], [-2, -0.2]]
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 16}, range=range,
                    title_fmt='.2f', verbose=False, hist_kwargs={"linewidth" : 2,
                    "density" : True})

# Fine-tune the formatting
ax_list = fig.axes
ax_list[0].set_title(r"$m_{\star}$ [M$_{\odot}$] $= 0.089 \pm {0.001}$", fontsize=16)
ax_list[-5].set_xlabel(r"$m_{\star}$ [$100\times$ M$_{\odot}$]", fontsize=22)
ax_list[-4].set_xlabel(r"$f_{sat}$", fontsize=22)
ax_list[-3].set_xlabel(r"$t_{sat}$ [Gyr]", fontsize=22)
ax_list[-2].set_xlabel(r"age [Gyr]", fontsize=22)
ax_list[-1].set_xlabel(r"$\beta_{XUV}$", fontsize=22)

ax_list[5].set_ylabel(r"$f_{sat}$", fontsize=22)
ax_list[10].set_ylabel(r"$t_{sat}$ [Gyr]", fontsize=22)
ax_list[15].set_ylabel(r"age [Gyr]", fontsize=22)
ax_list[20].set_ylabel(r"$\beta_{XUV}$", fontsize=22)

# Plot prior distributions on top of marginal posteriors
ax_list[0].axhline(2.5, lw=2, color="C0")
x = np.linspace(-3.4, -2.2, 100)
ax_list[6].plot(x, norm.pdf(x, loc=t1.fsatTrappist1, scale=t1.fsatTrappist1Sig),
                lw=2, color="C0")
ax_list[12].axhline(0.084, lw=2, color="C0")
x = np.linspace(0.1, 12, 100)
ax_list[18].plot(x, norm.pdf(x, loc=t1.ageTrappist1, scale=t1.ageTrappist1Sig),
                 lw=2, color="C0")
x = np.linspace(-2, 0, 100)
ax_list[24].plot(x, norm.pdf(x, loc=t1.betaTrappist1, scale=t1.betaTrappist1Sig),
                 lw=2, color="C0")

# Plot where forward model was evaluated - uncomment to plot!
# mass - fsat - tsat - age -beta
theta = np.load("../../Data/convergedAPFModelCache.npz")["theta"]
theta[:,0] = theta[:,0] * 1.0e2

# Size of initial training set
m0 = 50

# Plot initial training set
ax_list[5].scatter(theta[:m0,0], theta[:m0,1], s=10, color="C1", zorder=21) # (mass, fsat)
ax_list[10].scatter(theta[:m0,0], theta[:m0,2], s=10, color="C1", zorder=21) # (mass, tsat)
ax_list[11].scatter(theta[:m0,1], theta[:m0,2], s=10, color="C1", zorder=21) # (fsat, tsat)
ax_list[15].scatter(theta[:m0,0], theta[:m0,3], s=10, color="C1", zorder=21) # (mass, age)
ax_list[16].scatter(theta[:m0,1], theta[:m0,3], s=10, color="C1", zorder=21) # (fsat, age)
ax_list[17].scatter(theta[:m0,2], theta[:m0,3], s=10, color="C1", zorder=21) # (tsat, age)
ax_list[20].scatter(theta[:m0,0], theta[:m0,4], s=10, color="C1", zorder=21) # (mass, beta)
ax_list[21].scatter(theta[:m0,1], theta[:m0,4], s=10, color="C1", zorder=21) # (fsat, beta)
ax_list[22].scatter(theta[:m0,2], theta[:m0,4], s=10, color="C1", zorder=21) # (tsat, beta)
ax_list[23].scatter(theta[:m0,3], theta[:m0,4], s=10, color="C1", zorder=21) # (age, beta)

# Plot points selected by approxposterior
ax_list[5].scatter(theta[m0:,0], theta[m0:,1], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (mass, fsat)
ax_list[10].scatter(theta[m0:,0], theta[m0:,2], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (mass, tsat)
ax_list[11].scatter(theta[m0:,1], theta[m0:,2], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (fsat, tsat)
ax_list[15].scatter(theta[m0:,0], theta[m0:,3], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (mass, age)
ax_list[16].scatter(theta[m0:,1], theta[m0:,3], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (fsat, age)
ax_list[17].scatter(theta[m0:,2], theta[m0:,3], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (tsat, age)
ax_list[20].scatter(theta[m0:,0], theta[m0:,4], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (mass, beta)
ax_list[21].scatter(theta[m0:,1], theta[m0:,4], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (fsat, beta)
ax_list[22].scatter(theta[m0:,2], theta[m0:,4], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (tsat, beta)
ax_list[23].scatter(theta[m0:,3], theta[m0:,4], s=10, color="C0", zorder=20, alpha=0.75, edgecolor=None) # (age, beta)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("points.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("points.png", bbox_inches="tight", dpi=200)

# Done!
