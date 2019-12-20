#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of both the MCMC-derived ("true") and approxposterior-derived
posterior distributions.

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

script output:

True P(tsat >= age | data) = 0.400
True P(tsat <= 1Gyr | data) = 0.005
approxposterior P(tsat >= age | data) = 0.394
approxposterior P(tsat <= 1Gyr | data) = 0.002

"""

import numpy as np
import os
import sys
import corner
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
from trappist.mcmcUtils import extractMCMCResults

#Typical plot parameters that make for pretty plots
mpl.rcParams['font.size'] = 17.5

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1FiducialRatio.h5"
approxFilename = "../../Data/convergedAP.h5"

# Whether or not to plot blobs

# Extract true MCMC results
print("Loading true MCMC chain...")
trueChain = extractMCMCResults(filename, blobsExist=False, burn=500,
                               verbose=False, thinChains=False)

# Extract approxposterior MCMC results
print("Loading approxposterior MCMC chain...")
approxChain = extractMCMCResults(approxFilename, blobsExist=False, burn=500,
                                 verbose=False, thinChains=False)

# Define labels
labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
          r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# Scale Mass for readability
trueChain[:,0] = trueChain[:,0] * 1.0e2
approxChain[:,0] = approxChain[:,0] * 1.0e2

# Estimate probability that TRAPPIST-1 is still saturated at the age of the
# system using samples from the posterior distribution
mask = trueChain[:,2] >= trueChain[:,3]
print("True P(tsat >= age | data) = %0.3lf" % np.mean(mask))

# Estimate probability that tsat < 1 Gyr
mask = trueChain[:,2] <= 1
print("True P(tsat <= 1Gyr | data) = %0.3lf" % np.mean(mask))

# Estimate probability that TRAPPIST-1 is still saturated at the age of the
# system using samples from the posterior distribution
mask = approxChain[:,2] >= approxChain[:,3]
print("approxposterior P(tsat >= age | data) = %0.3lf" % np.mean(mask))

# Estimate probability that tsat < 1 Gyr
mask = approxChain[:,2] <= 1
print("approxposterior P(tsat <= 1Gyr | data) = %0.3lf" % np.mean(mask))

# Plot both true and approxposterior posterior distribution
bins = 20
range = [[8.7, 9.06], [-3.4, -2.2], [0, 12], [0, 12], [-2, -0.2]]

# approxposterior
fig = corner.corner(approxChain, quantiles=[], labels=labels,
                    bins=bins, show_titles=False, title_kwargs={"fontsize": 16},
                    title_fmt='.2f', verbose=False, hist_kwargs={"linewidth" : 2.5},
                    plot_contours=True, plot_datapoints=False, plot_density=False,
                    color="royalblue", range=range, no_fill_contours=True)

# True
fig = corner.corner(trueChain, quantiles=[], labels=labels, show_titles=False,
                    bins=bins, verbose=False, plot_density=True, fig=fig,
                    hist_kwargs={"linewidth" : 2.5}, plot_contours=True,
                    plot_datapoints=False, color="k", range=range,
                    no_fill_contours=True)

fig = corner.corner(approxChain, quantiles=[], labels=labels, fig=fig,
                    bins=bins, show_titles=False, title_kwargs={"fontsize": 16},
                    title_fmt='.2f', verbose=False, hist_kwargs={"linewidth" : 2.5},
                    plot_contours=False, plot_datapoints=False, plot_density=False,
                    color="royalblue", range=range, no_fill_contours=True)

# Add legend
fig.axes[1].text(0.13, 0.55, "emcee: 1,000,000 forward model evaluations", fontsize=26, color="k", zorder=99)
fig.axes[1].text(0.13, 0.375, r"approxposterior, 800 forward model evaluations", fontsize=26, color="royalblue",
                 zorder=99)

# Fine-tune the formatting
ax_list = fig.axes
ax_list[-5].set_xlabel(r"$m_{\star}$ [$100\times$ M$_{\odot}$]", labelpad=30)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("stacked.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("stacked.png", bbox_inches="tight", dpi=200)

# Done!
