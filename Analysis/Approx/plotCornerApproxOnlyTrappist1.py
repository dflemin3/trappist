#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the approxposterior-derived posterior distributions.

@author: David P. Fleming, 2019

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4672 0.4673 0.463  0.4716 0.4672 0.4756 0.4711 0.4701 0.4584 0.4668
 0.476  0.4744 0.4748 0.4723 0.477  0.4607 0.4713 0.481  0.4681 0.4641
 0.472  0.4638 0.474  0.4619 0.4672 0.4625 0.4707 0.451  0.473  0.4694
 0.4643 0.4764 0.464  0.4709 0.4759 0.4608 0.4617 0.4617 0.4616 0.4607
 0.4769 0.4693 0.4707 0.4643 0.4732 0.4645 0.4865 0.4527 0.4647 0.4554
 0.4695 0.4645 0.4622 0.4705 0.4534 0.4628 0.4646 0.4618 0.4688 0.4689
 0.4688 0.4682 0.4632 0.4734 0.4659 0.4569 0.458  0.4436 0.4501 0.4678
 0.4623 0.4649 0.45   0.4664 0.4736 0.4727 0.4708 0.4683 0.4653 0.4626
 0.4679 0.4666 0.4632 0.4722 0.4695 0.4615 0.4678 0.4594 0.4724 0.4624
 0.4699 0.4758 0.4787 0.4711 0.4653 0.475  0.4639 0.4812 0.4582 0.4726]
Mean acceptance fraction: 0.466941
Burnin, thin: 500 30
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [163.12017809  95.23874443  89.91209303 134.8759527  142.57474402]
Mean Number of iterations / tau: 125.14434245265241
$m_{\star}$ [M$_{\odot}$] = 8.897118e-02 + 5.266654e-04 - 5.288582e-04

$f_{sat}$ = -3.034367e+00 + 2.309017e-01 - 1.198365e-01

$t_{sat}$ [Gyr] = 6.755387e+00 + 3.520022e+00 - 3.099054e+00

Age [Gyr] = 7.574319e+00 + 1.871932e+00 - 1.932852e+00

$\beta_{XUV}$ = -1.154353e+00 + 2.897196e-01 - 2.888004e-01

P(tsat >= age | data) = 0.390
P(tsat <= 1Gyr | data) = 0.003

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

# Plot points selected by approxposterior?
plotAPPoints = False

# Path to data
filename = "../../Data/convergedAP.h5"

# Extract data
samples = extractMCMCResults(filename, blobsExist=False, burn=500,
                             thinChains=True, removeRogueChains=False)

labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
          r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# Output constraints
for ii in range(samples.shape[-1]):
    med = np.median(samples[:,ii])
    plus = np.percentile(samples[:,ii], 84) - med
    minus = med - np.percentile(samples[:,ii], 16)
    print("%s = %e + %e - %e" % (labels[ii], med, plus, minus))
    print()

# Scale Mass for readability
samples[:,0] = samples[:,0] * 1.0e2

# Estimate probability that TRAPPIST-1 is still saturated at the age of the
# system using samples from the posterior distribution
mask = samples[:,2] >= samples[:,3]
print("P(tsat >= age | data) = %0.3lf" % np.mean(mask))

# Estimate probability that tsat < 1 Gyr
mask = samples[:,2] <= 1
print("P(tsat <= 1Gyr | data) = %0.3lf" % np.mean(mask))

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
if plotAPPoints:
    theta = np.load("../../Data/apRunAPFModelCache.npz")["theta"]
    theta[:,0] = theta[:,0] * 1.0e2

    ax_list[5].scatter(theta[:,0], theta[:,1], s=10, color="red", zorder=20, alpha=0.2) # (mass, fsat)
    ax_list[10].scatter(theta[:,0], theta[:,2], s=10, color="red", zorder=20, alpha=0.2) # (mass, tsat)
    ax_list[11].scatter(theta[:,1], theta[:,2], s=10, color="red", zorder=20, alpha=0.2) # (fsat, tsat)
    ax_list[15].scatter(theta[:,0], theta[:,3], s=10, color="red", zorder=20, alpha=0.2) # (mass, age)
    ax_list[16].scatter(theta[:,1], theta[:,3], s=10, color="red", zorder=20, alpha=0.2) # (fsat, age)
    ax_list[17].scatter(theta[:,2], theta[:,3], s=10, color="red", zorder=20, alpha=0.2) # (tsat, age)
    ax_list[20].scatter(theta[:,0], theta[:,4], s=10, color="red", zorder=20, alpha=0.2) # (mass, beta)
    ax_list[21].scatter(theta[:,1], theta[:,4], s=10, color="red", zorder=20, alpha=0.2) # (fsat, beta)
    ax_list[22].scatter(theta[:,2], theta[:,4], s=10, color="red", zorder=20, alpha=0.2) # (tsat, beta)
    ax_list[23].scatter(theta[:,3], theta[:,4], s=10, color="red", zorder=20, alpha=0.2) # (age, beta)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("apCorner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("apCorner.png", bbox_inches="tight", dpi=200)

# Done!
