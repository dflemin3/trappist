#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions.

@author: David P. Fleming, 2019

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4771 0.4818 0.4753 0.4771 0.4844 0.4755 0.4745 0.4705 0.4874 0.4762
 0.4732 0.4684 0.4789 0.4793 0.478  0.4709 0.477  0.4739 0.4722 0.4739
 0.4786 0.4717 0.4795 0.4754 0.4841 0.475  0.4789 0.4873 0.4892 0.4673
 0.4669 0.4722 0.4788 0.49   0.4836 0.4701 0.4742 0.4783 0.476  0.4683
 0.4809 0.4934 0.4734 0.4743 0.4898 0.4866 0.4878 0.4901 0.4768 0.4752
 0.4835 0.4779 0.4842 0.4815 0.4784 0.4892 0.4814 0.4859 0.4656 0.4561
 0.4901 0.4681 0.489  0.4823 0.4834 0.4785 0.4839 0.4742 0.479  0.4924
 0.468  0.4727 0.48   0.4822 0.4904 0.4836 0.4816 0.4842 0.4872 0.4754
 0.4711 0.4618 0.4879 0.4704 0.488  0.4808 0.4817 0.4601 0.4806 0.4751
 0.476  0.4818 0.4704 0.474  0.4787 0.4941 0.4846 0.0051 0.4756 0.4732]
Mean acceptance fraction: 0.4738009999999999
Burnin, thin: 500 32
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [155.53291067 114.97383974  97.23057321 130.81622108 138.67385981]
Mean Number of iterations / tau: 127.44548090424023
$m_{\star}$ [M$_{\odot}$] = 8.896781e-02 + 5.902382e-04 - 5.881137e-04

$f_{sat}$ = -3.045454e+00 + 2.024326e-01 - 1.193354e-01

$t_{sat}$ [Gyr] = 6.803826e+00 + 3.537746e+00 - 3.017258e+00

Age [Gyr] = 7.634782e+00 + 1.796725e+00 - 1.877082e+00

$\beta_{XUV}$ = -1.152070e+00 + 2.911425e-01 - 2.821559e-01

P(tsat >= age | data) = 0.399
P(tsat <= 1Gyr | data) = 0.002

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
                             thinChains=True, removeRogueChains=True)

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
x = np.linspace(-3.3, -2.2, 100)
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
