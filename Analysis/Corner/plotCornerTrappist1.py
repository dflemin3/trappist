#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions. Note that we
remove rogue chains (via visual inspection and by indentifying those with
very low acceptance fractions, < 0.01).

@author: David P. Fleming, 2019

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4818 0.481  0.4825 0.4815 0.4701 0.4689 0.466  0.476  0.4804 0.4767
 0.4766 0.4811 0.466  0.4612 0.465  0.4845 0.4748 0.4797 0.4821 0.4753
 0.4668 0.4714 0.4715 0.4769 0.4773 0.482  0.4762 0.4676 0.4775 0.4898
 0.4606 0.4845 0.4705 0.4839 0.4846 0.4748 0.4788 0.4775 0.4715 0.485
 0.4734 0.4707 0.4715 0.4753 0.4766 0.474  0.4745 0.4851 0.4813 0.473
 0.4808 0.4852 0.4816 0.4673 0.4681 0.4694 0.4825 0.4802 0.4875 0.4845
 0.4687 0.4625 0.4875 0.4761 0.4828 0.4755 0.4744 0.4837 0.466  0.4882
 0.4817 0.4735 0.4766 0.4676 0.0064 0.4751 0.4731 0.4751 0.4769 0.4773
 0.4758 0.4771 0.4649 0.4863 0.4806 0.4914 0.4705 0.4845 0.4719 0.4823
 0.4734 0.4836 0.4634 0.4792 0.4778 0.4873 0.4518 0.485  0.4625 0.4787]
Mean acceptance fraction: 0.4715909999999999
Burnin, thin: 500 30
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [164.00771893 102.48784317  87.49975161 147.9347476  150.23351435]
Mean Number of iterations / tau: 130.43271513146442
$m_{\star}$ [M$_{\odot}$] = 8.896755e-02 + 5.626196e-04 - 5.469983e-04

$f_{sat}$ = -3.036436e+00 + 2.307563e-01 - 1.233763e-01

$t_{sat}$ [Gyr] = 6.645777e+00 + 3.540436e+00 - 3.119382e+00

Age [Gyr] = 7.465679e+00 + 2.036275e+00 - 2.109250e+00

$\beta_{XUV}$ = -1.160481e+00 + 3.119635e-01 - 3.075076e-01

P(tsat >= age | data) = 0.406
P(tsat <= 1Gyr | data) = 0.005
Quantiles:
[(0.16, 8.842055640345118), (0.5, 8.896755474291194), (0.84, 8.95301743419984)]
Quantiles:
[(0.16, -3.159811895289719), (0.5, -3.0364355453557934), (0.84, -2.8056792008684974)]
Quantiles:
[(0.16, 3.526395204705141), (0.5, 6.645776794552386), (0.84, 10.186212952128916)]
Quantiles:
[(0.16, 5.356428188012375), (0.5, 7.465678663800693), (0.84, 9.501953257219233)]
Quantiles:
[(0.16, -1.4679887752478038), (0.5, -1.1604812115772747), (0.84, -0.8485177489506157)]

"""

import numpy as np
import os
import sys
import corner
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
from trappist import trappist1 as t1
from trappist.mcmcUtils import extractMCMCResults
from scipy.stats import norm

#Typical plot parameters that make for pretty plots
mpl.rcParams['font.size'] = 17

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1FiducialRatio.h5"

# Whether or not to plot blobs
plotBlobs = False

# Extract results
if plotBlobs:
    chain, blobs = extractMCMCResults(filename, blobsExist=plotBlobs, burn=500,
                                      removeRogueChains=True)
else:
    chain = extractMCMCResults(filename, blobsExist=plotBlobs, burn=500,
                               removeRogueChains=True)


if plotBlobs:
    samples = np.hstack([chain, blobs])

    # Define Axis Labels
    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "Lum",
              "logLumXUV", "Radius"]

    # Make luminosity units more palatable
    samples[:,5] = samples[:,5]*1.0e3
else:
    # Just consider stellar data
    samples = chain
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
                    title_fmt='.2f', verbose=True, hist_kwargs={"linewidth" : 2,
                    "density" : True})

# Fine-tune the formatting
ax_list = fig.axes
ax_list[0].set_title(r"$m_{\star}$ [M$_{\odot}$] $= 0.089^{+0.001}_{-0.001}$", fontsize=16)
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

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("trappist1Corner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("trappist1Corner.png", bbox_inches="tight", dpi=200)

# Done!
