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
[0.4769 0.4906 0.4779 0.4852 0.4935 0.4789 0.482  0.4876 0.4857 0.4825
 0.4834 0.4762 0.4846 0.4736 0.4862 0.4884 0.4791 0.4823 0.4977 0.4903
 0.4813 0.468  0.4892 0.4775 0.4905 0.4939 0.4694 0.4804 0.4677 0.4765
 0.4803 0.489  0.4889 0.4896 0.488  0.4847 0.4816 0.4957 0.4731 0.4719
 0.4734 0.4943 0.4839 0.4755 0.4926 0.4761 0.4772 0.4817 0.4793 0.4748
 0.4907 0.4782 0.478  0.4911 0.461  0.4794 0.4935 0.467  0.482  0.4847
 0.4685 0.4788 0.481  0.4884 0.4741 0.4842 0.4811 0.4858 0.5036 0.4914
 0.4802 0.4869 0.4718 0.4742 0.4902 0.4913 0.476  0.4729 0.4721 0.484
 0.4746 0.4613 0.4831 0.4849 0.4848 0.4936 0.4812 0.4864 0.4811 0.4745
 0.489  0.4792 0.4853 0.4753 0.4708 0.4689 0.4788 0.4723 0.4791 0.4831]
Mean acceptance fraction: 0.48157999999999995
Burnin, thin: 500 29
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [170.93133396 106.17373707  92.85359035 137.65401429 150.5192855 ]
Mean Number of iterations / tau: 131.62639223402772
$m_{\star}$ [M$_{\odot}$] = 8.896846e-02 + 5.628460e-04 - 5.564086e-04

$f_{sat}$ = -3.034530e+00 + 2.294533e-01 - 1.236747e-01

$t_{sat}$ [Gyr] = 6.637545e+00 + 3.533881e+00 - 3.130945e+00

Age [Gyr] = 7.464342e+00 + 2.010969e+00 - 2.101029e+00

$\beta_{XUV}$ = -1.156503e+00 + 3.118561e-01 - 3.032991e-01

P(tsat >= age | data) = 0.400
P(tsat <= 1Gyr | data) = 0.005
Quantiles:
[(0.16, 8.841204722820336), (0.5, 8.896845578696698), (0.84, 8.953130173876524)]
Quantiles:
[(0.16, -3.1582045735650497), (0.5, -3.0345298670729246), (0.84, -2.8050765428323)]
Quantiles:
[(0.16, 3.5065999688022904), (0.5, 6.637545078889502), (0.84, 10.171426357019735)]
Quantiles:
[(0.16, 5.36331273883467), (0.5, 7.464342080273961), (0.84, 9.4753110394882)]
Quantiles:
[(0.16, -1.4598022795057397), (0.5, -1.1565031474355993), (0.84, -0.8446470168008263)]

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
                                      removeRogueChains=False)
else:
    chain = extractMCMCResults(filename, blobsExist=plotBlobs, burn=500,
                               removeRogueChains=False)

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

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("trappist1Corner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("trappist1Corner.png", bbox_inches="tight", dpi=200)

# Done!
