#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions.

@author: David P. Fleming, 2019

Script output:

Number of iterations: 5000
Acceptance fraction for each walker:
[0.4854 0.4798 0.4832 0.4972 0.4864 0.4712 0.483  0.502  0.48   0.47
 0.4902 0.482  0.477  0.484  0.4692 0.4878 0.5032 0.4616 0.4864 0.4938
 0.4912 0.491  0.4748 0.4828 0.4744 0.4864 0.4966 0.4844 0.4606 0.4892
 0.4848 0.4814 0.4588 0.4824 0.4822 0.4676 0.4726 0.4842 0.4832 0.4952
 0.4858 0.5096 0.4858 0.4756 0.4908 0.4808 0.4736 0.4838 0.4764 0.4768
 0.4932 0.4916 0.4876 0.5024 0.497  0.4776 0.4862 0.4838 0.494  0.5036
 0.5048 0.4668 0.4856 0.4806 0.4846 0.4906 0.491  0.481  0.4858 0.5124
 0.494  0.4874 0.486  0.4806 0.4892 0.4826 0.4902 0.4714 0.4796 0.4694
 0.4784 0.4674 0.507  0.4826 0.4832 0.4816 0.483  0.5002 0.497  0.4766
 0.4946 0.4842 0.4892 0.4822 0.4776 0.4806 0.4662 0.4758 0.4842 0.4816]
Mean acceptance fraction: 0.4844
Burnin, thin: 178 25
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [96.52836322 63.49553947 55.87738167 74.02378995 81.48683031]
Mean Number of iterations / tau: 74.28238092276904
P(tsat >= age | data) = 0.435
P(tsat <= 1Gyr | data) = 0.005
Quantiles:
[(0.16, 0.08840007945179734), (0.5, 0.0889512673405358), (0.84, 0.08949857434163117)]
Quantiles:
[(0.16, -3.140359091248909), (0.5, -3.0104761759179244), (0.84, -2.8019413363178507)]
Quantiles:
[(0.16, 3.751082988200065), (0.5, 6.881980826416042), (0.84, 10.3251393816065)]
Quantiles:
[(0.16, 5.244407412320776), (0.5, 7.421342182425143), (0.84, 9.453910189690117)]
Quantiles:
[(0.16, -1.4670832599777712), (0.5, -1.1569240393281999), (0.84, -0.8475822900325357)]


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
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 15.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1Fiducial.h5"

# Whether or not to plot blobs
plotBlobs = False

# Extract results
if plotBlobs:
    chain, blobs = extractMCMCResults(filename, blobsExist=plotBlobs, burn=500)
else:
    chain = extractMCMCResults(filename, blobsExist=plotBlobs, burn=500)


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

# Estimate probability that TRAPPIST-1 is still saturated at the age of the
# system using the posterior distribution
mask = samples[:,2] >= samples[:,3]
print("P(tsat >= age | data) = %0.3lf" % np.mean(mask))

# Estimate probability that tsat < 1 Gyr
mask = samples[:,2] <= 1
print("P(tsat <= 1Gyr | data) = %0.3lf" % np.mean(mask))

# Plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 14},
                    title_fmt='.2f', verbose=True)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("trappist1Corner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("trappist1Corner.png", bbox_inches="tight", dpi=200)

# Done!
