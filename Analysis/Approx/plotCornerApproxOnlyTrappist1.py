#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions.

@author: David P. Fleming, 2019

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4733 0.4647 0.4706 0.4731 0.46   0.4708 0.4647 0.4617 0.4803 0.4833
 0.4556 0.4682 0.4519 0.4593 0.4764 0.469  0.45   0.4555 0.4634 0.4608
 0.4696 0.4708 0.4841 0.4735 0.4584 0.4645 0.4583 0.4543 0.4682 0.4697
 0.4777 0.4788 0.4676 0.4683 0.4733 0.4684 0.4807 0.4665 0.4377 0.4707
 0.4701 0.47   0.477  0.4661 0.4584 0.4626 0.4641 0.468  0.4694 0.4724
 0.4689 0.4934 0.4456 0.4501 0.4588 0.4687 0.4645 0.4645 0.4653 0.4787
 0.4638 0.4728 0.4569 0.4774 0.4695 0.4884 0.4616 0.4705 0.4666 0.4537
 0.4655 0.4656 0.4634 0.4696 0.4716 0.4824 0.4598 0.4704 0.475  0.4693
 0.4812 0.4736 0.4721 0.4672 0.4703 0.4699 0.4745 0.4777 0.4639 0.469
 0.4785 0.4677 0.4721 0.4604 0.4798 0.4767 0.4635 0.4726 0.4805 0.4602]
Mean acceptance fraction: 0.468155
Burnin, thin: 500 31
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [158.99243628  87.54718736  73.10110259 133.00115384 138.44738198]
Mean Number of iterations / tau: 118.21785240676866
$m_{\star}$ [M$_{\odot}$] = 8.893236e-02 + 5.489587e-04 - 5.468893e-04

$f_{sat}$ = -3.031557e+00 + 1.940222e-01 - 1.013755e-01

$t_{sat}$ [Gyr] = 6.551915e+00 + 3.474280e+00 - 2.536282e+00

Age [Gyr] = 7.568291e+00 + 1.435620e+00 - 1.456144e+00

$\beta_{XUV}$ = -1.161093e+00 + 2.152944e-01 - 2.089716e-01

P(tsat >= age | data) = 0.367
P(tsat <= 1Gyr | data) = 0.000

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
mpl.rcParams['font.size'] = 17

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/convergedAP.h5"

# Extract data
samples = extractMCMCResults(filename, blobsExist=False, burn=500)

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
range = [[8.7, 9.06], [-3.3, -2.2], [0, 12], [0, 12], [-2, -0.2]]
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 16}, range=range,
                    title_fmt='.2f', verbose=False, hist_kwargs={"linewidth" : 1.5})

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

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("apCorner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("apCorner.png", bbox_inches="tight", dpi=200)

# Done!
