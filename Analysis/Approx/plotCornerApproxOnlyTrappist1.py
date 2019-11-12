#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions.

@author: David P. Fleming, 2019

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4545 0.4494 0.4598 0.4578 0.3081 0.4569 0.4727 0.4604 0.4621 0.4531
 0.4577 0.4623 0.4688 0.458  0.4509 0.4666 0.4737 0.4499 0.4617 0.4603
 0.4728 0.4563 0.4471 0.4561 0.4629 0.461  0.4441 0.4733 0.4449 0.4531
 0.4508 0.4539 0.4475 0.4648 0.4458 0.4525 0.4637 0.2762 0.464  0.4387
 0.4634 0.4638 0.4621 0.4459 0.4529 0.4578 0.4482 0.4584 0.4511 0.4623
 0.467  0.455  0.4599 0.4448 0.4451 0.4545 0.4591 0.4409 0.4506 0.4451
 0.4702 0.4552 0.457  0.4558 0.463  0.4488 0.4469 0.4617 0.4598 0.4662
 0.4598 0.461  0.4602 0.4583 0.4507 0.4596 0.4563 0.4568 0.4445 0.471
 0.4594 0.4574 0.4493 0.448  0.4604 0.4617 0.4546 0.4578 0.4487 0.4545
 0.453  0.4547 0.474  0.4643 0.4699 0.4648 0.4663 0.4421 0.4549 0.4391]
Mean acceptance fraction: 0.45359800000000006
Burnin, thin: 500 39
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [124.31849958  75.503293    65.95958776 112.97917982 126.51341164]
Mean Number of iterations / tau: 101.05479435919378
$m_{\star}$ [M$_{\odot}$] = 8.892028e-02 + 5.623543e-04 - 5.447724e-04

$f_{sat}$ = -3.033730e+00 + 1.967705e-01 - 9.928647e-02

$t_{sat}$ [Gyr] = 6.621841e+00 + 3.561250e+00 - 2.563445e+00

Age [Gyr] = 7.539407e+00 + 1.434721e+00 - 1.514535e+00

$\beta_{XUV}$ = -1.158456e+00 + 2.066506e-01 - 2.068156e-01

P(tsat >= age | data) = 0.387
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
range = None
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
