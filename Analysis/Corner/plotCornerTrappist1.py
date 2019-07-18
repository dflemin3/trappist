#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions.

@author: David P. Fleming, 2019

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4323 0.4515 0.4361 0.455  0.4574 0.4429 0.4623 0.4361 0.461  0.4529
 0.4436 0.4539 0.4477 0.4349 0.4525 0.451  0.4386 0.4459 0.4407 0.4518
 0.4578 0.4383 0.4533 0.4413 0.4526 0.4519 0.4463 0.4455 0.4561 0.4409
 0.436  0.4619 0.4285 0.4435 0.4557 0.438  0.4477 0.4486 0.4462 0.4515
 0.452  0.4442 0.4458 0.4512 0.4587 0.4569 0.4547 0.4408 0.4469 0.4529
 0.4509 0.4559 0.4535 0.4601 0.4591 0.4604 0.4516 0.4478 0.452  0.4461
 0.4576 0.4459 0.4445 0.4476 0.4428 0.4585 0.4297 0.4584 0.4384 0.4473
 0.4604 0.4639 0.4675 0.451  0.4572 0.4603 0.4582 0.4504 0.4424 0.452
 0.4328 0.4556 0.4519 0.456  0.4524 0.4513 0.4403 0.4547 0.452  0.4477
 0.4452 0.4518 0.453  0.4403 0.4558 0.4577 0.4481 0.4422 0.4588 0.4473]
Mean acceptance fraction: 0.449601
Burnin, thin: 500 35
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [140.59596201  78.50343436  75.26040711 120.65071601 137.41525736]
Mean Number of iterations / tau: 110.48515537086878
P(tsat >= age | data) = 0.426
P(tsat <= 1Gyr | data) = 0.004
Quantiles:
[(0.16, 0.08839547472054667), (0.5, 0.08894615011071999), (0.84, 0.08950534135762712)]
Quantiles:
[(0.16, -3.1495432034580046), (0.5, -3.0529929021776536), (0.84, -2.8113992554362333)]
Quantiles:
[(0.16, 3.699454860180092), (0.5, 6.852308213256522), (0.84, 10.277962336184737)]
Quantiles:
[(0.16, 5.308547676761598), (0.5, 7.435267547328605), (0.84, 9.471131458867541)]
Quantiles:
[(0.16, -1.4639834633001065), (0.5, -1.1595607830719579), (0.84, -0.8453924035645292)]

$m_{\star}$ [M$_{\odot}$] = 8.894615e-02 + 5.591912e-04 - 5.506754e-04

$f_{sat}$ = -3.052993e+00 + 2.415936e-01 - 9.655030e-02

$t_{sat}$ [Gyr] = 6.852308e+00 + 3.425654e+00 - 3.152853e+00

Age [Gyr] = 7.435268e+00 + 2.035864e+00 - 2.126720e+00

$\beta_{XUV}$ = -1.159561e+00 + 3.141684e-01 - 3.044227e-01

P(tsat >= age | data) = 0.426
P(tsat <= 1Gyr | data) = 0.004

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

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("trappist1Corner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("trappist1Corner.png", bbox_inches="tight", dpi=200)

# Done!
