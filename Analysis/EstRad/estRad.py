#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate the distribution of TRAPPIST-1's radius using our stellar mass posterior
distributions and the Delrez et al. (2018) density constraint following the
procedure outlined in Van Grootel et al. (2018).

Script output:

Radius [Rsun] = 0.120300 + 0.001963 - 0.001845

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from trappist import mcmcUtils
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['font.size'] = 20.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# CGS constants
MSUN = 1.988435e33 # mass of Sun in grams
RSUN = 6.957e10 # radius of Sun incm
RHOSUN = MSUN / (4./3. * np.pi * RSUN**3) # density of sun in g/cm^3

# Read in evolutionary tracks
chains = mcmcUtils.extractMCMCResults("../../Data/trappist1Fiducial.h5",
                                      verbose=False, applyBurnin=True,
                                      thinChains=True, blobsExist=False)

# Draw num samples
num = int(1.0e5) # Number of samples

# Draw mass samples with replacement in grams
masses = np.random.choice(chains[:,0], size=(num,), replace=True) * MSUN

# Draw density samples in g/cm^3 by approximating constraint as wide gaussian
rhos = norm.rvs(loc=51.1, scale=2.4, size=(num,)) * RHOSUN

# Compute radius via density equation: rho = M/V = M/(4/3 * pi * r^3)
# -> (rho/m * (4/3) * pi)^(1/3) = r, but convert to Rsun
rads = np.power(masses / (rhos * (4./3.) * np.pi), 1./3.) / RSUN

# Visualize final distribution, compute statistics of interest
rad = np.median(rads)
radPlus = np.percentile(rads, 84) - rad
radMinus = rad - np.percentile(rads, 16)
print("Radius [Rsun] = %lf + %lf - %lf" % (rad, radPlus, radMinus))

# Plot histogram
fig, ax = plt.subplots(figsize=(6,5))

# Plot histogram of samples
ax.hist(rads, bins="auto", color="C0", density=True, alpha=0.6);
ax.hist(rads, bins="auto", color="C0", density=True, histtype="step", lw=2.5);

# Overplot med, +/-
ax.axvline(rad, color="k", ls="--", lw=2.5, label="This Work")
ax.axvline(rad + radPlus, color="k", ls="--", lw=2.5)
ax.axvline(rad - radMinus, color="k", ls="--", lw=2.5)

# Overplot Van Grootel et al. (2018) constraints
ax.axvline(0.121, color="C1", ls="--", lw=2.5, label="Van Grootel et al. (2018)")
ax.axvline(0.121 + 0.003, color="C1", ls="--", lw=2.5)
ax.axvline(0.121 - 0.003, color="C1", ls="--", lw=2.5)

ax.set_ylabel("Density")
ax.set_xlabel(r"Radius [$R_{\odot}]$")
ax.legend(loc="best", framealpha=0.8)
fig.tight_layout()

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("estRad.pdf", bbox_inches="tight",
                dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("estRad.png", bbox_inches="tight",
                dpi=200)
# Done!
