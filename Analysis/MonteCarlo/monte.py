#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Plot LXUV/Lbol as a function of age for a synthetic population of ultracool
dwarfs with initial conditions sampled from our adopted prior distributions.

Script output:

Number of points in mask: 156
tsat: 6.740884e+00 - 2.788065e+00 + 3.419573e+00
fsat: -3.059389e+00 - 7.738391e-02 + 2.086102e-01
P(saturated) = 0.377

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from trappist import trappist1 as t1

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (6,5)
mpl.rcParams['font.size'] = 16.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Read in evolutionary tracks
data = pd.read_csv("../../Data/mcResults.csv", index_col=0)

# Estimate statistics for simulations consistent with 1 sigma LXUV/Lbol and age
# constraints for TRAPPIST-1
sig = 1
mask = (data["dAge"].values < t1.ageTrappist1 + sig*t1.ageTrappist1Sig)
mask = mask & (data["dAge"].values > t1.ageTrappist1 - sig*t1.ageTrappist1Sig)
mask = mask & (data["dLRatioAge"].values < t1.LRatioTrappist1 + sig*t1.LRatioTrappist1Sig)
mask = mask & (data["dLRatioAge"].values > t1.LRatioTrappist1 - sig*t1.LRatioTrappist1Sig)

print("Number of points in mask: %d" % np.sum(mask))
low, med, high = np.percentile(data["dSatXUVTime"].values[mask], [16, 50, 84])
print("tsat: %e - %e + %e" % (med, med - low, high - med))
low, med, high = np.percentile(data["dSatXUVFrac"].values[mask], [16, 50, 84])
print("fsat: %e - %e + %e" % (med, med - low, high - med))

# Naively estimate P(saturated) based on TRAPPIST priors
samps = t1.samplePriorTRAPPIST1(size=10000)
samps = np.array(samps)
mask = samps[:,2] >= samps[:,3]
print("P(saturated) = %0.3lf" % np.mean(mask))

# Now plot full population of ultracool dwarfs
fig, ax = plt.subplots()

im = ax.scatter(data["dAge"], data["dLRatioAge"], s=10, c=data["dSatXUVTime"],
                edgecolor=None, cmap="viridis", zorder=1, label="")
cbar = fig.colorbar(im)
cbar.set_label("$t_{sat}$ [Gyr]")

# Plot TRAPPIST-1 observations
ax.errorbar(t1.ageTrappist1, t1.LRatioTrappist1, xerr=t1.ageTrappist1Sig,
            yerr=t1.LRatioTrappist1Sig, color="w", lw=3, zorder=2, capthick=3,
            capsize=3, fmt='--o')
ax.errorbar(t1.ageTrappist1, t1.LRatioTrappist1, xerr=t1.ageTrappist1Sig,
            yerr=t1.LRatioTrappist1Sig, color="darkred", lw=2, zorder=3, capthick=2,
            capsize=2.5, fmt='--o')

# Dummy points for legend
ax.scatter(1, 1, s=45, color="k", label="Simulations", zorder=0)
ax.scatter(1, 1, s=45, color="darkred", label="TRAPPIST-1", zorder=0)

# Format
ax.set_xlim(0.02, 12.1)
ax.set_ylim(1.0e-6, 1.0e-2)
ax.set_yscale("log")
ax.set_xlabel("Age [Gyr]")
ax.set_ylabel("L$_{XUV}/$L$_{bol}$")
legend = ax.legend(loc="lower left", fontsize=13, framealpha=0, numpoints=1)
legend.legendHandles[0].set_color('k')

fig.tight_layout()

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("ultracoolMC.pdf", bbox_inches="tight",
                dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("ultracoolMC.png", bbox_inches="tight",
                dpi=200)

# Done!
