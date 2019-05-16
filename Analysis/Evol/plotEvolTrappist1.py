#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the evolution of TRAPPIST-1 with initial conditions sampled from the
posterior distributions.

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Luminosity [Lsun] = 5.220000e-04 + 1.900000e-05 - 1.900000e-05
LXUV [Lsun] = 3.866823e-07 + 4.876776e-08 - 4.998834e-08
Radius [Rsun] = 0.111608 + 0.000750 - 0.000760

"""

import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from trappist import mcmcUtils, trappist1 as t1

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 22.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Read in evolutionary tracks
data = np.load("../../Data/trappist1StarEvol.npz")
nsamples = len(data["Luminosity"])

chains, blobs = mcmcUtils.extractMCMCResults("../../Data/trappist1Fiducial.h5",
                                             verbose=True, burn=500,
                                             thinChains=True, blobsExist=True)

### Plot Lum, LumXUV, radius evolution and compare to observations ###

fig, axes = plt.subplots(ncols=3, figsize=(21,6))

for ii in range(nsamples):

    # Left: lum
    axes[0].plot(data["time"][ii], data["Luminosity"][ii], alpha=0.1, color="k",
                 lw=2, zorder=1)

    # Middle: lumLUX
    axes[1].plot(data["time"][ii], data["LXUVStellar"][ii], alpha=0.3, color="k",
                 lw=2)

    # Middle: lumLUX
    axes[2].plot(data["time"][ii], data["Radius"][ii], alpha=0.1, color="k",
                 lw=2)

# Plot constraints, format

# Output constraints
lum = np.median(blobs[:,0])
lumPlus = np.percentile(blobs[:,0], 84) - lum
lumMinus = lum - np.percentile(blobs[:,0], 16)
print("Luminosity [Lsun] = %e + %e - %e" % (lum, lumPlus, lumMinus))

lxuv = np.median(blobs[:,1])
lxuvPlus = np.percentile(blobs[:,1], 84) - lxuv
lxuvMinus = lxuv - np.percentile(blobs[:,1], 16)
print("LXUV [Lsun] = %e + %e - %e" % (lxuv, lxuvPlus, lxuvMinus))

rad = np.median(blobs[:,2])
radPlus = np.percentile(blobs[:,2], 84) - rad
radMinus = rad - np.percentile(blobs[:,2], 16)
print("Radius [Rsun] = %lf + %lf - %lf" % (rad, radPlus, radMinus))

# Luminosity from Grootel+2018
x = np.linspace(0, 1.2e10, 100)

# Plot 1-3 sigmas
for ii in range(1,4):
    axes[0].fill_between(x, 0.000522-(ii*0.000019), 0.000522+(ii*0.000019), color="C0",
                         alpha=0.22, zorder=0)
axes[0].axhline(0.000522, color="C0", lw=2, ls="--", zorder=2)

axes[0].set_ylabel(r"Luminosity [L$_{\odot}$]", fontsize=25)
axes[0].set_xlabel("Time [yr]", fontsize=25)
axes[0].set_ylim(4.0e-4, 1.5e-2)
axes[0].set_yscale("log")
axes[0].set_xscale("log")

# Luminosity inset
axLum = fig.add_axes([0.225, 0.55, 0.085, 0.3])
axLum.hist(blobs[:,0], bins=20, orientation="horizontal", color="gray",
           range=[0.00046, 0.000584], alpha=0.5);
axLum.hist(blobs[:,0], bins=20, orientation="horizontal", color="k",
          range=[0.00046, 0.000584], lw=2, histtype="step");
axLum.axhline(0.000522, lw=3, ls="--", color="C0")
axLum.axhline(0.000522+0.000019, lw=3, ls="--", color="C0")
axLum.axhline(0.000522-0.000019, lw=3, ls="--", color="C0")

# Format inset
axLum.set_xlabel("")
axLum.set_ylabel("")
axLum.set_title(r"Luminosity [L$_{\odot}$]", fontsize=18)

y2 = [0.000462, 0.00052, 0.000582]
y2_labels = ["0.00050", "0.00052", "0.00054"]
axLum.set_yticks(y2)
axLum.set_yticklabels(y2_labels, minor=False, fontsize=18)
axLum.set_xticklabels([])

# XUV Luminosity from Wheatley+2017 1-3 Sigmas

for ii in range(1,4):
    axes[1].fill_between(x, 3.9e-7-(ii*0.5e-7), 3.9e-7+(ii*0.5e-7), color="C0",
                         alpha=0.22, zorder=0)
axes[1].axhline(10**-6.4, color="C0", lw=2, ls="--", zorder=2)

axes[1].set_ylabel(r"XUV Luminosity [L$_{\odot}$]", fontsize=25)
axes[1].set_xlabel("Time [yr]", fontsize=25)
axes[1].set_ylim(1.5e-7, 5.0e-5)
axes[1].set_yscale("log")
axes[1].set_xscale("log")

# LXUV inset
axLXUV = fig.add_axes([0.555, 0.55, 0.085, 0.3])
axLXUV.hist(blobs[:,1], bins=20, orientation="horizontal", color="gray",
           range=[2.3e-7, 5.5e-7], alpha=0.5);
axLXUV.hist(blobs[:,1], bins=20, orientation="horizontal", color="k",
           range=[2.3e-7, 5.5e-7], lw=2, histtype="step");
axLXUV.axhline(3.4e-7, lw=3, ls="--", color="C0")
axLXUV.axhline(3.9e-7, lw=3, ls="--", color="C0")
axLXUV.axhline(4.4e-7, lw=3, ls="--", color="C0")

# Format inset
axLXUV.set_xlabel("")
axLXUV.set_ylabel("")
axLXUV.set_title(r"$L_{XUV}$ [$10^{-7} L_{\odot}$]", fontsize=18)

y2 = [2.4e-7, 3.9e-7, 5.4e-7]
y2_labels = ["2.4", "3.9 " ,"5.4"]
axLXUV.set_yticks(y2)
axLXUV.set_yticklabels(y2_labels, minor=False, fontsize=18)
axLXUV.set_xticklabels([])

# Radius from Grootel+2018 1-3 sigmas
for ii in range(1,4):
    axes[2].fill_between(x, 0.121-(ii*0.003), 0.121+(ii*0.003), color="C0",
                         alpha=0.22, zorder=0)
axes[2].axhline(0.121, color="C0", lw=2, ls="--", zorder=2)

axes[2].set_ylabel(r"Radius [R$_{\odot}$]", fontsize=25)
axes[2].set_xlabel("Time [yr]", fontsize=25)
axes[2].set_xscale("log")
axes[2].set_ylim(0.09, 0.40)

# Radius inset
axRad = fig.add_axes([0.885, 0.55, 0.085, 0.3])
axRad.hist(blobs[:,2], bins=20, orientation="horizontal", color="gray",
           alpha=0.5);
axRad.hist(blobs[:,2], bins=20, orientation="horizontal", color="k", lw=2,
           histtype="step");
axRad.axhline(0.118, lw=3, ls="--", color="C0")
axRad.axhline(0.121, lw=3, ls="--", color="C0")
axRad.axhline(0.124, lw=3, ls="--", color="C0")

# Format inset
axRad.set_xlabel("")
axRad.set_ylabel("")
axRad.set_title(r"Radius [R$_{\odot}$]", fontsize=18)

radMed = np.mean(blobs[:,2])
y2 = [radMed, 0.121]
y2_labels = ["%.3lf" % radMed, "0.121"]
axRad.set_yticks(y2)
axRad.set_yticklabels(y2_labels, minor=False, fontsize=18)
axRad.set_xticklabels([])

fig.tight_layout()
plt.subplots_adjust(wspace=0.225)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("trappist1Evol.pdf", bbox_inches="tight",
                dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("trappist1Evol.png", bbox_inches="tight",
                dpi=200)
# Done!
