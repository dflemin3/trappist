#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 26.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Read in evolutionary tracks
dataName = "trappist1WDelta20EpsBolmont"
dataDir = os.path.join("../../Data", dataName + "Evol.npz")
data = np.load(dataDir)
nsamples = len(data["Luminosity"])

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

# Luminosity from Grootel+2018
x = np.linspace(0, 1e10, 100)
axes[0].fill_between(x, 0.000522-0.000019, 0.000522+0.000019, color="C0",
                     alpha=0.3, zorder=0)
axes[0].axhline(0.000522, color="C0", lw=2, ls="--", zorder=2)

axes[0].set_ylabel("Luminosity [L$_{\odot}$]", fontsize=25)
axes[0].set_xlabel("Time [yr]", fontsize=25)
axes[0].set_ylim(4.0e-4, 1.5e-2)
axes[0].set_yscale("log")
axes[0].set_xscale("log")

# XUV Luminosity from Wheatley+2017
axes[1].fill_between(x, 10**(-6.4-0.1), 10**(-6.4+0.1), color="C0",
                     alpha=0.3, zorder=0)
axes[1].axhline(10**-6.4, color="C0", lw=2, ls="--", zorder=2)

axes[1].set_ylabel("XUV Luminosity [L$_{\odot}$]", fontsize=25)
axes[1].set_xlabel("Time [yr]", fontsize=25)
axes[1].set_ylim(2.0e-7, 2.0e-4)
axes[1].set_yscale("log")
axes[1].set_xscale("log")

# Radius from Grootel+2018
axes[2].fill_between(x, 0.121-0.003, 0.121+0.003, color="C0", alpha=0.3, zorder=0)
axes[2].axhline(0.121, color="C0", lw=2, ls="--", zorder=2)

axes[2].set_ylabel("Radius [R$_{\odot}$]", fontsize=25)
axes[2].set_xlabel("Time [yr]", fontsize=25)
axes[2].set_xscale("log")
axes[2].set_ylim(0.08, 0.42)

fig.savefig(os.path.join("../../Plots", dataName + "Trappist1Evol.png"),
            bbox_inches="tight", dpi=200)

### Plot runaway greenhouse HZ limit ###

fig, ax = plt.subplots(figsize=(7,6))

for ii in range(nsamples):

    # Plot inner HZ limit
    ax.plot(data["time"][ii], data["HZLimRunaway"][ii], alpha=0.1,
            color="k", lw=2, zorder=1)

    # Plot outer HZ limit
    ax.plot(data["time"][ii], data["HZLimEarlyMars"][ii], alpha=0.1,
            color="k", lw=2, zorder=1)

## Format, plot planet's semi-major axes from Gillon+2017 and Luger+2017
ba = .01111
baSig = 0.00034

ca = 0.01521
caSig = 0.00047

da = 0.02144
daSig = 0.00066

ea = 0.02817
eaSig = 0.00087

fa = 0.0371
faSig = 0.0011

ga = 0.0451
gaSig = 0.0014

ha = 0.06134
haSig = 0.002251

planets = [ba, ca, da, ea, fa, ga, ha]
planetsSig = [baSig, caSig, daSig, eaSig, faSig, gaSig, haSig]
planetsName = ["b", "c", "d", "e", "f", "g", "h"]
planetsColor = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

x = np.linspace(0, 1.0e10, 100)
for ii in range(len(planets)):
    ax.fill_between(x, planets[ii] - planetsSig[ii],
                    planets[ii] + planetsSig[ii], color=planetsColor[ii],
                    alpha=0.3, zorder=0)
    ax.axhline(planets[ii], color=planetsColor[ii], lw=2, ls="--", zorder=2,
               label=planetsName[ii])

ax.set_xscale("log")
ax.set_ylim(0, 0.12)
ax.set_xlabel("Time [yr]", fontsize=25)
ax.set_ylabel("Distance [AU]", fontsize=25)
ax.legend(loc="upper right", framealpha=0, fontsize=12)

fig.savefig(os.path.join("../../Plots", dataName + "Trappist1HZLimEvol.png"),
            bbox_inches="tight", dpi=200)

# Done!
