#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot convergence of stellar evolution quantities as a function of VPLanet's
timestepping parameter, dEta.

@author David P. Fleming, 2019
@email dflemin3 (at) uw (dot) edu
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 18.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

path = "../Sims/Convergence/"
dirs = ["Eta1", "Eta01", "Eta001", "Eta0001"]
labels = [r"$\eta = 1$", r"$\eta = 10^{-1}$", r"$\eta = 10^{-2}$",
          r"$\eta = 10^{-3}$"]
colors = ["C%d" % ii for ii in range(len(dirs))]

# Plot stellar convergence of luminosity, LXUV, radis
fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Load in "truth"
true = np.genfromtxt(os.path.join(path, "Eta00001", "Trappist.star.forward"), delimiter=" ")
trueLXUV = true[:,2]
trueLUM = true[:,1]
trueRAD = true[:,3]

# Loop over dirs contain same simulation, but different integration scaling factors
for ii, dir in enumerate(dirs):

    # Load data
    data = np.genfromtxt(os.path.join(path, dir, "Trappist.star.forward"), delimiter=" ")

    # Left: Luminosity
    axes[0].plot(data[:,0], np.fabs(data[:,1] - trueLUM)/trueLUM, lw=2, color=colors[ii], label=labels[ii])

    # Middle: Radius
    axes[1].plot(data[:,0], np.fabs(data[:,3] - trueRAD)/trueRAD, lw=2, color=colors[ii], label=labels[ii])

    # Right: LXUV
    axes[2].plot(data[:,0], np.fabs(data[:,2] - trueLXUV)/trueLXUV, lw=2, color=colors[ii], label=labels[ii])

# Format!
axes[0].set_xlabel("Time [yr]")
axes[0].set_xlim(data[1,0], data[-1,0])
axes[0].set_xscale("log")
axes[0].set_ylabel("Luminosity Relative Error")
axes[0].set_yscale("log")
axes[0].legend(loc="best", fontsize=10)

axes[1].set_xlabel("Time [yr]")
axes[1].set_xlim(data[1,0], data[-1,0])
axes[1].set_xscale("log")
axes[1].set_ylabel("Radius Relative Error")
axes[1].set_yscale("log")

axes[2].set_xlabel("Time [yr]")
axes[2].set_xlim(data[1,0], data[-1,0])
axes[2].set_xscale("log")
axes[2].set_ylabel("LXUV Error")
axes[2].set_yscale("log")

# Save!
fig.tight_layout()
fig.savefig("../Plots/trappist1Convergence.png", dpi=200, bbox_inches="tight")
