#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import os
import corner
import emcee
from statsmodels.stats.proportion import proportion_confint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ehi.mcmcUtils import extractMCMCResults

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 15.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/apRun9.h5"#"../../Data/TRAPPIST1/trappist1WRaymond2007EpsBolmont.h5"

# Whether or not to plot blobs
plotBlobs = False

# Extract results
if plotBlobs:
    chain, blobs = extractMCMCResults(filename, blobsExist=plotBlobs)
else:
    chain = extractMCMCResults(filename, blobsExist=plotBlobs)
if plotBlobs:
    # Select correct columns
    mask = [0, 1, 19, 20, 21, 22, 44, 45, 46, 47, 48, 49, 50]
    initWaterMask = [30, 31, 32, 33, 34, 35, 36]
    finalWaterMask = [37, 38, 39, 40, 41, 42, 43]
    initWater = blobs[:,initWaterMask]
    finalWater = blobs[:,finalWaterMask]
    deltaWater = finalWater - initWater
    blobs = blobs[:,mask]

    # Combine!
    samples = np.hstack([chain, blobs, deltaWater])

    # Define Axis Labels
    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "Lum",
              "logLumXUV", "dRGTimee", "dRGTimef", "dRGTimeg", "dRGTimeh",
              "OxygenMassb", "OxygenMassc", "OxygenMassd", "OxygenMasse",
              "OxygenMassf", "OxygenMassg", "OxygenMassh", "DeltaH20b", "DeltaH20c",
             " DeltaH20d", "DeltaH20e", "DeltaH20f", "DeltaH20g", "DeltaH20h"]

    # Convert RG Time to Myr
    samples[:,7] = samples[:,7]/1.0e6
    samples[:,8] = samples[:,8]/1.0e6
    samples[:,9] = samples[:,9]/1.0e6
    samples[:,10] = samples[:,10]/1.0e6

    # Make luminosity units more palatable
    samples[:,5] = samples[:,5]*1.0e3
else:
    # Just consider stellar data
    samples = chain
    labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
              r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# Plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 14})

# Save!
fig.savefig("../../Plots/trapist1UpdatedAPRun.png", bbox_inches="tight", dpi=200)
#fig.savefig("../../Plots/trappist1CornerWRaymond2007EpsBolmont.png", bbox_inches="tight", dpi=200)

# Done!
