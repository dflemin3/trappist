#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trappist-1 stellar evolution VPLanet simulations with initial conditions
sampled from the posterior distribution.

@author: David P Fleming, 2019
"""

import os
import numpy as np
import vplot
import emcee
from trappist import trappist1, mcmcUtils

# Define run parameters
nsamples = 100

# RNG seed
seed = 17
np.random.seed(seed)
dataDir = "../Data/trappist1Fiducial.h5"

# Options
kwargs = trappist1.kwargsTRAPPIST1
kwargs["LnPrior"] = trappist1.LnPriorTRAPPIST1
kwargs["PriorSample"] = trappist1.samplePriorTRAPPIST1
PATH = os.path.dirname(os.path.abspath(__file__))
kwargs["PATH"] = PATH

# Get the input files, save them as strings
with open(os.path.join(PATH, "star.in"), 'r') as f:
    star_in = f.read()
    kwargs["STARIN"] = star_in
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    vpl_in = f.read()
    kwargs["VPLIN"] = vpl_in

# Check for output dir, make it if it doesn't already exist
if not os.path.exists(os.path.join(PATH, "output")):
    os.makedirs(os.path.join(PATH, "output"))

# Draw nsmaples from the posterior distributions
samples = mcmcUtils.extractMCMCResults(dataDir, blobsExist=False)

# Get random initial conditions from posterior
sampleInds = np.random.choice(np.arange(len(samples)), size=nsamples, replace=True)

# Containers
lum = []
lumXUV = []
radius = []
temp = []
time = []

ii = 0
while ii < nsamples:

    # Run simulations and collect output
    output = mcmcUtils.GetEvol(samples[sampleInds[ii],:], **kwargs)

    # If simulation succeeded, extract data, move on to the next one
    if output is not None:
        # Extract simulation data
        time.append(output.star.Time)
        lum.append(output.star.Luminosity)
        lumXUV.append(output.star.LXUVStellar)
        radius.append(output.star.Radius)
        temp.append(output.star.Temperature)

        ii = ii + 1

# Cache results
np.savez("../Data/trappist1StarEvol.npz", time=time, Luminosity=lum,
         LXUVStellar=lumXUV, Radius=radius, Temperature=temp)

# Done!
