#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trappist1 runs with initial conditions sampled from posterior dist
"""

import os
import numpy as np
import vplot
import emcee
from ehi import trappist1, mcmcUtils

# Define run parameters
nsamples = 100
planetList = ["trappist1b.in", "trappist1c.in", "trappist1d.in", "trappist1e.in",
              "trappist1f.in", "trappist1g.in", "trappist1h.in"]
# RNG seed
seed = 90
np.random.seed(seed)
dataDir = "../../Data/trappist1W20EpsBolmont.h5"

# Options
kwargs = trappist1.kwargsTRAPPIST1
kwargs["LnPrior"] = trappist1.LnPriorTRAPPIST1
kwargs["PriorSample"] = trappist1.samplePriorTRAPPIST1
kwargs["WaterPrior"] = mcmcUtils.waterPriorDeltaSample
PATH = os.path.dirname(os.path.abspath(__file__))
kwargs["PATH"] = PATH
kwargs["planetList"] = planetList

# Get the input files, save them as strings
planet_ins = []
for planet in planetList:
    with open(os.path.join(PATH, planet), 'r') as f:
        planet_ins.append(f.read())
    kwargs["PLANETIN"] = planet_ins
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
reader = emcee.backends.HDFBackend(dataDir)
tau = reader.get_autocorr_time()
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

# Get random initial conditions from posterior
sampleInds = np.random.choice(np.arange(len(samples)), size=nsamples, replace=False)

# Containers
lum = []
lumXUV = []
radius = []
temp = []
hzlimrun = []
hzlimoutrun = []
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
        hzlimrun.append(output.star.HZLimRunaway)
        hzlimoutrun.append(output.star.HZLimEarlyMars)

        ii = ii + 1

# Cache results
np.savez("../../Data/trappistStarEvol.npz", time=time, Luminosity=lum,
         LXUVStellar=lumXUV, Radius=radius, Temperature=temp,
         HZLimRunaway=hzlimrun, HZLimEarlyMars=hzlimoutrun)

# Done!
