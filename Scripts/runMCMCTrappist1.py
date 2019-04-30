#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trappist1 stellar evolution MCMC run

@author: dflemin3
"""

import os
from trappist import pool
from trappist import trappist1, mcmcUtils

# Define run parameters
ndim = 5
nwalk = 40
nsteps = 10
nsamples = 0
restart = False
backend = "trappist1.h5"

# Open a pool, and let it rip!
with pool.Pool(pool='SerialPool') as pool:

    # Options
    kwargs = trappist1.kwargsTRAPPIST1
    kwargs["nsteps"] = nsteps
    kwargs["nsamples"] = nsamples
    kwargs["nwalk"] = nwalk
    kwargs["pool"] = pool
    kwargs["restart"] = restart
    kwargs["LnPrior"] = trappist1.LnPriorTRAPPIST1
    kwargs["PriorSample"] = trappist1.samplePriorTRAPPIST1
    PATH = os.path.dirname(os.path.abspath(__file__))
    kwargs["PATH"] = PATH
    kwargs["backend"] = backend

    # Check for output dir, make it if it doesn't already exist
    if not os.path.exists(os.path.join(PATH, "output")):
        os.makedirs(os.path.join(PATH, "output"))

    # Run
    mcmcUtils.RunMCMC(**kwargs)

# Done!