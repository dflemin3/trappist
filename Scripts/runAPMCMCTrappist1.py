#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRAPPIST-1 approxposterior run script.

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

"""

import os
from trappist import pool
from trappist import trappist1, mcmcUtils
from approxposterior import approx, utility, gpUtils
import emcee
import numpy as np
import george

# Define algorithm parameters
ndim = 5                         # Dimensionality of the problem
m0 = 50                          # Initial size of training set
m = 100                          # Number of new points to find each iteration
nmax = 20                        # Maximum number of iterations
seed = 91                        # RNG seed
nGPRestarts = 10                 # Number of times to restart GP hyperparameter optimizations
nMinObjRestarts = 5              # Number of times to restart objective fn minimization
optGPEveryN = 10                 # Optimize GP hyperparameters even this many iterations
bounds = ((0.07, 0.11),          # Prior bounds
          (-5.0, -1.0),
          (0.1, 12.0),
          (0.1, 12.0),
          (-2.0, 0.0))
algorithm = "bape"               # Kandasamy et al. (2017) formalism

# Set RNG seed
np.random.seed(seed)

# emcee.EnsembleSampler, emcee.EnsembleSampler.run_mcmc and GMM parameters
samplerKwargs = {"nwalkers" : 100}
mcmcKwargs = {"iterations" : int(1.0e4)}

# Loglikelihood function setup required to run VPLanet simulations
kwargs = trappist1.kwargsTRAPPIST1
PATH = os.path.dirname(os.path.abspath(__file__))
kwargs["PATH"] = PATH

# Get the input files, save them as strings
with open(os.path.join(PATH, "star.in"), 'r') as f:
    star_in = f.read()
    kwargs["STARIN"] = star_in
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    vpl_in = f.read()
    kwargs["VPLIN"] = vpl_in

# Generate initial training set using latin hypercube sampling over parameter bounds
# Evaluate forward model log likelihood + lnprior for each theta
if not os.path.exists("apRunAPFModelCache.npz"):
    y = np.zeros(m0)
    theta = np.zeros((m0,5))

    for ii in range(m0):
        theta[ii] = trappist1.samplePriorTRAPPIST1()
        y[ii] = mcmcUtils.LnLike(theta[ii], **kwargs)[0] + trappist1.LnPriorTRAPPIST1(theta[ii], **kwargs)
    np.savez("apRunAPFModelCache.npz", theta=theta, y=y)

else:
    print("Loading in cached simulations...")
    sims = np.load("apRunAPFModelCache.npz")
    theta = sims["theta"]
    y = sims["y"]

### Initialize GP ###

# Use ExpSquared kernel, the approxposterior default option
gp = gpUtils.defaultGP(theta, y, white_noise=-15, fitAmp=False)

# Initialize approxposterior
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=trappist1.LnPriorTRAPPIST1,
                            lnlike=mcmcUtils.LnLike,
                            priorSample=trappist1.samplePriorTRAPPIST1,
                            bounds=bounds,
                            algorithm=algorithm)

# Run!
ap.run(m=m, nmax=nmax, estBurnin=True, mcmcKwargs=mcmcKwargs, thinChains=True,
       samplerKwargs=samplerKwargs, verbose=True, nGPRestarts=nGPRestarts,
       nMinObjRestarts=nMinObjRestarts, optGPEveryN=optGPEveryN,
       seed=seed, cache=True, kmax=5, eps=0.1, convergenceCheck=True, **kwargs)
# Done!
