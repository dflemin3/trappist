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
m0 = 250                         # Initial size of training set
m = 100                          # Number of new points to find each iteration
nmax = 10                        # Maximum number of iterations
Dmax = 10.0                      # KL-Divergence convergence limit
kmax = 5                         # Number of iterations for Dmax convergence to kick in
seed = 90                        # RNG seed
nGPRestarts = 20                 # Number of times to restart GP hyperparameter optimizations
nMinObjRestarts = 10             # Number of times to restart objective fn minimization
optGPEveryN = 25                 # Optimize GP hyperparameters even this many iterations
nKLSamples = int(1.0e7)          # Number of samples from posterior to use to calculate KL-Divergence
bounds = ((0.07, 0.11),          # Prior bounds
          (-5.0, -1.0),
          (0.1, 12.0),
          (0.1, 12.0),
          (-2.0, 0.0))
algorithm = "bape"              # Use the Kandasamy et al. (2015) BAPE formalism

# Set RNG seed
np.random.seed(seed)

# emcee.EnsembleSampler, emcee.EnsembleSampler.run_mcmc and GMM parameters
samplerKwargs = {"nwalkers" : 100}
mcmcKwargs = {"iterations" : int(5.0e3)}
gmmKwargs = {"reg_covar" : 1.0e-5}

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
if not os.path.exists("apFModelCache.npz"):
    y = np.zeros(m0)
    theta = utility.latinHypercubeSampling(m0, bounds, criterion="maximin")
    for ii in range(m0):
        theta[ii,:] = trappist1.samplePriorTRAPPIST1()
        y[ii] = mcmcUtils.LnLike(theta[ii], **kwargs)[0] + trappist1.LnPriorTRAPPIST1(theta[ii], **kwargs)

else:
    print("Loading in cached simulations...")
    sims = np.load("apFModelCache.npz")
    theta = sims["theta"]
    y = sims["y"]

### Initialize GP ###

# Use ExpSquared kernel, the approxposterior default option
gp = gpUtils.defaultGP(theta, y)

# Initialize approxposterior
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=trappist1.LnPriorTRAPPIST1,
                            lnlike=mcmcUtils.LnLike,
                            priorSample=trappist1.samplePriorTRAPPIST1,
                            algorithm=algorithm)

# Run!
ap.run(m=m, nmax=nmax, Dmax=Dmax, kmax=kmax, bounds=bounds,  estBurnin=True,
       nKLSamples=nKLSamples, mcmcKwargs=mcmcKwargs, maxComp=12, thinChains=True,
       samplerKwargs=samplerKwargs, verbose=True, gmmKwargs=gmmKwargs,
       nGPRestarts=nGPRestarts, nMinObjRestarts=nMinObjRestarts,
       optGPEveryN=optGPEveryN, seed=seed, cache=True, **kwargs)
# Done!
