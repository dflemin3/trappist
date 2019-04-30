#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trappist1 b-h approxposterior run
"""

import os
from trappist import pool
from trappist import trappist1, mcmcUtils
from approxposterior import approx
import emcee, corner
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
bounds = ((0.1, 0.15),
          (-5.0, -2.0),
          (-0.3, 1.0),
          (1.0e-3, 8.0),
          (-2.0, 0.0))          # Prior bounds
algorithm = "bape"              # Use the Kandasamy et al. (2015) formalism

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

# Randomly sample initial conditions from the prior
# Evaluate forward model log likelihood + lnprior for each theta
if not os.path.exists("apFModelCache.npz"):
    y = np.zeros(m0)
    theta = np.zeros((m0, ndim))
    for ii in range(m0):
        theta[ii,:] = trappist1.samplePriorTRAPPIST1()
        y[ii] = ehimc.LnLike(theta[ii], **kwargs)[0] + trappist1.LnPriorTRAPPIST1(theta[ii], **kwargs)

else:
    print("Loading in cached simulations...")
    sims = np.load("apFModelCache.npz")
    theta = sims["theta"]
    y = sims["y"]

### Initialize GP ###

# Guess initial metric, or scale length of the covariances in loglikelihood space
initialMetric = [5.0*len(theta)**(1.0/theta.shape[-1]) for _ in range(theta.shape[-1])]

# Create kernel: We'll model coverianges in loglikelihood space using a
# Squared Expoential Kernel as we anticipate Gaussian-ish posterior
# distributions in our 2-dimensional parameter space
kernel = george.kernels.ExpSquaredKernel(initialMetric, ndim=ndim)

# Guess initial mean function
mean = np.mean(y)

# Create GP and compute the kernel
gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
gp.compute(theta)

# Initialize approxposterior
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=trappist1.LnPriorTRAPPIST1,
                            lnlike=ehimc.LnLike,
                            priorSample=trappist1.samplePriorTRAPPIST1,
                            algorithm=algorithm)

# Run!
ap.run(m=m, nmax=nmax, Dmax=Dmax, kmax=kmax, bounds=bounds,  estBurnin=True,
       nKLSamples=nKLSamples, mcmcKwargs=mcmcKwargs, maxComp=12, thinChains=True,
       samplerKwargs=samplerKwargs, verbose=True, gmmKwargs=gmmKwargs,
       nGPRestarts=nGPRestarts, nMinObjRestarts=nMinObjRestarts,
       optGPEveryN=optGPEveryN, seed=seed, cache=True, **kwargs)

# Check out the final posterior distribution!

# Load in chain from last iteration
reader = emcee.backends.HDFBackend(ap.backends[-1], read_only=True)
samples = reader.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

fig.savefig("apFinalPosterior.png", bbox_inches="tight") # Uncomment to save
# Done!
