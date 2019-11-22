#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4571 0.4403 0.4565 0.4299 0.4635 0.4397 0.455  0.4477 0.4505 0.4393
 0.4571 0.4442 0.4565 0.4455 0.4364 0.4491 0.437  0.4416 0.436  0.437
 0.4464 0.4403 0.4531 0.4365 0.442  0.4525 0.4409 0.4434 0.4455 0.4442
 0.4408 0.4416 0.4424 0.4499 0.4544 0.4554 0.4534 0.451  0.4468 0.4492
 0.4601 0.4531 0.4344 0.4437 0.4572 0.4366 0.4466 0.4539 0.4457 0.4547
 0.4574 0.4459 0.439  0.4495 0.4383 0.4447 0.4528 0.4558 0.4394 0.4523
 0.4643 0.449  0.4528 0.4496 0.4511 0.4488 0.4547 0.4405 0.4365 0.4592
 0.4535 0.4435 0.4545 0.4383 0.4514 0.4496 0.4445 0.457  0.4567 0.4535
 0.4432 0.4491 0.4563 0.4504 0.4397 0.4419 0.4505 0.4482 0.4394 0.4468
 0.4427 0.4401 0.4541 0.4595 0.4452 0.4532 0.4477 0.4578 0.4494 0.4509]
Mean acceptance fraction: 0.4478579999999999
Burnin, thin: 245 49
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [ 96.49808769 100.47998756  91.29981213  81.63147647  92.91741306]
Mean Number of iterations / tau: 92.56535538389817

"""

from approxposterior import approx, gpUtils
import sys
import os
import numpy as np
import corner
from scipy.optimize import rosen
from trappist.mcmcUtils import extractMCMCResults


# Define algorithm parameters
m0 = 250                                        # Initial size of training set
m = 50                                          # Number of new points to find each iteration
nmax = 5                                        # Maximum number of iterations
ndim = 5                                        # Dimensionality
bounds = list((-5,5) for _ in range(ndim))      # Prior bounds
algorithm = "alternate"                         # Use both bape and agp
seed = 57                                       # RNG seed

np.random.seed(seed)

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20 * ndim}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(1.0e4)} # emcee.EnsembleSampler.run_mcmc parameters


# Define lnprior, lnlike, and priorSample functions
def lnprior(x):
    if np.any(np.fabs(x) > 5):
        return -np.inf
    else:
        return 0.0


def lnlike(x):
    return -rosen(x)/100


def priorSample(n=1):
    return np.random.uniform(low=-5, high=5, size=(n,5)).squeeze()


# Only do this if the script hasn't been ran
if not os.path.exists("apRun4.h5"):

    # Sample initial conditions from prior
    theta = priorSample(m0)

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lnlike(theta[ii]) + lnprior(theta[ii])

    # Create the the default GP which uses an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y, white_noise=-10)

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                gp=gp,
                                lnprior=lnprior,
                                lnlike=lnlike,
                                priorSample=priorSample,
                                bounds=bounds,
                                algorithm=algorithm)

    # Run!
    ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=10, mcmcKwargs=mcmcKwargs,
           cache=True, samplerKwargs=samplerKwargs, verbose=True, thinChains=True,
           onlyLastMCMC=True, optGPEveryN=25, nMinObjRestarts=5,
           dropInitialTraining=True)

    # Load in chain from last iteration
    samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])
else:
    samples = samples = extractMCMCResults("apRun4.h5", blobsExist=False,
                                           thinChains=True, applyBurnin=True,
                                           verbose=True)

# Check out the final posterior distribution!
labels = ["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x_4$"]
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True, labels=labels)

# Overplot prior distributions on the marginal distributions
# Plot prior distributions on top of marginal posteriors
fig.axes[0].axhline(0.1, lw=2, color="C0")
fig.axes[6].axhline(0.1, lw=2, color="C0")
fig.axes[12].axhline(0.1, lw=2, color="C0")
fig.axes[18].axhline(0.1, lw=2, color="C0")
fig.axes[24].axhline(0.1, lw=2, color="C0")

# Save figure
if (sys.argv[1] == 'pdf'):
    fig.savefig("approxRD5Corner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("approxRD5Corner.png", bbox_inches="tight", dpi=200)
