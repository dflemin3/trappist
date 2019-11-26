#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4415 0.4436 0.4458 0.4458 0.4518 0.4421 0.4534 0.4495 0.4533 0.4526
 0.451  0.4551 0.4464 0.4435 0.4512 0.4444 0.4542 0.4534 0.4561 0.4406
 0.4619 0.4332 0.45   0.4499 0.4344 0.4519 0.4516 0.4463 0.4515 0.4435
 0.4325 0.4413 0.4387 0.4429 0.4524 0.4419 0.4451 0.454  0.4431 0.4532
 0.4411 0.4479 0.4458 0.4469 0.4429 0.4509 0.4267 0.4524 0.4462 0.4435
 0.451  0.4411 0.4461 0.4559 0.4366 0.4506 0.4275 0.4544 0.4433 0.4511
 0.4274 0.4485 0.4316 0.4486 0.4465 0.4442 0.4333 0.442  0.4489 0.4323
 0.4356 0.4448 0.4317 0.4514 0.4488 0.4453 0.4452 0.4585 0.4535 0.4418
 0.4491 0.4489 0.4379 0.4499 0.452  0.4549 0.4527 0.4564 0.449  0.4344
 0.4475 0.4531 0.4532 0.4523 0.4415 0.4436 0.4495 0.4484 0.4503 0.4489]
Mean acceptance fraction: 0.446299
Burnin, thin: 249 46
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [101.8065953  108.33310137  94.35649071  80.14005706  94.21795778]
Mean Number of iterations / tau: 95.77084044212332

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
seed = 70                                       # RNG seed

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
    gp = gpUtils.defaultGP(theta, y, white_noise=-10, fitAmp=True)

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
    ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=5, mcmcKwargs=mcmcKwargs,
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
    fig.savefig("apCorner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("apCorner.png", bbox_inches="tight", dpi=200)
