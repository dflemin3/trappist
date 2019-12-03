#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4494 0.4452 0.4546 0.4537 0.4565 0.4617 0.4455 0.4473 0.4506 0.4508
 0.443  0.4345 0.4406 0.4483 0.4494 0.4606 0.4491 0.4476 0.4394 0.4414
 0.4444 0.4372 0.4459 0.4486 0.4465 0.4363 0.4585 0.431  0.4415 0.4494
 0.4497 0.4304 0.4428 0.4493 0.4548 0.455  0.4487 0.4554 0.44   0.4472
 0.4223 0.4491 0.4544 0.4455 0.4493 0.4408 0.4527 0.4458 0.4645 0.4507
 0.4396 0.4516 0.4368 0.4519 0.4411 0.4475 0.4377 0.4515 0.4525 0.4335
 0.4405 0.4497 0.4348 0.4524 0.4499 0.4463 0.4244 0.4473 0.4409 0.4426
 0.4493 0.4444 0.4458 0.4478 0.4477 0.4485 0.4541 0.4572 0.462  0.4544
 0.4546 0.4342 0.4519 0.4446 0.4453 0.4483 0.4532 0.4468 0.4422 0.4365
 0.4435 0.4439 0.4529 0.4541 0.4481 0.4481 0.4486 0.4367 0.4509 0.4455]
Mean acceptance fraction: 0.44670499999999996
Burnin, thin: 258 47
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [ 94.05469769 105.03476529  89.5575247   77.39899609  95.45051129]
Mean Number of iterations / tau: 92.2992990137355


"""

from approxposterior import approx, gpUtils
import sys
import os
import numpy as np
import corner
from scipy.optimize import rosen
from trappist.mcmcUtils import extractMCMCResults


# Define algorithm parameters
m0 = 100                                        # Initial size of training set
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
           onlyLastMCMC=True, optGPEveryN=25, nMinObjRestarts=5)

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
