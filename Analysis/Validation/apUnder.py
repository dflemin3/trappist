#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4382 0.4493 0.4593 0.4562 0.4593 0.4646 0.4411 0.4575 0.4517 0.4445
 0.4491 0.4575 0.4498 0.4558 0.451  0.4399 0.4542 0.4613 0.4637 0.4588
 0.4598 0.4338 0.4474 0.4522 0.4485 0.4457 0.4547 0.4486 0.4631 0.4496
 0.4411 0.4575 0.4611 0.4629 0.4449 0.4534 0.455  0.4552 0.4323 0.4425
 0.4556 0.4604 0.4479 0.4649 0.4444 0.4693 0.4498 0.4599 0.4512 0.4534
 0.4579 0.4586 0.4539 0.4607 0.4515 0.4598 0.4345 0.4655 0.4568 0.4444
 0.4453 0.4587 0.4397 0.4438 0.4574 0.4551 0.4402 0.4586 0.4589 0.4506
 0.4558 0.4454 0.4449 0.4537 0.4442 0.4541 0.4611 0.4644 0.447  0.4535
 0.4547 0.4489 0.4504 0.4524 0.4592 0.4491 0.4572 0.4478 0.445  0.4486
 0.4589 0.4502 0.4362 0.4556 0.446  0.4579 0.4567 0.4666 0.4655 0.4371]
Mean acceptance fraction: 0.452489
Burnin, thin: 266 36
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [ 75.10029811  79.23617561  86.76665557 133.56744333 137.75846948]
Mean Number of iterations / tau: 102.48580841985279

"""

from approxposterior import approx, gpUtils
import sys
import os
import numpy as np
import corner
from scipy.optimize import rosen
from scipy.stats import norm
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

    x0, x1, x2, x3, x4 = x

    # Apply standard normal prior on final two dimensions
    lp = norm.logpdf(x3, loc=0, scale=1)
    lp += norm.logpdf(x4, loc=0, scale=1)

    x = [x0, x1, x2]

    if np.any(np.fabs(x) > 5):
        return -np.inf
    else:
        return lp


def lnlike(x):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    else:
        x0, x1, x2, x3, x4 = x
        x = [x0, x1, x2]
        return -rosen(x)/100


def priorSample(n=1):
    ret = []

    for _ in range(n):
        while True:
            tmp = [np.random.uniform(low=-5, high=5),
                   np.random.uniform(low=-5, high=5),
                   np.random.uniform(low=-5, high=5),
                   np.random.randn(),
                   np.random.randn()]
            if np.isfinite(lnprior(tmp)):
                ret.append(tmp)
                break
    return np.array(ret).squeeze()


# Only do this if the script hasn't been ran
if not os.path.exists("under4.h5"):

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
    ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=5, mcmcKwargs=mcmcKwargs,
           cache=True, samplerKwargs=samplerKwargs, verbose=True, thinChains=True,
           onlyLastMCMC=True, optGPEveryN=25, nMinObjRestarts=5, runName="under")

    # Load in chain from last iteration
    samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])
else:
    samples = extractMCMCResults("under4.h5", blobsExist=False,
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
    fig.savefig("apUnderCorner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("apUnderCorner.png", bbox_inches="tight", dpi=200)
