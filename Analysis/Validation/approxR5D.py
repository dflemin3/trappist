#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4703 0.441  0.4538 0.4522 0.4545 0.4428 0.4588 0.4483 0.4406 0.4505
 0.4348 0.4475 0.4474 0.4463 0.4549 0.4442 0.4352 0.4411 0.4436 0.4615
 0.4434 0.4521 0.4474 0.4445 0.4507 0.4291 0.4655 0.4262 0.4363 0.4378
 0.4489 0.4485 0.446  0.4569 0.4439 0.449  0.4455 0.447  0.4355 0.4529
 0.4591 0.4537 0.4501 0.4433 0.4527 0.4537 0.4547 0.4516 0.4538 0.4455
 0.4382 0.4511 0.4526 0.4532 0.4499 0.4459 0.4497 0.4439 0.4471 0.457
 0.4458 0.4391 0.45   0.4471 0.4235 0.4535 0.4577 0.4359 0.452  0.431
 0.4532 0.4506 0.4283 0.4503 0.4543 0.4479 0.4479 0.4364 0.4565 0.4491
 0.4608 0.4428 0.4275 0.437  0.4583 0.4255 0.4448 0.4569 0.4462 0.4323
 0.434  0.4518 0.4387 0.4457 0.4451 0.4303 0.4335 0.4348 0.4506 0.4401]
Mean acceptance fraction: 0.4463000000000001
Burnin, thin: 239 50
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [98.05706859 99.68684042 90.50281136 83.56602272 90.26758921]
Mean Number of iterations / tau: 92.41606645824956

"""

from approxposterior import approx, gpUtils
import sys
import os
import numpy as np
import corner
from scipy.optimize import rosen
from trappist.mcmcUtils import extractMCMCResults


# Define algorithm parameters
m0 = 150                                        # Initial size of training set
m = 50                                          # Number of new points to find each iteration
nmax = 10                                       # Maximum number of iterations
ndim = 5                                        # Dimensionality
bounds = list((-5,5) for _ in range(ndim))      # Prior bounds
algorithm = "bape"                              # Use the Kandasamy et al. (2015) formalism
seed = 27                                       # RNG seed

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
if not os.path.exists("apRun9.h5"):

    # Sample initial conditions from prior
    theta = priorSample(m0)

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lnlike(theta[ii]) + lnprior(theta[ii])

    # Create the the default GP which uses an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y, white_noise=-15)

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
           onlyLastMCMC=True, optGPEveryN=25, nMinObjRestarts=5)

    # Load in chain from last iteration
    samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])
else:
    samples = samples = extractMCMCResults("apRun9.h5", blobsExist=False,
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
