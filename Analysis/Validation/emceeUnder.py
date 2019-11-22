#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Bayesian posterior inference with a 5-dimensional Rosenbrock function using emcee.

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4511 0.4467 0.4524 0.4538 0.4571 0.448  0.4533 0.4528 0.4643 0.4551
 0.4491 0.4553 0.4475 0.4701 0.4604 0.4476 0.4401 0.4436 0.4473 0.4511
 0.4692 0.4481 0.4505 0.457  0.4607 0.4562 0.4617 0.4512 0.4568 0.4417
 0.4586 0.4504 0.4535 0.4588 0.4485 0.4525 0.4742 0.4619 0.4622 0.4468
 0.4462 0.4562 0.4544 0.449  0.4434 0.4318 0.4626 0.4551 0.4515 0.4739
 0.4456 0.4495 0.4543 0.4549 0.4532 0.4557 0.4523 0.4705 0.4477 0.4514
 0.4524 0.4544 0.4616 0.4453 0.4701 0.4613 0.4601 0.462  0.4569 0.4478
 0.4615 0.4447 0.448  0.4511 0.4617 0.4581 0.4594 0.4377 0.4644 0.4603
 0.4515 0.4458 0.4556 0.4518 0.4579 0.454  0.4436 0.4348 0.4483 0.4591
 0.4445 0.4503 0.4572 0.4677 0.4498 0.461  0.4659 0.4454 0.4646 0.4489]
Mean acceptance fraction: 0.45402899999999996
Burnin, thin: 274 36
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [ 72.89381859  81.71455823  89.37448911 136.40635668 132.44481868]
Mean Number of iterations / tau: 102.56680825837005

"""

import numpy as np
import emcee
import corner
import os
import sys
from scipy.optimize import rosen
from scipy.stats import norm
from trappist import mcmcUtils
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 17})

# Define inference parameters
ndim = 5          # Number of dimensions
nsteps = 10000    # Number of MCMC iterations
nwalk = 20 * ndim # Use 20 walkers per dimension

# Define logprobability function
def lnprob(x):

    x0, x1, x2, x3, x4 = x

    # Apply standard normal prior on final two dimensions
    lp = norm.logpdf(x3, loc=0, scale=1)
    lp += norm.logpdf(x4, loc=0, scale=1)

    # Only 1st 3 dimensions participate in likelihood calculation
    x = [x0, x1, x2]

    # Uniform prior over remaining parameters
    if np.any(np.fabs(x) > 5):
        return -np.inf
    else:
        return -rosen(x)/100 + lp
# end function

# Initial guess for walkers (random over prior)
p0 = [np.random.uniform(low=-5, high=5, size=(ndim)) for j in range(nwalk)]

# Initialize sampler and save results as an HDF5 file
filename = "emceeUnder.h5"

# Only actually run MCMC inference if it hasn't be done already
if not os.path.exists(filename):
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob,
                                    backend=emcee.backends.HDFBackend(filename))
    sampler.run_mcmc(p0, nsteps)

# Load in and inspect the final posterior distribution!
samples = mcmcUtils.extractMCMCResults(filename, verbose=True, applyBurnin=True,
                                       thinChains=True, blobsExist=False,
                                       burn=None, removeRogueChains=False)

# Corner plot!
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
    fig.savefig("emceeUnderCorner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("emceeUnderCorner.png", bbox_inches="tight", dpi=200)
