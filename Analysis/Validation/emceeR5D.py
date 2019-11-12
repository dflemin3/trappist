#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Bayesian posterior inference with a 5-dimensional Rosenbrock function using emcee.

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

Script output:

Number of iterations: 10000
Acceptance fraction for each walker:
[0.4474 0.4449 0.4562 0.4495 0.4495 0.4441 0.4352 0.4517 0.4425 0.4516
 0.4471 0.4529 0.4495 0.4425 0.4487 0.4595 0.4356 0.4495 0.4385 0.4436
 0.4424 0.4433 0.4487 0.4405 0.433  0.4483 0.4413 0.439  0.4355 0.441
 0.4548 0.4487 0.4367 0.4552 0.448  0.4544 0.4523 0.4495 0.4394 0.4624
 0.4471 0.4574 0.4484 0.4504 0.4398 0.4494 0.4422 0.4409 0.4453 0.4577
 0.4493 0.4415 0.4343 0.4529 0.4497 0.4497 0.4495 0.4462 0.4351 0.4428
 0.4473 0.4527 0.4438 0.4428 0.4509 0.4514 0.4478 0.4424 0.4494 0.4506
 0.4475 0.4527 0.4363 0.4471 0.4549 0.4549 0.4219 0.4369 0.4531 0.4435
 0.4535 0.447  0.4492 0.4531 0.4404 0.454  0.4418 0.4525 0.4561 0.4505
 0.4479 0.4584 0.4437 0.4549 0.4472 0.448  0.4547 0.4458 0.4427 0.4516]
Mean acceptance fraction: 0.44704900000000003
Burnin, thin: 254 49
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [ 97.66442009 100.08545413  94.17604387  78.68165842  93.65741193]
Mean Number of iterations / tau: 92.85299768948485


"""

import numpy as np
import emcee
import corner
import os
import sys
from scipy.optimize import rosen
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
    if np.any(np.fabs(x) > 5):
        return -np.inf
    else:
        return -rosen(x)/100
# end function

# Initial guess for walkers (random over prior)
p0 = [np.random.uniform(low=-5, high=5, size=(ndim)) for j in range(nwalk)]

# Initialize sampler and save results as an HDF5 file
filename = "emceeRosen5D.h5"

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
    fig.savefig("emceeRD5Corner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("emceeRD5Corner.png", bbox_inches="tight", dpi=200)
