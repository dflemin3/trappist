#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions.

@author: David P. Fleming, 2019

Script output:

Number of iterations: 5000
Acceptance fraction for each walker:
[0.443  0.4596 0.4702 0.4488 0.4586 0.4456 0.4742 0.4556 0.471  0.4576
 0.4546 0.4554 0.4624 0.4636 0.4496 0.4666 0.4656 0.4626 0.4678 0.459
 0.4346 0.4548 0.4564 0.477  0.4402 0.4518 0.446  0.448  0.451  0.4644
 0.4558 0.4602 0.4306 0.4368 0.4694 0.475  0.4674 0.4338 0.47   0.4674
 0.4572 0.458  0.4612 0.455  0.4652 0.4644 0.4472 0.462  0.4428 0.4544
 0.4396 0.4564 0.4552 0.4488 0.4526 0.4734 0.4628 0.457  0.457  0.4622
 0.4632 0.456  0.468  0.4634 0.464  0.4606 0.4518 0.446  0.4796 0.0064
 0.459  0.46   0.4758 0.4638 0.4592 0.448  0.4606 0.4378 0.4544 0.4706
 0.4532 0.4402 0.4534 0.4746 0.4592 0.4394 0.4404 0.4688 0.451  0.4624
 0.4554 0.4522 0.4664 0.4616 0.4768 0.4744 0.4766 0.464  0.4666 0.4472]
Mean acceptance fraction: 0.45346400000000003
Burnin, thin: 500 28
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [88.50884821 47.36353897 39.82590807 68.37328676 72.91224171]
Mean Number of iterations / tau: 63.39676474341238
P(tsat >= age | data) = 0.357
P(tsat <= 1Gyr | data) = 0.000
Quantiles:
[(0.16, 8.838475456368696), (0.5, 8.893464138990819), (0.84, 8.949730763349681)]
Quantiles:
[(0.16, -3.1301672913045318), (0.5, -3.023720769774166), (0.84, -2.8292393630383925)]
Quantiles:
[(0.16, 3.995229018508609), (0.5, 6.460306939252781), (0.84, 10.072290899634773)]
Quantiles:
[(0.16, 6.1602266787201625), (0.5, 7.6041374512502), (0.84, 9.05215882464596)]
Quantiles:
[(0.16, -1.3792802884504942), (0.5, -1.170167023313772), (0.84, -0.9561062896150441)]

"""

import numpy as np
import os
import sys
import corner
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
from trappist.mcmcUtils import extractMCMCResults

#Typical plot parameters that make for pretty plots
mpl.rcParams['font.size'] = 17

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/apRun9.h5"

# Extract data
samples = extractMCMCResults(filename, blobsExist=False, burn=500)

labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
          r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# Scale Mass for readability
samples[:,0] = samples[:,0] * 1.0e2

# Estimate probability that TRAPPIST-1 is still saturated at the age of the
# system using samples from the posterior distribution
mask = samples[:,2] >= samples[:,3]
print("P(tsat >= age | data) = %0.3lf" % np.mean(mask))

# Estimate probability that tsat < 1 Gyr
mask = samples[:,2] <= 1
print("P(tsat <= 1Gyr | data) = %0.3lf" % np.mean(mask))

# Plot!
range = [[8.7, 9.06], [-3.3, -2.2], [0, 12], [0, 12], [-2, -0.2]]
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 16}, range=range,
                    title_fmt='.2f', verbose=True, hist_kwargs={"linewidth" : 1.5})

# Fine-tune the formatting
ax_list = fig.axes
ax_list[0].set_title(r"$m_{\star}$ [M$_{\odot}$] $= 0.089^{+0.0005}_{-0.0005}$", fontsize=16)
ax_list[-5].set_xlabel(r"$m_{\star}$ [$100\times$ M$_{\odot}$]", fontsize=22)
ax_list[-4].set_xlabel(r"$f_{sat}$", fontsize=22)
ax_list[-3].set_xlabel(r"$t_{sat}$ [Gyr]", fontsize=22)
ax_list[-2].set_xlabel(r"age [Gyr]", fontsize=22)
ax_list[-1].set_xlabel(r"$\beta_{XUV}$", fontsize=22)

ax_list[5].set_ylabel(r"$f_{sat}$", fontsize=22)
ax_list[10].set_ylabel(r"$t_{sat}$ [Gyr]", fontsize=22)
ax_list[15].set_ylabel(r"age [Gyr]", fontsize=22)
ax_list[20].set_ylabel(r"$\beta_{XUV}$", fontsize=22)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("apCorner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("apCorner.png", bbox_inches="tight", dpi=200)

# Done!
