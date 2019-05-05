#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make a corner plot of the MCMC-derived posterior distributions.

@author: David P. Fleming, 2019

Script Output:

Number of iterations: 5000
Acceptance fraction for each walker:
[0.5054 0.5126 0.4994 0.52   0.5022 0.496  0.5062 0.5042 0.519  0.5174
 0.5078 0.511  0.5058 0.502  0.5172 0.4932 0.4928 0.4926 0.5096 0.4978
 0.5128 0.5192 0.4876 0.5152 0.5154 0.509  0.5156 0.518  0.515  0.5252
 0.509  0.51   0.5182 0.521  0.5208 0.5082 0.5118 0.509  0.515  0.5248
 0.5032 0.5164 0.5046 0.5202 0.5222 0.517  0.5154 0.5058 0.5032 0.5002
 0.5196 0.509  0.4952 0.5142 0.5194 0.5076 0.515  0.5226 0.5098 0.5148
 0.4998 0.517  0.5102 0.5052 0.5164 0.526  0.5126 0.5188 0.5218 0.5248
 0.5142 0.5014 0.4988 0.5348 0.504  0.5182 0.524  0.509  0.5168 0.5274
 0.4954 0.5072 0.5134 0.5112 0.5068 0.5034 0.5164 0.5266 0.5014 0.5124
 0.5042 0.5178 0.5152 0.518  0.5092 0.5172 0.508  0.523  0.5142 0.5176]
Mean acceptance fraction: 0.5116820000000001
Burnin, thin: 134 25
Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.
Number of iterations / tau: [97.70371834 81.97709956 74.38484188 90.89447927 85.56903943]
Mean Number of iterations / tau: 86.10583569702148

P(tsat >= age | data) = 0.606
P(tsat <= 1Gyr | data) = 0.000

Quantiles:
[(0.16, 0.08838708993607726), (0.5, 0.08893192062272381), (0.84, 0.08949327247272604)]
Quantiles:
[(0.16, -3.1109144853830353), (0.5, -3.0565854535385983), (0.84, -3.001718452419834)]
Quantiles:
[(0.16, 5.465865131230896), (0.5, 8.225280779687864), (0.84, 10.763059225706915)]
Quantiles:
[(0.16, 4.976807837867465), (0.5, 7.140779234838001), (0.84, 9.12832556937305)]
Quantiles:
[(0.16, -1.462264301177495), (0.5, -1.151435363351926), (0.84, -0.8365286850095074)]

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
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 15.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1Fiducial.h5"

# Whether or not to plot blobs
plotBlobs = False

# Extract results
if plotBlobs:
    chain, blobs = extractMCMCResults(filename, blobsExist=plotBlobs)
else:
    chain = extractMCMCResults(filename, blobsExist=plotBlobs)


if plotBlobs:
    samples = np.hstack([chain, blobs])

    # Define Axis Labels
    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "Lum",
              "logLumXUV", "Radius"]

    # Make luminosity units more palatable
    samples[:,5] = samples[:,5]*1.0e3
else:
    # Just consider stellar data
    samples = chain
    labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
              r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# Estimate probability that TRAPPIST-1 is still saturated at the age of the
# system using the posterior distribution
mask = samples[:,2] >= samples[:,3]
print("P(tsat >= age | data) = %0.3lf" % np.mean(mask))

# Estimate probability that tsat < 1 Gyr
mask = samples[:,2] <= 1
print("P(tsat <= 1Gyr | data) = %0.3lf" % np.mean(mask))

# Plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 14},
                    title_fmt='.2f', verbose=True)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("trappist1Corner.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("trappist1Corner.png", bbox_inches="tight", dpi=200)

# Done!
