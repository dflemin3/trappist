#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate's the distribution of TRAPPIST-1's XUV luminosity given the observations
from Wheatley et al. (2017), LXUV/LBOL in the range 6-9 x 10^-4, and the
observed bolometric luminosity from Van Grootel et al. (2018),
5.22 +/- 0.19 x 10^-4 Lsun.

We assume the Wheatley et al. (2017) LXUV/LBOL is distributed uniformly over the
quoted range and adopt the quoted normal distribution from Van Grootel et al.
(2018).

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu
"""


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 15.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Define observational parameters
W17Low = 6.0e-4
W17High = 9.0e-4
VG18Mean = 5.22e-4 # Lsun
VG18Sig = 0.19e-4 # Lsun

# Draw samples assuming Lbol ~ Van Grootel + 2018 distribution and the Wheatley
# 2017 values are uniformly distributed over the range
num = int(1.0e6)
LxuvLbolSamps = np.random.uniform(low=W17Low, high=W17High, size=num)
#LxuvLbolSamps = np.power(10.0, norm.rvs(loc=-6.4, scale=0.1, size=(num,)))
LbolSamps = norm.rvs(loc=VG18Mean, scale=VG18Sig, size=(num,))

# Convolve
Lxuv = LxuvLbolSamps * LbolSamps

# Visualize final distribution, compute statistics of interest
print("Mean LXUV x 10^-7: %e" % np.mean(Lxuv/1.0e-7))
print("Std LXUV x 10^-7: %e" % np.std(Lxuv/1.0e-7))

print("Mean log10LXUV: %e" % np.mean(np.log10(Lxuv)))
print("Std log10LXUV: %e" % np.std(np.log10(Lxuv)))

# Fit gaussian to log10Lxuv
muXUV, stdXUV = norm.fit(Lxuv/1.0e-7)

x = np.linspace(np.min(Lxuv/1.0e-7), np.max(Lxuv/1.0e-7), 250)
pdfXUV = norm.pdf(x, muXUV, stdXUV)

# Output fit parameters
print("XUV Gaussian Fit:", muXUV, stdXUV)

fig, ax = plt.subplots()

# Plot histogram of samples
ax.hist((Lxuv/1.0e-7), bins="auto", density=True, label="Convolved");

# Overplot gaussian fit on histograms
ax.plot(x, pdfXUV, color="k", ls="--", lw=2.5, label="Fit")

ax.set_ylabel("Density")
ax.set_xlabel(r"$L_{XUV} \times 10^{-7} [L_{\odot}]$")
ax.legend(loc="best")

fig.tight_layout()
plt.show()
