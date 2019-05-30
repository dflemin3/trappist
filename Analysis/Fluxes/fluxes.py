#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the distribution of the XUV fluxes the TRAPPIST-1 planets receive at
10, 100, and 1,000 Myr since TRAPPIST-1 entered the pre-main sequence.

@author: David P. Fleming, 2019
@email: dflemin3 (at) uw (dot) edu

"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib as mpl
import matplotlib.patches as mpatches
import joypy as jp
from scipy.stats import norm

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 22.0
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)


# Constants
LSUN = 3.848e33 # luminosity of the sun in ergs per second
AUCM = 1.496e13 # 1 AU in cm
MSUN = 1.988435e33 # mass of sun in grams
BIGG = 6.674e-8 # gravitational constant in cgs
DAYSEC = 3600.0 * 24.0 # seconds in a day

# From Wheatley+2017: The Earth typically receives 0.85 erg s−1 cm−2 in X-ray
# and 3.03 erg s−1 cm−2 in EUV in mid-Solar cycle (Ribas et al. 2005)
EARTHFLUXX = 0.85 # ergs/s/cm^2
EARTHFLUXEUV = 3.03 # ergs/s/cm^2
EARTHXUV = EARTHFLUXX + EARTHFLUXEUV

# TRAPPIST-1 planet orbital periods and uncertainties

# b
porbTrappist1b = 1.51087081
porbTrappist1bSig = 0.6e-6

# c
porbTrappist1c = 2.4218233
porbTrappist1cSig = 0.17e-5

# d
porbTrappist1d = 4.049610
porbTrappist1dSig = 0.63e-4

# e
porbTrappist1e = 6.099615
porbTrappist1eSig = 0.11e-4

# f
porbTrappist1f = 9.20669
porbTrappist1fSig = 0.15e-4

# g
porbTrappist1g = 12.35294
porbTrappist1gSig = 0.12e-3

# h
porbTrappist1h = 18.767
porbTrappist1hSig = 0.004


# Define helper functions
def Trappist1PlanetPorbSample(planet, size=1, **kwargs):
    """
    Sample Trappist1 system planet Porbs from Gaussian distributions based on
    Gillon+2017, Luger+2017 measurements.
    """

    # Light preprocessing of planet name
    name = str(planet).lower()

    ret = []
    for ii in range(size):
        if name == "trappist1b":
            ret.append(norm.rvs(loc=porbTrappist1b, scale=porbTrappist1bSig, size=1)[0])
        elif name == "trappist1c":
            ret.append(norm.rvs(loc=porbTrappist1c, scale=porbTrappist1cSig, size=1)[0])
        elif name == "trappist1d":
            ret.append(norm.rvs(loc=porbTrappist1d, scale=porbTrappist1dSig, size=1)[0])
        elif name == "trappist1e":
            ret.append(norm.rvs(loc=porbTrappist1e, scale=porbTrappist1eSig, size=1)[0])
        elif name == "trappist1f":
            ret.append(norm.rvs(loc=porbTrappist1f, scale=porbTrappist1fSig, size=1)[0])
        elif name == "trappist1g":
            ret.append(norm.rvs(loc=porbTrappist1g, scale=porbTrappist1gSig, size=1)[0])
        elif name == "trappist1h":
            ret.append(norm.rvs(loc=porbTrappist1h, scale=porbTrappist1hSig, size=1)[0])
        else:
            raise ValueError("Not a planet! Try trappist1x for x in [b-h]")

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def estXUVFluxes(planet, lum, mstar=0.089):
    """
    Estimate the XUV flux received by a given TRAPPIST-1 planet given its host
    star's mass and current luminosity and age.

    Parameters
    ----------
    planet : str
        name of planet of the form trappist1x for x in [b,h]
    lum : float/array
        XUV luminosity in Lsun at a given age
    mstar : float/array (optional)
        Mass of star in Msun. Defaults to 0.089 Msun

    Returns
    -------
    flux : float/array
        Incidence flux received by planet normalized by mid-solar cycle XUV
        flux received by the Earth
    """

    # Sample porb from Gillon+2017 observational constraints in [d] -> [s]
    porb = Trappist1PlanetPorbSample(planet, size=1) * DAYSEC

    # Convert to semi-major axis [cm] via Kepler's 3rd law
    # P^2 = 4 * pi^2 a^3 / G (m1 + m2), but here we assume m1 + m2 = mstar
    a = np.power((BIGG * porb * porb * mstar * MSUN) / (4.0 * np.pi**2), 1./3.)

    # Compute, return flux = L/(4.0*pi*a^2)

    return lum * LSUN / (4.0 * np.pi * a * a) / EARTHXUV
# end function


### Perform analysis ###


# Read in evolutionary tracks
data = np.load("../../Data/trappist1StarEvol.npz")
nsamples = len(data["Luminosity"])

# Find index the corresponds to given age in time array
ages = [1.0e7, 1.0e8, 1.0e9] # 10, 100, 1000 Myr
ageNames = ["10 Myr", "100 Myr", "1000 Myr"]
ageInds = []
for age in ages:
    ageInds.append(np.argmin(np.fabs(age - data["time"][0])))

# Define containers
t1b = []
t1c = []
t1d = []
t1e = []
t1f = []
t1g = []
t1h = []
age = []

# Loop over samples
for ii in range(nsamples):
    # Loop over ages
    for jj, ageInd in enumerate(ageInds):

        # Get XUV luminosity at current age in Lsun
        currentLum = data["LXUVStellar"][ii][ageInd]

        # Save age for dummy columns for subsequent group-by
        age.append(ageNames[jj])

        # Compute, save XUV for each planet for each sample at each age
        t1b.append(np.log10(estXUVFluxes("trappist1b", currentLum, mstar=0.089)))
        t1c.append(np.log10(estXUVFluxes("trappist1c", currentLum, mstar=0.089)))
        t1d.append(np.log10(estXUVFluxes("trappist1d", currentLum, mstar=0.089)))
        t1e.append(np.log10(estXUVFluxes("trappist1e", currentLum, mstar=0.089)))
        t1f.append(np.log10(estXUVFluxes("trappist1f", currentLum, mstar=0.089)))
        t1g.append(np.log10(estXUVFluxes("trappist1g", currentLum, mstar=0.089)))
        t1h.append(np.log10(estXUVFluxes("trappist1h", currentLum, mstar=0.089)))

# Turn it all into a pandas dataframe
df = pd.DataFrame({"b" : t1b, "c" : t1c, "d" : t1d, "e" : t1e, "f" : t1f,
                  "g" : t1g, "h" : t1h, "Age" : age})

# Make the ridge(joy?) plot, grouping by age
fig, axes = jp.joyplot(df, by="Age", xlabelsize=18, ylabelsize=18, alpha=0.7,
                       figsize=(6,6), linewidth=1, legend=False, overlap=0.5,
                       labels=["0.01 Gyr", "0.1 Gyr", "1 Gyr"], linecolor="k")

# Custom legend
custom = [mpatches.Patch(facecolor="C%d" % ii, edgecolor="C%d" % ii) for ii in range(7)]
legendLabels = ["%s" % let for let in ["b", "c", "d", "e", "f", "g", "h"]]
axes[0].legend(custom, legendLabels, bbox_to_anchor=[0.3,1.1], fontsize=15,
               ncol=2, framealpha=0, title="TRAPPIST-1")

# Format
xlabels = [r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]
axes[-1].set_xticklabels(xlabels)
axes[-1].set_xlabel(r"Incident $F_{XUV}/F_{XUV,\oplus}$", fontsize=20)

# Save!
if (sys.argv[1] == 'pdf'):
    fig.savefig("fluxes.pdf", bbox_inches="tight", dpi=200)
if (sys.argv[1] == 'png'):
    fig.savefig("fluxes.png", bbox_inches="tight", dpi=200)
# Done!
