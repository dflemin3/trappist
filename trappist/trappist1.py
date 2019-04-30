#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.stats import norm
from . import utils

__all__ = ["kwargsTRAPPIST1", "LnPriorTRAPPIST", "samplePriorTRAPPIST1"]

# Observational constraints

# Stellar properties: Trappist1 in nearly solar metallicity, so the Baraffe+2015 tracks will be good
lumTrappist1 = 0.000522           # Van Grootel et al. (2018) [Lsun]
lumTrappist1Sig = 0.000019        # Van Grootel et al. (2018) [Lsun]

radTrappist1 = 0.121              # Van Grootel et al. (2018) [Rsun]
radTrappist1Sig = 0.003           # Van Grootel et al. (2018) [Rsun]

logLXUVTrappist1 = -6.4           # Wheatley et al. (2017)
logLXUVTrappist1Sig = 0.1         # Wheatley et al. (2017)

betaTrappist1 = -1.18             # Jackson et al. (2012)
betaTrappist1Sig = 0.31           # Jackson et al. (2012)

ageTrappist1 = 7.6                # Burgasser et al. (2017) [Gyr]
ageTrappist1Sig = 2.2             # Burgasser et al. (2017) [Gyr]

fsatTrappist1 = -3.14             # Jackson et al. (2012)
fsatTrappist1Sig = 0.37           # Jackson et al. (2012)

# Planet properties: Masses [Mearth], Radii [Rearth] and ecc from Grimm+2018,
# Porbs from Gillon et al. (2017) and Luger et al. (2017)

# b
porbTrappist1b = 1.51087081
porbTrappist1bSig = 0.6e-6

eccTrappist1b = 0.00622
eccTrappist1bSig = 0.00304

massTrappist1b = 1.017
massTrappist1bSig = 0.154

radiusTrappist1b = 1.121
radiusTrappist1bSig = 0.032

# c
porbTrappist1c = 2.4218233
porbTrappist1cSig = 0.17e-5

eccTrappist1c = 0.00654
eccTrappist1cSig = 0.00188

massTrappist1c = 1.156
massTrappist1cSig = 0.142

radiusTrappist1c = 1.095
radiusTrappist1cSig = 0.031

# d
porbTrappist1d = 4.049610
porbTrappist1dSig = 0.63e-4

eccTrappist1d = 0.00837
eccTrappist1dSig = 0.00093

massTrappist1d = 0.297
massTrappist1dSig = 0.039

radiusTrappist1d = 0.784
radiusTrappist1dSig = 0.023

# e
porbTrappist1e = 6.099615
porbTrappist1eSig = 0.11e-4

eccTrappist1e = 0.00510
eccTrappist1eSig = 0.00058

massTrappist1e = 0.772
massTrappist1eSig = 0.079

radiusTrappist1e = 0.910
radiusTrappist1eSig = 0.027

# f
porbTrappist1f = 9.20669
porbTrappist1fSig = 0.15e-4

eccTrappist1f = 0.01007
eccTrappist1fSig = 0.00068

massTrappist1f = 0.934
massTrappist1fSig = 0.08

radiusTrappist1f = 1.046
radiusTrappist1fSig = 0.03

# g
porbTrappist1g = 12.35294
porbTrappist1gSig = 0.12e-3

eccTrappist1g = 0.00208
eccTrappist1gSig = 0.00058

massTrappist1g = 1.148
massTrappist1gSig = 0.098

radiusTrappist1g = 1.148
radiusTrappist1gSig = 0.033

# h
porbTrappist1h = 18.767
porbTrappist1hSig = 0.004

eccTrappist1h = 0.00567
eccTrappist1hSig = 0.00121

massTrappist1h = 0.331
massTrappist1hSig = 0.056

radiusTrappist1h = 0.773
radiusTrappist1hSig = 0.027

### Prior, likelihood, MCMC functions ###


def LnPriorTRAPPIST1(x, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x

    # Uniform prior for stellar mass [Msun]
    if (dMass < 0.07) or (dMass > 0.11):
        return -np.inf

    # Uniform prior on saturation timescale [100 Myr - 12 Gyr]
    if (dSatXUVTime < 0.1) or (dSatXUVTime > 12.0):
        return -np.inf

    # Large bound for age of system [Gyr] informed by Burgasser et al. (2017)
    if (dStopTime < 0.1) or (dStopTime > 12.0):
        return -np.inf

    # Hard bounds on XUVBeta to bracket realistic values
    if (dXUVBeta < -2.0) or (dXUVBeta > 0.0):
        return -np.inf

    # Hard bound on log10 saturation fraction (log10)
    if (dSatXUVFrac < -5) or (dSatXUVFrac > -1):
        return -np.inf

    # Age prior
    lnprior = norm.logpdf(dStopTime, ageTrappist1, ageTrappist1Sig)

    # Beta prior
    lnprior += norm.logpdf(dXUVBeta, betaTrappist1, betaTrappist1Sig)

    # fsat prior
    lnprior += norm.logpdf(dSatXUVFrac, fsatTrappist1, fsatTrappist1Sig)

    return lnprior
# end function


def samplePriorTRAPPIST1(size=1, **kwargs):
    """
    Sample dMass, dSatXUVFrac, dSatXUVTime, dStopTime, and dXUVBeta from their
    prior distributions.
    """

    ret = []
    for ii in range(size):
        while True:
            guess = [np.random.uniform(low=0.07, high=0.11),
                     norm.rvs(loc=fsatTrappist1, scale=fsatTrappist1Sig, size=1)[0],
                     np.random.uniform(low=0.1, high=12),
                     norm.rvs(loc=ageTrappist1, scale=ageTrappist1Sig, size=1)[0],
                     norm.rvs(loc=betaTrappist1, scale=betaTrappist1Sig, size=1)[0]]
            if not np.isinf(LnPriorTRAPPIST1(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def sampleFlatPriorTRAPPIST1(size=1, **kwargs):
    """
    Sample dMass, dSatXUVFrac, dSatXUVTime, dStopTime, and dXUVBeta from
    uniform distributions over the full prior range.
    """

    ret = []
    for ii in range(size):
        while True:
            guess = [np.random.uniform(low=0.07, high=0.11),
                     np.random.uniform(low=-5.0, high=-1.0),
                     np.random.uniform(low=0.1, high=12.0),
                     np.random.uniform(low=0.1, high=12.0),
                     np.random.uniform(low=-2.0, high=0.0)]
            if not np.isinf(LnPriorTRAPPIST1(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def Trappist1PlanetMassSample(planet, size=1, **kwargs):
    """
    Sample Trappist1 system planet masses from Gaussian distributions based on
    Grimm+2018 measurements.
    """

    # Light preprocessing of planet name
    name = str(planet).lower()

    ret = []
    for ii in range(size):
        if name == "trappist1b":
            ret.append(norm.rvs(loc=massTrappist1b, scale=massTrappist1bSig, size=1)[0])
        elif name == "trappist1c":
            ret.append(norm.rvs(loc=massTrappist1c, scale=massTrappist1cSig, size=1)[0])
        elif name == "trappist1d":
            ret.append(norm.rvs(loc=massTrappist1d, scale=massTrappist1dSig, size=1)[0])
        elif name == "trappist1e":
            ret.append(norm.rvs(loc=massTrappist1e, scale=massTrappist1eSig, size=1)[0])
        elif name == "trappist1f":
            ret.append(norm.rvs(loc=massTrappist1f, scale=massTrappist1fSig, size=1)[0])
        elif name == "trappist1g":
            ret.append(norm.rvs(loc=massTrappist1g, scale=massTrappist1gSig, size=1)[0])
        elif name == "trappist1h":
            ret.append(norm.rvs(loc=massTrappist1h, scale=massTrappist1hSig, size=1)[0])
        else:
            raise ValueError("Not a planet! Try trappist1x for x in [b-h]")

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def Trappist1PlanetRadiusSample(planet, size=1, **kwargs):
    """
    Sample Trappist1 system planet radii from Gaussian distributions based on
    Delrez+2018 measurements.
    """

    # Light preprocessing of planet name
    name = str(planet).lower()

    ret = []
    for ii in range(size):
        if name == "trappist1b":
            ret.append(norm.rvs(loc=radiusTrappist1b, scale=radiusTrappist1bSig, size=1)[0])
        elif name == "trappist1c":
            ret.append(norm.rvs(loc=radiusTrappist1c, scale=radiusTrappist1cSig, size=1)[0])
        elif name == "trappist1d":
            ret.append(norm.rvs(loc=radiusTrappist1d, scale=radiusTrappist1dSig, size=1)[0])
        elif name == "trappist1e":
            ret.append(norm.rvs(loc=radiusTrappist1e, scale=radiusTrappist1eSig, size=1)[0])
        elif name == "trappist1f":
            ret.append(norm.rvs(loc=radiusTrappist1f, scale=radiusTrappist1fSig, size=1)[0])
        elif name == "trappist1g":
            ret.append(norm.rvs(loc=radiusTrappist1g, scale=radiusTrappist1gSig, size=1)[0])
        elif name == "trappist1h":
            ret.append(norm.rvs(loc=radiusTrappist1h, scale=radiusTrappist1hSig, size=1)[0])
        else:
            raise ValueError("Not a planet! Try trappist1x for x in [b-h]")

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


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


def Trappist1PlanetEccSample(planet, size=1, **kwargs):
    """
    Sample Trappist1 system planet eccentricities from Gaussian distributions
    based on Grimm+2018 measurements.
    """

    # Light preprocessing of planet name
    name = str(planet).lower()

    ret = []
    for ii in range(size):
        while True:
            if name == "trappist1b":
                tmp = norm.rvs(loc=eccTrappist1b, scale=eccTrappist1bSig, size=1)[0]
            elif name == "trappist1c":
                tmp = norm.rvs(loc=eccTrappist1c, scale=eccTrappist1cSig, size=1)[0]
            elif name == "trappist1d":
                tmp = norm.rvs(loc=eccTrappist1d, scale=eccTrappist1dSig, size=1)[0]
            elif name == "trappist1e":
                tmp = norm.rvs(loc=eccTrappist1e, scale=eccTrappist1eSig, size=1)[0]
            elif name == "trappist1f":
                tmp = norm.rvs(loc=eccTrappist1f, scale=eccTrappist1fSig, size=1)[0]
            elif name == "trappist1g":
                tmp = norm.rvs(loc=eccTrappist1g, scale=eccTrappist1gSig, size=1)[0]
            elif name == "trappist1h":
                tmp = norm.rvs(loc=eccTrappist1h, scale=eccTrappist1hSig, size=1)[0]
            else:
                raise ValueError("Not a planet! Try trappist1x for x in [b-h]")

            # Make sure it's a physically possible eccentricity
            if tmp >= 0 and tmp < 1:
                break

        # Save valid guess
        ret.append(tmp)

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


# Dict to hold all constraints
kwargsTRAPPIST1 = {"PATH" : ".",
                   "LnPrior" : LnPriorTRAPPIST1,
                   "PriorSample" : samplePriorTRAPPIST1,
                   "LUM" : lumTrappist1,
                   "LUMSIG" : lumTrappist1Sig,
                   "LOGLUMXUV" : logLXUVTrappist1,
                   "LOGLUMXUVSIG" : logLXUVTrappist1Sig,
                   "PLANETLIST" : ["trappist1b", "trappist1c", "trappist1d",
                                   "trappist1e", "trappist1f", "trappist1g",
                                   "trappist1h"],
                   "PlanetMassSample" : Trappist1PlanetMassSample,
                   "PlanetEccSample" : Trappist1PlanetEccSample,
                   "PlanetRadiusSample" : Trappist1PlanetRadiusSample,
                   "PlanetPorbSample" : Trappist1PlanetPorbSample}
