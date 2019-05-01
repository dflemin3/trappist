#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRAPPIST-1 constraints and prior distributions
"""

import numpy as np
from scipy.stats import norm
from . import utils

__all__ = ["kwargsTRAPPIST1", "LnPriorTRAPPIST1", "samplePriorTRAPPIST1",
           "LnFlatPriorTRAPPIST1"]

# Observational constraints

# Stellar properties: Trappist1 in nearly solar metallicity, so the Baraffe+2015 tracks will be good
lumTrappist1 = 0.000522               # Van Grootel et al. (2018) [Lsun]
lumTrappist1Sig = 0.000019            # Van Grootel et al. (2018) [Lsun]

radTrappist1 = 0.121                  # Van Grootel et al. (2018) [Rsun]
radTrappist1Sig = 0.003               # Van Grootel et al. (2018) [Rsun]

logLXUVTrappist1 = -6.4               # Wheatley et al. (2017)
logLXUVTrappist1Sig = 0.1             # Wheatley et al. (2017)

betaTrappist1 = -1.18                 # Jackson et al. (2012)
betaTrappist1Sig = 0.31               # Jackson et al. (2012)

ageTrappist1 = 7.6                    # Burgasser et al. (2017) [Gyr]
ageTrappist1Sig = 2.2                 # Burgasser et al. (2017) [Gyr]

fsatTrappist1 = -3.05                 # Wright et al. (2018)
fsatTrappist1Sig = 0.06               # Wright et al. (2018)

fsatTrappist1Jackson2012 = -3.14      # Jackson et al. (2012)
fsatTrappist1Jackson2012Sig = 0.37    # Jackson et al. (2012)


### Prior, likelihood, MCMC functions ###


def LnFlatPriorTRAPPIST1(x, **kwargs):
    """
    log flat prior
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

    return 0
# end function


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


# Dict to hold all constraints
kwargsTRAPPIST1 = {"PATH" : ".",
                   "LnPrior" : LnPriorTRAPPIST1,
                   "PriorSample" : samplePriorTRAPPIST1,
                   "LUM" : lumTrappist1,
                   "LUMSIG" : lumTrappist1Sig,
                   "LOGLUMXUV" : logLXUVTrappist1,
                   "LOGLUMXUVSIG" : logLXUVTrappist1Sig}
