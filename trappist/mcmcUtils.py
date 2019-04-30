#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
docs
"""

import vplot as vpl
import numpy as np
from scipy.stats import norm
import emcee
import subprocess
import re
import os, sys
import random
import argparse
from . import utils
from . import proxima


__all__ = ["FunctionWrapper", "LnLike", "GetEvol", "RunMCMC",
           "waterPriorRaymond2007Sample", "waterPriorUniformSample",
           "waterPriorDeltaSample", "waterPriorLogUniformSample",
           "waterPriorDeltaBarnes2016Sample"]

### Utility functions ###

class FunctionWrapper(object):
    """"
    A simple function wrapper class. Stores :py:obj:`args` and :py:obj:`kwargs` and
    allows an arbitrary function to be called with a single parameter :py:obj:`x`
    """

    def __init__(self, f, *args, **kwargs):
        """
        """

        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        """
        """

        return self.f(x, *self.args, **self.kwargs)
# end class


def extractMCMCResults(filename, verbose=True, applyBurnin=True, thinChains=True,
                       blobsExist=True):
    """
    Extract and process MCMC results

    Parameters
    ----------
    filename : str
        path to emcee MCMC h5 file
    verbose : bool (optional)
        Output convergence diagnostics? Defaults to True.
    applyBurnin : bool (optional)
        Apply a burnin to reduce size of chains? Defaults to True.
    thinChains : bool (optional)
        Thin chains to reduce their size? Defaults to True.
    blobsExist : bool (optional)
        Whether or not blobs exist.  If True, return them! Defaults to True.

    Returns
    -------
    chain : numpy array
        MCMC chain
    blobs : numpy array
        MCMC ancillary and derived quantities. Only returned if blobsExist is True
    """

    # Open file
    reader = emcee.backends.HDFBackend(filename)

    if verbose:
        # Compute acceptance fraction for each walker
        print("Acceptance fraction for each walker:")
        print(reader.accepted / reader.iteration)
        print("Mean acceptance fraction:", np.mean(reader.accepted / reader.iteration))

    # Compute convergence diagnostics

    # Compute burnin?
    tau = reader.get_autocorr_time(tol=0)
    if applyBurnin:
        burnin = int(2*np.max(tau))
    else:
        burnin = 0
    if thinChains:
        thin = int(0.5*np.min(tau))
    else:
        thin = 1

    # Output convergence diagnostics?
    if verbose:
        print("Burnin, thin:", burnin, thin)

        # Is the length of the chain at least 50 tau?
        print("Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.")
        print("Number of iterations / tau:", reader.iteration / tau)
        print("Mean Number of iterations / tau:", np.mean(reader.iteration / tau))

    # Load data
    chain = reader.get_chain(discard=burnin, flat=True, thin=thin)

    # Properly shape blobs
    tmp = reader.get_blobs(discard=burnin, flat=True, thin=thin)
    if blobsExist:
        blobs = []
        for bl in tmp:
            blobs.append([bl[ii] for ii in range(len(bl))])
        blobs = np.array(blobs)

        return chain, blobs
    else:
        return chain
# end function


### Globally accessible priors ###

def waterPriorUniformSample(size=1, low=0.0, high=100.0, **kwargs):
    """
    Sample initial water inventory in TO from uniform distribution from [low,high)

    Parameters
    ----------
    size : int (optional)
        number of samples. Defaults to 1
    low : float (optional)
        Low range limit.  Defaults to 0
    high : float (optional)
        High range limit. Defaults to 100
    **kwargs

    Returns
    -------
    samples : float, array-like
        initial water inventory in TO of length size
    """

    ret = np.random.uniform(low=low, high=high, size=size)

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def waterPriorLogUniformSample(size=1, low=-5, high=-1.30, **kwargs):
    """
    Sample initial water inventory in TO from log-uniform distribution of water
    mass fractions based on results from Mulders+2015 and Ciesla+2015. Sample
    over [low,high) where the default values are [-1.0e-5,5.0e-2), or [-5,-1.3]

    Parameters
    ----------
    size : int (optional)
        number of samples. Defaults to 1
    low : float (optional)
        Low range limit.  Defaults to -5, or 1.0e-5
    high : float (optional)
        High range limit. Defaults to 5.0e-2, or -1.3
    **kwargs

    Returns
    -------
    samples : float, array-like
        initial water inventory in TO of length size
    """

    ret = 10**np.random.uniform(low=low, high=high, size=size) * utils.MEarth / utils.MTO

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def waterPriorRaymond2007Sample(size=1, **kwargs):
    """
    Sample initial water inventory in TO from Gaussian fit to Raymond+2007
    water delivery simulations.

    Parameters
    ----------
    size : int (optional)
        number of samples. Defaults to 1
    **kwargs

    Returns
    -------
    samples : float, array-like
        initial water inventory in TO of length size
    """

    # Fit mean, std to Raymond+2007 log10 water mass fractions. See waterPrior.py
    raymu, raysig = -2.17, 0.32

    # Return in terrestrial Earth ocean masses (TO)
    ret = 10**norm.rvs(raymu, raysig, size=size) * utils.MEarth / utils.MTO

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def waterPriorDeltaBarnes2016Sample(size=1, **kwargs):
    """

    Parameters
    ----------
    size : int (optional)
        number of samples. Defaults to 1
    loc : float (optional)
        Number of oceans to return.  Defaults to 5 following Barnes+2016's
        fiducial case.
    **kwargs

    Returns
    -------
    samples : float, array-like
        initial water inventory in TO of length size
    """

    ret = []
    for _ in range(size):
        ret.append(5.0)

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


def waterPriorDeltaSample(size=1, loc=20.0, **kwargs):
    """

    Parameters
    ----------
    size : int (optional)
        number of samples. Defaults to 1
    loc : float (optional)
        Number of oceans to return.  Defaults to 20.
    **kwargs

    Returns
    -------
    samples : float, array-like
        initial water inventory in TO of length size
    """

    ret = []
    for _ in range(size):
        ret.append(loc)

    if size > 1:
        return ret
    else:
        return ret[0]
# end function


### Loglikelihood and MCMC functions ###

def LnLike(x, **kwargs):
    """
    loglikelihood function: runs VPLanet simulation
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
    dSatXUVFrac = 10 ** dSatXUVFrac # Unlog
    dStopTime *= 1.e9 # Convert from Gyr -> yr
    dOutputTime = dStopTime # Output only at the end of the simulation

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)
    if np.isinf(lnprior):
        blobs = np.array([np.nan, np.nan] + 7*[np.nan for _ in kwargs["PLANETLIST"]])
        return -np.inf, blobs

    # Get strings containing VPLanet input files (they must be provided!)
    try:
        planet_ins = kwargs.get("PLANETIN")
        star_in = kwargs.get("STARIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: Must supply PLANETIN, STARIN, VPLIN.")
        raise

    # Get PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    planetNames = ['pl%012x' % random.randrange(16**12) for _ in kwargs["PLANETLIST"]]
    starName = 'st%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFiles = [pname + '.in' for pname in planetNames]
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFiles = ['%s.%s.forward' % (sysName, name) for name in kwargs["PLANETLIST"]]
    starFwFile = '%s.star.forward' % sysName

    # Get masses, initial eccentricities, Porbs in order from inner -> outer
    planetMasses = [kwargs["PlanetMassSample"](name) for name in kwargs["PLANETLIST"]]
    planetRadii = [kwargs["PlanetRadiusSample"](name) for name in kwargs["PLANETLIST"]]
    planetEccs = [kwargs["PlanetEccSample"](name) for name in kwargs["PLANETLIST"]]
    planetPorbs = [kwargs["PlanetPorbSample"](name) for name in kwargs["PLANETLIST"]]

    # Get water prior for each planet
    initWater = [kwargs["WaterPrior"]() for _ in kwargs["PLANETLIST"]]

    # Subtract water mass from dMass so total mass is conserved
    for ii in range(len(planetMasses)):
        planetMasses[ii] = planetMasses[ii] - (initWater[ii] * utils.MTO / utils.MEarth)

    # Populate the planet input files for each planet.  Note that Porbs negative
    # to make units days in VPLanet, and same for mass/rad but for Earth units
    for ii, planet_in in enumerate(planet_ins):
        planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", -planetMasses[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dRadius", "%s %.6e #" % ("dRadius", -planetRadii[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", planetEccs[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -planetPorbs[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dSurfWaterMass", "%s %.6e #" % ("dSurfWaterMass", -initWater[ii]), planet_in)
        with open(os.path.join(PATH, "output", planetFiles[ii]), 'w') as f:
            print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), star_in)
    star_in = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file

    # Populate list of planets
    saBodyFiles = str(starFile) + " "
    for pFile in planetFiles:
        saBodyFiles += str(pFile) + " "
    saBodyFiles = saBodyFiles.strip()

    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s #' % saBodyFiles, vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        for pFile in planetFiles:
            os.remove(os.path.join(PATH, "output", pFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        for pFile in planetFwFiles:
            os.remove(os.path.join(PATH, "output", pFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        blobs = np.array([np.nan, np.nan] + 7*[np.nan for _ in kwargs["PLANETLIST"]])
        return -np.inf, blobs

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age / utils.YEARSEC >= dStopTime:
        blobs = np.array([np.nan, np.nan] + 7*[np.nan for _ in kwargs["PLANETLIST"]])
        return -np.inf, blobs

    # Get planet output parameters. Porb and masses are determined by priors
    dEnvMasses = []
    dWaterMasses = []
    dInitWaterMasses = []
    dOxygenMasses = []
    dPorbs = []
    dPlanetMasses = []
    dRGTimes = []
    for ii, pName in enumerate(kwargs["PLANETLIST"]):
        name = str(pName).lower()
        dPlanetMasses.append(planetMasses[ii]) # Prior
        dPorbs.append(planetPorbs[ii]) # Prior
        dInitWaterMasses.append(initWater[ii]) # Prior
        dEnvMasses.append(float(output.log.final.__dict__[name].EnvelopeMass))
        dWaterMasses.append(float(output.log.final.__dict__[name].SurfWaterMass))
        dOxygenMasses.append(float(output.log.final.__dict__[name].OxygenMass) + float(output.log.final.__dict__[name].OxygenMantleMass))
        dRGTimes.append(float(output.log.final.__dict__[name].RGDuration))

    # Get stellar properties
    dLum = float(output.log.final.star.Luminosity)
    dLogLumXUV = np.log10(float(output.log.final.star.LXUVStellar)) # Logged!

    # Extract constraints
    # Must have luminosity, err for star
    lum = kwargs.get("LUM")
    lumSig = kwargs.get("LUMSIG")
    try:
        logLumXUV = kwargs.get("LOGLUMXUV")
        logLumXUVSig = kwargs.get("LOGLUMXUVSIG")
    except KeyError:
        logLumXUV = None
        logLumXUVSig = None

    # Compute the likelihood using provided constraints, assuming we have
    # luminosity constraints for host star
    lnlike = ((dLum - lum) / lumSig) ** 2
    if logLumXUV is not None:
        lnlike += ((dLogLumXUV - logLumXUV) / logLumXUVSig) ** 2
    lnlike = -0.5 * lnlike + lnprior

    # Return likelihood and blobs
    blobs = np.array([dLum, dLogLumXUV] + dPorbs + dPlanetMasses + dRGTimes + dEnvMasses + dWaterMasses + dInitWaterMasses + dOxygenMasses)
    return lnlike, blobs

# end function


def GetEvol(x, **kwargs):
    """
    Run a VPLanet simulation for this initial condition vector, x
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
    dSatXUVFrac = 10 ** dSatXUVFrac # Unlog
    dStopTime *= 1.e9 # Convert from Gyr -> yr

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)
    if np.isinf(lnprior):
        return None

    # Get strings containing VPLanet input files (they must be provided!)
    try:
        planet_ins = kwargs.get("PLANETIN")
        star_in = kwargs.get("STARIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: Must supply PLANETIN, STARIN, VPLIN.")
        raise

    # Get PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    planetNames = ['pl%012x' % random.randrange(16**12) for _ in kwargs["PLANETLIST"]]
    starName = 'st%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFiles = [pname + '.in' for pname in planetNames]
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFiles = ['%s.%s.forward' % (sysName, name) for name in kwargs["PLANETLIST"]]
    starFwFile = '%s.star.forward' % sysName

    # Get masses, initial eccentricities, Porbs in order from inner -> outer
    planetMasses = [kwargs["PlanetMassSample"](name) for name in kwargs["PLANETLIST"]]
    planetRadii = [kwargs["PlanetRadiusSample"](name) for name in kwargs["PLANETLIST"]]
    planetEccs = [kwargs["PlanetEccSample"](name) for name in kwargs["PLANETLIST"]]
    planetPorbs = [kwargs["PlanetPorbSample"](name) for name in kwargs["PLANETLIST"]]

    # Get water prior for each planet
    initWater = [kwargs["WaterPrior"]() for _ in kwargs["PLANETLIST"]]

    # Subtract water mass from dMass so total mass is conserved
    for ii in range(len(planetMasses)):
        planetMasses[ii] = planetMasses[ii] - (initWater[ii] * utils.MTO / utils.MEarth)

    # Populate the planet input files for each planet.  Note that Porbs negative
    # to make units days in VPLanet, and same for mass/rad but for Earth units
    for ii, planet_in in enumerate(planet_ins):
        planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", -planetMasses[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dRadius", "%s %.6e #" % ("dRadius", -planetRadii[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", planetEccs[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -planetPorbs[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dSurfWaterMass", "%s %.6e #" % ("dSurfWaterMass", -initWater[ii]), planet_in)

        with open(os.path.join(PATH, "output", planetFiles[ii]), 'w') as f:
            print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), star_in)
    star_in = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file

    # Populate list of planets
    saBodyFiles = str(starFile) + " "
    for pFile in planetFiles:
        saBodyFiles += str(pFile) + " "
    saBodyFiles = saBodyFiles.strip()

    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s #' % saBodyFiles, vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        for pFile in planetFiles:
            os.remove(os.path.join(PATH, "output", pFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        for pFile in planetFwFiles:
            os.remove(os.path.join(PATH, "output", pFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        return None

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age / utils.YEARSEC >= dStopTime:
        return None

    return output
# end function


def RunMCMC(x0=None, ndim=5, nwalk=100, nsteps=5000, pool=None, backend=None,
            restart=False, planetList=["planet.in"], npzCache=None, **kwargs):
    """
    """

    # Ensure LnPrior, prior sample are in kwargs
    try:
        kwargs["PriorSample"]
    except KeyError as err:
        print("ERROR: Must supply PriorSample function!")
        raise
    try:
        kwargs["LnPrior"]
    except KeyError as err:
        print("ERROR: Must supply LnPrior function!")
        raise

    # Extract path
    PATH = kwargs["PATH"]

    print("Running MCMC...")

    # Get the input files, save them as strings
    planet_ins = []
    for planet in planetList:
        with open(os.path.join(PATH, planet), 'r') as f:
            planet_ins.append(f.read())
        kwargs["PLANETIN"] = planet_ins
    with open(os.path.join(PATH, "star.in"), 'r') as f:
        star_in = f.read()
        kwargs["STARIN"] = star_in
    with open(os.path.join(PATH, "vpl.in"), 'r') as f:
        vpl_in = f.read()
        kwargs["VPLIN"] = vpl_in

    # Set up backend to save results
    if backend is not None:
        # Set up the backend
        handler = emcee.backends.HDFBackend(backend)

        # If restarting from a previous interation, initialize backend
        if not restart:
            handler.reset(nwalk, ndim)

    else:
        handler = None

    # Populate initial conditions for walkers using random samples over prior
    if not restart:
        # If MCMC isn't initialized, just sample from the prior
        if x0 is None:
            x0 = np.array([kwargs["PriorSample"](**kwargs) for w in range(nwalk)])

    ### Run MCMC ###

    # Initialize the sampler object
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLike, kwargs=kwargs,
                                    pool=pool, backend=handler)

    # Actually run the MCMC
    if restart:
        sampler.run_mcmc(None, nsteps)
    else:
        for ii, result in enumerate(sampler.sample(x0, iterations=nsteps)):
            print("MCMC: %d/%d..." % (ii + 1, nsteps))

    # Cache results into a npz?
    if npzCache is not None:
        # Estimate burnin, thin timescales
        tau = sampler.get_autocorr_time()
        burnin = int(2*np.max(tau))
        thin = int(0.5*np.min(tau))

        # Access samples, blobs
        chain = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        blobs = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

        # Now save it all!
        np.savez(npzCache, tau=tau, burnin=burnin, thin=thin, chain=chain,
                 blobs=blobs)

    print("Done!")
# end function
