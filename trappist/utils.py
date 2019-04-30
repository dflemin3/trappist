#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constants and math utility functions
"""

__all__ = ["LSUN", "YEARSEC", "BIGG", "DAYSEC", "MSUN", "AUCM", "loguniform"]

import numpy as np

# Constants
LSUN = 3.846e26 # Solar luminosity in ergs/s
YEARSEC = 3.15576e7 # seconds per year
BIGG = 6.67428e-11 # Universal Gravitational Constant in cgs
DAYSEC = 86400.0 # seconds per day
MSUN = 1.988416e30 # mass of sun in g
AUCM = 1.49598e11 # cm per AU
MTO = 1.4e24 # Mass of all of Earth's water in g
MEarth = 5.972e27 # Mass of Earth in g

def loguniform(low=0, high=1, size=None, base=10.0):
    """
    Sample from a log uniform distribution, base=base, over [low,high)

    Parameters
    ----------
    low : float
        lower bound of distribution
    high : float
        upper bound of distribution
    size : int
        number of samples to return
    base : float (optional)
        log base to use.  Defaults to 10

    Return
    ------
    sample(s) : float, array
        return a (,size) sample from a loguniform distribution
    """
    return np.power(base, np.random.uniform(low, high, size))
# end function
