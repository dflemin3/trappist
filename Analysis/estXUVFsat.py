#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate XUV f_sat for TRAPPIST-1 in the saturated phase from Lx/Lbol relation
from Wright et al. (2018). We estimate LEUV as a function of LX using Eqn. 2
from Chadney et al. (2015) following Wheatley et al. (2017) who showed that this
application was valid.

@author David P. Fleming, 2019
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

# Define constants
RSUN = 6.957e10 # Radius of Sun in cm
LSUN = 3.848e33 # Luminosity of Sun in erg/s


def calcLXUV(Lbol, LX, Rstar):
    """
    Convert bolometric luminosity into an LX luminosity given a saturation
    fraction, assuming that the star is in the saturated regime.

    Parameters
    ----------
    Lbol : float
        bolometric luminosity [Lsun]
    LX : float
        X-ray luminosity [Lsun]
    Rstar : float
        Stellar radius [Rsun]

    Returns
    -------
    LXUV : float
        EUV + X-ray luminosity [Lsun]
    """

    # Convert luminosities, radius to cgs units
    L = Lbol * LSUN
    R = Rstar * RSUN

    # Evaluate
    LXUV = (LX * LSUN) + 4.0*np.pi*R*R*425*np.power((LX * LSUN)/(4.0*np.pi*R*R), 0.58)

    # Return answer in Lsun
    return LXUV/LSUN
# end function


def calcTauCZ(mass):
    """
    Compute convective turnover time using Wright+2011 Eqn. 6

    Parameters
    ----------
    mass : float
        Stellar mass [Msun]

    Returns
    -------
    taucz : float
        convective turnover timescale [d]
    """

    logtau = 2.33 - 1.5*mass + 0.3*mass*mass

    return np.power(10.0, logtau)
# end function


# Run!
if __name__ == "__main__":

    # Load in Wright+2011 data
    columns = ["Vmag",  "V-K", "log10LX", "Prot", "Mass", "Radius", "LBOL",
               "Teff", "dcz", "Lx/bol"]
    data = pd.read_table("../Data/Wright2011.tsv", delimiter=";", header=None,
                         comment="#", names=columns, index_col=False)

    # Unlog LX [Lsun]
    data["LX"] = pd.Series(np.power(10.0, data["log10LX"].values)/LSUN,
                           index=data.index)

    # Compute bolometric luminosity [Lsun] because Wright+2011 data isn't
    # precise enough to have enough digits for small M dwarf luminosities
    data["LBOL"] = pd.Series(data["LX"].values/np.power(10.0,data["Lx/bol"].values),
                             index=data.index)

    # Compute LXUV
    data["LXUV"] = pd.Series(calcLXUV(data["LBOL"].values, data["LX"].values,
                             data["Radius"].values), index=data.index)

    # Compute LEUV
    data["LEUV"] = pd.Series(data["LXUV"].values - data["LX"].values,
                             index=data.index)

    # Compute LXUV/LBOL
    data["LXUVLBOL"] = pd.Series(data["LXUV"].values/data["LBOL"].values,
                                index=data.index)

    # Compute LX/LBOL
    data["LXLBOL"] = pd.Series(data["LX"].values/data["LBOL"].values,
                                index=data.index)

    # Compute convective turnover time
    data["Tau"] = pd.Series(calcTauCZ(data["Mass"].values), index=data.index)

    # Compute Rossby Number
    data["Ro"] = pd.Series(data["Prot"].values/data["Tau"].values,
                           index=data.index)

    # Only select fully-convective stars
    data = data[data.dcz < 1.0e-3]

    # Only select saturated stars with R0 <= 0.14 (Wright+2018)
    data = data[data.Ro <= 0.14]

    # Print mean, scatter
    print("LX/LBOL Mean, Scatter:", np.mean(np.log10(data["LXLBOL"].values)), np.std(np.log10(data["LXLBOL"].values)))
    print("LXUV/LBOL Mean, Scatter:", np.mean(np.log10(data["LXUVLBOL"].values)), np.std(np.log10(data["LXUVLBOL"].values)))

    # Make plots

    # Luminosity fraction vs Rossby Number
    fig, ax = plt.subplots()

    ax.scatter(data["Ro"], data["LXUVLBOL"], c="C0", label=r"$L_{XUV}/L_{Bol}$")
    ax.scatter(data["Ro"], data["LXLBOL"], c="k", label=r"$L_X/L_{Bol}$")

    ax.set_xlabel("Rossby Number", fontsize=20)
    ax.set_ylabel(r"Luminosity Ratio", fontsize=20)
    ax.set_xlim(1.0e-3, 4)
    ax.set_ylim(5.0e-7, 1.0e-2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=18)

    fig.tight_layout()
    fig.savefig("lumFracVsRossby.png", bbox_inches="tight", dpi=200)

    # Histogram of fsats
    fig, ax = plt.subplots()

    ax.hist(np.log10(data["LXUVLBOL"]), bins="auto", range=[-4.5, -2],
            color="C0", label=r"$L_{XUV}/L_{Bol}$", density=True,
            histtype="step", lw=3)

    ax.hist(np.log10(data["LXLBOL"]), bins="auto", range=[-4.5, -2],
            color="k", label=r"$L_{X}/L_{Bol}$", density=True,
            histtype="step", lw=3)

    ax.hist(np.log10(data["LXUVLBOL"]), bins="auto", range=[-4.5, -2],
            color="C0", label=r"$L_{XUV}/L_{Bol}$", density=True,
            alpha=0.5)

    ax.hist(np.log10(data["LXLBOL"]), bins="auto", range=[-4.5, -2],
            color="k", label=r"$L_{X}/L_{Bol}$", density=True,
            alpha=0.5)

    # Fit normal distributions
    muXUV, stdXUV = norm.fit(np.log10(data["LXUVLBOL"].values))
    muX, stdX = norm.fit(np.log10(data["LXLBOL"].values))

    # Output fit parameters
    print("X Gaussian Fit:", muX, stdX)
    print("XUV Gaussian Fit:", muXUV, stdXUV)

    # Overplot fits on histograms
    x = np.linspace(-4.5, -2, 250)
    pdfX = norm.pdf(x, muX, stdX)
    pdfXUV = norm.pdf(x, muXUV, stdXUV)

    ax.plot(x, pdfX, color="k", ls="--", lw=2.5, label="X Fit")
    ax.plot(x, pdfXUV, color="C0", ls="--", lw=2.5, label="XUV Fit")

    ax.legend(loc="best")
    ax.set_xlabel("Luminosity Ratio", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)

    plt.show()
