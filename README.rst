On the XUV Evolution of TRAPPIST-1
==================================

David P. Fleming, Rory Barnes, Rodrigo Luger, and Jacob T. VanderPlas

We model the long-term XUV luminosity of TRAPPIST-1 to constrain the evolving
high-energy radiation environment experienced by its planetary system. Using
Markov Chain Monte Carlo (MCMC), we derive probabilistic constraints for
TRAPPIST-1's stellar and XUV evolution that account for observational uncertainties,
degeneracies between model parameters, and empirical data of low-mass stars.
We constrain TRAPPIST-1's mass to 0.089 +/- 0.001 Msun and
find that its early XUV luminosity likely saturated at
log10(L_{XUV}/L_{bol}) = -3.03^{+0.23}_{-0.12}. From the posterior distribution,
we infer that there is a ~40% chance that TRAPPIST-1 is still in the
saturated phase today, suggesting that TRAPPIST-1 has maintained high activity
and L_{XUV}/L_{bol} ~ 10^-3 for several Gyrs. TRAPPIST-1's planetary
system therefore likely experienced a persistent and extreme XUV flux environment,
potentially driving significant atmospheric erosion and volatile loss. The inner
planets likely received XUV fluxes ~10^3 - 10^4 times that of the modern
Earth during TRAPPIST-1's ~1 Gyr-long pre-main sequence phase. Deriving
these constraints via MCMC is computationally non-trivial, so scaling our methods
to constrain the XUV evolution of a larger number of M dwarfs that harbor
terrestrial exoplanets would incur significant computational expenses. We
demonstrate that approxposterior, an open source Python machine learning
package for approximate Bayesian inference using Gaussian processes, accurately
and efficiently replicates our analysis using 980 times less computational
time and 1330 times fewer simulations than MCMC sampling using emcee. We
find that approxposterior derives constraints with mean errors on the best
fit values and 1 sigma uncertainties of 0.61% and 5.5%, respectively,
relative to emcee.

.. figure:: Analysis/Corner/trappist1Corner.png
   :width: 600px
   :align: center

   Joint and marginal posterior probability distributions for the TRAPPIST-1
   stellar and XUV parameters made using corner
   Foreman-Mackey+2016. The black vertical dashed lines on the
   marginalized distributions indicate the median values and lower and upper
   uncertainties from the 16th and 84th percentiles, respectively. From the
   posterior, we infer that there is a 40% chance that TRAPPIST-1 is still
   in the saturated phase today, potentially driving significant water loss
   and atmospheric escape from its planets.
