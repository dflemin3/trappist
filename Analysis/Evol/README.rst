TRAPPIST-1 Stellar Evolution Convergence
========================================

Overview
--------

Plot the evolution of TRAPPIST-1's luminosity, XUV luminosity, and radius using
samples from the posterior distribution.

===================   ============
**Date**              04/30/19
**Author**            David P. Fleming
**VPLanet Modules**   STELLAR
===================   ============

This example examines plots evolution of TRAPPIST-1's luminosity, XUV
luminosity, and radius using samples from the posterior distribution.

To make the plot
----------------

.. code-block:: bash

    python plotEvolTrappist1.py <pdf | png>


Expected output
---------------

.. figure:: trappist1Evol.png
   :width: 600px
   :align: center

   Evolution of the luminosity (left), XUV luminosity (middle), and radius
   (center) of TRAPPIST-1 using 100 samples (black) drawn from the posterior
   distribution. The insets display the marginalized values (blue) and the
   black dashed lines indicate the observed value and +/- 1 sigma uncertainties.
