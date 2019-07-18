Ultracool Dwarf XUV Evolution: Monte Carlo Simulations
======================================================

Overview
--------

Plot LXUV/Lbol as a function of age for a synthetic population of ultracool
dwarfs with initial conditions sampled from our adopted prior distributions.

===================   ============
**Date**              07/18/19
**Author**            David P. Fleming
**VPLanet Modules**   STELLAR
===================   ============

To make the plot
----------------

.. code-block:: bash

    python monte.py <pdf | png>


Expected output
---------------

.. figure:: monte.png
   :width: 600px
   :align: center

   LXUV/Lbol as a function of age for a population of 2,500 ultracool dwarfs
   evolved according to our model with initial conditions sampled from our
   prior distributions. We overplot the data for TRAPPIST-1, with uncertainties,
   in maroon.
