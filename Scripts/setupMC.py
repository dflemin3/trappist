"""

@author: David P. Fleming, University of Washington, Seattle, July 2019
@email: dflemin3 (at) uw (dot) edu

This script produces initial conditions for a synthetic population of ultracool
dwarfs to examine LXUV/Lbol as a function of time. All initial conditions are
sampled from the prior distributions used to constrain the evolutionary history
of TRAPPIST-1.

Assumptions:
- Template input (body.in) files live directory where this script exists.

"""

import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
import stat
import sys
from trappist import trappist1 as t1

# Number of sets of initial conditions to generate
num = 10

# Current path
PATH = os.path.dirname(os.path.realpath(__file__))

# Random seed
seed = int(os.getpid()*datetime.now().microsecond/100)

# Set RNG seed
np.random.seed(seed)

# File names
sys_name = "vpl.in"
star_name = "star.in"

# Containers
runfile_names = []
dirs = []
star_dMass = []
star_dSatXUVFrac = []
star_dSatXUVTime = []
star_dAge = []
star_dXUVBeta = []

### Make the simulation initial conditions! ###

for ii in range(num):

    # Create a directory for the simulation to live in
    directory = "simulation_" + str(ii)
    dirs.append(directory)
    if not os.path.exists(os.path.join(PATH,directory)):
        os.makedirs(os.path.join(PATH,directory))

    ### Populate the star input file ###

    # Read template input file
    with open(os.path.join(PATH, 'star.in'), 'r') as f:
        star_in = f.read()

    # Sample stellar parameters from TRAPPIST-1 prior distributions
    dMass, dSatXUVFrac, dSatXUVTime, _, dXUVBeta = t1.samplePriorTRAPPIST1()

    # Instead of using T1's age prior, we'll assume a flat age distribution
    # over 100 Myr to 12 Gyr
    dAge = np.random.uniform(low=0.1, high=12.0)

    # Save the values
    star_dMass.append(dMass)
    star_dSatXUVFrac.append(dSatXUVFrac)
    star_dSatXUVTime.append(dSatXUVTime)
    star_dAge.append(dAge)
    star_dXUVBeta.append(dXUVBeta)

    # Convert select values to proper VPLanet units
    dSatXUVTime = -dSatXUVTime
    dSatXUVFrac = 10**dSatXUVFrac
    dXUVBeta = -dXUVBeta
    dAge = dAge * 1.0e9

    # Write initial conditions to star input file
    star_in = re.sub('%s(.*?)#' % 'dMass', '%s %.5e #' % ('dMass', dMass), star_in)
    star_in = re.sub('%s(.*?)#' % 'dSatXUVFrac', '%s %.5e #' % ('dSatXUVFrac', dSatXUVFrac), star_in)
    star_in = re.sub('%s(.*?)#' % 'dSatXUVTime', '%s %.5e #' % ('dSatXUVTime', dSatXUVTime), star_in)
    star_in = re.sub('%s(.*?)#' % 'dXUVBeta', '%s %.5e #' % ('dXUVBeta', dXUVBeta), star_in)

    with open(os.path.join(PATH, directory, star_name), 'w') as f:
        print(star_in, file = f)

    ### Write vpl input file ###

    # Read template input file
    with open(os.path.join(PATH, sys_name), 'r') as f:
        sys_in = f.read()

    # Write vpl file
    sys_in = re.sub('%s(.*?)#' % 'dStopTime', '%s %.5e #' % ('dStopTime', dAge), sys_in)

    with open(os.path.join(PATH, directory, sys_name), 'w') as f:
        print(sys_in, file = f)

    # Generate *.sh file needed for cluster to run sims
    command = os.path.join(PATH, directory + ".sh")
    with open(command,"w") as g:
        g.write("#!/bin/bash\n")
        g.write("cd " + os.path.join(PATH, directory) + "\n") # Change dir to where sim is
        g.write("vplanet vpl.in\n") # Run sim command!

        # Now give that .sh file execute permissions
        st = os.stat(command)
        os.chmod(command, st.st_mode | stat.S_IEXEC)

        # Save file name for later...
        runfile_names.append(command)
# end for loop

# Write all runfile names to file needed for cluster
with open(os.path.join(PATH, "vplArgs.txt"), 'w') as f:
    for line in runfile_names:
        f.write(line + "\n")

# Save the initial condition - simulation dir pairs

# Put data into a pandas dataframe
df = pd.DataFrame({"dMass" : star_dMass, "dSatXUVFrac" : star_dSatXUVFrac,
                   "dSatXUVTime" : star_dSatXUVTime, "dAge" : star_dAge,
                   "dXUVBeta" : star_dXUVBeta}, index=dirs)

# Dump it into a CSV
df.to_csv(os.path.join(PATH,"mcInitialConditions.csv"))

# Done!
