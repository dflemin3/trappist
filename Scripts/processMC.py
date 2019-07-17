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
import os
import vplot

# Containers
lum = []
lumXUV = []
xuvBol = []
radius = []
temp = []

for ii in range(500):

    # Read output data
    data = np.genfromtxt(os.path.join("simulation_%d" % ii, "Trappist.star.forward"),
                         delimiter=" ")

    # Cache final values
    # saOutputOrder 	Time -Luminosity -LXUVStellar -Radius Temperature -RotPer # Outputs
    lum.append(data[-1,1])
    lumXUV.append(data[-1,2])
    xuvBol.append(data[-1,1]/data[-1,2])
    radius.append(data[-1,3])
    temp.append(data[-1,4])

# Load in initial conditions
dfInit = pd.read_csv("mcInitialConditions.csv", index_col=0, header=0)

# Convert to pandas data frame, join with initial conditions data frame
dfFinal = pd.DataFrame({"dLbolAge" : lum, "dLXUVAge" : lumXUV, "dRadiusAge" : radius,
                        "dLRatioAge" : xuvBol, "dTempAge" : temp},
                        index=dfInit.index)

# Join on index and dump to a csv
df = dfInit.join(dfFinal)
df.to_csv("mcResults.csv", header=True, index=True)

# Done!
