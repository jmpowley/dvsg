# DVSG
A Python module to calculate the $\Delta V_{\star-g}$ (pronounced 'DVSG') value of a galaxy.

## Background
Following **Powley et al. (in prep.)**, a galaxy's $\Delta V_{\star-g}$ value is defined as:

$$
\Delta V_{\star-g} = \frac{1}{N} \sum_{j} \left| {V_{\star,\text{norm}}^{j} - V^{j}_{g,\text{norm}}} \right|
$$

where:
- $V_{\star,\text{norm}}$ is the **normalised stellar velocity map**,
- $V_{g,\text{norm}}$ is the **normalised gas velocity map**,
- $\sum_{j} \left| {V_{\star,\text{norm}}^{j} - V^{j}_{g,\text{norm}}} \right|$ is the **sum** over all spaxels/bins, $j$, of the **absolute difference** between the normalised stellar and gas velocity maps
- $N$ is the **number of bins/spaxels** contributing towards the sum

A $\Delta V_{\star-g}$ value of 0 would imply no difference in the kinematics of stellar and gas velocity, whereas a $\Delta V_{\star-g}$ value close to 1 suggests one of the largest offsets between the stellar and gas velocity. The general trend is that the more kinematically disturbed galaxies tend to have larger $\Delta V_{\star-g}$ values. For more information about $\Delta V_{\star-g}$, please refer to Powley et al. (in prep.).

## Overview

The `calculate_dvsg` module contains all the necessary functions to calculate the $\Delta V_{\star-g}$ value of a galaxy, provided one has already obtained the stellar and gas velocity maps.

The `modelling` module contains the code used to create mock data and test how $\Delta V_{\star-g}$ changes as a function of offset angle or strength of assymetric drift.

The `plotting` module contains scripts used to quickly produce plots from possible output map products from `calculate_dvsg`.