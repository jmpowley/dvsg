# DVSG
A Python module to calculate the kinematic disturbance parameter $\Delta V_{\star-g}$ (pronounced 'DVSG') for a galaxy.

## Background
Following **Powley et al. (submitted.)**, a galaxy's $\Delta V_{\star-g}$ value is defined as:

$$
\Delta V_{\star-g} = \frac{1}{N} \sum_{j} \left| {V_{\star,\text{norm}}^{j} - V^{j}_{g,\text{norm}}} \right|
$$

where:
- $V_{\star,\text{norm}}$ is the **normalised stellar velocity map**,
- $V_{g,\text{norm}}$ is the **normalised gas velocity map**,
- $\sum_{j} \left| {V_{\star,\text{norm}}^{j} - V^{j}_{g,\text{norm}}} \right|$ is the **sum** over all spaxels/bins, $j$, of the **absolute difference** between the normalised stellar and gas velocity maps
- $N$ is the **number of bins/spaxels** contributing towards the sum

A lower $\Delta V_{\star-g}$ value implies that there is more similarity between the kinematics a galaxy's of stellar and gas velocity components, whereas a higher $\Delta V_{\star-g}$ implies a greater degree of kinematic disturbance. For more details information about the steps to $\Delta V_{\star-g}$,  please refer to Powley et al. (in submiited.).

## Overview

The `calculations` module contains functions to calculate the $\Delta V_{\star-g}$ value of a galaxy.

The `modelling` module contains the code used to create mock data to how $\Delta V_{\star-g}$ changes as a function of offset angle or strength of assymetric drift.

The `plotting` module contains funtions that can used to quickly produce science plots from a $\Delta V_{\star-g}$ calculation.

The `helpers` module contains commonly-used functions in the above modules