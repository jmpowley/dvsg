import numpy as np
import matplotlib.pyplot as plt

from .calculate_dvsg_v3 import load_map
from .dvsg.dvsg_tools import exclude_above_five_sigma, normalise_velocity_map

def mask_maps_for_plotting(sv_map, gv_map, sv_mask, gv_mask):
    """
    Apply masks to all spaxels in velocity map.

    Used only for generating plots.
    """
    sv_ma = sv_map.copy()
    gv_ma = gv_map.copy()

    sv_ma[sv_mask.astype(bool)] = np.nan
    gv_ma[gv_mask.astype(bool)] = np.nan

    return sv_ma, gv_ma

def make_stellar_gas_residual_maps_for_plotting(plateifu):

    # Load maps
    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_ra, bin_dec, x_as, y_as = load_map(plateifu, mode='local', bintype='VOR10')

    # Apply masks
    sv_ma, gv_ma = mask_maps_for_plotting(sv_map, gv_map, sv_mask, gv_mask)

    # Prepare maps
    sv_excl = exclude_above_five_sigma(sv_ma)
    gv_excl = exclude_above_five_sigma(gv_ma)
    sv_norm = normalise_velocity_map(sv_excl)
    gv_norm = normalise_velocity_map(gv_excl)

    residual = np.abs(sv_norm - gv_norm)

    return sv_norm, gv_norm, residual

def plot_stellar_gas_residual_maps(x_as, y_as, bin_x, bin_y, sv_norm, gv_norm, residual, dvsg, dvsg_stderr, plot_kwargs):

    # Extract plot kwargs 
    # -- formatting
    labsize = plot_kwargs.get('labsize')
    txtsize = plot_kwargs.get('txtsize')
    tcksize = plot_kwargs.get('tcksize')
    labelpad = plot_kwargs.get('labelpad')
    # -- booleans
    plot_stderr = plot_kwargs.get('plot_stderr')
    plot_bins = plot_kwargs.get('plot_bins')

    # Enable LaTeX
    plt.rcParams.update({
        'text.usetex': True,
        # default LaTeX serif (Computer Modern / Latin Modern)
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        # other preamble
        'text.latex.preamble':
            r'\usepackage[T1]{fontenc}' '\n'
            r'\usepackage{lmodern}'
    })

    labsize = 20
    txtsize = 20
    tcksize = 12

    # Create figure
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21,6))

    # Stellar
    im0 = ax[0].pcolormesh(x_as, y_as, sv_norm, cmap='RdBu_r', shading='auto')
    cb0 = fig.colorbar(im0, fraction=0.05, pad=0.03)
    cb0.set_label(r"$V_\star\ / \ \mathrm{Norm.\ (km\ s^{-1})}$", labelpad=labelpad, fontsize=labsize)

    # Gas
    im1 = ax[1].pcolormesh(x_as, y_as, gv_norm, cmap='RdBu_r', shading='auto')
    gv_cb = fig.colorbar(im1, fraction=0.05, pad=0.03)
    gv_cb.set_label(r"$V_{g}\ / \ \mathrm{Norm.\ (km\ s^{-1})}$", labelpad=labelpad, fontsize=labsize)

    # Residual
    im3 = ax[2].pcolormesh(x_as, y_as, residual, cmap='viridis', shading='auto')
    cb3 = fig.colorbar(im3, fraction=0.05, pad=0.03)
    cb3.set_label(r"Residual / Norm.", labelpad=labelpad, fontsize=labsize)

    # Add DVSG value ()
    dvsg_str = rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f}' if not plot_stderr else rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f} ± {dvsg_stderr:.2f}'
    ax[2].text(0.97, 0.03, dvsg_str, fontsize=txtsize, transform=ax[2].transAxes, va='bottom', ha='right')

    # Plotting code for each subplot
    for i in range(ncols):
        # Add labels
        ax[i].set_xlabel(r'$\Delta \alpha \ \;[\mathrm{arcsec}]$', size=labsize)
        ax[i].set_ylabel(r'$\Delta \delta \ \;[\mathrm{arcsec}]$', size=labsize)
        
        # Invert RA axis
        ax[i].invert_xaxis()

        # Plot bins
        if plot_bins:
            ax[i].scatter(bin_x, bin_y, color='k', marker='.', s=50, lw=0)

    plt.tight_layout()