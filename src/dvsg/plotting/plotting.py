import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from ..calculations.calculate_dvsg import load_map
from ..calculations.dvsg_tools import exclude_above_five_sigma, minmax_normalise_velocity_map, zscore1_normalise_velocity_map, zscore5_normalise_velocity_map, robust_scale_velocity_map, mad5_normalise_velocity_map

import numpy as np

import numpy as np
from matplotlib.ticker import FixedLocator

def formatter_using_normalisers(original_velocity_map, norm_method="minmax", n_ticks=5, debug=False):
    """
    Build 5 evenly spaced normalised ticks and their mapped original-velocity values
    by applying the same normalization functions you already use and inverting
    via interpolation.

    Returns:
        locator, labels  -> FixedLocator for the colorbar ticks, and list of labels
    Usage:
        ticks = locator.locs  # or use locator.get_values()
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
    """
    # finite-only original values
    v = np.asarray(original_velocity_map, dtype=float)
    finite_mask = np.isfinite(v)
    v_fin = v[finite_mask]

    if v_fin.size == 0:
        if debug: print("[formatter] no finite pixels -> return nan labels")
        ticks = np.linspace(-1.0, 1.0, n_ticks)
        labels = [r"${:.2f}$ ({})".format(t, r"\mathrm{nan}") for t in ticks]
        return FixedLocator(ticks), labels

    # create monotonic grid of original values to evaluate the normaliser on
    # using unique sorted finite values is simplest; if too big, sample evenly
    orig_sorted = np.unique(np.sort(v_fin))
    if orig_sorted.size > 2000:
        # sample 2000 points uniformly from the cdf to speed up interpolation
        idx = np.linspace(0, orig_sorted.size - 1, 2000).astype(int)
        orig_sorted = orig_sorted[idx]

    # compute normalised values using the selected normaliser
    if norm_method == "minmax":
        norm_vals = minmax_normalise_velocity_map(orig_sorted)
    elif norm_method == "zscore1":
        norm_vals = zscore1_normalise_velocity_map(orig_sorted)
    elif norm_method == "zscore5":
        norm_vals = zscore5_normalise_velocity_map(orig_sorted)
    elif norm_method == "robust":
        norm_vals = robust_scale_velocity_map(orig_sorted)
    elif norm_method == "mad5":
        norm_vals = mad5_normalise_velocity_map(orig_sorted)
    else:
        raise ValueError("Unknown norm_method: " + str(norm_method))

    # drop NaNs from the mapping
    valid = np.isfinite(norm_vals) & np.isfinite(orig_sorted)
    if valid.sum() < 2:
        # not enough valid mapping points
        if debug: print("[formatter] too few valid points after normalisation")
        ticks = np.linspace(-1.0, 1.0, n_ticks)
        labels = [r"${:.2f}$ ({})".format(t, r"nan") for t in ticks]
        return FixedLocator(ticks), labels

    orig_sorted = orig_sorted[valid]
    norm_vals = norm_vals[valid]

    # ensure monotonicity of norm_vals for interpolation: sort pairs by norm_vals
    order = np.argsort(norm_vals)
    norm_sorted = norm_vals[order]
    orig_for_norm_sorted = orig_sorted[order]

    # If there are duplicate norm_sorted values, np.interp still works,
    # but we prefer strictly monotonic so we unique them
    uniq_norm, uniq_idx = np.unique(norm_sorted, return_index=True)
    norm_sorted = uniq_norm
    orig_for_norm_sorted = orig_for_norm_sorted[uniq_idx]

    if debug:
        print("[formatter] norm_method:", norm_method)
        print("[formatter] norm_sorted range:", norm_sorted[0], norm_sorted[-1], "n_points:", norm_sorted.size)

    # Choose n_ticks evenly spaced in the normalised range
    norm_min, norm_max = norm_sorted[0], norm_sorted[-1]
    ticks = np.linspace(norm_min, norm_max, n_ticks)

    # Invert mapping by interpolation (safe because norm_sorted is monotonic)
    orig_ticks = np.interp(ticks, norm_sorted, orig_for_norm_sorted)

    if debug:
        print("[formatter] ticks (normalised):", ticks)
        print("[formatter] ticks (original):", orig_ticks)

    # Build nicely formatted LaTeX labels (normalised with 2 dp, original with 1 dp)
    labels = []
    for nt, ot in zip(ticks, orig_ticks):
        if np.isfinite(ot):
            labels.append(r"${:.2f}$ ({:.1f})".format(nt, ot))
        else:
            labels.append(r"${:.2f}$ ({})".format(nt, r"\mathrm{nan}"))

    # Return a FixedLocator (so you can set it on the colorbar without warnings) and labels
    return FixedLocator(ticks), labels

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

def create_stellar_gas_residual_maps_for_plotting(plateifu : str, norm_method : str = "minmax", **extras):

    # Load maps
    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_ra, bin_dec, x_as, y_as = load_map(plateifu, mode='local', bintype='VOR10')

    # Apply masks
    sv_ma, gv_ma = mask_maps_for_plotting(sv_map, gv_map, sv_mask, gv_mask)

    # Apply sigma clip
    sv_excl = exclude_above_five_sigma(sv_ma)
    gv_excl = exclude_above_five_sigma(gv_ma)

    sv_excl = np.array(sv_excl, copy=True)
    gv_excl = np.array(gv_excl, copy=True)

    # Normalise velocity map
    if norm_method == "minmax":
        sv_norm = minmax_normalise_velocity_map(sv_excl)
        gv_norm = minmax_normalise_velocity_map(gv_excl)
    elif norm_method == "zscore1":
        sv_norm = zscore1_normalise_velocity_map(sv_excl)
        gv_norm = zscore1_normalise_velocity_map(gv_excl)
    elif norm_method == "zscore5":
        sv_norm = zscore5_normalise_velocity_map(sv_excl)
        gv_norm = zscore5_normalise_velocity_map(gv_excl)
    elif norm_method == "robust":
        sv_norm = robust_scale_velocity_map(sv_excl)
        gv_norm = robust_scale_velocity_map(gv_excl)
    elif norm_method == "mad5":
        sv_norm = mad5_normalise_velocity_map(sv_excl)
        gv_norm = mad5_normalise_velocity_map(gv_excl)
    else:
        raise ValueError("norm_method must be 'minmax', 'zscore1', 'zscore5', 'robust' or 'mad5'")

    residual = np.abs(sv_norm - gv_norm)

    return sv_norm, gv_norm, residual

def plot_stellar_gas_residual_maps(x_as, y_as, bin_ra, bin_dec, sv_norm, gv_norm, residual, dvsg, dvsg_stderr, plot_kwargs : dict = None):

    plot_defaults = {
        "labsize" : 20,
        "txtsize" : 20,
        "tcksize" : 20,
        "labelpad" : 0,
        "plot_stderr" : False,
        "plot_bins" : False
    }

    if plot_kwargs is not None:
        for key in plot_defaults.keys():
            if key not in plot_kwargs.keys():
                plot_kwargs[key] = plot_defaults[key]
    else:
        plot_kwargs = plot_defaults

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
            ax[i].scatter(bin_ra, bin_dec, color='k', marker='.', s=50, lw=0)

    plt.tight_layout()

def plot_stellar_gas_residual_maps_on_axes(
    ax,
    plateifu,
    x_as, y_as,
    bin_ra, bin_dec,
    sv_map, gv_map,
    sv_norm, gv_norm, residual,
    dvsg, dvsg_stderr,
    dvsg_kwargs: dict,
    r_eff: float = None,
    plot_kwargs: dict = None
):
    """
    Plot the 3-panel stellar / gas / residual maps on pre-existing axes.

    ax : array-like of 3 matplotlib Axes (ax[0], ax[1], ax[2])
    """

    # ----- defaults and apply kwargs (same pattern as your other functions)
    plot_defaults = {
        "labsize": 20,
        "txtsize": 20,
        "tcksize": 20,
        "labelpad": 0,
        "plot_stderr": False,
        "plot_bins": False,
        "plot_r_eff": False,
    }
    if plot_kwargs is not None:
        for key in plot_defaults:
            if key not in plot_kwargs:
                plot_kwargs[key] = plot_defaults[key]
    else:
        plot_kwargs = plot_defaults

    print(plot_kwargs)

    labsize = plot_kwargs.get('labsize')
    txtsize = plot_kwargs.get('txtsize')
    tcksize = plot_kwargs.get('tcksize')
    labelpad = plot_kwargs.get('labelpad')
    plot_stderr = plot_kwargs.get('plot_stderr')
    plot_bins = plot_kwargs.get('plot_bins')
    plot_r_eff = plot_kwargs.get('plot_r_eff')

    # ----- Stellar panel
    im0 = ax[0].pcolormesh(x_as, y_as, sv_norm, cmap='RdBu_r', shading='auto')
    cb0 = ax[0].figure.colorbar(im0, ax=ax[0], fraction=0.05, pad=0.03)
    cb0.set_label(r"$V_{\star}~\rm{Norm.~(km~s^{-1})}$",
                  labelpad=labelpad, fontsize=labsize)

    # use the same formatter routine you use elsewhere
    locator, labels = formatter_using_normalisers(sv_map, norm_method=dvsg_kwargs["norm_method"])
    cb0.set_ticks(locator.locs)
    cb0.set_ticklabels(labels)
    cb0.ax.tick_params(axis='y', which='major', labelsize=tcksize)

    ax[0].text(0.03, 0.97, plateifu, fontsize=txtsize, transform=ax[0].transAxes, va='top', ha='left')
    if plot_r_eff and r_eff is not None:
        ax[0].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor='k',
                              linewidth=1.2, transform=ax[0].transData))

    # ----- Gas panel
    im1 = ax[1].pcolormesh(x_as, y_as, gv_norm, cmap='RdBu_r', shading='auto')
    cb1 = ax[1].figure.colorbar(im1, ax=ax[1], fraction=0.05, pad=0.03)
    cb1.set_label(r"$V_{\rm gas}~\rm{Norm.~(km~s^{-1})}$",
                  labelpad=labelpad, fontsize=labsize)

    locator, labels = formatter_using_normalisers(gv_map, norm_method=dvsg_kwargs["norm_method"])
    cb1.set_ticks(locator.locs)
    cb1.set_ticklabels(labels)
    cb1.ax.tick_params(axis='y', which='major', labelsize=tcksize)

    if plot_r_eff and r_eff is not None:
        ax[1].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor='k',
                              linewidth=1.2, transform=ax[1].transData))

    # ----- Residual panel
    im2 = ax[2].pcolormesh(x_as, y_as, residual, cmap='viridis', shading='auto')
    cb2 = ax[2].figure.colorbar(im2, ax=ax[2], fraction=0.05, pad=0.03)
    cb2.set_label(r"Residual / Norm.", labelpad=labelpad, fontsize=labsize)
    cb2.ax.tick_params(axis='y', which='major', labelsize=tcksize)

    dvsg_str = (rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f}'
                if not plot_stderr else
                rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f} ± {dvsg_stderr:.2f}')
    ax[2].text(0.97, 0.03, dvsg_str, fontsize=txtsize,
               transform=ax[2].transAxes, va='bottom', ha='right')

    if plot_r_eff and r_eff is not None:
        ax[2].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor='k',
                              linewidth=1.2, transform=ax[2].transData))

    # ----- Subplot formatting (labels, invert RA, bins, aspect)
    for i in range(3):
        ax[i].set_xlabel(r'$\Delta \alpha~[\rm{arcsec}]$', size=labsize)
        ax[i].set_ylabel(r'$\Delta \delta ~[\rm{arcsec}]$', size=labsize)

        ax[i].tick_params(axis='both', which='major', labelsize=tcksize)

        ax[i].invert_xaxis()
        if plot_bins:
            ax[i].scatter(bin_ra, bin_dec, color='k', marker='.', s=50, lw=0)
        ax[i].set_aspect('equal')

    # Caller can call plt.tight_layout() or manage the figure layout.
    return ax

def plot_stellar_gas_residual_visual_maps(x_as, y_as, bin_x, bin_y, sv_norm, gv_norm, residual, sdss_im, dvsg, dvsg_stderr, r_eff : float = None, plot_kwargs : dict = None):

    plot_defaults = {
        "labsize" : 20,
        "txtsize" : 20,
        "tcksize" : 20,
        "labelpad" : 0,
        "plot_stderr" : False,
        "plot_bins" : False,
        "plot_r_eff" : False,
    }

    # Set plot kwargs
    if plot_kwargs is not None:
        for key in plot_defaults.keys():
            if key not in plot_kwargs.keys():
                plot_kwargs[key] = plot_defaults[key]
    else:
        plot_kwargs = plot_defaults

    # Extract plot kwargs
    # -- formatting
    labsize = plot_kwargs.get('labsize')
    txtsize = plot_kwargs.get('txtsize')
    tcksize = plot_kwargs.get('tcksize')
    labelpad = plot_kwargs.get('labelpad')
    # -- booleans
    plot_stderr = plot_kwargs.get('plot_stderr')
    plot_bins = plot_kwargs.get('plot_bins')
    plot_r_eff = plot_kwargs.get('plot_r_eff')

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

    # Create figure
    nrows, ncols = 1, 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21,5))

    # Stellar
    im0 = ax[0].pcolormesh(x_as, y_as, sv_norm, cmap='RdBu_r', shading='auto')
    cb0 = fig.colorbar(im0, fraction=0.05, pad=0.03)
    cb0.set_label(r"$V_\star\ / \ \mathrm{Norm.\ (km\ s^{-1})}$", labelpad=labelpad, fontsize=labsize)

    # Gas
    im1 = ax[1].pcolormesh(x_as, y_as, gv_norm, cmap='RdBu_r', shading='auto')
    gv_cb = fig.colorbar(im1, fraction=0.05, pad=0.03)
    gv_cb.set_label(r"$V_{g}\ / \ \mathrm{Norm.\ (km\ s^{-1})}$", labelpad=labelpad, fontsize=labsize)

    # Residual
    im2 = ax[2].pcolormesh(x_as, y_as, residual, cmap='viridis', shading='auto')
    cb3 = fig.colorbar(im2, fraction=0.05, pad=0.03)
    cb3.set_label(r"Residual / Norm.", labelpad=labelpad, fontsize=labsize)
    # -- add DVSG value
    dvsg_str = rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f}' if not plot_stderr else rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f} ± {dvsg_stderr:.2f}'
    ax[2].text(0.97, 0.03, dvsg_str, fontsize=txtsize, transform=ax[2].transAxes, va='bottom', ha='right')

    # Visual
    im3 = ax[3].imshow(sdss_im, origin="upper")

    # Plotting code for each subplot
    for i in range(ncols):

        # First three subplots
        if i < 3:
            # -- add labels
            ax[i].set_xlabel(r'$\Delta \alpha \ \;[\mathrm{arcsec}]$', size=labsize)
            ax[i].set_ylabel(r'$\Delta \delta \ \;[\mathrm{arcsec}]$', size=labsize)
            # -- invert RA axis
            ax[i].invert_xaxis()
            # -- plot bins
            if plot_bins:
                ax[i].scatter(bin_x, bin_y, color='k', marker='.', s=50, lw=0)

        if i == 3:
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        ax[i].set_aspect('equal')

    plt.tight_layout()

def plot_stellar_gas_residual_visual_maps_on_axes(ax, plateifu, x_as, y_as, bin_ra, bin_dec, sv_map, gv_map, sv_norm, gv_norm, residual, im, dvsg, dvsg_stderr, dvsg_kwargs, r_eff : float = None, plot_kwargs : dict = None):
    """
    Plot the 4-panel stellar/gas/residual/visual for a single galaxy on pre-existing axes.
    
    ax : array of 4 matplotlib axes (one row of 4 panels)
    Other arguments same as original function.
    """

    # Set plot kwargs
    # -- define defaults
    plot_defaults = {
        "labsize" : 20,
        "txtsize" : 20,
        "tcksize" : 20,
        "labelpad" : 0,
        "plot_stderr" : False,
        "plot_bins" : False,
        "plot_r_eff" : False,
    }
    # -- apply kwargs
    if plot_kwargs is not None:
        for key in plot_defaults.keys():
            if key not in plot_kwargs.keys():
                plot_kwargs[key] = plot_defaults[key]
    else:
        plot_kwargs = plot_defaults
    # -- extract
    labsize = plot_kwargs.get('labsize')
    txtsize = plot_kwargs.get('txtsize')
    tcksize = plot_kwargs.get('tcksize')
    labelpad = plot_kwargs.get('labelpad')
    plot_stderr = plot_kwargs.get('plot_stderr')
    plot_bins = plot_kwargs.get('plot_bins')
    plot_r_eff = plot_kwargs.get('plot_r_eff')

    # Stellar
    im0 = ax[0].pcolormesh(x_as, y_as, sv_norm, cmap='RdBu_r', shading='auto')
    cb0 = ax[0].figure.colorbar(im0, ax=ax[0], fraction=0.05, pad=0.03)
    cb0.set_label(r"$V_\star\ / \ \mathrm{Norm.\ (km\ s^{-1})}$", labelpad=labelpad, fontsize=labsize)
    ax[0].text(0.03, 0.97, plateifu, fontsize=txtsize, transform=ax[0].transAxes, va='top', ha='left')
    # ticks = cb0.get_ticks()
    # cb0.set_ticks(ticks)
    # cb0.set_ticklabels(formatter(ticks, sv_map, norm_method=dvsg_kwargs["norm_method"]))
    locator, labels = formatter_using_normalisers(sv_map, norm_method=dvsg_kwargs["norm_method"])
    cb0.set_ticks(locator.locs)
    cb0.set_ticklabels(labels)
    if plot_r_eff and r_eff is not None:
        ax[0].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor='k', linewidth=1.2, transform=ax[0].transData))

    # Gas
    im1 = ax[1].pcolormesh(x_as, y_as, gv_norm, cmap='RdBu_r', shading='auto')
    cb1 = ax[1].figure.colorbar(im1, ax=ax[1], fraction=0.05, pad=0.03)
    cb1.set_label(r"$V_{g}\ / \ \mathrm{Norm.\ (km\ s^{-1})}$", labelpad=labelpad, fontsize=labsize)
    # ticks = cb1.get_ticks()
    # cb1.set_ticks(ticks)
    # cb1.set_ticklabels(formatter(ticks, gv_map, norm_method=dvsg_kwargs["norm_method"]))
    locator, labels = formatter_using_normalisers(gv_map, norm_method=dvsg_kwargs["norm_method"])
    cb1.set_ticks(locator.locs)
    cb1.set_ticklabels(labels)
    if plot_r_eff and r_eff is not None:
        ax[1].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor='k', linewidth=1.2, transform=ax[1].transData))

    # Residual
    im2 = ax[2].pcolormesh(x_as, y_as, residual, cmap='viridis', shading='auto')
    cb2 = ax[2].figure.colorbar(im2, ax=ax[2], fraction=0.05, pad=0.03)
    cb2.set_label(r"Residual / Norm.", labelpad=labelpad, fontsize=labsize)
    dvsg_str = rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f}' if not plot_stderr else rf'$\Delta V_{{\star-g}}$ = {dvsg:.2f} ± {dvsg_stderr:.2f}'
    ax[2].text(0.97, 0.03, dvsg_str, fontsize=txtsize, transform=ax[2].transAxes, va='bottom', ha='right')
    if plot_r_eff and r_eff is not None:
        ax[2].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor='k', linewidth=1.2, transform=ax[2].transData))

    # Visual
    im3 = ax[3].imshow(im, origin="upper")

    # Subplot formatting
    for i in range(4):
        if i < 3:
            ax[i].set_xlabel(r'$\Delta \alpha \ \;[\mathrm{arcsec}]$', size=labsize)
            ax[i].set_ylabel(r'$\Delta \delta \ \;[\mathrm{arcsec}]$', size=labsize)
            ax[i].invert_xaxis()
            if plot_bins:
                ax[i].scatter(bin_ra, bin_dec, color='k', marker='.', s=50, lw=0)
        else:
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        ax[i].set_aspect('equal')

    plt.tight_layout()

    return ax