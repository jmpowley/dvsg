import numpy as np

from astropy.table import Table

from .helpers import load_map, return_bin_indices, return_bin_coords
from .preprocessing import mask_map, apply_sigma_clip, normalise_map

def calculate_dvsg(sv_norm, gv_norm, return_stderr=False, return_residual=False, **extras):
    """Base function to calculate the DVSG value of a galaxy

    Parameters
    ----------
    sv_norm : array_like
        Normalised stellar velocity map
    gv_norm : array_like
        Normalised gas velocity map
    return_stderr : bool, optional
        Return the standard error on the DVSG value, by default False
    return_residual : bool, optional
        Return the residual map from the DVSG calculation, by default False

    Returns
    -------
    dvsg (dvsg_err, residual)
        DVSG value of the galaxy (optionally, with the error on the DVSG value and the residual)
    """

    if not np.shape(sv_norm) == np.shape(gv_norm):
        raise Exception('sv_norm and gv_norm must have the same shape. Currently have shapes ' + str(np.shape(sv_norm)) + ' and ' + str(np.shape(gv_norm)))
    
    residual = np.abs(sv_norm - gv_norm)
    dvsg = np.nanmean(residual)
    dvsg_stderr = np.nanstd(residual) / np.sqrt(np.count_nonzero(residual))

    # Optionally, return residual and standard error
    if return_residual:
        if return_stderr:
            return dvsg, dvsg_stderr, residual
        else:
            return dvsg, residual
    else:
        if return_stderr:
            return dvsg, dvsg_stderr
        else:
            return dvsg

def calculate_radial_dvsg(bin_coords, residual, sort_ascending : bool, **extras):
    """Calculate radial distance of bin co-ordinates and return with residuals

    Parameters
    ----------
    bin_coords : array_like
        Co-ordinates of bins.
    residual : array_like
        Residuals of each bin
    sort_ascending : bool
        Returns arrays sorted in ascending order by the bin distance

    Returns
    -------
    bin_dists : array_like
        Distances from each bin to the centre
    residual : array_like
        Same as input, but sorted by bin_dists if sort_ascending = True
    """
    
    # Calculate distance from each bin to centre
    bin_dists = np.sqrt(np.sum(bin_coords**2, axis=1))  # (x^2+y^2) summed along co-ordinate axis then sqrted

    if sort_ascending:
        bin_dists, residual = zip(*sorted(zip(bin_dists, residual)))

    return bin_dists, residual

def calculate_dvsg_error_analytic(sv_ivar, gv_ivar, sv_excl, gv_excl, norm_method, **extras):
    """Calculates the analytic uncertainty on the DVSG value of a galaxy

    Parameters
    ----------
    sv_ivar : array_like
        Inverse variance of stellar velocity map (default from MaNGA pipeline)
    gv_ivar : array_like
        Inverse variance of stellar velocity map
    sv_excl : array_like
        Sigma-clipped stellar velocity map
    gv_excl : array_like
        Sigma-clipped gas velocity map
    norm_method : str
        Normalisation method using in the DVSG calculation. Must be ``minmax`` for propogate error

    Returns
    -------
    dvsg_err : float (None if norm_method not ``minmax``)
        Uncertainty on DVSG from error propagation
    """

    if norm_method != "minmax":
        return None

    sv_range = np.nanmax(sv_excl) - np.nanmin(sv_excl)
    gv_range = np.nanmax(gv_excl) - np.nanmin(gv_excl)

    # Convert ivar to sigma, but only where ivar is positive and finite
    sv_ok = np.isfinite(sv_excl) & np.isfinite(sv_ivar) & (sv_ivar > 0)
    gv_ok = np.isfinite(gv_excl) & np.isfinite(gv_ivar) & (gv_ivar > 0)
    ok = sv_ok & gv_ok
    
    sv_err = np.sqrt(1.0 / sv_ivar[ok])
    gv_err = np.sqrt(1.0 / gv_ivar[ok])

    n = np.count_nonzero(ok)
    
    term = ((2.0 / sv_range) * sv_err) ** 2 + ((2.0 / gv_range) * gv_err) ** 2
    dvsg_err = (1.0 / n) * np.sqrt(np.sum(term))

    return dvsg_err

# ----------------------
# Routines from plateifu
# ----------------------
def calculate_dvsg_from_plateifu(plateifu, **dvsg_kwargs):

    # Load map
    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_ra, bin_dec, x_as, y_as = load_map(plateifu, **dvsg_kwargs)

    # Extract masked values from bins
    sv_flat, gv_flat = mask_map(sv_map, gv_map, sv_mask, gv_mask, bin_ids)
    sv_ivar_flat, gv_ivar_flat = mask_map(sv_ivar, gv_ivar, sv_mask, gv_mask, bin_ids)

    # Sigma clip and normalise maps
    sv_excl, gv_excl = apply_sigma_clip(sv_flat, gv_flat)
    sv_norm, gv_norm = normalise_map(sv_excl, gv_excl, **dvsg_kwargs)

    # Calculate DVSG
    dvsg, dvsg_stderr = calculate_dvsg(sv_norm, gv_norm, **dvsg_kwargs)
    
    # Calculate analytic error
    dvsg_properr = calculate_dvsg_error_analytic(sv_ivar_flat, gv_ivar_flat, sv_excl, gv_excl, **dvsg_kwargs)

    # Check error type
    err_type = dvsg_kwargs.get("error_type", "stderr")
    if err_type == "stderr":
        dvsg_err = dvsg_stderr
    elif err_type == "analytic":
        dvsg_err = dvsg_properr

    return dvsg, dvsg_err

def calculate_radial_dvsg_from_plateifu(plateifu : str, dvsg_kwargs : dict):

    # Execute pipeline to calculate residuals
    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_ra, bin_dec, x_as, y_as = load_map(plateifu, **dvsg_kwargs)
    sv_flat, gv_flat = mask_map(sv_map, gv_map, sv_mask, gv_mask, bin_ids, **dvsg_kwargs)
    sv_norm, gv_norm = normalise_map(sv_flat, gv_flat, **dvsg_kwargs)
    dvsg, dvsg_stderr, residual = calculate_dvsg(sv_norm, gv_norm, **dvsg_kwargs)
    
    # Load bin information
    sv_ubins, sv_uindx, gv_ubins, gv_uindx = return_bin_indices(bin_ids)
    bin_coords = return_bin_coords(bin_ra, bin_dec, sv_uindx, gv_uindx)

    # Calculate radial dvsg
    bin_dists, residual = calculate_radial_dvsg(bin_coords, residual, **dvsg_kwargs)

    return bin_dists, residual

def return_dvsg_table_from_plateifus(plateifu_list, **dvsg_kwargs):
    """
    Function to create a table of DVSG values for a set of MaNGA galaxies using their plateifus and kwargs to pass to the dvsg package

    Optionally adds the maps and binned data from each plateifu to the table
    """

    # Extract DVSG kwargs
    mode = dvsg_kwargs.get('mode')
    bintype = dvsg_kwargs.get('bintype')

    # Create lists to store DVSG data
    dvsg_list = []
    dvsg_err_list = []
    sv_map_list = []
    gv_map_list = []
    sv_norm_list = []
    gv_norm_list = []

    # Loop over each plateifu
    for plateifu in plateifu_list:

        dvsg, dvsg_err = calculate_dvsg_from_plateifu(plateifu, **dvsg_kwargs)

        # Add DVSG data to lists
        dvsg_list.append(dvsg)
        dvsg_err_list.append(dvsg_err)

    # Make astropy Table
    tb = Table()

    # Add columns of data to table
    tb['plateifu'] = plateifu_list
    tb['dvsg'] = dvsg_list
    tb['dvsg_err'] = dvsg_err_list
    # tb['sv_map'] = sv_map_list
    # tb['gv_map'] = gv_map_list
    # tb['sv_norm'] = sv_norm_list
    # tb['gv_norm'] = gv_norm_list

    return tb