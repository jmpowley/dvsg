import os

import numpy as np

from marvin import config
from marvin.tools import Maps
from mangadap.util.fitsutil import DAPFitsUtil

from astropy.io import fits
from astropy.table import Table

from .dvsg_tools import exclude_above_five_sigma, minmax_normalise_velocity_map, zscore1_normalise_velocity_map, zscore5_normalise_velocity_map, robust_scale_velocity_map, mad5_normalise_velocity_map

# -------------------
# Preprocessing steps
# -------------------
def load_map(plateifu : str, mode : str, bintype: str, **extras):
    """ 
    Loads map products for a MaNGA galaxy from an input plateifu

    Returns the velocity map, mask, inverse variance map, and bin information for the stellar and Halpha gas map.
    """

    # Access maps locally
    if mode == 'local':
        plate, ifu = plateifu.split('-')
        local_path = f'/Users/Jonah/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/VOR10-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-VOR10-MILESHC-MASTARSSP.fits.gz'

        # Only run if local path exists
        if os.path.exists(local_path):
            try:
                hdul = fits.open(local_path)

                # Extract stellar and gas velocity products
                # -- velocity maps
                sv_map = hdul['STELLAR_VEL'].data
                gv_map = hdul['EMLINE_GVEL'].data[23]
                # -- masks
                sv_mask = hdul['STELLAR_VEL_MASK'].data
                gv_mask = hdul['EMLINE_GVEL_MASK'].data[23]
                # -- inverse variance maps
                sv_ivar = hdul['STELLAR_VEL_IVAR'].data
                gv_ivar = hdul['EMLINE_GVEL_IVAR'].data[23]
                # -- bin information
                bin_ids = hdul['BINID'].data
                # -- spatial information
                x_as, y_as = hdul['SPX_SKYCOO'].data  # x/y spaxel offsets in arcseconds
                bin_ra, bin_dec = hdul['BIN_LWSKYCOO'].data  # bin offsets in RA and DEC
            except Exception as e:
                print(f'Error loading {plateifu} using local method: {e}')
        else:
            raise ValueError(f'Local path {local_path} could not be found')

    # Access maps remotely
    else:
        try:
            # Set Marvin release to DR17 and avoid API error
            config.setDR('DR17')
            config.switchSasUrl(sasmode='mirror')

            maps = Maps(plateifu=plateifu, mode='remote', bintype=bintype)
            print('Using remote file')
            
            # Extract stellar and gas velocity products
            # -- velocity maps
            sv_map = maps['stellar_vel'].value
            gv_map = maps['emline_gvel_ha_6564'].value
            # -- masks
            sv_mask = maps['stellar_vel_mask'].value
            gv_mask = maps['emline_gvel_mask_ha_6564'].value
            # -- inverse variance maps
            sv_ivar = maps['stellar_vel_ivar'].data
            gv_ivar = maps['emline_gvel_ivar_ha_6564'].value
            # -- bin ids
            # TODO: Add bin_id info for remote maps

        except Exception as e:
            print(f'Error loading {plateifu} using remote method: {e}')

    return sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_ra, bin_dec, x_as, y_as

def return_bin_indices(bin_ids):
    """
    Returns the unique bins and bin indices of a MaNGA galaxy from the input bin ids and x and y bin information.
    """

    # Get the unique indices of the stellar and gas velocity bins
    sv_bins = bin_ids[1]
    gv_bins = bin_ids[3]
    sv_ubins, sv_uindx = DAPFitsUtil.unique_bins(sv_bins, return_index=True)
    gv_ubins, gv_uindx = DAPFitsUtil.unique_bins(gv_bins, return_index=True)

    return sv_ubins, sv_uindx, gv_ubins, gv_uindx

def return_bin_coords(bin_ra, bin_dec, sv_uindx, gv_uindx):
    """
    Returns bin x and y coordinates of the stellar and gas velocity bins
    """

    # Use sv_uindx for bin ids as stellar and gas bins are the same
    bin_x = bin_ra.ravel()[sv_uindx]
    bin_y = bin_dec.ravel()[sv_uindx]

    bin_coords = np.vstack([bin_x, bin_y]).T

    return bin_coords

def mask_map(sv_map, gv_map, sv_mask, gv_mask, bin_ids, **extras):
    """
    Extracts stellar and gas velocity values in map from each bin.
    
    Returns values as flattened and masked numpy arrays.
    """

    # Get the unique indices of the stellar and gas velocity bins
    sv_ubins, sv_uindx, gv_ubins, gv_uindx = return_bin_indices(bin_ids)

    # Apply mask and save a masked array
    sv_flat = np.ma.MaskedArray(sv_map.ravel()[sv_uindx], mask=sv_mask.ravel()[sv_uindx] > 0)  # Masked stellar velocity positions
    gv_flat = np.ma.MaskedArray(gv_map.ravel()[gv_uindx], mask=gv_mask.ravel()[gv_uindx] > 0)  # Masked gas velocity positions

    return sv_flat, gv_flat

def apply_sigma_clip(sv_flat, gv_flat):

    # Apply sigma clip
    sv_excl = exclude_above_five_sigma(sv_flat)
    gv_excl = exclude_above_five_sigma(gv_flat)

    return sv_excl, gv_excl

def normalise_map(sv_excl, gv_excl, norm_method, **extras):
    """
    Apply preprocessing steps to velocity maps before DVSG calculation.

    Currently applies five sigma clip and normalises between -1 and 1.

    Returns
    -------
    sv_norm, gv_norm : np.ndarray
        Preprocessed stellar and gas velocity maps.
    """

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
        raise ValueError("norm_method must be 'minmax', 'zscore1', 'zscore5' or 'robust'")

    return sv_norm, gv_norm

def calculate_dvsg(sv_norm, gv_norm, return_stderr=False, return_residual=False, **extras):
    """
    Calculates the DVSG value and standard error using the normalised stellar and gas velocity maps.

    Optionally returns residual map used in calculation.
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
    
def calculate_dvsg_error_analytic(sv_ivar, gv_ivar, sv_excl, gv_excl, norm_method, **extras):

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
    if n == 0:
        return np.nan
    
    term = ((2.0 / sv_range) * sv_err) ** 2 + ((2.0 / gv_range) * gv_err) ** 2
    dvsg_err = (1.0 / n) * np.sqrt(np.sum(term))

    return dvsg_err

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

def return_dvsg_table(plateifu_list, **dvsg_kwargs):
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