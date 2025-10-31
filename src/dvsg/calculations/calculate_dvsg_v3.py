import os

import numpy as np

from marvin import config
from marvin.tools import Maps
from mangadap.util.fitsutil import DAPFitsUtil

from astropy.io import fits
from astropy.table import Table

from .dvsg_tools import exclude_above_five_sigma, normalise_velocity_map, minmax_normalise_velocity_map, zscore_normalise_velocity_map

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

    return bin_x, bin_y

def mask_map(sv_map, gv_map, sv_mask, gv_mask, bin_ids):
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

def prepare_map(sv_flat, gv_flat, norm_method, **extras):
    """
    Apply preprocessing steps to velocity maps before DVSG calculation.

    Currently applies five sigma clip and normalises between -1 and 1.

    Returns
    -------
    sv_norm, gv_norm : np.ndarray
        Preprocessed stellar and gas velocity maps.
    """

    # Apply sigma clip
    sv_excl = exclude_above_five_sigma(sv_flat)
    gv_excl = exclude_above_five_sigma(gv_flat)

    # Normalise velocity map
    if norm_method == "minmax":
        sv_norm = minmax_normalise_velocity_map(sv_excl)
        gv_norm = minmax_normalise_velocity_map(gv_excl)
    elif norm_method == "zscore":
        sv_norm = zscore_normalise_velocity_map(sv_excl)
        gv_norm = zscore_normalise_velocity_map(gv_excl)
    else:
        raise ValueError("norm_method must be 'minmax' or 'zscore'")

    return sv_norm, gv_norm

def calculate_dvsg(sv_norm, gv_norm, return_residual=False, **extras):
    """
    Calculates the DVSG value and standard error using the normalised stellar and gas velocity maps.

    Optionally returns residual map used in calculation.
    """

    if not np.shape(sv_norm) == np.shape(gv_norm):
        raise Exception('sv_norm and gv_norm must have the same shape. Currently have shapes ' + str(np.shape(sv_norm)) + ' and ' + str(np.shape(gv_norm)))
    
    residual_map = np.abs(sv_norm - gv_norm)
    dvsg = np.nanmean(residual_map)
    dvsg_stderr = np.nanstd(residual_map) / np.sqrt(np.count_nonzero(residual_map))
    
    if return_residual:
        return dvsg, dvsg_stderr, residual_map
    else:
        return dvsg, dvsg_stderr

def calculate_dvsg_from_plateifu(plateifu, **dvsg_kwargs):

    # Load map
    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_ra, bin_dec, x_as, y_as = load_map(plateifu, **dvsg_kwargs)

    # Extract masked values from bins
    sv_flat, gv_flat = mask_map(sv_map, gv_map, sv_mask, gv_mask, bin_ids)

    # Sigma clip and normalise maps
    sv_norm, gv_norm = prepare_map(sv_flat, gv_flat, **dvsg_kwargs)

    # Calculate DVSG
    dvsg, dvsg_stderr = calculate_dvsg(sv_norm, gv_norm, **dvsg_kwargs)

    return dvsg, dvsg_stderr

def calculate_radial_dvsg():
    pass

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
    dvsg_stderr_list = []
    sv_map_list = []
    gv_map_list = []
    sv_norm_list = []
    gv_norm_list = []

    # Loop over each plateifu
    for plateifu in plateifu_list:

        # Load map
        sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_ra, bin_dec, x_as, y_as = load_map(plateifu, **dvsg_kwargs)

        # Extract masked values from bins
        sv_flat, gv_flat = mask_map(sv_map, gv_map, sv_mask, gv_mask, bin_ids)

        # Sigma clip and normalise maps
        sv_norm, gv_norm = prepare_map(sv_flat, gv_flat, **dvsg_kwargs)

        # Calculate DVSG
        dvsg, dvsg_stderr = calculate_dvsg(sv_norm, gv_norm, return_residual=False)

        # Add DVSG data to lists
        dvsg_list.append(dvsg)
        dvsg_stderr_list.append(dvsg_stderr)
        # sv_map_list.append(sv_map)
        # gv_map_list.append(gv_map)
        # sv_norm_list.append(sv_norm)
        # gv_norm_list.append(gv_norm)

    # Make astropy Table
    tb = Table()

    # Add columns of data to table
    tb['plateifu'] = plateifu_list
    tb['dvsg'] = dvsg_list
    tb['dvsg_stderr'] = dvsg_stderr_list
    # tb['sv_map'] = sv_map_list
    # tb['gv_map'] = gv_map_list
    # tb['sv_norm'] = sv_norm_list
    # tb['gv_norm'] = gv_norm_list

    return tb