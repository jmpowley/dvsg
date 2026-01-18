import os

import numpy as np

from astropy.io import fits

from marvin import config
from marvin.tools import Maps

from mangadap.util.fitsutil import DAPFitsUtil

# -----------------
# Loading functions
# -----------------
def download_map_from_plateifu(plateifu, bintype):

    # Set Marvin release to DR17 and avoid API error
    config.setDR("DR17")
    config.switchSasUrl(sasmode="mirror")

    # Check map not already downloaded
    plate, ifu = plateifu.split("-")
    local_path = f"/Users/Jonah/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/{bintype}-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-{bintype}-MILESHC-MASTARSSP.fits.gz"
    if os.path.exists(local_path):
        print(f"PLATEIFU {plateifu} already downloaded")
        return
    try:
        map = Maps(plateifu, mode="remote", bintype=bintype)
        map.download()
        print(f"Map {plateifu} downloaded!")
    except Exception as e:
        print(f"Error: unable to download map {plateifu}: {e}")


def load_local_hdul(plateifu : str, bintype: str):

    plate, ifu = plateifu.split("-")
    local_path = f"/Users/Jonah/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/VOR10-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-{bintype}-MILESHC-MASTARSSP.fits.gz"
    
    # Check local path exists
    if os.path.exists(local_path):
        hdul = fits.open(local_path)
    else:
        raise ValueError(f"No local path {local_path}")
    
    return hdul


def load_maps(plateifu : str, mode : str, bintype: str, **extras):
    """Loads map products for a MaNGA galaxy from an input plateifu.

    Returns the velocity map, mask, inverse variance map, and bin information for the stellar and Halpha gas map.
    """

    # Access maps locally
    if mode == "local":
        hdul = load_local_hdul(plateifu, bintype)

        # velocity maps
        sv_map = hdul["STELLAR_VEL"].data
        gv_map = hdul["EMLINE_GVEL"].data[23]
        
        # masks
        sv_mask = hdul["STELLAR_VEL_MASK"].data
        gv_mask = hdul["EMLINE_GVEL_MASK"].data[23]
        
        # inverse variance maps
        sv_ivar = hdul["STELLAR_VEL_IVAR"].data
        gv_ivar = hdul["EMLINE_GVEL_IVAR"].data[23]
        
        # bin information
        bin_ids = hdul["BINID"].data
        bin_snr = hdul["BIN_SNR"].data
    
    # Access maps remotely
    elif mode == "remote":
        try:
            # Set Marvin release to DR17 and avoid API error
            config.setDR("DR17")
            config.switchSasUrl(sasmode="mirror")

            maps = Maps(plateifu=plateifu, mode="remote", bintype=bintype)
            print("Using remote file")
            
            # Extract stellar and gas velocity products
            # -- velocity maps
            sv_map = maps["stellar_vel"].value
            gv_map = maps["emline_gvel_ha_6564"].value
            # -- masks
            sv_mask = maps["stellar_vel_mask"].value
            gv_mask = maps["emline_gvel_mask_ha_6564"].value
            # -- inverse variance maps
            sv_ivar = maps["stellar_vel_ivar"].data
            gv_ivar = maps["emline_gvel_ivar_ha_6564"].value
            # -- bin ids
            # TODO: Add bin info for remote maps
            bin_ids = None
            bin_snr = None

        except Exception as e:
            print(f"Error loading {plateifu} using remote method: {e}")

    else:
        raise ValueError(f"Invalid mode, got {mode}")

    return sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_snr


def load_map_coords(plateifu : str, mode : str, bintype: str, **extras):

    # Access maps locally
    if mode == "local":
        hdul = load_local_hdul(plateifu, bintype)

        # x/y spaxel offsets in arcseconds
        x_as, y_as = hdul["SPX_SKYCOO"].data

        # bin offsets in RA and DEC
        bin_ra, bin_dec = hdul["BIN_LWSKYCOO"].data

    # Access maps remotely
    elif mode == "remote":
        pass
        # TODO: Add spatial info for remote maps

    else:
        raise ValueError(f"Invalid mode, got {mode}")
    
    return x_as, y_as, bin_ra, bin_dec

# ----------------
# Return functions
# ----------------
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

def return_bin_coord_centres(bin_ra, bin_dec, sv_uindx, gv_uindx):
    """
    Returns the coordinate centres of the stellar and gas velocity bins
    """

    # Use sv_uindx for bin ids as stellar and gas bins are the same
    bin_x = bin_ra.ravel()[sv_uindx]
    bin_y = bin_dec.ravel()[sv_uindx]

    bin_centres = np.vstack([bin_y, bin_x]).T

    return bin_centres

def return_bin_coords(bin_ids):
    """Returns the co-ordinates of each spaxel in a given bin"""

    nbins = np.size(np.unique(bin_ids)) - 1
    bin_coords_list = []

    # Loop over bins
    for i in range(nbins):
        bin_ys, bin_xs = np.where(bin_ids == i)
        bin_coords = np.vstack([bin_ys, bin_xs]).T
        bin_coords_list.append(bin_coords)

    return bin_coords_list