import os

import numpy as np

from astropy.io import fits

from marvin import config
from marvin.tools import Maps

from mangadap.util.fitsutil import DAPFitsUtil

__all__ = [
    "download_map_from_plateifu",
    "load_local_hdul",
    "load_maps",
    "load_map_coords",
    "return_bin_indices",
    "return_bin_coord_centres",
    "return_bin_coords",
]

def download_map_from_plateifu(plateifu, bintype, **extras):
    """Attempts to download the map for a given plateifu and bintype, and prints an error if it fails.

    Parameters
    ----------
    plateifu : _type_
        _description_
    bintype : _type_
        _description_
    """

    # Set Marvin release to DR17 and avoid API error
    config.setDR("DR17")
    config.switchSasUrl(sasmode="mirror")

    try:
        map = Maps(plateifu, mode="remote", bintype=bintype)
        map.download()
        print(f"Map {plateifu} downloaded!")
    except Exception as e:
        print(f"Error: unable to download map {plateifu}: {e}")


def load_local_hdul(plateifu : str, bintype: str, **extras):
    """Loads a local DAP product from the MANGA_SPECTRO_ANALYSIS directory for a given plateifu and bintype.

    Parameters
    ----------
    plateifu : str
        MaNGA plateifu of the galaxy
    bintype : str
        Binning type of the DAP product

    Returns
    -------
    hdul : astropy.io.fits.HDUList
        The HDUList of the local DAP product for the given plateifu and bintype
    """

    # Load local path
    analysis_dir = os.environ.get("MANGA_SPECTRO_ANALYSIS")
    if analysis_dir is None:
        raise EnvironmentError("MANGA_SPECTRO_ANALYSIS is not set. Please set this environment variable.")
    
    plate, ifu = plateifu.split("-")
    local_path = os.path.join(
        analysis_dir,
        f"{bintype}-MILESHC-MASTARSSP",
        plate,
        ifu,
        f"manga-{plate}-{ifu}-MAPS-{bintype}-MILESHC-MASTARSSP.fits.gz"
    )
    
    if os.path.exists(local_path):
        hdul = fits.open(local_path)
        return hdul
    else:
        raise FileNotFoundError(f"No local file found at {local_path}")


def load_maps(plateifu: str, mode: str, bintype: str, **extras):
    """
    Loads stellar and gas velocity maps, masks, inverse variance maps, and bin information for a MaNGA galaxy.

    Parameters
    ----------
    plateifu : str
        MaNGA plateifu of the galaxy
    mode : str
        How to access the maps: "local" for local files, "remote" for Marvin API
    bintype : str
        Binning type of the DAP product

    Returns
    -------
    sv_map : np.ndarray
        Stellar velocity map
    gv_map : np.ndarray
        Gas velocity map (H-alpha)
    sv_mask : np.ndarray
        Stellar velocity mask
    gv_mask : np.ndarray
        Gas velocity mask
    sv_ivar : np.ndarray
        Stellar velocity inverse variance
    gv_ivar : np.ndarray
        Gas velocity inverse variance
    bin_ids : np.ndarray or None
        Bin IDs for each spaxel (local only)
    bin_snr : np.ndarray or None
        Signal-to-noise ratio per bin (local only)
    """

    if mode == "local":
        hdul = load_local_hdul(plateifu, bintype)

        # Extract velocity maps, masks, and inverse variance
        sv_map = hdul["STELLAR_VEL"].data
        gv_map = hdul["EMLINE_GVEL"].data[23]  # H-alpha
        sv_mask = hdul["STELLAR_VEL_MASK"].data
        gv_mask = hdul["EMLINE_GVEL_MASK"].data[23]
        sv_ivar = hdul["STELLAR_VEL_IVAR"].data
        gv_ivar = hdul["EMLINE_GVEL_IVAR"].data[23]
        bin_ids = hdul["BINID"].data
        bin_snr = hdul["BIN_SNR"].data

    elif mode == "remote":
        try:
            config.setDR("DR17")
            config.switchSasUrl(sasmode="mirror")

            maps = Maps(plateifu=plateifu, mode="remote", bintype=bintype)

            # Extract maps and masks from Marvin object
            sv_map = maps["stellar_vel"].value
            gv_map = maps["emline_gvel_ha_6564"].value
            sv_mask = maps["stellar_vel_mask"].value
            gv_mask = maps["emline_gvel_mask_ha_6564"].value
            sv_ivar = maps["stellar_vel_ivar"].data
            gv_ivar = maps["emline_gvel_ivar_ha_6564"].value

            # Bin info is not available remotely
            bin_ids = None
            bin_snr = None

        except Exception as e:
            print(f"Error loading {plateifu} using remote method: {e}")
            return None

    else:
        raise ValueError(f"Invalid mode, got {mode}")

    return sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_snr


def load_map_coords(plateifu: str, mode: str, bintype: str, **extras):
    """
    Loads spatial coordinates of MaNGA maps in arcseconds for spaxels and bins.

    Parameters
    ----------
    plateifu : str
        MaNGA plateifu of the galaxy
    mode : str
        How to access the maps: "local" or "remote"
    bintype : str
        Binning type of the DAP product

    Returns
    -------
    x_as : np.ndarray
        Spaxel x offsets in arcseconds
    y_as : np.ndarray
        Spaxel y offsets in arcseconds
    bin_ra : np.ndarray
        RA of each bin
    bin_dec : np.ndarray
        DEC of each bin
    """

    if mode == "local":
        hdul = load_local_hdul(plateifu, bintype)

        # Spaxel offsets in sky coordinates
        x_as, y_as = hdul["SPX_SKYCOO"].data

        # Bin centre coordinates in RA/DEC
        bin_ra, bin_dec = hdul["BIN_LWSKYCOO"].data

    elif mode == "remote":
        # TODO: Add spatial info for remote maps
        x_as, y_as, bin_ra, bin_dec = None, None, None, None

    else:
        raise ValueError(f"Invalid mode, got {mode}")

    return x_as, y_as, bin_ra, bin_dec


def return_bin_indices(bin_ids):
    """
    Returns the unique bin IDs and their corresponding indices for stellar and gas velocity maps.

    Parameters
    ----------
    bin_ids : np.ndarray
        Bin ID array for the galaxy

    Returns
    -------
    sv_ubins : np.ndarray
        Unique stellar velocity bin IDs
    sv_uindx : np.ndarray
        Indices corresponding to the unique stellar bins
    gv_ubins : np.ndarray
        Unique gas velocity bin IDs
    gv_uindx : np.ndarray
        Indices corresponding to the unique gas bins
    """

    sv_bins = bin_ids[1]
    gv_bins = bin_ids[3]

    sv_ubins, sv_uindx = DAPFitsUtil.unique_bins(sv_bins, return_index=True)
    gv_ubins, gv_uindx = DAPFitsUtil.unique_bins(gv_bins, return_index=True)

    return sv_ubins, sv_uindx, gv_ubins, gv_uindx


def return_bin_coord_centres(bin_ra, bin_dec, sv_uindx, gv_uindx):
    """
    Returns the RA/DEC centres of each stellar and gas bin.

    Parameters
    ----------
    bin_ra : np.ndarray
        RA of all bins
    bin_dec : np.ndarray
        DEC of all bins
    sv_uindx : np.ndarray
        Indices of unique stellar bins
    gv_uindx : np.ndarray
        Indices of unique gas bins

    Returns
    -------
    bin_centres : np.ndarray
        Array of shape (nbins, 2) containing [DEC, RA] of each bin centre
    """

    # Stellar bins used as reference; RA/DEC of each bin centre
    bin_x = bin_ra.ravel()[sv_uindx]
    bin_y = bin_dec.ravel()[sv_uindx]

    bin_centres = np.vstack([bin_y, bin_x]).T
    return bin_centres


def return_bin_coords(bin_ids):
    """
    Returns the spaxel coordinates for each bin in the galaxy.

    Parameters
    ----------
    bin_ids : np.ndarray
        Bin ID array of the galaxy

    Returns
    -------
    bin_coords_list : list of np.ndarray
        List of (y, x) coordinates for each bin
    """

    nbins = np.size(np.unique(bin_ids)) - 1
    bin_coords_list = []

    for i in range(nbins):
        bin_ys, bin_xs = np.where(bin_ids == i)
        bin_coords = np.vstack([bin_ys, bin_xs]).T
        bin_coords_list.append(bin_coords)

    return bin_coords_list