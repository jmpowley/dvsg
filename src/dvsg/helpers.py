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


def _initialise_remote_access():
    """Initialise remote access to MaNGA data by setting Marvin configuration."""

    # Set Marvin release to DR17 and avoid API error
    config.setDR("DR17")
    config.switchSasUrl(sasmode="mirror")


# ----------------------
# Data loading functions
# ----------------------
def download_map_from_plateifu(plateifu, bintype, **extras):
    """Download a remote MaNGA MAPS file for a plateifu and bintype.

    Parameters
    ----------
    plateifu : str
        MaNGA plate-IFU identifier.
    bintype : str
        DAP binning type (for example ``VOR10``).
    """

    try:
        _initialise_remote_access()
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
    drp_version = os.environ.get("MANGADRP_VER")
    if drp_version is None:
        raise EnvironmentError("MANGADRP_VER is not set. Please set this environment variable.")
    dap_version = os.environ.get("MANGADAP_VER")
    if dap_version is None:
        raise EnvironmentError("MANGADAP_VER is not set. Please set this environment variable.")
    
    plate, ifu = plateifu.split("-")
    local_path = os.path.join(
        analysis_dir,
        drp_version,
        dap_version,
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
    Load MaNGA stellar/gas velocity products and bin metadata.

    Parameters
    ----------
    plateifu : str
        MaNGA plateifu of the galaxy
    mode : str
        Data access mode: ``local`` or ``remote``.
    bintype : str
        Binning type of the DAP product

    Returns
    -------
    sv_map : np.ndarray
        Stellar velocity map
    gv_map : np.ndarray
        Gas velocity map
    sv_mask : np.ndarray
        Stellar velocity mask
    gv_mask : np.ndarray
        Gas velocity mask
    sv_ivar : np.ndarray
        Stellar velocity inverse variance
    gv_ivar : np.ndarray
        Gas velocity inverse variance
    bin_ids : np.ndarray
        BINID cube in DAP channel order.
    bin_snr : np.ndarray
        Bin signal-to-noise map.
    """

    if mode == "local":
        try:
            hdul = load_local_hdul(plateifu, bintype)

            # Extract velocity products
            sv_map = hdul["STELLAR_VEL"].data
            gv_map = hdul["EMLINE_GVEL"].data[23]  # H-alpha channel
            sv_mask = hdul["STELLAR_VEL_MASK"].data
            gv_mask = hdul["EMLINE_GVEL_MASK"].data[23]
            sv_ivar = hdul["STELLAR_VEL_IVAR"].data
            gv_ivar = hdul["EMLINE_GVEL_IVAR"].data[23]
            
            # Extract bin information
            bin_ids = hdul["BINID"].data
            bin_snr = hdul["BIN_SNR"].data

        except Exception as e:
            raise RuntimeError(f"Error loading {plateifu} using local method: {e}") from e

    elif mode == "remote":
        try:
            _initialise_remote_access()
            maps = Maps(plateifu=plateifu, mode="remote", bintype=bintype)

            # Extract velocity products
            sv = maps.stellar_vel
            gv = maps.emline_gvel_ha_6564
            sv_map = sv.value
            gv_map = gv.value
            sv_mask = np.asarray(sv.mask)
            gv_mask = np.asarray(gv.mask)
            sv_ivar = np.asarray(sv.ivar)
            gv_ivar = np.asarray(gv.ivar)

            # Extract bin information
            bin_ids = np.stack([
                maps.binid_binned_spectra.value,
                maps.binid_stellar_continua.value,
                maps.binid_em_line_moments.value,
                maps.binid_em_line_models.value,
                maps.binid_spectral_indices.value,
                ],
                axis=0,
            )
            bin_snr = maps.bin_snr.value

        except Exception as e:
            raise RuntimeError(f"Error loading {plateifu} using remote method: {e}") from e

    else:
        raise ValueError(f"Invalid mode, got {mode}")

    return sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_snr


def load_map_coords(plateifu: str, mode: str, bintype: str, **extras):
    """
    Load map coordinate products for a MaNGA target.

    Parameters
    ----------
    plateifu : str
        MaNGA plateifu of the galaxy
    mode : str
        Data access mode: ``local`` or ``remote``.
    bintype : str
        Binning type of the DAP product

    Returns
    -------
    x_as, y_as : np.ndarray
        Spaxel offsets in arcseconds.
    bin_ra, bin_dec : np.ndarray
        Bin centre sky coordinates.
    """

    if mode == "local":
        hdul = load_local_hdul(plateifu, bintype)

        # Spaxel offsets in sky coordinates
        x_as, y_as = hdul["SPX_SKYCOO"].data

        # Bin centre coordinates in RA/DEC
        bin_ra, bin_dec = hdul["BIN_LWSKYCOO"].data

    elif mode == "remote":
        _initialise_remote_access()
        maps = Maps(plateifu=plateifu, mode="remote", bintype=bintype)
        
        x_as = maps.spx_skycoo_x.value
        y_as = maps.spx_skycoo_y.value
        bin_ra = maps.bin_lwskycoo_ra.value
        bin_dec = maps.bin_lwskycoo_dec.value

    else:
        raise ValueError(f"Invalid mode, got {mode}")

    return x_as, y_as, bin_ra, bin_dec


# ---------------------
# Bin utility functions
# ---------------------
def return_bin_indices(bin_ids):
    """
    Return unique stellar/gas bin IDs and representative indices.

    Parameters
    ----------
    bin_ids : np.ndarray
        Bin ID array for the galaxy

    Returns
    -------
    sv_ubins, gv_ubins : np.ndarray
        Unique stellar and gas bin IDs.
    sv_uindx, gv_uindx : np.ndarray
        Indices of representative spaxels for each unique bin.
    """

    sv_bins = bin_ids[1]
    gv_bins = bin_ids[3]

    sv_ubins, sv_uindx = DAPFitsUtil.unique_bins(sv_bins, return_index=True)
    gv_ubins, gv_uindx = DAPFitsUtil.unique_bins(gv_bins, return_index=True)

    return sv_ubins, sv_uindx, gv_ubins, gv_uindx


def return_bin_coord_centres(bin_ra, bin_dec, sv_uindx, gv_uindx):
    """
    Return bin centre coordinates for stellar-bin index ordering.

    Parameters
    ----------
    bin_ra : np.ndarray
        RA of all bins
    bin_dec : np.ndarray
        DEC of all bins
    sv_uindx : np.ndarray
        Representative stellar bin indices.
    gv_uindx : np.ndarray
        Representative gas bin indices (accepted for API consistency).

    Returns
    -------
    bin_centres : np.ndarray
        Array of shape ``(nbins, 2)`` with ``[DEC, RA]`` rows.
    """

    # Stellar bins used as reference; RA/DEC of each bin centre
    bin_x = bin_ra.ravel()[sv_uindx]
    bin_y = bin_dec.ravel()[sv_uindx]

    bin_centres = np.vstack([bin_y, bin_x]).T
    return bin_centres


def return_bin_coords(bin_ids):
    """
    Return spaxel coordinate lists for each bin ID.

    Parameters
    ----------
    bin_ids : np.ndarray
        Bin ID array of the galaxy

    Returns
    -------
    bin_coords_list : list[np.ndarray]
        One ``(n_spaxel, 2)`` array of ``(y, x)`` coordinates per bin.
    """

    nbins = np.size(np.unique(bin_ids)) - 1
    bin_coords_list = []

    for i in range(nbins):
        bin_ys, bin_xs = np.where(bin_ids == i)
        bin_coords = np.vstack([bin_ys, bin_xs]).T
        bin_coords_list.append(bin_coords)

    return bin_coords_list
