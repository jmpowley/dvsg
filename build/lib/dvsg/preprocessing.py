import numpy as np

from .helpers import return_bin_indices, load_maps

__all__ = [
    "exclude_above_n_sigma",
    "minmax_normalise_velocity_map",
    "zscore1_normalise_velocity_map",
    "zscore5_normalise_velocity_map",
    "robust_scale_velocity_map",
    "mad5_normalise_velocity_map",
    "mask_velocity_maps",
    "mask_binned_map",
    "apply_bin_snr_threshold",
    "apply_velocity_snr_threshold",
    "apply_sigma_clip",
    "normalise_map",
    "preprocess_maps_from_plateifu",
]

# ------------------------
# Sigma-clipping functions
# ------------------------

def exclude_above_n_sigma(velocity_map, n: int):
    """Excludes any values in a velocity map greater than n standard deviations
    from the mean velocity of the map (excluding NaNs).

    The standard deviation used is the sample standard deviation (ddof=1).

    Parameters
    ----------
    velocity_map : array_like
        The original velocity map.

    Returns
    -------
    excluded_velocity_map : np.ndarray
        The n-sigma clipped velocity map
    """

    velocity_above_n_sigma = np.nanmean(velocity_map) + n * np.nanstd(velocity_map, ddof=1)
    velocity_below_n_sigma = np.nanmean(velocity_map) - n * np.nanstd(velocity_map, ddof=1)
    
    excluded_velocity_map = velocity_map.copy()

    excluded_velocity_map[velocity_map > velocity_above_n_sigma] = np.nan
    excluded_velocity_map[velocity_map < velocity_below_n_sigma] = np.nan

    return excluded_velocity_map


# -----------------------
# Normalisation functions
# -----------------------
def minmax_normalise_velocity_map(velocity_map):
    """
    Normalises the given velocity map to the range [-1, 1] using the formula:

        x' = 2 * ((x - min(x)) / (max(x) - min(x))) - 1 

    Parameters
    ----------
    velocity_map : array_like
        Input velocity values.

    Returns
    -------
    np.ndarray
        Normalised map. NaNs are preserved.
    """

    velocity_map = np.asarray(velocity_map, dtype=float)  # ensure float copy

    min_val, max_val = np.nanmin(velocity_map), np.nanmax(velocity_map)

    if min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
        return np.full_like(velocity_map, np.nan)

    normalised_velocity_map = 2 * (velocity_map - min_val) / (max_val - min_val) - 1

    return normalised_velocity_map


def zscore1_normalise_velocity_map(velocity_map):
    """Apply z-score normalisation ``(X - mean) / std`` to a velocity map.

    Parameters
    ----------
    velocity_map : array_like
        Input velocity values.

    Returns
    -------
    np.ndarray
        Z-score normalised map.
    """
    velocity_map = np.asarray(velocity_map, dtype=float)  # ensure float copy

    if np.nanmin(velocity_map) == np.nanmax(velocity_map):
        return np.full_like(velocity_map, np.nan)

    mean_val = np.nanmean(velocity_map)
    std_val = np.nanstd(velocity_map, ddof=1)

    if std_val == 0 or np.isnan(std_val):
        return np.full_like(velocity_map, np.nan)

    normalised_map = (velocity_map - mean_val) / std_val

    return normalised_map


def zscore5_normalise_velocity_map(velocity_map):
    """Apply 5-sigma z-score scaling ``(X - mean) / (5*std)``.

    Parameters
    ----------
    velocity_map : array_like
        Input velocity values.

    Returns
    -------
    np.ndarray
        Scaled z-score map.
    """
    velocity_map = np.asarray(velocity_map, dtype=float)  # ensure float copy

    if np.nanmin(velocity_map) == np.nanmax(velocity_map):
        return np.full_like(velocity_map, np.nan)

    mean_val = np.nanmean(velocity_map)
    std_val = np.nanstd(velocity_map, ddof=1)

    if std_val == 0 or np.isnan(std_val):
        return np.full_like(velocity_map, np.nan)

    normalised_map = (velocity_map - mean_val) / (5 * std_val)

    return normalised_map


def robust_scale_velocity_map(velocity_map):
    """Apply robust median/IQR scaling to a velocity map.

    Parameters
    ----------
    velocity_map : array_like
        Input velocity values.

    Returns
    -------
    np.ndarray
        Robust-scaled map.
    """

    velocity_25, velocity_50, velocity_75 = np.nanpercentile(velocity_map, q=[25, 50, 75])

    robust_scaled_map = (velocity_map - velocity_50) / (velocity_75 - velocity_25)

    return robust_scaled_map


def mad5_normalise_velocity_map(velocity_map):
    """
    Apply 5σ robust (MAD-based) normalisation to a velocity map.

    Z = (X - median(X)) / (5 * 1.4826 * MAD)

    Where MAD = median(|X - median(X)|).
    This is more robust to outliers than a standard deviation-based z-score.

    The resulting map has median ≈ 0.
    Values within ±5σ (in a Gaussian sense) map roughly to [-1, 1].

    Parameters
    ----------
    velocity_map : array_like
        Input velocity values.

    Returns
    -------
    np.ndarray
        MAD-based normalised map.
    """
    velocity_map = np.asarray(velocity_map, dtype=float)

    if np.all(np.isnan(velocity_map)):
        return np.full_like(velocity_map, np.nan)
    if np.nanmin(velocity_map) == np.nanmax(velocity_map):
        return np.full_like(velocity_map, np.nan)

    median_val = np.nanmedian(velocity_map)
    mad = np.nanmedian(np.abs(velocity_map - median_val))

    if mad == 0 or np.isnan(mad):
        return np.full_like(velocity_map, np.nan)

    # Convert MAD to an equivalent σ estimate and normalise
    scale = 5.0 * 1.4826 * mad
    normalised_map = (velocity_map - median_val) / scale

    # Optional: hard clip to [-1, 1]
    # normalised_map = np.clip(normalised_map, -1.0, 1.0)

    return normalised_map

# -----------------------
# Preprocessing functions
# -----------------------
def mask_velocity_maps(sv_map: np.ndarray, gv_map: np.ndarray, sv_mask: np.ndarray, gv_mask: np.ndarray, bin_ids: np.ndarray, **extras):
    """Flatten stellar/gas maps to one value per bin and apply masks.

    Parameters
    ----------
    sv_map, gv_map : np.ndarray
        Stellar and gas map arrays.
    sv_mask, gv_mask : np.ndarray
        Bitmask arrays for the corresponding maps.
    bin_ids : np.ndarray
        BINID cube used to index bin representatives.

    Returns
    -------
    sv_flat, gv_flat : np.ma.MaskedArray
        Flattened, masked stellar and gas values.
    """

    # Get the unique indices of the stellar and gas velocity bins
    _, sv_uindx, _, gv_uindx = return_bin_indices(bin_ids)

    # Apply mask and save a masked array
    sv_flat = np.ma.MaskedArray(sv_map.ravel()[sv_uindx], mask=sv_mask.ravel()[sv_uindx] > 0)
    gv_flat = np.ma.MaskedArray(gv_map.ravel()[gv_uindx], mask=gv_mask.ravel()[gv_uindx] > 0)

    return sv_flat, gv_flat


def mask_binned_map(map, mask, bin_ids, **extras):
    """Flatten a binned map to stellar-bin representatives and apply mask."""

    # Get the unique indices of the stellar and gas velocity bins
    _, sv_uindx, _, _ = return_bin_indices(bin_ids)

    flat = np.ma.MaskedArray(map.ravel()[sv_uindx], mask=mask.ravel()[sv_uindx] > 0)

    return flat


def apply_bin_snr_threshold(sv_flat, gv_flat, bin_snr_flat, snr_threshold: float, **extras):
    """Mask stellar/gas bins below a bin-level SNR threshold."""

    if snr_threshold is None:
        return sv_flat, gv_flat

    reject = bin_snr_flat < snr_threshold

    sv_flat[reject] = np.nan
    gv_flat[reject] = np.nan

    return sv_flat, gv_flat


def apply_velocity_snr_threshold(sv_flat, gv_flat, sv_ivar_flat, gv_ivar_flat, snr_threshold: float, **extras):
    """Mask bins below velocity SNR threshold computed from IVAR."""

    # Calculate S/N ratio of velocity values
    sv_snr = np.abs(sv_flat) / (1 / np.sqrt(sv_ivar_flat))
    gv_snr = np.abs(gv_flat) / (1 / np.sqrt(gv_ivar_flat))

    # Apply mask
    sv_reject = sv_snr < snr_threshold
    gv_reject = gv_snr < snr_threshold

    sv_flat[sv_reject] = np.nan
    gv_flat[gv_reject] = np.nan

    return sv_flat, gv_flat


def apply_sigma_clip(sv_flat, gv_flat, n_sigma: float, **extras):
    """Apply symmetric n-sigma clipping to stellar and gas arrays."""

    # Apply sigma clip
    sv_excl = exclude_above_n_sigma(sv_flat, n_sigma)
    gv_excl = exclude_above_n_sigma(gv_flat, n_sigma)

    return sv_excl, gv_excl


def normalise_map(sv_excl, gv_excl, norm_method, **extras):
    """
    Normalise preprocessed stellar and gas arrays with one method.

    Applies one of:
    - ``minmax``: map to [-1, 1]
    - ``zscore1``: standard z-score
    - ``zscore5``: z-score scaled by 5σ
    - ``robust``: median/IQR scaling
    - ``mad5``: median/MAD scaling with 5σ equivalent

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
        raise ValueError("norm_method must be 'minmax', 'zscore1', 'zscore5', 'robust' or 'mad5'")

    return sv_norm, gv_norm


# -----------------------------------
# Multi-stage preprocessing functions
# -----------------------------------
def preprocess_maps_from_plateifu(plateifu: str, **dvsg_kwargs):
    """Run the standard preprocessing chain for one plateifu.

    Steps are: load maps, flatten/mask bins, optional bin-SNR cut,
    sigma clipping, then map normalisation.
    """

    # Load map
    sv_map, gv_map, sv_mask, gv_mask, _, _, bin_ids, bin_snr = load_maps(plateifu, **dvsg_kwargs)
    # Extract masked values and flatten
    sv_flat, gv_flat = mask_velocity_maps(sv_map, gv_map, sv_mask, gv_mask, bin_ids)
    bin_snr_flat = mask_binned_map(bin_snr, sv_mask, bin_ids)  # use stellar velocity mask

    # Apply SNR threshold
    if dvsg_kwargs.get("snr_threshold") is not None:
        sv_flat, gv_flat = apply_bin_snr_threshold(sv_flat, gv_flat, bin_snr_flat, **dvsg_kwargs)

    # Sigma clip and normalise maps
    sv_clip, gv_clip = apply_sigma_clip(sv_flat, gv_flat, **dvsg_kwargs)
    sv_norm, gv_norm = normalise_map(sv_clip, gv_clip, **dvsg_kwargs)

    return sv_norm, gv_norm
