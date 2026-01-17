import numpy as np

from .helpers import return_bin_indices

# ------------------------
# Sigma-clipping functions
# ------------------------

def exclude_above_n_sigma(velocity_map: np.ndarray, n: int):
    """Excludes any values in a velocity map greater than n standard deviations
    from the mean velocity of the map (excluding NaNs).

    The standard deviation used is the sample standard deviation (ddof=1).

    Parameters
    ----------
    velocity_map : np.ndarray
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
def minmax_normalise_velocity_map(velocity_map: np.ndarray):
    """
    Normalises the given velocity map to the range [-1, 1] using the formula:

        x' = 2 * ((x - min(x)) / (max(x) - min(x))) - 1 

    NaN values are ignored when computing the minimum and maximum.
    """

    velocity_map = np.asarray(velocity_map, dtype=float)  # ensure float copy

    min_val, max_val = np.nanmin(velocity_map), np.nanmax(velocity_map)

    if min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
        return np.full_like(velocity_map, np.nan)

    normalised_velocity_map = 2 * (velocity_map - min_val) / (max_val - min_val) - 1

    return normalised_velocity_map


def zscore1_normalise_velocity_map(velocity_map: np.ndarray):
    '''
    Apply Gaussian/Z-score normalisation to a velocity map

    Z = (X - mu) / sigma

    Where X is the data, mu is the mean of the data and sigma is the standard deviation

    The normalised data has a mean of 0 and standard deviation of 1
    '''
    velocity_map = np.asarray(velocity_map, dtype=float)  # ensure float copy

    if np.nanmin(velocity_map) == np.nanmax(velocity_map):
        return np.full_like(velocity_map, np.nan)

    mean_val = np.nanmean(velocity_map)
    std_val = np.nanstd(velocity_map, ddof=1)

    if std_val == 0 or np.isnan(std_val):
        return np.full_like(velocity_map, np.nan)

    normalised_map = (velocity_map - mean_val) / std_val

    return normalised_map


def zscore5_normalise_velocity_map(velocity_map: np.ndarray):
    '''
    Apply Gaussian/Z-score normalisation to a velocity map

    Z = (X - mu) / (5 * sigma)

    Where X is the data, mu is the mean of the data and sigma is the standard deviation

    The normalised data has a mean of 0 and a range [-1, 1]
    '''
    velocity_map = np.asarray(velocity_map, dtype=float)  # ensure float copy

    if np.nanmin(velocity_map) == np.nanmax(velocity_map):
        return np.full_like(velocity_map, np.nan)

    mean_val = np.nanmean(velocity_map)
    std_val = np.nanstd(velocity_map, ddof=1)

    if std_val == 0 or np.isnan(std_val):
        return np.full_like(velocity_map, np.nan)

    normalised_map = (velocity_map - mean_val) / (5 * std_val)

    return normalised_map


def robust_scale_velocity_map(velocity_map: np.ndarray):

    velocity_25, velocity_50, velocity_75 = np.nanpercentile(velocity_map, q=[25, 50, 75])

    robust_scaled_map = (velocity_map - velocity_50) / (velocity_75 - velocity_25)

    return robust_scaled_map


def mad5_normalise_velocity_map(velocity_map: np.ndarray):
    """
    Apply 5σ robust (MAD-based) normalisation to a velocity map.

    Z = (X - median(X)) / (5 * 1.4826 * MAD)

    Where MAD = median(|X - median(X)|).
    This is more robust to outliers than a standard deviation-based z-score.

    The resulting map has median ≈ 0.
    Values within ±5σ (in a Gaussian sense) map roughly to [-1, 1].
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
def mask_velocity_maps(sv_map, gv_map, sv_mask, gv_mask, bin_ids, **extras):
    """
    Extracts stellar and gas velocity values in map from each bin.
    
    Returns values as flattened and masked numpy arrays.
    """

    # Get the unique indices of the stellar and gas velocity bins
    sv_ubins, sv_uindx, gv_ubins, gv_uindx = return_bin_indices(bin_ids)

    # Apply mask and save a masked array
    sv_flat = np.ma.MaskedArray(sv_map.ravel()[sv_uindx], mask=sv_mask.ravel()[sv_uindx] > 0)
    gv_flat = np.ma.MaskedArray(gv_map.ravel()[gv_uindx], mask=gv_mask.ravel()[gv_uindx] > 0)

    return sv_flat, gv_flat


def mask_binned_map(map, mask, bin_ids, **extras):

    # Get the unique indices of the stellar and gas velocity bins
    sv_ubins, sv_uindx, gv_ubins, gv_uindx = return_bin_indices(bin_ids)

    flat = np.ma.MaskedArray(map.ravel()[sv_uindx], mask=mask.ravel()[sv_uindx] > 0)

    return flat


def apply_bin_snr_threshold(sv_flat, gv_flat, bin_snr_flat, snr_threshold, **extras):

    reject = bin_snr_flat < snr_threshold

    sv_flat[reject] = np.nan
    gv_flat[reject] = np.nan

    return sv_flat, gv_flat


def apply_velocity_snr_threshold(sv_flat, gv_flat, sv_ivar_flat, gv_ivar_flat, snr_threshold: float, **extras):

    # Calculate S/N ratio of velocity values
    sv_snr = np.abs(sv_flat) / (1 / np.sqrt(sv_ivar_flat))
    gv_snr = np.abs(gv_flat) / (1 / np.sqrt(gv_ivar_flat))

    # Apply mask
    sv_reject = sv_snr < snr_threshold
    gv_reject = gv_snr < snr_threshold

    sv_flat[sv_reject] = np.nan
    gv_flat[gv_reject] = np.nan

    return sv_flat, gv_flat


def apply_sigma_clip(sv_flat, gv_flat, n: int = 3):

    # Apply sigma clip
    sv_excl = exclude_above_n_sigma(sv_flat, n=n)
    gv_excl = exclude_above_n_sigma(gv_flat, n=n)

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