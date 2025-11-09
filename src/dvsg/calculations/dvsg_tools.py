import os

import numpy as np

from marvin import config
from marvin.tools import Maps

def download_map_from_plateifu(plateifu, bintype):

    # Set Marvin release to DR17 and avoid API error
    config.setDR('DR17')
    config.switchSasUrl(sasmode='mirror')

    # Check map not already downloaded
    plate, ifu = plateifu.split('-')
    local_path = f'/Users/Jonah/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/{bintype}-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-{bintype}-MILESHC-MASTARSSP.fits.gz'
    if os.path.exists(local_path):
        print(f"PLATEIFU {plateifu} already downloaded")
        return
    
    try:
        map = Maps(plateifu, mode='remote', bintype=bintype)
        map.download()
        print(f'Map {plateifu} downloaded!')
    except Exception as e:
        print(f'Error: unable to download map {plateifu}: {e}')

def exclude_above_five_sigma(velocity_map: np.ndarray):
    '''Excludes any values in a velocity map greater than 5 standard deviations
    from the mean velocity of the map.
    
    NaN values are ignored when computing the mean and standard deviation.

    The standard deviation used is the sample standard deviation (ddof=1).

    Parameters
    ----------
    velocity_map : np.ndarray
        The original velocity map.

    Returns
    -------
    excluded_velocity_map : np.ndarray
        The original velocity map, but with all velocity values greater 
        than 5 standard deviations from the mean set to NaN.
    '''

    velocity_above_five_sigma = np.nanmean(velocity_map) + 5 * np.nanstd(velocity_map, ddof=1)
    velocity_below_five_sigma = np.nanmean(velocity_map) - 5 * np.nanstd(velocity_map, ddof=1)
    
    excluded_velocity_map = velocity_map.copy()

    excluded_velocity_map[velocity_map > velocity_above_five_sigma] = np.nan
    excluded_velocity_map[velocity_map < velocity_below_five_sigma] = np.nan

    return excluded_velocity_map

def normalise_velocity_map(velocity_map: np.ndarray):
    '''
    Normalises the given velocity map to the range [-1, 1] using the formula:

        x' = 2 * ((x - min(x)) / (max(x) - min(x))) - 1 

    NaN values are ignored when computing the minimum and maximum.
    '''

    min_val, max_val = np.nanmin(velocity_map), np.nanmax(velocity_map)

    if min_val == max_val:
        return np.full_like(velocity_map, np.nan)  # Avoid division by zero

    normalised_velocity_map = 2 * (velocity_map - min_val) / (max_val - min_val) - 1

    return normalised_velocity_map

def minmax_normalise_velocity_map(velocity_map: np.ndarray):
    '''
    Normalises the given velocity map to the range [-1, 1] using the formula:

        x' = 2 * ((x - min(x)) / (max(x) - min(x))) - 1 

    NaN values are ignored when computing the minimum and maximum.
    '''
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