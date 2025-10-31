import numpy as np
from marvin import config

def download_map_from_plateifu(plateifu, bintype):

    # Set Marvin release to DR17 and avoid API error
    config.setDR('DR17')
    config.switchSasUrl(sasmode='mirror')

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

def zscore_normalise_velocity_map(velocity_map: np.ndarray):
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