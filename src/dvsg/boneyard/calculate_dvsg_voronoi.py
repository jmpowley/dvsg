import numpy as np
import pandas as pd
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from marvin import config
from marvin.tools import Maps
from astropy.io import fits
import os

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
    '''Normalises the given velocity map to the range [-1, 1].

    Normalisation uses the formula:

        x' = 2 * ((x - min(x)) / (max(x) - min(x))) - 1 

    NaN values are ignored when computing the minimum and maximum.

    Parameters
    ----------
    velocity_map : np.ndarray
        The unnormalised velocity map.

    Returns
    -------
    normalised_velocity_map : np.ndarray
        The normalised velocity map, with values in the range [-1, 1].
        If all finite values in velocity_map are identical, returns NaNs.
    '''

    min_val, max_val = np.nanmin(velocity_map), np.nanmax(velocity_map)

    if min_val == max_val:
        return np.full_like(velocity_map, np.nan)  # Avoid division by zero

    normalised_velocity_map = 2 * (velocity_map - min_val) / (max_val - min_val) - 1

    return normalised_velocity_map

def calculate_residual_map(sv_map, gv_map):

    if not np.shape(sv_map) == np.shape(gv_map):
        raise Exception('sv_map and gv_map must have the same shape. Currently have shapes ' + str(np.shape(sv_map)) + ' and ' + str(np.shape(gv_map)))
    
    residual_map = np.abs(sv_map - gv_map)
    residual_mean = np.nanmean(residual_map)
    residual_stderr = np.nanstd(residual_map) / np.sqrt(np.count_nonzero(residual_map))
    
    return residual_map, residual_mean, residual_stderr

def calculate_DVSG_voronoi(residual_map, noise_map, weighted, target_sn):
    xs = []
    ys = []
    signals = []
    noises = []

    x_size, y_size = np.shape(residual_map)
    for x in range(x_size):
        for y in range(y_size):
            if np.isfinite(residual_map[x, y]) and np.isfinite(noise_map[x, y]):
                xs.append(x)
                ys.append(y)
                signals.append(residual_map[x, y])
                noises.append(noise_map[x, y])
                
    xs = np.asarray(xs, dtype=int)
    ys = np.asarray(ys, dtype=int)
    signals = np.asarray(signals)
    noises = np.asarray(noises)

    # Run the binning algorithm for target S/N
    bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
        xs, ys, signals, noises, target_sn,
        cvt=False, pixelsize=None, plot=False, quiet=True, sn_func=None, wvt=True
    )

    nx, ny = residual_map.shape
    x0, y0 = nx//2, ny//2

    # Find unique bins
    unique_bins = np.unique(bin_number)
    n_bins = len(unique_bins)

    # Initialise arrays for rebinned data
    binned_signal = np.zeros(n_bins)
    binned_noise = np.zeros(n_bins)
    radii = np.zeros(n_bins)

    # Find mean signal in each bin
    for i, b in enumerate(unique_bins):
        # Find mean signal in each bin
        indices = np.where(bin_number == b)[0]    
        binned_signal[i] = np.mean(signals[indices])
        binned_noise[i] = np.sqrt(np.sum(noises[indices]**2)) / len(indices)

        # Calculate radius of bin centroid
        radii[i] = np.hypot(x_bar[i] - x0,  y_bar[i] - y0)

    # Create DataFrame for radial DVSG information
    radial_df = pd.DataFrame({
        'bin'      : unique_bins,
        'x_center' : x_bar,
        'y_center' : y_bar,
        'radius'   : radii,
        'DVSG_in_bin': binned_signal
    })

    if weighted:
        weights = nPixels  # Pixel counts per bin
        variance = np.sum((weights**2) * (binned_noise**2)) / (np.sum(weights)**2)  # Weighted standard error
        dvsg = np.sum(binned_signal * weights) / np.sum(weights)
        dvsg_stderr = np.sqrt(variance) / np.sqrt(np.size(binned_signal))
    else:
        # Recalculate DVSG and error
        dvsg = np.mean(binned_signal)
        dvsg_stderr = np.std(binned_signal, ddof=1) / np.sqrt(len(binned_signal))

    # Create output map
    dvsg_map = np.full((x_size, y_size), np.nan)
    bin_to_value = {b: binned_signal[i] for i, b in enumerate(unique_bins)}  # Dictionary to bin number to aggregated value
    for i in range(len(xs)):
        x_coord = xs[i]
        y_coord = ys[i]
        b = bin_number[i]
        dvsg_map[x_coord, y_coord] = bin_to_value[b]

    return dvsg_map, dvsg, dvsg_stderr

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

def load_map_from_plateifu(plateifu, mode, bintype):

    # Access maps locally
    if mode == 'local':
        plate, ifu = plateifu.split('-')
        local_path = f'/Users/Jonah/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/VOR10-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-VOR10-MILESHC-MASTARSSP.fits.gz'

        # Only run if local path exists
        if os.path.exists(local_path):
            try:
                hdul = fits.open(local_path)

                # Extract stellar and gas velocity maps and their masks (local)
                sv_map = hdul['STELLAR_VEL'].data
                gv_map = hdul['EMLINE_GVEL'].data[23]
                sv_mask = hdul['STELLAR_VEL_MASK'].data
                gv_mask = hdul['EMLINE_GVEL_MASK'].data[23]
                sv_ivar = hdul['STELLAR_VEL_IVAR'].data
                gv_ivar = hdul['EMLINE_GVEL_IVAR'].data[23]
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
        except Exception as e:
            print(f'Error loading {plateifu} using remote method: {e}')

    return sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar

def prepare_map_for_dvsg_calculation(sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar):

    # Apply masks
    sv_map[sv_mask.astype(bool)] = np.nan
    gv_map[gv_mask.astype(bool)] = np.nan
    sv_ivar[sv_mask.astype(bool)] = np.nan
    gv_ivar[gv_mask.astype(bool)] = np.nan

    # Exclude values over 5 sigma
    sv_excl = exclude_above_five_sigma(sv_map)
    gv_excl = exclude_above_five_sigma(gv_map)

    # Normalise values to between -1 and 1
    sv_norm = normalise_velocity_map(sv_excl)
    gv_norm = normalise_velocity_map(gv_excl)

    # Match spaxels between inverse variance map and excluded map
    sv_ivar[np.isnan(sv_excl)] = np.nan
    gv_ivar[np.isnan(gv_excl)] = np.nan

    # Calculate combined quadrature noise of stellar and gas velocity
    sv_err = np.sqrt(1 / sv_ivar)
    gv_err = np.sqrt(1/ gv_ivar)
    sv_var = (1 / sv_ivar) * (2 / (np.nanmax(sv_excl)- np.nanmin(sv_excl)))**2
    gv_var = (1 / gv_ivar) * (2 / (np.nanmax(gv_excl) - np.nanmin(gv_excl)))**2
    noise_map = np.sqrt(sv_var + gv_var)

    return sv_norm, gv_norm, noise_map

def get_DVSG_from_plateifu(plateifu, target_sn, weighted, download, bintype, mode, return_stellar_gas_maps):

    # Download map if not already downloaded and change mode to local if not already
    if download:
        plate, ifu = plateifu.split('-')
        local_path = f'/Users/Jonah/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/VOR10-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-{bintype}-MILESHC-MASTARSSP.fits.gz'
        if not os.path.exists(local_path):
            download_map_from_plateifu(plateifu, bintype)
        mode = 'local'

    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar = load_map_from_plateifu(plateifu, mode=mode, bintype='VOR10')

    sv_norm, gv_norm, noise_map = prepare_map_for_dvsg_calculation(sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar)

    residual_map,_,_ = calculate_residual_map(sv_norm, gv_norm)

    dvsg_map, dvsg, dvsg_stderr = calculate_DVSG_voronoi(residual_map, noise_map, weighted=weighted, target_sn=target_sn)

    if return_stellar_gas_maps:
        return dvsg_map, dvsg, dvsg_stderr, sv_norm, gv_norm
    else:
        return dvsg_map, dvsg, dvsg_stderr

def calculate_radial_DVSG(residual_map, noise_map, target_sn):

    xs = []
    ys = []
    signals = []
    noises = []

    x_size, y_size = np.shape(residual_map)
    for x in range(x_size):
        for y in range(y_size):
            if np.isfinite(residual_map[x, y]) and np.isfinite(noise_map[x, y]):
                xs.append(x)
                ys.append(y)
                signals.append(residual_map[x, y])
                noises.append(noise_map[x, y])
                
    xs = np.asarray(xs, dtype=int)
    ys = np.asarray(ys, dtype=int)
    signals = np.asarray(signals)
    noises = np.asarray(noises)

    # Run the binning algorithm for target S/N
    bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
        xs, ys, signals, noises, target_sn,
        cvt=False, pixelsize=None, plot=False, quiet=True, sn_func=None, wvt=True
    )

    # Find centre of map
    nx, ny = residual_map.shape
    x0, y0 = nx//2, ny//2

    # Find unique bins
    unique_bins = np.unique(bin_number)
    n_bins = len(unique_bins)

    # Initialise arrays for rebinned data
    binned_signal = np.zeros(n_bins)
    binned_noise = np.zeros(n_bins)
    radii = np.zeros(n_bins)

    # Find mean signal in each bin
    for i, b in enumerate(unique_bins):
        # Find mean signal in each bin
        indices = np.where(bin_number == b)[0]    
        binned_signal[i] = np.mean(signals[indices])
        binned_noise[i] = np.sqrt(np.sum(noises[indices]**2)) / len(indices)

        # Calculate radius of bin centroid
        radii[i] = np.hypot(x_bar[i] - x0,  y_bar[i] - y0)

    # Create DataFrame for radial DVSG information
    radial_df = pd.DataFrame({
        'bin'      : unique_bins,
        'x_center' : x_bar,
        'y_center' : y_bar,
        'radius'   : radii,
        'DVSG_in_bin': binned_signal
    })

    return radial_df

def get_radial_DVSG_from_plateifu(plateifu, target_sn, download, bintype, mode):

    # Download map if not already downloaded and change mode to local if not already
    if download:
        plate, ifu = plateifu.split('-')
        local_path = f'/Users/Jonah/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/VOR10-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-{bintype}-MILESHC-MASTARSSP.fits.gz'
        if not os.path.exists(local_path):
            download_map_from_plateifu(plateifu, bintype)
        mode = 'local'

    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar = load_map_from_plateifu(plateifu, mode=mode, bintype=bintype)

    sv_norm, gv_norm, noise_map = prepare_map_for_dvsg_calculation(sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar)

    residual_map,_,_ = calculate_residual_map(sv_norm, gv_norm)

    radial_df = calculate_radial_DVSG(residual_map, noise_map, target_sn)

    return radial_df