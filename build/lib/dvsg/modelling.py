from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import rotate

from .preprocessing import minmax_normalise_velocity_map

__all__ = [
    "cookie_cutter",
    "circular_mask",
    "MapModel",
]

# -------------
# Mask helpers
# -------------
def cookie_cutter(array: np.ndarray, size: int, set_edges_to_nan: bool):
    """Apply an octagon-like edge mask to a square 2D array."""

    size = array.shape[0]
    if array.shape != (size, size):
        raise ValueError("input_array must be shape (size, size)")

    # Distance from diagonal to perform cut
    diag_cut = int(0.65 * size)

    i = np.arange(size)[:, None]
    j = np.arange(size)[None, :]

    # Conditions for octagon-like mask
    c1 = (j - i) <= diag_cut
    c2 = (i - j) <= diag_cut
    c3 = (i + j) <= (size - 1) + diag_cut
    c4 = (i + j) >= (size - 1) - diag_cut
    edge_mask = c1 & c2 & c3 & c4  # True inside octagon, False in cut corners

    # Apply mask to input array
    result = array.copy()
    if set_edges_to_nan:
        result[~edge_mask] = np.nan
    else:
        result[~edge_mask] = 0.0

    return result


def circular_mask(array: np.ndarray,
                  centre: Tuple[float, float],
                  pixel_scale: float = 1.0,
                  radius: Optional[float] = None,
                  radius_units: str = "pixels",
                  set_edges_to_nan: bool = True,
                  return_mask: bool = False,
                  margin: int = 0):
    """Create or apply a circular mask to a 2D array."""

    if array.ndim != 2:
        raise ValueError("array must be 2D")

    ny, nx = array.shape
    x0, y0 = centre

    # Determine radius in pixels
    if radius is None:
        max_r = min(x0, y0, nx - 1 - x0, ny - 1 - y0) - margin
        r_pix = float(max_r)
    else:
        if radius_units == "pixels":
            r_pix = float(radius)
        elif radius_units == "arcsec":
            r_pix = float(radius) / pixel_scale
        else:
            raise ValueError("radius_units must be 'pixels' or 'arcsec'")

    if r_pix < 0:
        raise ValueError("Computed radius is negative")

    # Build distance grid
    j, i = np.indices((ny, nx))
    dx = i - x0
    dy = j - y0
    mask = (dx * dx + dy * dy) <= (r_pix * r_pix)

    if return_mask:
        return mask

    # Apply mask
    out = array.copy()
    if set_edges_to_nan:
        out[~mask] = np.nan
    else:
        out[~mask] = 0.0

    return out


# ---------------
# Map model class
# ---------------
class MapModel:
    """Create synthetic velocity maps for controlled DVSG tests."""

    def __init__(self,
                 map_type: str,
                 size: Optional[int] = 40,
                 pixel_scale: Optional[float] = 1.0,
                 center: Optional[Tuple[int, int]] = None,
                 seed: Optional[int] = None,
                 input_map: Optional[np.ndarray] = None,
                 input_mask: Optional[np.ndarray] = None,
                 map_kwargs: Optional[dict] = None,
                 ):

        self.size = size
        self.map_type = map_type
        self.pixel_scale = pixel_scale
        self.center = center if center is not None else (size // 2, size // 2)
        self.rng = np.random.default_rng(seed)
        self.map_kwargs = map_kwargs or {}
        self.input_map = input_map
        self.input_mask = input_mask

        self._initialise_map()

    def _initialise_map(self):
        """Build map from requested model type."""
        map_builders = {
            "rotation_dominated": self.rotation_dominated_map,
            "dispersion_dominated": self.dispersion_dominated_map,
            "input": self._load_input,
        }

        try:
            builder = map_builders[self.map_type]
        except KeyError:
            raise ValueError(f"Invalid map type: {self.map_type}")

        self.map = builder(**self.map_kwargs)

    def _load_input(self):
        """Load user-provided map and mask into the model."""
        map = self.input_map
        mask = self.input_mask

        self.map = map
        self.mask = mask
        self.size = map.shape[0]

        return self.map

    def _grid_r_theta(self):
        """Return coordinate grids in map-centred Cartesian and polar form."""
        ny = nx = self.size
        y, x = np.indices((ny, nx))
        x0, y0 = self.center
        x_rel = (x - x0) * self.pixel_scale
        y_rel = (y - y0) * self.pixel_scale
        R = np.hypot(x_rel, y_rel)
        theta = np.arctan2(y_rel, x_rel)  # radians
        return x_rel, y_rel, R, theta

    def rotate_map(self, angle, set_edges_to_nan: bool = True):
        """Rotate model map by angle and reapply edge/mask logic."""

        # Replace NaNs only in the temporary array for interpolation
        map_to_rotate = self.map.copy()
        nan_mask = np.isnan(map_to_rotate)
        map_to_rotate[nan_mask] = 0.0

        # Rotate map with original shape
        map_off = rotate(input=map_to_rotate, angle=angle, reshape=False)

        if self.map_type in ["rotation_dominated", "dispersion_dominated"]:
            out = cookie_cutter(map_off, self.size, set_edges_to_nan=set_edges_to_nan)
        elif self.map_type == "input":
            mask_to_rotate = self.mask.astype(int)
            mask_off = rotate(input=mask_to_rotate, angle=angle, order=0, reshape=False)
            keep = (~self.mask & ~mask_off).astype(bool)
            out = map_off.copy()
            out[~keep] = np.nan

        return out

    def rotation_dominated_map(self, v_max=200., r_turn=5., incl=60., pa=0., v_sys=0., normalise=True, return_meta=False):
        """
        Build a toy axisymmetric rotation-dominated LOS velocity map.
        """

        # Convert to radians
        pa = np.deg2rad(pa)
        cosi = np.cos(np.deg2rad(incl))
        sini = np.sin(np.deg2rad(incl))

        x_rel, y_rel, R, theta = self._grid_r_theta()

        # Rotate by PA
        x_r = x_rel * np.cos(pa) + y_rel * np.sin(pa)
        y_r = -x_rel * np.sin(pa) + y_rel * np.cos(pa)

        # Deproject minor axis to get intrinsic radius in disc plane
        x_deproj = x_r
        y_deproj = y_r / cosi
        R = np.hypot(x_deproj, y_deproj)
        phi = np.arctan2(y_deproj, x_deproj)

        # Create toy intrinsic rotation curve
        v_rot = v_max * (1.0 - np.exp(-R / r_turn))

        # Project into LOS
        v_los = v_sys + v_rot * sini * np.cos(phi)

        # Normalise map after applying cookie cutter
        v_los_cut = cookie_cutter(v_los, self.size, set_edges_to_nan=True)
        if normalise:
            v_los_cut = minmax_normalise_velocity_map(v_los_cut)

        # Return map and, optionally, metadata
        meta = dict(v_max=v_max, r_turn=r_turn, incl_deg=incl, pa_deg=pa, v_sys=v_sys)
        if return_meta:
            return v_los_cut, meta
        else:
            return v_los_cut

    def dispersion_dominated_map(self, sigma0=80.0, normalise=True):
        """Build a toy dispersion-dominated velocity map."""
        # Zero mean, Gaussian scatter everywhere
        v_map = self.rng.normal(loc=0.0, scale=sigma0, size=(self.size, self.size))

        # Normalise map after applying cookie cutter
        v_map_cut = cookie_cutter(v_map, self.size, set_edges_to_nan=True)
        if normalise:
            v_map_cut = minmax_normalise_velocity_map(v_map_cut)

        return v_map_cut

    def apply_asymmetric_drift(self, v_circ_map, sigma_R_map, surface_density_map,
                               kappa_factor=1.0):
        """
        Apply a simplified asymmetric-drift correction to a velocity field.
        """
        # compute radial derivative in log-space
        eps = 1e-8
        _, _, R, _ = self._grid_r_theta()

        # avoid divide by zero at center
        log_term = np.log(surface_density_map * (sigma_R_map**2) + eps)
        # radial derivative: d ln(...) / d ln R = R * d/dR log_term
        # compute d/dR via central differences in radial bins
        # first compute radial profile by binning (simple approach)
        r_flat = R.ravel()
        bins = np.unique(np.round(r_flat)).astype(int)
        # compute radial derivative profile
        bin_centers = bins + 0.5
        profile = np.zeros_like(bin_centers, dtype=float)
        counts = np.zeros_like(bin_centers, dtype=int)
        for i, b in enumerate(bins):
            mask = (np.floor(R) == b)
            if np.any(mask):
                profile[i] = np.nanmean(log_term[mask])
                counts[i] = np.count_nonzero(mask)
            else:
                profile[i] = np.nan

        # finite-diff d profile/d ln r
        # avoid r=0 bin
        dln = np.zeros_like(profile)
        for i in range(1, len(profile)-1):
            if np.isfinite(profile[i-1]) and np.isfinite(profile[i+1]):
                dln[i] = (profile[i+1] - profile[i-1]) / (np.log(bin_centers[i+1]) - np.log(bin_centers[i-1]))
        # map radial derivative back to map
        dln_map = np.zeros_like(R)
        for i, b in enumerate(bins):
            mask = (np.floor(R) == b)
            dln_map[mask] = dln[i]

        # adopt sigma_phi^2 / sigma_R^2 ~ 0.5 * kappa_factor (user can tune)
        sigma_phi2_over_sigma_R2 = 0.5 * kappa_factor

        v_circ2 = v_circ_map**2
        correction = sigma_R_map**2 * (dln_map + 1.0 - sigma_phi2_over_sigma_R2)
        v_phi2 = np.maximum(0.0, v_circ2 - correction)
        v_phi = np.sqrt(v_phi2)

        # projection to LOS depends on geometry; if v_circ_map was LOS-projected
        # we must de-project and re-project; for this simple demo assume v_circ_map is intrinsic v_phi already.
        # Return v_phi (intrinsic streaming). Caller should project it via same geometry as rotation_map.
        return v_phi
