from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .preprocessing import minmax_normalise_velocity_map

class MapModel:
    """
    Minimal MapModel with physical parameters and asymmetric-drift hook.
    Returns 2D numpy arrays (velocity maps) and optional metadata dict.
    """
    
    def __init__(self, size: int, pixel_scale: float = 1.0,
                center: Optional[Tuple[int, int]] = None,
                seed: Optional[int] = None):
        self.size = size
        self.pixel_scale = pixel_scale
        self.center = center if center is not None else (size//2, size//2)
        self.rng = np.random.default_rng(seed)

    def _grid_r_theta(self):
        ny = nx = self.size
        y, x = np.indices((ny, nx))
        x0, y0 = self.center
        x_rel = (x - x0) * self.pixel_scale
        y_rel = (y - y0) * self.pixel_scale
        R = np.hypot(x_rel, y_rel)
        theta = np.arctan2(y_rel, x_rel)  # radians
        return x_rel, y_rel, R, theta

    def cookie_cutter(self, input_array: np.ndarray, set_edges_to_nan: bool):
        """
        Cut the corners of an NxN square to make an 'octagon-like' mask,
        then apply it to input_array. Works for NaNs too (they are preserved).
        The mask is created by four linear inequalities (diagonal cuts).
        """

        size = self.size

        if input_array.shape != (size, size):
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
        result = input_array.copy()
        if set_edges_to_nan:
            result[~edge_mask] = np.nan
        else:
            result[~edge_mask] = 0.0

        return result

    def rotation_map(self, v_max=200., r_turn=5., incl_deg=60., pa_deg=0., v_sys=0., normalise=True, return_meta=False):
        """
        Build a simple axisymmetric rotation map (projected LOS velocity).
        v_rot(R) = v_max * (1 - exp(-R / r_turn))
        """
        incl = np.deg2rad(incl_deg)
        x_rel, y_rel, R, theta = self._grid_r_theta()

        # intrinsic rotation curve
        v_rot = v_max * (1.0 - np.exp(-R / r_turn))  # simple form

        # Project into LOS: v_los = v_rot * sin(i) * cos(phi)
        # phi measured from major axis; if PA != 0 you'd rotate coords by PA
        pa = np.deg2rad(pa_deg)
        # rotate coords by pa
        x_r = x_rel * np.cos(pa) + y_rel * np.sin(pa)
        y_r = -x_rel * np.sin(pa) + y_rel * np.cos(pa)
        R_r = np.hypot(x_r, y_r)
        phi = np.arctan2(y_r, x_r)

        # recompute v_rot with rotated radius if desired:
        v_rot = v_max * (1.0 - np.exp(-R_r / r_turn))

        # Convert to v_los
        v_los = v_rot * np.cos(phi) + v_sys

        # Normalise map after applying cookie cutter
        out = self.cookie_cutter(v_los, set_edges_to_nan=True)
        if normalise:
            out = minmax_normalise_velocity_map(out)

        # return map + metadata
        meta = dict(v_max=v_max, r_turn=r_turn, incl_deg=incl_deg, pa_deg=pa_deg, v_sys=v_sys)
        if return_meta:
            return out, meta
        else:
            return out

    def dispersion_map(self, sigma0=80.0, normalise=True):
        """Return a toy dispersion-dominated gas velocity map."""
        # Zero mean, Gaussian scatter everywhere
        v_map = self.rng.normal(loc=0.0, scale=sigma0, size=(self.size, self.size))

        # Normalise map after applying cookie cutter
        out = self.cookie_cutter(v_map, set_edges_to_nan=True)
        if normalise:
            out = minmax_normalise_velocity_map(out)

        return out

    def apply_asymmetric_drift(self, v_circ_map, sigma_R_map, surface_density_map,
                               kappa_factor=1.0):
        """
        Apply a simple asymmetric drift correction to a circular velocity map.
        This is a simplified version of the asymmetric drift eqn:
          v_phi^2 = v_c^2 - sigma_R^2 * ( d ln(n sigma_R^2) / d ln R + 1 - sigma_phi^2/sigma_R^2 )
        For simplicity we:
         - approximate d ln(n sigma_R^2) / d ln R with finite differences
         - assume sigma_phi^2/sigma_R^2 ~ 0.5 (epicyclic approx) or let user pass kappa_factor
        Returns corrected stellar azimuthal velocity map v_phi (projected LOS unchanged).
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

    def rotate(self, array, angle):

        # Set edge NaNs to zero for rotation
        nan_mask = np.isnan(array)
        array[nan_mask] = 0.0
        
        # Rotate map keeping original shape
        rotated = ndimage.rotate(input=array, angle=angle, reshape=False)
        
        # Apply cookie cutter and mask set edges back to NaN
        out = self.cookie_cutter(rotated, set_edges_to_nan=True)
        return out