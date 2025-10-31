import numpy as np
from scipy import ndimage
# from calculate_dvsg_v3 import calculate_DVSG

class MapModel:
    """Models simaulted gas and velocity maps"""

    def __init__(self, size):
        """
        Initialize the models
        """
        
        # Set size of model
        self.size = size

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
        result[~edge_mask] = 0.0

        if set_edges_to_nan:
            result[~edge_mask] = np.nan

        return result
    
    def rotation_dominated(self, set_edges_to_nan: bool, rng=None):
        """
        Returns an NxN array with a smooth velocity gradient (rotation-dominated).
        Bands (by rows) have different uniform ranges. Vectorised implementation.
        """

        size = self.size

        if rng is None:
            rng = np.random.default_rng()

        out = np.empty((size, size), dtype=float)

        # row bands as fractions of size — matches your original 5 bands
        bands = [
            (0.0, 0.2, 1.00, 0.66),
            (0.2, 0.4, 0.66, 0.33),
            (0.4, 0.6, 0.33, -0.33),
            (0.6, 0.8, -0.33, -0.66),
            (0.8, 1.0, -0.66, -1.00),
        ]

        for start_frac, end_frac, low, high in bands:
            r0 = int(np.floor(start_frac * size))
            r1 = int(np.floor(end_frac * size))
            if r1 <= r0:
                continue
            rows = r1 - r0
            out[r0:r1, :] = rng.uniform(low=min(low, high), high=max(low, high), size=(rows, size))

        # apply cookie cutter
        out = self.cookie_cutter(out, set_edges_to_nan=set_edges_to_nan)
        return out
    
    def dispersion_dominated(self, set_edges_to_nan: bool, rng=None):
        """
        Returns an NxN array where the center has a narrow dispersion [-0.25,0.25]
        and the outer region is wider [-1,1]. Vectorised.
        """

        size = self.size

        if rng is None:
            rng = np.random.default_rng()

        i = np.arange(size)[:, None]
        j = np.arange(size)[None, :]
        r_centre = np.sqrt((i - size/2)**2 + (j - size/2)**2)
        central_mask = r_centre < (size / 4)

        out = np.empty((size, size), dtype=float)
        # Fill central region and outer region with vectorised random draws
        out[central_mask] = rng.uniform(-0.5, 0.5, size=np.count_nonzero(central_mask))
        out[~central_mask] = rng.uniform(-1.0, 1.0, size=np.count_nonzero(~central_mask))

        # Apply cookie cutter
        out = self.cookie_cutter(out, set_edges_to_nan=set_edges_to_nan)
        return out
    
    def rotate(self, array, angle):
        
        # Rotate map keeping original shape
        rotated = ndimage.rotate(input=array, angle=angle, reshape=False)
        
        # Apply cookie cutter
        out = self.cookie_cutter(rotated, set_edges_to_nan=True)
        return out