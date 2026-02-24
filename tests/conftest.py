import sys
import types
from pathlib import Path

import numpy as np


# Ensure local src-layout package import works without editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Provide lightweight fallbacks for optional external dependencies used at import time.
try:
    import marvin  # noqa: F401
except ImportError:
    marvin = types.ModuleType("marvin")

    class _Config:
        @staticmethod
        def setDR(*args, **kwargs):
            return None

        @staticmethod
        def switchSasUrl(*args, **kwargs):
            return None

    marvin.config = _Config()

    marvin_tools = types.ModuleType("marvin.tools")

    class _Maps:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Mock marvin.tools.Maps: not available in unit-test fallback")

    marvin_tools.Maps = _Maps

    sys.modules["marvin"] = marvin
    sys.modules["marvin.tools"] = marvin_tools


try:
    from mangadap.util.fitsutil import DAPFitsUtil  # noqa: F401
except ImportError:
    mangadap = types.ModuleType("mangadap")
    mangadap_util = types.ModuleType("mangadap.util")
    mangadap_fitsutil = types.ModuleType("mangadap.util.fitsutil")

    class _DAPFitsUtil:
        @staticmethod
        def unique_bins(bins, return_index=False):
            bins = np.asarray(bins)
            uniq, idx = np.unique(bins, return_index=True)
            if return_index:
                return uniq, idx
            return uniq

        @staticmethod
        def reconstruct_map(map_shape, bins, flat):
            out = np.full(map_shape, np.nan, dtype=float)
            bins = np.asarray(bins).reshape(map_shape)
            values = np.asarray(flat)
            uniq = np.unique(bins)
            for i, b in enumerate(uniq):
                if i < len(values):
                    out[bins == b] = values[i]
            return out

    mangadap_fitsutil.DAPFitsUtil = _DAPFitsUtil

    sys.modules["mangadap"] = mangadap
    sys.modules["mangadap.util"] = mangadap_util
    sys.modules["mangadap.util.fitsutil"] = mangadap_fitsutil
