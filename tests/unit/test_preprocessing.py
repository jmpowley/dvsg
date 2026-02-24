import numpy as np
import pytest

from dvsg.preprocessing import (
    apply_bin_snr_threshold,
    apply_sigma_clip,
    exclude_above_n_sigma,
    mad5_normalise_velocity_map,
    minmax_normalise_velocity_map,
    normalise_map,
    zscore1_normalise_velocity_map,
)


def test_minmax_normalise_maps_to_minus1_plus1():
    arr = np.array([[-3.0, -1.0, 1.0, 3.0]])
    out = minmax_normalise_velocity_map(arr)
    expected = np.array([[-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]])

    assert np.allclose(out, expected)
    assert np.nanmin(out) == pytest.approx(-1.0)
    assert np.nanmax(out) == pytest.approx(1.0)


def test_minmax_normalise_returns_nan_for_constant_map():
    arr = np.full((2, 2), 5.0)
    out = minmax_normalise_velocity_map(arr)

    assert np.isnan(out).all()


def test_zscore1_normalise_zero_mean_unit_std_for_simple_case():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = zscore1_normalise_velocity_map(arr)

    assert np.nanmean(out) == pytest.approx(0.0, abs=1e-12)
    assert np.nanstd(out, ddof=1) == pytest.approx(1.0, abs=1e-12)


def test_mad5_normalise_returns_nan_for_zero_mad():
    arr = np.array([2.0, 2.0, 2.0, 2.0])
    out = mad5_normalise_velocity_map(arr)

    assert np.isnan(out).all()


def test_exclude_above_n_sigma_removes_outlier():
    arr = np.array([0.0, 0.0, 0.0, 100.0])
    out = exclude_above_n_sigma(arr, n=1)

    assert np.isnan(out[-1])
    assert np.allclose(out[:-1], [0.0, 0.0, 0.0], equal_nan=True)


def test_apply_bin_snr_threshold_masks_below_threshold():
    sv = np.array([1.0, 2.0, 3.0], dtype=float)
    gv = np.array([4.0, 5.0, 6.0], dtype=float)
    snr = np.array([5.0, 15.0, 8.0], dtype=float)

    sv_out, gv_out = apply_bin_snr_threshold(sv.copy(), gv.copy(), snr, snr_threshold=10.0)

    assert np.isnan(sv_out[[0, 2]]).all()
    assert np.isnan(gv_out[[0, 2]]).all()
    assert sv_out[1] == pytest.approx(2.0)
    assert gv_out[1] == pytest.approx(5.0)


def test_apply_sigma_clip_applies_to_both_maps_consistently():
    sv = np.array([0.0, 0.0, 0.0, 100.0])
    gv = np.array([0.0, 0.0, -100.0, 0.0])

    sv_out, gv_out = apply_sigma_clip(sv, gv, n_sigma=1)

    assert np.isnan(sv_out[-1])
    assert np.isnan(gv_out[2])


def test_normalise_map_rejects_unknown_method():
    sv = np.array([0.0, 1.0, 2.0])
    gv = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="norm_method"):
        normalise_map(sv, gv, norm_method="unknown")


def test_regression_sigma_clip_then_minmax_output_snapshot():
    sv = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 100.0])
    gv = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, -100.0])

    sv_clip, gv_clip = apply_sigma_clip(sv, gv, n_sigma=2)
    sv_norm, gv_norm = normalise_map(sv_clip, gv_clip, norm_method="minmax")

    expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, np.nan])

    assert np.allclose(sv_norm, expected, equal_nan=True)
    assert np.allclose(gv_norm, expected, equal_nan=True)
