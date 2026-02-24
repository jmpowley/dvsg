import numpy as np
import pytest

from dvsg.calculations import (
    _validate_error_type,
    calculate_dvsg,
    calculate_dvsg_diagnostics,
)
from dvsg.preprocessing import minmax_normalise_velocity_map


def test_calculate_dvsg_zero_for_identical_maps():
    arr = np.array([[0.0, 1.0], [-1.0, np.nan]])
    dvsg = calculate_dvsg(arr, arr.copy())
    assert dvsg == pytest.approx(0.0)


def test_calculate_dvsg_matches_manual_mean_abs_residual():
    sv = np.array([[0.0, 1.0], [2.0, np.nan]])
    gv = np.array([[1.0, 1.0], [0.0, np.nan]])

    dvsg = calculate_dvsg(sv, gv)
    # residual = [[1, 0], [2, nan]] -> mean over finite values = 1.0
    assert dvsg == pytest.approx(1.0)


def test_calculate_dvsg_raises_on_shape_mismatch():
    sv = np.zeros((2, 2))
    gv = np.zeros((2, 3))

    with pytest.raises(Exception, match="must have the same shape"):
        calculate_dvsg(sv, gv)


def test_calculate_dvsg_diagnostics_returns_expected_keys():
    sv = np.array([[0.0, 1.0], [2.0, np.nan]])
    gv = np.array([[1.0, 1.0], [0.0, np.nan]])

    out = calculate_dvsg_diagnostics(sv, gv)

    assert set(out.keys()) == {"dvsg", "dvsg_stderr", "residual"}
    assert out["dvsg"] == pytest.approx(1.0)
    assert out["residual"].shape == sv.shape


def test_calculate_dvsg_diagnostics_stderr_counts_finite_points_not_nonzero():
    sv = np.array([0.0, 0.0, 2.0, np.nan])
    gv = np.array([0.0, 0.0, 0.0, np.nan])

    out = calculate_dvsg_diagnostics(sv, gv)
    residual = np.array([0.0, 0.0, 2.0, np.nan])
    n_valid = np.count_nonzero(np.isfinite(residual))
    expected = np.nanstd(residual) / np.sqrt(n_valid)

    assert n_valid == 3
    assert out["dvsg_stderr"] == pytest.approx(expected)


def test_calculate_dvsg_diagnostics_stderr_nan_when_no_finite_bins():
    sv = np.array([np.nan, np.nan])
    gv = np.array([np.nan, np.nan])

    out = calculate_dvsg_diagnostics(sv, gv)

    assert np.isnan(out["dvsg_stderr"])


def test_error_type_validation_accepts_only_stderr_and_analytic():
    assert _validate_error_type(None) is None
    assert _validate_error_type("stderr") == "stderr"
    assert _validate_error_type("analytic") == "analytic"

    with pytest.raises(ValueError, match="Allowed values"):
        _validate_error_type("bad-value")


def test_regression_toy_pipeline_minmax_then_dvsg():
    sv = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, np.nan]])
    gv = np.array([[-2.0, 0.0, 0.0], [2.0, 2.0, np.nan]])

    sv_norm = minmax_normalise_velocity_map(sv)
    gv_norm = minmax_normalise_velocity_map(gv)

    dvsg = calculate_dvsg(sv_norm, gv_norm)

    # residual finite values: [0.0, 0.5, 0.0, 0.5, 0.0] -> mean = 0.2
    assert dvsg == pytest.approx(0.2)
