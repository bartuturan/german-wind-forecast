import numpy as np

from windfc.config import PowerCurve
from windfc.power import capacity_factor, to_hub_height, wind_speed

CURVE = PowerCurve(speeds=(0, 3, 12, 25), cf=(0, 0, 1, 1), cut_out=25.0, hub_height=100)


def test_wind_speed():
    assert wind_speed(np.array(3.0), np.array(4.0)) == 5.0


def test_below_cut_in_is_zero():
    assert capacity_factor(np.array([0.0, 2.9, 3.0]), CURVE).tolist() == [0, 0, 0]


def test_linear_between_points_and_rated():
    cf = capacity_factor(np.array([7.5, 12.0, 20.0]), CURVE)
    np.testing.assert_allclose(cf, [0.5, 1.0, 1.0])


def test_above_cut_out_is_zero():
    assert capacity_factor(np.array([25.0, 25.1, 40.0]), CURVE).tolist() == [1.0, 0.0, 0.0]


def test_nan_stays_nan():
    assert np.isnan(capacity_factor(np.array([np.nan]), CURVE)[0])


def test_shear():
    np.testing.assert_allclose(to_hub_height(np.array(10.0), 100, 0.14), 10.0)
    assert to_hub_height(np.array(10.0), 120, 0.14) > 10.0
