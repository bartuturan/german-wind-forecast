import pandas as pd
import pytest

from windfc.calibrate import fit_scales, load_scales, save_scales


def _pair(times, fc, act):
    members = pd.DataFrame({"valid_time": times, "step": 0, "member": 1,
                            "onshore_mw": fc, "offshore_mw": fc})
    actual = pd.DataFrame({"onshore_mw": act, "offshore_mw": act}, index=times)
    return members, actual


def test_scale_is_least_squares_ratio():
    times = pd.date_range("2026-10-01", periods=2, freq="h", tz="UTC")
    scales = fit_scales([_pair(times, [100.0, 200.0], [80.0, 160.0])])
    assert scales["onshore_mw"] == pytest.approx(0.8)


def test_before_ignores_actuals_not_yet_known():
    times = pd.date_range("2026-10-01", periods=2, freq="h", tz="UTC")
    pair = _pair(times, [100.0, 100.0], [50.0, 150.0])
    assert fit_scales([pair], before=times[1])["onshore_mw"] == pytest.approx(0.5)
    assert fit_scales([pair])["onshore_mw"] == pytest.approx(1.0)


def test_load_prefers_the_runs_own_out_of_sample_scales(tmp_path):
    path = tmp_path / "calibration.json"
    save_scales(path, {"onshore_mw": 0.9, "offshore_mw": 0.8},
                {"2026-10-10T00": {"onshore_mw": 0.7, "offshore_mw": 0.6}})
    assert load_scales(path, "2026-10-10T00") == ({"onshore_mw": 0.7, "offshore_mw": 0.6}, "out-of-sample")
    assert load_scales(path, "2026-10-11T00") == ({"onshore_mw": 0.9, "offshore_mw": 0.8}, "all-runs")
    assert load_scales(tmp_path / "missing.json") is None
