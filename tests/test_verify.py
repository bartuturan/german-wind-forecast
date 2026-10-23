import numpy as np
import pandas as pd
import pytest

from windfc.verify import crps_ensemble, verify


def test_crps_single_member_is_absolute_error():
    assert crps_ensemble(np.array([[3.0]]), np.array([1.0]))[0] == pytest.approx(2.0)


def test_crps_two_members():
    # members {0, 2}, obs 1: E|X-y| = 1, E|X-X'| = (0+2+2+0)/4 = 1 -> CRPS = 0.5
    assert crps_ensemble(np.array([[0.0, 2.0]]), np.array([1.0]))[0] == pytest.approx(0.5)


def test_crps_perfect_forecast_is_zero():
    assert crps_ensemble(np.full((1, 10), 5.0), np.array([5.0]))[0] == pytest.approx(0.0)


def test_verify_scores():
    times = pd.date_range("2026-10-01", periods=4, freq="3h", tz="UTC")
    rows = [{"valid_time": t, "step": 3 * i, "member": m, "total_mw": 1000.0 + 100 * m}
            for i, t in enumerate(times) for m in range(11)]
    members = pd.DataFrame(rows)  # median 1500, P10 1100, P90 1900
    actual = pd.Series([1500.0, 1600.0, 2500.0, np.nan], index=times)
    res = verify(members, actual)
    o = res["overall"]
    assert o["n"] == 3
    assert o["mae_mw"] == pytest.approx((0 + 100 + 1000) / 3)
    assert o["coverage_p10_p90"] == pytest.approx(2 / 3)
    assert res["by_lead"][0]["lead_h"] == "0-24"
