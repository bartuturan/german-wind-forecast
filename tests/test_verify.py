import numpy as np
import pandas as pd
import pytest

from windfc.verify import (crps_ensemble, is_day_ahead, randomized_pit, rank_histogram,
                           score_table, verify)


def test_crps_single_member_is_absolute_error():
    assert crps_ensemble(np.array([[3.0]]), np.array([1.0]))[0] == pytest.approx(2.0)


def test_crps_two_members():
    # members {0, 2}, obs 1: E|X-y| = 1, E|X-X'| = (0+2+2+0)/4 = 1 -> CRPS = 0.5
    assert crps_ensemble(np.array([[0.0, 2.0]]), np.array([1.0]))[0] == pytest.approx(0.5)


def test_crps_perfect_forecast_is_zero():
    assert crps_ensemble(np.full((1, 10), 5.0), np.array([5.0]))[0] == pytest.approx(0.0)


def _members(times, n=11):
    rows = [{"valid_time": t, "step": 3 * i, "member": m, "total_mw": 1000.0 + 100 * m}
            for i, t in enumerate(times) for m in range(n)]
    return pd.DataFrame(rows)  # median 1500, P10 1100, P90 1900


def test_verify_scores():
    times = pd.date_range("2026-10-01", periods=4, freq="3h", tz="UTC")
    actual = pd.Series([1500.0, 1600.0, 2500.0, np.nan], index=times)
    res = verify(_members(times), actual)
    o = res["overall"]
    assert o["n"] == 3
    assert o["mae_mw"] == pytest.approx((0 + 100 + 1000) / 3)
    assert o["coverage_p10_p90"] == pytest.approx(2 / 3)
    assert res["by_lead"][0]["lead_h"] == "0-24"
    assert sum(res["rank_histogram"]["counts"]) == 3


def test_persistence_is_the_step0_actual():
    times = pd.date_range("2026-10-01", periods=4, freq="3h", tz="UTC")
    actual = pd.Series([1200.0, 1600.0, 2500.0, 1000.0], index=times)
    table = score_table(_members(times), actual)
    assert (table["persistence"] == 1200.0).all()
    o = verify(_members(times), actual)["overall"]
    assert o["mae_persistence_mw"] == pytest.approx((0 + 400 + 1300 + 200) / 4)
    assert o["crpss_vs_persistence"] == pytest.approx(1 - table["crps"].mean() / o["mae_persistence_mw"])


def test_pit_extremes_and_uniformity():
    ens = np.tile(np.arange(10.0), (3, 1))
    pit = randomized_pit(ens, np.array([-1.0, 4.5, 99.0]))
    assert 0 <= pit[0] < 1 / 11
    assert 5 / 11 <= pit[1] < 6 / 11
    assert 10 / 11 <= pit[2] < 1
    # obs drawn from the same distribution as the members -> flat histogram
    rng = np.random.default_rng(1)
    pit = randomized_pit(rng.normal(size=(20000, 50)), rng.normal(size=20000))
    counts = rank_histogram(pd.Series(pit))["counts"]
    assert max(counts) / min(counts) < 1.1


def test_pit_ties_are_spread():
    # every member equals the obs (saturated fleet): ranks must spread over 0..m, not pile up
    pit = randomized_pit(np.full((5000, 9), 3.0), np.full(5000, 3.0))
    assert np.histogram(pit, bins=10, range=(0, 1))[0].min() > 350


def test_day_ahead_window_cest():
    # 00 UTC run on 1 Oct (CEST, UTC+2): delivery day 2 Oct = 1 Oct 22:00 .. 2 Oct 22:00 UTC
    base = pd.Timestamp("2026-10-01T00:00", tz="UTC")
    valid = base + pd.to_timedelta(np.arange(0, 49, 3), unit="h")
    flag = is_day_ahead(pd.Series([base] * len(valid)), pd.Series(valid))
    assert list(np.arange(0, 49, 3)[flag.to_numpy()]) == [24, 27, 30, 33, 36, 39, 42, 45]


def test_day_ahead_excludes_12utc_runs():
    base = pd.Timestamp("2026-10-01T12:00", tz="UTC")
    valid = base + pd.to_timedelta(np.arange(0, 49, 3), unit="h")
    assert not is_day_ahead(pd.Series([base] * len(valid)), pd.Series(valid)).any()
