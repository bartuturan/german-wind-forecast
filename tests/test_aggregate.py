from datetime import datetime, timezone

import numpy as np
import pytest

from windfc.aggregate import national_members, region_weights, summarize
from windfc.config import load_config

CFG = load_config()


def _grid():
    lat = np.arange(CFG.bbox.lat_min, CFG.bbox.lat_max + 0.01, 0.25)
    lon = np.arange(CFG.bbox.lon_min, CFG.bbox.lon_max + 0.01, 0.25)
    return lat, lon


@pytest.mark.parametrize("region", CFG.regions, ids=lambda r: r.name)
def test_weights_normalised(region):
    lat, lon = _grid()
    w = region_weights(region, lat, lon)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_mask_restricts_to_land_for_onshore():
    lat, lon = _grid()
    lsm = np.zeros((len(lat), len(lon)))
    lsm[:, lon > 9] = 1.0  # pretend everything east of 9E is land
    north = next(r for r in CFG.regions if r.name == "north")
    w = region_weights(north, lat, lon, lsm)
    assert w[:, lon <= 9].sum() == 0


def test_rated_wind_gives_installed_capacity():
    lat, lon = _grid()
    shape = (3, 2, len(lat), len(lon))
    cube = {"u100": np.full(shape, 15.0, np.float32), "v100": np.zeros(shape, np.float32),
            "members": np.arange(3), "steps": np.array([0, 3]), "lat": lat, "lon": lon,
            "base_time": datetime(2026, 10, 1, tzinfo=timezone.utc)}
    df = national_members(cube, CFG)
    expected = sum(r.capacity_mw * CFG.availability[r.type] for r in CFG.regions)
    np.testing.assert_allclose(df["total_mw"], expected)
    assert len(df) == 6

    s = summarize(df)
    assert list(s.columns) == ["valid_time", "step", "mean", "p10", "p25", "p50", "p75", "p90"]
    assert len(s) == 2
