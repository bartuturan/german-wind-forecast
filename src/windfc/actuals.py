"""Actual German wind generation from Energy-Charts (Fraunhofer ISE), with SMARD as fallback."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

ENERGY_CHARTS_URL = "https://api.energy-charts.info/public_power"
SMARD_URL = "https://www.smard.de/app/chart_data"
SMARD_FILTERS = {"onshore_mw": 4067, "offshore_mw": 1225}
TIMEOUT = 60


def fetch_energy_charts(start: datetime, end: datetime) -> pd.DataFrame:
    """15-min onshore/offshore wind in MW, UTC index."""
    params = {"country": "de", "start": start.strftime("%Y-%m-%dT%H:%MZ"),
              "end": end.strftime("%Y-%m-%dT%H:%MZ")}
    r = requests.get(ENERGY_CHARTS_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    index = pd.to_datetime(js["unix_seconds"], unit="s", utc=True)
    series = {p["name"]: p["data"] for p in js["production_types"]}
    df = pd.DataFrame({
        "onshore_mw": series["Wind onshore"],
        "offshore_mw": series["Wind offshore"],
    }, index=index, dtype=float)
    df.index.name = "time"
    return df


def fetch_smard(start: datetime, end: datetime) -> pd.DataFrame:
    """Quarter-hourly onshore/offshore wind in MW from SMARD's chart_data JSON files (UTC)."""
    cols = {}
    for col, flt in SMARD_FILTERS.items():
        idx = requests.get(f"{SMARD_URL}/{flt}/DE/index_quarterhour.json", timeout=TIMEOUT)
        idx.raise_for_status()
        # each timestamp (ms) starts a weekly file; keep the ones overlapping [start, end]
        t0 = (start - timedelta(days=7)).timestamp() * 1000
        stamps = [t for t in idx.json()["timestamps"] if t0 <= t <= end.timestamp() * 1000]
        rows = []
        for ts in stamps:
            r = requests.get(f"{SMARD_URL}/{flt}/DE/{flt}_DE_quarterhour_{ts}.json", timeout=TIMEOUT)
            r.raise_for_status()
            rows += r.json()["series"]
        s = pd.Series({t: v for t, v in rows if v is not None}, dtype=float)
        # SMARD reports MWh per quarter hour; x4 gives mean MW
        s.index = pd.to_datetime(s.index, unit="ms", utc=True)
        cols[col] = s * 4
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "time"
    return df.loc[start:end]


def fetch_actuals(start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = fetch_energy_charts(start, end)
    except Exception as exc:
        log.warning("Energy-Charts failed (%s); falling back to SMARD", exc)
        df = fetch_smard(start, end)
    df["total_mw"] = df["onshore_mw"] + df["offshore_mw"]
    return df


def align_to(actuals: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly mean centred on each forecast valid time (the forecast is an instantaneous wind)."""
    smooth = actuals.sort_index().rolling("60min", center=True).mean()
    out = smooth.reindex(times, method="nearest", tolerance=pd.Timedelta("15min"))
    out.index.name = "valid_time"
    return out
