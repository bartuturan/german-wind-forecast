"""FastAPI service over the cached forecasts and actuals.

The API only reads the data/ cache; downloading and processing happen in scripts/run_pipeline.py.
Run: uvicorn windfc.api:app --reload
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from .actuals import align_to
from .aggregate import summarize
from .calibrate import apply_scales, load_scales
from .chart import fan_chart
from .config import DEFAULT_CONFIG, load_config
from .store import Store
from .verify import verify

cfg = load_config(os.environ.get("WINDFC_CONFIG", DEFAULT_CONFIG))
store = Store(cfg.data_dir)

app = FastAPI(title="German wind power forecast", version="0.1.0")


def _resolve(run: str) -> str:
    runs = store.runs()
    if not runs:
        raise HTTPException(404, "no forecast runs cached; run scripts/run_pipeline.py first")
    if run == "latest":
        return runs[-1]
    if run not in runs:
        raise HTTPException(404, f"run {run} not found; available: {runs}")
    return run


def _members(run: str, calibrated: bool) -> tuple[pd.DataFrame, str | None]:
    """Member forecast of a run and the calibration scope ("out-of-sample", "all-runs" or None)."""
    members = store.load_forecast(run)
    if not calibrated:
        return members, None
    loaded = load_scales(store.calibration_path(), run)
    if loaded is None:
        raise HTTPException(404, "no calibration; run scripts/calibrate.py first")
    scales, scope = loaded
    return apply_scales(members, scales), scope


def _utc(t: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df.round(1).to_dict(orient="records")


@app.get("/", response_class=HTMLResponse)
def index():
    runs = store.runs()
    latest = runs[-1] if runs else None
    img = f'<img src="/chart?run={latest}" style="max-width:100%">' if latest else "<p>No runs cached.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Wind forecast</title></head>
<body style="font-family:sans-serif;margin:24px">
<h2>German wind power: ECMWF ENS forecast vs actual</h2>
<p>Runs cached: {', '.join(runs) or 'none'} · <a href="/docs">API docs</a></p>{img}</body></html>"""


@app.get("/runs")
def runs():
    return {"runs": store.runs(), "installed_mw": cfg.installed_mw(),
            "calibrated_available": store.calibration_path().exists()}


@app.get("/forecast")
def forecast(run: str = "latest", fleet: str = Query("total", pattern="^(total|onshore|offshore)$"),
             calibrated: bool = False):
    run = _resolve(run)
    members, scope = _members(run, calibrated)
    summary = summarize(members, f"{fleet}_mw")
    return {"run": run, "fleet": fleet, "unit": "MW", "calibrated": calibrated,
            "calibration_scope": scope, "data": _records(summary)}


@app.get("/actuals")
def actuals(start: datetime, end: datetime):
    df = store.load_actuals()
    if df.empty:
        raise HTTPException(404, "no actuals cached")
    sub = df.loc[_utc(start):_utc(end)].reset_index()
    return {"unit": "MW", "source": "Energy-Charts / SMARD", "data": _records(sub)}


@app.get("/verify")
def verify_run(run: str = "latest", fleet: str = Query("total", pattern="^(total|onshore|offshore)$"),
               calibrated: bool = False):
    run = _resolve(run)
    members, scope = _members(run, calibrated)
    times = pd.DatetimeIndex(members["valid_time"].unique())
    aligned = align_to(store.load_actuals(), times)
    return {"run": run, "fleet": fleet, "calibrated": calibrated, "calibration_scope": scope,
            **verify(members, aligned[f"{fleet}_mw"], f"{fleet}_mw")}


@app.get("/chart")
def chart(run: str = "latest", fleet: str = Query("total", pattern="^(total|onshore|offshore)$"),
          calibrated: bool = False):
    run = _resolve(run)
    members, scope = _members(run, calibrated)
    act = store.load_actuals()
    actual = act[f"{fleet}_mw"] if not act.empty else None
    label = {"total": "onshore + offshore", "onshore": "onshore", "offshore": "offshore"}[fleet]
    cal = f" calibrated ({scope})" if calibrated else ""
    title = f"ECMWF IFS ENS {run}Z, German wind ({label}){cal}"
    return Response(fan_chart(members, actual, title, f"{fleet}_mw"), media_type="image/png")
