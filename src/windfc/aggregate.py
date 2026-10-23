"""Capacity-weighted aggregation of gridded capacity factors to national MW."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .config import Config, Region
from .power import capacity_factor, to_hub_height, wind_speed

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def region_weights(region: Region, lat: np.ndarray, lon: np.ndarray,
                   lsm: np.ndarray | None = None) -> np.ndarray:
    """Normalised (sum 1) cos(lat) area weights over the region's cells.

    With a land-sea mask, onshore regions keep land cells (lsm >= 0.5) and offshore regions
    keep sea cells. If the mask leaves nothing, every cell in the box is used.
    """
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    inside = ((lat2 >= region.lat[0]) & (lat2 <= region.lat[1])
              & (lon2 >= region.lon[0]) & (lon2 <= region.lon[1]))
    if lsm is not None:
        surface = lsm >= 0.5 if region.type == "onshore" else lsm < 0.5
        if (inside & surface).any():
            inside &= surface
    if not inside.any():
        raise ValueError(f"region {region.name} contains no grid cells")
    w = np.where(inside, np.cos(np.deg2rad(lat2)), 0.0)
    return w / w.sum()


def national_members(cube: dict, cfg: Config, lsm: np.ndarray | None = None) -> pd.DataFrame:
    """MW per member and valid time, split onshore/offshore.

    Returns long format: valid_time, step, member, onshore_mw, offshore_mw, total_mw.
    """
    ws100 = wind_speed(cube["u100"], cube["v100"])  # (member, step, lat, lon)
    fleet_mw = {}
    for fleet in ("onshore", "offshore"):
        curve = cfg.curves[fleet]
        ws = to_hub_height(ws100, curve.hub_height, cfg.shear_alpha) if cfg.shear_enabled else ws100
        cf = capacity_factor(ws, curve)
        total = np.zeros(cf.shape[:2])
        for region in (r for r in cfg.regions if r.type == fleet):
            w = region_weights(region, cube["lat"], cube["lon"], lsm)
            region_cf = np.nansum(cf * w, axis=(-2, -1))  # (member, step)
            total += region_cf * region.capacity_mw
        fleet_mw[fleet] = total * cfg.availability.get(fleet, 1.0)

    members, steps = cube["members"], cube["steps"]
    m_idx, s_idx = np.meshgrid(np.arange(len(members)), np.arange(len(steps)), indexing="ij")
    df = pd.DataFrame({
        "valid_time": [cube["base_time"] + timedelta(hours=int(steps[s])) for s in s_idx.ravel()],
        "step": steps[s_idx.ravel()],
        "member": members[m_idx.ravel()],
        "onshore_mw": fleet_mw["onshore"].ravel(),
        "offshore_mw": fleet_mw["offshore"].ravel(),
    })
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["total_mw"] = df["onshore_mw"] + df["offshore_mw"]
    return df


def summarize(members: pd.DataFrame, column: str = "total_mw") -> pd.DataFrame:
    """Ensemble quantiles and mean per valid time: columns step, mean, p10, p25, p50, p75, p90."""
    g = members.groupby("valid_time")
    out = pd.DataFrame({"step": g["step"].first(), "mean": g[column].mean()})
    for q in QUANTILES:
        out[f"p{int(q * 100)}"] = g[column].quantile(q)
    return out.reset_index()
