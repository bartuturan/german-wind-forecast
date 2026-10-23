"""Simple statistical calibration: one multiplicative scale per fleet (onshore, offshore).

Fitted by least squares of the ensemble mean against actuals: scale = sum(f*a) / sum(f^2).
It absorbs systematic errors in the capacity split, power curve and availability, but does
not fix the spread. EMOS or quantile regression would be the next step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FLEETS = ("onshore_mw", "offshore_mw")


def fit_scales(pairs: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, float]:
    """pairs: (member forecast, actuals aligned to its valid times) per run."""
    scales = {}
    for col in FLEETS:
        num = den = 0.0
        for members, actual in pairs:
            mean = members.groupby("valid_time")[col].mean()
            a = actual[col].reindex(mean.index)
            ok = a.notna()
            num += float((mean[ok] * a[ok]).sum())
            den += float((mean[ok] ** 2).sum())
        scales[col] = num / den if den > 0 else 1.0
    return scales


def apply_scales(members: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    out = members.copy()
    for col in FLEETS:
        out[col] = out[col] * scales.get(col, 1.0)
    out["total_mw"] = out["onshore_mw"] + out["offshore_mw"]
    return out


def save_scales(path: Path, scales: dict[str, float]) -> None:
    path.write_text(json.dumps(scales, indent=2))


def load_scales(path: Path) -> dict[str, float] | None:
    return json.loads(path.read_text()) if path.exists() else None
