"""Simple statistical calibration: one multiplicative scale per fleet (onshore, offshore).

Fitted by least squares of the ensemble mean against actuals: scale = sum(f*a) / sum(f^2).
It absorbs systematic errors in the capacity split, power curve and availability, but does
not fix the spread. EMOS or quantile regression would be the next step.

calibration.json holds the scales fitted on every run (for new runs) and, under "by_run",
the rolling-origin scales each past run would have had at its base time (out of sample).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FLEETS = ("onshore_mw", "offshore_mw")


def fit_scales(pairs: list[tuple[pd.DataFrame, pd.DataFrame]],
               before: pd.Timestamp | None = None) -> dict[str, float]:
    """pairs: (member forecast, actuals aligned to its valid times) per run.

    With before, only valid times earlier than it are used: the actuals known at that moment.
    """
    scales = {}
    for col in FLEETS:
        num = den = 0.0
        for members, actual in pairs:
            mean = members.groupby("valid_time")[col].mean()
            a = actual[col].reindex(mean.index)
            ok = a.notna()
            if before is not None:
                ok &= mean.index < before
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


def save_scales(path: Path, scales: dict[str, float],
                by_run: dict[str, dict[str, float]] | None = None) -> None:
    path.write_text(json.dumps({**scales, "by_run": by_run or {}}, indent=2))


def load_scales(path: Path, run: str | None = None) -> tuple[dict[str, float], str] | None:
    """(scales, scope) for a run: its rolling-origin scales if it has them ("out-of-sample"),
    else the scales fitted on all runs ("all-runs", in sample for any run they were fitted on)."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    by_run = raw.pop("by_run", {})
    if run in by_run:
        return by_run[run], "out-of-sample"
    return raw, "all-runs"
