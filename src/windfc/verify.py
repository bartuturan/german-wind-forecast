"""Deterministic and probabilistic scores of the ensemble against actual generation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def crps_ensemble(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble per case: E|X - y| - 0.5 E|X - X'|.

    ens has shape (n_cases, n_members), obs has shape (n_cases,).
    """
    ens = np.asarray(ens, dtype=float)
    obs = np.asarray(obs, dtype=float)
    term1 = np.mean(np.abs(ens - obs[:, None]), axis=1)
    term2 = np.mean(np.abs(ens[:, :, None] - ens[:, None, :]), axis=(1, 2))
    return term1 - 0.5 * term2


def _scores(group: pd.DataFrame) -> pd.Series:
    err = group["p50"] - group["actual"]
    inside = (group["actual"] >= group["p10"]) & (group["actual"] <= group["p90"])
    return pd.Series({
        "n": len(group),
        "mae_mw": err.abs().mean(),
        "rmse_mw": np.sqrt((err ** 2).mean()),
        "bias_mw": err.mean(),
        "crps_mw": group["crps"].mean(),
        "coverage_p10_p90": inside.mean(),
    })


def score_table(members: pd.DataFrame, actual: pd.Series, column: str = "total_mw") -> pd.DataFrame:
    """One row per valid time with quantiles, actual and CRPS; times without actuals are dropped."""
    wide = members.pivot_table(index="valid_time", columns="member", values=column)
    steps = members.groupby("valid_time")["step"].first()
    actual = actual.reindex(wide.index)
    ok = actual.notna()
    wide, actual = wide[ok], actual[ok]
    q = np.quantile(wide.to_numpy(), [0.1, 0.5, 0.9], axis=1)
    return pd.DataFrame({
        "step": steps[ok],
        "p10": q[0], "p50": q[1], "p90": q[2],
        "actual": actual,
        "crps": crps_ensemble(wide.to_numpy(), actual.to_numpy()),
    }, index=wide.index)


def verify(members: pd.DataFrame, actual: pd.Series, column: str = "total_mw",
           lead_bins: tuple[int, ...] = (0, 24, 48, 72, 96, 120, 144, 240, 360)) -> dict:
    """Overall scores plus scores per lead-time bin (hours)."""
    table = score_table(members, actual, column)
    if table.empty:
        return {"overall": None, "by_lead": []}
    overall = _scores(table).to_dict()
    bins = pd.cut(table["step"], bins=list(lead_bins), include_lowest=True, right=False)
    by_lead = []
    for interval, grp in table.groupby(bins, observed=True):
        row = _scores(grp).to_dict()
        row["lead_h"] = f"{int(interval.left)}-{int(interval.right)}"
        by_lead.append(row)
    return {"overall": overall, "by_lead": by_lead}
