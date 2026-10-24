"""Deterministic and probabilistic scores of the ensemble against actual generation.

score_table() turns one run into one row per valid time; tables from several runs can be
concatenated and passed to summarize_scores() to pool them.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LEAD_BINS = (0, 24, 48, 72, 96, 120, 144, 240, 360)
MARKET_TZ = "Europe/Berlin"
PIT_BINS = 10


def crps_ensemble(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble per case: E|X - y| - 0.5 E|X - X'|.

    ens has shape (n_cases, n_members), obs has shape (n_cases,).
    """
    ens = np.asarray(ens, dtype=float)
    obs = np.asarray(obs, dtype=float)
    term1 = np.mean(np.abs(ens - obs[:, None]), axis=1)
    term2 = np.mean(np.abs(ens[:, :, None] - ens[:, None, :]), axis=(1, 2))
    return term1 - 0.5 * term2


def randomized_pit(ens: np.ndarray, obs: np.ndarray, seed: int = 0) -> np.ndarray:
    """Rank of the observation among the members, mapped to a uniform value in [0, 1).

    With m members the rank r is in 0..m; ties with members are broken at random, and
    u = (r + V) / (m + 1) with V ~ U(0, 1). For a calibrated ensemble u is uniform, so a
    histogram of u is the rank histogram with a flat expectation for any number of bins.
    """
    ens = np.asarray(ens, dtype=float)
    obs = np.asarray(obs, dtype=float)
    rng = np.random.default_rng(seed)
    below = (ens < obs[:, None]).sum(axis=1)
    ties = (ens == obs[:, None]).sum(axis=1)
    rank = below + np.floor(rng.random(len(obs)) * (ties + 1))
    return (rank + rng.random(len(obs))) / (ens.shape[1] + 1)


def is_day_ahead(base_time: pd.Series, valid_time: pd.Series) -> pd.Series:
    """True where a 00 UTC run forecasts the next delivery day (Berlin calendar date + 1).

    That is the run a trader has before the 12:00 CET day-ahead auction gate closure; 12 UTC
    runs arrive after it, so they never count as day-ahead.
    """
    base_day = base_time.dt.tz_convert(MARKET_TZ).dt.normalize()
    valid_day = valid_time.dt.tz_convert(MARKET_TZ).dt.normalize()
    return (base_time.dt.hour == 0) & (valid_day - base_day == pd.Timedelta(days=1))


def score_table(members: pd.DataFrame, actual: pd.Series, column: str = "total_mw") -> pd.DataFrame:
    """One row per valid time with quantiles, actual, CRPS, PIT and the persistence forecast.

    The persistence forecast is the actual at the run's base time (its step-0 valid time),
    held for every lead, so ENS and persistence use information up to the same moment.
    Times without actuals are dropped.
    """
    wide = members.pivot_table(index="valid_time", columns="member", values=column)
    steps = members.groupby("valid_time")["step"].first()
    actual = actual.reindex(wide.index)
    base = pd.Series(wide.index - pd.to_timedelta(steps.to_numpy(), unit="h"), index=wide.index)
    persistence = actual.reindex(base).to_numpy()
    ok = actual.notna().to_numpy()
    ens, obs = wide.to_numpy()[ok], actual.to_numpy()[ok]
    q = np.quantile(ens, [0.1, 0.5, 0.9], axis=1)
    table = pd.DataFrame({
        "valid_time": wide.index[ok],
        "base_time": base.to_numpy()[ok],
        "step": steps.to_numpy()[ok],
        "p10": q[0], "p50": q[1], "p90": q[2],
        "actual": obs,
        "crps": crps_ensemble(ens, obs),
        "pit": randomized_pit(ens, obs),
        "persistence": persistence[ok],
    })
    table["day_ahead"] = is_day_ahead(table["base_time"], table["valid_time"])
    return table


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _scores(group: pd.DataFrame) -> dict:
    err = group["p50"] - group["actual"]
    inside = (group["actual"] >= group["p10"]) & (group["actual"] <= group["p90"])
    # skill against persistence, on the cases where both forecasts exist
    both = group["persistence"].notna()
    mae_pers = (group.loc[both, "persistence"] - group.loc[both, "actual"]).abs().mean()
    crps_both = group.loc[both, "crps"].mean()
    # persistence is deterministic, so its CRPS equals its MAE
    skill = 1.0 - crps_both / mae_pers if both.any() and mae_pers > 0 else math.nan
    return {
        "n": int(len(group)),
        "mae_mw": _finite(err.abs().mean()),
        "rmse_mw": _finite(np.sqrt((err ** 2).mean())),
        "bias_mw": _finite(err.mean()),
        "crps_mw": _finite(group["crps"].mean()),
        "coverage_p10_p90": _finite(inside.mean()),
        "mae_persistence_mw": _finite(mae_pers) if both.any() else None,
        "crpss_vs_persistence": _finite(skill),
    }


def rank_histogram(pit: pd.Series, bins: int = PIT_BINS) -> dict:
    """Counts of the randomized PIT in equal bins; a calibrated ensemble gives n / bins each."""
    counts, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return {"bins": bins, "counts": [int(c) for c in counts], "expected": len(pit) / bins}


def summarize_scores(table: pd.DataFrame, lead_bins: tuple[int, ...] = LEAD_BINS) -> dict:
    """Overall scores, scores per lead-time bin (hours), the day-ahead window and the rank histogram."""
    if table.empty:
        return {"overall": None, "by_lead": [], "day_ahead": None, "rank_histogram": None}
    by_lead = []
    bins = pd.cut(table["step"], bins=list(lead_bins), include_lowest=True, right=False)
    for interval, grp in table.groupby(bins, observed=True):
        row = _scores(grp)
        row["lead_h"] = f"{int(interval.left)}-{int(interval.right)}"
        by_lead.append(row)
    da = table[table["day_ahead"]]
    return {
        "overall": _scores(table),
        "by_lead": by_lead,
        "day_ahead": _scores(da) if not da.empty else None,
        "rank_histogram": rank_histogram(table["pit"]),
    }


def verify(members: pd.DataFrame, actual: pd.Series, column: str = "total_mw",
           lead_bins: tuple[int, ...] = LEAD_BINS) -> dict:
    """Scores of one run; see summarize_scores()."""
    return summarize_scores(score_table(members, actual, column), lead_bins)
