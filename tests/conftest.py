import pandas as pd
import pytest

from windfc.verify import score_table, scores_by, summarize_scores


def _members(times, offset=0.0, scale=1.0):
    rows = [{"valid_time": t, "step": 3 * i, "member": m,
             "total_mw": scale * (20000.0 + 500 * m + 1000 * i) + offset}
            for i, t in enumerate(times) for m in range(11)]
    return pd.DataFrame(rows)


@pytest.fixture
def evaluation():
    """An evaluation.json-shaped dict from two synthetic 00Z runs, 0-48 h (so a day-ahead window)."""
    tables = {"raw": [], "calibrated": []}
    runs = ["2026-10-01T00", "2026-10-02T00"]
    for k, run in enumerate(runs):
        times = pd.date_range(run.replace("T00", " 00:00"), periods=17, freq="3h", tz="UTC")
        actual = pd.Series(25000.0 + 300 * pd.RangeIndex(17).to_numpy(), index=times)
        for variant, scale in (("raw", 1.0), ("calibrated", 0.8)):
            table = score_table(_members(times, offset=500 * k, scale=scale), actual)
            tables[variant].append(table.assign(run=run))
    out = {"fleet": "total", "runs_scored": runs, "min_train": 7}
    for variant, parts in tables.items():
        pooled = pd.concat(parts, ignore_index=True)
        out[variant] = {**summarize_scores(pooled), "by_run": scores_by(pooled, "run")}
    return out
