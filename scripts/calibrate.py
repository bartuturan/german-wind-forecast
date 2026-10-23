"""Fit per-fleet scale factors over all cached runs and report leave-one-run-out scores.

    python scripts/calibrate.py
"""
from __future__ import annotations

import argparse

import pandas as pd

from windfc.actuals import align_to
from windfc.calibrate import apply_scales, fit_scales, save_scales
from windfc.config import DEFAULT_CONFIG, load_config
from windfc.store import Store
from windfc.verify import verify


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args()

    store = Store(load_config(args.config).data_dir)
    actuals = store.load_actuals()
    pairs = {}
    for run in store.runs():
        members = store.load_forecast(run)
        aligned = align_to(actuals, pd.DatetimeIndex(members["valid_time"].unique()))
        if aligned["total_mw"].notna().any():
            pairs[run] = (members, aligned)
    if len(pairs) < 2:
        raise SystemExit(f"need at least 2 runs with actuals, have {len(pairs)}")

    # leave-one-run-out: fit on the other runs, score the held-out run
    rows = []
    for run, (members, aligned) in pairs.items():
        scales = fit_scales([v for k, v in pairs.items() if k != run])
        raw = verify(members, aligned["total_mw"])["overall"]
        cal = verify(apply_scales(members, scales), aligned["total_mw"])["overall"]
        rows.append({"run": run, "mae_raw": raw["mae_mw"], "mae_cal": cal["mae_mw"],
                     "crps_raw": raw["crps_mw"], "crps_cal": cal["crps_mw"],
                     "cov_raw": raw["coverage_p10_p90"], "cov_cal": cal["coverage_p10_p90"]})
    table = pd.DataFrame(rows).set_index("run")
    print(table.round(2).to_string())
    print("\nmean:\n" + table.mean().round(2).to_string())

    scales = fit_scales(list(pairs.values()))
    save_scales(store.calibration_path(), scales)
    print(f"\nscales fitted on all {len(pairs)} runs: {scales} -> {store.calibration_path()}")


if __name__ == "__main__":
    main()
