"""Rolling-origin evaluation of the per-fleet scale calibration, then a final fit on all runs.

    python scripts/calibrate.py
    python scripts/calibrate.py --min-train 10 --fleet onshore

Runs are taken in base-time order. Each run is calibrated with scales fitted only on earlier
runs and only on actuals that were known at its base time, which is how the scales would have
been used live. The first --min-train runs are a warm-up and are not scored. Scores are pooled
over all scored runs: overall, by lead time, the day-ahead window, per run, against persistence,
plus a rank histogram (data/rank_histogram.png). Results go to data/evaluation.json and are
drawn as a scorecard (data/scores.png), which the API also serves at /evaluation/chart.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from windfc.actuals import align_to
from windfc.calibrate import apply_scales, fit_scales, save_scales
from windfc.chart import rank_histogram_chart, scores_chart
from windfc.config import DEFAULT_CONFIG, load_config
from windfc.download import parse_run
from windfc.store import Store
from windfc.verify import score_table, scores_by, summarize_scores

COLUMNS = [("mae_mw", "MAE"), ("crps_mw", "CRPS"), ("bias_mw", "bias"),
           ("coverage_p10_p90", "cov"), ("crpss_vs_persistence", "CRPSS")]


def comparison(summaries: dict[str, dict]) -> pd.DataFrame:
    """One row per slice (overall, lead bins, day-ahead), raw and calibrated side by side."""
    def slices(s: dict) -> dict[str, dict]:
        out = {"overall": s["overall"]}
        out.update({f"lead {r['lead_h']} h": r for r in s["by_lead"]})
        if s["day_ahead"]:
            out["day-ahead"] = s["day_ahead"]
        return out

    per_variant = {v: slices(s) for v, s in summaries.items()}
    rows = []
    for name, raw in per_variant["raw"].items():
        row = {"slice": name, "n": raw["n"], "persist MAE": raw["mae_persistence_mw"]}
        for key, label in COLUMNS:
            for variant, sl in per_variant.items():
                row[f"{label} {variant[:3]}"] = sl[name][key]
        rows.append(row)
    return pd.DataFrame(rows).set_index("slice")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--min-train", type=int, default=7, help="earlier runs needed before a run is scored")
    p.add_argument("--fleet", default="total", choices=["total", "onshore", "offshore"])
    args = p.parse_args()
    column = f"{args.fleet}_mw"

    store = Store(load_config(args.config).data_dir)
    actuals = store.load_actuals()
    pairs = {}
    for run in store.runs():
        members = store.load_forecast(run)
        aligned = align_to(actuals, pd.DatetimeIndex(members["valid_time"].unique()))
        if aligned["total_mw"].notna().any():
            pairs[run] = (members, aligned)
    runs = sorted(pairs, key=parse_run)
    if len(runs) <= args.min_train:
        raise SystemExit(f"need more than {args.min_train} runs with actuals, have {len(runs)}")

    tables = {"raw": [], "calibrated": []}
    by_run = {}
    for i, run in enumerate(runs):
        base = pd.Timestamp(parse_run(run))
        earlier = [pairs[r] for r in runs[:i]]
        if len(earlier) < args.min_train:
            continue
        scales = fit_scales(earlier, before=base)
        by_run[run] = scales
        members, aligned = pairs[run]
        for variant, m in (("raw", members), ("calibrated", apply_scales(members, scales))):
            tables[variant].append(score_table(m, aligned[column], column).assign(run=run))

    pooled = {v: pd.concat(t, ignore_index=True) for v, t in tables.items()}
    summaries = {v: {**summarize_scores(t), "by_run": scores_by(t, "run")} for v, t in pooled.items()}
    scored = list(by_run)
    print(f"{args.fleet}: {len(scored)} runs scored ({scored[0]} .. {scored[-1]}), "
          f"{len(runs) - len(scored)} warm-up runs\n")
    print(comparison(summaries).round(2).to_string())

    scale_table = pd.DataFrame(by_run).T
    print("\nrolling-origin scales (first, last):")
    print(scale_table.iloc[[0, -1]].round(3).to_string())

    da = {v: t.loc[t["day_ahead"], "pit"] for v, t in pooled.items()}
    hist_path = store.rank_histogram_path()
    hist_path.write_bytes(rank_histogram_chart(
        {"raw, all leads": pooled["raw"]["pit"], "calibrated, all leads": pooled["calibrated"]["pit"],
         "calibrated, day-ahead": da["calibrated"]},
        f"Rank histogram, German wind ({args.fleet}), {len(scored)} runs"))
    evaluation = {"fleet": args.fleet, "runs_scored": scored, "min_train": args.min_train, **summaries}
    eval_path = store.evaluation_path()
    eval_path.write_text(json.dumps(evaluation, indent=2))
    scores_path = store.scores_chart_path()
    scores_path.write_bytes(scores_chart(evaluation))
    print(f"\n-> {hist_path}\n-> {eval_path}\n-> {scores_path}")

    scales = fit_scales(list(pairs.values()))
    save_scales(store.calibration_path(), scales, by_run)
    print(f"scales fitted on all {len(pairs)} runs (for new runs): "
          f"{ {k: round(v, 3) for k, v in scales.items()} } -> {store.calibration_path()}")


if __name__ == "__main__":
    main()
