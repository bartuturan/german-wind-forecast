"""Download one ENS run, turn it into a national wind power forecast, fetch actuals and score it.

Examples:
    python scripts/run_pipeline.py --run 2026-10-18T00 --max-step 48
    python scripts/run_pipeline.py --run latest --source ecmwf
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone

import pandas as pd

from windfc import download
from windfc.actuals import align_to, fetch_actuals
from windfc.aggregate import national_members
from windfc.config import DEFAULT_CONFIG, load_config
from windfc.decode import decode_ensemble, decode_field
from windfc.store import Store
from windfc.verify import verify

log = logging.getLogger("run_pipeline")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="latest", help="base time as YYYY-MM-DDTHH (UTC) or 'latest'")
    p.add_argument("--source", help="ecmwf | aws | azure (default from config)")
    p.add_argument("--max-step", type=int, default=144, help="forecast horizon in hours")
    p.add_argument("--step-interval", type=int, default=3, help="step spacing up to 144 h")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--keep-raw", action="store_true", help="keep the global GRIB files after cropping")
    p.add_argument("--refresh", action="store_true", help="re-download even if a cube is cached")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    store = Store(cfg.data_dir)
    source = args.source or cfg.source

    base_time = download.latest_run(source) if args.run == "latest" else download.parse_run(args.run)
    run = download.run_id(base_time)
    steps = download.ens_steps(args.max_step, args.step_interval)
    log.info("run %s, %d steps (0-%d h), source %s", run, len(steps), steps[-1], source)

    # 1. ENS 100 m wind, cropped to Germany
    if store.cube_path(run).exists() and not args.refresh:
        cube = store.load_cube(run)
        log.info("using cached cube %s", store.cube_path(run))
    else:
        files = download.download_run(base_time, steps, store.raw_dir(run), source=source)
        cube = decode_ensemble(files, cfg.bbox)
        store.save_cube(run, cube)
        if not args.keep_raw:
            shutil.rmtree(store.raw_dir(run), ignore_errors=True)
    log.info("cube u100 %s (member, step, lat, lon)", cube["u100"].shape)

    # 2. land-sea mask (optional)
    lsm = None
    lsm_file = download.download_lsm(store.lsm_path(), base_time, source=source)
    if lsm_file is not None:
        values, lat, lon = decode_field(lsm_file, cfg.bbox)
        if values.shape == cube["u100"].shape[-2:]:
            lsm = values
    log.info("land-sea mask: %s", "yes" if lsm is not None else "no")

    # 3. wind -> power
    members = national_members(cube, cfg, lsm)
    store.save_forecast(run, members)
    log.info("forecast saved: %s", store.forecast_path(run))

    # 4. actuals for the forecast window (only what has already happened)
    now = datetime.now(timezone.utc)
    start = base_time - timedelta(hours=6)
    end = min(members["valid_time"].max().to_pydatetime() + timedelta(hours=3), now)
    if end > start:
        store.save_actuals(fetch_actuals(start, end))

    # 5. scores
    times = pd.DatetimeIndex(members["valid_time"].unique())
    aligned = align_to(store.load_actuals(), times)
    result = verify(members, aligned["total_mw"])
    print(json.dumps({"run": run, **result}, indent=2, default=float))


if __name__ == "__main__":
    main()
