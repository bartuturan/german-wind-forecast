"""Run the pipeline for a range of past runs, one subprocess per run, so a failure skips only that run.

    python scripts/backfill.py --start 2026-09-23 --end 2026-10-19
    python scripts/backfill.py --start 2026-10-01 --end 2026-10-07 --hours 0 12 --step-interval 3
    python scripts/backfill.py --start 2026-09-23 --end 2026-10-21 --actuals-only

Runs that already have a forecast in data/forecast/ are skipped. Each run's output goes to
data/logs/pipeline_<run>.log. At the end the actuals for the whole range are fetched again in
one request, which fills hours that had not happened yet when a run was first processed (and
any gaps from backfills running in parallel).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from windfc.actuals import fetch_actuals
from windfc.config import DEFAULT_CONFIG, load_config
from windfc.store import Store

PIPELINE = Path(__file__).with_name("run_pipeline.py")


def run_with_timeout(cmd: list[str], log, timeout_s: float) -> int | str:
    """Exit code of cmd, or "timeout" after killing its whole process tree.

    The download client has no read timeout, so a dropped connection can hang a run forever.
    On Windows the venv python.exe is a launcher with a child process, hence the tree kill.
    """
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.kill()
        proc.wait()
        return "timeout"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, type=date.fromisoformat, help="first base date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, type=date.fromisoformat, help="last base date, inclusive")
    p.add_argument("--hours", type=int, nargs="+", default=[0], help="base hours to run (UTC)")
    p.add_argument("--source", default="azure")
    p.add_argument("--max-step", type=int, default=48)
    p.add_argument("--step-interval", type=int, default=6)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--attempts", type=int, default=3, help="tries per run (network errors are common)")
    p.add_argument("--timeout-min", type=float, default=30, help="kill a run that takes longer than this")
    p.add_argument("--actuals-only", action="store_true", help="only refetch the actuals for the range")
    args = p.parse_args()

    store = Store(load_config(args.config).data_dir)
    log_dir = store.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    day = args.start
    failed = []
    while day <= args.end and not args.actuals_only:
        for hour in args.hours:
            run = datetime(day.year, day.month, day.day, hour).strftime("%Y-%m-%dT%H")
            if store.forecast_path(run).exists():
                print(f"{run} cached, skipping", flush=True)
                continue
            print(f"{run} running", flush=True)
            cmd = [sys.executable, str(PIPELINE), "--run", run, "--source", args.source,
                   "--max-step", str(args.max_step), "--step-interval", str(args.step_interval),
                   "--config", args.config]
            for attempt in range(1, args.attempts + 1):
                with open(log_dir / f"pipeline_{run}.log", "w") as log:
                    code = run_with_timeout(cmd, log, args.timeout_min * 60)
                print(f"{run} attempt {attempt}: {'ok' if code == 0 else f'FAILED ({code})'}", flush=True)
                if code == 0:
                    break
                if attempt < args.attempts:
                    time.sleep(60 * attempt)  # Azure's SAS token service has short outages
            else:
                failed.append(run)
        day += timedelta(days=1)

    start = datetime(args.start.year, args.start.month, args.start.day, tzinfo=timezone.utc)
    last_valid = timedelta(days=(args.end - args.start).days, hours=max(args.hours) + args.max_step + 3)
    end = min(start + last_valid, datetime.now(timezone.utc))
    store.save_actuals(fetch_actuals(start - timedelta(hours=6), end))
    print(f"actuals refreshed {start:%Y-%m-%d} .. {end:%Y-%m-%d %H:%M}", flush=True)
    if failed:
        raise SystemExit(f"failed runs: {failed}")


if __name__ == "__main__":
    main()
