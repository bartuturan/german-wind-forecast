"""Retrieve ECMWF IFS ensemble 100 m wind from ECMWF open data."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from ecmwf.opendata import Client

log = logging.getLogger(__name__)

PARAMS = ["100u", "100v"]
# Open data ENS files (…-enfo-ef.grib2) carry the 50 perturbed members; the control is not included.
TYPES = ["pf"]


def run_id(base_time: datetime) -> str:
    return base_time.strftime("%Y-%m-%dT%H")


def parse_run(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)


def ens_steps(max_step: int, interval: int = 3) -> list[int]:
    """ENS output steps: 3-hourly to 144 h, 6-hourly beyond."""
    steps = [s for s in range(0, min(max_step, 144) + 1, interval)]
    steps += [s for s in range(150, max_step + 1, 6)]
    return steps


def latest_run(source: str) -> datetime:
    client = Client(source=source, model="ifs", resol="0p25")
    dt = client.latest(stream="enfo", type="pf", step=0, param=PARAMS)
    return dt.replace(tzinfo=timezone.utc)


def download_run(
    base_time: datetime,
    steps: list[int],
    out_dir: Path,
    source: str = "ecmwf",
    chunk: int = 4,
) -> list[Path]:
    """Download the run in chunks of steps, one GRIB2 file per chunk; existing files are skipped.

    ecmwf-opendata reads the .index file and fetches only the byte ranges of the requested
    fields, but every field is still global, so files are large. Crop and discard after decoding.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(source=source, model="ifs", resol="0p25")
    files = []
    for i in range(0, len(steps), chunk):
        part = steps[i : i + chunk]
        target = out_dir / f"ens_{part[0]:03d}-{part[-1]:03d}.grib2"
        if not target.exists():
            log.info("downloading steps %s -> %s", part, target.name)
            tmp = target.with_suffix(".part")
            client.retrieve(
                date=base_time.strftime("%Y%m%d"),
                time=base_time.hour,
                stream="enfo",
                type=TYPES,
                param=PARAMS,
                step=part,
                target=str(tmp),
            )
            tmp.replace(target)
        files.append(target)
    return files


def download_lsm(target: Path, base_time: datetime, source: str = "ecmwf") -> Path | None:
    """Land-sea mask from step 0 of the ENS run (member 1). Returns None if unavailable."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    client = Client(source=source, model="ifs", resol="0p25")
    try:
        client.retrieve(date=base_time.strftime("%Y%m%d"), time=base_time.hour, stream="enfo",
                        type="pf", number=1, step=0, param="lsm", target=str(target))
    except Exception as exc:  # the mask is a nice-to-have
        log.warning("land-sea mask not available (%s); regions will use all cells", exc)
        target.unlink(missing_ok=True)
        return None
    return target
