"""Decode GRIB2 with eccodes and crop global fields to the configured box."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import eccodes
import numpy as np

from .config import BBox


def grid_coords(ni: int, nj: int, lat0: float, lon0: float, dlat: float, dlon: float,
                j_scans_positively: bool) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude vectors of a regular_ll grid, longitudes wrapped to [-180, 180)."""
    lats = lat0 + (1 if j_scans_positively else -1) * dlat * np.arange(nj)
    lons = lon0 + dlon * np.arange(ni)
    lons = (lons + 180.0) % 360.0 - 180.0
    return lats, lons


def crop_index(lats: np.ndarray, lons: np.ndarray, bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices inside the box, ordered so the result has ascending lat and lon."""
    rows = np.where((lats >= bbox.lat_min) & (lats <= bbox.lat_max))[0]
    cols = np.where((lons >= bbox.lon_min) & (lons <= bbox.lon_max))[0]
    rows = rows[np.argsort(lats[rows])]
    cols = cols[np.argsort(lons[cols])]
    return rows, cols


def iter_messages(path: Path) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield (metadata, 2-D values) for every message in a GRIB file."""
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                meta = {
                    "shortName": eccodes.codes_get(h, "shortName"),
                    "number": eccodes.codes_get(h, "number"),
                    "step": int(eccodes.codes_get(h, "endStep")),
                    "dataDate": eccodes.codes_get(h, "dataDate"),
                    "dataTime": eccodes.codes_get(h, "dataTime"),
                    "Ni": eccodes.codes_get(h, "Ni"),
                    "Nj": eccodes.codes_get(h, "Nj"),
                    "lat0": eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                    "lon0": eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
                    "dlat": eccodes.codes_get(h, "jDirectionIncrementInDegrees"),
                    "dlon": eccodes.codes_get(h, "iDirectionIncrementInDegrees"),
                    "jpos": bool(eccodes.codes_get(h, "jScansPositively")),
                }
                values = eccodes.codes_get_values(h).reshape(meta["Nj"], meta["Ni"])
            finally:
                eccodes.codes_release(h)
            yield meta, values


def _coords_of(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    return grid_coords(meta["Ni"], meta["Nj"], meta["lat0"], meta["lon0"],
                       meta["dlat"], meta["dlon"], meta["jpos"])


def decode_ensemble(files: Iterable[Path], bbox: BBox) -> dict:
    """Read u100/v100 for all members and steps into cubes of shape (member, step, lat, lon).

    Returns a dict with base_time, members, steps, lat, lon, u100, v100 (float32).
    """
    fields: dict[tuple[str, int, int], np.ndarray] = {}
    rows = cols = lats = lons = None
    base_time = None
    for path in files:
        for meta, values in iter_messages(path):
            if rows is None:
                glats, glons = _coords_of(meta)
                rows, cols = crop_index(glats, glons, bbox)
                lats, lons = glats[rows], glons[cols]
                base_time = datetime.strptime(
                    f"{meta['dataDate']}{meta['dataTime']:04d}", "%Y%m%d%H%M"
                ).replace(tzinfo=timezone.utc)
            name = {"100u": "u100", "100v": "v100"}.get(meta["shortName"], meta["shortName"])
            fields[(name, meta["number"], meta["step"])] = values[np.ix_(rows, cols)].astype(np.float32)

    if not fields:
        raise ValueError("no GRIB messages decoded")
    members = sorted({k[1] for k in fields})
    steps = sorted({k[2] for k in fields})
    out = {"base_time": base_time, "members": np.array(members), "steps": np.array(steps),
           "lat": lats, "lon": lons}
    for var in ("u100", "v100"):
        cube = np.full((len(members), len(steps), len(lats), len(lons)), np.nan, dtype=np.float32)
        for (name, m, s), arr in fields.items():
            if name == var:
                cube[members.index(m), steps.index(s)] = arr
        out[var] = cube
    return out


def decode_field(path: Path, bbox: BBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First message of a file (e.g. the land-sea mask), cropped: (values, lat, lon)."""
    meta, values = next(iter_messages(path))
    glats, glons = _coords_of(meta)
    rows, cols = crop_index(glats, glons, bbox)
    return values[np.ix_(rows, cols)], glats[rows], glons[cols]


def valid_times(base_time: datetime, steps: np.ndarray) -> list[datetime]:
    return [base_time + timedelta(hours=int(s)) for s in steps]
