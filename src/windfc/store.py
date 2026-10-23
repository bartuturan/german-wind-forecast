"""File cache under data/: cropped wind cubes (npz), forecasts and actuals (parquet)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)

    # layout
    def raw_dir(self, run: str) -> Path:
        return self.root / "raw" / run

    def cube_path(self, run: str) -> Path:
        return self.root / "cube" / f"{run}.npz"

    def forecast_path(self, run: str) -> Path:
        return self.root / "forecast" / f"{run}.parquet"

    def actuals_path(self) -> Path:
        return self.root / "actuals" / "wind_de.parquet"

    def lsm_path(self) -> Path:
        return self.root / "static" / "lsm.grib2"

    def calibration_path(self) -> Path:
        return self.root / "calibration.json"

    # wind cube
    def save_cube(self, run: str, cube: dict) -> Path:
        path = self.cube_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, u100=cube["u100"], v100=cube["v100"], members=cube["members"],
            steps=cube["steps"], lat=cube["lat"], lon=cube["lon"],
            base_time=np.array(cube["base_time"].timestamp()),
        )
        return path

    def load_cube(self, run: str) -> dict:
        with np.load(self.cube_path(run)) as z:
            cube = {k: z[k] for k in z.files}
        cube["base_time"] = datetime.fromtimestamp(float(cube["base_time"]), tz=timezone.utc)
        return cube

    # forecasts (member-level, long format)
    def save_forecast(self, run: str, members: pd.DataFrame) -> Path:
        path = self.forecast_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        members.to_parquet(path, index=False)
        return path

    def load_forecast(self, run: str) -> pd.DataFrame:
        return pd.read_parquet(self.forecast_path(run))

    def runs(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "forecast").glob("*.parquet"))

    # actuals: one growing table, de-duplicated on time
    def save_actuals(self, df: pd.DataFrame) -> Path:
        path = self.actuals_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df])
            df = df[~df.index.duplicated(keep="last")]
        df.sort_index().to_parquet(path)
        return path

    def load_actuals(self) -> pd.DataFrame:
        path = self.actuals_path()
        if not path.exists():
            return pd.DataFrame(columns=["onshore_mw", "offshore_mw", "total_mw"])
        return pd.read_parquet(path)
