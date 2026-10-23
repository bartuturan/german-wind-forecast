"""Load config.yaml into typed objects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class PowerCurve:
    speeds: tuple[float, ...]
    cf: tuple[float, ...]
    cut_out: float
    hub_height: float


@dataclass(frozen=True)
class Region:
    name: str
    type: str  # "onshore" | "offshore"
    capacity_mw: float
    lat: tuple[float, float]
    lon: tuple[float, float]


@dataclass(frozen=True)
class Config:
    data_dir: Path
    source: str
    bbox: BBox
    shear_enabled: bool
    shear_alpha: float
    curves: dict[str, PowerCurve]
    availability: dict[str, float]
    regions: tuple[Region, ...]

    def installed_mw(self, fleet: str | None = None) -> float:
        return sum(r.capacity_mw for r in self.regions if fleet is None or r.type == fleet)


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    data_dir = Path(raw.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = path.parent / data_dir

    curves = {}
    for name, c in raw["power_curves"].items():
        pts = sorted(c["points"])
        curves[name] = PowerCurve(
            speeds=tuple(float(p[0]) for p in pts),
            cf=tuple(float(p[1]) for p in pts),
            cut_out=float(c["cut_out"]),
            hub_height=float(c.get("hub_height", 100)),
        )

    regions = tuple(
        Region(
            name=name,
            type=r["type"],
            capacity_mw=float(r["capacity_mw"]),
            lat=(float(r["lat"][0]), float(r["lat"][1])),
            lon=(float(r["lon"][0]), float(r["lon"][1])),
        )
        for name, r in raw["regions"].items()
    )

    return Config(
        data_dir=data_dir,
        source=raw.get("source", "aws"),
        bbox=BBox(**raw["bbox"]),
        shear_enabled=bool(raw["shear"]["enabled"]),
        shear_alpha=float(raw["shear"]["alpha"]),
        curves=curves,
        availability={k: float(v) for k, v in raw["availability"].items()},
        regions=regions,
    )
