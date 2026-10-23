"""Wind speed to capacity factor."""
from __future__ import annotations

import numpy as np

from .config import PowerCurve


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.hypot(u, v)


def to_hub_height(ws100: np.ndarray, hub_height: float, alpha: float) -> np.ndarray:
    """Power-law shear profile: v(h) = v(100 m) * (h / 100)^alpha."""
    return ws100 * (hub_height / 100.0) ** alpha


def capacity_factor(ws: np.ndarray, curve: PowerCurve) -> np.ndarray:
    """Interpolate the power curve; zero above cut-out. NaN input stays NaN."""
    ws = np.asarray(ws, dtype=float)
    cf = np.interp(ws, curve.speeds, curve.cf, left=0.0, right=curve.cf[-1])
    cf = np.where(ws > curve.cut_out, 0.0, cf)
    return np.where(np.isnan(ws), np.nan, cf)
