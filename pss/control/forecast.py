from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class Forecast:
    object_id: str
    origin_time: float
    times: np.ndarray
    centers: np.ndarray
    radii: np.ndarray | float


def freeze_forecast(forecast, now, sample_dt) -> tuple[Forecast, np.ndarray]:
    if not isinstance(forecast, Forecast):
        raise TypeError("family members must be Forecast objects")
    times = np.array(forecast.times, dtype=float, copy=True)
    centers = np.array(forecast.centers, dtype=float, copy=True)
    radii = np.array(forecast.radii, dtype=float, copy=True)
    if radii.ndim == 0:
        radii = np.full(times.shape, radii, dtype=float)
    origin = float(forecast.origin_time)
    if (
        not isinstance(forecast.object_id, str)
        or not forecast.object_id.strip()
        or times.ndim != 1
        or (times.size < 2)
        or (centers.shape != (times.size, 2))
        or (radii.shape != times.shape)
        or (not np.all(np.isfinite(times)))
        or (not np.all(np.isfinite(centers)))
        or (not np.all(np.isfinite(radii)))
        or np.any(radii < 0)
        or np.any(np.diff(times) <= 0)
        or (not np.isfinite(origin))
        or (origin > times[0] + 1e-09)
        or (origin > now + 1e-09)
        or (times[0] > now + 1e-09)
        or (times[-1] <= now + 1e-08)
    ):
        raise ValueError("invalid geometry, observation origin, or absolute-time support")
    grid_parts = [
        np.linspace(a, b, int(np.ceil((b - a) / sample_dt)) + 1) for a, b in pairwise(times)
    ]
    grid = np.unique(np.concatenate(grid_parts))
    for array in (times, centers, radii, grid):
        array.setflags(write=False)
    return (Forecast(forecast.object_id, origin, times, centers, radii), grid)


def geometry_at(forecast, when) -> tuple[np.ndarray, float]:
    if (
        not np.isfinite(when)
        or when < forecast.times[0] - 1e-09
        or when > forecast.times[-1] + 1e-09
    ):
        raise ValueError("forecast queried outside absolute-time support")
    center = np.array(
        [np.interp(when, forecast.times, forecast.centers[:, axis]) for axis in range(2)]
    )
    radius = float(np.interp(when, forecast.times, forecast.radii))
    return (center, radius)
