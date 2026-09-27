from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .commands import ControllerCommands
from .constraints import constraint_rows, predictive_row
from .flow import braking_flow
from .forecast import Forecast, freeze_forecast, geometry_at
from .qp import solve_acceleration
from .retreat import RetreatConfig, RetreatFlow

__all__ = ["CodeWorldController", "Forecast", "braking_flow"]


class CodeWorldController(ControllerCommands):
    def __init__(
        self,
        *,
        robot_radius: float = 0.403,
        clearance: float = 0.05,
        max_accel: float = 2.0,
        brake_gain: float = 2.5,
        sample_dt: float = 0.15,
        alpha: float = 3.0,
        terminal_speed: float = 0.12,
        static_obstacles: Sequence = (),
        inactive: bool = False,
        backup_policy: str = "braking",
        retreat_config: RetreatConfig | None = None,
    ):
        if backup_policy not in {"braking", "retreat"}:
            raise ValueError("backup_policy must be braking or retreat")
        if retreat_config is not None and (not isinstance(retreat_config, RetreatConfig)):
            raise TypeError("retreat_config must be RetreatConfig")
        self.backup_policy = backup_policy
        self.retreat_config = retreat_config or RetreatConfig()
        nonnegative = np.array([robot_radius, clearance], dtype=float)
        positive = np.array([max_accel, brake_gain, sample_dt, alpha, terminal_speed], dtype=float)
        if (
            not np.all(np.isfinite(nonnegative))
            or np.any(nonnegative < 0)
            or (not np.all(np.isfinite(positive)))
            or np.any(positive <= 0)
        ):
            raise ValueError("geometry margins must be nonnegative and controller gains positive")
        self.robot_radius = float(robot_radius)
        self.clearance = float(clearance)
        self.max_accel = float(max_accel)
        self.brake_gain = float(brake_gain)
        self.sample_dt = float(sample_dt)
        self.alpha = float(alpha)
        self.terminal_speed = float(terminal_speed)
        self.inactive = bool(inactive)
        self._family: tuple[Forecast, ...] = ()
        self._grids: tuple[np.ndarray, ...] = ()
        self._emergency_reason = None if inactive else "awaiting_forecast"
        self._last_time: float | None = None
        self._last_command: dict = {"status": "not_run", "enforced_rows": []}
        static = []
        for i, obstacle in enumerate(static_obstacles):
            if isinstance(obstacle, dict):
                name = str(obstacle.get("object_id", f"static_{i}"))
                center, radius = (obstacle["center"], obstacle["radius"])
            else:
                name = f"static_{i}"
                center, radius = obstacle
            center = np.array(center, dtype=float, copy=True)
            radius = float(radius)
            if (
                center.shape != (2,)
                or not np.all(np.isfinite(center))
                or (not np.isfinite(radius))
                or (radius < 0)
            ):
                raise ValueError("static obstacle must contain a finite center and radius")
            center.setflags(write=False)
            static.append((name, center, radius))
        self._static = tuple(static)

    @staticmethod
    def _state(state, now):
        x = np.asarray(state, dtype=float)
        t = float(now)
        if x.shape != (4,) or not np.all(np.isfinite(x)) or (not np.isfinite(t)):
            raise ValueError("state must be finite [x,y,vx,vy] and time must be finite")
        return (x, t)

    @staticmethod
    def _member(rows, terminal_margin):
        return bool(
            all(
                (
                    row["margin"] >= -1e-09 and row.get("hocbf_membership", 0.0) >= -1e-09
                    for row in rows
                )
            )
            and (terminal_margin is None or terminal_margin >= -1e-09)
        )

    def update(self, forecasts, now, state) -> dict:
        try:
            x, t = self._state(state, now)
            if self._last_time is not None and t < self._last_time - 1e-09:
                raise ValueError("time_reversed")
            frozen = [self._freeze(item, t) for item in forecasts]
            if not frozen:
                raise ValueError("empty_active_family")
            candidate = tuple((item[0] for item in frozen))
            grids = tuple((item[1] for item in frozen))
            names = [item.object_id for item in candidate]
            if len(names) != len(set(names)):
                raise ValueError("duplicate_object_id")
            previous = {item.object_id: item for item in self._family}
            if not set(previous).issubset(names):
                raise ValueError("active_object_omitted")
            for item in candidate:
                if (
                    item.object_id in previous
                    and item.origin_time <= previous[item.object_id].origin_time + 1e-09
                ):
                    raise ValueError("observation_origin_not_new")
            rows, minimum, terminal = self._constraints(x, t, candidate, grids)
            if not self._member(rows, terminal):
                raise ValueError("candidate_backup_membership_failed")
            _, status = self._solve(self.backup_input(x, t, candidate), rows)
            if status != "solved":
                raise ValueError(status)
        except (
            TypeError,
            ValueError,
            KeyError,
            OverflowError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            status = f"update_rejected:{exc}"
            if not self._family:
                self.invalidate(status)
            return {"accepted": False, "status": status}
        self._family, self._grids = (candidate, grids)
        self._emergency_reason = None
        self.inactive = False
        self._last_time = t
        self._last_command = {"status": "family_committed", "enforced_rows": []}
        return {
            "accepted": True,
            "status": "committed",
            "min_margin": minimum,
            "terminal_margin": terminal,
            "object_count": len(candidate),
        }

    def invalidate(self, reason):
        self._emergency_reason = str(reason)
        return {"accepted": False, "status": self._emergency_reason}

    def snapshot(self) -> dict:
        return {
            "backup_policy": self.backup_policy,
            "retreat_config": asdict(self.retreat_config)
            if self.backup_policy == "retreat"
            else None,
            "terminal_continuation_verified": False,
            "coverage": "uncertainty_in_forecast_geometry",
            "q_m": 0.0,
            "robot_radius_m": self.robot_radius,
            "clearance_m": self.clearance,
            "inactive": self.inactive,
            "emergency_reason": self._emergency_reason,
            "forecasts": [
                {
                    "object_id": item.object_id,
                    "origin_time": item.origin_time,
                    "times": item.times.tolist(),
                    "centers": item.centers.tolist(),
                    "radii": item.radii.tolist(),
                    "constraint_times": grid.tolist(),
                }
                for item, grid in zip(self._family, self._grids, strict=True)
            ],
            "static_obstacles": [
                {"object_id": name, "center": center.tolist(), "radius": radius}
                for name, center, radius in self._static
            ],
            "last_command": deepcopy(self._last_command),
            "enforced_rows": deepcopy(self._last_command.get("enforced_rows", [])),
        }

    def _freeze(self, forecast, now):
        return freeze_forecast(forecast, now, self.sample_dt)

    geometry_at = staticmethod(geometry_at)

    def _predictive_row(self, x, now, when, forecast):
        return predictive_row(self, x, now, when, forecast)

    def _constraints(self, x, now, family, grids):
        return constraint_rows(self, x, now, family, grids)

    def _solve(self, nominal, rows):
        return solve_acceleration(nominal, rows, self.max_accel)

    def backup_input(self, state, now, family=None):
        x, t = self._state(state, now)
        family = self._family if family is None else family
        if self.backup_policy == "retreat" and family:
            if t >= min((f.times[-1] for f in family)) - 1e-08:
                raise ValueError("retreat forecast expired")
            if self._last_time is not None and t < self._last_time - 1e-09:
                raise ValueError("time_reversed")
            return RetreatFlow(self, family).evaluate(x, t)[0][2:]
        return np.clip(-self.brake_gain * x[2:], -self.max_accel, self.max_accel)
