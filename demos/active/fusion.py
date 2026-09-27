from copy import deepcopy

import numpy as np

from pss.control.forecast import Forecast, freeze_forecast, geometry_at


class OracleGround:
    def __init__(self, objects, horizon=2.5):
        self.spec = {o["id"]: o for o in objects}
        self.horizon = horizon
        self.grounded = set()
        self.measurements = {}
        self.time = None

    def observe(self, grounded_ids, measurements, now):
        if self.time is not None and now <= self.time:
            raise ValueError("Oracle ground observations must be strictly chronological")
        active = self.grounded | set(grounded_ids)
        if not active.issubset(self.spec) or set(measurements) != active:
            raise ValueError("Complete per-body ground inventory is required")
        candidate = {}
        for name, values in measurements.items():
            center, velocity = [np.asarray(value, float).copy() for value in values]
            if (
                center.shape != (2,)
                or velocity.shape != (2,)
                or (not np.isfinite(np.r_[center, velocity]).all())
            ):
                raise ValueError("Oracle ground measurement must be finite XY and velocity")
            candidate[name] = (center, velocity)
        newly = active - self.grounded
        self.measurements, self.grounded, self.time = (candidate, active, float(now))
        return sorted(newly)

    def forecast(self, name, now):
        if self.time is None or abs(now - self.time) > 1e-08 or name not in self.grounded:
            raise ValueError("A fresh oracle ground observation is required")
        position, velocity = self.measurements[name]
        offsets = np.linspace(0, self.horizon, 6)
        return (
            Forecast(
                name,
                now,
                now + offsets,
                position + offsets[:, None] * velocity,
                self.spec[name]["radius"],
            ),
            dict(
                source="oracle_current_ground_position_velocity",
                original_origin=now,
                original_support=[now, now + self.horizon],
                center=position.tolist(),
                velocity=velocity.tolist(),
                model="constant_current_velocity; no future simulation rollout",
                calibrated=False,
            ),
        )


def fused_family(semantic, ground, names, now):
    family, sources = ([], {})
    for name in names:
        if name in ground.grounded:
            forecast, sources[name] = ground.forecast(name, now)
        else:
            if name not in semantic:
                raise ValueError(f"No semantic support for airborne {name}")
            raw = semantic[name]
            old, _ = freeze_forecast(raw, raw.origin_time, 0.15)
            if now < old.times[0] - 1e-09 or now >= old.times[-1] - 1e-08:
                raise ValueError(f"Airborne semantic support expired: {name}")
            times = np.r_[now, np.asarray(old.times)[np.asarray(old.times) > now + 1e-09]]
            geometry = [geometry_at(old, float(t)) for t in times]
            forecast = Forecast(
                name,
                now,
                times,
                np.array([x[0] for x in geometry]),
                np.array([x[1] for x in geometry]),
            )
            sources[name] = dict(
                source="code_as_world",
                original_origin=old.origin_time,
                original_support=[float(old.times[0]), float(old.times[-1])],
            )
        family.append(forecast)
    return (family, deepcopy(sources))
