import numpy as np

from .flow import braking_flow


def predictive_row(self, x, now, when, forecast):
    position, velocity, sensitivity = braking_flow(
        x, float(when - now), self.max_accel, self.brake_gain
    )
    center, physical_radius = self.geometry_at(forecast, float(when))
    delta = position - center
    distance = float(np.linalg.norm(delta))
    if distance < 1e-12:
        raise ValueError("backup position coincides with hazard center")
    normal = delta / distance
    margin = distance - physical_radius - self.robot_radius - self.clearance
    row = normal * sensitivity[0]
    lower = -self.alpha * margin - normal @ (x[2:] - velocity)
    return {
        "kind": "predictive",
        "object_id": forecast.object_id,
        "time": float(when),
        "margin": float(margin),
        "coefficients": row.tolist(),
        "lower_bound": float(lower),
    }


def constraint_rows(self, x, now, family, grids):
    if self.backup_policy == "retreat" and family:
        from .retreat import retreat_rows

        rows, minimum, terminal = retreat_rows(self, x, now, family, grids)
        static = static_rows(self, x, now)
        rows.extend(static)
        return (rows, min([minimum, *[r["margin"] for r in static]]), terminal)
    rows = []
    margins = []
    for forecast, grid in zip(family, grids, strict=True):
        if now < forecast.times[0] - 1e-09 or now >= forecast.times[-1] - 1e-08:
            raise ValueError("forecast_expired_or_unsupported")
        times = np.unique(np.concatenate(([now], grid[grid > now + 1e-09])))
        for when in times:
            row = self._predictive_row(x, now, when, forecast)
            rows.append(row)
            margins.append(row["margin"])
    terminal_margin = None
    if family:
        endpoint = min((float(item.times[-1]) for item in family))
        _, velocity, sensitivity = braking_flow(x, endpoint - now, self.max_accel, self.brake_gain)
        terminal_margin = self.terminal_speed**2 - float(velocity @ velocity)
        row = -2.0 * velocity * sensitivity[1]
        final_brake = np.clip(-self.brake_gain * velocity, -self.max_accel, self.max_accel)
        lower = -self.alpha * terminal_margin - 2.0 * velocity @ final_brake
        rows.append(
            {
                "kind": "terminal",
                "object_id": "__terminal__",
                "time": endpoint,
                "margin": terminal_margin,
                "coefficients": row.tolist(),
                "lower_bound": float(lower),
            }
        )
    static = static_rows(self, x, now)
    rows.extend(static)
    margins.extend((row["margin"] for row in static))
    min_margin = min(margins) if margins else None
    return (rows, min_margin, terminal_margin)


def static_rows(self, x, now):
    rows = []
    for name, center, radius in self._static:
        delta = x[:2] - center
        expanded = radius + self.robot_radius + self.clearance
        h = float(delta @ delta - expanded**2)
        hdot = float(2.0 * delta @ x[2:])
        lower = -2.0 * float(x[2:] @ x[2:]) - 2.0 * self.alpha * hdot - self.alpha**2 * h
        rows.append(
            {
                "kind": "static_hocbf",
                "object_id": name,
                "time": now,
                "margin": float(np.linalg.norm(delta) - expanded),
                "hocbf_membership": hdot + self.alpha * h,
                "coefficients": (2.0 * delta).tolist(),
                "lower_bound": lower,
            }
        )
    return rows
