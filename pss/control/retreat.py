from dataclasses import asdict, dataclass

import numpy as np

from .forecast import geometry_at

RETREAT_SCHEMA = "pss.absolute-time-retreat.v2"


@dataclass(frozen=True)
class RetreatConfig:
    speed: float = 0.5
    gain: float = 3.0
    threat_length: float = 0.3
    reference_cap: float = 1.2
    integration_dt: float = 0.025
    terminal_clearance: float = 0.08

    def __post_init__(self):
        if not all((np.isfinite(v) and v > 0 for v in asdict(self).values())):
            raise ValueError("retreat settings must be finite and positive")


def controller_descriptor(backup_policy):
    if backup_policy not in {"braking", "retreat"}:
        raise ValueError("backup_policy must be braking or retreat")
    return dict(
        backup_policy=backup_policy,
        schema=RETREAT_SCHEMA if backup_policy == "retreat" else "braking_v1",
        retreat_config=asdict(RetreatConfig()) if backup_policy == "retreat" else None,
        terminal_continuation_verified=False,
    )


class RetreatFlow:
    def __init__(self, controller, family):
        self.family = family
        self.static = controller._static
        self.config = controller.retreat_config
        self.radius = controller.robot_radius + controller.clearance
        self.max_accel = controller.max_accel

    def geometry(self, when):
        values = []
        for forecast in self.family:
            center, radius = geometry_at(forecast, when)
            i = min(
                np.searchsorted(forecast.times, when, side="right") - 1, len(forecast.times) - 2
            )
            i = max(i, 0)
            dt = forecast.times[i + 1] - forecast.times[i]
            velocity = (forecast.centers[i + 1] - forecast.centers[i]) / dt
            growth = (forecast.radii[i + 1] - forecast.radii[i]) / dt
            values.append((forecast.object_id, center, radius, velocity, growth))
        values.extend(
            ((name, center, radius, np.zeros(2), 0.0) for name, center, radius in self.static)
        )
        if not values:
            raise ValueError("retreat requires supported geometry")
        names, centers, radii, velocities, growth = zip(*values, strict=True)
        return (names, np.array(centers), np.array(radii), np.array(velocities), np.array(growth))

    def evaluate(self, state, when):
        _, centers, radii, velocities, growth = self.geometry(when)
        delta = state[:2] - centers
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances < 1e-09):
            raise ValueError("backup position coincides with hazard center")
        normals = delta / distances[:, None]
        margins = distances - radii - self.radius
        weights = np.exp(-(margins - margins.min()) / self.config.threat_length)
        weights /= weights.sum()
        weight_jac = -weights[:, None] / self.config.threat_length * (normals - weights @ normals)
        normal_jac = (np.eye(2) - normals[:, :, None] * normals[:, None, :]) / distances[
            :, None, None
        ]
        speed = self.config.speed + np.maximum(growth, 0.0)
        targets = velocities + speed[:, None] * normals
        reference = weights @ targets
        reference_jac = np.einsum("ni,nj->ij", targets, weight_jac)
        reference_jac += np.einsum("n,nij->ij", weights * speed, normal_jac)
        norm = np.linalg.norm(reference)
        if norm > 1e-09:
            q = norm / self.config.reference_cap
            scale = np.tanh(q) / q
            slope = (1 - np.tanh(q) ** 2 - scale) / norm**2
            reference_jac = (
                scale * np.eye(2) + slope * np.outer(reference, reference)
            ) @ reference_jac
            reference = scale * reference
        raw = self.config.gain * (reference - state[2:])
        saturated = np.tanh(raw / self.max_accel)
        acceleration = self.max_accel * saturated
        jac = np.zeros((4, 4))
        jac[:2, 2:] = np.eye(2)
        jac[2:, :2] = self.config.gain * reference_jac
        jac[2:, 2:] = -self.config.gain * np.eye(2)
        jac[2:] *= (1 - saturated**2)[:, None]
        return (np.r_[state[2:], acceleration], jac)

    def rollout(self, state, now, times):
        targets = np.asarray(times, dtype=float)
        if (
            targets.ndim != 1
            or not len(targets)
            or (not np.all(np.isfinite(targets)))
            or (targets[0] < now - 1e-09)
            or np.any(np.diff(targets) <= 0)
        ):
            raise ValueError("invalid absolute rollout grid")
        if any(
            (now < f.times[0] - 1e-09 or targets[-1] > f.times[-1] + 1e-09 for f in self.family)
        ):
            raise ValueError("common rollout exceeds forecast support")
        knots = sorted({float(t) for f in self.family for t in f.times if now < t < targets[-1]})
        boundaries = np.unique(np.r_[targets, knots])
        y = np.r_[state, np.eye(4).ravel()]
        integration_time = float(now)
        samples = {}

        def rhs(t, value):
            f, jac = self.evaluate(value[:4], t)
            return np.r_[f, (jac @ value[4:].reshape(4, 4)).ravel()]

        for boundary in boundaries:
            n = max(1, int(np.ceil((boundary - integration_time) / self.config.integration_dt)))
            start = integration_time
            for step in range(n):
                integration_time = start + (boundary - start) * step / n
                right = (
                    float(boundary)
                    if step == n - 1
                    else start + (boundary - start) * (step + 1) / n
                )
                dt = right - integration_time
                end = np.nextafter(right, integration_time) if dt else integration_time
                k1 = rhs(integration_time, y)
                k2 = rhs(integration_time + dt / 2, y + dt * k1 / 2)
                k3 = rhs(integration_time + dt / 2, y + dt * k2 / 2)
                k4 = rhs(end, y + dt * k3)
                y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            integration_time = float(boundary)
            if boundary in targets:
                if not np.all(np.isfinite(y)):
                    raise ValueError("nonfinite retreat rollout")
                samples[float(boundary)] = (y[:4].copy(), y[4:].reshape(4, 4).copy())
        return samples


def retreat_rows(controller, state, now, family, grids):
    flow = RetreatFlow(controller, family)
    endpoint = min((f.times[-1] for f in family))
    times = np.unique(
        np.concatenate([[now, endpoint], *[g[(g > now) & (g <= endpoint)] for g in grids]])
    )
    if endpoint <= now + 1e-08:
        raise ValueError("forecast_expired_or_unsupported")
    samples = flow.rollout(state, now, times)
    backup = flow.evaluate(state, now)[0][2:]
    rows = []

    def add(kind, name, when, margin, gradient, sensitivity):
        coefficients = (gradient @ sensitivity)[2:]
        rows.append(
            dict(
                kind=kind,
                object_id=name,
                time=float(when),
                margin=float(margin),
                coefficients=coefficients.tolist(),
                lower_bound=float(coefficients @ backup - controller.alpha * margin),
            )
        )

    for when in times:
        y, sensitivity = samples[float(when)]
        names, centers, radii, velocities, growth = flow.geometry(when)
        delta = y[:2] - centers
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance < 1e-09):
            raise ValueError("backup position coincides with hazard center")
        normals = delta / distance[:, None]
        for i, name in enumerate(names):
            margin = distance[i] - radii[i] - flow.radius
            add(
                "predictive" if i < len(family) else "static_backup",
                name,
                when,
                margin,
                np.r_[normals[i], 0.0, 0.0],
                sensitivity,
            )
            if when == endpoint:
                add(
                    "terminal_clearance",
                    name,
                    when,
                    margin - controller.retreat_config.terminal_clearance,
                    np.r_[normals[i], 0.0, 0.0],
                    sensitivity,
                )
                if i >= len(family):
                    expanded = radii[i] + flow.radius
                    h = distance[i] ** 2 - expanded**2
                    psi = 2 * delta[i] @ y[2:] + controller.alpha * h
                    gradient = np.r_[2 * y[2:] + 2 * controller.alpha * delta[i], 2 * delta[i]]
                    add("terminal_static_hocbf", name, when, psi, gradient, sensitivity)
                    continue
                relative = y[2:] - velocities[i]
                rate = normals[i] @ relative - growth[i]
                grad_pos = (np.eye(2) - np.outer(normals[i], normals[i])) @ relative / distance[i]
                add(
                    "terminal_separation",
                    name,
                    when,
                    rate,
                    np.r_[grad_pos, normals[i]],
                    sensitivity,
                )
    minimum = min((r["margin"] for r in rows if r["kind"] in {"predictive", "static_backup"}))
    return (rows, minimum, None)
