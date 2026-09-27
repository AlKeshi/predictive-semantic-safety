from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecoveryConfig:
    clearance_band: float = 0.35
    retreat_speed: float = 0.2
    velocity_gain: float = 2.5
    weight_scale: float = 0.15

    def __post_init__(self):
        values = np.array(list(vars(self).values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("recovery settings must be finite and positive")


def recovery_objective(state, nominal, occupancies, config=None):
    config = RecoveryConfig() if config is None else config
    x = np.asarray(state, dtype=float)
    desired = np.asarray(nominal, dtype=float)
    if x.shape != (4,) or desired.shape != (2,) or (not np.all(np.isfinite(x))):
        raise ValueError("invalid recovery state or objective")
    if not np.all(np.isfinite(desired)):
        raise ValueError("nonfinite recovery objective")
    margins, directions = ([], [])
    for occupancy in occupancies:
        samples = np.asarray(occupancy, dtype=float)
        if (
            samples.ndim != 2
            or samples.shape[1] != 3
            or (not len(samples))
            or (not np.all(np.isfinite(samples)))
            or np.any(samples[:, 2] < 0)
        ):
            raise ValueError("invalid recovery occupancy")
        delta = x[:2] - samples[:, :2]
        distances = np.linalg.norm(delta, axis=1)
        clearance = distances - samples[:, 2]
        index = int(np.argmin(clearance))
        if distances[index] < 1e-10:
            raise ValueError("undefined outward direction at hazard center")
        margins.append(float(clearance[index]))
        directions.append(delta[index] / distances[index])
    if not margins:
        return (desired.copy(), {"active": False, "blend": 0.0})
    minimum = min(margins)
    proximity = float(np.clip(1.0 - minimum / config.clearance_band, 0.0, 1.0))
    blend = proximity**2 * (3.0 - 2.0 * proximity)
    weights = np.exp(-(np.asarray(margins) - minimum) / config.weight_scale)
    weights /= weights.sum()
    velocity = config.retreat_speed * (weights @ np.asarray(directions))
    reference = config.velocity_gain * (velocity - x[2:])
    objective = (1.0 - blend) * desired + blend * reference
    return (
        objective,
        {
            "active": blend > 0.0,
            "blend": blend,
            "min_clearance_m": minimum,
            "hazard_weights": weights.tolist(),
            "outward_velocity_mps": velocity.tolist(),
            "goal_acceleration": desired.tolist(),
            "objective_acceleration": objective.tolist(),
            "settings": dict(vars(config)),
            "schema": "occupancy_recovery_objective_v1_uncalibrated",
        },
    )
