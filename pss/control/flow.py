import numpy as np


def braking_flow(
    state: np.ndarray, duration: float, max_accel: float = 2.0, brake_gain: float = 2.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(state, dtype=float)
    if (
        x.shape != (4,)
        or not np.all(np.isfinite(x))
        or (not np.isfinite(duration))
        or (duration < 0)
        or (not np.isfinite(max_accel))
        or (max_accel <= 0)
        or (not np.isfinite(brake_gain))
        or (brake_gain <= 0)
    ):
        raise ValueError("invalid braking state, duration, or gains")
    speed = np.abs(x[2:])
    sign = np.sign(x[2:])
    limit = max_accel / brake_gain
    saturated_time = np.maximum((speed - limit) / max_accel, 0.0)
    saturated = duration <= saturated_time
    tail = np.maximum(duration - saturated_time, 0.0)
    decay = np.exp(-brake_gain * tail)
    initial_tail = np.minimum(speed, limit)
    velocity = initial_tail * decay
    displacement = (
        speed * saturated_time
        - 0.5 * max_accel * saturated_time**2
        + initial_tail * -np.expm1(-brake_gain * tail) / brake_gain
    )
    dpdv = np.where(
        speed > limit,
        (speed - velocity) / max_accel,
        -np.expm1(-brake_gain * duration) / brake_gain,
    )
    dvdv = decay
    velocity = np.where(saturated, speed - max_accel * duration, velocity)
    displacement = np.where(
        saturated, speed * duration - 0.5 * max_accel * duration**2, displacement
    )
    dpdv = np.where(saturated, duration, dpdv)
    dvdv = np.where(saturated, 1.0, dvdv)
    return (x[:2] + sign * displacement, sign * velocity, np.stack((dpdv, dvdv)))
