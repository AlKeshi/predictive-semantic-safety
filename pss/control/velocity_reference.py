import numpy as np

from .qp import project_velocity


def project_velocity_reference(
    position,
    reference,
    obstacles,
    *,
    robot_radius,
    clearance,
    alpha,
    max_speed,
    tracking_error_reserve=0.0,
):
    position = np.asarray(position, dtype=float)
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(
        [robot_radius, clearance, alpha, max_speed, tracking_error_reserve], dtype=float
    )
    if (
        position.shape != (2,)
        or reference.shape != (2,)
        or (not np.all(np.isfinite(np.r_[position, reference, values])))
        or (robot_radius < 0)
        or (clearance < 0)
        or (alpha <= 0)
        or (max_speed <= 0)
        or (tracking_error_reserve < 0)
    ):
        raise ValueError("invalid velocity-reference inputs")
    rows = []
    for item in obstacles:
        center = np.asarray(item["center"], dtype=float)
        radius = float(item["radius"])
        if (
            center.shape != (2,)
            or not np.all(np.isfinite(center))
            or (not np.isfinite(radius))
            or (radius < 0)
        ):
            raise ValueError("invalid static obstacle")
        delta = position - center
        h = float(delta @ delta - (radius + robot_radius + clearance) ** 2)
        rows.append(
            {
                "kind": "static_velocity_reference",
                "object_id": item["object_id"],
                "coefficients": (2 * delta).tolist(),
                "lower_bound": -alpha * h
                + 2 * float(np.linalg.norm(delta)) * tracking_error_reserve,
                "margin": h,
            }
        )
    result = project_velocity(reference, rows, max_speed)
    return (
        result.copy(),
        {
            "reference_before_projection": reference.tolist(),
            "reference_after_projection": result.tolist(),
            "reference_velocity_rows": rows,
            "tracking_error_reserve": float(tracking_error_reserve),
            "reference_projection_change": float(np.linalg.norm(result - reference)),
        },
    )
