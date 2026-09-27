from itertools import combinations

import numpy as np
import osqp
from scipy import sparse


def solve_acceleration(nominal, rows, max_accel):
    a = np.array([row["coefficients"] for row in rows], dtype=float).reshape(-1, 2)
    lower = np.array([row["lower_bound"] for row in rows], dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(lower)):
        return (None, "invalid_qp_rows")
    full_a = np.vstack((a, np.eye(2)))
    full_lower = np.concatenate((lower, np.full(2, -max_accel)))
    full_upper = np.concatenate((np.full(len(rows), np.inf), np.full(2, max_accel)))
    try:
        problem = osqp.OSQP()
        problem.setup(
            P=sparse.eye(2, format="csc"),
            q=-nominal,
            A=sparse.csc_matrix(full_a),
            l=full_lower,
            u=full_upper,
            verbose=False,
            eps_abs=1e-07,
            eps_rel=1e-07,
            max_iter=5000,
            polishing=False,
        )
        result = problem.solve(raise_error=False)
        if result.info.status_val != 1 or result.x is None or (not np.all(np.isfinite(result.x))):
            return (None, f"qp_{str(result.info.status).replace(' ', '_')}")
        control = np.clip(np.asarray(result.x), -max_accel, max_accel)
        if np.any(a @ control < lower - 1e-05):
            return (None, "qp_residual_rejected")
        return (control, "solved")
    except Exception as exc:
        return (None, f"qp_error_{type(exc).__name__}")


def project_velocity(reference, rows, max_speed):
    reference = np.asarray(reference, dtype=float)
    if (
        reference.shape != (2,)
        or not np.isfinite(reference).all()
        or (not np.isfinite(max_speed))
        or (max_speed <= 0)
    ):
        raise ValueError("Invalid velocity projection inputs")
    A = np.asarray(
        [row["coefficients"] for row in rows] + [[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float
    )
    b = np.asarray([row["lower_bound"] for row in rows] + [-max_speed] * 4)
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Invalid velocity projection rows")
    clipped = np.clip(reference, -max_speed, max_speed)
    if np.all(A @ clipped >= b - 1e-10):
        return clipped
    candidates = [reference]
    for normal, lower in zip(A, b, strict=True):
        norm = float(normal @ normal)
        if norm > 1e-24:
            candidates.append(reference + normal * (lower - normal @ reference) / norm)
    for i, j in combinations(range(len(A)), 2):
        pair = A[[i, j]]
        if abs(np.linalg.det(pair)) > 1e-12:
            candidates.append(np.linalg.solve(pair, b[[i, j]]))
    feasible = [v for v in candidates if np.all(A @ v >= b - 1e-10)]
    if not feasible:
        raise ValueError("infeasible velocity-reference projection")
    result = min(feasible, key=lambda v: float(np.sum((v - reference) ** 2)))
    return result.copy()
