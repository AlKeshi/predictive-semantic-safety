import numpy as np

from pss.control.code_world import CodeWorldController


class PlainCBF:
    def __init__(self, robot_radius=0.403, clearance=0.05, alpha=3.0, max_accel=2.0):
        self.solver = CodeWorldController(
            robot_radius=robot_radius,
            clearance=clearance,
            alpha=alpha,
            max_accel=max_accel,
            inactive=True,
        )

    def geometry_rows(self, state, now, obstacles):
        x = np.asarray(state)
        rows = []
        for o in obstacles:
            if o.get("polygon") is not None:
                polygon = np.asarray(o["polygon"])
                candidates = []
                inside = True
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                    edge = b - a
                    length2 = float(edge @ edge)
                    fraction = float(np.clip((x[:2] - a) @ edge / length2, 0.0, 1.0))
                    point = a + fraction * edge
                    delta = x[:2] - point
                    inside &= bool(edge[0] * (x[1] - a[1]) - edge[1] * (x[0] - a[0]) >= -1e-10)
                    hessian = (
                        2 * (np.eye(2) - np.outer(edge, edge) / length2)
                        if 0.0 < fraction < 1.0
                        else 2 * np.eye(2)
                    )
                    candidates.append((float(delta @ delta), delta, hessian))
                squared, delta, hessian = min(candidates, key=lambda item: item[0])
                if inside:
                    squared = 0.0
                    delta = np.zeros(2)
                    hessian = np.zeros((2, 2))
                r = self.solver.robot_radius + self.solver.clearance
                h = squared - r * r
                gradient = 2 * delta
                hd = float(gradient @ x[2:])
                rows.append(
                    {
                        "kind": "known_polygon_hocbf",
                        "object_id": o["object_id"],
                        "coefficients": gradient.tolist(),
                        "lower_bound": float(
                            -x[2:] @ hessian @ x[2:]
                            - 2 * self.solver.alpha * hd
                            - self.solver.alpha**2 * h
                        ),
                        "h": h,
                        "psi1": hd + self.solver.alpha * h,
                        "margin": float(np.sqrt(squared) - r),
                        "time": now,
                    }
                )
                continue
            delta = x[:2] - np.asarray(o["center"])
            r = o["radius"] + self.solver.robot_radius + self.solver.clearance
            h = float(delta @ delta - r * r)
            hd = float(2 * delta @ x[2:])
            rows.append(
                {
                    "kind": "current_geometry_hocbf",
                    "object_id": o["object_id"],
                    "coefficients": (2 * delta).tolist(),
                    "lower_bound": float(
                        -2 * (x[2:] @ x[2:]) - 2 * self.solver.alpha * hd - self.solver.alpha**2 * h
                    ),
                    "h": h,
                    "psi1": hd + self.solver.alpha * h,
                    "margin": float(np.linalg.norm(delta) - r),
                    "time": now,
                }
            )
        return rows

    def command(self, state, nominal, now, obstacles):
        x = np.asarray(state)
        rows = self.geometry_rows(state, now, obstacles)
        u, status = self.solver._solve(np.asarray(nominal), rows)
        feasible = u is not None
        if not feasible:
            u = np.clip(
                -self.solver.brake_gain * x[2:], -self.solver.max_accel, self.solver.max_accel
            )
        return (
            u,
            {
                "status": status if feasible else "brake:" + status,
                "feasible": feasible,
                "qp_attempted": True,
                "qp_status": status,
                "membership_valid": all((r["h"] >= 0 and r["psi1"] >= 0 for r in rows)),
                "candidate_rows": rows,
                "enforced_rows": rows if feasible else [],
                "obstacle_count": len(obstacles),
                "forecast_count": 0,
                "no_vision": True,
                "no_prediction": True,
                "q_m": 0.0,
                "obstacle_model": "current_geometry_frozen_within_each_QP",
                "min_margin": min((r["margin"] for r in rows), default=None),
            },
        )
