import numpy as np

from pss.control.code_world import CodeWorldController


class MeasuredGroundController(CodeWorldController):
    def __init__(self, *, moving, **kwargs):
        super().__init__(**kwargs)
        from copy import deepcopy

        self.moving = deepcopy(moving)

    def _constraints(self, x, now, family, grids):
        rows, minimum, terminal = super()._constraints(x, now, family, grids)
        margins = [] if minimum is None else [minimum]
        for item in self.moving:
            delta = x[:2] - item["center"]
            relative = x[2:] - item["velocity"]
            expanded = item["radius"] + self.robot_radius + self.clearance
            h = float(delta @ delta - expanded**2)
            hdot = float(2 * delta @ relative)
            lower = (
                -2 * float(relative @ relative)
                - 2 * self.alpha * hdot
                - self.alpha**2 * h
                + 2 * np.linalg.norm(delta) * 0.5
            )
            margin = float(np.linalg.norm(delta) - expanded)
            rows.append(
                dict(
                    kind="moving_ground_hocbf",
                    object_id=item["object_id"],
                    time=now,
                    margin=margin,
                    hocbf_membership=hdot + self.alpha * h,
                    coefficients=(2 * delta).tolist(),
                    lower_bound=float(lower),
                    obstacle_velocity=item["velocity"].tolist(),
                    assumed_obstacle_acceleration_bound_mps2=0.5,
                )
            )
            margins.append(margin)
        return (rows, min(margins) if margins else None, terminal)

    def snapshot(self):
        from copy import deepcopy

        result = super().snapshot()
        result["moving_ground_obstacles"] = deepcopy(self.moving)
        return result


def ground_motion_ready(measured, history, quiet):
    moving = [i for i, o in enumerate(measured) if i not in quiet]
    for i in moving:
        ball = measured[i]
        for j in quiet:
            solid = measured[j]
            if (
                np.dot(ball["center"] - solid["center"], ball["velocity"] - solid["velocity"])
                < -1e-05
            ):
                return False
        samples = list(history)
        for (t0, a), (t1, b) in zip(samples[:-1], samples[1:], strict=False):
            if t1 <= t0 or np.linalg.norm(b[i]["velocity"] - a[i]["velocity"]) / (t1 - t0) > 0.5:
                return False
    return True
