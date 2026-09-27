import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from pss.control.qp import project_velocity

from .ground_models import MeasuredGroundController, ground_motion_ready


class GroundHandoff:
    def __init__(self, world, out, static):
        self.world = world
        self.out = Path(out)
        self.static = static
        self.history = deque(maxlen=31)
        self.active = False
        self.reserve = 0.0
        self.events = []
        self.scratch = mujoco.MjData(world.model)
        self.previous = None
        self.previous_time = None
        self.latest = []
        self.last_gate = {"status": "awaiting_history"}

    def measured(self):
        w = self.world
        m = w.model
        d = self.scratch
        d.qpos[:] = w.data.qpos
        d.qvel[:] = w.data.qvel
        d.time = w.data.time
        d.mocap_pos[:] = w.data.mocap_pos
        d.mocap_quat[:] = w.data.mocap_quat
        d.eq_active[:] = w.data.eq_active
        mujoco.mj_forward(m, d)
        floor = m.geom("floor").id
        touching = set()
        for c in d.contact[: d.ncon]:
            a, b = (int(c.geom1), int(c.geom2))
            if a == floor:
                touching.add(int(m.geom_bodyid[b]))
            if b == floor:
                touching.add(int(m.geom_bodyid[a]))
        items = []
        for spec, bid in zip(
            w.scenario.dynamic_obstacles, w.dynamic_obstacle_body_ids, strict=True
        ):
            gid = next((g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid))
            rot = d.geom_xmat[gid].reshape(3, 3)
            size = m.geom_size[gid]
            sphere = spec.shape == "sphere"
            if sphere:
                radius = float(size[0])
            else:
                corners = (
                    np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]) * size
                )
                radius = float(np.linalg.norm((corners @ rot.T)[:, :2], axis=1).max())
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
            items.append(
                dict(
                    object_id=spec.name,
                    center=d.geom_xpos[gid, :2].copy(),
                    radius=radius,
                    grounded=bid in touching,
                    low_pose=bool(sphere or np.abs(rot[2]) @ size <= np.median(size) + 0.02),
                    linear_speed=float(np.linalg.norm(v[3:])),
                    angular_speed=float(np.linalg.norm(v[:3])),
                    velocity=v[3:5].copy(),
                    sphere=sphere,
                )
            )
        return items

    def update(self, controller, state, now):
        measured = self.measured()
        expected = [o.name for o in self.world.scenario.dynamic_obstacles]
        if not expected or [o["object_id"] for o in measured] != expected:
            raise ValueError("Incomplete or reordered native ground inventory")
        if self.history and now <= self.history[-1][0]:
            raise ValueError("Non-increasing ground observation time")
        for o in measured:
            if (
                not np.all(
                    np.isfinite(
                        np.r_[
                            o["center"],
                            o["velocity"],
                            o["radius"],
                            o["linear_speed"],
                            o["angular_speed"],
                        ]
                    )
                )
                or o["radius"] < 0
            ):
                raise ValueError("Invalid native ground measurement")
        self.latest = measured
        self.history.append((now, measured))
        if len(self.history) < 31 or now - self.history[0][0] < 0.6 - 1e-08:
            return controller
        quiet = {
            i
            for i in range(len(measured))
            if all(
                (
                    o[i]["grounded"]
                    and o[i]["linear_speed"] < 0.02
                    and (o[i]["angular_speed"] < 0.08)
                    for _, o in self.history
                )
            )
        }
        eligible = all((o["grounded"] for _, items in self.history for o in items))
        eligible = eligible and all((o["low_pose"] for _, items in self.history for o in items))
        eligible = eligible and all((i in quiet or o["sphere"] for i, o in enumerate(measured)))
        eligible = eligible and ground_motion_ready(measured, self.history, quiet)
        self.last_gate = {
            "time": now,
            "eligible": eligible,
            "quiet_ids": [measured[i]["object_id"] for i in quiet],
            "status": "candidate_pending" if eligible else "premises_not_met",
        }
        if not eligible:
            if self.active:
                raise RuntimeError("Grounded complete-inventory premise revoked; stopped episode")
            return controller
        static = [
            dict(object_id=f"known_static_{i}", center=np.asarray(p), radius=r)
            for i, (p, r) in enumerate(self.static)
        ]
        static.extend(
            (
                dict(object_id=o["object_id"], center=o["center"], radius=o["radius"])
                for i, o in enumerate(measured)
                if i in quiet
            )
        )
        moving = [o for i, o in enumerate(measured) if i not in quiet]
        if self.previous is not None:
            dt = now - self.previous_time
            previous = {o["object_id"]: o for o in self.previous}
            if any(
                (
                    np.linalg.norm(o["velocity"] - previous[o["object_id"]]["velocity"]) / dt > 0.5
                    for o in moving
                )
            ):
                if self.active:
                    raise RuntimeError("Declared ground acceleration bound exceeded")
                return controller
        candidate = MeasuredGroundController(
            robot_radius=controller.robot_radius,
            clearance=controller.clearance,
            max_accel=controller.max_accel,
            brake_gain=controller.brake_gain,
            sample_dt=controller.sample_dt,
            alpha=controller.alpha,
            terminal_speed=controller.terminal_speed,
            inactive=True,
            static_obstacles=static,
            moving=moving,
        )
        _, check = candidate.command(state, np.zeros(2), now)
        self.last_gate.update(
            status="committed" if check.get("feasible") else "candidate_rejected",
            candidate_check=check,
        )
        if not check.get("feasible"):
            if self.active:
                raise RuntimeError("Ground candidate failed membership/QP; stopped episode")
            return controller
        if not self.active:
            receipt = dict(
                time=now,
                kind="empirical_native_ground_handoff",
                prior=controller.snapshot(),
                candidate=candidate.snapshot(),
                check=check,
                measurement="complete native geometry, direct floor contacts, low settled pose and receding rolling bodies; 0.6 seconds history",
                assumptions="settled solids remain inert; moving spheres have acceleration at most 0.5 m/s^2",
                no_waypoints=True,
            )
            self.events.append(receipt)
            self.out.joinpath("ground_handoff.json").write_text(
                json.dumps(receipt, indent=2, default=lambda x: x.tolist())
            )
        self.active = True
        self.previous = measured
        self.previous_time = now
        return candidate

    def project(self, state, reference, previous_command, controller):
        self.reserve = max(self.reserve, float(np.linalg.norm(previous_command - state[2:])))
        snapshot = controller.snapshot()
        obstacles = snapshot["static_obstacles"] + snapshot["moving_ground_obstacles"]
        rows = []
        for o in obstacles:
            delta = state[:2] - np.asarray(o["center"])
            normal = 2 * delta
            h = float(
                delta @ delta - (o["radius"] + controller.robot_radius + controller.clearance) ** 2
            )
            lower = (
                -controller.alpha * h
                + float(normal @ np.asarray(o.get("velocity", [0, 0])))
                + np.linalg.norm(normal) * self.reserve
            )
            rows.append(
                dict(
                    object_id=o["object_id"], coefficients=normal.tolist(), lower_bound=float(lower)
                )
            )
        result = project_velocity(reference, rows, 0.5)
        return (
            result.copy(),
            dict(
                reference_before_projection=reference.tolist(),
                reference_after_projection=result.tolist(),
                reference_velocity_rows=rows,
                tracking_error_reserve=self.reserve,
                tracking_reserve_status="empirical_not_certified",
            ),
        )


__all__ = ["GroundHandoff", "MeasuredGroundController", "ground_motion_ready"]
