from __future__ import annotations

import numpy as np

from .backup_cbf import admit_current_geometry
from .geometry import geom_bounds, update_geometry_measurement


class ReactiveControl:
    def control_tick(self):
        state = (
            np.r_[self.data.qpos[:2], self.data.qvel[:2]]
            if self.physics
            else self.world.planar_state()
        )
        if self.goal_time is None and np.linalg.norm(state[:2] - self.goal) < 0.25:
            self.goal_time = self.now
        update_geometry_measurement(self.model, self.data, self.geometry_data)
        obstacle_records = []
        for g in self.candidates:
            self.g = g
            center, radius, bottom, self.corners = geom_bounds(
                self.model, self.geometry_data, self.g
            )
            if (
                self.g in self.hazard_geoms
                and admit_current_geometry(bottom)
                and (self.g not in self.active)
            ):
                self.active.add(self.g)
                self.admissions.append(
                    {
                        "time": self.now,
                        "object_id": self.model.geom(self.g).name,
                        "reason": "current_solid_intersects_robot_height_band",
                        "bottom": bottom,
                    }
                )
            if self.g not in self.active:
                continue
            if self.g in self.active:
                obstacle_records.append(
                    {
                        "object_id": self.model.geom(self.g).name,
                        "center": center[:2].tolist(),
                        "radius": radius,
                        "z": float(center[2]),
                        "bottom": bottom,
                        "polygon": self.known_polygons.get(self.g),
                        "dynamic": self.g in self.hazard_geoms,
                    }
                )
        desired = (
            self.case["speed"]
            * (self.goal - state[:2])
            / max(np.linalg.norm(self.goal - state[:2]), 0.25 if self.physics else 1e-08)
        )
        if self.goal_time is not None or self.now < self.case["warmup"]:
            desired[:] = 0.0
        feedback = state[2:] if self.demo_profile and self.physics else self.command_velocity
        nominal = np.clip(2.5 * (desired - feedback), -1.5, 1.5)
        if self.now < self.case["warmup"]:
            acceleration = np.clip(-2.5 * state[2:], -2.0, 2.0)
            diag = {
                "status": "common_warmup",
                "feasible": False,
                "qp_attempted": False,
                "membership_valid": True,
                "enforced_rows": [],
                "candidate_rows": [],
            }
        else:
            acceleration, diag = self.controller.command(state, nominal, self.now, obstacle_records)
        previous_command = self.command_velocity.copy()
        self.command_velocity += self.policy_dt * acceleration
        if self.physics:
            self.command_velocity = np.clip(self.command_velocity, -0.5, 0.5)
            if not diag["feasible"]:
                self.command_velocity *= 0.9
        elif not diag["feasible"]:
            self.command_velocity[:] = 0.0
            diag["execution_override"] = "zero_translation_emergency_stop"
        if self.now < self.case["warmup"]:
            self.command_velocity[:] = 0.0
        rot = self.data.xmat[self.root].reshape(3, 3)
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        co, si = (np.cos(yaw), np.sin(yaw))
        command = np.r_[
            np.array([[co, si], [-si, co]]) @ self.command_velocity,
            np.clip(
                -2 * (yaw - (self.case.get("yaw_rad", 0.0) if self.demo_profile else 0.0)),
                -0.5,
                0.5,
            ),
        ]
        self.policy.apply(self.model, self.data, command)
        self.trace.append(
            {
                "time": self.now,
                "state": state.tolist(),
                "command": command.tolist(),
                "nominal_accel": nominal.tolist(),
                "desired_velocity": desired.tolist(),
                "previous_command_velocity": previous_command.tolist(),
                "filtered_reference_accel": acceleration.tolist(),
                "command_world_velocity": self.command_velocity.tolist(),
                "diagnostic": diag,
                "mode": diag["status"],
                "obstacles": obstacle_records,
                "robot_z": float(self.data.qpos[2]),
            }
        )
        self.truth.append(
            {
                "time": self.now,
                "objects": {
                    self.model.geom(g).name: self.geometry_data.geom_xpos[g].tolist()
                    for g in sorted(self.hazard_geoms)
                },
            }
        )
        if self.step % self.substeps == 0:
            self.state_times.append(self.now)
            self.qpos_log.append(self.data.qpos.copy())
            self.qvel_log.append(self.data.qvel.copy())
        if self.step == 0:
            self.snapshots = [
                {
                    "geom_id": g,
                    "name": self.model.geom(g).name,
                    "center": geom_bounds(self.model, self.data, g)[0].tolist(),
                    "radius": geom_bounds(self.model, self.data, g)[1],
                    "corners": None
                    if geom_bounds(self.model, self.data, g)[3] is None
                    else geom_bounds(self.model, self.data, g)[3].tolist(),
                }
                for g in self.candidates
            ]
