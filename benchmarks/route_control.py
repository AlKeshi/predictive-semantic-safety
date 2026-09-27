from __future__ import annotations

import numpy as np

from pss.sim.mujoco.go1_rollout import yaw_from_quaternion

from .pipeline import enforce_required_origins
from .route_geometry import _actual_hazards


class RouteControl:
    def command(self):
        goal = np.asarray(self.scenario.robot_goal) - self.state[:2]
        velocity = self.config.speed * goal / max(np.linalg.norm(goal), 1e-08)
        if self.goal_time is not None or self.now < self.config.warmup:
            velocity = np.zeros(2)
        feedback_velocity = (
            self.state[2:]
            if self.ground is not None and self.ground.active
            else self.command_velocity
        )
        nominal = np.clip(2.5 * (velocity - feedback_velocity), -1.5, 1.5)
        if self.now < self.config.warmup or (
            self.transform is not None and (not self.transform.seen)
        ):
            acceleration = np.clip(-2.5 * self.state[2:], -1.5, 1.5)
            self.diagnostic = {"mode": "observing"}
        else:
            enforce_required_origins(self.controller, self.transform, self.now, self.state)
            acceleration, self.diagnostic = self.controller.command(self.state, nominal, self.now)
        self.truth[-1]["enforced_object_ids"] = sorted(
            {
                row.get("object_id")
                for row in self.diagnostic.get("enforced_rows", [])
                if row.get("object_id")
            }
        )
        mode = str(self.diagnostic.get("mode", self.diagnostic.get("status", "unknown")))
        self.fallback_ticks += int(
            any((k in mode.lower() for k in ("brak", "emergency", "invalid", "expired", "await")))
        )
        self.intervention_ticks += int(np.linalg.norm(acceleration - nominal) > 0.1)
        self.total_policy_ticks += 1
        previous_command = self.command_velocity.copy()
        self.command_velocity = self.command_velocity + acceleration * self.cfg.policy_dt
        if (
            self.ground is not None
            and self.ground.active
            and self.diagnostic.get("feasible", False)
        ):
            try:
                self.command_velocity, reference_diag = self.ground.project(
                    self.state, self.command_velocity, previous_command, self.controller
                )
                self.diagnostic.update(reference_diag)
            except ValueError as exc:
                self.command_velocity[:] = 0
                self.diagnostic.update(
                    status="brake:reference_projection_failed",
                    feasible=False,
                    enforced_rows=[],
                    num_constraints=0,
                    reference_error=str(exc),
                )
        desired_velocity = self.command_velocity.copy()
        if not self.diagnostic.get("feasible", False) and self.now >= self.config.warmup:
            self.command_velocity[:] = 0
            desired_velocity[:] = 0
            self.diagnostic["execution_override"] = "zero_translation_emergency_stop"
        if self.now < self.config.warmup or (
            self.transform is not None and (not self.transform.seen)
        ):
            desired_velocity = np.zeros(2)
            self.command_velocity[:] = 0
        yaw = yaw_from_quaternion(self.world.data.qpos[3:7])
        co, si = (np.cos(yaw), np.sin(yaw))
        body_velocity = np.array([[co, si], [-si, co]]) @ desired_velocity
        if self.demo_navigation:
            yaw_error = np.arctan2(
                np.sin(self.heading_reference - yaw), np.cos(self.heading_reference - yaw)
            )
            yaw_rate = float(np.clip(2.0 * yaw_error, -0.5, 0.5))
            self.diagnostic["heading_hold"] = dict(
                reference_rad=float(self.heading_reference), measured_rad=float(yaw)
            )
        else:
            yaw_rate, gaze_receipt = self.gaze.command(
                self.state[:2], yaw, self.now, self.controller.snapshot()
            )
            self.diagnostic["gaze"] = gaze_receipt
        mode = str(self.diagnostic.get("mode", self.diagnostic.get("status", "unknown")))
        self.current_command = np.r_[body_velocity, yaw_rate]
        if self.external is not None:
            self.current_command, extra = self.external.command(
                self.observation.rgb,
                self.state,
                yaw,
                np.asarray(self.scenario.robot_goal),
                self.now,
                self.config.warmup,
                self.goal_time is not None,
            )
            self.diagnostic = dict(
                status="external_policy", feasible=True, enforced_rows=[], **extra
            )
            mode = "external_policy"
            self.external_records.append(dict(time=self.now, **extra))
        self.world.policy.apply(self.world.model, self.world.data, self.current_command)
        self.trace.append(
            {
                "time": self.now,
                "state": self.state,
                "command": self.current_command,
                "nominal_accel": nominal,
                "filtered_reference_accel": None if self.external is not None else acceleration,
                "command_world_velocity": None
                if self.external is not None
                else self.command_velocity.copy(),
                "previous_command_velocity": previous_command.copy(),
                "mode": mode,
                "diagnostic": self.diagnostic,
                "clearance_m": self.clearance,
            }
        )

    def control_tick(self):
        self.state = self.world.planar_state()
        self.truth.append({"time": self.now, "objects": _actual_hazards(self.world)})
        self.clearance = self.clearance_audit.sample(
            self.world, self.robot_geoms, self.hazard_geoms
        )
        self.minimum_height = min(self.minimum_height, float(self.world.data.qpos[2]))
        if self.last_xy is not None:
            self.path_length += float(np.linalg.norm(self.state[:2] - self.last_xy))
        self.last_xy = self.state[:2].copy()
        distance = float(np.linalg.norm(np.asarray(self.scenario.robot_goal) - self.state[:2]))
        if distance < 0.25 and self.goal_time is None:
            self.goal_time = self.now
        if self.ground is not None:
            self.controller = self.ground.update(self.controller, self.state, self.now)
        if (
            self.now >= self.next_update - 1e-09
            and self.config.method != "nominal"
            and (not (self.ground is not None and self.ground.active))
        ):
            self.forecast()
        self.command()
