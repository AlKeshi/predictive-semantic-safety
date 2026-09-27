from __future__ import annotations

import numpy as np

from pss.control.velocity_reference import project_velocity_reference

from .pipeline import enforce_required_origins


class CeilingControl:
    def control_tick(self):
        if np.linalg.norm(self.state[:2] - self.goal) < 0.25:
            self.reached_at = self.now if self.reached_at is None else self.reached_at
        delta = self.goal - self.state[:2]
        desired_velocity = self.config.speed * delta / max(np.linalg.norm(delta), 0.25)
        if self.reached_at is not None:
            desired_velocity[:] = 0
        feedback = self.state[2:] if self.demo_navigation else self.command_velocity
        nominal = np.clip(2.5 * (desired_velocity - feedback), -1.5, 1.5)
        if self.now < self.case["warmup"] or (
            self.config.method == "cw"
            and (not self.active)
            and (not self.secure_observation or self.now - self.last_secure > 1.5)
        ):
            accel = np.clip(
                -self.controller.brake_gain * self.state[2:],
                -self.controller.max_accel,
                self.controller.max_accel,
            )
            diag = {
                "status": "observation_warmup_or_unresolved",
                "feasible": False,
                "enforced_rows": [],
                "intervened": False,
            }
        else:
            enforce_required_origins(self.controller, self.transform, self.now, self.state)
            accel, diag = self.controller.command(self.state, nominal, self.now)
        if self.active and diag.get("intervened") and (self.first_intervention is None):
            self.first_intervention = self.now
        if self.ground_handoff:
            self.tracking_error_reserve = max(
                self.tracking_error_reserve,
                float(np.linalg.norm(self.command_velocity - self.state[2:])),
            )
        previous_command = self.command_velocity.copy()
        self.command_velocity = np.clip(self.command_velocity + 0.02 * accel, -0.5, 0.5)
        if not diag["feasible"]:
            self.command_velocity *= 0.9
        elif self.ground_handoff:
            try:
                self.command_velocity, reference_diag = project_velocity_reference(
                    self.state[:2],
                    self.command_velocity,
                    self.controller.snapshot()["static_obstacles"],
                    robot_radius=self.controller.robot_radius,
                    clearance=self.controller.clearance,
                    alpha=self.controller.alpha,
                    max_speed=0.5,
                    tracking_error_reserve=self.tracking_error_reserve,
                )
                diag.update(reference_diag)
            except ValueError as exc:
                self.command_velocity[:] = 0.0
                diag.update(
                    status="brake:velocity_reference_projection_failed",
                    feasible=False,
                    enforced_rows=[],
                    num_constraints=0,
                    reference_projection_error=str(exc),
                )
        trunk_rotation = self.data.xmat[self.root_robot].reshape(3, 3)
        local_velocity = trunk_rotation[:2, :2].T @ self.command_velocity
        yaw = np.arctan2(trunk_rotation[1, 0], trunk_rotation[0, 0])
        yaw_command = np.clip(-2 * yaw, -0.5, 0.5)
        applied_command = np.r_[local_velocity, yaw_command]
        if self.external is not None:
            applied_command, extra = self.external.command(
                self.rgb,
                self.state,
                yaw,
                self.goal,
                self.now,
                self.case["warmup"],
                self.reached_at is not None,
            )
            self.external_records.append(dict(time=self.now, **extra))
            diag = dict(status="external_policy", feasible=True, enforced_rows=[], **extra)
        self.policy.apply(self.model, self.data, applied_command)
        self.min_height = min(self.min_height, float(self.data.qpos[2]))
        robot_geom_ids = [
            gi
            for gi in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gi]) in self.robot_bodies
            and (self.model.geom_contype[gi] or self.model.geom_conaffinity[gi])
        ]
        self.clearance_audit.sample(
            self.audit_world, robot_geom_ids, [self.model.geom("fixture_housing").id]
        )
        self.controls.append(
            dict(
                state=self.state.tolist(),
                applied_command=applied_command.tolist(),
                desired_velocity=desired_velocity.tolist(),
                previous_command_velocity=previous_command.tolist(),
                command_velocity=self.command_velocity.tolist(),
                active=self.active,
                settled=self.settled,
                support_until=None
                if not np.isfinite(self.last_valid_support)
                else self.last_valid_support,
                nominal_accel=nominal.tolist(),
                **{**diag, "time": self.now},
            )
        )
        self.truth.append(
            {
                "time": self.now,
                "fixture": self.data.xpos[self.physics.body].tolist(),
                "robot": self.data.qpos[:7].tolist(),
                "fixture_quaternion": self.data.xquat[self.physics.body].tolist(),
                "supports": self.data.eq_active[self.physics.ids].tolist(),
            }
        )
