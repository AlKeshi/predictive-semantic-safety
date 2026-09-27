import math

import numpy as np

from .emergency import fresh_ground_fallback
from .snapshot import waypoint


class Control:
    def control_tick(self):
        self.state = np.r_[self.data.qpos[:2], self.data.qvel[:2]]
        self.min_height = min(self.min_height, float(self.data.qpos[2]))
        if self.semantic_enabled:
            if self.now >= self.next_update - 1e-08 and len(self.ground.grounded) < len(self.names):
                self.forecast()
            if self.now >= self.next_ground - 1e-08:
                self.ground_update()
            if (
                len(self.ground.grounded) == len(self.names)
                and self.live_family
                and (self.now >= self.next_plan - 1e-08)
            ):
                planned, self.navigation_receipt = waypoint(
                    self.state[:2], self.goal, self.live_family, self.static
                )
                self.navigation_receipt["planning_time"] = self.now
                self.navigation_target = self.goal.copy() if planned is None else planned
                self.next_plan = self.now + 0.5
            delta = self.navigation_target - self.state[:2]
            desired = 1.1 * delta
            desired *= min(1.0, self.speed / max(np.linalg.norm(desired), 1e-08))
            if self.now < 0.8:
                desired[:] = 0
            nominal = np.clip(2.5 * (desired - self.command), -1.5, 1.5)
            acceleration, diagnostic = self.controller.command(self.state, nominal, self.now)
            acceleration, diagnostic = fresh_ground_fallback(
                self.controller,
                self.state,
                self.now,
                nominal,
                acceleration,
                diagnostic,
                self.live_family,
                self.live_sources,
            )
            self.command += acceleration * 0.02
            if self.now < 0.8 or (
                not diagnostic["feasible"] and (not diagnostic["execute_backup"])
            ):
                self.command[:] = 0
                diagnostic["execution_override"] = "zero_translation_emergency_stop"
        else:
            desired = 1.1 * (self.goal - self.state[:2])
            desired *= min(1.0, self.speed / max(np.linalg.norm(desired), 1e-08))
            acceleration = np.clip((desired - self.command) / 0.02, -0.9, 0.9)
            nominal = acceleration.copy()
            self.command += acceleration * 0.02
            diagnostic = dict(
                status="semantic_off",
                feasible=False,
                execute_backup=False,
                certificate_valid=False,
                enforced_rows=[],
            )
        rotation = self.data.xmat[self.model.body("trunk").id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        yaw_rate, gaze_receipt = (
            self.gaze.command(self.state[:2], yaw, self.now, self.controller.snapshot())
            if self.semantic_enabled
            else (np.clip(-2 * yaw, -0.5, 0.5), {})
        )
        co, si = (math.cos(yaw), math.sin(yaw))
        body_command = np.r_[np.array([[co, si], [-si, co]]) @ self.command, yaw_rate]
        self.policy.apply(self.model, self.data, body_command)
        if np.linalg.norm(self.goal - self.state[:2]) < 0.2 and self.goal_time is None:
            self.goal_time = self.now
        positions = {n: self.data.xpos[self.model.body(n).id].tolist() for n in self.names}
        self.trace.append(
            dict(
                time=self.now,
                state=self.state.tolist(),
                command=self.command.tolist(),
                body_command=body_command.tolist(),
                nominal=nominal.tolist(),
                acceleration=acceleration.tolist(),
                diagnostic=diagnostic,
                gaze=gaze_receipt,
                navigation_target=self.navigation_target.tolist(),
                navigation=self.navigation_receipt,
            )
        )
        self.rows.append(
            dict(
                time=self.now,
                robot=self.state.tolist(),
                command=self.command.tolist(),
                positions=positions,
                phase=diagnostic["status"],
            )
        )
