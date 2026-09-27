from __future__ import annotations

import numpy as np

from pss.sim.mujoco.go1_policy import _HOME_ANGLES


class Go1EpisodeMixin:
    def prepare_online_episode(self) -> dict[str, object]:
        if self.scenario.objects or self.config.stack_settle_time:
            raise ValueError("Use dynamic obstacles for the supported native-contact scenarios")
        cfg = self.config
        data = self.data
        mj = self.mj
        initial_yaw = float(self.scenario.meta.get("robot_initial_yaw_rad", 0.0))
        if not np.isfinite(initial_yaw):
            raise ValueError("Declared initial robot yaw must be finite")
        mj.mj_resetData(self.model, data)
        data.qpos[:3] = [*self.scenario.robot_start, 0.278]
        data.qpos[3:7] = [np.cos(initial_yaw / 2), 0.0, 0.0, np.sin(initial_yaw / 2)]
        data.qpos[7:19] = _HOME_ANGLES
        data.ctrl[:] = _HOME_ANGLES
        mj.mj_forward(self.model, data)
        self.policy.reset()
        stack_state: list[tuple[slice, slice, np.ndarray]] = []
        for body_id in self.stack_body_ids:
            joint_id = int(self.model.body_jntadr[body_id])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            stack_state.append(
                (
                    slice(qpos_adr, qpos_adr + 7),
                    slice(dof_adr, dof_adr + 6),
                    data.qpos[qpos_adr : qpos_adr + 7].copy(),
                )
            )
        dynamic_obstacle_state: list[tuple[int, slice, slice, np.ndarray]] = []
        for body_id in self.dynamic_obstacle_body_ids:
            joint_id = int(self.model.body_jntadr[body_id])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            dynamic_obstacle_state.append(
                (
                    int(body_id),
                    slice(qpos_adr, qpos_adr + 7),
                    slice(dof_adr, dof_adr + 6),
                    data.qpos[qpos_adr : qpos_adr + 7].copy(),
                )
            )

        def restore_initial_stack() -> None:
            for qpos_slice, qvel_slice, initial_qpos in stack_state:
                data.qpos[qpos_slice] = initial_qpos
                data.qvel[qvel_slice] = 0.0

        def restore_initial_dynamic_obstacles() -> None:
            for _, qpos_slice, qvel_slice, initial_qpos in dynamic_obstacle_state:
                data.qpos[qpos_slice] = initial_qpos
                data.qvel[qvel_slice] = 0.0

        substeps = round(cfg.policy_dt / cfg.sim_dt)
        for step in range(int(np.ceil(cfg.settle_time / cfg.sim_dt))):
            if step % substeps == 0:
                self.policy.apply(self.model, data, np.zeros(3))
            data.qfrc_applied[:] = 0.0
            mj.mj_step(self.model, data)
            if cfg.hold_stack_during_settle:
                restore_initial_stack()
            restore_initial_dynamic_obstacles()
        if cfg.hold_stack_during_settle:
            restore_initial_stack()
        restore_initial_dynamic_obstacles()
        mj.mj_forward(self.model, data)
        details: dict[str, object] = {
            "robot_settle_time_s": float(cfg.settle_time),
            "stack_held_during_robot_settle": bool(cfg.hold_stack_during_settle),
            "free_stack_settle_time_s": float(cfg.stack_settle_time),
            "quiescence_gate_enabled": bool(cfg.reject_nonquiescent_stack),
        }
        restore_initial_dynamic_obstacles()
        mj.mj_forward(self.model, data)
        data.time = 0.0
        self._activated_dynamic_obstacle_body_ids = set()
        self._dynamic_obstacle_initial_state = {
            body_id: (qpos_slice, qvel_slice, initial_qpos.copy())
            for body_id, qpos_slice, qvel_slice, initial_qpos in dynamic_obstacle_state
        }
        self._apply_dynamic_obstacle_initial_velocities()
        return details

    def _apply_dynamic_obstacle_initial_velocities(self) -> None:
        for body_id, obstacle in self._dynamic_obstacle_by_body_id.items():
            if obstacle.activation_distance is not None:
                continue
            self._apply_dynamic_obstacle_velocity(body_id, obstacle)

    def apply_dynamic_obstacle_triggers(self) -> bool:
        activated = False
        active = getattr(self, "_activated_dynamic_obstacle_body_ids", set())
        self._activated_dynamic_obstacle_body_ids = active
        robot_xy = np.asarray(self.data.qpos[:2], dtype=float)
        for body_id, obstacle in self._dynamic_obstacle_by_body_id.items():
            if obstacle.activation_distance is None or body_id in active:
                continue
            center = np.asarray(obstacle.activation_center or obstacle.position[:2], dtype=float)
            launch = self.scenario.meta.get("timed_launches", {}).get(obstacle.name)
            waiting = (
                self.data.time + 1e-09 < launch
                if launch is not None
                else float(np.linalg.norm(robot_xy - center)) > float(obstacle.activation_distance)
            )
            if waiting:
                self._hold_dynamic_obstacle_at_initial_state(body_id)
                continue
            self._apply_dynamic_obstacle_velocity(body_id, obstacle)
            active.add(body_id)
            activated = True
        if activated:
            self.mj.mj_forward(self.model, self.data)
        return activated

    def _apply_dynamic_obstacle_velocity(self, body_id: int, obstacle) -> None:
        joint_id = int(self.model.body_jntadr[body_id])
        qvel_adr = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[qvel_adr : qvel_adr + 3] = np.asarray(obstacle.linear_velocity, dtype=float)
        self.data.qvel[qvel_adr + 3 : qvel_adr + 6] = np.asarray(
            obstacle.angular_velocity, dtype=float
        )

    def _hold_dynamic_obstacle_at_initial_state(self, body_id: int) -> None:
        state = getattr(self, "_dynamic_obstacle_initial_state", {}).get(body_id)
        if state is None:
            return
        qpos_slice, qvel_slice, initial_qpos = state
        self.data.qpos[qpos_slice] = initial_qpos
        self.data.qvel[qvel_slice] = 0.0
        self.mj.mj_forward(self.model, self.data)
