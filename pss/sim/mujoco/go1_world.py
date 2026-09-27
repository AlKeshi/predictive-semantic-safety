from __future__ import annotations

import numpy as np

from pss.data.episode import rot_to_quat
from pss.scenarios.spec import ScenarioSpec
from pss.sim.mujoco.go1_episode import Go1EpisodeMixin
from pss.sim.mujoco.go1_policy import Go1OnnxPolicy
from pss.sim.mujoco.go1_rollout import yaw_from_quaternion
from pss.sim.mujoco.go1_scene import _go1_assets, _go1_scene_xml, _require_mujoco
from pss.sim.mujoco.go1_sensors import Go1SensorsMixin
from pss.sim.mujoco.go1_types import Go1WorldConfig


class Go1MujocoWorld(Go1EpisodeMixin, Go1SensorsMixin):
    def __init__(
        self, scenario: ScenarioSpec, config: Go1WorldConfig | None = None, *, scene_xml=None
    ):
        self.scenario = scenario
        self.config = config or Go1WorldConfig()
        self.mj = _require_mujoco()
        self.model = self.mj.MjModel.from_xml_string(
            _go1_scene_xml(scenario, self.config) if scene_xml is None else scene_xml,
            assets=_go1_assets(full_body_collisions=self.config.full_body_collisions),
        )
        self.data = self.mj.MjData(self.model)
        self.policy = Go1OnnxPolicy()
        self.stack_body_ids = [self.model.body(f"stack_{obj.name}").id for obj in scenario.objects]
        self._stack_body_id_set = frozenset(self.stack_body_ids)
        self._stack_body_names = {
            body_id: obj.name
            for body_id, obj in zip(self.stack_body_ids, scenario.objects, strict=True)
        }
        self.dynamic_obstacle_body_ids = [
            self.model.body(f"dynamic_obstacle_{obstacle.name}").id
            for obstacle in scenario.dynamic_obstacles
        ]
        self.scene_prop_body_ids = [
            self.model.body(f"scene_prop_{prop.name}").id for prop in scenario.scene_props
        ]
        self._dynamic_obstacle_by_body_id = {
            body_id: obstacle
            for body_id, obstacle in zip(
                self.dynamic_obstacle_body_ids, scenario.dynamic_obstacles, strict=True
            )
        }
        self.floor_geom_id = int(self.model.geom("floor").id)
        self.table_body_ids = [
            int(self.model.body("table" if index == 0 else f"table_{index}").id)
            for index in range(len(scenario.tables))
        ]
        self._table_body_id_set = frozenset(self.table_body_ids)
        excluded = {
            0,
            *self.table_body_ids,
            *self.stack_body_ids,
            *self.dynamic_obstacle_body_ids,
            *self.scene_prop_body_ids,
        }
        self.robot_body_ids = {
            body_id for body_id in range(self.model.nbody) if body_id not in excluded
        }
        self.top_body_id = self.stack_body_ids[-1] if self.stack_body_ids else None
        self.ego_camera_id = self.model.camera("ego_rgbd").id
        self.model.cam_fovy[self.ego_camera_id] = self.config.rgbd_fovy_deg
        self.omnivla_camera_id = self.model.camera("omnivla_ego").id
        self.model.cam_fovy[self.omnivla_camera_id] = self.config.rgbd_fovy_deg
        self._apply_scene_camera_mount_overrides()
        self._rgbd_camera = None
        self._forward_rgbd_camera = None
        self._omnivla_rgb_camera = None

    def planar_state(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]],
            dtype=float,
        )

    def close(self) -> None:
        if self._rgbd_camera is not None:
            self._rgbd_camera.close()
            self._rgbd_camera = None
        if self._forward_rgbd_camera is not None:
            self._forward_rgbd_camera.close()
            self._forward_rgbd_camera = None
        if self._omnivla_rgb_camera is not None:
            self._omnivla_rgb_camera.close()
            self._omnivla_rgb_camera = None

    def _apply_scene_camera_mount_overrides(self) -> None:
        spec = {"position": (0.42, 0.0, 0.42), "pitch_down_deg": 24.0, "fovy_deg": 80.0}
        override = self.scenario.meta.get("omnivla_camera_mount")
        if isinstance(override, dict):
            spec.update(override)
        if "position" in spec:
            self.model.cam_pos[self.omnivla_camera_id] = np.asarray(spec["position"], dtype=float)
        pitch_down = np.deg2rad(float(spec.get("pitch_down_deg", 0.0)))
        if "fovy_deg" in spec:
            self.model.cam_fovy[self.omnivla_camera_id] = float(spec["fovy_deg"])
        image_right = np.array([0.0, -1.0, 0.0])
        camera_z = np.array([-np.cos(pitch_down), 0.0, np.sin(pitch_down)])
        image_up = np.cross(camera_z, image_right)
        camera_xmat = np.column_stack([image_right, image_up, camera_z])
        self.model.cam_quat[self.omnivla_camera_id] = rot_to_quat(camera_xmat)


__all__ = ["Go1MujocoWorld", "Go1OnnxPolicy", "Go1WorldConfig", "yaw_from_quaternion"]
