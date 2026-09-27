from __future__ import annotations

import numpy as np


class Go1SensorsMixin:
    def capture_egocentric_rgb(self) -> np.ndarray:
        return self._egocentric_camera().capture_rgb(self.data)

    def capture_forward_egocentric(self):
        return self._forward_camera().capture(self.data)

    def capture_omnivla_rgb(self) -> np.ndarray:
        if self._omnivla_rgb_camera is None:
            from pss.sim.mujoco.rgbd import MujocoRGBDCamera

            self._omnivla_rgb_camera = MujocoRGBDCamera(
                self.mj,
                self.model,
                self.scenario,
                camera_name="omnivla_ego",
                width=self.config.rgbd_width,
                height=self.config.rgbd_height,
                max_depth=self.config.rgbd_max_depth,
            )
        return self._omnivla_rgb_camera.capture_rgb(self.data)

    def _forward_camera(self):
        if self._forward_rgbd_camera is None:
            from pss.sim.mujoco.rgbd import MujocoRGBDCamera

            self._forward_rgbd_camera = MujocoRGBDCamera(
                self.mj,
                self.model,
                self.scenario,
                camera_name="omnivla_ego",
                width=self.config.rgbd_width,
                height=self.config.rgbd_height,
                max_depth=self.config.rgbd_max_depth,
            )
        return self._forward_rgbd_camera

    def _egocentric_camera(self):
        if self._rgbd_camera is None:
            from pss.sim.mujoco.rgbd import MujocoRGBDCamera

            self._rgbd_camera = MujocoRGBDCamera(
                self.mj,
                self.model,
                self.scenario,
                camera_name="ego_rgbd",
                width=self.config.rgbd_width,
                height=self.config.rgbd_height,
                max_depth=self.config.rgbd_max_depth,
            )
        return self._rgbd_camera
