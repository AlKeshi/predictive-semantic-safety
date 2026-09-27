from __future__ import annotations

import numpy as np

from pss.data.episode import ObservationBundle
from pss.scenarios.spec import ScenarioSpec


class MujocoRGBDCamera:
    def __init__(
        self,
        mujoco_module,
        model,
        scenario: ScenarioSpec,
        *,
        camera_name: str = "ego_rgbd",
        width: int = 640,
        height: int = 480,
        max_depth: float = 10.0,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("RGB-D width and height must be positive")
        if max_depth <= 0.0:
            raise ValueError("RGB-D maximum depth must be positive")
        self.mj = mujoco_module
        self.model = model
        self.scenario = scenario
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.max_depth = float(max_depth)
        self.camera_id = int(model.camera(camera_name).id)
        self.renderer = self.mj.Renderer(model, height=self.height, width=self.width)
        self._stack_body_to_instance = {
            int(model.body(f"stack_{obj.name}").id): i for i, obj in enumerate(scenario.objects)
        }
        try:
            self._table_body_id = int(model.body("table").id)
        except KeyError:
            self._table_body_id = -1

    def capture(self, data) -> ObservationBundle:
        self.renderer.update_scene(data, camera=self.camera_name)
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy().astype(np.float32, copy=False)
        self.renderer.enable_segmentation_rendering()
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        if rgb.shape != (self.height, self.width, 3):
            raise RuntimeError(f"Unexpected MuJoCo RGB shape {rgb.shape}")
        if depth.shape != (self.height, self.width):
            raise RuntimeError(f"Unexpected MuJoCo depth shape {depth.shape}")
        if segmentation.shape != (self.height, self.width, 2):
            raise RuntimeError(f"Unexpected MuJoCo segmentation shape {segmentation.shape}")
        valid = np.isfinite(depth) & (depth > 0.0) & (depth <= self.max_depth)
        depth = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
        instance = self._instances_from_segmentation(segmentation)
        intrinsics = self._intrinsics()
        cam_pose = self._camera_to_world(data)
        return ObservationBundle(
            rgb=rgb.astype(np.uint8, copy=False),
            depth=depth,
            instance=instance,
            cam_intrinsics=intrinsics,
            cam_pose=cam_pose,
        )

    def capture_rgb(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera_name)
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        rgb = self.renderer.render().copy()
        if rgb.shape != (self.height, self.width, 3):
            raise RuntimeError(f"Unexpected MuJoCo RGB shape {rgb.shape}")
        return rgb.astype(np.uint8, copy=False)

    def close(self) -> None:
        self.renderer.close()

    def _instances_from_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        instance = np.full((self.height, self.width), -1, dtype=np.int32)
        object_id = segmentation[..., 0]
        object_type = segmentation[..., 1]
        geom_pixels = object_type == int(self.mj.mjtObj.mjOBJ_GEOM)
        if not np.any(geom_pixels):
            return instance
        for geom_id in np.unique(object_id[geom_pixels]):
            geom_id = int(geom_id)
            if geom_id < 0 or geom_id >= self.model.ngeom:
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id in self._stack_body_to_instance:
                value = self._stack_body_to_instance[body_id]
            elif body_id == self._table_body_id:
                value = -2
            else:
                continue
            instance[geom_pixels & (object_id == geom_id)] = value
        return instance

    def _intrinsics(self) -> np.ndarray:
        fovy = np.deg2rad(float(self.model.cam_fovy[self.camera_id]))
        focal = 0.5 * self.height / np.tan(0.5 * fovy)
        return np.array(
            [
                [focal, 0.0, (self.width - 1.0) / 2.0],
                [0.0, focal, (self.height - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _camera_to_world(self, data) -> np.ndarray:
        rotation_mj = np.asarray(data.cam_xmat[self.camera_id], dtype=float).reshape(3, 3)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation_mj @ np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = np.asarray(data.cam_xpos[self.camera_id], dtype=float)
        return pose


__all__ = ["MujocoRGBDCamera"]
