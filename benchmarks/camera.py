import mujoco
import numpy as np


class Camera:
    def __init__(self, model):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=480, width=720)
        self.id = model.camera("ego_rgbd").id
        self.head = model.body("rgbd_head").id
        self.pitch, self.yaw = (np.deg2rad(13.5), 0.0)
        focal = 240 / np.tan(np.deg2rad(model.cam_fovy[self.id]) / 2)
        self.k = np.array([[focal, 0, 359.5], [0, focal, 239.5], [0, 0, 1.0]])

    def track(self, data, observed_center):
        if observed_center is None:
            return
        trunk = self.model.body("trunk").id
        local = data.xmat[trunk].reshape(3, 3).T @ (observed_center - data.cam_xpos[self.id])
        desired_yaw = np.clip(np.arctan2(local[1], local[0]), -1.65, 1.65)
        desired_pitch = np.clip(np.arctan2(local[2], np.linalg.norm(local[:2])), -0.75, 0.95)
        self.pitch += np.clip(desired_pitch - self.pitch, -0.055, 0.055)
        self.yaw += np.clip(desired_yaw - self.yaw, -0.055, 0.055)
        cy, sy = (np.cos(self.yaw / 2), np.sin(self.yaw / 2))
        cp, sp = (np.cos(self.pitch / 2), np.sin(self.pitch / 2))
        self.model.body_quat[self.head] = [cy * cp, sy * sp, -cy * sp, sy * cp]
        mujoco.mj_forward(self.model, data)

    def capture(self, data):
        self.renderer.update_scene(data, camera="ego_rgbd")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[self.id].reshape(3, 3) @ np.diag([1, -1, -1])
        pose[:3, 3] = data.cam_xpos[self.id]
        return (rgb, depth, pose)
