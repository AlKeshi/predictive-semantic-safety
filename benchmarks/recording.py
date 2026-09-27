from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from pss.visualization.code_world_render import EpisodeRenderer


class NativeRecorder:
    def __init__(self, model, data, out):
        self.model, self.data, self.out = (model, data, Path(out))
        self.times = []
        self.qpos = []
        self.qvel = []
        self.body_quat = []
        self.cam_quat = []
        self.eq = []

    def capture(self):
        self.times.append(float(self.data.time))
        self.qpos.append(self.data.qpos.copy())
        self.qvel.append(self.data.qvel.copy())
        self.body_quat.append(self.model.body_quat.copy())
        self.cam_quat.append(self.model.cam_quat.copy())
        self.eq.append(self.data.eq_active.copy())

    def close(self):
        np.savez_compressed(
            self.out / "native_video_states.npz",
            times=self.times,
            qpos=self.qpos,
            qvel=self.qvel,
            body_quat=self.body_quat,
            cam_quat=self.cam_quat,
            eq_active=self.eq,
        )
        mujoco.mj_saveLastXML(str(self.out / "scene.xml"), self.model)


class RecordedRenderer(EpisodeRenderer):
    def __init__(self, world, out, *a, **kw):
        super().__init__(world, out, *a, **kw)
        self.native = NativeRecorder(world.model, world.data, out)

    def capture(self, *a, **kw):
        frame = super().capture(*a, **kw)
        self.gifs.clear()
        self.native.capture()
        return frame

    def close(self):
        self.writer.close()
        self.hero.close()
        self.native.close()


class PlainVideo:
    def __init__(self, model, data, world, out, label="Plain HOCBF-QP"):
        self.model, self.data, self.world, self.out = (model, data, world, Path(out))
        self.frames = 0
        if world is not None:
            self.route = RecordedRenderer(world, out, label, 20)
        else:
            self.native = NativeRecorder(model, data, out)
            self.left = mujoco.Renderer(model, height=480, width=720)
            self.right = mujoco.Renderer(model, height=480, width=720)
            self.writer = imageio.get_writer(
                str(self.out / "demo.mp4"), fps=20, codec="libx264", quality=8, macro_block_size=1
            )

    def capture(self):
        if self.world is not None:
            obs = self.world.capture_forward_egocentric()
            self.route.capture(obs, {"feasible": False}, {}, None)
        else:
            self.left.update_scene(self.data, camera="scene_preview")
            self.right.update_scene(self.data, camera="ego_rgbd")
            frame = np.concatenate((self.left.render().copy(), self.right.render().copy()), axis=1)
            self.writer.append_data(frame)
            self.native.capture()
            if self.frames == 0:
                Image.fromarray(frame).save(self.out / "preview.png")
        self.frames += 1

    def close(self):
        if self.world is not None:
            self.route.close()
        else:
            self.writer.close()
            self.left.close()
            self.right.close()
            self.native.close()
