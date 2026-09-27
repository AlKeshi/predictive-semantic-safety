from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image

from pss.control.code_world import CodeWorldController
from pss.io import save_json
from pss.perception.ceiling_fixture import observe


class CeilingObservation:
    def ground_update(self):
        if (
            not self.settled
            and self.active
            and (len(self.grounded_history) >= 10)
            and (self.grounded_history[-1][0] - self.grounded_history[0][0] >= 0.44)
        ):
            points = np.array([x[1] for x in self.grounded_history])
            if (
                np.max(points[:, 2]) < 0.16
                and np.max(np.linalg.norm(points - points.mean(0), axis=1)) < 0.055
            ):
                self.settled = True
        if (
            self.demo_navigation
            and self.settled
            and (not self.ground_handoff)
            and (self.detection is not None)
            and (len(self.grounded_history) >= 10)
        ):
            pts = np.array([item[1] for item in self.grounded_history])
            normal = np.asarray(self.diagnostic.get("normal", [0, 0, 0]))
            flat = abs(normal[2]) > 0.985
            quiet = (
                np.max(pts[:, 2]) < 0.16
                and np.max(np.linalg.norm(pts - pts.mean(0), axis=1)) < 0.025
            )
            if flat and quiet:
                measured = pts.mean(0)
                candidate = CodeWorldController(
                    clearance=0.05,
                    inactive=True,
                    static_obstacles=[
                        {
                            "object_id": "ceiling_fixture",
                            "center": measured[:2],
                            "radius": self.detection.radius + self.demo_margin,
                        }
                    ],
                )
                _, checked = candidate.command(self.state, np.zeros(2), self.now)
                if checked["feasible"]:
                    self.handoff_receipt = {
                        "time": self.now,
                        "observed_center": measured.tolist(),
                        "prior_snapshot": self.controller.snapshot(),
                        "candidate_snapshot": candidate.snapshot(),
                        "criterion": "flat, low, stationary RGB-D surface; hard static HOCBF feasible",
                        "assumption": "fallen fixture remains inert on floor after observed settling",
                        "q_m": self.demo_margin,
                        "status": "atomic empirical measured-debris handoff",
                    }
                    self.controller = candidate
                    self.ground_handoff = True
                    save_json(self.out / "ground_handoff.json", self.handoff_receipt)

    def observe(self):
        camera_target = self.previous
        if self.ground_handoff and self.previous is not None:
            relative = self.data.xmat[self.root_robot].reshape(3, 3).T @ (
                self.previous - self.data.qpos[:3]
            )
            if relative[0] < 0.1:
                camera_target = self.data.qpos[:3] + self.data.xmat[self.root_robot].reshape(
                    3, 3
                ) @ np.array([4.0, 0.0, 0.35])
        if self.config.method == "omnivla":
            self.model.body_quat[self.camera.head] = [1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
        else:
            self.camera.track(self.data, camera_target)
        self.rgb, self.depth, self.pose = self.camera.capture(self.data)
        self.detection, self.diagnostic = observe(
            self.rgb, self.depth, self.camera.k, self.pose, self.previous
        )
        detections = {} if self.detection is None else {"ceiling_fixture": self.detection}
        if self.detection is not None:
            self.previous = self.detection.center.copy()
            self.grounded_history.append((self.now, self.previous.copy()))
        else:
            self.grounded_history.clear()
        self.sensor_log.append(
            dict(
                time=self.now,
                center=None if self.detection is None else self.detection.center.tolist(),
                camera_to_world=self.pose.tolist(),
                camera_intrinsics=self.camera.k.tolist(),
                **self.diagnostic,
            )
        )
        self.frames.append(self.rgb)
        self.timestamps.append(self.now)
        self.history.append((self.now, detections))
        if self.config.render or self.step % 1000 == 0:
            self.outside.update_scene(self.data, camera="scene_preview")
            paired = np.concatenate((self.outside.render().copy(), self.rgb), axis=1)
            if self.config.render:
                self.writer.append_data(paired)
                self.native.capture()
            if self.step % 1000 == 0:
                Image.fromarray(paired).save(self.out / f"frame_{self.now:05.2f}.png")
        self.ground_update()
        if (
            self.config.method == "cw"
            and (not self.ground_handoff)
            and (self.now + 1e-08 >= self.next_forecast)
            and (len(self.frames) >= 8)
        ):
            self.forecast()
