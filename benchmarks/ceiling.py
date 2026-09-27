from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np

from datagen import attachments, ceiling_xml
from pss.control.code_world import CodeWorldController
from pss.control.recovery import RecoveryController
from pss.evaluation.mujoco_clearance import ClearanceAudit
from pss.io import save_json
from pss.predictors.code_as_world import create_predictor as CodeAsWorldPredictor
from pss.predictors.structured_json import install_schema_generation
from pss.scenarios.ceiling_fixture import initialize_robot
from pss.sim.mujoco.go1_policy import Go1OnnxPolicy

from .camera import Camera
from .ceiling_control import CeilingControl
from .ceiling_forecast import CeilingForecast
from .ceiling_observation import CeilingObservation
from .ceiling_report import CeilingReport
from .config import CeilingConfig
from .recording import NativeRecorder


class CeilingEpisode(CeilingForecast, CeilingObservation, CeilingControl, CeilingReport):
    def __init__(
        self, config: CeilingConfig, case, external=None, transform=None, uncertainty=None
    ):
        self.config = config
        self.case = case
        self.external = external
        self.transform = transform
        self.uncertainty = uncertainty
        self.demo_navigation = self.config.navigation_profile == "demo-ground-truth"
        self.demo_margin = self.config.demo_margin_m if self.demo_navigation else 0.0
        if self.demo_navigation and (self.transform is not None or self.external is not None):
            raise ValueError(
                "Demo navigation cannot be combined with calibrated or external policies"
            )
        controller_type = CodeWorldController
        if self.demo_navigation:
            controller_type = RecoveryController
        self.out = self.config.output
        self.out.mkdir(parents=True, exist_ok=True)
        self.xml = self.config.scene_xml or ceiling_xml(self.case)
        (self.out / "scene.xml").write_text(self.xml)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.policy = Go1OnnxPolicy()
        initialize_robot(self.model, self.data, self.policy)
        self.data.qpos[0] = self.case["start_x"]
        self.data.qpos[1] = self.case.get("start_y", 0.0)
        yaw0 = self.case.get("yaw_rad", 0.0)
        self.data.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.physics = attachments(self.model, self.case)
        self.camera = Camera(self.model)
        self.outside = mujoco.Renderer(self.model, height=480, width=720)
        self.predictor = None
        if self.config.method == "cw":
            self.predictor = CodeAsWorldPredictor(
                self.config.model,
                max_new_tokens=3000,
                max_frames=12,
                max_pixels=262144,
                device=self.config.device,
            )
            install_schema_generation(self.predictor)
        self.external_records = []
        self.controller = controller_type(clearance=0.05, inactive=True)
        self.frames, self.timestamps, self.history = (
            deque(maxlen=25),
            deque(maxlen=25),
            deque(maxlen=25),
        )
        self.sensor_log, self.controls, self.truth, self.proposals = ([], [], [], [])
        self.previous = None
        self.active = self.secure_observation = False
        self.last_secure = -np.inf
        self.last_valid_support = -np.inf
        self.last_committed_forecast = None
        self.next_forecast = 0.75 if self.transform is None else self.transform.origins[0]
        self.horizon = 8.0
        self.command_velocity = np.zeros(2)
        self.tracking_error_reserve = 0.0
        self.goal = np.array([self.case["hazard_x"] + 2.0, 0.0])
        self.settled = False
        self.ground_handoff = False
        self.handoff_receipt = None
        self.grounded_history = deque(maxlen=12)
        self.contact_steps = 0
        self.environment_contact_steps = 0
        self.first_contact = None
        self.first_hazard = None
        self.first_intervention = None
        self.min_height = 10.0
        self.min_clearance = 2.0
        self.clearance_audit = ClearanceAudit()
        self.audit_world = SimpleNamespace(mj=mujoco, model=self.model, data=self.data)
        self.reached_at = None
        self.start_wall = time.monotonic()
        self.robot_bodies = set()
        self.root_robot = self.model.body("trunk").id
        for body in range(1, self.model.nbody):
            parent = body
            while parent:
                if parent == self.root_robot:
                    self.robot_bodies.add(body)
                    break
                parent = int(self.model.body_parentid[parent])

        class NullWriter:
            def append_data(self, frame):
                pass

            def close(self):
                pass

        self.writer = (
            imageio.get_writer(
                self.out / "demo.mp4", fps=20, codec="libx264", quality=8, macro_block_size=1
            )
            if self.config.render
            else NullWriter()
        )
        self.native = NativeRecorder(self.model, self.data, self.out)

    def run(self):
        try:
            for step in range(round(self.config.duration / self.model.opt.timestep) + 1):
                self.step = step
                self.now = float(self.data.time)
                if self.transform is not None and self.transform.needs_truth(self.now):
                    self.transform.observe_simulation(
                        self.now, self.model, self.data, {"ceiling_fixture": self.physics.body}
                    )
                self.state = np.r_[self.data.qpos[:2], self.data.qvel[:2]]
                if self.step % 25 == 0:
                    self.observe()
                if self.step % 10 == 0:
                    self.control_tick()
                touched = False
                environment_touched = False
                for contact in self.data.contact[: self.data.ncon]:
                    bodies = [
                        int(self.model.geom_bodyid[int(contact.geom1)]),
                        int(self.model.geom_bodyid[int(contact.geom2)]),
                    ]
                    if any((body in self.robot_bodies for body in bodies)):
                        other = (
                            int(contact.geom2)
                            if bodies[0] in self.robot_bodies
                            else int(contact.geom1)
                        )
                        if (
                            int(self.model.geom_bodyid[other]) == 0
                            and self.model.geom(other).name != "floor"
                        ):
                            environment_touched = True
                    if self.physics.body in bodies and any(
                        (body in self.robot_bodies for body in bodies)
                    ):
                        touched = True
                self.environment_contact_steps += int(environment_touched)
                if touched:
                    self.contact_steps += 1
                    self.first_contact = (
                        self.now if self.first_contact is None else self.first_contact
                    )
                self.physics.step(self.data)
                mujoco.mj_step(self.model, self.data)
        finally:
            self.writer.close()
            self.native.close()
            self.camera.renderer.close()
            self.outside.close()
            save_json(self.out / "external_policy.json", self.external_records)
            save_json(self.out / "sensor_log.json", self.sensor_log)
            save_json(self.out / "controls.json", self.controls)
            save_json(self.out / "evaluator_truth.json", self.truth)
            save_json(self.out / "proposals.json", self.proposals)
        return self.report()


def run_episode(config: CeilingConfig, case, external=None, transform=None, uncertainty=None):
    return CeilingEpisode(config, case, external, transform, uncertainty).run()
