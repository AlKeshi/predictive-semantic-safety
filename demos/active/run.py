import hashlib
import math
import time
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from benchmarks.sensor import detections_for_observation
from pss.control.body_gaze import BodyGazeController
from pss.control.code_world import CodeWorldController
from pss.io import save_json, source_hashes
from pss.predictors.code_as_world import create_predictor
from pss.sim.mujoco.go1_policy import Go1OnnxPolicy

from .control import Control
from .environment import build, sensor
from .forecast import Forecast
from .fusion import OracleGround
from .ground import Ground
from .report import Report
from .scene import scene_inventory


class ActiveEpisode(Forecast, Ground, Control, Report):
    def __init__(
        self,
        out,
        duration=42.0,
        period=0.6,
        horizon=2.5,
        speed=0.55,
        *,
        model=None,
        semantic_enabled=True,
        render=False,
        device="cuda",
    ):
        self.out = out
        self.duration = duration
        self.period = period
        self.horizon = horizon
        self.speed = speed
        self.model = model
        self.semantic_enabled = semantic_enabled
        self.render = render
        self.device = device
        self.predictor = (
            create_predictor(self.model, max_frames=8, max_new_tokens=1536, device=self.device)
            if self.semantic_enabled
            else None
        )
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.mj, self.model, self.data, self.robot_geoms, xml, self.initial = build()
        self.objects, self.geo, self.static = scene_inventory(self.model, self.data)
        self.names = [obj["id"] for obj in self.objects]
        (self.out / "scene.xml").write_text(xml)
        save_json(
            self.out / "protocol.json",
            dict(
                schema="active-caw-scene-inventory-uncalibrated-v3",
                calibrated=False,
                scripted_robot=False,
                controller="public CodeWorldController backup_policy=retreat"
                if self.semantic_enabled
                else "direct goal, no safety filter",
                perception="actual robot-mounted RGB-D/CaW before floor contact; oracle ground XY/velocity afterwards"
                if self.semantic_enabled
                else "robot proprioception only; no visual forecasts",
                oracle_ground_state=self.semantic_enabled,
                runner_source="demos/active/run.py",
                nominal_planner="A* on current ground disks and known static map; no manual waypoints",
                nominal_planner_mode="current_snapshot_monotone_egress",
                emergency_policy="public retreat feedback with fresh validated ground geometry; explicitly uncertified",
                camera="omnivla_ego; repository causal ideal pan/tilt and body gaze",
                inference_execution="offline synchronous, simulator paused during inference",
                initial=self.initial,
                declared_objects=self.objects,
                declared_static_obstacles=self.static,
                duration=self.duration,
                period=self.period,
                horizon=self.horizon,
                speed=self.speed,
                warmup=0.8,
                q_m=0.0,
                terminal_continuation_verified=False,
                scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_files=source_hashes("pss", "benchmarks", "datagen", "demos"),
            ),
        )
        self.world = sensor(self.mj, self.model, self.data, self.out, self.objects)
        self.controller = CodeWorldController(backup_policy="retreat", static_obstacles=self.static)
        self.ground = OracleGround(self.objects, horizon=self.horizon)
        self.semantic = {}
        self.semantic_failed = False
        self.live_family, self.live_sources = ([], {})
        self.floor_seen = set()
        self.floor_events = []
        self.fusion_updates = []
        self.gaze = BodyGazeController()
        self.policy = Go1OnnxPolicy()
        self.frames, self.tracks, self.times, self.poses, self.depths = [
            deque(maxlen=29) for _ in range(5)
        ]
        self.goal = np.array(self.initial["goal"])
        self.command = np.zeros(2)
        self.updates, self.trace, self.contacts, self.rows = ([], [], [], [])
        self.saved_qpos, self.saved_qvel, self.saved_times, self.saved_cam = ([], [], [], [])
        self.floor = self.model.geom("floor").id
        self.striker = self.model.jnt_dofadr[self.model.joint("striker_free").id]
        self.direction = np.array(
            [
                -self.initial["config"]["diagonal"],
                -math.sqrt(1 - self.initial["config"]["diagonal"] ** 2),
            ]
        )
        self.released = False
        self.impact = self.goal_time = None
        self.next_update = 0.8
        self.next_ground = 0.0
        self.next_plan = 0.0
        self.navigation_target = self.goal.copy()
        self.navigation_receipt = dict(status="direct_goal_until_all_ground")
        self.min_height = float(self.data.qpos[2])
        self.input_video = None
        if self.render:
            self.input_video = imageio.get_writer(
                self.out / "actual-ego.mp4", fps=20, codec="libx264", quality=8
            )

    def run(self):
        try:
            for step in range(round(self.duration / 0.001) + 1):
                self.now = float(self.data.time)
                if not self.released and self.now >= self.initial["ball_release"] - 1e-09:
                    velocity = self.initial["config"]["ball_speed"] * self.direction
                    self.data.qvel[self.striker : self.striker + 3] = np.r_[velocity, 0.0]
                    self.data.qvel[self.striker + 3 : self.striker + 6] = (
                        np.cross([0, 0, 1], np.r_[velocity, 0.0])
                        / self.model.geom("striker").size[0]
                    )
                    self.released = True
                if step % 20 == 0 or step % 50 == 0:
                    self.mj.mj_forward(self.model, self.data)
                for c in self.data.contact[: self.data.ncon]:
                    pair = {int(c.geom1), int(c.geom2)}
                    if self.floor in pair:
                        for g in pair & set(self.geo):
                            if self.geo[g] not in self.floor_seen:
                                self.floor_events.append(dict(time=self.now, object_id=self.geo[g]))
                                self.floor_seen.add(self.geo[g])
                    if (
                        self.model.geom("striker").id in pair
                        and pair & set(self.geo) - {self.model.geom("striker").id}
                        and (self.impact is None)
                    ):
                        self.impact = self.now
                    if (
                        pair & self.robot_geoms
                        and self.floor not in pair
                        and pair - self.robot_geoms
                    ):
                        kind = "hazard" if pair & set(self.geo) else "environment"
                        event = dict(
                            time=self.now, kind=kind, geoms=[self.model.geom(g).name for g in pair]
                        )
                        if not any((x["kind"] == kind for x in self.contacts)):
                            self.contacts.append(event)
                if step % 50 == 0:
                    self.observation = self.world.capture_forward_egocentric()
                    self.detections = detections_for_observation(
                        self.world, self.observation, self.objects
                    )
                    self.gaze.observe(self.detections, self.now)
                    self.frames.append(self.observation.rgb.copy())
                    self.tracks.append((self.now, self.detections))
                    self.times.append(self.now)
                    self.poses.append(self.observation.cam_pose.copy())
                    self.depths.append(self.observation.depth.copy())
                    if self.input_video:
                        self.input_video.append_data(self.observation.rgb)
                    self.saved_qpos.append(self.data.qpos.copy())
                    self.saved_qvel.append(self.data.qvel.copy())
                    self.saved_times.append(self.now)
                    self.saved_cam.append(self.model.cam_quat[self.world.omnivla_camera_id].copy())
                if step % 20 == 0:
                    self.control_tick()
                if step < round(self.duration / 0.001):
                    self.mj.mj_step(self.model, self.data)
        finally:
            if self.input_video:
                self.input_video.close()
            self.world.close()
            np.savez_compressed(
                self.out / "states.npz",
                qpos=self.saved_qpos,
                qvel=self.saved_qvel,
                times=self.saved_times,
                camera_quat=self.saved_cam,
            )
            save_json(self.out / "trace.json", self.trace)
            save_json(self.out / "updates.json", self.updates)
            save_json(self.out / "fusion-updates.json", self.fusion_updates)
            save_json(self.out / "floor-events.json", self.floor_events)
        return self.report()


def run(
    out,
    duration=42.0,
    period=0.6,
    horizon=2.5,
    speed=0.55,
    *,
    model=None,
    semantic_enabled=True,
    render=False,
    device="cuda",
):
    return ActiveEpisode(
        out,
        duration,
        period,
        horizon,
        speed,
        model=model,
        semantic_enabled=semantic_enabled,
        render=render,
        device=device,
    ).run()
