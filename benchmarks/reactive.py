from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from datagen import attachments, ceiling_xml, route_scene
from pss.io import save_json, source_hashes
from pss.scenarios.ceiling_fixture import initialize_robot
from pss.scenarios.code_world_suite import configure_native_contacts, world_config
from pss.sim.mujoco import Go1MujocoWorld
from pss.sim.mujoco.go1_policy import Go1OnnxPolicy

from .backup_cbf import MatchedBackupCBF
from .geometry import geom_bounds
from .plain_cbf import PlainCBF
from .reactive_contacts import ReactiveContacts
from .reactive_control import ReactiveControl
from .reactive_report import ReactiveReport
from .recording import PlainVideo


def create_world(case, scene_xml=None):
    if case["family"] == "ceiling":
        model = mujoco.MjModel.from_xml_string(scene_xml or ceiling_xml(case))
        data = mujoco.MjData(model)
        policy = Go1OnnxPolicy()
        initialize_robot(model, data, policy)
        data.qpos[0] = case["start_x"]
        data.qpos[1] = case.get("start_y", 0.0)
        yaw0 = case.get("yaw_rad", 0.0)
        data.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        mujoco.mj_forward(model, data)
        physical = attachments(model, case)
        return (model, data, policy, physical, None)
    world = Go1MujocoWorld(
        route_scene(case), world_config(case["duration"], render=False), scene_xml=scene_xml
    )
    configure_native_contacts(world)
    world.prepare_online_episode()
    return (world.model, world.data, world.policy, None, world)


class ReactiveEpisode(ReactiveControl, ReactiveContacts, ReactiveReport):
    def __init__(self, case, out, method, render=False, *, scene_xml=None, demo_profile=False):
        self.case = case
        self.out = out
        self.method = method
        self.render = render
        self.scene_xml = scene_xml
        self.demo_profile = demo_profile
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=False)
        save_json(self.out / "case.json", self.case)
        self.source_files = source_hashes("pss", "benchmarks", "datagen", "demos")
        self.model, self.data, self.policy, self.physics, self.world = create_world(
            self.case, self.scene_xml
        )
        self.root = self.model.body("trunk").id
        robot_bodies = set()
        for bid in range(1, self.model.nbody):
            ancestor = bid
            while ancestor:
                if ancestor == self.root:
                    robot_bodies.add(bid)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        self.robot_geoms = {
            i
            for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in robot_bodies
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        }
        hazard_bodies = (
            {self.physics.body} if self.physics else set(self.world.dynamic_obstacle_body_ids)
        )
        self.hazard_geoms = {
            i
            for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in hazard_bodies
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        }
        self.environment_geoms = {
            i
            for i in range(self.model.ngeom)
            if i not in self.robot_geoms | self.hazard_geoms
            and self.model.geom(i).name not in ("floor", "ground")
            and (int(self.model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_PLANE))
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        }
        self.candidates = sorted(self.hazard_geoms | self.environment_geoms)
        initial_bottom = {
            g: geom_bounds(self.model, self.data, g)[2] for g in self.environment_geoms
        }
        self.known_polygons = {}
        for g in self.environment_geoms:
            self.g = g
            self.corners = geom_bounds(self.model, self.data, self.g)[3]
            if self.corners is not None:
                points = np.unique(self.corners[:, :2], axis=0)
                self.known_polygons[self.g] = points[ConvexHull(points).vertices].tolist()
        self.active = {g for g in self.environment_geoms if initial_bottom[g] <= 0.55}
        self.ground_events = []
        self.admissions = [
            {"time": 0.0, "object_id": self.model.geom(g).name, "reason": "initial_known_obstacle"}
            for g in sorted(self.active)
        ]
        assert self.method in {"plain_cbf", "backup_cbf"}
        self.controller = (
            PlainCBF()
            if self.method == "plain_cbf"
            else MatchedBackupCBF(horizon=8.0 if self.case["family"] == "ceiling" else 2.5)
        )
        self.geometry_data = mujoco.MjData(self.model)
        self.goal = np.array([self.case["hazard_x"] + 2.0, 0.0])
        self.command_velocity = np.zeros(2)
        self.trace, self.truth, self.contacts, self.snapshots = ([], [], [], [])
        self.first_pairs = set()
        self.contact_ticks = self.environment_ticks = 0
        self.goal_time = None
        self.start_wall = time.monotonic()
        self.policy_dt = 0.02
        self.substeps = round(self.policy_dt / self.model.opt.timestep)
        self.qpos_log, self.qvel_log, self.state_times = ([], [], [])
        self.video = None
        if self.render:
            self.video = PlainVideo(
                self.model,
                self.data,
                self.world,
                self.out,
                "CBF-QP" if self.method == "plain_cbf" else "Backup-CBF",
            )
        self.next_video = 0.0

    def run(self):
        try:
            count = round(self.case["duration"] / self.model.opt.timestep)
            for step in range(count):
                self.step = step
                self.now = float(self.data.time)
                if self.step % self.substeps == 0:
                    self.control_tick()
                if self.video is not None and self.now + 1e-09 >= self.next_video:
                    self.video.capture()
                    self.next_video += 0.05
                if self.physics:
                    self.physics.step(self.data)
                else:
                    self.data.qfrc_applied[:] = 0.0
                    self.world.apply_dynamic_obstacle_triggers()
                mujoco.mj_step(self.model, self.data)
                self.record_contacts()
                if self.demo_profile and (self.hazard_tick or self.environment_tick):
                    break
            return self.report()
        finally:
            if self.video is not None:
                self.video.close()
            if self.world is not None:
                self.world.close()


def run(case, out, method, render=False, *, scene_xml=None, demo_profile=False):
    return ReactiveEpisode(
        case, out, method, render, scene_xml=scene_xml, demo_profile=demo_profile
    ).run()
