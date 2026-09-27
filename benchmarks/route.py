from __future__ import annotations

import time
from collections import deque

import numpy as np

from pss.control.body_gaze import BodyGazeController
from pss.control.code_world import CodeWorldController
from pss.control.recovery import RecoveryController
from pss.evaluation.mujoco_clearance import ClearanceAudit
from pss.io import save_json
from pss.predictors.code_as_world import create_predictor as CodeAsWorldPredictor
from pss.scenarios.code_world_suite import configure_native_contacts, world_config
from pss.sim.mujoco import Go1MujocoWorld
from pss.sim.mujoco.go1_rollout import yaw_from_quaternion

from .config import RouteConfig
from .ground_handoff import GroundHandoff
from .recording import RecordedRenderer as EpisodeRenderer
from .route_control import RouteControl
from .route_forecast import RouteForecast
from .route_geometry import _contacts, _geometry_groups, _objects
from .route_report import RouteReport
from .sensor import detections_for_observation, install_sensor


class RouteEpisode(RouteForecast, RouteControl, RouteReport):
    def __init__(
        self,
        config: RouteConfig,
        scenario,
        predictor=None,
        external=None,
        transform=None,
        uncertainty=None,
    ):
        self.config = config
        self.scenario = scenario
        self.predictor = predictor
        self.external = external
        self.transform = transform
        self.uncertainty = uncertainty
        self.demo_navigation = self.config.navigation_profile == "demo-ground-truth"
        self.demo_margin = self.config.demo_margin_m if self.demo_navigation else 0.0
        if self.demo_navigation and (self.transform is not None or self.external is not None):
            raise ValueError(
                "Demo navigation cannot be combined with calibrated or external policies"
            )
        self.cfg = world_config(self.config.duration, render=self.config.render)
        self.out = self.config.output
        self.out.mkdir(parents=True, exist_ok=True)
        self.objects = _objects(self.scenario)
        self.world = Go1MujocoWorld(self.scenario, self.cfg, scene_xml=self.config.scene_xml)
        configure_native_contacts(self.world)
        if self.external is None:
            install_sensor(self.world, self.out, _objects)
        static_obstacles = [
            (table.position, float(np.linalg.norm(np.asarray(table.size[:2]) / 2)))
            for table in self.scenario.tables
        ]
        for prop in self.scenario.scene_props:
            if getattr(prop, "collidable", True):
                static_obstacles.append(
                    (prop.position[:2], float(np.linalg.norm(np.asarray(prop.size) / 2)))
                )
        controller_type = CodeWorldController
        self.ground = None
        if self.demo_navigation:
            controller_type = RecoveryController
            self.ground = GroundHandoff(self.world, self.out, static_obstacles)
        self.controller = controller_type(
            inactive=self.config.method == "nominal",
            static_obstacles=() if self.config.method == "nominal" else static_obstacles,
        )
        if self.config.method == "code_world" and self.predictor is None:
            self.predictor = CodeAsWorldPredictor(
                model_path=self.config.model,
                max_new_tokens=1536,
                device=self.config.device,
            )
        self.observation_frames = deque(maxlen=16)
        self.observation_times = deque(maxlen=16)
        self.track_history = deque(maxlen=16)
        self.model_updates = []
        self.trace = []
        self.truth = []
        self.contact_events = []
        self.seen_contacts = set()
        self.gaze = BodyGazeController()
        self.renderer = None
        self.wall_start = time.perf_counter()
        self.next_sensor = 0.0
        self.next_update = (
            self.config.warmup if self.transform is None else self.transform.origins[0]
        )
        self.next_render = 0.0
        self.current_command = np.zeros(3)
        self.command_velocity = np.zeros(2)
        self.diagnostic = {"mode": "observing"}
        self.goal_time = None
        self.clearance_audit = ClearanceAudit()
        self.minimum_height = 10.0
        self.total_policy_ticks = 0
        self.fallback_ticks = 0
        self.intervention_ticks = 0
        self.last_xy = None
        self.path_length = 0.0
        self.observation = None
        self.external_records = []

    def run(self):
        try:
            self.world.prepare_online_episode()
            self.heading_reference = yaw_from_quaternion(self.world.data.qpos[3:7])
            self.robot_geoms, self.hazard_geoms, environment_geoms = _geometry_groups(self.world)
            if self.config.render:
                method_label = {"code_world": "Code-as-World", "nominal": "Nominal"}[
                    self.config.method
                ]
                self.renderer = EpisodeRenderer(
                    self.world,
                    self.out,
                    self.config.scene.replace("_", " ").title() + "  |  " + method_label,
                    20,
                    predictor_label=f"{method_label} forecast"
                    if self.config.method != "nominal"
                    else "Unfiltered goal navigation",
                )
            substeps = round(self.cfg.policy_dt / self.cfg.sim_dt)
            step = 0
            while self.world.data.time < self.config.duration - 1e-09:
                self.now = float(self.world.data.time)
                if self.transform is not None and self.transform.needs_truth(self.now):
                    self.transform.observe_simulation(
                        self.now,
                        self.world.model,
                        self.world.data,
                        {
                            ob.name: body
                            for ob, body in zip(
                                self.world.scenario.dynamic_obstacles,
                                self.world.dynamic_obstacle_body_ids,
                                strict=True,
                            )
                        },
                    )
                if self.now + 1e-09 >= self.next_sensor:
                    self.observation = self.world.capture_forward_egocentric()
                    self.observation_frames.append(self.observation.rgb.copy())
                    self.sensor_detections = detections_for_observation(
                        self.world, self.observation, self.objects
                    )
                    self.gaze.observe(self.sensor_detections, self.now)
                    self.track_history.append((self.now, self.sensor_detections))
                    self.observation_times.append(self.now)
                    self.next_sensor += 0.05
                    if len(self.observation_frames) == 16 and (
                        not (self.out / "observation_clip.npz").exists()
                    ):
                        np.savez_compressed(
                            self.out / "observation_clip.npz",
                            frames=np.asarray(self.observation_frames),
                            timestamps=np.asarray(self.observation_times),
                        )
                if step % substeps == 0:
                    self.control_tick()
                if self.renderer is not None and self.now + 1e-09 >= self.next_render:
                    self.renderer.capture(
                        self.observation,
                        self.diagnostic,
                        self.controller.snapshot(),
                        self.clearance,
                        contact=bool(self.contact_events),
                        goal_reached=self.goal_time is not None,
                    )
                    self.next_render += 0.05
                self.world.data.qfrc_applied[:] = 0
                self.world.apply_dynamic_obstacle_triggers()
                self.world.mj.mj_step(self.world.model, self.world.data)
                for event in _contacts(
                    self.world, self.robot_geoms, self.hazard_geoms, environment_geoms
                ):
                    key = (event["kind"], event["robot_geom_id"], event["other_geom_id"])
                    if key not in self.seen_contacts:
                        self.seen_contacts.add(key)
                        self.contact_events.append(event)
                step += 1
            return self.report()
        except Exception as exc:
            save_json(self.out / "trace.json", self.trace)
            save_json(self.out / "truth.json", self.truth)
            save_json(self.out / "contacts.json", self.contact_events)
            save_json(self.out / "updates.json", self.model_updates)
            save_json(
                self.out / "failure.json",
                {
                    "status": "incomplete",
                    "time": float(self.world.data.time),
                    "error": repr(exc),
                    "snapshot": self.controller.snapshot(),
                },
            )
            raise
        finally:
            if self.renderer is not None:
                self.renderer.close()
            self.world.close()


def run_episode(
    config: RouteConfig, scenario, predictor=None, external=None, transform=None, uncertainty=None
):
    return RouteEpisode(config, scenario, predictor, external, transform, uncertainty).run()
