from __future__ import annotations

import itertools
import time
from dataclasses import asdict

import numpy as np

from pss.evaluation.code_world_metrics import forecast_metrics
from pss.io import save_json


class RouteReport:
    def report(self):
        if self.transform is not None and self.transform.needs_truth(float(self.world.data.time)):
            self.transform.observe_simulation(
                float(self.world.data.time),
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
        if len(self.observation_frames) == 16 and (
            not (self.out / "observation_clip.npz").exists()
        ):
            np.savez_compressed(
                self.out / "observation_clip.npz",
                frames=np.asarray(self.observation_frames),
                timestamps=np.asarray(self.observation_times),
            )
        save_json(self.out / "external_policy.json", self.external_records)
        save_json(self.out / "truth.json", self.truth)
        save_json(self.out / "contacts.json", self.contact_events)
        if self.transform is None:
            stats = forecast_metrics(self.model_updates, self.truth, self.objects, self.demo_margin)
            committed_stats = forecast_metrics(
                self.model_updates, self.truth, self.objects, self.demo_margin, committed_only=True
            )
            stats["forecast_metrics_schema"] = stats.pop("schema", None)
            stats["committed_forecast_metrics"] = committed_stats
            stats["forecast_metrics_are_proposal_metrics"] = True
        else:
            stats = {"forecast_coverage": "not estimated; use complete indexed calibration traces"}
        for before, after in itertools.pairwise(self.trace):
            dt = after["time"] - before["time"]
            before["measured_planar_accel"] = (
                np.asarray(after["state"][2:]) - before["state"][2:]
            ) / dt
        save_json(self.out / "trace.json", self.trace)
        latencies = [u["latency_s"] for u in self.model_updates if u["latency_s"] > 0]
        report = dict(
            schema="code_world_episode_v2",
            scene=self.config.scene,
            seed=self.config.seed,
            method=self.config.method,
            settings=asdict(self.config),
            scenario=asdict(self.scenario),
            camera={
                "width": self.observation.rgb.shape[1],
                "height": self.observation.rgb.shape[0],
                "projection": "forward",
                "horizontal_fov_deg": float(
                    np.degrees(
                        2
                        * np.arctan(
                            self.observation.rgb.shape[1]
                            / (2 * self.observation.cam_intrinsics[0, 0])
                        )
                    )
                ),
                "vertical_fov_deg": float(
                    np.degrees(
                        2
                        * np.arctan(
                            self.observation.rgb.shape[0]
                            / (2 * self.observation.cam_intrinsics[1, 1])
                        )
                    )
                ),
                "robot_mounted": True,
            },
            predictor_inputs="RGB video + causal measured world-XYZ track history; known fixture shapes/colors/extents",
            forecast_space="metric",
            inference_execution="offline_synchronous_simulator_paused",
            body_gaze="observed RGB-D or supported committed forecast; last-observed search after expiry; yaw only",
            actuator_interface="integrated filtered reference acceleration to Go1 velocity command; measured velocity in barrier; no tracking certificate",
            uncertainty=self.uncertainty or {"calibrated": False, "residual_m": 0.0},
            goal_reached=self.goal_time is not None,
            goal_time_s=self.goal_time,
            hazard_contact=any((c["kind"] == "hazard" for c in self.contact_events)),
            environment_contact=any((c["kind"] == "environment" for c in self.contact_events)),
            contacts=self.contact_events,
            **self.clearance_audit.report(),
            minimum_trunk_height_m=self.minimum_height,
            path_length_m=self.path_length,
            fallback_ticks=self.fallback_ticks,
            intervention_ticks=self.intervention_ticks,
            policy_ticks=self.total_policy_ticks,
            forecast_updates=len(self.model_updates),
            accepted_updates=sum((bool(u.get("accepted")) for u in self.model_updates)),
            latency_median_s=float(np.median(latencies)) if latencies else None,
            latency_p95_s=float(np.quantile(latencies, 0.95)) if latencies else None,
            wall_time_s=time.perf_counter() - self.wall_start,
            **stats,
        )
        if self.demo_navigation:
            report.update(
                navigation_profile="demo-ground-truth",
                ground_handoff=self.ground.active,
                ground_handoff_events=self.ground.events,
                ground_last_gate=self.ground.last_gate,
                ground_last_measurements=self.ground.latest,
                ground_measurement="simulator current geometry and floor contacts; oracle, not camera-only",
                body_gaze="hold initial heading independently of the causal camera gimbal",
                controller_schema="demo_navigation_ground_truth_v1_uncalibrated",
                heuristic_margin_m=self.demo_margin,
            )
        save_json(self.out / "report.json", report)
        return report
