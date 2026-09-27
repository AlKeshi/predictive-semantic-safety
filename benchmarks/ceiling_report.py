from __future__ import annotations

import hashlib
import time

from pss.io import save_json


class CeilingReport:
    def report(self):
        receipt = {
            "pipeline_schema": "pss_ceiling_direct_goal_tracking_v4",
            "nominal_policy": "direct_goal_measured_velocity_feedback_without_intermediate_waypoints"
            if self.demo_navigation
            else "direct_goal_previous_command_velocity_feedback",
            "locomotion_reference_policy": "static_hocbf_minimum_change_projection",
            "tracking_error_reserve_m_s": self.tracking_error_reserve,
            "tracking_error_reserve_status": "empirical_observed_maximum_not_certified",
            "seed": self.config.seed,
            "goal_xy_m": self.goal.tolist(),
            "nominal_speed_m_s": self.config.speed,
            "status": "completed empirical closed-loop run",
            "robot_goal_reached": self.reached_at is not None,
            "reached_at_s": self.reached_at,
            "fixture_release_events": self.physics.events,
            "first_predicted_hazard_s": self.first_hazard,
            "first_filter_intervention_s": self.first_intervention,
            "rgbd_settled_observed": self.settled,
            "grounded_static_handoff": self.ground_handoff,
            "forecast_method": "Code-as-World two-stage event hypothesis plus executable gravity/contact rollout",
            "hazard_contact_physics_steps": self.contact_steps,
            "environment_contact": self.environment_contact_steps > 0,
            "environment_contact_physics_steps": self.environment_contact_steps,
            "first_contact_s": self.first_contact,
            "forecast_attempts": len(self.proposals),
            "committed_forecasts": sum((bool(p.get("accepted")) for p in self.proposals)),
            "duration_s": float(self.data.time),
            "wall_elapsed_s": time.monotonic() - self.start_wall,
            "model": self.predictor.provenance
            if self.predictor is not None
            else {"baseline": self.config.method},
            "minimum_trunk_height_m": self.min_height,
            "minimum_geom_clearance_m": self.min_clearance,
            "benchmark_method": self.config.method,
            "q_m": self.demo_margin,
            "uncertainty": self.uncertainty or {"calibrated": False, "residual_m": 0.0},
            "scene_sha256": hashlib.sha256(self.xml.encode()).hexdigest(),
            "simulation_paused_during_inference": True,
            "camera": "actual RGB-D; idealized actuated head tracks observed points",
            "state_input_boundary": "hazard from RGB-D only; robot proprioception; simulator fixture state only in evaluator",
        }
        receipt.update(self.clearance_audit.report())
        if self.demo_navigation:
            receipt.update(
                navigation_profile="demo-ground-truth",
                heuristic_margin_m=self.demo_margin,
                controller_schema="demo_navigation_ground_truth_v1_uncalibrated",
            )
        save_json(self.out / "receipt.json", receipt)
        return receipt
