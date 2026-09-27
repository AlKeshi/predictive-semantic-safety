from __future__ import annotations

import hashlib
import json
import platform
import sys
import time

import mujoco
import numpy as np

from pss.io import save_json, source_hashes


class ReactiveReport:
    def report(self):
        save_json(self.out / "trace.json", self.trace)
        save_json(self.out / "truth.json", self.truth)
        save_json(self.out / "contacts.json", self.contacts)
        save_json(self.out / "admissions.json", self.admissions)
        save_json(self.out / "ground_events.json", self.ground_events)
        save_json(self.out / "initial_geometry.json", self.snapshots)
        np.savez_compressed(
            self.out / "native_states.npz",
            times=self.state_times,
            qpos=self.qpos_log,
            qvel=self.qvel_log,
        )
        mujoco.mj_saveLastXML(str(self.out / "scene.xml"), self.model)
        save_json(
            self.out / "report.json",
            {
                "schema": "matched_current_geometry_cbf_v1",
                "method": self.method,
                "case_id": self.case["id"],
                "family": self.case["family"],
                "goal_reached": self.goal_time is not None,
                "goal_time_s": self.goal_time,
                "hazard_contact": self.contact_ticks > 0,
                "environment_contact": self.environment_ticks > 0,
                "hazard_contact_physics_steps": self.contact_ticks,
                "environment_contact_physics_steps": self.environment_ticks,
                "minimum_trunk_height_m": min((t["robot_z"] for t in self.trace)),
                "duration_s": float(self.data.time),
                "input_boundary": "shared current native solid geometry in z<=0.55 m; retain admitted identities; no vision or future hazard state",
                "geometry_update_hz": 50.0,
                "overhead_admission": "current oriented solid bottom<=0.55 m at 50 Hz; same rule for both controllers",
                "controller": "original HOCBF"
                if self.method == "plain_cbf"
                else "original PSS braking flow and sensitivities with stationary current geometry",
                "q_m": 0.0,
                "robot_radius_m": 0.403,
                "clearance_m": 0.05,
                "comparison_scope": "matched controller comparison; not a bare model-code deletion or RGB-D end-to-end ablation",
                "moving_obstacle_derivatives": "not modeled; no moving-obstacle safety guarantee",
                "model_calls": 0,
                "calibration_performed": False,
                "future_world_prediction_count": 0,
                "elapsed_s": time.monotonic() - self.start_wall,
            },
        )
        assert source_hashes("pss", "benchmarks", "datagen", "demos") == self.source_files
        save_json(
            self.out / "benchmark_receipt.json",
            {
                "status": "complete",
                "method": self.method,
                "case_id": self.case["id"],
                "source_files": self.source_files,
                "case_sha256": hashlib.sha256(
                    json.dumps(self.case, sort_keys=True).encode()
                ).hexdigest(),
                "platform": platform.platform(),
                "python": sys.version,
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
                "model_calls": 0,
                "vision_calls": 0,
            },
        )
        return json.loads((self.out / "report.json").read_text())
