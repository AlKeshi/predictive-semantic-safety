from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

OMNIVLA_SOURCE_REPOSITORY = "https://github.com/NHirose/OmniVLA"
OMNIVLA_SOURCE_REVISION = "5182600cb4a9ee07684e17cdd2a6cbafc56b8a68"
OMNIVLA_CHECKPOINT_ID = "NHirose/omnivla-original"
OMNIVLA_CHECKPOINT_REVISION = "e36a84d4923c041149d441f93f3bdb7092bb5f07"
OMNIVLA_CHECKPOINT_STEP = 120000
OMNIVLA_ACTION_PERIOD_S = 1.0 / 3.0
OMNIVLA_ACTION_HORIZON = 8
OMNIVLA_ACTION_DIM = 4
OMNIVLA_POSE_DIM = 4
OMNIVLA_LANGUAGE_POSE_MODALITY_ID = 8
OMNIVLA_GOAL_PROMPT = "move toward the specified goal while avoiding obstacles"
OMNIVLA_WAYPOINT_INDEX = 4
OMNIVLA_WAYPOINT_SPACING_M = 0.1
OMNIVLA_GOAL_DISTANCE_CAP_M = 30.0
_CHECKPOINT_SHA256 = {
    "model-00001-of-00004.safetensors": "3bd0dd7d75924136ecf1a584bdaa8e8b994d4f5d80a7b8869814976f371c05d2",
    "model-00002-of-00004.safetensors": "3c21dcdb5f797631a5e555f749e0ec83a34f4d7227265e7ce3a6c320f8b42994",
    "model-00003-of-00004.safetensors": "5af986297dfbf994d5cf1343071d3fc30c5ba531662c9ac8c1a8364ce26fa1af",
    "model-00004-of-00004.safetensors": "a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60",
    "action_head--120000_checkpoint.pt": "d22c198064c5c90202833162a5edb719dd0607070c549e31345785a2623281e7",
    "proprio_projector--120000_checkpoint.pt": "1bd796dfeb4becda52baf09c9fbba6121818a57069750e925d5769f9a02feb04",
}


@dataclass(frozen=True)
class OmniVLAPolicyConfig:
    source_dir: Path
    checkpoint_dir: Path
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dir", Path(self.source_dir).expanduser().resolve())
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir).expanduser().resolve())
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty string")

    def report(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_dir"] = str(self.source_dir)
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        return payload


@dataclass(frozen=True)
class OmniVLAInference:
    inference_index: int
    goal_pose: np.ndarray
    normalized_waypoints: np.ndarray
    metric_waypoints: np.ndarray
    selected_waypoint: np.ndarray
    command: np.ndarray
    wall_latency_s: float

    def summary(self) -> dict[str, Any]:
        return {
            "inference_index": int(self.inference_index),
            "goal_pose_forward_left_cos_sin": self.goal_pose.tolist(),
            "normalized_waypoints": self.normalized_waypoints.tolist(),
            "metric_waypoints": self.metric_waypoints.tolist(),
            "selected_waypoint": self.selected_waypoint.tolist(),
            "command_vx_vy_wz": self.command.tolist(),
            "wall_latency_s": float(self.wall_latency_s),
        }
