from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Go1WorldConfig:
    sim_dt: float = 0.004
    policy_dt: float = 0.02
    settle_time: float = 0.45
    hold_stack_during_settle: bool = True
    stack_settle_time: float = 0.0
    reject_nonquiescent_stack: bool = False
    stack_quiescent_linear_speed: float = 0.04
    stack_quiescent_angular_speed: float = 0.25
    duration: float = 9.5
    fps: int = 20
    width: int = 1600
    height: int = 900
    top_width: int = 480
    top_height: int = 300
    rgbd_width: int = 640
    rgbd_height: int = 480
    rgbd_fovy_deg: float = 70.0
    rgbd_max_depth: float = 10.0
    full_body_collisions: bool = False
    impulse_duration: float = 0.06
    route_height: float = 0.026
    omnivla_goal_beacon: bool = False

    def __post_init__(self) -> None:
        if self.sim_dt <= 0.0 or self.policy_dt <= 0.0:
            raise ValueError("Simulation and policy timesteps must be positive")
        ratio = self.policy_dt / self.sim_dt
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("policy_dt must be an integer multiple of sim_dt")
        if self.duration <= 0.0 or self.fps <= 0:
            raise ValueError("Duration and fps must be positive")
        if self.settle_time < 0.0 or self.stack_settle_time < 0.0:
            raise ValueError("Settling durations must be nonnegative")
        if self.stack_quiescent_linear_speed <= 0.0 or self.stack_quiescent_angular_speed <= 0.0:
            raise ValueError("Stack quiescence speed thresholds must be positive")
        if (
            min(
                self.width,
                self.height,
                self.top_width,
                self.top_height,
                self.rgbd_width,
                self.rgbd_height,
            )
            <= 0
        ):
            raise ValueError("Render dimensions must be positive")
        if not 1.0 < self.rgbd_fovy_deg < 179.0:
            raise ValueError("RGB-D vertical field of view must lie in (1, 179) degrees")
        if self.rgbd_max_depth <= 0.0:
            raise ValueError("RGB-D maximum depth must be positive")
