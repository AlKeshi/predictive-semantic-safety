from pathlib import Path

import numpy as np


class OmniVLA:
    def __init__(self, source, checkpoint, device="cuda"):
        from pss.vla.omnivla_policy import OmniVLAPolicy, OmniVLAPolicyConfig

        self.policy = OmniVLAPolicy(OmniVLAPolicyConfig(Path(source), Path(checkpoint), device))
        self.next_time = 0.0
        self.latest = np.zeros(3)
        self.receipt = {}

    def command(self, rgb, state, yaw, goal, now, warmup, reached):
        if now < warmup or reached:
            return (np.zeros(3), {"method": "omnivla", "reason": "common_warmup_or_goal_stop"})
        if now + 1e-08 >= self.next_time:
            co, si = (np.cos(yaw), np.sin(yaw))
            body_goal = np.array([[co, si], [-si, co]]) @ (goal - state[:2])
            self.latest = self.policy.command(rgb, body_goal_delta=body_goal, body_goal_yaw=-yaw)
            if not np.all(np.isfinite(self.latest)):
                raise ValueError("Nonfinite OmniVLA action")
            self.receipt = self.policy.last_inference.summary()
            self.next_time = now + 1.0 / 3.0
        return (self.latest.copy(), {"method": "omnivla", "model_inference": self.receipt})
