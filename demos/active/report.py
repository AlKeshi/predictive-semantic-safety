import time
from collections import Counter

import numpy as np

from pss.io import save_json

from .render import render_video


class Report:
    def report(self):
        xs = np.array([r["robot"][0] for r in self.rows])
        summary = dict(
            schema="active-caw-scene-inventory-uncalibrated-v3",
            pss=self.semantic_enabled,
            calibrated=False,
            scripted_robot=False,
            oracle_ground_state=self.semantic_enabled,
            grounded_ids=sorted(self.ground.grounded),
            goal=self.goal.tolist(),
            goal_time=self.goal_time,
            duration=self.duration,
            first_contact=next((c for c in self.contacts if c["kind"] == "hazard"), None),
            first_environment_contact=next(
                (c for c in self.contacts if c["kind"] == "environment"), None
            ),
            contacts=self.contacts,
            ball_release=self.initial["ball_release"],
            ball_impact=self.impact,
            backward_excursion=float((np.maximum.accumulate(xs) - xs).max()),
            reverse_seconds=sum((r["command"][0] < -0.03 for r in self.rows)) * 0.02,
            accepted_updates=sum((bool(u.get("accepted")) for u in self.fusion_updates)),
            updates=len(self.fusion_updates),
            semantic_calls=len(self.updates),
            fresh_model_calls=sum(("request_id" in u for u in self.updates)),
            minimum_trunk_height=self.min_height,
            statuses=dict(Counter((t["diagnostic"]["status"] for t in self.trace))),
            elapsed_s=time.monotonic() - self.started,
            q_m=0.0,
            terminal_continuation_verified=False,
            successful_recorded_demo=self.goal_time is not None
            and (not self.contacts)
            and (self.min_height > 0.2),
        )
        save_json(self.out / "summary.json", summary)
        save_json(self.out / "report.json", dict(summary, rows=self.rows))
        if self.render:
            render_video(self.out, self.semantic_enabled)
        return summary
