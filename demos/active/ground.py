from dataclasses import asdict

import numpy as np

from .fusion import fused_family


class Ground:
    def ground_update(self):
        measurements = {}
        for name in self.floor_seen | self.ground.grounded:
            body = self.model.body(name).id
            twist = np.zeros(6)
            self.mj.mj_objectVelocity(
                self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, body, twist, 0
            )
            measurements[name] = (self.data.xpos[body, :2].copy(), twist[3:5].copy())
        newly = self.ground.observe(self.floor_seen, measurements, self.now)
        entry = dict(
            time=self.now,
            newly_grounded=newly,
            grounded_ids=sorted(self.ground.grounded),
            ground_measurements={n: [p.tolist(), v.tolist()] for n, (p, v) in measurements.items()},
        )
        before = self.controller.snapshot()["forecasts"]
        self.live_family, self.live_sources = ([], {})
        try:
            if self.semantic_failed and set(self.names) - self.ground.grounded:
                raise ValueError("Active airborne semantic observation invalid")
            family, sources = fused_family(self.semantic, self.ground, self.names, self.now)
            self.live_family, self.live_sources = (family, sources)
            entry["sources"] = sources
            entry["proposed_forecasts"] = [asdict(f) for f in family]
            entry.update(self.controller.update(family, self.now, self.state))
        except ValueError as error:
            entry["error"] = str(error)
            entry.update(self.controller.update([], self.now, self.state))
        if not entry["accepted"]:
            assert before == self.controller.snapshot()["forecasts"]
        entry["committed_forecasts"] = self.controller.snapshot()["forecasts"]
        self.fusion_updates.append(entry)
        self.next_ground += 0.1
