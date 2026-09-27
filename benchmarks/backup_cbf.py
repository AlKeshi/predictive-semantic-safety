from copy import deepcopy

import numpy as np

from pss.control.code_world import CodeWorldController, Forecast

from .plain_cbf import PlainCBF


def admit_current_geometry(bottom, height=0.55):
    if not np.isfinite(bottom):
        raise ValueError("Invalid measured current solid bound")
    return bool(bottom <= height)


class _CommonStaticController(CodeWorldController):
    def __init__(self):
        super().__init__(inactive=True)
        self.current_static_rows = []

    def _constraints(self, x, now, family, grids):
        rows, minimum, terminal = super()._constraints(x, now, family, grids)
        static = deepcopy(self.current_static_rows)
        for row in static:
            row["hocbf_membership"] = row["psi1"]
        rows.extend(static)
        values = [row["margin"] for row in rows if row["kind"] != "terminal"]
        return (rows, min(values) if values else minimum, terminal)


class MatchedBackupCBF:
    def __init__(self, horizon):
        self.horizon = float(horizon)
        self.controller = _CommonStaticController()
        self.static = PlainCBF()

    def command(self, state, nominal, now, obstacles):
        dynamic = [o for o in obstacles if o["dynamic"]]
        static = [o for o in obstacles if not o["dynamic"]]
        self.controller.current_static_rows = self.static.geometry_rows(state, now, static)
        decision = {"accepted": False, "status": "no_observed_ground_obstacle"}
        if dynamic:
            times = float(now) + np.linspace(0.0, self.horizon, 11)
            forecasts = [
                Forecast(
                    o["object_id"],
                    float(now),
                    times,
                    np.tile(o["center"], (len(times), 1)),
                    o["radius"],
                )
                for o in dynamic
            ]
            decision = self.controller.update(forecasts, now, state)
            if not decision["accepted"]:
                self.controller.invalidate("current_geometry_invalid")
        elif self.controller.snapshot()["forecasts"]:
            decision = self.controller.update([], now, state)
        acceleration, diagnostic = self.controller.command(state, nominal, now)
        diagnostic.update(
            geometry_update=decision,
            current_obstacle_count=len(obstacles),
            dynamic_obstacle_count=len(dynamic),
            future_world_prediction=False,
            input_model="same current native geometry as Plain CBF; frozen during braking rollout",
        )
        return (acceleration, diagnostic)
