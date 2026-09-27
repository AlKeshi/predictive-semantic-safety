from copy import deepcopy

import numpy as np

from pss.control.code_world import CodeWorldController

from .recovery_objective import recovery_objective


class RecoveryController(CodeWorldController):
    def command(self, state, nominal_accel, now):
        requested = nominal_accel
        recovery = {"active": False, "reason": "no_valid_committed_family"}
        if self._emergency_reason is None and self._family:
            try:
                x, t = self._state(state, now)
                if self._last_time is not None and t < self._last_time - 1e-09:
                    raise ValueError("time reversed")
                occupancy = []
                for f, grid in zip(self._family, self._grids, strict=True):
                    if t < f.times[0] - 1e-09 or t >= f.times[-1] - 1e-08:
                        raise ValueError("unsupported forecast")
                    times = np.unique(np.r_[t, grid[grid > t + 1e-09]])
                    samples = []
                    for when in times:
                        center, radius = self.geometry_at(f, float(when))
                        samples.append([*center, radius + self.robot_radius + self.clearance])
                    occupancy.append(samples)
                requested, recovery = recovery_objective(x, nominal_accel, occupancy)
            except (ValueError, TypeError, FloatingPointError):
                requested = np.full(2, np.nan)
                recovery = {"active": False, "reason": "invalid_recovery_input"}
        control, diagnostic = super().command(state, requested, now)
        diagnostic["recovery_reference"] = recovery
        try:
            recorded = np.asarray(requested, dtype=float)
            diagnostic["qp_objective_acceleration"] = (
                recorded.tolist()
                if recorded.shape == (2,) and np.all(np.isfinite(recorded))
                else None
            )
        except (ValueError, TypeError):
            diagnostic["qp_objective_acceleration"] = None
        diagnostic["recovery_enforced"] = bool(diagnostic["feasible"] and recovery["active"])
        self._last_command = deepcopy(diagnostic)
        return (control, diagnostic)
