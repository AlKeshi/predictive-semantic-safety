from __future__ import annotations

from copy import deepcopy

import numpy as np

from .constraints import static_rows
from .retreat import RETREAT_SCHEMA


class ControllerCommands:
    def command(self, state, nominal_accel, now) -> tuple[np.ndarray, dict]:
        rows = []
        minimum = terminal = None
        try:
            x, t = self._state(state, now)
        except (TypeError, ValueError, OverflowError):
            diagnostic = {
                "status": "brake_invalid_state",
                "feasible": False,
                "certificate_valid": False,
                "coverage": "uncertainty_in_forecast_geometry",
                "enforced_rows": [],
                "acceleration": [0.0, 0.0],
                "backup_policy": self.backup_policy,
                "terminal_continuation_verified": False,
            }
            self._last_command = diagnostic
            return (np.zeros(2), deepcopy(diagnostic))
        brake = np.clip(-self.brake_gain * x[2:], -self.max_accel, self.max_accel)
        try:
            nominal = np.asarray(nominal_accel, dtype=float)
        except (TypeError, ValueError, OverflowError):
            nominal = np.full(2, np.nan)
        reason = self._emergency_reason
        if nominal.shape != (2,) or not np.all(np.isfinite(nominal)):
            reason = "invalid_nominal"
        if self._last_time is not None and t < self._last_time - 1e-09:
            reason = "time_reversed"
        if not self._family and (not self.inactive):
            reason = reason or "awaiting_forecast"
        control = None
        status = reason
        if reason is None:
            try:
                rows, minimum, terminal = self._constraints(x, t, self._family, self._grids)
                if not self._member(rows, terminal):
                    status = "backup_membership_lost"
                elif rows:
                    control, status = self._solve(nominal, rows)
                else:
                    control = np.clip(nominal, -self.max_accel, self.max_accel)
                    status = "inactive_nominal"
            except (ValueError, FloatingPointError):
                status = "forecast_expired_or_invalid"
        feasible = control is not None
        fallback_source = None
        if not feasible:
            control = brake
            rows = []
            status = f"brake:{status}"
            fallback_source = "braking"
            if self.backup_policy == "retreat" and self._family:
                try:
                    proposed = self.backup_input(x, t, self._family)
                    emergency_rows = static_rows(self, x, t)
                    checked, emergency_status = self._solve(proposed, emergency_rows)
                    if checked is not None and emergency_status == "solved":
                        control = checked
                        rows = emergency_rows
                        fallback_source = "retreat_static_filtered"
                        status = f"uncertified_retreat:{status.removeprefix('brake:')}"
                except (ValueError, FloatingPointError):
                    pass
        diagnostic = {
            "status": status,
            "feasible": feasible,
            "intervened": bool(
                nominal.shape != (2,)
                or not np.all(np.isfinite(nominal))
                or np.linalg.norm(control - nominal) > 0.001
            ),
            "certificate_valid": False,
            "coverage": "uncertainty_in_forecast_geometry",
            "q_m": 0.0,
            "min_margin": minimum,
            "terminal_margin": terminal,
            "enforced_rows": rows,
            "num_constraints": len(rows),
            "acceleration": control.tolist(),
            "time": t,
            "backup_policy": self.backup_policy,
            "controller_schema": RETREAT_SCHEMA
            if self.backup_policy == "retreat"
            else "braking_v1",
            "fallback_source": fallback_source,
            "execute_backup": fallback_source == "retreat_static_filtered",
            "terminal_continuation_verified": False,
            "terminal_guard": "dynamic_separation_and_static_hocbf"
            if self.backup_policy == "retreat"
            else "low_speed",
        }
        self._last_time = max(t, self._last_time) if self._last_time is not None else t
        self._last_command = deepcopy(diagnostic)
        return (control.copy(), diagnostic)
