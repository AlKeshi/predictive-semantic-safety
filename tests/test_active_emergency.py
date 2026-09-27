import numpy as np

from demos.active.emergency import fresh_ground_fallback
from pss.control.code_world import CodeWorldController
from pss.control.forecast import Forecast


def test_fresh_ground_feedback_is_uncertified_and_does_not_commit_rejected_family():
    c = CodeWorldController(backup_policy="retreat")
    state = np.array([0.0, 0.0, 0.0, 0.0])
    f = Forecast("rolling", 1.0, np.array([1.0, 3.5]), np.array([[0.5, 0.0], [-0.5, 0.0]]), 0.2)
    assert not c.update([f], 1.0, state)["accepted"]
    before = c.snapshot()["forecasts"]
    acceleration, d = c.command(state, np.zeros(2), 1.0)
    action, result = fresh_ground_fallback(
        c,
        state,
        1.0,
        np.zeros(2),
        acceleration,
        d,
        [f],
        {"rolling": {"source": "oracle_current_ground_position_velocity"}},
    )
    assert action[0] < 0
    assert (
        result["execute_backup"] and (not result["feasible"]) and (not result["certificate_valid"])
    )
    assert c.snapshot()["forecasts"] == before == []


def test_expired_emergency_geometry_cannot_drive_retreat():
    c = CodeWorldController(backup_policy="retreat")
    state = np.zeros(4)
    f = Forecast("rolling", 0.0, np.array([0.0, 1.0]), np.array([[1.0, 0.0], [0.5, 0.0]]), 0.2)
    action, d = c.command(state, np.zeros(2), 1.1)
    got, result = fresh_ground_fallback(
        c,
        state,
        1.1,
        np.zeros(2),
        action,
        d,
        [f],
        {"rolling": {"source": "oracle_current_ground_position_velocity"}},
    )
    assert np.array_equal(action, got)
    assert result == d


def test_fresh_ground_emergency_still_enforces_known_wall_rows():
    c = CodeWorldController(backup_policy="retreat", static_obstacles=[([-0.8, 0.0], 0.12)])
    state = np.array([0.0, 0.0, -0.2, 0.0])
    f = Forecast("rolling", 1.0, np.array([1.0, 3.5]), np.array([[0.5, 0.0], [-0.5, 0.0]]), 0.2)
    action, d = c.command(state, np.zeros(2), 1.0)
    got, result = fresh_ground_fallback(
        c,
        state,
        1.0,
        np.zeros(2),
        action,
        d,
        [f],
        {"rolling": {"source": "oracle_current_ground_position_velocity"}},
    )
    assert result["execute_backup"] and (not result["feasible"])
    assert result["enforced_rows"]
    assert all(
        (
            np.dot(r["coefficients"], got) >= r["lower_bound"] - 1e-06
            for r in result["enforced_rows"]
        )
    )
