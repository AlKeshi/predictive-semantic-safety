from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pss.control import CodeWorldController, Forecast, braking_flow


def forecast(name="block", origin=0.0, center=(3.0, 0.8)):
    return Forecast(
        name, origin, origin + np.array([0.0, 0.5, 1.5, 3.0]), np.tile(center, (4, 1)), 0.13
    )


@pytest.mark.parametrize("speed", [0.0, 0.2, 0.79, 0.8, 0.9, 1.5, -1.5])
def test_exact_braking_and_sensitivity(speed):
    x = np.array([0.1, -0.2, speed, -speed])
    for duration in [0.01, 0.4, 1.0, 2.5]:
        p, v, sensitivity = braking_flow(x, duration)
        numerical = solve_ivp(
            lambda _, y: np.r_[y[2:], np.clip(-2.5 * y[2:], -2, 2)],
            [0, duration],
            x,
            rtol=1e-10,
            atol=1e-12,
        ).y[:, -1]
        np.testing.assert_allclose(np.r_[p, v], numerical, atol=1e-08)
        for axis in range(2):
            epsilon = np.zeros(4)
            epsilon[axis + 2] = 1e-06
            pp, vp, _ = braking_flow(x + epsilon, duration)
            pm, vm, _ = braking_flow(x - epsilon, duration)
            np.testing.assert_allclose(
                [(pp[axis] - pm[axis]) / 2e-06, (vp[axis] - vm[axis]) / 2e-06],
                sensitivity[:, axis],
                atol=1e-06,
            )


def test_absolute_deadline_derivative():
    c = CodeWorldController(inactive=True)
    f, _ = c._freeze(forecast(), 0.0)
    x = np.array([0, 0, 0.6, -0.2])
    now, when, dt = (0.3, 1.4, 1e-06)
    u = np.array([0.4, -0.1])
    row = c._predictive_row(x, now, when, f)
    direction = np.r_[x[2:], u]
    plus = c._predictive_row(x + dt * direction, now + dt, when, f)["margin"]
    minus = c._predictive_row(x - dt * direction, now - dt, when, f)["margin"]
    derivative = (plus - minus) / (2 * dt)
    expected = np.dot(row["coefficients"], u) - row["lower_bound"] - c.alpha * row["margin"]
    assert derivative == pytest.approx(expected, abs=1e-07)


def test_independent_hazards_have_separate_hard_rows():
    c = CodeWorldController()
    x = [0, 0, 0.2, 0]
    assert c.update([forecast("a"), forecast("b", center=(3, -0.8))], 0, x)["accepted"]
    u, diagnostic = c.command(x, [0.3, 0.1], 0.1)
    assert diagnostic["feasible"]
    assert {r["object_id"] for r in diagnostic["enforced_rows"] if r["kind"] == "predictive"} == {
        "a",
        "b",
    }
    for row in diagnostic["enforced_rows"]:
        assert np.dot(row["coefficients"], u) >= row["lower_bound"] - 1e-05


def test_rejected_family_preserves_state_and_enforces_committed_rows():
    c = CodeWorldController()
    x = np.array([0, 0, 0.2, 0])
    assert c.update([forecast()], 0, x)["accepted"]
    before = deepcopy(c.snapshot())
    rejection = c.update([forecast(origin=0.1, center=(0, 0))], 0.1, x)
    assert not rejection["accepted"]
    assert c.snapshot() == before
    u, diagnostic = c.command(x, [0.3, 0], 0.1)
    assert diagnostic["feasible"] and diagnostic["enforced_rows"]
    for row in diagnostic["enforced_rows"]:
        assert np.dot(row["coefficients"], u) >= row["lower_bound"] - 1e-05


@pytest.mark.parametrize("failure", ["expired", "unsafe", "invalidated"])
def test_rejected_update_cannot_revive_invalid_committed_filter(failure):
    c = CodeWorldController(static_obstacles=[([0, 2], 0.2)])
    x = np.array([0, 0, 0.2, 0])
    assert c.update([forecast()], 0, x)["accepted"]
    now = 3.1 if failure == "expired" else 0.1
    if failure == "unsafe":
        x[1] = 2
    if failure == "invalidated":
        c.invalidate("required prediction missing")
    assert not c.update([], now, x)["accepted"]
    _, diagnostic = c.command(x, [0.3, 0], now)
    assert not diagnostic["feasible"] and not diagnostic["enforced_rows"]


def test_active_object_cannot_silently_disappear():
    c = CodeWorldController()
    x = [0, 0, 0.2, 0]
    c.update([forecast("a"), forecast("b", center=(3, -0.8))], 0, x)
    assert not c.update([forecast("a", origin=0.1)], 0.1, x)["accepted"]


def test_forecast_input_and_snapshot_do_not_mutate_committed_arrays():
    c = CodeWorldController()
    f = forecast()
    c.update([f], 0, [0, 0, 0.2, 0])
    original = deepcopy(c.snapshot()["forecasts"])
    f.centers[:] = 100
    snapshot = c.snapshot()
    snapshot["forecasts"][0]["centers"][0][0] = -100
    assert c.snapshot()["forecasts"] == original


def test_expired_family_brakes_and_is_not_clamped():
    c = CodeWorldController()
    c.update([forecast()], 0, [0, 0, 0.2, 0])
    _, diagnostic = c.command([0, 0, 0.2, 0], [0.3, 0], 3.1)
    assert not diagnostic["feasible"] and (not diagnostic["enforced_rows"])
    with pytest.raises(ValueError):
        c.geometry_at(forecast(), 3.1)


def test_no_hazard_is_distinct_from_missing_active_forecast():
    u, a = CodeWorldController(inactive=True).command([0, 0, 0.2, 0], [0.3, 0], 1)
    np.testing.assert_array_equal(u, [0.3, 0])
    assert a["status"] == "inactive_nominal"
    u, b = CodeWorldController().command([0, 0, 0.2, 0], [0.3, 0], 1)
    np.testing.assert_array_equal(u, [-0.5, 0])
    assert not b["feasible"]


def test_qp_rejects_invalid_rows_and_public_controller_has_no_tuned_quantile():
    c = CodeWorldController(inactive=True)
    assert c._solve(np.zeros(2), [{"coefficients": [np.nan, 0], "lower_bound": 0}])[0] is None
    with pytest.raises(TypeError):
        CodeWorldController(q=0.2)
