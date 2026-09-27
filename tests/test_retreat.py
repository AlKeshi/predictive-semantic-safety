from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pss.control import CodeWorldController, Forecast, RetreatConfig
from pss.control.retreat import RetreatFlow, controller_descriptor, retreat_rows


def forecast(name="a", origin=0.0, center=(1.8, 0.1), velocity=(-0.25, 0.0), growth=0.06, end=3.0):
    offsets = np.array([0.0, 0.7, 1.4, end])
    return Forecast(
        name,
        origin,
        origin + offsets,
        np.array(center) + offsets[:, None] * velocity,
        0.2 + growth * offsets,
    )


def frozen(controller, forecasts, now=0.0):
    pairs = [controller._freeze(f, now) for f in forecasts]
    return (tuple((p[0] for p in pairs)), tuple((p[1] for p in pairs)))


@pytest.mark.parametrize("velocity", [(0.0, 0.0), (4.0, -3.0)])
def test_full_feedback_jacobian_including_saturation_and_threat_weights(velocity):
    c = CodeWorldController(backup_policy="retreat", static_obstacles=[((-0.8, 2), 0.15)])
    family, _ = frozen(c, [forecast(), forecast("b", center=(2.0, -2.0))])
    flow = RetreatFlow(c, family)
    x = np.r_[[-0.4, 0.2], velocity]
    eps = 1e-05
    actual, jac = flow.evaluate(x, 0.3)
    numerical = np.column_stack(
        [
            (flow.evaluate(x + eps * e, 0.3)[0] - flow.evaluate(x - eps * e, 0.3)[0]) / (2 * eps)
            for e in np.eye(4)
        ]
    )
    np.testing.assert_allclose(jac, numerical, atol=2e-08)
    assert np.max(np.abs(actual[2:])) <= c.max_accel


def test_variational_flow_matches_independent_solver_and_initial_state_differences():
    c = CodeWorldController(backup_policy="retreat")
    family, _ = frozen(c, [forecast()])
    flow = RetreatFlow(c, family)
    x = np.array([-0.2, 0.15, 0.3, -0.1])
    t = 0.13
    end = 1.31
    value, jac = flow.rollout(x, t, [end])[end]
    independent = solve_ivp(
        lambda now, y: flow.evaluate(y, now)[0], (t, end), x, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(value, independent.y[:, -1], atol=3e-07)
    eps = 1e-05
    numerical = np.column_stack(
        [
            (
                flow.rollout(x + eps * e, t, [end])[end][0]
                - flow.rollout(x - eps * e, t, [end])[end][0]
            )
            / (2 * eps)
            for e in np.eye(4)
        ]
    )
    np.testing.assert_allclose(jac, numerical, atol=2e-07)


def test_fixed_absolute_endpoint_rows_and_terminal_derivatives():
    c = CodeWorldController(backup_policy="retreat", static_obstacles=[((-0.4, 3.0), 0.2)])
    family, grids = frozen(c, [forecast(), forecast("b", center=(2.0, 2.0), velocity=(-0.1, 0.0))])
    x = np.array([-0.2, 0.15, 0.3, -0.1])
    now = 0.13
    u = np.array([0.2, -0.1])
    eps = 1e-05
    rows, _, _ = retreat_rows(c, x, now, family, grids)
    direction = np.r_[x[2:], u]
    plus = retreat_rows(c, x + eps * direction, now + eps, family, grids)[0]
    minus = retreat_rows(c, x - eps * direction, now - eps, family, grids)[0]

    def key(r):
        return (r["kind"], r["object_id"], r["time"])

    p = {key(r): r for r in plus}
    m = {key(r): r for r in minus}
    checked = set()
    for row in rows:
        if row["time"] <= now + eps:
            continue
        numerical = (p[key(row)]["margin"] - m[key(row)]["margin"]) / (2 * eps)
        analytic = np.dot(row["coefficients"], u) - row["lower_bound"] - c.alpha * row["margin"]
        assert numerical == pytest.approx(analytic, abs=3e-05)
        checked.add(row["kind"])
    assert checked == {
        "predictive",
        "static_backup",
        "terminal_clearance",
        "terminal_separation",
        "terminal_static_hocbf",
    }


def test_rollout_splits_piecewise_velocity_and_growth_knots():
    c = CodeWorldController(backup_policy="retreat")
    raw = forecast()
    raw.centers[2:] += np.array([0.1, 0.15])
    raw.radii[2:] += 0.08
    family, _ = frozen(c, [raw])
    flow = RetreatFlow(c, family)
    refined = CodeWorldController(
        backup_policy="retreat", retreat_config=replace(RetreatConfig(), integration_dt=0.005)
    )
    other = RetreatFlow(refined, family)
    x = np.array([-0.3, 0.1, 0.2, 0])
    np.testing.assert_allclose(
        flow.rollout(x, 0, [2.0])[2.0][0], other.rollout(x, 0, [2.0])[2.0][0], atol=2e-05
    )


def test_stationary_robot_retreats_and_terminal_guard_does_not_require_stopping():
    c = CodeWorldController(backup_policy="retreat")
    x = np.zeros(4)
    assert c.update([forecast(center=(1.8, 0.0))], 0, x)["accepted"]
    assert c.backup_input(x, 0)[0] < -0.5
    np.testing.assert_array_equal(CodeWorldController().backup_input(x, 0), [0, 0])
    u, d = c.command(x, [0.4, 0], 0.1)
    assert d["feasible"] and (not d["terminal_continuation_verified"])
    assert not any((row["kind"] == "terminal" for row in d["enforced_rows"]))
    for row in d["enforced_rows"]:
        assert np.dot(row["coefficients"], u) >= row["lower_bound"] - 1e-05


def test_rejected_update_retains_filter_and_explicit_invalidation_uses_retreat():
    c = CodeWorldController(backup_policy="retreat")
    x = np.zeros(4)
    assert c.update([forecast(center=(1.8, 0.0))], 0, x)["accepted"]
    before = deepcopy(c.snapshot()["forecasts"])
    assert not c.update([forecast(origin=0.1, center=(0.0, 0.0))], 0.1, x)["accepted"]
    assert c.snapshot()["forecasts"] == before
    u, d = c.command(x, [0.5, 0], 0.1)
    assert d["feasible"] and d["enforced_rows"]
    for row in d["enforced_rows"]:
        assert np.dot(row["coefficients"], u) >= row["lower_bound"] - 1e-05
    c.invalidate("current observation contradicts forecast")
    u, d = c.command(x, [0.5, 0], 0.1)
    assert u[0] < 0 and (not d["feasible"])
    assert d["status"].startswith("uncertified_retreat")
    assert d["execute_backup"]
    assert d["enforced_rows"] == [] and (not d["certificate_valid"])


@pytest.mark.parametrize("when", [3.0, 3.2, -0.1])
def test_expired_or_reversed_time_never_extrapolates_escape(when):
    c = CodeWorldController(backup_policy="retreat")
    assert c.update([forecast()], 0, np.zeros(4))["accepted"]
    _, d = c.command([0, 0, 0.2, 0], [0.5, 0], when)
    assert d["fallback_source"] == "braking" and d["enforced_rows"] == []
    assert not d["execute_backup"]


def test_retreat_into_wall_is_rejected_and_multiple_hazards_are_not_dropped():
    c = CodeWorldController(backup_policy="retreat", static_obstacles=[((-0.85, 0.0), 0.2)])
    x = np.zeros(4)
    decision = c.update([forecast(center=(1.8, 0.0), growth=0.2)], 0, x)
    assert not decision["accepted"]
    assert not c.snapshot()["forecasts"]
    c = CodeWorldController(backup_policy="retreat")
    family, grids = frozen(c, [forecast(), forecast("b", center=(2.0, 1.0))])
    rows, _, _ = retreat_rows(c, x, 0, family, grids)
    assert {r["object_id"] for r in rows if r["kind"] == "predictive"} == {"a", "b"}


def test_unequal_support_uses_common_deadline_without_clamping_other_hazard():
    c = CodeWorldController(backup_policy="retreat")
    family, grids = frozen(c, [forecast(end=2.0), forecast("b", center=(2, 1), end=3.0)])
    rows, _, _ = retreat_rows(c, np.zeros(4), 0.1, family, grids)
    assert max((r["time"] for r in rows)) == 2.0
    assert {r["object_id"] for r in rows if r["kind"] == "terminal_clearance"} == {"a", "b"}
    with pytest.raises(ValueError):
        retreat_rows(c, np.zeros(4), 2.1, family, grids)


def test_surrounding_static_obstacles_allow_hocbf_admissible_corridor_motion():
    obstacles = [
        ((x + dx, side * 3.6), 0.78)
        for side in [-1, 1]
        for x in [-3.0, -0.4, 2.2, 4.8, 7.4]
        for dx in [-0.8, 0.0, 0.8]
    ]
    c = CodeWorldController(backup_policy="retreat", static_obstacles=obstacles)
    x = np.array([-0.3, 0.0, 0.6, 0.0])
    f = Forecast(
        "expanding",
        2.8,
        2.8 + np.array([0, 0.5, 1, 2, 3, 4]),
        np.tile([1.4, 0.25], (6, 1)),
        np.array([0.15, 0.17, 0.2, 0.3, 0.4, 0.5]),
    )
    assert c.update([f], 2.8, x)["accepted"]
    u, d = c.command(x, [1.0, 0.0], 2.8)
    assert d["feasible"]
    rows = d["enforced_rows"]
    terminal_static = [r for r in rows if r["kind"] == "terminal_static_hocbf"]
    assert len(terminal_static) == len(obstacles)
    assert all((r["margin"] >= 0 for r in terminal_static))
    assert all((r["object_id"] == "expanding" for r in rows if r["kind"] == "terminal_separation"))
    assert all((np.dot(r["coefficients"], u) >= r["lower_bound"] - 1e-05 for r in rows))


def test_configuration_and_pipeline_identity():
    for value in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            RetreatConfig(gain=value)
    with pytest.raises(ValueError):
        CodeWorldController(backup_policy="fire")
    assert controller_descriptor("retreat") != controller_descriptor("braking")


def test_infeasible_qp_fallback_is_bounded_and_never_claims_predictive_enforcement(monkeypatch):
    c = CodeWorldController(backup_policy="retreat")
    state = np.zeros(4)
    assert c.update([forecast()], 0, state)["accepted"]
    monkeypatch.setattr(c, "_solve", lambda *_: (None, "qp_infeasible"))
    u, d = c.command(state, [0.4, 0], 0.1)
    assert np.max(np.abs(u)) <= c.max_accel
    assert not d["feasible"] and (not d["certificate_valid"])
    assert not d["execute_backup"]
    assert d["enforced_rows"] == []


def test_active_family_cannot_drop_a_hazard_and_initial_missing_prediction_is_not_nominal():
    c = CodeWorldController(backup_policy="retreat")
    _, d = c.command([0, 0, 0.2, 0], [0.4, 0], 0)
    assert not d["feasible"] and d["status"].startswith("brake:")
    x = np.zeros(4)
    family = [forecast(), forecast("b", center=(2.0, 1.0))]
    assert c.update(family, 0, x)["accepted"]
    before = deepcopy(c.snapshot()["forecasts"])
    assert not c.update([forecast(origin=0.1)], 0.1, x)["accepted"]
    assert c.snapshot()["forecasts"] == before


def test_braking_calibration_record_is_rejected_for_retreat(tmp_path):
    from pss.calibration import CalibrationStore, fit_calibration, load_calibration

    pipeline = {"synthetic_test": True, "controller": controller_descriptor("braking")}

    def corpus(prefix):
        return dict(
            pipeline=pipeline,
            indices=[dict(object_id="a", origin_time=0.0, future_time=1.0)],
            shape_radii={"a": 0.2},
            sessions=[
                dict(
                    session_id=f"{prefix}-{i}",
                    acquisition_record_digest="1" * 64,
                    prediction=[[0.0, 0.0]],
                    truth=[[0.1, 0.0]],
                )
                for i in range(3)
            ],
        )

    record = fit_calibration(corpus("train"), corpus("cal"), delta=0.5)
    path = CalibrationStore(tmp_path).save(record)
    different = dict(pipeline, controller=controller_descriptor("retreat"))
    with pytest.raises(ValueError, match="does not match"):
        load_calibration(different, path=path)
