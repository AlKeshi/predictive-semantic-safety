from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.demo_margin import pad_forecasts
from benchmarks.ground_handoff import GroundHandoff, ground_motion_ready
from benchmarks.navigation import run_navigation
from benchmarks.run import run
from pss.control.code_world import CodeWorldController, Forecast
from pss.control.recovery import RecoveryController
from pss.control.recovery_objective import recovery_objective


def item(name="debris", **changes):
    result = dict(
        object_id=name,
        center=np.array([2.0, 0.4]),
        radius=0.2,
        velocity=np.zeros(2),
        grounded=True,
        low_pose=True,
        sphere=False,
        linear_speed=0.0,
        angular_speed=0.0,
    )
    result.update(changes)
    return result


def handoff(tmp_path, items):
    h = GroundHandoff.__new__(GroundHandoff)
    h.world = SimpleNamespace(
        scenario=SimpleNamespace(
            dynamic_obstacles=[SimpleNamespace(name=x["object_id"]) for x in items]
        )
    )
    h.out, h.static = (tmp_path, [])
    h.history, h.active, h.reserve, h.events = (deque(maxlen=31), False, 0.0, [])
    h.previous, h.previous_time, h.latest = (None, None, [])
    h.measured = lambda: deepcopy(items)
    return h


def fill(h, controller):
    for i in range(31):
        controller = h.update(controller, np.zeros(4), i * 0.02)
    return controller


@pytest.mark.parametrize(
    "field,value",
    [("grounded", False), ("low_pose", False), ("linear_speed", 0.1), ("angular_speed", 0.2)],
)
def test_airborne_unsettled_or_upright_inventory_preserves_old_controller(tmp_path, field, value):
    c = CodeWorldController()
    before = deepcopy(c.snapshot())
    h = handoff(tmp_path, [item("a"), item("b", **{field: value})])
    assert fill(h, c) is c
    assert c.snapshot() == before and (not h.active) and (not h.events)


def test_rejected_handoff_preserves_candidate_and_source_state(tmp_path):
    c = CodeWorldController()
    before = deepcopy(c.snapshot())
    h = handoff(tmp_path, [item(center=np.zeros(2))])
    assert fill(h, c) is c
    assert c.snapshot() == before and (not h.active) and (not h.events)
    assert h.previous is None


def test_complete_handoff_checks_every_independent_obstacle_and_gains(tmp_path):
    c = CodeWorldController(max_accel=1.3, brake_gain=1.7, alpha=2.2)
    h = handoff(tmp_path, [item("a"), item("b", center=np.array([2.0, -0.4]))])
    ground = fill(h, c)
    assert h.active and ground is not c
    assert ground.max_accel == 1.3 and ground.brake_gain == 1.7 and (ground.alpha == 2.2)
    _, diag = ground.command(np.zeros(4), np.array([0.3, 0.0]), 0.62)
    assert {r["object_id"] for r in diag["enforced_rows"]} == {"a", "b"}
    assert diag["feasible"] and (not diag["certificate_valid"])
    assert c.snapshot()["emergency_reason"] == "awaiting_forecast"
    h.measured = lambda: [item("a", grounded=False), item("b")]
    before = deepcopy(ground.snapshot())
    with pytest.raises(RuntimeError, match="premise revoked"):
        h.update(ground, np.zeros(4), 0.64)
    assert ground.snapshot() == before


def test_missing_inventory_and_reversed_time_fail(tmp_path):
    c = CodeWorldController()
    h = handoff(tmp_path, [item()])
    h.update(c, np.zeros(4), 0.0)
    with pytest.raises(ValueError, match="Non-increasing"):
        h.update(c, np.zeros(4), 0.0)
    h.measured = lambda: []
    with pytest.raises(ValueError, match="Incomplete"):
        h.update(c, np.zeros(4), 0.02)


def test_approaching_ball_and_excess_acceleration_block_handoff():
    a = [
        item("solid", center=np.zeros(2)),
        item("ball", sphere=True, velocity=np.array([-1.0, 0.0])),
    ]
    assert not ground_motion_ready(a, [(0.0, a), (0.02, a)], {0})
    a[1]["velocity"] = np.array([1.0, 0.0])
    assert ground_motion_ready(a, [(0.0, a), (0.02, a)], {0})
    b = deepcopy(a)
    b[1]["velocity"] += 0.1
    assert not ground_motion_ready(b, [(0.0, a), (0.02, b)], {0})


def test_recovery_rotation_symmetry_and_no_hazard_equivalence():
    state = np.array([0.0, 0.0, 0.1, 0.0])
    nominal = np.array([0.3, 0.0])
    occupied = [np.array([[0.7, 0.1, 0.6], [0.8, 0.1, 0.6]])]
    u, _ = recovery_objective(state, nominal, occupied)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    moved = [np.c_[o[:, :2] @ rotation.T, o[:, 2]] for o in occupied]
    v, _ = recovery_objective(
        np.r_[rotation @ state[:2], rotation @ state[2:]], rotation @ nominal, moved
    )
    np.testing.assert_allclose(v, rotation @ u)
    np.testing.assert_array_equal(recovery_objective(state, nominal, [])[0], nominal)


def test_recovery_preserves_hard_rows_and_expiry_brakes():
    c = RecoveryController()
    f = Forecast("object", 0.0, np.array([0.0, 2.0]), np.array([[0.75, 0.0], [0.75, 0.0]]), 0.2)
    assert c.update([f], 0.0, np.zeros(4))["accepted"]
    u, diag = c.command(np.zeros(4), np.array([1.0, 0.0]), 0.1)
    assert diag["feasible"] and diag["recovery_enforced"]
    assert any((r["kind"] == "predictive" for r in diag["enforced_rows"]))
    for r in diag["enforced_rows"]:
        assert np.dot(r["coefficients"], u) >= r["lower_bound"] - 1e-05
    before = c.snapshot()["forecasts"]
    _, diag = c.command(np.zeros(4), np.array([1.0, 0.0]), 2.0)
    assert not diag["feasible"] and (not diag["recovery_enforced"])
    assert not diag["enforced_rows"] and c.snapshot()["forecasts"] == before


def test_invalid_objective_reports_a_brake_without_breaking_logging():
    c = RecoveryController(inactive=True)
    _, diag = c.command(np.zeros(4), ["invalid"], 0.0)
    assert not diag["feasible"] and (not diag["recovery_enforced"])
    assert diag["qp_objective_acceleration"] is None


def test_native_measurement_does_not_modify_live_physics(tmp_path):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="ball" pos="1 0 .1"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world = SimpleNamespace(
        model=model,
        data=data,
        dynamic_obstacle_body_ids=[model.body("ball").id],
        scenario=SimpleNamespace(dynamic_obstacles=[SimpleNamespace(name="ball", shape="sphere")]),
    )
    h = GroundHandoff(world, tmp_path, [])
    data.qpos[0] = 2.0
    names = ["qpos", "qvel", "qacc", "qacc_warmstart", "geom_xpos", "geom_xmat"]
    before = {key: getattr(data, key).copy() for key in names}
    measured = h.measured()
    assert measured[0]["center"][0] == pytest.approx(2.0)
    for key in names:
        np.testing.assert_array_equal(getattr(data, key), before[key])


def test_projection_and_ground_rows_use_moving_relative_velocity(tmp_path):
    items = [item("ball", sphere=True, velocity=np.array([0.1, 0.0]), linear_speed=0.1)]
    h = handoff(tmp_path, items)
    ground = fill(h, CodeWorldController())
    state = np.zeros(4)
    velocity, diag = h.project(state, np.array([1.0, 0.0]), np.zeros(2), ground)
    assert np.max(abs(velocity)) <= 0.5 + 1e-10
    for row in diag["reference_velocity_rows"]:
        assert np.dot(row["coefficients"], velocity) >= row["lower_bound"] - 1e-10
    _, receipt = ground.command(state, np.array([1.0, 0.0]), 0.62)
    assert receipt["enforced_rows"][0]["kind"] == "moving_ground_hocbf"


def test_navigation_cannot_change_calibrated_protocol_or_shorten_horizon(tmp_path):
    with pytest.raises(ValueError, match="separate uncalibrated"):
        run({}, "pss", tmp_path, navigation_profile="demo-ground-truth")
    with pytest.raises(ValueError, match="entire original"):
        run_navigation({"family": "stack", "duration": 14}, tmp_path, None, duration=10)
    with pytest.raises(ValueError, match="cannot alter"):
        run({}, "pss_uncalibrated", tmp_path, demo_margin_m=0.2)


def test_fixed_demo_padding_changes_only_new_geometry():
    f = Forecast(
        "object",
        0.0,
        np.array([0.0, 2.0]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([0.3, 0.4]),
    )
    padded = pad_forecasts([f], 0.2)[0]
    np.testing.assert_allclose(padded.radii, [0.5, 0.6])
    np.testing.assert_array_equal(f.radii, [0.3, 0.4])
    np.testing.assert_array_equal(padded.times, f.times)
    np.testing.assert_array_equal(padded.centers, f.centers)
    assert padded.origin_time == f.origin_time and padded.object_id == f.object_id
    with pytest.raises(ValueError):
        pad_forecasts([f], float("nan"))


def test_stopping_contact_free_is_not_navigation_success(tmp_path, monkeypatch):
    import benchmarks.reactive

    def stopped(case, output, *args, **kwargs):
        output.mkdir()
        return dict(environment_contact=False, goal_reached=False)

    monkeypatch.setattr(benchmarks.reactive, "run", stopped)
    result = run(
        dict(id="test", family="stack", intended_hazard=True), "plain_cbf", tmp_path / "run"
    )
    assert result["contact_free"] and (not result["navigation_success"])
