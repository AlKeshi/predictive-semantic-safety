import numpy as np
import osqp
import pytest
from scipy import sparse

from pss.control.velocity_reference import project_velocity_reference

PARAMS = dict(robot_radius=0.4, clearance=0.05, alpha=3.0, max_speed=0.5)


@pytest.fixture
def project():
    return project_velocity_reference


def test_unconstrained_reference_and_speed_box(project):
    v, _ = project([0, 0], [0.8, -0.7], [], **PARAMS)
    np.testing.assert_allclose(v, [0.5, -0.5], atol=1e-12)
    v, _ = project([0, 0], [0.12, -0.07], [], **PARAMS)
    np.testing.assert_array_equal(v, [0.12, -0.07])


def test_reference_reserve_implies_velocity_condition_for_bounded_error(project):
    obstacle = dict(object_id="observed", center=[0, 0], radius=0.8)
    p = np.array([-1.5, 0.2])
    reserve = 0.3
    v, diag = project(p, [0.4, -0.1], [obstacle], tracking_error_reserve=reserve, **PARAMS)
    normal = 2 * p
    worst_actual = v - reserve * normal / np.linalg.norm(normal)
    h = np.dot(p, p) - (0.8 + 0.4 + 0.05) ** 2
    assert normal @ worst_actual + 3 * h >= -1e-10
    assert diag["tracking_error_reserve"] == reserve


def test_no_prescribed_passing_side(project):
    p = np.array([-1.4, 0.3])
    ref = np.array([0.4, -0.05])
    obstacle = dict(object_id="observed", center=[0, 0], radius=0.8)
    v, _ = project(p, ref, [obstacle], tracking_error_reserve=0.2, **PARAMS)
    mirrored, _ = project(
        p * [1, -1], ref * [1, -1], [obstacle], tracking_error_reserve=0.2, **PARAMS
    )
    np.testing.assert_allclose(mirrored, v * [1, -1], atol=1e-12)
    R = np.array([[0, -1], [1, 0]])
    rotated, _ = project(R @ p, R @ ref, [obstacle], tracking_error_reserve=0.2, **PARAMS)
    np.testing.assert_allclose(rotated, R @ v, atol=1e-12)


@pytest.mark.parametrize(
    "p,ref,reserve", [([np.nan, 0], [0, 0], 0), ([0, 0], [np.inf, 0], 0), ([0, 0], [0, 0], -0.1)]
)
def test_invalid_inputs_fail_closed(project, p, ref, reserve):
    with pytest.raises(ValueError):
        project(p, ref, [], tracking_error_reserve=reserve, **PARAMS)


def test_infeasible_reference_raises(project):
    with pytest.raises(ValueError, match="infeasible"):
        project([0, 0], [0, 0], [dict(object_id="overlap", center=[0, 0], radius=1)], **PARAMS)


def test_projection_matches_independent_osqp(project):
    rng = np.random.default_rng(91916)
    for _ in range(80):
        p = rng.uniform(-2, 2, 2)
        reference = rng.uniform(-0.8, 0.8, 2)
        reserve = rng.uniform(0, 0.3)
        obstacles = [
            dict(object_id=str(i), center=rng.uniform(-1, 1, 2), radius=rng.uniform(0.1, 0.7))
            for i in range(2)
        ]
        normals = []
        lower = []
        for obstacle in obstacles:
            delta = p - obstacle["center"]
            h = delta @ delta - (obstacle["radius"] + 0.45) ** 2
            normals.append(2 * delta)
            lower.append(-3 * h + np.linalg.norm(2 * delta) * reserve)
        A = np.vstack([normals, np.eye(2)])
        lo = np.r_[lower, [-0.5, -0.5]]
        hi = np.r_[[np.inf, np.inf], [0.5, 0.5]]
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.eye(2, format="csc"),
            q=-reference,
            A=sparse.csc_matrix(A),
            l=lo,
            u=hi,
            verbose=False,
            eps_abs=1e-09,
            eps_rel=1e-09,
            max_iter=10000,
            polishing=False,
        )
        result = solver.solve(raise_error=False)
        if result.info.status_val == 1:
            projected, _ = project(
                p, reference, obstacles, tracking_error_reserve=reserve, **PARAMS
            )
            np.testing.assert_allclose(projected, result.x, atol=2e-06, rtol=0)
        else:
            assert result.info.status_val == 3
            with pytest.raises(ValueError, match="infeasible"):
                project(p, reference, obstacles, tracking_error_reserve=reserve, **PARAMS)
