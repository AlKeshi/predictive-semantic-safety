import numpy as np

from demos.active.snapshot import waypoint
from pss.control.forecast import Forecast


def test_egress_continues_into_a_route_instead_of_stopping_at_nearest_free_cell():
    f = Forecast("rolling", 0.0, np.array([0.0, 2.5]), np.array([[0.0, 0.0], [-0.5, 0.0]]), 0.2)
    target, receipt = waypoint([-0.65, 0.0], [2.0, 0.0], [f], [])
    assert receipt["status"] == "planned_egress"
    path = np.asarray(receipt["path"])
    margin = np.linalg.norm(path, axis=1) - (0.2 + 0.403 + 0.12 + 0.15 / np.sqrt(2))
    exit_index = np.flatnonzero(margin > 0)[0]
    assert np.all(np.diff(margin[: exit_index + 1]) > 0)
    assert np.all(margin[exit_index:] > 0)
    assert np.linalg.norm(target - [-0.65, 0.0]) >= 0.65
    assert abs(target[1]) > 0.2
