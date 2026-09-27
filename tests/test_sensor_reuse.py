from types import SimpleNamespace

import numpy as np

from benchmarks.sensor import detections_for_observation
from pss.perception.code_world import Detection


def test_reused_detections_have_no_shared_mutable_state():
    observation = object()
    specifications = [{"id": "a", "radius": 0.2}]
    original = Detection("a", np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), 5.0, 0.2, 20)
    world = SimpleNamespace(
        _current_sensor_detections=(observation, specifications, {"a": original})
    )
    first = detections_for_observation(world, observation, specifications)
    np.testing.assert_array_equal(first["a"].center, original.center)
    first["a"].center[:] = -100
    second = detections_for_observation(world, observation, specifications)
    np.testing.assert_array_equal(second["a"].center, [3, 4, 5])


def test_new_frame_or_changed_inventory_recomputes(monkeypatch):
    observation, fresh = (object(), object())
    specifications = [{"id": "a", "radius": 0.2}]
    world = SimpleNamespace(_current_sensor_detections=(observation, specifications, {"old": None}))
    calls = []

    def detect(ob, specs):
        calls.append((ob, specs))
        return {"fresh": None}

    monkeypatch.setattr("benchmarks.sensor.detect_objects", detect)
    assert detections_for_observation(world, fresh, specifications) == {"fresh": None}
    assert detections_for_observation(world, observation, [{"id": "a", "radius": 0.3}]) == {
        "fresh": None
    }
    assert len(calls) == 2
