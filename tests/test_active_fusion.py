import numpy as np
import pytest

from demos.active.fusion import OracleGround, fused_family
from pss.control.forecast import Forecast

SPECS = [{"id": "a", "radius": 0.2}, {"id": "b", "radius": 0.2}]


def test_ground_handoff_keeps_other_airborne_support_and_all_ids():
    g = OracleGround(SPECS)
    g.observe({"a"}, {"a": ([1.0, 0.0], [-0.4, 0.0])}, 1.0)
    old = Forecast("b", 0.0, np.array([0.0, 2.0]), np.array([[2.0, 1.0], [2.0, -1.0]]), 0.2)
    family, sources = fused_family({"b": old}, g, ["a", "b"], 1.0)
    assert [f.object_id for f in family] == ["a", "b"]
    assert family[1].times[-1] == 2.0
    assert sources["b"]["original_origin"] == 0.0
    assert np.allclose(family[0].centers[-1], [0.0, 0.0])
    assert np.allclose(family[1].centers[0], [2.0, 0.0])
    assert old.times.tolist() == [0.0, 2.0]


def test_expired_airborne_cannot_be_extended_by_fresh_ground_measurement():
    g = OracleGround(SPECS)
    g.observe({"a"}, {"a": ([1.0, 0.0], [0.0, 0.0])}, 2.0)
    old = Forecast("b", 0.0, np.array([0.0, 2.0]), np.array([[2.0, 0.0], [2.0, 0.0]]), 0.2)
    with pytest.raises(ValueError, match="expired"):
        fused_family({"b": old}, g, ["a", "b"], 2.0)


def test_missing_or_nonfinite_ground_measurement_is_atomic():
    g = OracleGround(SPECS)
    g.observe({"a"}, {"a": ([1.0, 0.0], [0.0, 0.0])}, 1.0)
    with pytest.raises(ValueError):
        g.observe({"b"}, {"a": ([1.0, 0.0], [0.0, 0.0]), "b": ([np.nan, 0.0], [0.0, 0.0])}, 1.1)
    assert g.grounded == {"a"} and g.time == 1.0
    with pytest.raises(ValueError):
        g.observe(set(), {}, 1.2)
    assert g.grounded == {"a"} and g.time == 1.0


def test_all_ground_needs_no_vlm_and_camera_dropout_never_removes_hazards():
    g = OracleGround(SPECS)
    g.observe({"a", "b"}, {"a": ([1.0, 0.0], [-0.4, 0.0]), "b": ([2.0, 0.0], [-0.3, 0.0])}, 1.0)
    family, sources = fused_family({}, g, ["a", "b"], 1.0)
    assert len(family) == 2
    assert all((s["source"] == "oracle_current_ground_position_velocity" for s in sources.values()))
    with pytest.raises(ValueError, match="fresh"):
        fused_family({}, g, ["a", "b"], 1.1)
