import csv
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.backup_cbf import MatchedBackupCBF, admit_current_geometry
from benchmarks.geometry import geom_bounds, update_geometry_measurement
from benchmarks.plain_cbf import PlainCBF
from benchmarks.summarize import summarize


def obstacle(center=(2.0, 0.8), dynamic=True):
    return {
        "object_id": "block",
        "center": list(center),
        "radius": 0.13,
        "dynamic": dynamic,
        "polygon": None,
    }


def test_stationary_backup_uses_flow_and_terminal_rows():
    controller = MatchedBackupCBF(2.5)
    u, receipt = controller.command([0.0, 0.0, 0.5, 0.0], [0.2, 0.0], 1.0, [obstacle()])
    assert receipt["geometry_update"]["accepted"] and receipt["feasible"]
    assert {"predictive", "terminal"} <= {row["kind"] for row in receipt["enforced_rows"]}
    for forecast in controller.controller.snapshot()["forecasts"]:
        np.testing.assert_array_equal(
            forecast["centers"], np.tile([2.0, 0.8], (len(forecast["times"]), 1))
        )
    for row in receipt["enforced_rows"]:
        assert np.dot(row["coefficients"], u) - row["lower_bound"] >= -1e-05


def test_unobserved_hazard_does_not_trigger_backup():
    u, receipt = MatchedBackupCBF(2.5).command([0.0, 0.0, 0.2, 0.0], [0.3, 0.0], 1.0, [])
    np.testing.assert_array_equal(u, [0.3, 0.0])
    assert receipt["status"] == "inactive_nominal"
    assert not admit_current_geometry(2.0)
    assert admit_current_geometry(0.55)
    with pytest.raises(ValueError):
        admit_current_geometry(float("nan"))


def test_rejected_replacement_brakes_and_preserves_family():
    controller = MatchedBackupCBF(2.5)
    state = [0.0, 0.0, 0.2, 0.0]
    controller.command(state, [0.2, 0.0], 1.0, [obstacle()])
    before = deepcopy(controller.controller.snapshot()["forecasts"])
    _, receipt = controller.command(state, [0.2, 0.0], 1.02, [obstacle((0.01, 0.0))])
    assert not receipt["geometry_update"]["accepted"] and (not receipt["feasible"])
    assert receipt["enforced_rows"] == []
    assert controller.controller.snapshot()["forecasts"] == before


def test_fresh_geometry_is_separate_from_physics():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 .57"><freejoint/><geom type="box" size=".12 .05 .57"/></body></worldbody></mujoco>'
    )
    data, measured = (mujoco.MjData(model), mujoco.MjData(model))
    mujoco.mj_forward(model, data)
    data.qpos[0] = 0.37
    before = {
        name: getattr(data, name).copy()
        for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "geom_xpos", "geom_xmat")
    }
    update_geometry_measurement(model, data, measured)
    center, radius, bottom, _ = geom_bounds(model, measured, 0)
    assert center[0] == pytest.approx(0.37)
    assert radius == pytest.approx(0.13) and bottom == pytest.approx(0.0)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)


def test_plain_polygon_does_not_enclose_entire_lane():
    item = {
        "object_id": "wall",
        "center": [0.0, -2.0],
        "radius": 20.0,
        "dynamic": False,
        "polygon": [[-20.0, -2.1], [20.0, -2.1], [20.0, -1.9], [-20.0, -1.9]],
    }
    _, receipt = PlainCBF().command([0.0, 0.0, 0.2, 0.0], [0.2, 0.0], 1.0, [item])
    assert receipt["feasible"] and receipt["min_margin"] > 1.0


def test_summary_rejects_contact_inconsistency(tmp_path):
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, ["method", "trial", "contact_free", "hazard_contact", "environment_contact"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "plain_cbf",
                "trial": "case",
                "contact_free": True,
                "hazard_contact": True,
                "environment_contact": False,
            }
        )
    with pytest.raises(ValueError, match="Inconsistent"):
        summarize(path)


class RecordFixture:
    indices = [{"origin_time": 0.8}, {"origin_time": 1.8}]

    def inflate_forecasts(self, forecasts, pipeline_digest):
        from pss.calibration import WholePlaneForecast

        if not forecasts:
            raise WholePlaneForecast("Missing fixture")
        return forecasts

    def receipt(self):
        return {"fitted_record": True}


def test_missed_calibration_origin_forces_brake_before_any_forecast():
    from benchmarks.pipeline import CalibratedSession, enforce_required_origins
    from pss.control.code_world import CodeWorldController

    session = CalibratedSession(RecordFixture(), "fixture")
    controller = CodeWorldController(inactive=True)
    enforce_required_origins(controller, session, 0.82, [0.0, 0.0, 0.2, 0.0])
    _, diagnostic = controller.command([0.0, 0.0, 0.2, 0.0], [0.3, 0.0], 0.82)
    assert not diagnostic["feasible"]
    assert session.receipt()["invalid_reason"] is not None
    assert not session.receipt()["all_required_predictions_present"]


def test_invalid_calibration_prediction_cannot_resume_on_later_forecast():
    from benchmarks.pipeline import CalibratedSession
    from pss.calibration import WholePlaneForecast
    from pss.control.code_world import Forecast

    session = CalibratedSession(RecordFixture(), "fixture")
    with pytest.raises(WholePlaneForecast):
        session([])
    forecast = Forecast(
        "fixture", 1.8, np.array([1.8, 2.8]), np.array([[2.0, 0.0], [2.0, 0.0]]), 0.2
    )
    with pytest.raises(WholePlaneForecast):
        session([forecast])


def test_checkpoint_configuration_is_content_validated(tmp_path, monkeypatch):
    import hashlib

    from pss.vla import omnivla_support

    value = b'{"config": true}'
    path = tmp_path / "config.json"
    path.write_bytes(value)
    monkeypatch.setattr(
        omnivla_support, "_CHECKPOINT_SHA256", {"config.json": hashlib.sha256(value).hexdigest()}
    )
    assert "config.json" in omnivla_support._validate_checkpoint(tmp_path)
    path.write_bytes(b'{"config": false}')
    with pytest.raises(ValueError, match="hash mismatch"):
        omnivla_support._validate_checkpoint(tmp_path)


def test_omnivla_source_rejects_modified_tracked_files(tmp_path):
    import subprocess

    from pss.vla.omnivla_support import _require_git_revision

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    source = tmp_path / "model.py"
    source.write_text("value = 1\n")
    git("add", "model.py")
    git("commit", "-m", "fixture")
    revision = git("rev-parse", "HEAD")
    assert _require_git_revision(tmp_path, revision, label="test") == revision
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="modified tracked"):
        _require_git_revision(tmp_path, revision, label="test")
