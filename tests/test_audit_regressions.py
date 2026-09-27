import json
from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_calibration_integration import record

from benchmarks.ceiling_observation import CeilingObservation
from benchmarks.pipeline import CalibratedSession, enforce_required_origins
from pss.calibration.export import export_corpus
from pss.control import CodeWorldController, Forecast


@pytest.mark.parametrize("demo", [False, True])
def test_debris_handoff_is_selected_by_demo_profile(tmp_path, demo):
    controller = CodeWorldController(inactive=True)
    episode = SimpleNamespace(
        settled=True,
        active=True,
        demo_navigation=demo,
        ground_handoff=False,
        detection=SimpleNamespace(radius=0.2),
        demo_margin=0.2,
        grounded_history=deque((i * 0.05, [2.0, 0.0, 0.05]) for i in range(10)),
        diagnostic={"normal": [0, 0, 1]},
        state=np.zeros(4),
        now=1.0,
        controller=controller,
        out=tmp_path,
    )
    CeilingObservation.ground_update(episode)
    assert episode.ground_handoff == demo
    assert (episode.controller is controller) != demo


def test_required_origin_failure_revokes_even_valid_old_geometry():
    fitted = record()
    session = CalibratedSession(fitted, fitted.pipeline_digest)
    controller = CodeWorldController()
    forecast = Forecast("block", 0, np.array([0.0, 5.0]), np.array([[10.0, 0.0]] * 2), 0.2)
    assert controller.update([forecast], 0, np.zeros(4))["accepted"]
    enforce_required_origins(controller, session, 0.81, np.zeros(4))
    _, decision = controller.command(np.zeros(4), [0.3, 0], 0.81)
    assert not decision["feasible"] and not decision["enforced_rows"]


def test_export_rejects_mixed_acquisition_and_margin_ablation(tmp_path):
    fitted = record()
    session = CalibratedSession(fitted, fitted.pipeline_digest)
    trace = session.trace("first")
    trace["truth"] = [[0, 0]] * len(fitted.indices)
    assert trace["acquisition_record_digest"] == fitted.record_digest
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    paths[0].write_text(json.dumps(trace))
    other = deepcopy(trace)
    other.update(session_id="second", acquisition_record_digest="different")
    paths[1].write_text(json.dumps(other))
    with pytest.raises(ValueError, match="different acquisition"):
        export_corpus(paths)
    other.update(acquisition_record_digest=fitted.record_digest, residual_margin_applied=False)
    paths[1].write_text(json.dumps(other))
    with pytest.raises(ValueError, match="Margin ablations"):
        export_corpus(paths)
