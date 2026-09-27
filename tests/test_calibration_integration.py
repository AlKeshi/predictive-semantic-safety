import json

import numpy as np
import pytest

from benchmarks.pipeline import CalibratedSession, enforce_required_origins
from pss.calibration import WholePlaneForecast, fit_calibration
from pss.calibration.export import export_corpus
from pss.control import CodeWorldController, Forecast


def record():
    indices = [
        {"object_id": "block", "origin_time": origin, "future_time": origin + offset}
        for origin in (0.8, 1.8)
        for offset in np.arange(1, 6) * 0.5
    ]
    common = {
        "pipeline": {"acquisition_mode": "calibrated_closed_loop", "synthetic_test": True},
        "indices": indices,
        "shape_radii": {"block": 0.2},
    }

    def corpus(prefix, error):
        return dict(
            common,
            sessions=[
                {
                    "session_id": f"{prefix}-{i}",
                    "acquisition_record_digest": "1" * 64,
                    "prediction": [[10.0, 0.0]] * 10,
                    "truth": [[10.0 + error, 0.0]] * 10,
                }
                for i in range(3)
            ],
        )

    return fit_calibration(corpus("training", 0.1), corpus("calibration", 0.2), delta=0.5)


def test_complete_finite_session_export_fit_and_guards(tmp_path):
    fitted = record()
    session = CalibratedSession(fitted, fitted.pipeline_digest, duration=4.3, family="stack")
    controller = CodeWorldController()
    state = np.zeros(4)
    next_origin = session.origins[0]
    for now in np.arange(44) / 10:
        if abs(now - next_origin) < 1e-08:
            forecast = Forecast(
                "block",
                now,
                now + np.arange(6) * 0.5,
                np.tile([10.0, 0.0], (6, 1)),
                np.full(6, 0.2),
            )
            adjusted = session([forecast])
            assert adjusted[0].radii[0] == 0.2
            assert np.all(adjusted[0].radii[1:] > 0.2)
            controller.update(adjusted, now, state)
            next_origin = session.next_origin_after(now)
        if session.needs_truth(now):
            session.observe_truth(now, {"block": [10.2, 0.0, 0.3]})
        if session.seen:
            enforce_required_origins(controller, session, now, state)
            controller.command(state, np.zeros(2), now)
    assert np.isinf(next_origin)
    session.check_time(4.31)
    assert session.receipt()["all_required_predictions_present"]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(session.trace("independent-synthetic-session")))
    exported = export_corpus([trace])
    assert len(exported["sessions"][0]["truth"]) == 10
    assert all((point is not None for point in exported["sessions"][0]["truth"]))
    with pytest.raises(ValueError, match="beyond"):
        CalibratedSession(fitted, fitted.pipeline_digest, duration=4.2, family="stack")
    missing = CalibratedSession(fitted, fitted.pipeline_digest)
    with pytest.raises(WholePlaneForecast, match="Missing"):
        missing.check_time(0.81)
    incomplete = session.trace("incomplete")
    incomplete["truth"][0] = None
    trace.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError):
        export_corpus([trace])
    raw = session.trace("wrong-policy")
    raw["pipeline"]["acquisition_mode"] = "raw"
    trace.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Raw-policy"):
        export_corpus([trace])
