import json
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks import pipeline
from benchmarks.config import CeilingConfig, RouteConfig
from benchmarks.route_geometry import retained_forecast_receipt
from benchmarks.run import run
from pss.calibration import CalibrationRecord, CalibrationStore, fit_calibration
from pss.control import Forecast


def fitted(family):
    origin, horizon, count = (0.75, 8.0, 80) if family == "ceiling" else (0.1, 2.5, 5)
    indices = [
        {"object_id": "hazard", "origin_time": origin, "future_time": origin + offset}
        for offset in np.arange(1, count + 1) * horizon / count
    ]
    common = {
        "pipeline": {"synthetic_test": True, "family": family},
        "indices": indices,
        "shape_radii": {"hazard": 0.2},
    }

    def corpus(prefix, errors):
        return {
            **common,
            "sessions": [
                {
                    "session_id": f"{prefix}-{i}",
                    "acquisition_record_digest": "1" * 64,
                    "prediction": [[e, 0]] * count,
                    "truth": [[0, 0]] * count,
                }
                for i, e in enumerate(errors)
            ],
        }

    return fit_calibration(corpus("train", [0.1, 0.2]), corpus("cal", [0.05, 0.1, 0.15]), delta=0.5)


def case_for(family):
    return {
        "id": f"synthetic-{family}",
        "family": family,
        "duration": 9.0 if family == "ceiling" else 3.0,
        "seed": 1,
        "speed": 0.2,
        "warmup": 0.0,
        "intended_hazard": True,
    }


@pytest.mark.parametrize("family", ["ceiling", "stack", "dominos"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("method", ["pss", "pss_uncalibrated"])
def test_runner_loads_applies_and_records_calibration(
    tmp_path, monkeypatch, family, explicit, method
):
    record = fitted(family)
    directory = tmp_path / "artifacts"
    artifact = CalibrationStore(directory).save(record)
    monkeypatch.setattr(pipeline, "describe", lambda *args, **kwargs: record.to_dict()["pipeline"])
    called = []

    def episode(args, scene, *, external, transform, uncertainty):
        called.append(True)
        assert isinstance(args, CeilingConfig if family == "ceiling" else RouteConfig)
        assert args.model == "synthetic-checkpoint"
        assert args.navigation_profile == "benchmark"
        assert args.duration == case_for(family)["duration"]
        output = args.output
        assert (
            CalibrationRecord.load(output / "calibration_record.json").record_digest
            == record.record_digest
        )
        assert uncertainty["q"] == record.q
        origin = transform.origins[0]
        times = np.r_[origin, [item["future_time"] for item in record.indices]]
        raw = Forecast("hazard", origin, times, np.zeros((len(times), 2)), 0.2)
        adjusted = transform([raw])[0]
        np.testing.assert_allclose(
            adjusted.radii, np.r_[0.2, np.full(len(times) - 1, 0.3)] if method == "pss" else 0.2
        )
        assert raw.radii == 0.2
        for when in times[1:]:
            transform.observe_truth(when, {"hazard": [0.0, 0.0]})
        return {"hazard_contact": False, "environment_contact": False, "q_m": 0.0}

    backend = ModuleType(f"benchmarks.{('ceiling' if family == 'ceiling' else 'route')}")
    backend.run_episode = episode
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    datagen = ModuleType("datagen")
    datagen.route_scene = lambda case: SimpleNamespace(meta={})
    monkeypatch.setitem(sys.modules, "datagen", datagen)
    output = tmp_path / "run"
    result = run(
        case_for(family),
        method,
        output,
        model="synthetic-checkpoint",
        calibration=artifact if explicit else None,
        calibration_dir=directory,
    )
    assert called == [True]
    assert result["calibration_record_digest"] == record.record_digest
    assert result["calibration_predictions_complete"]
    assert result["residual_margin_applied"] == (method == "pss")
    report = json.loads(
        (output / ("receipt.json" if family == "ceiling" else "report.json")).read_text()
    )
    assert "q_m" not in report
    assert report["uncertainty"]["calibration_record_digest"] == record.record_digest
    trace = json.loads((output / "calibration_trace.json").read_text())
    assert trace["acquisition_record_digest"] == record.record_digest
    assert all((value is not None for value in trace["prediction"]))
    assert all((value is not None for value in trace["truth"]))


def test_runner_missing_record_stops_before_backend_or_output(tmp_path, monkeypatch):
    record = fitted("ceiling")
    monkeypatch.setattr(pipeline, "describe", lambda *args, **kwargs: record.to_dict()["pipeline"])
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="no fixed-margin fallback"):
        run(
            case_for("ceiling"),
            "pss",
            output,
            model="synthetic",
            calibration_dir=tmp_path / "empty",
        )
    assert not output.exists()


@pytest.mark.parametrize("option", ["calibration", "calibration_dir"])
def test_explicit_calibration_cannot_be_ignored_by_a_different_method(tmp_path, option):
    with pytest.raises(ValueError, match="only valid"):
        run(
            case_for("ceiling"),
            "backup_cbf",
            tmp_path / "run",
            model="synthetic",
            **{option: tmp_path / "record.json"},
        )


def snapshot():
    return {
        "emergency_reason": None,
        "forecasts": [
            {
                "object_id": name,
                "times": [0.0, 1.0, 2.0],
                "centers": [[0.0, 0.0]] * 3,
                "radii": [0.2, 0.5, 0.8],
            }
            for name in ("a", "b")
        ],
    }


def test_observation_gap_uses_committed_radius_not_a_global_q():
    value = snapshot()
    before = deepcopy(value)
    controller = SimpleNamespace(snapshot=lambda: deepcopy(value))
    detection = SimpleNamespace(center=np.array([0.25, 0.0, 0.0]), radius=0.2)
    result = retained_forecast_receipt(
        controller, {"a": detection}, [{"id": "a"}, {"id": "b"}], 1.0
    )
    assert result["retained_committed_family"]
    assert result["carry_until_time"] == 2.0
    assert value == before
    with pytest.raises(ValueError, match="contradicts"):
        retained_forecast_receipt(controller, {"a": detection}, [{"id": "a"}, {"id": "b"}], 0.5)


@pytest.mark.parametrize("radius", [-0.1, np.nan, np.inf, 0.6])
def test_observation_gap_rejects_invalid_or_uncontained_observed_shape(radius):
    controller = SimpleNamespace(snapshot=snapshot)
    detection = SimpleNamespace(center=np.zeros(3), radius=radius)
    with pytest.raises(ValueError, match="contradicts"):
        retained_forecast_receipt(controller, {"a": detection}, [{"id": "a"}, {"id": "b"}], 1.0)


def test_retention_never_extends_expired_support_or_clears_emergency():
    value = snapshot()
    controller = SimpleNamespace(snapshot=lambda: value)
    objects = [{"id": "a"}, {"id": "b"}]
    with pytest.raises(ValueError, match="expired"):
        retained_forecast_receipt(controller, {}, objects, 2.0)
    value["emergency_reason"] = "invalid"
    with pytest.raises(ValueError, match="latched"):
        retained_forecast_receipt(controller, {}, objects, 1.0)
