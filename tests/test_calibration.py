import json
from copy import deepcopy

import numpy as np
import pytest

from pss.calibration import (
    CalibrationRecord,
    WholePlaneForecast,
    conformal_quantile,
    fit_calibration,
    session_scores,
)
from pss.control import Forecast


def corpus(prefix, errors):
    return {
        "pipeline": {
            "model": "fixed-model-revision",
            "prompt": "fixed-prompt",
            "acquisition": "fixed-schedule",
            "randomness": "fixed-mechanism",
        },
        "indices": [
            {"object_id": "block", "origin_time": 0.0, "future_time": t} for t in [1.0, 2.0]
        ],
        "shape_radii": {"block": 0.2},
        "sessions": [
            {
                "session_id": f"{prefix}-{i}",
                "acquisition_record_digest": "1" * 64,
                "prediction": [[e, 0], [2 * e, 0]],
                "truth": [[0, 0], [0, 0]],
            }
            for i, e in enumerate(errors)
        ],
    }


def record():
    return fit_calibration(
        corpus("train", [0.1, 0.2]), corpus("calibration", [0.05, 0.1, 0.15]), delta=0.5
    )


@pytest.mark.parametrize("digest", [None, "", " ", "bad", "g" * 64, 12, {}])
def test_fit_and_record_loading_require_acquisition_provenance(digest):
    training, calibration = corpus("train", [0.1, 0.2]), corpus("cal", [0.05, 0.1, 0.15])
    for data in (training, calibration):
        for session in data["sessions"]:
            session["acquisition_record_digest"] = digest
    with pytest.raises(ValueError, match="acquisition record digest"):
        fit_calibration(training, calibration, delta=0.5)
    data = record().to_dict()
    for records in data["acquisition_records"].values():
        for name in records:
            records[name] = digest
    with pytest.raises(ValueError, match="acquisition record digest"):
        CalibrationRecord(data)


@pytest.mark.parametrize("change", ["missing", "empty", "extra", "mixed", "legacy"])
def test_record_loading_rejects_incomplete_or_mixed_acquisition(change):
    data = record().to_dict()
    if change == "missing":
        del data["acquisition_records"]
    elif change == "empty":
        data["acquisition_records"]["training"] = {}
    elif change == "extra":
        data["acquisition_records"]["training"]["unknown"] = "1" * 64
    elif change == "mixed":
        data["acquisition_records"]["training"]["train-0"] = "2" * 64
    else:
        data["schema"] = "pss.session-position-calibration.v1"
    with pytest.raises(ValueError):
        CalibrationRecord(data)


@pytest.mark.parametrize("split", ["training", "calibration", "between_splits"])
def test_fit_rejects_inconsistent_acquisition_controllers(split):
    training, calibration = corpus("train", [0.1, 0.2]), corpus("cal", [0.05, 0.1])
    changed = training if split == "training" else calibration
    sessions = changed["sessions"] if split == "between_splits" else changed["sessions"][:1]
    for session in sessions:
        session["acquisition_record_digest"] = "2" * 64
    with pytest.raises(ValueError, match="acquisition record digest"):
        fit_calibration(training, calibration, delta=0.5)


def test_exact_augmented_order_statistic_and_insufficient_sample_size():
    assert conformal_quantile([1, 2, 3], 0.5) == 2
    assert conformal_quantile([1, 2, 3], 0.25) == 3
    assert np.isinf(conformal_quantile([1, 2, 3], 0.1))
    assert conformal_quantile([1, 1, 1], 0.5) == 1


def test_complete_session_maximum_across_all_positions():
    np.testing.assert_allclose(session_scores([[0.2, 1.2], [0.6, 0.4]], [0.2, 0.4]), [3, 3])
    with pytest.raises(ValueError):
        session_scores([[0.1, 0.2]], [1, 0])


def test_training_only_scales_and_known_rank():
    r = record()
    np.testing.assert_array_equal(r.to_dict()["scales"], [0.2, 0.4])
    assert r.q == pytest.approx(0.5)
    assert r.receipt()["calibration_sessions"] == 3
    changed = corpus("calibration", [100, 200, 300])
    r2 = fit_calibration(corpus("train", [0.1, 0.2]), changed, delta=0.5)
    np.testing.assert_array_equal(r2.to_dict()["scales"], r.to_dict()["scales"])


def test_missing_required_prediction_scores_infinity_without_dropping_session():
    calibration = corpus("calibration", [0.05, 0.1, 0.15])
    calibration["sessions"][0]["prediction"][1] = [None, None]
    r = fit_calibration(corpus("train", [0.1, 0.2]), calibration, delta=0.25)
    assert len(r.to_dict()["scores"]) == 3 and r.to_dict()["scores"][0] is None
    assert np.isinf(r.q)
    with pytest.raises(WholePlaneForecast):
        r.inflate_forecasts([], pipeline_digest=r.pipeline_digest)


def test_physical_session_overlap_and_duplicate_clips_are_rejected():
    with pytest.raises(ValueError):
        fit_calibration(corpus("same", [0.1, 0.2]), corpus("same", [0.1, 0.2]), delta=0.5)
    duplicated = corpus("calibration", [0.1, 0.2])
    duplicated["sessions"][1]["session_id"] = duplicated["sessions"][0]["session_id"]
    with pytest.raises(ValueError):
        fit_calibration(corpus("train", [0.1, 0.2]), duplicated, delta=0.5)


def test_record_roundtrip_and_tampered_quantile_or_scales(tmp_path):
    r = record()
    path = tmp_path / "calibration.json"
    r.save(path)
    assert CalibrationRecord.load(path).to_dict() == r.to_dict()
    for field in ("q", "scales", "pipeline_digest"):
        bad = deepcopy(r.to_dict())
        bad[field] = {"q": 0.2, "scales": [0.1, 0.2], "pipeline_digest": "0" * 64}[field]
        with pytest.raises(ValueError):
            CalibrationRecord(bad)


def test_prediction_radius_includes_shape_spread_and_calibrated_error():
    r = record()
    f = Forecast(
        "block", 0.0, np.array([0.0, 1.0, 2.0]), np.zeros((3, 2)), np.array([0.2, 0.3, 0.4])
    )
    inflated = r.inflate_forecasts([f], pipeline_digest=r.pipeline_digest)[0]
    np.testing.assert_allclose(inflated.radii, [0.2, 0.4, 0.6])
    np.testing.assert_array_equal(f.radii, [0.2, 0.3, 0.4])
    np.testing.assert_array_equal(inflated.centers, f.centers)


@pytest.mark.parametrize("change", ["pipeline", "missing", "grid", "origin", "shape", "nan"])
def test_runtime_rejects_mismatch_instead_of_using_an_uncalibrated_radius(change):
    r = record()
    f = Forecast("block", 0.0, np.array([0.0, 1.0, 2.0]), np.zeros((3, 2)), 0.2)
    family, digest = ([f], r.pipeline_digest)
    if change == "pipeline":
        digest = "0" * 64
    elif change == "missing":
        family = []
    elif change == "grid":
        f.times[1] = 1.1
    elif change == "origin":
        f = Forecast("block", 0.1, f.times + 0.1, f.centers, f.radii)
        family = [f]
    elif change == "shape":
        family = [Forecast("block", 0.0, f.times, f.centers, 0.1)]
    elif change == "nan":
        f.centers[1, 0] = np.nan
    with pytest.raises(WholePlaneForecast):
        r.inflate_forecasts(family, pipeline_digest=digest)


def test_saved_records_use_standard_json_for_infinite_values(tmp_path):
    r = fit_calibration(corpus("train", [0.1]), corpus("calibration", [0.1]), delta=0.01)
    p = tmp_path / "record.json"
    r.save(p)
    assert "Infinity" not in p.read_text() and json.loads(p.read_text())["q"] is None


@pytest.mark.parametrize(
    "prediction",
    [None, [[None, None], [0, 0]], [["bad", 0], [0, 0]], [[0, 0]], [[True, 0], [0, 0]]],
)
def test_invalid_required_forecast_is_an_infinite_score(prediction):
    calibration = corpus("calibration", [0.1])
    calibration["sessions"][0]["prediction"] = prediction
    fitted = fit_calibration(corpus("train", [0.1]), calibration, delta=0.5)
    assert fitted.to_dict()["scores"] == [None]
    assert np.isinf(fitted.q)


@pytest.mark.parametrize(
    "field,value", [("q", True), ("scales", ["0.2", "0.4"]), ("scores", [True, 0.5, 0.75])]
)
def test_record_rejects_non_numeric_scoring_fields(field, value):
    data = record().to_dict()
    data[field] = value
    with pytest.raises((TypeError, ValueError)):
        CalibrationRecord(data)
