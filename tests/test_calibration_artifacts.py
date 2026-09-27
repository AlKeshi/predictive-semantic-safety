import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pss.calibration import (
    CalibrationRecord,
    CalibrationStore,
    WholePlaneForecast,
    fit_and_store,
    fit_calibration,
    load_calibration,
    save_calibration,
)
from pss.control import Forecast


def corpus(prefix, errors):
    return {
        "pipeline": {"synthetic_test": True, "model": "frozen", "schedule": "fixed"},
        "indices": [
            {"object_id": "block", "origin_time": 0.0, "future_time": t} for t in (1.0, 2.0)
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


def fitted(errors=(0.05, 0.1, 0.15), delta=0.5):
    return fit_calibration(corpus("train", [0.1, 0.2]), corpus("cal", errors), delta=delta)


def test_fit_store_load_and_apply_indexed_radii(tmp_path):
    training, calibration = (corpus("train", [0.1, 0.2]), corpus("cal", [0.05, 0.1, 0.15]))
    record = fit_and_store(training, calibration, delta=0.5, directory=tmp_path)
    loaded = load_calibration(training["pipeline"], directory=tmp_path)
    assert loaded.to_dict() == record.to_dict()
    assert loaded.q == pytest.approx(0.5)
    raw = Forecast(
        "block", 0.0, np.array([0.0, 1.0, 2.0]), np.zeros((3, 2)), np.array([0.2, 0.3, 0.4])
    )
    inflated = loaded.inflate_forecasts([raw], pipeline_digest=loaded.pipeline_digest)[0]
    np.testing.assert_allclose(inflated.radii, [0.2, 0.4, 0.6])
    np.testing.assert_allclose(raw.radii, [0.2, 0.3, 0.4])
    assert loaded.receipt()["quantile_units"] == "dimensionless"
    assert loaded.receipt()["calibration_record_digest"] == record.record_digest


def test_default_store_and_corpus_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, data in (
        ("train", corpus("train", [0.1, 0.2])),
        ("cal", corpus("cal", [0.05, 0.1, 0.15])),
    ):
        Path(f"{name}.json").write_text(json.dumps(data))
    record = fit_and_store("train.json", Path("cal.json"), delta=0.5)
    assert (
        CalibrationStore().load(record.to_dict()["pipeline"]).record_digest == record.record_digest
    )
    assert (tmp_path / "calibration_records" / f"{record.pipeline_digest}.json").is_file()


def test_missing_record_has_no_fixed_margin_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="no fixed-margin fallback"):
        load_calibration({"model": "new"}, directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_explicit_record_does_not_fall_back_to_store(tmp_path):
    record = fitted()
    store = CalibrationStore(tmp_path)
    store.save(record)
    with pytest.raises(FileNotFoundError):
        load_calibration(
            record.to_dict()["pipeline"], directory=tmp_path, path=tmp_path / "missing.json"
        )
    with pytest.raises(ValueError, match="does not match"):
        load_calibration({"model": "different"}, path=store.path_for(record.to_dict()["pipeline"]))


@pytest.mark.parametrize(
    "field,value", [("q", 0.123), ("scales", [0.3, 0.4]), ("pipeline_digest", "bad")]
)
def test_tampered_artifact_is_rejected(tmp_path, field, value):
    record = fitted()
    data = record.to_dict()
    data[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_calibration(record.to_dict()["pipeline"], path=path)


def test_corrupt_json_is_not_replaced_by_a_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{invalid")
    with pytest.raises(ValueError):
        load_calibration(fitted().to_dict()["pipeline"], path=path)


def test_pipeline_change_cannot_reuse_record(tmp_path):
    record = fitted()
    store = CalibrationStore(tmp_path)
    store.save(record)
    pipeline = {**record.to_dict()["pipeline"], "model": "changed"}
    with pytest.raises(FileNotFoundError):
        store.load(pipeline)
    store.path_for(pipeline).write_text(json.dumps(record.to_dict()))
    with pytest.raises(ValueError, match="does not match"):
        store.load(pipeline)


def test_identical_publication_is_idempotent_and_refit_is_explicit(tmp_path):
    store = CalibrationStore(tmp_path)
    first, second = (fitted(), fitted((0.2, 0.3, 0.4)))
    path = store.save(first)
    assert store.save(first) == path
    with pytest.raises(FileExistsError, match="different calibration"):
        store.save(second)
    assert store.load(first.to_dict()["pipeline"]).record_digest == first.record_digest
    store.save(second, replace=True)
    assert store.load(second.to_dict()["pipeline"]).q == second.q
    assert first.record_digest != second.record_digest
    assert not list(tmp_path.glob(".calibration-*"))


def test_inventory_collision_cannot_silently_choose_latest(tmp_path):
    first = fitted()
    training, calibration = (corpus("train", [0.1, 0.2]), corpus("cal", [0.05, 0.1, 0.15]))
    for data in (training, calibration):
        data["indices"][1]["future_time"] = 3.0
    second = fit_calibration(training, calibration, delta=0.5)
    store = CalibrationStore(tmp_path)
    store.save(first)
    with pytest.raises(FileExistsError):
        store.save(second)


def test_concurrent_identical_publication_and_temporary_cleanup(tmp_path):
    store, record = (CalibrationStore(tmp_path), fitted())
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: store.save(record), range(8)))
    assert len(set(paths)) == 1
    assert store.load(record.to_dict()["pipeline"]).record_digest == record.record_digest
    assert not list(tmp_path.glob(".calibration-*"))


def test_failed_write_preserves_previous_record(tmp_path, monkeypatch):
    from pss.calibration import artifacts

    store, first = (CalibrationStore(tmp_path), fitted())
    path = store.save(first)
    before = path.read_bytes()

    def fail(*args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError):
        store.save(fitted((0.2, 0.3, 0.4)), replace=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".calibration-*"))


def test_infinite_quantile_survives_storage_and_fails_closed(tmp_path):
    record = fitted(delta=0.01)
    CalibrationStore(tmp_path).save(record)
    loaded = load_calibration(record.to_dict()["pipeline"], directory=tmp_path)
    assert np.isinf(loaded.q)
    assert loaded.receipt()["q"] is None
    with pytest.raises(WholePlaneForecast, match="Infinite"):
        loaded.inflate_forecasts([], pipeline_digest=loaded.pipeline_digest)


def test_overlap_and_scalar_shortcuts_cannot_publish(tmp_path):
    data = corpus("same", [0.1, 0.2])
    with pytest.raises(ValueError, match="disjoint"):
        fit_and_store(data, deepcopy(data), delta=0.5, directory=tmp_path)
    with pytest.raises(TypeError, match="scalar"):
        CalibrationStore(tmp_path).save(0.2)
    with pytest.raises(TypeError, match="scalar"):
        save_calibration(0.2, tmp_path / "scalar.json")
    assert not list(tmp_path.iterdir())


def test_cli_fit_publishes_to_default_lookup_and_preserves_explicit_output(tmp_path):
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": root}
    for name, value in (
        ("train", corpus("train", [0.1, 0.2])),
        ("cal", corpus("cal", [0.05, 0.1, 0.15])),
    ):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    command = [sys.executable, "-m", "pss.calibration", "train.json", "cal.json", "--delta", ".5"]
    result = subprocess.run(
        command, cwd=tmp_path, env=env, check=True, capture_output=True, text=True
    )
    receipt = json.loads(result.stdout)
    loaded = CalibrationRecord.load(tmp_path / receipt["calibration_path"])
    assert receipt["calibration_record_digest"] == loaded.record_digest
    assert receipt["q"] == pytest.approx(0.5)
    exported = subprocess.run(
        command + ["--output", "export.json"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(exported.stdout)["calibration_path"] == "export.json"
    assert CalibrationRecord.load(tmp_path / "export.json").to_dict() == loaded.to_dict()
    bad = subprocess.run(command + ["--q", ".2"], cwd=tmp_path, env=env, capture_output=True)
    assert bad.returncode != 0
