from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .data import pipeline_digest
from .fit import fit_calibration
from .record import CalibrationRecord

DEFAULT_CALIBRATION_DIR = "calibration_records"


def save_calibration(record: CalibrationRecord, path: str | Path, *, replace: bool = False) -> Path:
    if not isinstance(record, CalibrationRecord):
        raise TypeError("A fitted CalibrationRecord is required, not a scalar quantile")
    record = CalibrationRecord(record.to_dict())
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".calibration-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record.to_dict(), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                existing = CalibrationRecord.load(path)
                if existing.record_digest != record.record_digest:
                    raise FileExistsError(
                        f"A different calibration already exists at {path}; "
                        "choose another directory or explicitly use replace=True / --replace."
                    ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


class CalibrationStore:
    def __init__(self, directory: str | Path = DEFAULT_CALIBRATION_DIR):
        self.directory = Path(directory).expanduser()

    def path_for(self, pipeline: dict) -> Path:
        return self.directory / f"{pipeline_digest(pipeline)}.json"

    def save(self, record: CalibrationRecord, *, replace: bool = False) -> Path:
        if not isinstance(record, CalibrationRecord):
            raise TypeError("A fitted CalibrationRecord is required, not a scalar quantile")
        return save_calibration(
            record, self.path_for(record.to_dict()["pipeline"]), replace=replace
        )

    def load(self, pipeline: dict) -> CalibrationRecord:
        return load_calibration(pipeline, path=self.path_for(pipeline))


def load_calibration(
    pipeline: dict,
    *,
    directory: str | Path = DEFAULT_CALIBRATION_DIR,
    path: str | Path | None = None,
) -> CalibrationRecord:
    expected = pipeline_digest(pipeline)
    selected = (
        Path(path).expanduser()
        if path is not None
        else CalibrationStore(directory).path_for(pipeline)
    )
    try:
        record = CalibrationRecord.load(selected)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No fitted calibration record at {selected}. Run pss-calibrate on "
            "disjoint training/calibration sessions from this frozen pipeline first. "
            "PSS has no fixed-margin fallback."
        ) from exc
    if record.pipeline_digest != expected:
        raise ValueError("Calibration pipeline does not match current source and checkpoint")
    return record


def _corpus(value: dict | str | Path) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, (str, Path)):
        raise TypeError("A session corpus dictionary or JSON path is required")
    data = json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("A session corpus must be a JSON object")
    return data


def fit_and_store(
    training: dict | str | Path,
    calibration: dict | str | Path,
    *,
    delta: float,
    directory: str | Path = DEFAULT_CALIBRATION_DIR,
    minimum_scale: float = 0.001,
    replace: bool = False,
) -> CalibrationRecord:
    record = fit_calibration(
        _corpus(training), _corpus(calibration), delta=delta, minimum_scale=minimum_scale
    )
    CalibrationStore(directory).save(record, replace=replace)
    return record
