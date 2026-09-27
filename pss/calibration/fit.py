import numpy as np

from .data import SCHEMA, conformal_quantile, pipeline_digest, session_scores, validate_corpus
from .record import CalibrationRecord


def fit_calibration(training, calibration, *, delta, minimum_scale=0.001):
    training, train_ids, train_errors = validate_corpus(training)
    calibration, calibration_ids, calibration_errors = validate_corpus(calibration)
    if set(train_ids) & set(calibration_ids):
        raise ValueError("Training and calibration physical sessions must be disjoint")
    for field in ("pipeline", "indices", "shape_radii"):
        if training[field] != calibration[field]:
            raise ValueError(f"Training and calibration differ in frozen {field}")
    if isinstance(minimum_scale, bool) or not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("Minimum training scale must be positive and finite")
    finite = np.isfinite(train_errors)
    if not np.all(finite.any(axis=0)):
        raise ValueError("Every training index needs at least one valid position forecast")
    maxima = np.where(finite, train_errors, 0.0).max(axis=0)
    scales = np.maximum(maxima, minimum_scale)
    scores = session_scores(calibration_errors, scales)
    q = conformal_quantile(scores, delta)
    return CalibrationRecord(
        {
            "schema": SCHEMA,
            "pipeline": training["pipeline"],
            "pipeline_digest": pipeline_digest(training["pipeline"]),
            "indices": training["indices"],
            "shape_radii": training["shape_radii"],
            "acquisition_records": {
                split: {
                    session["session_id"]: session.get("acquisition_record_digest")
                    for session in corpus["sessions"]
                }
                for split, corpus in (("training", training), ("calibration", calibration))
            },
            "training_session_ids": train_ids,
            "calibration_session_ids": calibration_ids,
            "training_max_errors": maxima.tolist(),
            "minimum_scale": minimum_scale,
            "scales": scales.tolist(),
            "scores": [float(s) if np.isfinite(s) else None for s in scores],
            "delta": float(delta),
            "q": float(q) if np.isfinite(q) else None,
        }
    )
