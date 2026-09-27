import hashlib
import json

import numpy as np

SCHEMA = "pss.session-position-calibration.v2"


def pipeline_digest(pipeline):
    if not isinstance(pipeline, dict) or not pipeline:
        raise ValueError("A complete frozen pipeline description is required")
    encoded = json.dumps(pipeline, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_indices(indices):
    if not isinstance(indices, list) or not indices:
        raise ValueError("A nonempty fixed forecast inventory is required")
    keys = []
    for item in indices:
        if not isinstance(item, dict) or set(item) != {"object_id", "origin_time", "future_time"}:
            raise ValueError("Invalid object/origin/future-time index")
        name, origin, future = (item["object_id"], item["origin_time"], item["future_time"])
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Object identity must be nonempty")
        if any((isinstance(v, bool) or not isinstance(v, (int, float)) for v in (origin, future))):
            raise ValueError("Forecast times must be numeric")
        if not np.all(np.isfinite([origin, future])) or not 0 <= origin < future:
            raise ValueError("Required forecast time must follow its observation origin")
        keys.append((name, float(origin), float(future)))
    if len(set(keys)) != len(keys):
        raise ValueError("Required forecast indices must be unique")
    return keys


def validate_shapes(shapes, keys):
    names = {key[0] for key in keys}
    if not isinstance(shapes, dict) or set(shapes) != names:
        raise ValueError("Every indexed object needs an enclosing shape radius")
    for radius in shapes.values():
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise TypeError("Shape radius must be numeric")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Shape radius must be finite and nonnegative")


def validate_corpus(value):
    keys = validate_indices(value["indices"])
    validate_shapes(value["shape_radii"], keys)
    pipeline_digest(value["pipeline"])
    sessions = value["sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("At least one complete physical session is required")
    ids, errors = ([], [])
    for session in sessions:
        name = session["session_id"]
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Physical session identifiers must be nonempty and unique")
        truth = np.asarray(session["truth"], dtype=float)
        prediction = np.full((len(keys), 2), np.nan)
        supplied = session.get("prediction")
        if isinstance(supplied, list) and len(supplied) == len(keys):
            for index, row in enumerate(supplied):
                try:
                    point = np.asarray(row, dtype=float)
                    if point.shape == (2,) and (not any((isinstance(v, bool) for v in row))):
                        prediction[index] = point
                except (TypeError, ValueError, OverflowError):
                    pass
        if truth.shape != (len(keys), 2) or not np.all(np.isfinite(truth)):
            raise ValueError("Complete finite truth is required for every indexed position")
        with np.errstate(over="ignore", invalid="ignore"):
            error = np.linalg.norm(truth - prediction, axis=1)
        error[~np.isfinite(error)] = np.inf
        ids.append(name)
        errors.append(error)
    return (value, ids, np.asarray(errors))


def session_scores(errors, scales):
    errors, scales = (np.asarray(errors, dtype=float), np.asarray(scales, dtype=float))
    if errors.ndim != 2 or scales.shape != (errors.shape[1],):
        raise ValueError("Error and scale dimensions disagree")
    if np.any(np.isnan(errors)) or np.any(errors < 0):
        raise ValueError("Errors must be nonnegative, with infinity for failed forecasts")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Scales must be positive and finite")
    with np.errstate(over="ignore"):
        return np.max(errors / scales, axis=1)


def numeric_vector(values, *, infinity_for_null=False):
    if not isinstance(values, list):
        raise TypeError("A numeric list is required")
    for value in values:
        if value is None and infinity_for_null:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Numeric records cannot contain boolean or text values")
    return np.asarray([np.inf if v is None else v for v in values], dtype=float)


def conformal_quantile(scores, delta):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not scores.size or np.any(np.isnan(scores)) or np.any(scores < 0):
        raise ValueError("Nonnegative physical-session scores are required")
    if isinstance(delta, bool) or not np.isfinite(delta) or (not 0 < delta < 1):
        raise ValueError("Failure probability must lie strictly between zero and one")
    rank = int(np.ceil((scores.size + 1) * (1 - delta)))
    return float(np.sort(np.r_[scores, np.inf])[rank - 1])
