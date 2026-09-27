import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from pss.control.forecast import Forecast

from .data import (
    SCHEMA,
    conformal_quantile,
    numeric_vector,
    pipeline_digest,
    validate_indices,
    validate_shapes,
)


class WholePlaneForecast(ValueError):
    pass


class CalibrationRecord:
    def __init__(self, data):
        data = deepcopy(data)
        if data.get("schema") != SCHEMA:
            raise ValueError("Unsupported calibration schema")
        keys = validate_indices(data["indices"])
        validate_shapes(data["shape_radii"], keys)
        if data["pipeline_digest"] != pipeline_digest(data["pipeline"]):
            raise ValueError("Pipeline fingerprint does not match its description")
        training, calibration = (data["training_session_ids"], data["calibration_session_ids"])
        for ids in (training, calibration):
            if (
                not isinstance(ids, list)
                or not ids
                or any((not isinstance(i, str) or not i.strip() for i in ids))
            ):
                raise ValueError("Complete physical-session identities are required")
            if len(set(ids)) != len(ids):
                raise ValueError("A physical session cannot be counted twice")
        if set(training) & set(calibration):
            raise ValueError("Training and calibration sessions overlap")
        acquisition = data.get("acquisition_records")
        if not isinstance(acquisition, dict) or set(acquisition) != {"training", "calibration"}:
            raise ValueError("Acquisition records are required for both session splits")
        digests = []
        for split, ids in (("training", training), ("calibration", calibration)):
            records = acquisition[split]
            if not isinstance(records, dict) or set(records) != set(ids):
                raise ValueError("Acquisition records must identify every physical session")
            digests.extend(records.values())
        if (
            any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or len(set(digests)) != 1
        ):
            raise ValueError("Sessions require one nonempty SHA-256 acquisition record digest")
        scales = numeric_vector(data["scales"])
        training_maxima = numeric_vector(data["training_max_errors"])
        minimum = data["minimum_scale"]
        if isinstance(minimum, bool) or not np.isfinite(minimum) or minimum <= 0:
            raise ValueError("Minimum training scale must be positive and finite")
        if scales.shape != (len(keys),) or training_maxima.shape != scales.shape:
            raise ValueError("Scale inventory differs from the frozen index")
        if not np.all(np.isfinite(training_maxima)) or np.any(training_maxima < 0):
            raise ValueError("Training maximum errors must be finite and nonnegative")
        if not np.array_equal(scales, np.maximum(training_maxima, minimum)):
            raise ValueError("Scales do not match the declared training-only construction")
        scores = numeric_vector(data["scores"], infinity_for_null=True)
        if scores.shape != (len(calibration),):
            raise ValueError("Exactly one score per calibration session is required")
        q = conformal_quantile(scores, data["delta"])
        if isinstance(data["q"], bool) or (
            data["q"] is not None and (not isinstance(data["q"], (int, float)))
        ):
            raise ValueError("Quantile must be numeric or null for infinity")
        supplied_q = np.inf if data["q"] is None else float(data["q"])
        if q != supplied_q:
            raise ValueError("Quantile differs from the finite-sample order statistic")
        self._data, self._keys = (data, keys)
        self._q = q
        self._scales = scales
        self._scales.setflags(write=False)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @property
    def pipeline_digest(self):
        return self._data["pipeline_digest"]

    @property
    def record_digest(self):
        encoded = json.dumps(self._data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def indices(self):
        return deepcopy(self._data["indices"])

    @property
    def q(self):
        return self._q

    @property
    def delta(self):
        return self._data["delta"]

    def to_dict(self):
        return deepcopy(self._data)

    def save(self, path):
        Path(path).write_text(json.dumps(self._data, indent=2, allow_nan=False) + "\n")

    def receipt(self):
        return {
            "schema": SCHEMA,
            "pipeline_digest": self.pipeline_digest,
            "calibration_record_digest": self.record_digest,
            "calibrated": True,
            "quantile_source": "fitted_session_scores",
            "quantile_units": "dimensionless",
            "residual_radius_rule": "q * training_scale_at_object_origin_future_index",
            "delta": self.delta,
            "q": None if not np.isfinite(self.q) else self.q,
            "calibration_sessions": len(self._data["calibration_session_ids"]),
            "required_positions_per_session": len(self._keys),
            "scope": "declared objects and sampled future times",
        }

    def inflate_forecasts(self, forecasts, *, pipeline_digest):
        try:
            return self._inflate(forecasts, pipeline_digest)
        except (TypeError, ValueError, KeyError, OverflowError, FloatingPointError) as exc:
            raise WholePlaneForecast(str(exc)) from exc

    def _inflate(self, forecasts, actual_digest):
        if actual_digest != self.pipeline_digest:
            raise ValueError("Runtime differs from the frozen calibration pipeline")
        if not np.isfinite(self.q):
            raise ValueError("Infinite conformal quantile defines the whole plane")
        forecasts = list(forecasts)
        if not forecasts or any((not isinstance(f, Forecast) for f in forecasts)):
            raise ValueError("Missing required forecast family")
        origin = float(forecasts[0].origin_time)
        indices = [
            i for i, key in enumerate(self._keys) if np.isclose(key[1], origin, atol=1e-08, rtol=0)
        ]
        names = {self._keys[i][0] for i in indices}
        if not indices or len(forecasts) != len(names) or {f.object_id for f in forecasts} != names:
            raise ValueError("Forecast inventory or observation origin is not calibrated")
        result = []
        for forecast in forecasts:
            slots = sorted(
                (i for i in indices if self._keys[i][0] == forecast.object_id),
                key=lambda i: self._keys[i][2],
            )
            future_times = np.array([self._keys[i][2] for i in slots])
            times = np.array(forecast.times, dtype=float, copy=True)
            centers = np.array(forecast.centers, dtype=float, copy=True)
            radii = np.array(forecast.radii, dtype=float, copy=True)
            if radii.ndim == 0:
                radii = np.full(times.shape, radii)
            expected = np.r_[origin, future_times]
            if times.shape != expected.shape or not np.allclose(
                times, expected, atol=1e-08, rtol=0
            ):
                raise ValueError("Forecast times differ from the fixed calibration grid")
            if not np.isclose(forecast.origin_time, origin, atol=1e-08, rtol=0):
                raise ValueError("Family observation origins disagree")
            if centers.shape != (len(times), 2) or radii.shape != times.shape:
                raise ValueError("Forecast geometry has invalid dimensions")
            if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
                raise ValueError("Invalid test forecast defines the whole plane")
            if np.any(radii < self._data["shape_radii"][forecast.object_id]):
                raise ValueError("Forecast radius omits the declared enclosing shape")
            with np.errstate(over="ignore"):
                radii[1:] += self.q * self._scales[slots]
            if not np.all(np.isfinite(radii)):
                raise ValueError("Unbounded prediction region")
            result.append(Forecast(forecast.object_id, origin, times, centers, radii))
        return result
