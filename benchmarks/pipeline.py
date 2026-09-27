import numpy as np

from .identity import describe as describe


class CalibratedSession:
    def __init__(self, record, actual_digest, *, duration=None, family=None, margin=True):
        self.record = record
        self.digest = actual_digest
        self.margin = margin
        self.origins = sorted({float(item["origin_time"]) for item in record.indices})
        self.seen = set()
        self.invalid_reason = None
        self.prediction = [None] * len(record.indices)
        self.truth = [None] * len(record.indices)
        if duration is not None:
            self.validate_schedule(duration, family)

    def validate_schedule(self, duration, family):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("A finite positive session duration is required")
        earliest, cadence, horizon, count = (
            (0.75, 0.05, 8.0, 80) if family == "ceiling" else (0.1, 0.1, 2.5, 5)
        )
        for origin in self.origins:
            if origin < earliest - 1e-08 or not np.isclose(
                origin / cadence, round(origin / cadence), atol=1e-07, rtol=0
            ):
                raise ValueError("Required origin is unavailable on the acquisition/control grid")
            if origin + horizon > duration + 1e-08:
                raise ValueError("Required future truth extends beyond the session duration")
            rows = [i for i in self.record.indices if abs(i["origin_time"] - origin) < 1e-08]
            for name in {i["object_id"] for i in rows}:
                actual = sorted((i["future_time"] for i in rows if i["object_id"] == name))
                expected = origin + np.arange(1, count + 1) * horizon / count
                if len(actual) != count or not np.allclose(actual, expected, atol=1e-08, rtol=0):
                    raise ValueError("Required future grid differs from the frozen predictor")

    def next_origin_after(self, now):
        return next((origin for origin in self.origins if origin > now + 1e-08), np.inf)

    def needs_truth(self, now):
        return any(
            (
                value is None and abs(item["future_time"] - now) <= 1e-07
                for item, value in zip(self.record.indices, self.truth, strict=True)
            )
        )

    def observe_simulation(self, now, model, data, object_bodies):
        import mujoco

        scratch = getattr(self, "_truth_data", None)
        if scratch is None:
            scratch = self._truth_data = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        scratch.mocap_pos[:] = data.mocap_pos
        scratch.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(model, scratch)
        self.observe_truth(now, {name: scratch.xpos[body] for name, body in object_bodies.items()})

    def observe_truth(self, now, objects):
        for slot, item in enumerate(self.record.indices):
            if abs(item["future_time"] - now) <= 1e-07:
                point = np.asarray(objects[item["object_id"]], dtype=float)[:2]
                if point.shape != (2,) or not np.all(np.isfinite(point)):
                    raise ValueError("Indexed physical truth must be finite XY")
                self.truth[slot] = point.tolist()

    def trace(self, session_id):
        data = self.record.to_dict()
        return {
            "schema": "pss.calibration-session-trace.v1",
            "pipeline": data["pipeline"],
            "pipeline_digest": self.digest,
            "indices": self.record.indices,
            "shape_radii": data["shape_radii"],
            "acquisition_record_digest": self.record.record_digest,
            "residual_margin_applied": self.margin,
            "session_id": session_id,
            "prediction": self.prediction,
            "truth": self.truth,
        }

    def invalidate(self, reason):
        self.invalid_reason = str(reason)

    def __call__(self, forecasts):
        forecasts = list(forecasts)
        from pss.calibration import WholePlaneForecast

        if self.invalid_reason is not None:
            raise WholePlaneForecast(self.invalid_reason)
        try:
            result = self.record.inflate_forecasts(forecasts, pipeline_digest=self.digest)
        except ValueError as error:
            self.invalidate(error)
            raise
        for slot, item in enumerate(self.record.indices):
            for forecast in forecasts:
                if (
                    item["object_id"] == forecast.object_id
                    and abs(item["origin_time"] - forecast.origin_time) <= 1e-08
                ):
                    indices = np.flatnonzero(
                        np.isclose(forecast.times, item["future_time"], atol=1e-08, rtol=0)
                    )
                    if len(indices) == 1:
                        self.prediction[slot] = np.asarray(forecast.centers[indices[0]]).tolist()
        self.seen.update((float(forecast.origin_time) for forecast in forecasts))
        return result if self.margin else forecasts

    def check_time(self, now):
        from pss.calibration import WholePlaneForecast

        missing = [
            origin
            for origin in self.origins
            if origin < now - 1e-07
            and (not any((abs(origin - seen) <= 1e-07 for seen in self.seen)))
        ]
        if missing:
            self.invalidate(f"Missing required prediction origins: {missing}")
        if self.invalid_reason is not None:
            raise WholePlaneForecast(self.invalid_reason)

    def receipt(self):
        result = self.record.receipt()
        result.update(
            calibrated=self.margin,
            residual_margin_applied=self.margin,
            required_origins=self.origins,
            observed_origins=sorted(self.seen),
            invalid_reason=self.invalid_reason,
            all_required_predictions_present=self.invalid_reason is None
            and len(self.seen) == len(self.origins),
        )
        return result


def enforce_required_origins(controller, transform, now, state):
    if transform is None:
        return
    from pss.calibration import WholePlaneForecast

    try:
        transform.check_time(now)
    except WholePlaneForecast as error:
        controller.invalidate(error)
