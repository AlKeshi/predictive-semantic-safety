from __future__ import annotations

import numpy as np

OFFSETS_S = (0.5, 1.0, 2.0, 2.5)
_TOL = 1e-08


def _finite_time(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("metric times must be finite")
    return number


def _forecasts(update):
    result = {}
    for forecast in update.get("proposed_forecasts", []) or []:
        try:
            name = str(forecast["object_id"])
            times = np.asarray(forecast["times"], dtype=float)
            centers = np.asarray(forecast["centers"], dtype=float)
            if name in result:
                return {}
            if (
                times.ndim != 1
                or times.size < 2
                or centers.shape != (times.size, 2)
                or (not np.all(np.isfinite(times)))
                or (not np.all(np.isfinite(centers)))
                or np.any(np.diff(times) <= 0)
            ):
                continue
            result[name] = (times, centers)
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _unchanged_carry(update, committed_receipt):
    if update.get("retained_committed_family") is not True or not committed_receipt:
        return False
    try:
        carried = update["carried_forecasts"]
        left = {item["object_id"]: item for item in carried}
        right = {item["object_id"]: item for item in committed_receipt}
        if len(left) != len(carried) or set(left) != set(right):
            return False
        for name, item in left.items():
            for key in ("origin_time", "times", "centers", "radii"):
                a = np.asarray(item[key], dtype=float)
                b = np.asarray(right[name][key], dtype=float)
                if key == "radii":
                    a, b = np.broadcast_arrays(a, b)
                if not np.all(np.isfinite(a)) or not np.array_equal(a, b):
                    return False
            when = float(update["origin_time"])
            if when < item["times"][0] - _TOL or when >= item["times"][-1] - _TOL:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _predict(forecast, when):
    if forecast is None:
        return None
    times, centers = forecast
    if when < times[0] - _TOL or when > times[-1] + _TOL:
        return None
    return np.array([np.interp(when, times, centers[:, axis]) for axis in range(2)])


def _truth_at(times, points, when):
    if not len(times) or when < times[0] - _TOL or when > times[-1] + _TOL:
        return None
    index = int(np.searchsorted(times, when))
    if index < len(times) and abs(times[index] - when) <= _TOL:
        value = points[index]
        return value if np.all(np.isfinite(value)) else None
    if index > 0 and abs(times[index - 1] - when) <= _TOL:
        value = points[index - 1]
        return value if np.all(np.isfinite(value)) else None
    if (
        index == 0
        or index >= len(times)
        or (not np.all(np.isfinite(points[index - 1 : index + 1])))
    ):
        return None
    weight = (when - times[index - 1]) / (times[index] - times[index - 1])
    return (1 - weight) * points[index - 1] + weight * points[index]


def _error(forecast, name, when, truth_times, truth_points):
    truth = _truth_at(truth_times, truth_points[name], when)
    prediction = _predict(forecast, when)
    if truth is None:
        return (None, "truth_unavailable")
    if prediction is None:
        return (None, "missing_or_unsupported_forecast")
    return (float(np.linalg.norm(prediction - truth)), None)


def _errors_over(forecast, name, start, end, truth_times, truth_points):
    times = np.unique(np.r_[truth_times[(truth_times > start) & (truth_times < end)], end])
    errors = []
    missing = 0
    for when in times:
        error, _ = _error(forecast, name, float(when), truth_times, truth_points)
        if error is None:
            missing += 1
        else:
            errors.append(error)
    return (errors, missing, len(times))


def _rate(count, total):
    return count / total if total else None


def _horizon(update, forecasts):
    declared = update.get("horizon_s")
    if declared is not None:
        try:
            value = float(declared)
            return value if np.isfinite(value) and value > 0 else None
        except (ValueError, TypeError):
            return None
    ends = [times[-1] - float(update["origin_time"]) for times, _ in forecasts.values()]
    return max(ends) if ends and max(ends) > 0 else None
