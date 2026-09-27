import numpy as np

from .forecast_errors import _TOL, OFFSETS_S, _error, _errors_over, _rate


def offset_metrics(updates, selected, horizons, names, truth_times, truth_points, q):
    fixed = {}
    for offset in OFFSETS_S:
        metric = {
            "offset_s": offset,
            "eligible_object_predictions_n": 0,
            "censored_object_predictions_n": 0,
            "outside_declared_horizon_n": 0,
            "unknown_horizon_n": 0,
            "missing_or_invalid_n": 0,
            "truth_unavailable_n": 0,
            "error_observed_n": 0,
            "covered_n": 0,
        }
        fde, ade = ([], [])
        for update, forecasts, horizon in zip(updates, selected, horizons, strict=True):
            if horizon is None:
                metric["unknown_horizon_n"] += len(names)
                continue
            if offset > horizon + _TOL:
                metric["outside_declared_horizon_n"] += len(names)
                continue
            origin = float(update["origin_time"])
            endpoint = origin + offset
            if (
                not len(truth_times)
                or origin < truth_times[0] - _TOL
                or endpoint > truth_times[-1] + _TOL
            ):
                metric["censored_object_predictions_n"] += len(names)
                continue
            for name in names:
                metric["eligible_object_predictions_n"] += 1
                error, reason = _error(
                    forecasts.get(name), name, endpoint, truth_times, truth_points
                )
                if error is None:
                    key = (
                        "truth_unavailable_n"
                        if reason == "truth_unavailable"
                        else "missing_or_invalid_n"
                    )
                    metric[key] += 1
                    continue
                fde.append(error)
                metric["error_observed_n"] += 1
                metric["covered_n"] += int(error <= q + _TOL)
                errors, missing, _ = _errors_over(
                    forecasts.get(name), name, origin, endpoint, truth_times, truth_points
                )
                if not missing:
                    ade.append(float(np.mean(errors)))
        metric["fde_m"] = float(np.mean(fde)) if fde else None
        metric["ade_m"] = float(np.mean(ade)) if ade else None
        metric["ade_observed_forecasts_n"] = len(ade)
        metric["coverage_fraction"] = _rate(
            metric["covered_n"], metric["eligible_object_predictions_n"]
        )
        metric["coverage_scope"] = (
            "endpoint center error <= q; ADE uses future observed-grid points excluding the origin"
        )
        fixed[f"{offset:g}"] = metric
    return fixed
