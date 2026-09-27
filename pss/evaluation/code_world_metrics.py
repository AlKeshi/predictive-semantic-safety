from __future__ import annotations

import numpy as np

from .forecast_errors import (
    _TOL,
    _error,
    _errors_over,
    _finite_time,
    _forecasts,
    _horizon,
    _rate,
    _unchanged_carry,
)
from .forecast_offsets import offset_metrics


def forecast_metrics(updates, truth, objects, q, committed_only=False):
    q = float(q)
    if not np.isfinite(q) or q < 0:
        raise ValueError("q must be finite and nonnegative")
    names = [str(obj.get("id", obj.get("object_id", ""))) for obj in objects]
    if not names or any((not name for name in names)) or len(names) != len(set(names)):
        raise ValueError("objects must declare unique nonempty IDs")
    truth_times = np.asarray([_finite_time(sample["time"]) for sample in truth])
    if np.any(np.diff(truth_times) <= 0):
        raise ValueError("truth times must be strictly increasing")
    truth_points = {}
    for name in names:
        points = np.full((len(truth), 2), np.nan)
        for index, sample in enumerate(truth):
            try:
                value = np.asarray(sample["objects"][name], dtype=float)
                if value.ndim == 1 and value.size >= 2 and np.all(np.isfinite(value[:2])):
                    points[index] = value[:2]
            except (KeyError, ValueError, TypeError):
                pass
        truth_points[name] = points
    origins = np.asarray([_finite_time(update["origin_time"]) for update in updates])
    if np.any(np.diff(origins) < 0):
        raise ValueError("scheduled updates must be chronological")
    parsed = [_forecasts(update) for update in updates]
    horizons = [
        _horizon(update, forecasts) for update, forecasts in zip(updates, parsed, strict=True)
    ]
    selected = [
        forecasts if not committed_only or bool(update.get("accepted")) else {}
        for update, forecasts in zip(updates, parsed, strict=True)
    ]
    executed = {
        "scheduled_object_time_points_n": 0,
        "covered_n": 0,
        "error_observed_n": 0,
        "missing_or_unsupported_forecast_n": 0,
        "not_enforced_n": 0,
        "truth_unavailable_n": 0,
        "start_time_s": None,
        "end_time_s": None,
    }
    executed_errors = []
    next_update = 0
    active = {}
    committed_receipt = None
    rejected_latch = True
    enforcement_metadata_complete = True
    for sample_index, when in enumerate(truth_times):
        if not len(origins) or when < origins[0] - _TOL:
            continue
        while next_update < len(updates) and origins[next_update] <= when + _TOL:
            if committed_only:
                if bool(updates[next_update].get("accepted")):
                    active = parsed[next_update]
                    committed_receipt = updates[next_update].get("proposed_forecasts")
                    rejected_latch = False
                elif not rejected_latch and _unchanged_carry(
                    updates[next_update], committed_receipt
                ):
                    pass
                else:
                    rejected_latch = True
            else:
                active = parsed[next_update]
            next_update += 1
        if executed["start_time_s"] is None:
            executed["start_time_s"] = float(when)
        executed["end_time_s"] = float(when)
        enforced = truth[sample_index].get("enforced_object_ids")
        if committed_only and enforced is None:
            enforcement_metadata_complete = False
        for name in names:
            executed["scheduled_object_time_points_n"] += 1
            if committed_only and (
                rejected_latch or (enforced is not None and name not in enforced)
            ):
                executed["not_enforced_n"] += 1
                continue
            error, reason = _error(active.get(name), name, float(when), truth_times, truth_points)
            if error is None:
                executed[reason + "_n"] += 1
            else:
                executed_errors.append(error)
                executed["error_observed_n"] += 1
                executed["covered_n"] += int(error <= q + _TOL)
    count = executed["scheduled_object_time_points_n"]
    executed["fraction"] = _rate(executed["covered_n"], count)
    executed["all_observed_scope_covered"] = bool(executed["covered_n"] == count) if count else None
    executed["enforcement_metadata_complete"] = (
        enforcement_metadata_complete if committed_only else None
    )
    executed["scope"] = (
        "actual committed-and-enforced geometry on observed execution times"
        if committed_only and enforcement_metadata_complete
        else "committed geometry on observed times; actual enforcement metadata incomplete"
        if committed_only
        else "latest scheduled proposal on observed execution times, including rejected proposals"
    )
    full = {
        "scheduled_updates_n": len(updates),
        "eligible_updates_n": 0,
        "covered_updates_n": 0,
        "censored_updates_n": 0,
        "unknown_horizon_updates_n": 0,
        "eligible_missing_or_rejected_updates_n": 0,
        "eligible_object_predictions_n": 0,
        "error_observed_points_n": 0,
        "missing_points_n": 0,
        "covered_points_n": 0,
    }
    full_errors, final_errors = ([], [])
    for update, forecasts, horizon in zip(updates, selected, horizons, strict=True):
        if horizon is None:
            full["unknown_horizon_updates_n"] += 1
            continue
        origin = float(update["origin_time"])
        endpoint = origin + horizon
        if (
            not len(truth_times)
            or origin < truth_times[0] - _TOL
            or endpoint > truth_times[-1] + _TOL
        ):
            full["censored_updates_n"] += 1
            continue
        full["eligible_updates_n"] += 1
        full["eligible_object_predictions_n"] += len(names)
        complete_inventory = all((name in forecasts for name in names))
        full["eligible_missing_or_rejected_updates_n"] += int(not complete_inventory)
        whole_covered = True
        for name in names:
            errors, missing, _ = _errors_over(
                forecasts.get(name), name, origin, endpoint, truth_times, truth_points
            )
            full_errors.extend(errors)
            full["error_observed_points_n"] += len(errors)
            full["missing_points_n"] += missing
            full["covered_points_n"] += sum((error <= q + _TOL for error in errors))
            whole_covered &= (
                not missing and bool(errors) and all((error <= q + _TOL for error in errors))
            )
            final_error, _ = _error(forecasts.get(name), name, endpoint, truth_times, truth_points)
            if final_error is not None:
                final_errors.append(final_error)
        full["covered_updates_n"] += int(whole_covered)
    full["update_coverage_fraction"] = _rate(full["covered_updates_n"], full["eligible_updates_n"])
    full["pointwise_coverage_fraction"] = _rate(
        full["covered_points_n"], full["error_observed_points_n"] + full["missing_points_n"]
    )
    full["ade_m"] = float(np.mean(full_errors)) if full_errors else None
    full["fde_m"] = float(np.mean(final_errors)) if final_errors else None
    full["fde_observed_object_predictions_n"] = len(final_errors)
    full["sampled_interval"] = "(observation origin, declared horizon endpoint]"
    full["scope"] = (
        "Each eligible accepted family evaluated over its entire declared horizon, including after replacement; rejected or missing scheduled updates count as uncovered"
        if committed_only
        else "Each eligible proposal evaluated over its entire declared horizon; missing scheduled updates count as uncovered"
    )
    fixed = offset_metrics(updates, selected, horizons, names, truth_times, truth_points, q)
    actual_whole = executed["all_observed_scope_covered"]
    if committed_only and (not enforcement_metadata_complete):
        actual_whole = None
    return {
        "schema": "code_world_forecast_metrics_v3",
        "forecast_metric_scope": "committed_and_enforced" if committed_only else "all_proposals",
        "q_m": q,
        "calibrated": False,
        "coverage_guarantee": None,
        "forecast_ade_m": float(np.mean(executed_errors)) if executed_errors else None,
        "forecast_max_error_m": max(executed_errors) if executed_errors else None,
        "empirical_pointwise_coverage": executed["fraction"],
        "whole_executed_trace_covered": actual_whole,
        "missing_forecast_updates": sum(
            (not all((name in forecasts for name in names)) for forecasts in selected)
        ),
        "scheduled_updates_n": len(updates),
        "accepted_updates_n": sum((bool(update.get("accepted")) for update in updates)),
        "executed_time_coverage": executed,
        "full_horizon": full,
        "horizon_metrics": fixed,
        "truth_sampling": "Recorded simulator centers; linear interpolation only between available bracketing truth samples for exact horizon endpoints. No truth extrapolation.",
        "interpretation": "Full-horizon coverage concerns eligible forecasts only and is not whole-session coverage. Late forecasts remain in executed-time coverage. Missing predictions reduce coverage; error averages report only numerically observed predictions, with missing counts separate.",
    }
