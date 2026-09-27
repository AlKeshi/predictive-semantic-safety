from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from pss.control.code_world import Forecast

MAX_FUTURE_CAMERA_RANGE_M = 20.0
MAX_FORECAST_SPEED_M_S = 8.0
FORECAST_SAMPLES = 5
HISTORY_SAMPLES = 6


def _finite_array(value, shape, label):
    values = np.asarray(value, dtype=object)
    if values.shape != shape or any(
        (
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, float, np.integer, np.floating))
            for item in values.flat
        )
    ):
        raise ValueError(f"{label} must contain numeric values with shape {shape}")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


def _times(origin, horizon):
    origin = float(_finite_array(origin, (), "origin"))
    horizon = float(_finite_array(horizon, (), "horizon"))
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    offsets = np.arange(1, FORECAST_SAMPLES + 1, dtype=float) * horizon / FORECAST_SAMPLES
    absolute_times = np.r_[origin, origin + offsets]
    if not np.all(np.isfinite(absolute_times)) or np.any(np.diff(absolute_times) <= 0):
        raise ValueError("forecast support must be finite and strictly increasing")
    return (origin, offsets)


def _identities(objects):
    objects = list(objects)
    if not objects or any((not isinstance(item, Mapping) for item in objects)):
        raise ValueError("a nonempty declared object inventory is required")
    names = [item.get("id") for item in objects]
    if any((not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError("declared object identities must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("duplicate declared object identity")
    return (objects, names)


def _camera_pose(value):
    pose = _finite_array(value, (4, 4), "camera pose")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-06, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-06, rtol=0)
        or (not np.isclose(np.linalg.det(rotation), 1, atol=1e-06, rtol=0))
    ):
        raise ValueError("camera pose must be a rigid optical-camera-to-world transform")
    return pose


def _measured_center(name, detection):
    if getattr(detection, "object_id", None) != name:
        raise ValueError(f"observed identity conflicts with {name}")
    center = _finite_array(detection.center, (3,), f"observed center for {name}")
    depth = float(_finite_array(detection.depth, (), f"observed depth for {name}"))
    radius = float(_finite_array(detection.radius, (), f"observed radius for {name}"))
    if depth <= 0 or radius < 0:
        raise ValueError(f"invalid observed depth or radius for {name}")
    return (center, radius)


def metric_trajectory_prompt(
    objects, history, *, origin, horizon=2.5, camera_pose=None, route=None
):
    objects, names = _identities(objects)
    origin, future_times = _times(origin, horizon)
    history = list(history)
    if not history:
        raise ValueError("observed RGB-D history is required")
    times = _finite_array([item[0] for item in history], (len(history),), "history times")
    if np.any(np.diff(times) <= 0) or np.any(times > origin + 1e-06):
        raise ValueError("RGB-D history must be chronological and causal")
    if not np.isclose(times[-1], origin, atol=1e-06, rtol=0):
        raise ValueError("forecast origin must equal the latest observed timestamp")
    tracks = {name: [] for name in names}
    validated = []
    for _, detections in history:
        if not isinstance(detections, Mapping) or not set(detections).issubset(names):
            raise ValueError("RGB-D history contains an undeclared identity")
        validated.append(
            {name: _measured_center(name, detection)[0] for name, detection in detections.items()}
        )
    if set(validated[-1]) != set(names):
        raise ValueError("every declared object requires a current RGB-D observation")
    indices = np.linspace(0, len(history) - 1, min(HISTORY_SAMPLES, len(history)), dtype=int)
    for index in indices:
        for name, center in validated[index].items():
            tracks[name].append(
                [round(float(times[index] - origin), 4), *np.round(center, 4).tolist()]
            )
    inventory = []
    for spec in objects:
        name = spec["id"]
        item = {"id": name, "current_center_m": np.round(validated[-1][name], 4).tolist()}
        if "shape" in spec:
            item["shape"] = str(spec["shape"])
        if "color" in spec:
            color = _finite_array(spec["color"], (3,), f"color for {name}")
            if np.any(color < 0) or np.any(color > 1):
                raise ValueError("declared colors must be RGB in [0,1]")
            item["color_rgb"] = np.round(color, 3).tolist()
        if spec.get("size") is not None:
            size = _finite_array(spec["size"], (3,), f"size for {name}")
            if np.any(size <= 0):
                raise ValueError("declared dimensions must be positive")
            item["size_m"] = np.round(size, 4).tolist()
        item["extent_radius_m"] = float(_measured_center(name, history[-1][1][name])[1])
        inventory.append(item)
    context = {"objects": inventory, "observed_history_s_xyz_m": tracks}
    if camera_pose is not None:
        pose = _camera_pose(camera_pose)
        context["last_camera_optical_to_world"] = np.round(pose, 5).tolist()
    if route is not None:
        waypoints = np.asarray(route)
        if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 2:
            raise ValueError("nominal route must contain at least two world-XY waypoints")
        context["nominal_route_xy_m"] = _finite_array(waypoints, waypoints.shape, "route").tolist()
    return (
        "Predict physical motion of EVERY listed object after the LAST video frame. Use the video to infer support, gravity, rolling, collisions and momentum transfer. An object stationary now may move after a visible incoming impact or loss of support; a moving object may stop or redirect after contact. Estimate event timing from the observed geometry and motion. Do not invent an unseen impact, wall or force. The following are actual past RGB-D center measurements, not future states. They are already in a FIXED WORLD frame: metres, z up, ground z=0. World x/y are fixed horizontal axes, NOT image right/down and NOT the robot heading. The optical camera axes are x right, y down, z forward; its transform explains the last image orientation only. Camera motion does not move world coordinates. History rows are [seconds relative to last frame,world_x,world_y,world_z]. "
        + json.dumps(context, separators=(",", ":"), allow_nan=False)
        + f" Return one JSON object with key objects containing exactly one entry for each of these exact IDs: {json.dumps(names)}. Each entry has id, a short motion_description, and trajectory: exactly five numeric [dt,dx,dy,dz] rows at dt="
        + json.dumps(np.round(future_times, 8).tolist())
        + " seconds. Each displacement is TOTAL displacement in WORLD METRES from that object's current_center_m, NOT displacement from the previous predicted point. The measured center is fixed at dt=0; do not output or replace it. Use observed distances and elapsed seconds for scale. Predict each interaction's effect on subsequent motion. Preserve observed motion when no interaction changes it. Include every object even if it is predicted to miss the route or leave the image. Do not output pixel coordinates, absolute future positions, code, or markdown."
    )


def _entries(payload, names):
    values = payload.get("objects", payload) if isinstance(payload, dict) else payload
    if isinstance(values, dict):
        if set(values) != set(names):
            raise ValueError("model object map must match the entire declared inventory")
        entries = []
        for name, entry in values.items():
            if not isinstance(entry, dict):
                raise ValueError("each model object must be an object")
            if "id" in entry and entry["id"] != name:
                raise ValueError("conflicting mapped and embedded model identity")
            entries.append({"id": name, **entry})
        return entries
    if not isinstance(values, list):
        raise ValueError("model objects must be a list or exact identity map")
    return values


def lift_metric_response(payload, detections, objects, *, origin, horizon=2.5, camera_pose=None):
    objects, names = _identities(objects)
    origin, expected_times = _times(origin, horizon)
    if not isinstance(detections, Mapping) or set(detections) != set(names):
        raise ValueError("every declared object requires a current RGB-D observation")
    pose = _camera_pose(camera_pose) if camera_pose is not None else None
    entries = _entries(payload, names)
    if len(entries) != len(names) or any((not isinstance(item, dict) for item in entries)):
        raise ValueError("model response must include the entire object inventory")
    returned_names = [entry.get("id") for entry in entries]
    if any((not isinstance(name, str) for name in returned_names)):
        raise ValueError("model identities must be exact strings")
    if len(set(returned_names)) != len(names) or set(returned_names) != set(names):
        raise ValueError("model identity missing, duplicated, or undeclared")
    by_name = {entry["id"]: entry for entry in entries}
    forecasts = []
    for name in names:
        center, radius = _measured_center(name, detections[name])
        trajectory = by_name[name].get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != FORECAST_SAMPLES:
            raise ValueError("metric trajectory requires exactly five explicit-time samples")
        rows = []
        for sample in trajectory:
            if isinstance(sample, dict):
                if not all((key in sample for key in ("dt", "dx", "dy", "dz"))):
                    raise ValueError("metric sample requires dt,dx,dy,dz")
                sample = [sample[key] for key in ("dt", "dx", "dy", "dz")]
            rows.append(_finite_array(sample, (4,), "metric [dt,dx,dy,dz] sample"))
        values = np.asarray(rows)
        if not np.allclose(values[:, 0], expected_times, atol=1e-08, rtol=0):
            raise ValueError("metric times must match the five declared future offsets")
        points = np.vstack((center, center + values[:, 1:]))
        intervals = np.diff(np.r_[0.0, values[:, 0]])
        if np.any(
            np.linalg.norm(np.diff(points, axis=0), axis=1)
            > MAX_FORECAST_SPEED_M_S * intervals + 1e-08
        ):
            raise ValueError("consecutive metric forecast speed exceeds numeric limit")
        if not np.all(np.isfinite(points)):
            raise ValueError("metric world points must be finite")
        if pose is not None and np.any(
            np.linalg.norm(points - pose[:3, 3], axis=1) > MAX_FUTURE_CAMERA_RANGE_M
        ):
            raise ValueError("metric forecast exceeds numeric camera range")
        forecasts.append(
            Forecast(name, origin, origin + np.r_[0.0, values[:, 0]], points[:, :2].copy(), radius)
        )
    return forecasts
