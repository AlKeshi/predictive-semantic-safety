from __future__ import annotations

import json

import numpy as np

from pss.control.code_world import Forecast


def causal_facts(history, diagnostic):
    observed = [(t, ds["ceiling_fixture"].center) for t, ds in history if "ceiling_fixture" in ds]
    if len(observed) < 4 or history[-1][1].get("ceiling_fixture") is None:
        raise ValueError("current fixture and at least four past RGB-D observations required")
    rows = np.array([[t, *center] for t, center in observed])
    rows[:, 0] -= rows[-1, 0]
    selected = np.linspace(0, len(rows) - 1, min(6, len(rows)), dtype=int)
    represented = np.round(rows[selected], 4)
    times = represented[:, 0] - represented[:, 0].mean()
    if times @ times < 1e-05:
        raise ValueError("insufficient observed elapsed time")
    velocity = times @ (represented[:, 1:] - represented[:, 1:].mean(0)) / (times @ times)
    if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) > 8:
        raise ValueError("unusable RGB-D measured velocity")
    return {
        "current_center_m": represented[-1, 1:].tolist(),
        "past_observed_s_xyz_m": represented.tolist(),
        "measured_world_velocity_m_s": velocity.tolist(),
        "current_surface_normal_world": diagnostic.get("normal"),
        "dimensions_m": [1.18, 0.46, 0.1],
    }


def parse_event(payload, horizon):
    entry = payload
    if isinstance(entry, dict) and "objects" in entry:
        entry = entry["objects"]
    if isinstance(entry, list):
        if len(entry) != 1:
            raise ValueError("one exact fixture event is required")
        entry = entry[0]
    if isinstance(entry, dict) and "ceiling_fixture" in entry:
        entry = entry["ceiling_fixture"]
    if not isinstance(entry, dict) or entry.get("id") != "ceiling_fixture":
        raise ValueError("missing exact fixture identity")
    state = entry.get("attachment_state")
    if state not in {"secure", "loose", "detached", "grounded", "uncertain"}:
        raise ValueError("invalid attachment state")
    mode = entry.get("evolution")
    if mode not in {"supported", "free_fall", "rest"}:
        raise ValueError("unsupported event mechanism")
    delay = entry.get("release_interval_s")
    if mode == "free_fall":
        if (
            not isinstance(delay, list)
            or len(delay) != 2
            or any((isinstance(v, bool) or not isinstance(v, (int, float)) for v in delay))
        ):
            raise ValueError("release interval must have two numeric endpoints")
        delay = np.array(delay, dtype=float)
        if not np.all(np.isfinite(delay)) or not 0 <= delay[0] <= delay[1] <= horizon:
            raise ValueError("release interval outside forecast support")
    elif delay is not None:
        raise ValueError("non-release mechanism must have null release interval")
    if mode == "supported" and state != "secure":
        raise ValueError("uncertain or failing support cannot declare no active hazard")
    if mode == "rest" and state != "grounded":
        raise ValueError("rest mechanism requires grounded observation")
    return dict(entry)


def compile_event(event, detection, velocity, origin, horizon=8.0):
    center = np.array(detection.center, dtype=float)
    velocity = np.array(velocity, dtype=float)
    if (
        center.shape != (3,)
        or velocity.shape != (3,)
        or (not np.all(np.isfinite(np.r_[center, velocity])))
    ):
        raise ValueError("invalid measured initial state")
    offsets = np.linspace(0, horizon, round(horizon / 0.1) + 1)
    mode = event["evolution"]
    floor_center = 0.05
    if mode == "supported":
        if center[2] < 1.3:
            raise ValueError("supported-overhead hypothesis contradicts measured height")
    if mode in {"supported", "rest"}:
        if mode == "rest" and center[2] > 0.2:
            raise ValueError("rest hypothesis contradicts measured height")
        trajectories = np.repeat(center[None, None, :], len(offsets), axis=1)
    else:
        earliest, latest = event["release_interval_s"]
        release_times = np.unique(np.linspace(earliest, latest, 17))
        trajectories = []
        for release in release_times:
            released_height = max(floor_center, center[2] + velocity[2] * release)
            fall_time = (
                velocity[2]
                + np.sqrt(velocity[2] ** 2 + 2 * 9.81 * (released_height - floor_center))
            ) / 9.81
            impact = release + max(0.0, fall_time)
            moving_time = np.minimum(offsets, impact)
            points = center + moving_time[:, None] * velocity
            points[:, 2] -= 0.5 * 9.81 * np.maximum(moving_time - release, 0) ** 2
            points[:, 2] = np.maximum(points[:, 2], floor_center)
            points[0] = center
            trajectories.append(points)
        trajectories = np.asarray(trajectories)
    centers = trajectories[:, :, :2].mean(0)
    spread = np.linalg.norm(trajectories[:, :, :2] - centers, axis=2).max(0)
    forecast = Forecast(
        "ceiling_fixture", origin, origin + offsets, centers, detection.radius + spread
    )
    return (
        forecast,
        {
            "evolution": mode,
            "offsets_s": offsets.tolist(),
            "trajectories_xyz_m": trajectories.tolist(),
            "physical_radius_m": float(detection.radius),
            "hypothesis_spread_m": spread.tolist(),
            "gravity_m_s2": 9.81,
            "floor_center_m": floor_center,
            "post_impact_model": "inert debris remains at computed contact location",
            "claim": "finite sampled executable hypotheses; uncertainty handled by the forecast consumer",
        },
    )


def analysis_prompt(facts):
    return (
        "Examine the changing ceiling light and its suspension in this video. Infer its CURRENT physical condition and what happens NEXT. Distinguish a supported object, a failing attachment, free fall, and resting debris. Use actual measured motion below; it is already corrected for robot camera motion. Treat velocities below 0.02 m/s as RGB-D measurement jitter, not evidence of free fall; larger coherent changes are actual object motion. World coordinates are metres, z up, floor z=0. History rows are [relative_time_seconds, world_x, world_y, world_z]. The normal is the FIXTURE PANEL surface normal, not the ceiling normal. These are only observations up through the final frame: "
        + json.dumps(facts)
        + ". Return JSON with objects as a list containing id ceiling_fixture, attachment_state (secure/loose/detached/grounded/uncertain), motion_description explaining visible evidence and predicted mechanism, trajectory: exactly five [dt,dx,dy,dz] rows at dt=1,2,3,4,5 seconds. Displacements are TOTAL world displacements from the CURRENT center, not between samples. Match numerical motion to the mechanism; stop descent when the physical object contacts the floor. No control actions."
    )


def refinement_prompt(facts, prior):
    return (
        "Construct an executable physical forecast from this video and causal observations. Coordinates are FIXED WORLD XYZ in metres; Z is height above floor z=0, NOT camera depth. Every history row is [relative_time_seconds, world_x, world_y, world_z]. Surface normal means the FIXTURE PANEL normal, not the ceiling. Evolution describes a possible FUTURE mechanism: a currently loose suspended object may enter free_fall later. A first physical analysis of this SAME observed video prefix returned: "
        + json.dumps(prior)
        + ". Actual measurements through the final frame: "
        + json.dumps(facts)
        + ". Infer the next physical mechanism. Return one JSON object with exactly id (ceiling_fixture), attachment_state (secure/loose/detached/grounded/uncertain), evolution (supported/free_fall/rest), release_interval_s (earliest and latest plausible loss of support in seconds after the final frame, within [0,8], or null for supported/rest), motion_description (evidence and physical consequence). Supported means continued secure attachment. Free_fall represents a plausible falling hazard following loss of the remaining support, not just extrapolation of the slow current swing. Rest means stationary floor debris. Assess whether the visibly inferred condition permits such a falling hazard, even when continued support is another possible outcome. Estimate a release interval only from the observed condition; no hidden failure schedule is known. A physics engine will integrate measured initial velocity, gravity and ground contact for the selected event. Return only the compact JSON object, without a thinking block. Do not provide robot actions."
    )
