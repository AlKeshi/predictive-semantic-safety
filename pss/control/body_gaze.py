from __future__ import annotations

import numpy as np


def wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class BodyGazeController:
    def __init__(self):
        self.observed = {}
        self.target = None
        self.observation_time = None
        self.visible_ids = []

    def observe(self, detections, now):
        if not np.isfinite(now):
            raise ValueError("Observation timestamp must be finite")
        if self.observation_time is not None and now < self.observation_time:
            raise ValueError("Observation time cannot reverse")
        self.observation_time = float(now)
        self.visible_ids = []
        for name, detection in detections.items():
            point = np.asarray(detection.center[:2], dtype=float)
            if point.shape == (2,) and np.all(np.isfinite(point)):
                self.observed[name] = (float(now), point.copy())
                self.visible_ids.append(name)

    def command(self, xy, yaw, now, snapshot):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)) or (not np.all(np.isfinite([yaw, now]))):
            raise ValueError("Gaze requires finite measured position, yaw, and time")
        points = {
            name: (point, "last_observed_search", now - stamp)
            for name, (stamp, point) in self.observed.items()
        }
        if snapshot.get("emergency_reason") is None:
            for item in snapshot.get("forecasts", []):
                times = np.asarray(item["times"], dtype=float)
                centers = np.asarray(item["centers"], dtype=float)
                if times[0] <= now < times[-1] and item["object_id"] in self.observed:
                    point = np.array([np.interp(now, times, centers[:, axis]) for axis in range(2)])
                    points[item["object_id"]] = (
                        point,
                        "supported_committed_prediction",
                        now - float(item["origin_time"]),
                    )
        for name, (stamp, point) in self.observed.items():
            if now - stamp <= 0.075:
                points[name] = (
                    point,
                    "latest_rgbd" if name in self.visible_ids else "recent_rgbd",
                    now - stamp,
                )
        bearings = []
        targets = []
        for name, (point, source, age) in sorted(points.items()):
            delta = point - np.asarray(xy)
            if np.linalg.norm(delta) > 0.05:
                bearing = float(np.arctan2(delta[1], delta[0]))
                bearings.append(bearing % (2 * np.pi))
                targets.append(
                    dict(
                        object_id=name,
                        source=source,
                        age_s=age,
                        bearing_rad=bearing,
                        position_xy=point.tolist(),
                    )
                )
        span = None
        if bearings:
            ordered = np.sort(bearings)
            gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
            choices = np.flatnonzero(gaps >= gaps.max() - 0.1)
            candidates = [
                (
                    wrap(ordered[(index + 1) % len(ordered)] + (2 * np.pi - gaps[index]) / 2),
                    2 * np.pi - gaps[index],
                )
                for index in choices
            ]
            reference = yaw if self.target is None else self.target
            desired, span = min(candidates, key=lambda item: abs(wrap(item[0] - reference)))
        else:
            desired = yaw
        if self.target is None or abs(wrap(desired - self.target)) > np.deg2rad(8.0):
            self.target = desired
        error = wrap(self.target - yaw)
        rate = float(np.clip(1.6 * error, -0.8, 0.8)) if abs(error) > np.deg2rad(3.0) else 0.0
        return (
            rate,
            dict(
                target_yaw_rad=self.target,
                measured_yaw_rad=float(yaw),
                yaw_rate_command_rad_s=rate,
                target_span_rad=span,
                targets=targets,
                sensor="single_forward_perspective",
                scope="gaze_only_no_translation_override",
                latest_sensor_time_s=self.observation_time,
                visible_object_ids=sorted(self.visible_ids),
            ),
        )
