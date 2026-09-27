import json
import math
from copy import deepcopy
from pathlib import Path
from types import MethodType

import numpy as np

from pss.data.episode import rot_to_quat
from pss.perception.code_world import detect_objects


def install_sensor(world, out, specifications_for):
    original = world.capture_forward_egocentric

    def capture(world):
        now = float(world.data.time)
        if not hasattr(world, "_head_state"):
            world._head_state = {
                "yaw": 0.0,
                "pitch": -math.radians(
                    float(world.scenario.meta["omnivla_camera_mount"]["pitch_down_deg"])
                ),
                "time": now,
                "points": {},
                "records": [],
            }
        head = world._head_state
        dt = max(0.0, now - head["time"])
        cid = world.omnivla_camera_id
        parent = int(world.model.cam_bodyid[cid])
        rotation = world.data.xmat[parent].reshape(3, 3)
        cam_position = world.data.cam_xpos[cid].copy()
        desired = np.array([head["yaw"], head["pitch"]])
        span = None
        if head["points"]:
            targets = np.array([v[1] for v in head["points"].values()])
            directions = (targets - cam_position) @ rotation
            azimuth = np.arctan2(directions[:, 1], directions[:, 0]) % (2 * np.pi)
            elevation = np.arctan2(directions[:, 2], np.linalg.norm(directions[:, :2], axis=1))
            ordered = np.sort(azimuth)
            gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
            i = int(np.argmax(gaps))
            span = 2 * np.pi - gaps[i]
            yaw = (ordered[(i + 1) % len(ordered)] + span / 2 + np.pi) % (2 * np.pi) - np.pi
            desired = np.clip(
                [yaw, (elevation.min() + elevation.max()) / 2],
                np.deg2rad([-70, -35]),
                np.deg2rad([70, 35]),
            )
        current = np.array([head["yaw"], head["pitch"]])
        current += np.clip(desired - current, -np.array([1.8, 1.2]) * dt, np.array([1.8, 1.2]) * dt)
        yaw, pitch = current
        forward = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        back = -forward
        up = np.cross(back, right)
        world.model.cam_quat[cid] = rot_to_quat(np.column_stack([right, up, back]))
        world.mj.mj_camlight(world.model, world.data)
        ob = original()
        specifications = specifications_for(world.scenario)
        detections = detect_objects(ob, specifications)
        world._current_sensor_detections = (ob, deepcopy(specifications), detections)
        for name, value in detections.items():
            head["points"][name] = (now, value.center.copy())
        head.update(yaw=float(yaw), pitch=float(pitch), time=now)
        head["records"].append(
            {
                "time": now,
                "yaw_rad": float(yaw),
                "pitch_rad": float(pitch),
                "desired": desired.tolist(),
                "target_span_rad": None if span is None else float(span),
                "visible_ids": sorted(detections),
                "camera_pose": ob.cam_pose.tolist(),
                "source": "current_and_past_actual_RGBD_only",
                "mechanism": "ideal_pan_tilt_gimbal_yaw70_pitch35_deg_rate1p8_1p2_rad_s",
                "fov_changed": False,
            }
        )
        return ob

    world.capture_forward_egocentric = MethodType(capture, world)
    oldclose = world.close

    def close(world):
        if hasattr(world, "_head_state"):
            Path(out, "sensor_head_receipts.json").write_text(
                json.dumps(world._head_state["records"], indent=2)
            )
        return oldclose()

    world.close = MethodType(close, world)


def detections_for_observation(world, observation, specifications):
    cached = getattr(world, "_current_sensor_detections", None)
    if cached is not None and cached[0] is observation and (cached[1] == specifications):
        return deepcopy(cached[2])
    return detect_objects(observation, specifications)
