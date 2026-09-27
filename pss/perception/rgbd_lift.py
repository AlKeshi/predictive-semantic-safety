from __future__ import annotations

import numpy as np


def pixel_to_camera(
    u_px: float, v_px: float, depth_m: float, cam_intrinsics: np.ndarray
) -> np.ndarray:
    depth_value = float(depth_m)
    if not np.isfinite(depth_value) or depth_value <= 0.0:
        raise ValueError("depth_m must be positive and finite")
    intrinsics = np.asarray(cam_intrinsics, dtype=float)
    if intrinsics.shape != (3, 3):
        raise ValueError("cam_intrinsics must have shape (3, 3)")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0 or (not np.all(np.isfinite([fx, fy, cx, cy]))):
        raise ValueError("cam_intrinsics must contain finite positive focal lengths")
    return np.array(
        [(float(u_px) - cx) * depth_value / fx, (float(v_px) - cy) * depth_value / fy, depth_value],
        dtype=float,
    )


def camera_to_world(point_cam: np.ndarray, cam_pose: np.ndarray) -> np.ndarray:
    point = np.asarray(point_cam, dtype=float).reshape(3)
    pose = np.asarray(cam_pose, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("cam_pose must have shape (4, 4) and be finite")
    homogeneous = np.r_[point, 1.0]
    return (pose @ homogeneous)[:3]


def pixel_to_world(
    u_px: float, v_px: float, depth_m: float, cam_intrinsics: np.ndarray, cam_pose: np.ndarray
) -> np.ndarray:
    return camera_to_world(pixel_to_camera(u_px, v_px, depth_m, cam_intrinsics), cam_pose)
