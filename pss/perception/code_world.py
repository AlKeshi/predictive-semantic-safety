from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

MAX_FUTURE_CAMERA_RANGE_M = 20.0
MAX_FORECAST_SPEED_M_S = 8.0


@dataclass
class Detection:
    object_id: str
    pixel: np.ndarray
    center: np.ndarray
    depth: float
    radius: float
    pixels: int


def _surface_center(observation, xx, yy, spec):
    intrinsics = np.asarray(observation.cam_intrinsics, dtype=float)
    depth_image = np.asarray(observation.depth)
    pixel = np.array([xx.mean(), yy.mean()])
    order = np.argsort((xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2)
    depth = float(np.median(depth_image[yy[order[:9]], xx[order[:9]]]))
    ray = np.linalg.solve(intrinsics, np.r_[pixel, 1.0])
    camera_center = depth * ray
    offset = float(spec.get("surface_offset", spec["radius"]))
    camera_center += offset * ray / np.linalg.norm(ray)
    if spec.get("shape") == "sphere" and len(xx) >= 12:
        take = np.linspace(0, len(xx) - 1, min(512, len(xx)), dtype=int)
        z = depth_image[yy[take], xx[take]]
        uv = np.c_[xx[take], yy[take], np.ones(len(take))]
        points = uv @ np.linalg.inv(intrinsics).T * z[:, None]
        anchor = points.mean(axis=0)
        centered = points - anchor
        center_offset, _, rank, singular = np.linalg.lstsq(
            2 * centered, np.sum(centered * centered, axis=1), rcond=None
        )
        candidate = anchor + center_offset
        radius = float(spec["radius"])
        residual = np.abs(np.linalg.norm(points - candidate, axis=1) - radius)
        if (
            rank == 3
            and singular[-1] > 0.0001 * singular[0]
            and (candidate[2] > 0)
            and (np.linalg.norm(center_offset) <= 1.5 * radius)
            and (np.quantile(residual, 0.95) <= 0.005 + 0.05 * radius)
        ):
            camera_center = candidate
    center = (observation.cam_pose @ np.r_[camera_center, 1.0])[:3]
    if not np.all(np.isfinite(center)) or camera_center[2] <= 0:
        raise ValueError("Unusable measured depth surface")
    projected = intrinsics @ camera_center
    return (projected[:2] / projected[2], center, float(camera_center[2]))


def detect_objects(observation, objects):
    rgb = np.asarray(observation.rgb, dtype=float) / 255.0
    r, g, b = (rgb[..., 0], rgb[..., 1], rgb[..., 2])
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    delta = high - low
    saturation = delta / np.maximum(high, 1e-08)
    hue = np.zeros_like(high)
    red = (rgb[..., 0] == high) & (delta > 1e-08)
    green = (rgb[..., 1] == high) & ~red & (delta > 1e-08)
    blue = (delta > 1e-08) & ~red & ~green
    hue[red] = (rgb[..., 1] - rgb[..., 2])[red] / delta[red] % 6
    hue[green] = (rgb[..., 2] - rgb[..., 0])[green] / delta[green] + 2
    hue[blue] = (rgb[..., 0] - rgb[..., 1])[blue] / delta[blue] + 4
    hue /= 6
    targets = [colorsys.rgb_to_hsv(*spec["color"]) for spec in objects]
    distances = []
    for target_h, target_s, _ in targets:
        difference = np.abs(hue - target_h)
        distances.append(
            np.minimum(difference, 1 - difference)
            if target_s >= 0.18
            else np.full_like(hue, np.inf)
        )
    nearest = np.argmin(distances, axis=0) if distances else np.zeros_like(hue, dtype=int)
    valid_depth = (
        np.isfinite(observation.depth) & (observation.depth > 0) & (observation.depth < 9.5)
    )
    detections = {}
    for object_index, (spec, target) in enumerate(zip(objects, targets, strict=True)):
        name = spec["id"]
        if target[1] < 0.18:
            mask = (saturation < 0.18) & (high > 0.7)
        else:
            mask = (
                (distances[object_index] < 0.065)
                & (nearest == object_index)
                & (saturation > 0.35)
                & (high > 0.17)
            )
        mask &= valid_depth
        labels, _ = ndimage.label(mask)
        best = None
        for index, slices in enumerate(ndimage.find_objects(labels), 1):
            if slices is None:
                continue
            yy, xx = np.where(labels[slices] == index)
            yy = yy + slices[0].start
            xx = xx + slices[1].start
            size = len(xx)
            if size < 5:
                continue
            if best is None or size > best[0]:
                best = (size, xx, yy)
        if best is None:
            continue
        size, xx, yy = best
        try:
            pixel, center, depth = _surface_center(observation, xx, yy, spec)
            detections[name] = Detection(name, pixel, center, depth, float(spec["radius"]), size)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return detections
