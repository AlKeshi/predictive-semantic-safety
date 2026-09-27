from __future__ import annotations

import numpy as np
from scipy import ndimage

from pss.perception.code_world import Detection

OBJECTS = [
    {
        "id": "ceiling_fixture",
        "shape": "rectangular ceiling luminaire",
        "size": [1.18, 0.46, 0.1],
        "color": [0.68, 0.43, 0.15],
        "radius": 0.636,
    }
]


def observe(rgb, depth, intrinsics, pose, previous=None):
    rgb = np.asarray(rgb, dtype=float)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 9.5)
    r, g, b = np.moveaxis(rgb, -1, 0)
    warm = valid & (r > 1.025 * g) & (g > 1.1 * b) & (r > 55)
    warm = ndimage.binary_closing(warm, iterations=1)
    labels, count = ndimage.label(warm)
    inverse = np.linalg.inv(intrinsics)
    candidates = []
    for index, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        mask = ndimage.binary_fill_holes(labels[slices] == index)
        yy, xx = np.where(mask & valid[slices])
        yy = yy + slices[0].start
        xx = xx + slices[1].start
        if len(xx) < 25 or np.ptp(xx) < 5 or np.ptp(yy) < 3:
            continue
        z = depth[yy, xx]
        points_cam = np.c_[xx, yy, np.ones(len(xx))] @ inverse.T * z[:, None]
        points = points_cam @ pose[:3, :3].T + pose[:3, 3]
        anchor = np.median(points, axis=0)
        if previous is None:
            if anchor[2] < 1.3:
                continue
        elif np.linalg.norm(anchor - previous) > 1.35:
            continue
        centered = points - anchor
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        local = centered @ vt.T
        lo, hi = np.quantile(local, [0.01, 0.99], axis=0)
        extent = hi - lo
        if extent[0] < 0.2 or extent[0] > 1.35 or extent[1] > 0.65 or (extent[2] > 0.17):
            continue
        center = anchor + (lo + hi) / 2 @ vt
        normal = vt[2].copy()
        if normal @ (pose[:3, 3] - center) < 0:
            normal *= -1
        center -= normal * 0.042
        cam_center = pose[:3, :3].T @ (center - pose[:3, 3])
        if cam_center[2] <= 0:
            continue
        uv = intrinsics @ cam_center
        uv = uv[:2] / uv[2]
        score = len(xx) if previous is None else -np.linalg.norm(center - previous)
        candidates.append(
            (
                score,
                Detection(
                    "ceiling_fixture",
                    uv,
                    center,
                    float(cam_center[2]),
                    OBJECTS[0]["radius"],
                    len(xx),
                ),
                {
                    "extent": extent.tolist(),
                    "normal": normal.tolist(),
                    "pixel_bounds": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                },
            )
        )
    if not candidates:
        return (None, {"reason": "no_usable_rgbd_fixture_surface"})
    _, detection, diagnostic = max(candidates, key=lambda item: item[0])
    return (detection, diagnostic)
