from __future__ import annotations

import numpy as np


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
