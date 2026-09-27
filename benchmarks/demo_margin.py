from dataclasses import replace

import numpy as np


def pad_forecasts(forecasts, margin):
    margin = float(margin)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Demo margin must be finite and nonnegative")
    return [replace(f, radii=np.array(f.radii, dtype=float, copy=True) + margin) for f in forecasts]
