import numpy as np

from pss.control.forecast import Forecast

from .navigation import waypoint as swept_waypoint


def waypoint(position, goal, forecasts, static, robot_radius=0.403):
    snapshots = [
        Forecast(
            f.object_id,
            f.origin_time,
            np.asarray(f.times),
            np.repeat(np.asarray(f.centers)[:1], len(f.times), axis=0),
            float(np.broadcast_to(np.asarray(f.radii), np.asarray(f.times).shape)[0]),
        )
        for f in forecasts
    ]
    target, receipt = swept_waypoint(position, goal, snapshots, static, robot_radius)
    receipt["occupancy"] = "current snapshot; dynamic CBF uses full forecast"
    return (target, receipt)
