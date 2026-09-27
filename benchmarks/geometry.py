import itertools

import mujoco
import numpy as np


def update_geometry_measurement(model, physical_data, measured_data):
    measured_data.qpos[:] = physical_data.qpos
    measured_data.mocap_pos[:] = physical_data.mocap_pos
    measured_data.mocap_quat[:] = physical_data.mocap_quat
    measured_data.time = physical_data.time
    mujoco.mj_kinematics(model, measured_data)


def geom_bounds(model, data, gid):
    center = data.geom_xpos[gid].copy()
    typ = int(model.geom_type[gid])
    size = model.geom_size[gid]
    if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0])
        return (center, radius, center[2] - radius, None)
    if typ == int(mujoco.mjtGeom.mjGEOM_BOX):
        corners = np.array(list(itertools.product([-1.0, 1.0], repeat=3))) * size
        corners = corners @ data.geom_xmat[gid].reshape(3, 3).T
        radius = float(np.linalg.norm(corners[:, :2], axis=1).max())
        return (center, radius, float(center[2] + corners[:, 2].min()), corners + center)
    radius = float(model.geom_rbound[gid])
    return (center, radius, center[2] - radius, None)
