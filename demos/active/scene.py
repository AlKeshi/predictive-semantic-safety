from itertools import product

import mujoco
import numpy as np


def scene_inventory(model, data, *, robot_body="trunk", floor_geom="floor"):
    robot = model.body(robot_body).id
    floor = model.geom(floor_geom).id
    roots = model.body_rootid[model.geom_bodyid]
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    if (
        model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
        or model.body_weldid[model.geom_bodyid[floor]] != 0
        or (not np.allclose(data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1]))
    ):
        raise ValueError("The floor role must be a fixed horizontal plane")
    objects, geom_names = ([], {})
    for body in sorted(set(model.body_rootid) - {0, robot}):
        if model.body_weldid[body] == 0:
            continue
        name = model.body(body).name
        geoms = np.flatnonzero((roots == body) & collides)
        joints = np.flatnonzero(model.body_rootid[model.jnt_bodyid] == body)
        if (
            not name
            or len(geoms) != 1
            or len(joints) != 1
            or (model.jnt_type[joints[0]] != mujoco.mjtJoint.mjJNT_FREE)
        ):
            raise ValueError(f"Hazard {name!r} requires one free body and one collision primitive")
        g = int(geoms[0])
        if model.geom_bodyid[g] != body or not np.allclose(model.geom_pos[g], 0):
            raise ValueError(f"Hazard {name!r} requires a body-centered primitive")
        kind, size = (model.geom_type[g], model.geom_size[g])
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            shape, radius, dimensions = ("sphere", size[0], [2 * size[0]] * 3)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape, radius = ("cylinder", float(np.hypot(*size[:2])))
            dimensions = [2 * size[0], 2 * size[0], 2 * size[1]]
        else:
            raise ValueError(f"Unsupported visual grounding primitive for {name!r}")
        material = model.geom_matid[g]
        color = model.geom_rgba[g] if material < 0 else model.mat_rgba[material]
        if color[3] < 1:
            raise ValueError(f"Hazard {name!r} must be opaque for color/depth grounding")
        objects.append(
            dict(
                id=name,
                shape=shape,
                color=color[:3].tolist(),
                radius=float(radius),
                surface_offset=float(size[0]),
                size=dimensions,
            )
        )
        geom_names[g] = name
    if not objects:
        raise ValueError("Active avoidance requires at least one declared hazard")
    static = []
    fixed = model.body_weldid[model.geom_bodyid] == 0
    covered = set(geom_names) | set(np.flatnonzero(fixed | (roots == robot) | ~collides))
    if covered != set(range(model.ngeom)):
        raise ValueError("Unsupported articulated or moving scene geometry")
    for g in np.flatnonzero(fixed & collides):
        if g == floor:
            continue
        kind, size = (model.geom_type[g], model.geom_size[g])
        if kind == mujoco.mjtGeom.mjGEOM_PLANE or model.body_mocapid[model.geom_bodyid[g]] >= 0:
            raise ValueError("Unbounded planes and mocap obstacles need a dedicated adapter")
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            vertices = np.array(list(product((-1, 1), repeat=3))) * size
            rotated = vertices @ data.geom_xmat[g].reshape(3, 3).T
            radius = float(np.linalg.norm(rotated[:, :2], axis=1).max())
        else:
            radius = float(model.geom_rbound[g])
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Static obstacle has no finite bounding radius")
        static.append(
            dict(
                object_id=model.geom(g).name or f"geom_{g}",
                center=data.geom_xpos[g, :2].tolist(),
                radius=radius,
            )
        )
    return (objects, geom_names, static)
