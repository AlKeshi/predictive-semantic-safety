import numpy as np

from pss.data.episode import rot_to_quat
from pss.scenarios.spec import DynamicObstacleSpec, ObjectSpec, ScenePropSpec, object_rotation


def _table_xml(table, suffix: str = "") -> str:
    hx, hy, hz = np.asarray(table.size, dtype=float) / 2.0
    tx, ty = table.position
    return f'\n    <body name="table{suffix}" pos="{tx:.8g} {ty:.8g} 0">\n      <geom name="table_pedestal{suffix}" type="box" size="0.25 0.25 {max(hz - 0.05, 0.05):.8g}"\n            pos="0 0 {max(hz - 0.05, 0.05):.8g}" material="pss_table_base"\n            friction="{table.friction:.8g} 0.01 0.001"/>\n      <geom name="table_top{suffix}" type="box" size="{hx:.8g} {hy:.8g} 0.05"\n            pos="0 0 {table.size[2] - 0.05:.8g}" material="pss_table_top"\n            friction="{table.friction:.8g} 0.01 0.001"/>\n    </body>'


def _stack_object_xml(obj: ObjectSpec) -> str:
    half = np.asarray(obj.size, dtype=float) / 2.0
    px, py, pz = obj.position
    quat = " ".join((f"{v:.8g}" for v in rot_to_quat(object_rotation(obj))))
    rgba = " ".join((f"{v:.6g}" for v in (*obj.color, 1.0)))
    if obj.shape == "cuboid":
        geom = f'type="box" size="{half[0]:.8g} {half[1]:.8g} {half[2]:.8g}"'
        volume = float(np.prod(obj.size))
    elif obj.shape in ("cylinder", "cylinder_side"):
        geom = f'type="cylinder" size="{half[0]:.8g} {half[2]:.8g}"'
        volume = float(np.pi * half[0] ** 2 * obj.size[2])
    elif obj.shape == "sphere":
        geom = f'type="sphere" size="{half[0]:.8g}"'
        volume = float(4.0 / 3.0 * np.pi * half[0] ** 3)
    else:
        raise ValueError(f"Unsupported Go1-world stack shape '{obj.shape}'")
    mass = max(float(obj.density * volume), 0.0001)
    inertia = _object_diagonal_inertia(obj, mass)
    com = " ".join((f"{float(v):.8g}" for v in obj.com_offset))
    inertia_text = " ".join((f"{float(v):.8g}" for v in inertia))
    return f'\n    <body name="stack_{obj.name}" pos="{px:.8g} {py:.8g} {pz:.8g}" quat="{quat}">\n      <freejoint/>\n      <inertial pos="{com}" mass="{mass:.8g}" diaginertia="{inertia_text}"/>\n      <geom name="stack_geom_{obj.name}" {geom} mass="{mass:.8g}"\n            rgba="{rgba}" friction="{obj.friction:.8g} 0.01 0.001"\n            contype="1" conaffinity="1"/>\n    </body>'


def _dynamic_obstacle_xml(obstacle: DynamicObstacleSpec) -> str:
    px, py, pz = obstacle.position
    geom, volume, inertia_scale = _dynamic_obstacle_geom(obstacle)
    mass = max(float(obstacle.density) * volume, 0.0001)
    inertia = inertia_scale * mass
    quat = " ".join((f"{v:.8g}" for v in _euler_xyz_quat(obstacle.euler_xyz)))
    rgba = " ".join((f"{v:.6g}" for v in (*obstacle.color, 1.0)))
    damping_ratio = float(np.clip(1.0 - obstacle.restitution, 0.05, 1.0))
    inertia_text = " ".join((f"{float(v):.8g}" for v in inertia))
    return f'\n    <body name="dynamic_obstacle_{obstacle.name}" pos="{px:.8g} {py:.8g} {pz:.8g}" quat="{quat}">\n      <freejoint/>\n      <inertial pos="0 0 0" mass="{mass:.8g}" diaginertia="{inertia_text}"/>\n      <geom name="dynamic_obstacle_geom_{obstacle.name}" {geom}\n            rgba="{rgba}" friction="{obstacle.friction:.8g} 0.01 0.001"\n            solref="0.004 {damping_ratio:.3g}" solimp="0.95 0.99 0.001"\n            contype="1" conaffinity="1"/>\n    </body>'


def _dynamic_obstacle_geom(obstacle: DynamicObstacleSpec) -> tuple[str, float, np.ndarray]:
    if obstacle.shape == "sphere":
        if obstacle.radius <= 0.0:
            raise ValueError("Dynamic sphere obstacle radius must be positive")
        radius = float(obstacle.radius)
        volume = 4.0 / 3.0 * np.pi * radius**3
        inertia_scale = np.full(3, 0.4 * radius**2)
        return (f'type="sphere" size="{radius:.8g}"', volume, inertia_scale)
    if obstacle.shape == "cuboid":
        if obstacle.size is None or min(obstacle.size) <= 0.0:
            raise ValueError("Dynamic cuboid obstacle size entries must be positive")
        sx, sy, sz = np.asarray(obstacle.size, dtype=float)
        half = np.array([sx, sy, sz], dtype=float) / 2.0
        volume = float(sx * sy * sz)
        inertia_scale = (
            np.array([sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy], dtype=float) / 12.0
        )
        return (
            f'type="box" size="{half[0]:.8g} {half[1]:.8g} {half[2]:.8g}"',
            volume,
            inertia_scale,
        )
    if obstacle.shape in ("cylinder", "cylinder_side"):
        if obstacle.size is None or min(obstacle.size) <= 0.0:
            raise ValueError("Dynamic cylinder obstacle size entries must be positive")
        diameter, _, length = np.asarray(obstacle.size, dtype=float)
        radius = float(diameter / 2.0)
        half_length = float(length / 2.0)
        volume = float(np.pi * radius**2 * length)
        transverse = (3.0 * radius * radius + length * length) / 12.0
        axial = 0.5 * radius * radius
        inertia_scale = np.array([transverse, transverse, axial], dtype=float)
        return (f'type="cylinder" size="{radius:.8g} {half_length:.8g}"', volume, inertia_scale)
    raise ValueError(f"Unsupported dynamic obstacle shape '{obstacle.shape}'")


def _scene_prop_xml(prop: ScenePropSpec) -> str:
    if min(prop.size) <= 0.0:
        raise ValueError("Scene prop size entries must be positive")
    px, py, pz = prop.position
    quat = " ".join((f"{v:.8g}" for v in _euler_xyz_quat(prop.euler_xyz)))
    rgba = " ".join((f"{v:.6g}" for v in (*prop.color, 1.0)))
    visual = f'material="{prop.material}"' if prop.material else f'rgba="{rgba}"'
    contype = 1 if prop.collidable else 0
    conaffinity = 1 if prop.collidable else 0
    if prop.shape != "cuboid":
        raise ValueError(f"Unsupported scene prop shape '{prop.shape}'")
    hx, hy, hz = np.asarray(prop.size, dtype=float) / 2.0
    return f'\n    <body name="scene_prop_{prop.name}" pos="{px:.8g} {py:.8g} {pz:.8g}" quat="{quat}">\n      <geom name="scene_prop_geom_{prop.name}" type="box" size="{hx:.8g} {hy:.8g} {hz:.8g}"\n            {visual} friction="{prop.friction:.8g} 0.01 0.001"\n            contype="{contype}" conaffinity="{conaffinity}"/>\n    </body>'


def _euler_xyz_quat(euler_xyz: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler_xyz, dtype=float)
    cr, sr = (np.cos(roll), np.sin(roll))
    cp, sp = (np.cos(pitch), np.sin(pitch))
    cy, sy = (np.cos(yaw), np.sin(yaw))
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot_to_quat(rot_z @ rot_y @ rot_x)


def _object_diagonal_inertia(obj: ObjectSpec, mass: float) -> np.ndarray:
    sx, sy, sz = np.asarray(obj.size, dtype=float)
    if obj.shape == "sphere":
        value = 0.4 * mass * (sx / 2.0) ** 2
        return np.full(3, value)
    if obj.shape in ("cylinder", "cylinder_side"):
        radius = sx / 2.0
        transverse = mass * (3.0 * radius * radius + sz * sz) / 12.0
        axial = 0.5 * mass * radius * radius
        return np.array([transverse, transverse, axial])
    return mass / 12.0 * np.array([sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy])
