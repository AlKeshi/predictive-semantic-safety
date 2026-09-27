from __future__ import annotations

import math

import numpy as np

from pss.scenarios.spec import DynamicObstacleSpec, ScenarioSpec
from pss.sim.mujoco.go1_types import Go1WorldConfig

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_FOVY_DEG = 85.0
CAMERA_FOVX_DEG = math.degrees(
    2 * math.atan(math.tan(math.radians(CAMERA_FOVY_DEG / 2)) * CAMERA_WIDTH / CAMERA_HEIGHT)
)


def object_radius(obj: DynamicObstacleSpec) -> float:
    if obj.shape == "sphere":
        return float(obj.radius)
    if obj.size is None:
        raise ValueError(f"Object {obj.name!r} needs a size")
    if obj.shape in ("cylinder", "cylinder_side"):
        return float(math.hypot(obj.size[0] / 2.0, obj.size[2] / 2.0))
    return float(np.linalg.norm(np.asarray(obj.size, dtype=float) / 2.0))


def hazard_metadata(scenario: ScenarioSpec) -> dict[str, dict]:
    return {
        obj.name: {
            "object_id": obj.name,
            "shape": obj.shape,
            "color_rgb": [round(255.0 * channel) for channel in obj.color],
            "radius_m": object_radius(obj),
            "geometry_source": "declared_simulation_fixture",
        }
        for obj in scenario.dynamic_obstacles
    }


def configure_native_contacts(world) -> dict[str, dict]:
    settings = world.scenario.meta.get("native_contact", {})
    for object_id, contact in settings.items():
        geom_id = world.model.geom(f"dynamic_obstacle_geom_{object_id}").id
        world.model.geom_priority[geom_id] = int(contact["priority"])
        world.model.geom_solref[geom_id] = np.asarray(contact["solref"], dtype=float)
    return settings


def world_config(duration: float, *, render: bool = False, **overrides) -> Go1WorldConfig:
    settings = {
        "duration": float(duration),
        "fps": 20,
        "width": 1280 if render else 640,
        "height": 720 if render else 360,
        "rgbd_width": CAMERA_WIDTH,
        "rgbd_height": CAMERA_HEIGHT,
        "rgbd_fovy_deg": CAMERA_FOVY_DEG,
        "stack_settle_time": 0.0,
        "reject_nonquiescent_stack": False,
        "full_body_collisions": True,
    }
    settings.update(overrides)
    return Go1WorldConfig(**settings)
