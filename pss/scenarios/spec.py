from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SHAPE_CUBOID = "cuboid"
SHAPE_CYLINDER = "cylinder"
SHAPE_CYLINDER_SIDE = "cylinder_side"
SHAPE_SPHERE = "sphere"
SHAPES = (SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_CYLINDER_SIDE, SHAPE_SPHERE)


@dataclass
class ObjectSpec:
    name: str
    shape: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    yaw: float = 0.0
    density: float = 600.0
    friction: float = 0.6
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TableSpec:
    size: tuple[float, float, float] = (0.9, 0.9, 0.75)
    position: tuple[float, float] = (0.0, 0.0)
    friction: float = 0.8


@dataclass
class DisturbanceSpec:
    time: float = 1.0
    target_object: str = ""
    impulse: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DynamicObstacleSpec:
    name: str
    position: tuple[float, float, float]
    radius: float = 0.0
    shape: str = SHAPE_SPHERE
    size: tuple[float, float, float] | None = None
    euler_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    activation_distance: float | None = None
    activation_center: tuple[float, float] | None = None
    density: float = 450.0
    friction: float = 0.4
    restitution: float = 0.85
    color: tuple[float, float, float] = (0.9, 0.12, 0.08)


@dataclass
class ScenePropSpec:
    name: str
    shape: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    euler_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: float = 0.6
    color: tuple[float, float, float] = (0.75, 0.64, 0.5)
    material: str | None = None
    collidable: bool = True


@dataclass
class ScenarioSpec:
    scenario_id: str
    seed: int
    table: TableSpec
    objects: list[ObjectSpec]
    vcom: list[int] = field(default_factory=list)
    vpsf: list[int] = field(default_factory=list)
    disturbance: DisturbanceSpec | None = None
    robot_start: tuple[float, float] = (-2.5, -1.6)
    robot_goal: tuple[float, float] = (2.5, -1.6)
    horizon: float = 4.0
    dynamic_obstacles: list[DynamicObstacleSpec] = field(default_factory=list)
    scene_props: list[ScenePropSpec] = field(default_factory=list)
    extra_tables: list[TableSpec] = field(default_factory=list)
    extra_disturbances: list[DisturbanceSpec] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_stable_by_construction(self) -> bool:
        return not self.vcom and (not self.vpsf)

    @property
    def tables(self) -> list[TableSpec]:
        return [self.table, *self.extra_tables]

    @property
    def stack_height(self) -> int:
        return len(self.objects)

    def stack_top_z(self) -> float:
        return max((o.position[2] + o.size[2] / 2.0 for o in self.objects))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ScenarioSpec:
        d = dict(d)
        d["table"] = TableSpec(**_tupled(d["table"]))
        d["objects"] = [ObjectSpec(**_tupled(o)) for o in d["objects"]]
        d["dynamic_obstacles"] = [
            DynamicObstacleSpec(**_tupled(o)) for o in d.get("dynamic_obstacles", [])
        ]
        d["scene_props"] = [ScenePropSpec(**_tupled(o)) for o in d.get("scene_props", [])]
        d["extra_tables"] = [TableSpec(**_tupled(t)) for t in d.get("extra_tables", [])]
        if d.get("disturbance") is not None:
            d["disturbance"] = DisturbanceSpec(**_tupled(d["disturbance"]))
        d["extra_disturbances"] = [
            DisturbanceSpec(**_tupled(item)) for item in d.get("extra_disturbances", [])
        ]
        return ScenarioSpec(**d)


def _tupled(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: tuple(v) if isinstance(v, list) and all((isinstance(x, (int, float)) for x in v)) else v
        for k, v in d.items()
    }


def object_rotation(obj: ObjectSpec) -> np.ndarray:
    c, s = (np.cos(obj.yaw), np.sin(obj.yaw))
    yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if obj.shape == SHAPE_CYLINDER_SIDE:
        roll = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        return yaw @ roll
    return yaw
