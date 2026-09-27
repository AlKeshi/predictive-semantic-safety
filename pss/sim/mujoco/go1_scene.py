from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from pss.scenarios.spec import (
    ScenarioSpec,
)
from pss.sim.mujoco.go1_types import Go1WorldConfig

from .scene_geometry import _dynamic_obstacle_xml, _scene_prop_xml, _stack_object_xml, _table_xml

_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "go1"
_GO1_MESHES = ("trunk.stl", "hip.stl", "thigh_mirror.stl", "calf.stl", "thigh.stl")


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "The Go1 MuJoCo runtime needs MuJoCo. Install `pip install -e .`."
        ) from exc
    return mujoco


def _go1_assets(*, full_body_collisions: bool = False) -> dict[str, bytes]:
    required = [
        _ASSET_ROOT / "go1.xml",
        _ASSET_ROOT / "go1_policy.onnx",
        *(_ASSET_ROOT / "meshes" / name for name in _GO1_MESHES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing vendored Go1 assets: {missing}")
    robot_xml = (_ASSET_ROOT / "go1.xml").read_bytes()
    if full_body_collisions:
        tree = ET.fromstring(robot_xml)
        collision = tree.find(".//default[@class='collision']/geom")
        if collision is None:
            raise ValueError("The Go1 asset is missing its collision default")
        collision.set("contype", "0")
        collision.set("conaffinity", "1")
        robot_xml = ET.tostring(tree, encoding="utf-8")
    assets = {"go1.xml": robot_xml}
    for name in _GO1_MESHES:
        assets[f"meshes/{name}"] = (_ASSET_ROOT / "meshes" / name).read_bytes()
    return assets


def _go1_scene_xml(scenario: ScenarioSpec, cfg: Go1WorldConfig) -> str:
    scene_material_xml = _scene_material_xml(scenario)
    table_xml = "\n".join(
        (
            _table_xml(table, suffix="" if index == 0 else f"_{index}")
            for index, table in enumerate(scenario.tables)
        )
    )
    object_xml = "\n".join((_stack_object_xml(obj) for obj in scenario.objects))
    scene_prop_xml = "\n".join((_scene_prop_xml(prop) for prop in scenario.scene_props))
    dynamic_obstacle_xml = "\n".join(
        (_dynamic_obstacle_xml(obstacle) for obstacle in scenario.dynamic_obstacles)
    )
    object_xml = "\n".join(
        (chunk for chunk in (object_xml, scene_prop_xml, dynamic_obstacle_xml) if chunk)
    )
    gx, gy = np.asarray(scenario.robot_goal, dtype=float)
    goal_beacon_xml = (
        f'\n    <body name="omnivla_goal_beacon" pos="{gx:.8g} {gy:.8g} 0">\n      <geom name="omnivla_goal_beacon_post" type="cylinder" pos="0 0 0.55"\n            size="0.055 0.55" rgba="0.02 0.24 0.95 1"\n            contype="0" conaffinity="0"/>\n      <geom name="omnivla_goal_beacon_head" type="sphere" pos="0 0 1.18"\n            size="0.14" rgba="0.02 0.38 1 1"\n            contype="0" conaffinity="0"/>\n    </body>'
        if cfg.omnivla_goal_beacon
        else ""
    )
    return _gallery_visuals(
        f'\n<mujoco model="pss_go1_online">\n  <include file="go1.xml"/>\n  <option timestep="{cfg.sim_dt:.9g}" gravity="0 0 -9.81" integrator="Euler"\n          iterations="20" ls_iterations="10"/>\n  <visual>\n    <global offwidth="{max(cfg.width, cfg.top_width, cfg.rgbd_width)}"\n            offheight="{max(cfg.height, cfg.top_height, cfg.rgbd_height)}"/>\n    <quality shadowsize="4096" offsamples="4"/>\n    <map znear="0.01" zfar="30" fogstart="8" fogend="18"/>\n    <headlight diffuse="0.46 0.48 0.50" ambient="0.18 0.19 0.21"\n               specular="0.10 0.11 0.12"/>\n    <rgba haze="0.75 0.79 0.84 1"/>\n  </visual>\n  <asset>\n    <texture name="pss_sky" type="skybox" builtin="gradient"\n             rgb1="0.60 0.70 0.82" rgb2="0.96 0.97 0.98" width="512" height="3072"/>\n    {scene_material_xml}\n    <material name="pss_table_top" rgba="0.48 0.29 0.14 1" specular="0.30" shininess="0.22"/>\n    <material name="pss_table_base" rgba="0.22 0.25 0.29 1" specular="0.15"/>\n    <material name="pss_wall" rgba="0.84 0.86 0.88 1" roughness="0.9"/>\n  </asset>\n  <worldbody>\n    <light name="pss_key" pos="-3.2 -4.2 6.5" dir="0.35 0.45 -1" directional="true"\n           diffuse="0.58 0.55 0.52" specular="0.16 0.15 0.14" castshadow="true"/>\n    <light name="pss_fill" pos="3.2 -1.0 4.0" dir="-0.55 0.15 -1" directional="true"\n           diffuse="0.20 0.26 0.34" specular="0.08 0.10 0.13" castshadow="false"/>\n    <geom name="floor" type="plane" size="12 12 0.1" material="pss_floor"\n          friction="0.80 0.01 0.001" contype="1" conaffinity="0" priority="1"/>\n    <geom name="back_wall" type="box" size="4.7 0.04 1.35" pos="0 2.55 1.35"\n          material="pss_wall" contype="0" conaffinity="0"/>\n    <geom name="left_marker" type="box" size="0.02 1.7 0.002" pos="-2.35 -0.72 0.003"\n          rgba="0.36 0.39 0.43 0.35" contype="0" conaffinity="0"/>\n    <geom name="right_marker" type="box" size="0.02 1.7 0.002" pos="2.35 -0.72 0.003"\n          rgba="0.36 0.39 0.43 0.35" contype="0" conaffinity="0"/>\n    {goal_beacon_xml}\n    {table_xml}\n    {object_xml}\n  </worldbody>\n</mujoco>\n',
        scenario,
    )


def _scene_material_xml(scenario: ScenarioSpec) -> str:
    chunks = [
        '<material name="pss_floor" rgba="0.64 0.66 0.68 1"\n              reflectance="0.02" roughness="0.92"/>'
    ]
    if any((prop.material == "pss_wood_grid" for prop in scenario.scene_props)):
        chunks.extend(
            [
                '<texture name="pss_wood_grid_tex" type="2d" builtin="checker" mark="edge"\n             rgb1="0.62 0.42 0.24" rgb2="0.78 0.58 0.36" markrgb="0.42 0.27 0.14"\n             width="512" height="512"/>',
                '<material name="pss_wood_grid" texture="pss_wood_grid_tex" texrepeat="8 3"\n              reflectance="0.03" roughness="0.82"/>',
            ]
        )
    return "\n    ".join(chunks)


def _gallery_visuals(xml: str, scenario: ScenarioSpec) -> str:
    if scenario.meta.get("visual_style") != "physion_gallery":
        return xml
    from xml.etree import ElementTree as ET

    root = ET.fromstring(xml)
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name") in {"back_wall", "left_marker", "right_marker"}:
            if geom.get("contype") != "0" or geom.get("conaffinity") != "0":
                raise ValueError("Gallery may only remove noncolliding decoration")
            world.remove(geom)
    root.find("visual/headlight").set("diffuse", "0.24 0.25 0.27")
    root.find("visual/headlight").set("ambient", "0.18 0.19 0.20")
    root.find("visual/headlight").set("specular", "0 0 0")
    for light in world.findall("light"):
        if light.get("name") == "pss_key":
            light.set("diffuse", "0.58 0.56 0.53")
            light.set("specular", "0.12 0.12 0.12")
        elif light.get("name") == "pss_fill":
            light.set("diffuse", "0.22 0.26 0.30")
    for material in root.findall("asset/material"):
        if material.get("name") == "pss_floor":
            material.set("rgba", "0.62 0.65 0.66 1")
            material.set("reflectance", "0")
            material.set("specular", "0")
            material.set("shininess", "0")
        elif material.get("name") in {"pss_table_top", "pss_table_base"}:
            rgba = material.get("rgba").split()
            material.set("rgba", " ".join(rgba[:3] + ["0"]))
    for texture in root.findall("asset/texture"):
        if texture.get("name") == "pss_sky":
            texture.set("rgb1", "0.80 0.84 0.87")
            texture.set("rgb2", "0.94 0.95 0.96")
    return ET.tostring(root, encoding="unicode")
