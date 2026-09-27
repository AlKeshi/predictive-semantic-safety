from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from pss.scenarios.spec import DynamicObstacleSpec, ScenarioSpec

_ASSETS = Path(__file__).resolve().parent / "assets"
_COLORS = ((0.88, 0.075, 0.055), (0.1, 0.68, 0.19), (0.04, 0.34, 0.88), (0.94, 0.48, 0.035))


def route_scene(case: dict) -> ScenarioSpec:
    from pss.scenarios.code_world_suite import hazard_metadata

    family = case["family"]
    if family not in {"stack", "dominos"}:
        raise ValueError("A route scene must be stack or dominos")
    count = case["count"]
    if isinstance(count, bool) or count not in {2, 3, 4}:
        raise ValueError("The scene requires two, three or four blocks")
    scene = ScenarioSpec.from_dict(json.loads((_ASSETS / f"{family}_stage.json").read_text()))
    scene.seed = int(case["seed"])
    x, y = (case["hazard_x"], case["hazard_y"])
    speed, lead = (case["ball_speed"], case["ball_lead"])
    danger = case["intended_hazard"]
    if family == "stack":
        primitives = json.loads((_ASSETS / "stack_primitives.json").read_text())["objects"]
        objects = [DynamicObstacleSpec(**item) for item in primitives]
        ball = objects[0]
        blocks = sorted((o for o in objects if o.shape == "cuboid"), key=lambda o: o.position[2])[
            :count
        ]
        target = next((o for o in objects if o.name == "red_target"))
        bottom = 0.0
        for block in blocks:
            block.position = (x, y, bottom + block.size[2] / 2)
            bottom += block.size[2]
        target.position = (x, y, bottom + target.radius)
        ball.position = (
            x + (0 if danger else 0.65),
            y + blocks[0].size[1] / 2 + ball.radius + speed * lead,
            ball.radius,
        )
        ball.linear_velocity = (0.0, -speed, 0.0)
        ball.angular_velocity = (speed / ball.radius, 0.0, 0.0)
        scene.dynamic_obstacles = [ball, *blocks, target]
    else:
        height, radius = (case["height"], case["ball_radius"])
        spacing = height * case["spacing"]
        blocks = [
            DynamicObstacleSpec(
                name="red_target" if i == 0 else f"relay_{i}",
                shape="cuboid",
                position=(x, y + i * spacing, height / 2),
                size=(0.24, 0.1, height),
                color=_COLORS[i],
                density=550.0,
                friction=0.65,
                restitution=0.08,
            )
            for i in range(count)
        ]
        if not danger and count > 2:
            shifted = blocks[count // 2]
            shifted.position = (x + 0.65, shifted.position[1], shifted.position[2])
        ball = DynamicObstacleSpec(
            name="purple_ball",
            radius=radius,
            position=(
                x + (0 if danger or count > 2 else 0.65),
                y + (count - 1) * spacing + 0.05 + radius + speed * lead,
                radius,
            ),
            linear_velocity=(0.0, -speed, 0.0),
            angular_velocity=(speed / radius, 0.0, 0.0),
            color=(0.63, 0.06, 0.88),
            density=3000.0,
            friction=0.12,
            restitution=0.15,
        )
        scene.dynamic_obstacles = [ball, *blocks]
    scene.robot_start = (case.get("start_x", -2.0), case.get("start_y", 0.0))
    scene.robot_goal = (x + 2.0, 0.0)
    scene.scenario_id = "pss-paper-benchmark-v1/" + case["id"]
    scene.meta["robot_initial_yaw_rad"] = case.get("yaw_rad", 0.0)
    if case.get("mirror_y"):
        for obstacle in scene.dynamic_obstacles:
            obstacle.position = tuple(np.asarray(obstacle.position) * [1, -1, 1])
            obstacle.linear_velocity = tuple(np.asarray(obstacle.linear_velocity) * [1, -1, 1])
            obstacle.angular_velocity = tuple(np.asarray(obstacle.angular_velocity) * [-1, 1, -1])
        scene.robot_start = (scene.robot_start[0], -scene.robot_start[1])
    if "ball_launch_time_s" in case:
        launch = float(case["ball_launch_time_s"])
        if not np.isfinite(launch) or launch < 0:
            raise ValueError("Ball launch time must be finite and nonnegative")
        ball.activation_distance = 0.0
        ball.activation_center = (1000000.0, 1000000.0)
        scene.meta["timed_launches"] = {ball.name: launch}
    scene.meta["hazards"] = hazard_metadata(scene)
    scene.meta["code_as_world_scene"] = (
        "physion_"
        + ("support" if family == "stack" else "dominos")
        + ("_hit" if danger else "_clear")
    )
    scene.meta["native_contact"] = {
        o.name: {"priority": 1, "solref": [0.008, 1.0]} for o in scene.dynamic_obstacles
    }
    return scene


def ceiling_xml(case: dict) -> str:
    from pss.scenarios.ceiling_fixture import build_scene

    if case["family"] != "ceiling":
        raise ValueError("A ceiling scene requires a ceiling case")
    root = ET.fromstring(build_scene())
    shift = [case["hazard_x"] - 1.73, case["hazard_y"], case["height"] - 2.13]
    for node in root.find("worldbody"):
        name = node.get("name", "")
        if name == "loose_fixture" or name.startswith(
            (
                "left_mount",
                "right_mount",
                "left_ceiling_anchor",
                "right_ceiling_anchor",
                "left_remaining_wire",
                "right_remaining_wire",
            )
        ):
            if node.get("pos"):
                position = np.fromstring(node.get("pos"), sep=" ") + shift
                node.set("pos", " ".join(map(str, position)))
            if node.get("fromto"):
                points = np.fromstring(node.get("fromto"), sep=" ").reshape(2, 3) + shift
                node.set("fromto", " ".join(map(str, points.ravel())))
    return ET.tostring(root, encoding="unicode")


def attachments(model, case: dict):
    from pss.scenarios.ceiling_fixture import LoadDrivenAttachments

    result = LoadDrivenAttachments(model, case["seed"])
    if "attachment_lifetimes" in case:
        lifetimes = np.asarray(case["attachment_lifetimes"], float)
        damage = np.asarray(case["attachment_damage"], float)
        if (
            lifetimes.shape != (2,)
            or damage.shape != (2,)
            or (not np.isfinite(np.r_[lifetimes, damage]).all())
            or np.any(lifetimes <= 0)
            or np.any(damage < 0)
            or np.any(damage > 1)
        ):
            raise ValueError("Invalid two-attachment demo configuration")
        result.lifetimes, result.damage = (lifetimes.copy(), damage.copy())
        return result
    result.lifetimes *= case["lifetime_scale"] if case["intended_hazard"] else 1000.0
    return result
