import json
import math
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from datagen import attachments, ceiling_xml, list_cases, load_case, route_scene
from pss.scenarios.code_world_suite import configure_native_contacts, world_config
from pss.sim.mujoco import Go1MujocoWorld


def test_paired_manifest_preserves_physics():
    cases = list_cases()
    assert len(cases) == 150
    assert len({case["trial_seed"] for case in cases}) == 150
    for family in ["ceiling", "stack", "dominos"]:
        assert len(list_cases(family)) == 50
    grouped = {}
    for case in cases:
        grouped.setdefault(case["base_case_id"], []).append(case)
        pose = case["initial_condition_perturbation"]
        assert abs(pose["dx_m"]) <= 0.03
        assert abs(pose["dy_m"]) <= 0.03
        assert abs(pose["dyaw_rad"]) <= math.pi / 90
    for group in grouped.values():
        assert len(group) == 5
        for key in ["seed", "hazard_x", "hazard_y", "count", "speed", "intended_hazard"]:
            assert len({case[key] for case in group}) == 1
    case = load_case("stack_00_r00")
    case["hazard_x"] = -100
    assert load_case(case["id"])["hazard_x"] != -100


@pytest.mark.parametrize("family", ["stack", "dominos"])
def test_native_scene_inventory_and_initialization(family):
    case = load_case(family + "_02_r00")
    scene = route_scene(case)
    assert len(scene.dynamic_obstacles) == case["count"] + (2 if family == "stack" else 1)
    assert not scene.objects and scene.disturbance is None
    assert all((o.activation_distance is None for o in scene.dynamic_obstacles))
    world = Go1MujocoWorld(scene, world_config(case["duration"]))
    try:
        configure_native_contacts(world)
        world.prepare_online_episode()
        first = (world.data.qpos.copy(), world.data.qvel.copy(), world.policy.last_action.copy())
        world.prepare_online_episode()
        for expected, actual in zip(
            first, [world.data.qpos, world.data.qvel, world.policy.last_action], strict=True
        ):
            np.testing.assert_array_equal(expected, actual)
        assert len(world.dynamic_obstacle_body_ids) == len(scene.dynamic_obstacles)
    finally:
        world.close()


@pytest.mark.parametrize("family", ["ceiling", "stack", "dominos"])
def test_export_is_portable_and_preserves_native_geometry(tmp_path, family):
    case_id = family + "_00_r00"
    path = tmp_path / family
    subprocess.run(
        [sys.executable, "-m", "datagen", "--case", case_id, "--output", str(path)], check=True
    )
    loaded = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    case = json.loads((path / "case.json").read_text())
    world = None
    if family == "ceiling":
        expected = mujoco.MjModel.from_xml_string(ceiling_xml(case))
        assert np.all(np.isfinite(attachments(expected, case).lifetimes))
    else:
        world = Go1MujocoWorld(route_scene(case), world_config(case["duration"]))
        configure_native_contacts(world)
        expected = world.model
    try:
        for name in [
            "body_mass",
            "body_inertia",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_solref",
            "geom_priority",
            "geom_contype",
            "geom_conaffinity",
            "qpos0",
        ]:
            np.testing.assert_array_equal(getattr(expected, name), getattr(loaded, name))
    finally:
        if world:
            world.close()
