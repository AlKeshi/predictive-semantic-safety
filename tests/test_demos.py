import hashlib
import json
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from datagen import attachments, route_scene
from demos import ROOT, case, scene_path, scene_xml
from demos.__main__ import catalogue
from pss.scenarios.code_world_suite import world_config
from pss.sim.mujoco import Go1MujocoWorld


@pytest.mark.parametrize("name", sorted({v[0] for v in catalogue().values()}))
def test_packaged_scene_loads_from_file_and_string(name):
    a = mujoco.MjModel.from_xml_path(str(scene_path(name)))
    b = mujoco.MjModel.from_xml_string(scene_xml(name))
    assert a.nq == b.nq and a.ngeom == b.ngeom
    np.testing.assert_array_equal(a.qpos0, b.qpos0)
    assert a.nq > 19


@pytest.mark.parametrize("name", ["ceiling-a", "ceiling-b", "ceiling-c", "ceiling-omnivla"])
def test_ceiling_starts_with_intact_declared_attachments(name):
    settings = case(name)
    model = mujoco.MjModel.from_xml_string(scene_xml(name))
    physical = attachments(model, settings)
    assert np.all(model.eq_solref[physical.ids, 0] < 0.01)
    np.testing.assert_array_equal(model.eq_active0[physical.ids], settings["supports"])
    np.testing.assert_array_equal(physical.damage, settings["attachment_damage"])
    np.testing.assert_array_equal(physical.lifetimes, settings["attachment_lifetimes"])


@pytest.mark.parametrize("name", ["dominos-omnivla", "stack-omnivla"])
def test_timed_ball_waits_then_launches_once(name):
    settings = case(name)
    world = Go1MujocoWorld(route_scene(settings), world_config(1), scene_xml=scene_xml(name))
    try:
        world.prepare_online_episode()
        bid, ball = next(
            (
                (b, o)
                for b, o in world._dynamic_obstacle_by_body_id.items()
                if o.name == "purple_ball"
            )
        )
        adr = world.model.jnt_dofadr[world.model.body_jntadr[bid]]
        world.data.time = settings["ball_launch_time_s"] - 0.004
        world.apply_dynamic_obstacle_triggers()
        np.testing.assert_array_equal(world.data.qvel[adr : adr + 6], 0)
        world.data.time = settings["ball_launch_time_s"]
        assert world.apply_dynamic_obstacle_triggers()
        np.testing.assert_array_equal(world.data.qvel[adr : adr + 3], ball.linear_velocity)
        world.data.qvel[adr] = 0.123
        assert not world.apply_dynamic_obstacle_triggers()
        assert world.data.qvel[adr] == 0.123
    finally:
        world.close()


def test_mirrored_case_reflects_spin_as_an_axial_vector():
    settings = case("dominos-b")
    normal = deepcopy(settings)
    normal["mirror_y"] = False
    for a, b in zip(
        route_scene(normal).dynamic_obstacles, route_scene(settings).dynamic_obstacles, strict=True
    ):
        np.testing.assert_array_equal(b.position, np.asarray(a.position) * [1, -1, 1])
        np.testing.assert_array_equal(
            b.angular_velocity, np.asarray(a.angular_velocity) * [-1, 1, -1]
        )


def test_all_twenty_recipes_have_initial_assets_not_historical_answers():
    assert len(catalogue()) == 20
    for name, _ in catalogue().values():
        assert scene_path(name).is_file()
        assert (ROOT / "cases" / f"{name}.json").is_file()
    initial = json.loads((ROOT / "cases/active-avoidance.json").read_text())
    assert not {"goal_time", "first_contact", "trace", "forecasts"} & initial.keys()


def test_packaged_provenance_hashes_match_assets():
    for name, record in json.loads((ROOT / "provenance.json").read_text()).items():
        assert (
            hashlib.sha256(scene_path(name).read_bytes()).hexdigest()
            == record["portable_xml_sha256"]
        )
        assert (
            hashlib.sha256((ROOT / "cases" / f"{name}.json").read_bytes()).hexdigest()
            == record["case_sha256"]
        )


def test_demo_xml_cannot_bypass_calibration_identity(tmp_path):
    from benchmarks.run import run

    with pytest.raises(ValueError, match="frozen calibration"):
        run(case("ceiling-a"), "pss", tmp_path / "run", scene_xml="<mujoco/>")
