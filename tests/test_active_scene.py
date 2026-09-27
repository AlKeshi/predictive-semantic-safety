from itertools import product

import mujoco
import numpy as np
import pytest

from demos.active.run import build
from demos.active.scene import scene_inventory


def scene(extra="", shape="cylinder", size=".2 .3", offset="0 0 0"):
    xml = f"""<mujoco><worldbody>
      <geom name="floor" type="plane" size="10 10 .1"/>
      <body name="trunk"><freejoint/><geom type="sphere" size=".1"/></body>
      <body name="changed_identity" pos="1 2 3"><freejoint/>
        <geom name="different_geom_name" type="{shape}" size="{size}"
              pos="{offset}" rgba=".1 .2 .9 1"/>
      </body>{extra}</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return (model, data)


def test_inventory_follows_names_count_sizes_and_not_dynamic_poses():
    model, data = scene(
        '<body name="another"><freejoint/><geom type="sphere" size=".41" rgba=".9 .1 .2 1"/></body>'
    )
    objects, geoms, static = scene_inventory(model, data)
    assert [o["id"] for o in objects] == ["changed_identity", "another"]
    assert geoms[model.geom("different_geom_name").id] == "changed_identity"
    assert objects[0]["radius"] == pytest.approx(np.hypot(0.2, 0.3))
    assert objects[0]["size"] == pytest.approx([0.4, 0.4, 0.6])
    assert objects[1]["radius"] == 0.41
    assert static == []
    data.qpos[7:10] = [9, -8, 7]
    mujoco.mj_forward(model, data)
    assert scene_inventory(model, data) == (objects, geoms, static)


def test_static_map_includes_new_rotated_nested_fixed_obstacles():
    model, data = scene(
        '<body name="new_fixed" pos="-1 3 0" euler="30 45 60">'
        '<geom name="new_wall" type="box" size=".1 .2 .7"/></body>'
    )
    _, _, static = scene_inventory(model, data)
    assert [o["object_id"] for o in static] == ["new_wall"]
    gid = model.geom("new_wall").id
    vertices = np.array(list(product((-1, 1), repeat=3))) * [0.1, 0.2, 0.7]
    world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    distances = np.linalg.norm(world[:, :2] - static[0]["center"], axis=1)
    assert np.all(distances <= static[0]["radius"] + 1e-12)
    assert static[0]["radius"] > np.hypot(0.1, 0.2)


@pytest.mark.parametrize("kwargs", [dict(shape="box", size=".2 .2 .3"), dict(offset=".1 0 0")])
def test_unsupported_hazard_cannot_be_silently_dropped(kwargs):
    with pytest.raises(ValueError, match="primitive"):
        scene_inventory(*scene(**kwargs))


@pytest.mark.parametrize(
    "extra",
    [
        '<body name="hinged"><joint type="hinge"/><geom type="sphere" size=".2"/></body>',
        '<body name="fixed_root"><body name="hinged"><joint type="hinge"/>'
        '<geom type="sphere" size=".2"/></body></body>',
        '<geom name="wall_plane" type="plane" size="2 2 .1" euler="90 0 0"/>',
        '<body name="tracked" mocap="true"><geom type="sphere" size=".2"/></body>',
    ],
)
def test_unsupported_environment_cannot_be_omitted(extra):
    with pytest.raises(ValueError):
        scene_inventory(*scene(extra))


def test_original_demo_geometry_is_preserved():
    _, model, data, _, _, _ = build()
    objects, geoms, static = scene_inventory(model, data)
    assert len(objects) == len(geoms) == 5
    for i, obj in enumerate(objects):
        assert obj["id"] == (f"cylinder_{i}" if i < 4 else "striker")
        assert obj["radius"] == pytest.approx(np.hypot(0.18, 0.18) if i < 4 else 0.16)
        assert obj["surface_offset"] == (0.18 if i < 4 else 0.16)
    assert {o["object_id"] for o in static} == {
        "ball_launch_shelf",
        "ball_launch_stand",
        "table_top",
        "table_pedestal",
    }
    for obj in static:
        gid = model.geom(obj["object_id"]).id
        np.testing.assert_array_equal(obj["center"], data.geom_xpos[gid, :2])
        assert obj["radius"] == pytest.approx(np.linalg.norm(model.geom_size[gid, :2]))
