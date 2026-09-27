from types import SimpleNamespace

from benchmarks.sensor import install_sensor
from demos import case, scene_xml
from pss.sim.mujoco.go1_scene import _go1_assets, _require_mujoco
from pss.sim.mujoco.rgbd import MujocoRGBDCamera


def build():
    mj = _require_mujoco()
    xml = scene_xml("active-avoidance")
    initial = case("active-avoidance")
    model = mj.MjModel.from_xml_string(xml, _go1_assets())
    data = mj.MjData(model)
    robot_geoms = {g for g in range(model.ngeom) if model.geom_group[g] == 3}
    for g in robot_geoms:
        model.geom_conaffinity[g] |= 1
        model.body_conaffinity[model.geom_bodyid[g]] |= 1
    data.qpos[:] = initial["initial_qpos"]
    data.qvel[:] = initial["initial_qvel"]
    mj.mj_forward(model, data)
    return (mj, model, data, robot_geoms, xml, initial)


def sensor(mj, model, data, out, objects):
    scenario = SimpleNamespace(objects=[], meta={"omnivla_camera_mount": {"pitch_down_deg": 8}})
    camera = MujocoRGBDCamera(mj, model, scenario, camera_name="omnivla_ego", width=640, height=480)
    world = SimpleNamespace(
        mj=mj,
        model=model,
        data=data,
        scenario=scenario,
        omnivla_camera_id=model.camera("omnivla_ego").id,
        capture_forward_egocentric=lambda: camera.capture(data),
        close=camera.close,
    )
    install_sensor(world, out, lambda _: objects)
    return world
