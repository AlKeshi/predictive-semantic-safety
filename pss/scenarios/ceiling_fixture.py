import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from pss.sim.mujoco.go1_policy import _HOME_ANGLES


def build_scene():
    assets = Path(__file__).resolve().parents[1] / "sim" / "mujoco" / "assets"
    root = ET.parse(assets / "go1" / "go1.xml").getroot()
    room = ET.parse(assets / "ceiling_fixture.xml").getroot()
    root.set("model", "Go1 ceiling-fixture physical attachment failure")
    root.find("compiler").set("meshdir", str(assets / "go1" / "meshes"))
    old = root.find("option")
    position = list(root).index(old)
    root.remove(old)
    root.insert(position, copy.deepcopy(room.find("option")))
    root.append(copy.deepcopy(room.find("visual")))
    collision = root.find(".//default[@class='collision']/geom")
    collision.set("contype", "0")
    collision.set("conaffinity", "1")
    for tag in ["asset", "worldbody"]:
        for node in room.find(tag):
            root.find(tag).append(copy.deepcopy(node))
    root.find(".//body[@name='rgbd_head']").set(
        "quat", "0.9930684569549263 0 -0.11753739745783764 0"
    )
    root.find(".//camera[@name='ego_rgbd']").set("fovy", "64")
    for tag in ["equality", "tendon"]:
        root.append(copy.deepcopy(room.find(tag)))
    return ET.tostring(root, encoding="unicode")


class LoadDrivenAttachments:
    def __init__(self, model, seed=0):
        self.model = model
        rng = np.random.default_rng(seed)
        self.names = ("left", "right")
        self.ids = np.array([model.equality(name + "_attachment").id for name in self.names])
        self.wires = np.array([model.tendon(name + "_suspension_wire").id for name in self.names])
        self.body = model.body("loose_fixture").id
        self.reference_load = float(model.body_mass[self.body]) * 9.81 / 2
        self.lifetimes = np.array([15.0, 3.0]) * rng.uniform(0.97, 1.03, 2)
        self.damage = np.zeros(2)
        self.loads = np.zeros(2)
        self.events = []
        self.phase = float(rng.uniform(0, 2 * np.pi))
        self.dof = model.jnt_dofadr[model.body_jntadr[self.body]]

    def step(self, data):
        dt = float(self.model.opt.timestep)
        if any(data.eq_active[self.ids]):
            data.xfrc_applied[self.body, 3] = 0.1 * np.sin(2 * np.pi * 3.1 * data.time + self.phase)
            data.qfrc_applied[self.dof + 3 : self.dof + 6] = (
                -60 * data.qvel[self.dof + 3 : self.dof + 6]
            )
        else:
            data.xfrc_applied[self.body] = 0
            data.qfrc_applied[self.dof + 3 : self.dof + 6] = 0
        for index, eqid in enumerate(self.ids):
            if not data.eq_active[eqid]:
                self.loads[index] = 0
                continue
            rows = (data.efc_type[: data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (
                data.efc_id[: data.nefc] == eqid
            )
            load = float(np.linalg.norm(data.efc_force[: data.nefc][rows]))
            self.loads[index] = load
            self.damage[index] += dt * (load / self.reference_load) ** 2 / self.lifetimes[index]
            softness = np.clip((self.damage[index] - 0.3) / 0.7, 0, 1)
            self.model.eq_solref[eqid, 0] = 0.004 + 0.085 * softness**2
            self.model.eq_solimp[eqid, :2] = [0.95 - 0.85 * softness, 0.99 - 0.65 * softness]
            if self.damage[index] >= 1:
                data.eq_active[eqid] = 0
                self.model.tendon_rgba[self.wires[index], 3] = 0
                self.events.append(
                    {
                        "time_s": float(data.time),
                        "support": self.names[index],
                        "event": "load_damage_release",
                        "load_n": load,
                    }
                )


def initialize_robot(model, data, policy):
    data.qpos[:7] = [-1.3, 0, 0.278, 1, 0, 0, 0]
    data.qpos[7:19] = _HOME_ANGLES
    data.ctrl[:] = _HOME_ANGLES
    mujoco.mj_forward(model, data)
    policy.reset()
    for step in range(500):
        if step % 10 == 0:
            policy.apply(model, data, np.zeros(3))
        mujoco.mj_step(model, data)
    data.time = 0
