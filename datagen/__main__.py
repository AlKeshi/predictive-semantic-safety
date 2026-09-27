import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from datagen import ceiling_xml, list_cases, load_case, route_scene


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss scenes", description="Export a native MuJoCo benchmark scene"
    )
    parser.add_argument("--case", default="stack_00_r00")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join((case["id"] for case in list_cases())))
        return
    if args.output is None:
        parser.error("--output is required when exporting a scene")
    case = load_case(args.case)
    args.output.mkdir(parents=True, exist_ok=True)
    if case["family"] == "ceiling":
        root = ET.fromstring(ceiling_xml(case))
        root.find("compiler").set("meshdir", "meshes")
        xml = ET.tostring(root, encoding="unicode")
    else:
        from pss.scenarios.code_world_suite import world_config
        from pss.sim.mujoco.go1_scene import _go1_assets, _go1_scene_xml

        scene = route_scene(case)
        root = ET.fromstring(_go1_scene_xml(scene, world_config(case["duration"])))
        for name, contact in scene.meta["native_contact"].items():
            geom = root.find(f".//geom[@name='dynamic_obstacle_geom_{name}']")
            geom.set("priority", str(contact["priority"]))
            geom.set("solref", " ".join(map(str, contact["solref"])))
        xml = ET.tostring(root, encoding="unicode")
        (args.output / "scene.json").write_text(json.dumps(scene.to_dict(), indent=2) + "\n")
        assets = _go1_assets(full_body_collisions=True)
        (args.output / "go1.xml").write_bytes(assets["go1.xml"])
    from pss.sim.mujoco.go1_policy import _ASSET_ROOT

    shutil.copytree(_ASSET_ROOT / "meshes", args.output / "meshes", dirs_exist_ok=True)
    (args.output / "scene.xml").write_text(xml)
    (args.output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    print(args.output / "scene.xml")


if __name__ == "__main__":
    main()
