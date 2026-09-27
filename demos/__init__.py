import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def case(name):
    return json.loads((ROOT / "cases" / f"{name}.json").read_text())


def scene_path(name):
    return ROOT / json.loads((ROOT / "provenance.json").read_text())[name]["scene_file"]


def scene_xml(name):
    path = scene_path(name)
    root = ET.fromstring(path.read_text())
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        compiler.set("meshdir", str((path.parent / compiler.get("meshdir")).resolve()))
    for include in root.findall(".//include"):
        include.set("file", str((path.parent / include.get("file")).resolve()))
    return ET.tostring(root, encoding="unicode")
