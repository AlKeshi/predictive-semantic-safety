import json
import math
from pathlib import Path

from pss.io import save_json

from . import ROOT, case, scene_xml


def recipes():
    return json.loads((ROOT / "recipes.json").read_text())


def catalogue():
    return {name: (item["case"], item["method"]) for name, item in recipes().items()}


def run_demo(
    demo,
    output,
    *,
    model=None,
    duration=None,
    render=False,
    device="cuda",
    omnivla_source=None,
    omnivla_checkpoint=None,
):
    options = recipes()[demo]
    name, method = (options.pop("case"), options.pop("method"))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be finite and positive")
    if method in {"on", "pss_uncalibrated"} and model is None:
        raise ValueError("A Code-as-World checkpoint is required")
    if method in {"on", "off"}:
        from .active.run import run

        result = run(
            output,
            duration=duration or options["duration"],
            model=model,
            semantic_enabled=method == "on",
            render=render,
            device=device,
        )
    else:
        from benchmarks.run import run

        settings = case(name)
        if duration is not None:
            settings["duration"] = duration
        result = run(
            settings,
            method,
            output,
            model=model,
            render=render,
            device=device,
            omnivla_source=omnivla_source,
            omnivla_checkpoint=omnivla_checkpoint,
            scene_xml=scene_xml(name),
            **options,
        )
    save_json(
        output / "demo.json",
        dict(
            demo=demo,
            source=json.loads((ROOT / "provenance.json").read_text())[name],
            duration_override_s=duration,
            paper_comparable=False,
            calibration_performed=False,
        ),
    )
    return result
