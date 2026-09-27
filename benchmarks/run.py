import argparse
import hashlib
import json
import math
from pathlib import Path

from pss.calibration import DEFAULT_CALIBRATION_DIR

from .config import CeilingConfig, EpisodeConfig, RouteConfig
from .summarize import LABELS


def run(
    case,
    method,
    output,
    model=None,
    omnivla_source=None,
    omnivla_checkpoint=None,
    calibration=None,
    render=False,
    device="cuda",
    calibration_dir=DEFAULT_CALIBRATION_DIR,
    navigation_profile="benchmark",
    demo_margin_m=0.0,
    scene_xml=None,
    demo_baseline=False,
):
    if navigation_profile not in {"benchmark", "demo-ground-truth"}:
        raise ValueError("Unknown navigation profile")
    if navigation_profile != "benchmark" and method != "pss_uncalibrated":
        raise ValueError("Demo navigation is a separate uncalibrated PSS experiment")
    if not math.isfinite(demo_margin_m) or demo_margin_m < 0:
        raise ValueError("Demo margin must be finite and nonnegative")
    if navigation_profile == "benchmark" and demo_margin_m != 0:
        raise ValueError("Demo margin cannot alter the original benchmark protocol")
    if method not in LABELS:
        raise ValueError("Unknown comparison method")
    if method == "pss" and (scene_xml is not None or demo_baseline):
        raise ValueError("Demo environment overrides are outside the frozen calibration pipeline")
    if demo_baseline and method not in {"plain_cbf", "backup_cbf"}:
        raise ValueError("Demo baseline settings require a reactive baseline method")
    supplied_record = calibration is not None or Path(calibration_dir) != Path(
        DEFAULT_CALIBRATION_DIR
    )
    matched = method == "pss" or (method == "pss_uncalibrated" and supplied_record)
    if supplied_record and not matched:
        raise ValueError("Calibration options are only valid for PSS methods")
    if matched and (navigation_profile != "benchmark" or scene_xml is not None):
        raise ValueError("Matched calibration experiments cannot use demo overrides")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if method in {"plain_cbf", "backup_cbf"}:
        from .reactive import run as simulate

        report = simulate(
            case, output, method, render=render, scene_xml=scene_xml, demo_profile=demo_baseline
        )
    else:
        transform = None
        uncertainty = {"calibrated": False, "residual_m": 0.0}
        if navigation_profile != "benchmark":
            uncertainty = {
                "calibrated": False,
                "residual_m": demo_margin_m,
                "source": "fixed metric demo padding; no fitting or coverage claim",
            }
        external = None
        if method == "omnivla":
            if omnivla_source is None or omnivla_checkpoint is None:
                raise ValueError("OmniVLA source and checkpoint directories are required")
            from .omnivla import OmniVLA

            external = OmniVLA(omnivla_source, omnivla_checkpoint, device)
        elif model is None:
            raise ValueError("Code-as-World checkpoint directory is required")
        if matched:
            from pss.calibration import load_calibration, pipeline_digest

            from .pipeline import CalibratedSession, describe

            pipeline = describe(case["family"], model, case=case, device=device)
            record = load_calibration(pipeline, directory=calibration_dir, path=calibration)
            actual = pipeline_digest(pipeline)
            transform = CalibratedSession(
                record,
                actual,
                duration=case["duration"],
                family=case["family"],
                margin=method == "pss",
            )
            uncertainty = transform.receipt()
        output.mkdir(parents=True)
        if matched:
            record.save(output / "calibration_record.json")
        (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        settings = dict(
            output=output,
            duration=case["duration"],
            seed=case["seed"],
            speed=case["speed"],
            model=model,
            render=render,
            device=device,
            navigation_profile=navigation_profile,
            demo_margin_m=demo_margin_m,
            scene_xml=scene_xml,
        )
        config: EpisodeConfig
        if case["family"] == "ceiling":
            from .ceiling import run_episode

            config = CeilingConfig(method="omnivla" if external else "cw", **settings)
            report = run_episode(
                config, case, external=external, transform=transform, uncertainty=uncertainty
            )
        else:
            from datagen import route_scene

            from .route import run_episode

            scenario = route_scene(case)
            if external is not None:
                scenario.meta["omnivla_camera_mount"]["pitch_down_deg"] = 0.0
            config = RouteConfig(
                scene=case["id"],
                method="nominal" if external else "code_world",
                warmup=case["warmup"],
                **settings,
            )
            report = run_episode(
                config, scenario, external=external, transform=transform, uncertainty=uncertainty
            )
    if matched:
        from pss.calibration import WholePlaneForecast

        try:
            transform.check_time(case["duration"] + 1e-06)
        except WholePlaneForecast:
            pass
        (output / "calibration_trace.json").write_text(
            json.dumps(transform.trace(case["id"]), indent=2, allow_nan=False) + "\n"
        )
        report["uncertainty"] = transform.receipt()
        report.pop("q_m", None)
        report_path = output / ("receipt.json" if case["family"] == "ceiling" else "report.json")
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    hazard_contact = bool(
        report.get("hazard_contact", report.get("hazard_contact_physics_steps", 0))
    )
    environment_contact = bool(report["environment_contact"])
    result = {
        "method": method,
        "trial": case["id"],
        "family": case["family"],
        "base_case_id": case.get("base_case_id", case["id"]),
        "intended_hazard": case["intended_hazard"],
        "contact_free": not hazard_contact and (not environment_contact),
        "hazard_contact": hazard_contact,
        "environment_contact": environment_contact,
        "status": "complete",
        "paper_comparable": navigation_profile == "benchmark"
        and scene_xml is None
        and not demo_baseline,
        "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
    }
    result["goal_reached"] = bool(
        report.get("goal_reached", report.get("robot_goal_reached", False))
    )
    result["goal_time_s"] = report.get("goal_time_s", report.get("reached_at_s"))
    result["navigation_success"] = bool(result["contact_free"] and result["goal_reached"])
    if navigation_profile != "benchmark":
        result["navigation_profile"] = navigation_profile
        result["controller_schema"] = "demo_navigation_ground_truth_v1_uncalibrated"
        result["paper_comparable"] = False
        result["heuristic_margin_m"] = demo_margin_m
    if matched:
        result["calibration_record_digest"] = record.record_digest
        result["calibration_pipeline_digest"] = record.pipeline_digest
        result["residual_margin_applied"] = transform.margin
        result["calibration_predictions_complete"] = transform.receipt()[
            "all_required_predictions_present"
        ]
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss benchmark", description="Run one of the five paper comparison methods."
    )
    parser.add_argument("--method", choices=LABELS, required=True)
    parser.add_argument("--case", required=True, help="Frozen case ID or a case JSON file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--omnivla-source", type=Path)
    parser.add_argument("--omnivla-checkpoint", type=Path)
    calibration_source = parser.add_mutually_exclusive_group()
    calibration_source.add_argument("--calibration", type=Path, help="Explicit fitted record")
    calibration_source.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path(DEFAULT_CALIBRATION_DIR),
        help="Automatically load the record matching the frozen runtime pipeline",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render", action="store_true")
    args = vars(parser.parse_args(argv))
    from datagen import load_case

    case_arg = args.pop("case")
    case = (
        json.loads(Path(case_arg).read_text()) if Path(case_arg).is_file() else load_case(case_arg)
    )
    run(case=case, **args)
    print(f"Saved {args['output']}")


if __name__ == "__main__":
    main()
