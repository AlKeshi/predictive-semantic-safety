import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from .run import run


def run_navigation(case, output, model, *, duration=None, render=False, margin_m=0.2):
    original = deepcopy(case)
    case = deepcopy(case)
    duration = (40.0 if case["family"] == "ceiling" else 35.0) if duration is None else duration
    if not math.isfinite(duration) or duration < case["duration"]:
        raise ValueError("Navigation duration must include the entire original benchmark horizon")
    case["duration"] = float(duration)
    output = Path(output)
    result = run(
        case,
        "pss_uncalibrated",
        output,
        model=model,
        render=render,
        navigation_profile="demo-ground-truth",
        demo_margin_m=margin_m,
    )
    result["original_case_sha256"] = hashlib.sha256(
        json.dumps(original, sort_keys=True).encode()
    ).hexdigest()
    result["original_duration_s"] = original["duration"]
    result["navigation_duration_s"] = case["duration"]
    result["heuristic_margin_m"] = margin_m
    result["ground_measurement"] = (
        "RGB-D settling"
        if case["family"] == "ceiling"
        else "oracle native current state and floor contacts"
    )
    (output / "navigation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "original_case.json").write_text(json.dumps(original, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Navigation")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--margin-m",
        type=float,
        default=0.2,
        help="Fixed predictive metric padding from the demos; no calibration",
    )
    parser.add_argument("--render", action="store_true")
    args = vars(parser.parse_args())
    from datagen import load_case

    source = args.pop("case")
    case = json.loads(Path(source).read_text()) if Path(source).is_file() else load_case(source)
    run_navigation(case, **args)
    print(f"Saved {args['output']}")


if __name__ == "__main__":
    main()
