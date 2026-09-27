import argparse
import json
from pathlib import Path

from .artifacts import DEFAULT_CALIBRATION_DIR, CalibrationStore, save_calibration
from .fit import fit_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss calibrate",
        description="Fit session calibration and save the artifact used automatically by PSS.",
    )
    parser.add_argument("training", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--minimum-scale", type=float, default=0.001)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--calibration-dir", type=Path, default=Path(DEFAULT_CALIBRATION_DIR))
    destination.add_argument(
        "--output", type=Path, help="Explicit record file instead of automatic lookup"
    )
    parser.add_argument(
        "--replace", action="store_true", help="Explicitly replace an existing record"
    )
    args = parser.parse_args(argv)
    record = fit_calibration(
        json.loads(args.training.read_text(encoding="utf-8")),
        json.loads(args.calibration.read_text(encoding="utf-8")),
        delta=args.delta,
        minimum_scale=args.minimum_scale,
    )
    if args.output is not None:
        path = save_calibration(record, args.output, replace=args.replace)
    else:
        path = CalibrationStore(args.calibration_dir).save(record, replace=args.replace)
    print(json.dumps({**record.receipt(), "calibration_path": str(path)}))


if __name__ == "__main__":
    main()
