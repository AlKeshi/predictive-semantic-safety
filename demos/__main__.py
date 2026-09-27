import argparse
from pathlib import Path

from .api import catalogue as catalogue
from .api import run_demo


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss demo", description="Run a packaged PSS or baseline demo."
    )
    parser.add_argument("demo", nargs="?", choices=catalogue())
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--omnivla-source", type=Path)
    parser.add_argument("--omnivla-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, help="Override simulation duration in seconds")
    parser.add_argument("--render", action="store_true")
    args = vars(parser.parse_args(argv))
    if args.pop("list"):
        print("\n".join(catalogue()))
        return
    if not args["demo"] or args["output"] is None:
        parser.error("demo and --output are required")
    try:
        run_demo(**args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Saved {args['output']}")


if __name__ == "__main__":
    main()
