import argparse
from importlib import import_module

COMMANDS = {
    "demo": "demos.__main__",
    "benchmark": "benchmarks.run",
    "scenes": "datagen.__main__",
    "calibrate": "pss.calibration.__main__",
    "summarize": "benchmarks.summarize",
    "checkpoint": "pss.predictors.checkpoint",
    "serve": "pss.predictors.inference_router",
    "wait": "pss.predictors.service",
    "doctor": "pss.doctor",
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pss", description="Predictive Semantic Safety")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    import_module(COMMANDS[args.command]).main(args.arguments)


if __name__ == "__main__":
    main()
