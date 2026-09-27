import argparse
import json
import math
import time
from pathlib import Path


def wait_ready(queue, timeout=300):
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Timeout must be finite and nonnegative")
    deadline = time.monotonic() + timeout
    while True:
        try:
            state = json.loads((Path(queue) / "ready.json").read_text())
            if state["status"] == "ready" and 0 <= time.time() - state["heartbeat_unix"] <= 15:
                return state
            if state["status"] == "stopped":
                raise RuntimeError("Inference worker stopped; inspect the worker log")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Inference worker is not ready; inspect the worker log")
        time.sleep(0.2)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss wait", description="Wait for a live resident GPU worker."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    state = wait_ready(args.queue, args.timeout)
    print(f"Ready: {state['host']} / {state['worker_id']}")
