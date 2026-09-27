import argparse
import copy
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .checkpoint import MODEL_ID, MODEL_REVISION


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class RoutedCodeAsWorldPredictor:
    def __init__(
        self,
        model_path=MODEL_ID,
        *,
        max_new_tokens=768,
        max_frames=16,
        max_pixels=262144,
        device="cuda",
        revision=MODEL_REVISION,
        queue=None,
    ):
        if device != "cuda" or revision != MODEL_REVISION:
            raise ValueError("Router requires the pinned CUDA checkpoint")
        if max_new_tokens < 1 or max_frames < 2 or max_pixels < 1024:
            raise ValueError("Invalid inference limits")
        queue = queue or os.environ.get("PSS_INFERENCE_QUEUE")
        if queue is None:
            raise ValueError("Set queue or PSS_INFERENCE_QUEUE to the resident worker directory")
        self.queue = Path(queue).resolve()
        ready = json.loads((self.queue / "ready.json").read_text())
        if ready["status"] != "ready" or time.time() - ready["heartbeat_unix"] > 15:
            raise RuntimeError("Resident inference worker is not ready")
        self.model_path = Path(ready["model"]["model_path"])
        if str(model_path) != MODEL_ID and Path(model_path).resolve() != self.model_path.resolve():
            raise ValueError("Resident checkpoint differs from the requested checkpoint")
        self.worker_id = ready["worker_id"]
        self.max_new_tokens, self.max_frames, self.max_pixels = (
            max_new_tokens,
            max_frames,
            max_pixels,
        )
        self.inference_profile = "fast"
        self.provenance = dict(
            ready["model"],
            transport="shared_filesystem_resident_v1",
            max_new_tokens=max_new_tokens,
            max_frames=max_frames,
            max_pixels=max_pixels,
        )

    def infer(self, frames, prompt, timestamps, *, forecast_space="pixel", response_schema=None):
        started = time.perf_counter()
        result = dict(
            success=False,
            raw_response="",
            parsed_json=None,
            error=None,
            provenance=dict(self.provenance),
            forecast_space=forecast_space,
        )
        try:
            if response_schema is None and hasattr(self, "schema_stage"):
                from .structured_json import ANALYSIS, EVENT

                response_schema = ANALYSIS if self.schema_stage == "analysis" else EVENT
            if response_schema is None:
                raise ValueError("Fast inference requires an explicit response schema")
            request_id = uuid.uuid4().hex
            folder = self.queue / "requests" / request_id
            folder.mkdir(parents=True, exist_ok=False)
            np.savez(folder / "input.npz", rgb=np.asarray(frames), times=np.asarray(timestamps))
            request = dict(
                worker_id=self.worker_id,
                prompt=prompt,
                forecast_space=forecast_space,
                response_schema=response_schema,
                limits={
                    key: getattr(self, key)
                    for key in ("max_new_tokens", "max_frames", "max_pixels")
                },
            )
            atomic_json(folder / "request.json", request)
            response = folder / "response.json"
            while not response.exists():
                ready = json.loads((self.queue / "ready.json").read_text())
                if (
                    ready["worker_id"] != self.worker_id
                    or ready["status"] != "ready"
                    or time.time() - ready["heartbeat_unix"] > 15
                ):
                    raise RuntimeError("Inference worker stopped or changed during the request")
                if time.perf_counter() - started > 600:
                    raise TimeoutError(
                        "Inference did not finish within 600 seconds; request retained"
                    )
                time.sleep(0.02)
            result = json.loads(response.read_text())
            result["router"] = dict(
                request_id=request_id,
                worker_id=self.worker_id,
                server_latency_s=result.get("latency_s"),
            )
        except Exception as exc:
            result["error"] = dict(type=type(exc).__name__, message=str(exc))
        result["latency_s"] = time.perf_counter() - started
        return result


def serve(queue, model_path, *, gpu_memory_utilization=0.5):
    import fcntl

    from .vllm_backend import VLLMCodeAsWorldPredictor

    queue = Path(queue).resolve()
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            status="loading",
            worker_id=uuid.uuid4().hex,
            pid=os.getpid(),
            host=socket.gethostname(),
            job_id=os.getenv("SLURM_JOB_ID"),
        )
        stop = threading.Event()

        def heartbeat():
            while not stop.is_set():
                atomic_json(queue / "ready.json", dict(state, heartbeat_unix=time.time()))
                stop.wait(2)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            resident = VLLMCodeAsWorldPredictor(
                model_path,
                max_new_tokens=3000,
                max_frames=12,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            state.update(status="ready", model=resident.provenance, model_loads=1)
            requests = queue / "requests"
            requests.mkdir(exist_ok=True)
            while True:
                for path in sorted(requests.glob("*/request.json")):
                    response = path.parent / "response.json"
                    if response.exists():
                        continue
                    request = json.loads(path.read_text())
                    if request["worker_id"] != state["worker_id"]:
                        continue
                    state["request_id"] = path.parent.name
                    started = time.perf_counter()
                    try:
                        predictor = copy.copy(resident)
                        predictor.provenance = dict(resident.provenance)
                        for key in ("max_new_tokens", "max_frames", "max_pixels"):
                            value = request["limits"][key]
                            setattr(predictor, key, value)
                            predictor.provenance[key] = value
                        with np.load(path.parent / "input.npz", allow_pickle=False) as clip:
                            result = predictor.infer(
                                clip["rgb"],
                                request["prompt"],
                                clip["times"],
                                forecast_space=request["forecast_space"],
                                response_schema=request["response_schema"],
                            )
                    except Exception as exc:
                        result = dict(
                            success=False,
                            raw_response="",
                            parsed_json=None,
                            error=dict(type=type(exc).__name__, message=str(exc)),
                            latency_s=time.perf_counter() - started,
                            provenance=dict(resident.provenance),
                        )
                    atomic_json(response, result)
                    state.pop("request_id", None)
                time.sleep(0.02)
        finally:
            stop.set()
            thread.join()
            atomic_json(
                queue / "ready.json", dict(state, status="stopped", heartbeat_unix=time.time())
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss serve", description="Keep Code-as-World resident on a CUDA GPU."
    )
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args(argv)
    serve(args.queue, args.model, gpu_memory_utilization=args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
