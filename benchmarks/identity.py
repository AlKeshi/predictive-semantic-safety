import hashlib
import os
import platform
from importlib.metadata import version
from pathlib import Path


def describe(family, model, *, case, device="cuda"):
    root = Path(__file__).resolve().parents[1]
    files = {}
    for package in ("pss", "datagen", "benchmarks"):
        for path in sorted((root / package).rglob("*.py")):
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in ("pss/sim/mujoco/assets", "datagen/assets"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    omitted = {
        "id",
        "trial_seed",
        "replicate_index",
        "start_x",
        "start_y",
        "yaw_rad",
        "initial_condition_perturbation",
        "construction_probe",
        "physics_seed_note",
    }
    population = {key: value for key, value in case.items() if key not in omitted}
    population["base_start_x"] = round(
        case["start_x"] - case.get("initial_condition_perturbation", {}).get("dx_m", 0), 12
    )
    dependencies = {
        name: version(name)
        for name in (
            "numpy",
            "scipy",
            "osqp",
            "mujoco",
            "onnxruntime",
            "Pillow",
        )
    }
    backend = os.environ.get("PSS_INFERENCE_BACKEND", "transformers")
    profile = (
        "fast"
        if backend.startswith("vllm")
        else os.environ.get("PSS_INFERENCE_PROFILE", "baseline")
    )
    accelerator = None
    model = Path(model).expanduser().resolve()
    if backend == "vllm-router":
        from pss.predictors.service import wait_ready

        resident = wait_ready(os.environ["PSS_INFERENCE_QUEUE"], timeout=0)["model"]
        if Path(resident["model_path"]).resolve() != model:
            raise ValueError("Resident checkpoint differs from the calibration pipeline")
        dependencies["resident_inference"] = resident["acceleration_versions"]
        dependencies["resident_python"] = resident["python"]
        accelerator = resident["accelerator"]
    else:
        import torch

        dependencies.update(
            {name: version(name) for name in ("torch", "transformers", "accelerate")}
        )
        if profile == "fast":
            packages = (
                ("vllm",)
                if backend == "vllm"
                else ("flash-linear-attention", "fla-core", "causal-conv1d", "triton")
            )
            dependencies.update(
                {name: version(name) for name in (*packages, "xgrammar", "jsonschema")}
            )
        if str(device).startswith("cuda"):
            accelerator = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
    from pss.predictors.checkpoint import CHECKPOINT_FILES

    weights = {}
    for name in sorted(CHECKPOINT_FILES):
        path = model / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        weights[name] = digest.hexdigest()
    if not weights or not any((name.endswith((".safetensors", ".bin")) for name in weights)):
        raise ValueError("Model directory has no verifiable checkpoint weights")
    return {
        "schema": "pss_mujoco_fixed_pipeline_v6",
        "acquisition_mode": "calibrated_closed_loop",
        "case_population": population,
        "pose_sampling": "datagen fixed physical case with independent x/y/yaw perturbations",
        "device": str(device),
        "dtype": "bfloat16" if str(device).startswith("cuda") else "float32",
        "accelerator": accelerator,
        "python": platform.python_version(),
        "dependencies": dependencies,
        "family": family,
        "source_files": files,
        "checkpoint_files": weights,
        "control_hz": 50,
        "sensor_period_s": 0.05,
        "query_schedule": "exact origins declared in calibration record",
        "forecast_horizon_s": 8.0 if family == "ceiling" else 2.5,
        "inference": "synchronous_simulation_paused",
        "inference_profile": profile,
        "inference_backend": backend,
        "scope": "fixed sampled prediction indices",
    }
