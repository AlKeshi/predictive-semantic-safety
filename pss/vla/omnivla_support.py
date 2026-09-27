from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

OMNIVLA_SOURCE_REPOSITORY = "https://github.com/NHirose/OmniVLA"
OMNIVLA_SOURCE_REVISION = "5182600cb4a9ee07684e17cdd2a6cbafc56b8a68"
OMNIVLA_CHECKPOINT_ID = "NHirose/omnivla-original"
OMNIVLA_CHECKPOINT_REVISION = "e36a84d4923c041149d441f93f3bdb7092bb5f07"
OMNIVLA_CHECKPOINT_STEP = 120000
OMNIVLA_ACTION_PERIOD_S = 1.0 / 3.0
OMNIVLA_ACTION_HORIZON = 8
OMNIVLA_ACTION_DIM = 4
OMNIVLA_POSE_DIM = 4
OMNIVLA_LANGUAGE_POSE_MODALITY_ID = 8
OMNIVLA_GOAL_PROMPT = "move toward the specified goal while avoiding obstacles"
OMNIVLA_WAYPOINT_INDEX = 4
OMNIVLA_WAYPOINT_SPACING_M = 0.1
OMNIVLA_GOAL_DISTANCE_CAP_M = 30.0
_CHECKPOINT_SHA256 = {
    "model-00001-of-00004.safetensors": "3bd0dd7d75924136ecf1a584bdaa8e8b994d4f5d80a7b8869814976f371c05d2",
    "model-00002-of-00004.safetensors": "3c21dcdb5f797631a5e555f749e0ec83a34f4d7227265e7ce3a6c320f8b42994",
    "model-00003-of-00004.safetensors": "5af986297dfbf994d5cf1343071d3fc30c5ba531662c9ac8c1a8364ce26fa1af",
    "model-00004-of-00004.safetensors": "a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60",
    "action_head--120000_checkpoint.pt": "d22c198064c5c90202833162a5edb719dd0607070c549e31345785a2623281e7",
    "proprio_projector--120000_checkpoint.pt": "1bd796dfeb4becda52baf09c9fbba6121818a57069750e925d5769f9a02feb04",
}
_CHECKPOINT_SHA256.update(
    {
        "added_tokens.json": "ab43123267b190cb7990bc9f2ae4ad32dcb7ed029fb0fccb1b4e062c6f54a2a1",
        "config.json": "1546e9d71e6cd37cbfd0f5a5122292b9c81412054baa36195b1404a6f7d7a76a",
        "generation_config.json": "67fbe17b12e8beb5be98eb911483971cc78589e51a37ac0cce9cd499d06a8bf1",
        "model.safetensors.index.json": "ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659",
        "preprocessor_config.json": "bd024783fb785cbdfe35b710b621c5e8678d4c157a9ba625675e67d40d1757da",
        "processor_config.json": "167e026d8121f6cb0bf5c18310b97ba8139152f970822bb3824213bd37dc7a76",
        "special_tokens_map.json": "cb90ee5cf5793aa444039af9795b00db59d3fef8942e7eca9cf1990bab370d61",
        "tokenizer.json": "8f5e2869e1807b8bb3c7717a294539c37e9de5d728ad60e31ef57e83cc5ea527",
        "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
        "tokenizer_config.json": "f5f1d3ed015ebb71cf11686ff00bc8e0c25957bbf3952f95312cecc9b168fbea",
    }
)


def _official_waypoint_pd_command(waypoint: np.ndarray) -> np.ndarray:
    dx, dy, hx, hy = np.asarray(waypoint, dtype=float)
    dt = OMNIVLA_ACTION_PERIOD_S
    epsilon = 1e-08
    if abs(dx) < epsilon and abs(dy) < epsilon:
        linear = 0.0
        angular = _wrap_angle(float(np.arctan2(hy, hx))) / dt
    elif abs(dx) < epsilon:
        linear = 0.0
        angular = float(np.sign(dy) * np.pi / (2.0 * dt))
    else:
        linear = float(dx / dt)
        angular = float(np.arctan(dy / dx) / dt)
    linear = float(np.clip(linear, 0.0, 0.5))
    angular = float(np.clip(angular, -1.0, 1.0))
    max_v = 0.3
    max_w = 0.3
    if abs(linear) <= max_v:
        if abs(angular) <= max_w:
            limited_v, limited_w = (linear, angular)
        else:
            radius = linear / angular
            limited_v = max_w * np.sign(linear) * abs(radius)
            limited_w = max_w * np.sign(angular)
    elif abs(angular) <= 0.001:
        limited_v = max_v * np.sign(linear)
        limited_w = 0.0
    else:
        radius = linear / angular
        if abs(radius) >= max_v / max_w:
            limited_v = max_v * np.sign(linear)
            limited_w = max_v * np.sign(angular) / abs(radius)
        else:
            limited_v = max_w * np.sign(linear) * abs(radius)
            limited_w = max_w * np.sign(angular)
    return np.array([limited_v, 0.0, limited_w], dtype=np.float32)


def _load_state_dict(path: Path, torch_module) -> dict[str, Any]:
    state = torch_module.load(path, map_location="cpu")
    return {key.removeprefix("module."): value for key, value in state.items()}


def _safe_register(registry, *args) -> None:
    try:
        registry.register(*args)
    except ValueError as exc:
        if "already" not in str(exc).lower():
            raise


def _require_git_revision(path: Path, expected: str, *, label: str) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    actual = completed.stdout.strip()
    if actual != expected:
        raise ValueError(f"{label} revision mismatch: {actual} != {expected}")
    changed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changed:
        raise ValueError(f"{label} has modified tracked files")
    return actual


def _validate_checkpoint(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected in _CHECKPOINT_SHA256.items():
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing OmniVLA checkpoint file: {candidate}")
        if candidate.suffix in {".safetensors", ".pt"} and candidate.stat().st_size < 1000000:
            raise ValueError(f"OmniVLA checkpoint file is still an LFS pointer: {candidate}")
        digest = _file_sha256(candidate)
        if digest != expected:
            raise ValueError(f"OmniVLA checkpoint hash mismatch for {name}: {digest}")
        hashes[name] = digest
    return hashes


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)
