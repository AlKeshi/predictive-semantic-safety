from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path

_CASES = Path(__file__).resolve().parent / "cases"


@lru_cache(maxsize=1)
def _trials():
    result = []
    for index, path in enumerate(sorted(_CASES.glob("*.json"))):
        base = json.loads(path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for repeat in range(5):
            seed = 202609100000 + index * 10 + repeat
            rng = random.Random(seed)
            dx = rng.uniform(-0.03, 0.03)
            dy = rng.uniform(-0.03, 0.03)
            yaw = rng.uniform(-math.pi / 90, math.pi / 90)
            case = copy.deepcopy(base)
            case.update(
                base_case_id=base["id"],
                id=f"{base['id']}_r{repeat:02d}",
                replicate_index=repeat,
                trial_seed=seed,
                start_x=base["start_x"] + dx,
                start_y=dy,
                yaw_rad=yaw,
                initial_condition_perturbation={"dx_m": dx, "dy_m": dy, "dyaw_rad": yaw},
                base_case_sha256=digest,
                physics_seed_note="seed preserves fixed case physics; trial_seed generates only the independent x/y/yaw perturbation",
            )
            result.append(case)
    return result


def list_cases(family: str | None = None) -> list[dict]:
    if family is not None and family not in {"ceiling", "stack", "dominos"}:
        raise ValueError("Family must be ceiling, stack or dominos")
    return copy.deepcopy([case for case in _trials() if family is None or case["family"] == family])


def load_case(case_id: str) -> dict:
    for case in _trials():
        if case["id"] == case_id:
            return copy.deepcopy(case)
    raise KeyError(f"Unknown case {case_id!r}")
