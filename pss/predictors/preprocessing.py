from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import numpy as np


def parse_json_response(text: str) -> dict[str, Any] | list[dict[str, Any]]:
    raw = str(text).strip()
    while True:
        think = re.match("^<think>.*?</think>\\s*", raw, flags=re.DOTALL)
        fence = re.fullmatch("```(?:json)?[ \\t]*\\r?\\n(.*?)\\r?\\n```", raw, flags=re.DOTALL)
        if think:
            raw = raw[think.end() :].strip()
        elif fence:
            raw = fence.group(1).strip()
        else:
            break

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(payload, dict) and (
        not (
            isinstance(payload, list)
            and payload
            and all((isinstance(item, dict) for item in payload))
        )
    ):
        raise ValueError("model response must be a JSON object or nonempty object list")

    def finite_values(value: Any) -> None:
        if isinstance(value, float) and (not np.isfinite(value)):
            raise ValueError("non-finite JSON number")
        if isinstance(value, dict):
            for child in value.values():
                finite_values(child)
        elif isinstance(value, list):
            for child in value:
                finite_values(child)

    finite_values(payload)
    return payload


def prepare_clip(
    frames: Sequence[np.ndarray], timestamps: Sequence[float], max_frames: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(frames)
    times = np.asarray(timestamps, dtype=float)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB with shape [time,height,width,3]")
    if min(values.shape[1:3]) < 32 or len(values) < 2:
        raise ValueError("at least two RGB frames of at least 32 pixels are required")
    if times.shape != (len(values),) or not np.all(np.isfinite(times)):
        raise ValueError("each RGB frame requires one finite timestamp")
    if np.any(np.diff(times) <= 0):
        raise ValueError("frame timestamps must be strictly increasing")
    if max_frames < 2:
        raise ValueError("max_frames must be at least two")
    indices = np.linspace(0, len(values) - 1, min(max_frames, len(values)), dtype=int)
    return (np.ascontiguousarray(values[indices]), times[indices])
