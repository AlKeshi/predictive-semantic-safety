from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .preprocessing import parse_json_response, prepare_clip
from .video_inputs import VIDEO_PREPROCESSING_SCHEMA, _TokenTiming, prepare_video_inputs


class TransformersGeneration:
    def infer(
        self,
        frames: Sequence[np.ndarray],
        prompt: str,
        timestamps: Sequence[float],
        *,
        forecast_space: str = "pixel",
        response_schema: dict | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result: dict[str, Any] = {
            "success": False,
            "raw_response": "",
            "parsed_json": None,
            "latency_s": 0.0,
            "provenance": dict(self.provenance),
            "error": None,
            "forecast_space": forecast_space,
        }
        timings = {}
        phase = started
        try:
            if forecast_space not in ("pixel", "metric"):
                raise ValueError("forecast_space must be pixel or metric")
            rgb, times = prepare_clip(frames, timestamps, self.max_frames)
            if not str(prompt).strip():
                raise ValueError("a non-empty task prompt is required")
            result.update(
                frame_timestamps_s=times.tolist(),
                forecast_origin_s=float(times[-1]),
                input_shape=list(rgb.shape),
                input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                prompt_sha256=hashlib.sha256(str(prompt).encode()).hexdigest(),
            )
            _, _, inputs = prepare_video_inputs(
                self.processor,
                rgb,
                times,
                prompt,
                forecast_space,
                self.max_pixels,
                video_cache=getattr(self, "video_cache", None),
            )
            result["video_cache_hit"] = bool(
                getattr(getattr(self, "video_cache", None), "last_hit", False)
            )
            timings["prepare_and_process_s"] = time.perf_counter() - phase
            grid = inputs["video_grid_thw"].tolist()
            if len(grid) != 1 or len(grid[0]) != 3 or min(grid[0]) < 1:
                raise ValueError("invalid processed video grid")
            temporal = int(self.processor.video_processor.temporal_patch_size)
            patch = int(self.processor.video_processor.patch_size)
            grid_t, grid_h, grid_w = map(int, grid[0])
            processed_height, processed_width = (grid_h * patch, grid_w * patch)
            represented_frames = grid_t * temporal
            result["video_preprocessing"] = {
                "schema": VIDEO_PREPROCESSING_SCHEMA,
                "video_grid_thw": grid,
                "observed_frame_count": len(rgb),
                "represented_frame_count": represented_frames,
                "temporal_padding_frames": represented_frames - len(rgb),
                "processed_frame_height": processed_height,
                "processed_frame_width": processed_width,
                "processed_pixels_per_frame": processed_height * processed_width,
                "requested_max_pixels_per_frame": self.max_pixels,
                "patch_size": patch,
                "temporal_patch_size": temporal,
            }
            expected_frames = (len(rgb) + temporal - 1) // temporal * temporal
            if represented_frames != expected_frames:
                raise ValueError("processed video changed the observed frame count")
            if processed_height * processed_width > self.max_pixels:
                raise ValueError("processed video exceeds the requested per-frame pixel limit")
            if len(inputs["pixel_values_videos"]) != grid_t * grid_h * grid_w:
                raise ValueError("processed video patches do not match the reported grid")
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            result["prompt_tokens"] = prompt_tokens
            if prompt_tokens + self.max_new_tokens > 8192:
                raise ValueError("video and prompt exceed the frozen 8192-token inference budget")
            phase = time.perf_counter()
            self._active_response_schema = response_schema
            constraints = getattr(self, "_generation_constraints", None)
            if response_schema is not None:
                from .structured_json import generation_constraints

                generation_kwargs = generation_constraints(self, response_schema)
            elif constraints is not None:
                generation_kwargs = constraints()
            elif getattr(self, "inference_profile", "baseline") == "fast":
                raise ValueError("Fast inference requires an explicit response schema")
            else:
                generation_kwargs = {}
            timings["constraints_setup_s"] = time.perf_counter() - phase
            phase = time.perf_counter()
            inputs = inputs.to(self.model.device)
            if self.model.device.type == "cuda":
                self.torch.cuda.synchronize(self.model.device)
            timings["input_transfer_s"] = time.perf_counter() - phase
            token_timing = _TokenTiming()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    streamer=token_timing,
                    **generation_kwargs,
                )
            if self.model.device.type == "cuda":
                self.torch.cuda.synchronize(self.model.device)
            timings["generation_s"] = time.perf_counter() - token_timing.started
            timings["time_to_first_token_s"] = token_timing.first_token_s
            phase = time.perf_counter()
            continuation = generated[0, prompt_tokens:]
            raw = self.processor.decode(continuation, skip_special_tokens=True).strip()
            result.update(
                raw_response=raw,
                prompt_tokens=prompt_tokens,
                generated_tokens=int(continuation.shape[0]),
                finish_reason="length" if len(continuation) >= self.max_new_tokens else "stop",
            )
            if result["finish_reason"] == "length":
                raise ValueError("model response reached the output token limit")
            result["parsed_json"] = parse_json_response(raw)
            if getattr(self, "inference_profile", "baseline") == "fast":
                from jsonschema import Draft202012Validator

                Draft202012Validator(self._active_response_schema).validate(result["parsed_json"])
            result["success"] = True
            timings["decode_and_parse_s"] = time.perf_counter() - phase
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["latency_s"] = time.perf_counter() - started
        result["timings"] = timings
        return result
