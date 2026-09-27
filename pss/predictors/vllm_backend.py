import hashlib
import importlib.metadata
import json
import platform
import time

from .checkpoint import MODEL_ID, MODEL_REVISION, resolve_checkpoint
from .code_as_world import FAST_INFERENCE_SCHEMA, VIDEO_PREPROCESSING_SCHEMA, prepare_video_inputs
from .fast_json import bounded_grammar
from .preprocessing import parse_json_response, prepare_clip
from .video_cache import CACHE_SCHEMA, VideoPreprocessingCache


def video_transport_prompt(rendered):
    marker = "<|vision_start|><|video_pad|><|vision_end|>"
    if rendered.count(marker) != 1:
        raise ValueError("Expected exactly one frozen video placeholder")
    return rendered.replace(marker, "<|vision_start|>" + marker + "<|vision_end|>", 1)


class VLLMCodeAsWorldPredictor:
    def __init__(
        self,
        model_path=MODEL_ID,
        *,
        max_new_tokens=768,
        max_frames=16,
        max_pixels=262144,
        device="cuda",
        revision=MODEL_REVISION,
        gpu_memory_utilization=0.5,
    ):
        import torch
        from transformers import AutoProcessor
        from vllm import LLM

        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("The vLLM backend requires CUDA")
        if max_new_tokens < 1 or max_frames < 2 or max_pixels < 1024:
            raise ValueError("Invalid inference limits")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("GPU memory utilization must be between zero and one")
        self.max_new_tokens, self.max_frames, self.max_pixels = (
            max_new_tokens,
            max_frames,
            max_pixels,
        )
        self.inference_profile = "fast"
        self.model_path, identity = resolve_checkpoint(model_path, revision)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        self.video_cache = VideoPreprocessingCache(self.processor.video_processor)
        started = time.perf_counter()
        self.engine = LLM(
            model=str(self.model_path),
            dtype="bfloat16",
            trust_remote_code=False,
            max_model_len=8192,
            max_num_seqs=1,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"video": {"count": 1, "num_frames": 16}, "image": 0},
            mm_processor_kwargs={
                "do_sample_frames": False,
                "size": {"shortest_edge": 16 * 16384, "longest_edge": 16 * max_pixels},
            },
            enable_prefix_caching=True,
            generation_config="vllm",
            seed=0,
            structured_outputs_config={"backend": "xgrammar"},
        )
        self.provenance = dict(
            identity,
            model_path=str(self.model_path),
            backend="vllm",
            load_latency_s=time.perf_counter() - started,
            config_sha256=hashlib.sha256(
                (self.model_path / "config.json").read_bytes()
            ).hexdigest(),
            max_frames=max_frames,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            dtype="bfloat16",
            gpu=torch.cuda.get_device_name(),
            gpu_memory_utilization=gpu_memory_utilization,
            python=platform.python_version(),
            accelerator={
                "name": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
            inference_profile="fast",
            inference_schema=FAST_INFERENCE_SCHEMA,
            video_preprocessing_schema=VIDEO_PREPROCESSING_SCHEMA,
            video_cache_schema=CACHE_SCHEMA,
            acceleration_versions={
                k: importlib.metadata.version(k)
                for k in [
                    "vllm",
                    "torch",
                    "transformers",
                    "xgrammar",
                    "numpy",
                    "Pillow",
                    "jsonschema",
                ]
            },
            calibration_compatible=False,
            calibrated=False,
            coverage_guarantee=None,
            do_sample=False,
            prefix_caching=True,
            video_transport_schema="qwen3vl_hf_outer_video_wrappers_v1",
        )

    def infer(self, frames, prompt, timestamps, *, forecast_space="pixel", response_schema=None):
        from jsonschema import Draft202012Validator
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        started = time.perf_counter()
        result = dict(
            success=False,
            raw_response="",
            parsed_json=None,
            error=None,
            latency_s=0.0,
            provenance=dict(self.provenance),
            forecast_space=forecast_space,
        )
        try:
            if forecast_space not in {"pixel", "metric"} or not str(prompt).strip():
                raise ValueError("Invalid forecast space or empty prompt")
            rgb, times = prepare_clip(frames, timestamps, self.max_frames)
            result.update(
                frame_timestamps_s=times.tolist(),
                forecast_origin_s=float(times[-1]),
                input_shape=list(rgb.shape),
                input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                prompt_sha256=hashlib.sha256(str(prompt).encode()).hexdigest(),
            )
            rendered, metadata, expected = prepare_video_inputs(
                self.processor,
                rgb,
                times,
                prompt,
                forecast_space,
                self.max_pixels,
                video_cache=self.video_cache,
            )
            result["video_cache_hit"] = self.video_cache.last_hit
            grid = expected["video_grid_thw"].tolist()
            temporal = self.processor.video_processor.temporal_patch_size
            patch = self.processor.video_processor.patch_size
            _, h, w = grid[0]
            if (
                h * w * patch * patch > self.max_pixels
                or grid[0][0] * temporal != (len(rgb) + temporal - 1) // temporal * temporal
            ):
                raise ValueError("Processed video violates frozen frame/pixel budget")
            expected_ids = expected["input_ids"][0].tolist()
            if len(expected_ids) + self.max_new_tokens > 8192:
                raise ValueError("Video and prompt exceed the 8192-token inference budget")
            if response_schema is None and hasattr(self, "schema_stage"):
                from .structured_json import ANALYSIS, EVENT

                response_schema = ANALYSIS if self.schema_stage == "analysis" else EVENT
            if response_schema is None:
                raise ValueError("Fast inference requires an explicit response schema")
            grammar = bounded_grammar(json.dumps(response_schema))
            request = {
                "prompt": video_transport_prompt(rendered),
                "multi_modal_data": {"video": [(rgb, metadata)]},
                "mm_processor_kwargs": {
                    "do_sample_frames": False,
                    "size": {
                        "shortest_edge": len(rgb) * min(16384, self.max_pixels),
                        "longest_edge": len(rgb) * self.max_pixels,
                    },
                },
            }
            params = SamplingParams(
                temperature=0,
                max_tokens=self.max_new_tokens,
                structured_outputs=StructuredOutputsParams(grammar=grammar),
            )
            prepared = time.perf_counter()
            answer = self.engine.generate([request], params, use_tqdm=False)[0]
            generation_end = time.perf_counter()
            prediction = answer.outputs[0]
            actual_ids = list(answer.prompt_token_ids)
            result.update(
                raw_response=prediction.text.strip(),
                generated_tokens=len(prediction.token_ids),
                prompt_tokens=len(answer.prompt_token_ids),
                finish_reason=prediction.finish_reason,
                input_token_ids_match=actual_ids == expected_ids,
                video_preprocessing={
                    "schema": VIDEO_PREPROCESSING_SCHEMA,
                    "video_grid_thw": grid,
                    "observed_frame_count": len(rgb),
                    "processed_pixels_per_frame": h * w * patch * patch,
                },
            )
            result["timings"] = dict(
                prepare_and_process_s=prepared - started, generation_s=generation_end - prepared
            )
            if answer.metrics is not None:
                result["timings"]["time_to_first_token_s"] = (
                    answer.metrics.first_token_time - answer.metrics.arrival_time
                )
            if not result["input_token_ids_match"]:
                result["input_token_mismatch"] = {
                    "expected": expected_ids,
                    "actual": actual_ids,
                    "first_difference": next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(expected_ids, actual_ids, strict=False))
                            if a != b
                        ),
                        min(len(expected_ids), len(actual_ids)),
                    ),
                }
                raise ValueError(
                    "vLLM input tokens differ from the frozen HF video/prompt processing"
                )
            if prediction.finish_reason != "stop":
                raise ValueError(f"Model generation did not finish: {prediction.finish_reason}")
            result["parsed_json"] = parse_json_response(result["raw_response"])
            Draft202012Validator(response_schema).validate(result["parsed_json"])
            result["success"] = True
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["latency_s"] = time.perf_counter() - started
        return result
