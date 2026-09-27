from __future__ import annotations

import hashlib
import importlib.metadata
import os
import time
from pathlib import Path

from .checkpoint import MODEL_ID, MODEL_REVISION, resolve_checkpoint
from .generation import TransformersGeneration
from .preprocessing import parse_json_response, prepare_clip
from .video_inputs import FAST_INFERENCE_SCHEMA, VIDEO_PREPROCESSING_SCHEMA, prepare_video_inputs


class CodeAsWorldPredictor(TransformersGeneration):
    def __init__(
        self,
        model_path: str | Path = MODEL_ID,
        *,
        max_new_tokens: int = 768,
        max_frames: int = 16,
        max_pixels: int = 262144,
        device: str = "cuda",
        revision: str = MODEL_REVISION,
        inference_profile: str | None = None,
    ) -> None:
        if max_new_tokens < 1 or max_frames < 2 or max_pixels < 1024:
            raise ValueError("invalid inference limits")
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if str(device).startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError("The requested CUDA device is unavailable")
        self.torch = torch
        self.inference_profile = inference_profile or os.environ.get(
            "PSS_INFERENCE_PROFILE", "baseline"
        )
        if self.inference_profile not in {"baseline", "fast"}:
            raise ValueError("inference_profile must be baseline or fast")
        self.max_new_tokens = int(max_new_tokens)
        self.max_frames = int(max_frames)
        self.max_pixels = int(max_pixels)
        self.model_path, identity = resolve_checkpoint(model_path, revision)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        if (
            type(self.processor).__name__ != "Qwen3VLProcessor"
            or type(self.processor.video_processor).__name__ != "Qwen3VLVideoProcessor"
        ):
            raise ValueError("unsupported video processor for the frozen pixel-limit contract")
        started = time.perf_counter()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            device_map=str(device),
            attn_implementation="sdpa",
        ).eval()
        self.model.config.use_cache = True
        self.model.config.text_config.use_cache = True
        self.load_latency_s = time.perf_counter() - started
        self.provenance = {
            "model_id": identity["model_id"],
            "revision": identity["revision"],
            "model_path": str(self.model_path),
            "config_sha256": hashlib.sha256(
                (self.model_path / "config.json").read_bytes()
            ).hexdigest(),
            "backend": "transformers",
            "transformers_version": importlib.metadata.version("transformers"),
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(self.model.device)
            if self.model.device.type == "cuda"
            else None,
            "load_latency_s": self.load_latency_s,
            "max_frames": self.max_frames,
            "max_pixels": self.max_pixels,
            "video_preprocessing_schema": VIDEO_PREPROCESSING_SCHEMA,
            "processor_class": type(self.processor).__name__,
            "video_processor_class": type(self.processor.video_processor).__name__,
            "processor_config_sha256": hashlib.sha256(
                (self.model_path / "processor_config.json").read_bytes()
            ).hexdigest(),
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "calibrated": False,
            "coverage_guarantee": None,
        }
        if self.inference_profile == "fast":
            self.enable_fast_inference()

    def enable_fast_inference(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5

        names = (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "FusedRMSNormGated",
        )
        available = {name: getattr(modeling_qwen3_5, name, None) is not None for name in names}
        if self.model.device.type != "cuda" or not all(available.values()):
            raise RuntimeError(
                f"Fast inference requires CUDA and all acceleration kernels: {available}"
            )
        self.inference_profile = "fast"
        from .video_cache import CACHE_SCHEMA, VideoPreprocessingCache

        self.video_cache = VideoPreprocessingCache(self.processor.video_processor)
        self.provenance.update(
            inference_profile="fast",
            video_cache_schema=CACHE_SCHEMA,
            inference_schema=FAST_INFERENCE_SCHEMA,
            acceleration_kernels=available,
            acceleration_versions={
                name: importlib.metadata.version(name)
                for name in (
                    "flash-linear-attention",
                    "fla-core",
                    "causal-conv1d",
                    "triton",
                    "xgrammar",
                    "jsonschema",
                )
            },
            numeric_spelling_limits={
                "integer_digits": 8,
                "fractional_digits": 6,
                "exponent_digits": 2,
            },
            calibration_compatible=False,
        )


def create_predictor(*args, backend=None, queue=None, **kwargs):
    backend = backend or (
        "vllm-router"
        if queue is not None
        else os.environ.get("PSS_INFERENCE_BACKEND", "transformers")
    )
    if queue is not None and backend != "vllm-router":
        raise ValueError("queue requires the vllm-router backend")
    if backend == "transformers":
        return CodeAsWorldPredictor(*args, **kwargs)
    if backend == "vllm":
        from .vllm_backend import VLLMCodeAsWorldPredictor

        return VLLMCodeAsWorldPredictor(*args, **kwargs)
    if backend == "vllm-router":
        from .inference_router import RoutedCodeAsWorldPredictor

        return RoutedCodeAsWorldPredictor(*args, queue=queue, **kwargs)
    raise ValueError("PSS_INFERENCE_BACKEND must be transformers, vllm or vllm-router")


__all__ = [
    "CodeAsWorldPredictor",
    "create_predictor",
    "FAST_INFERENCE_SCHEMA",
    "VIDEO_PREPROCESSING_SCHEMA",
    "prepare_video_inputs",
    "parse_json_response",
    "prepare_clip",
]
