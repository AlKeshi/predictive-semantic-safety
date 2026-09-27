from __future__ import annotations

import sys
from typing import Any

import numpy as np

from pss.vla.omnivla_support import _load_state_dict, _safe_register

from .omnivla_types import (
    OMNIVLA_ACTION_DIM,
    OMNIVLA_ACTION_HORIZON,
    OMNIVLA_CHECKPOINT_ID,
    OMNIVLA_CHECKPOINT_STEP,
    OMNIVLA_GOAL_PROMPT,
    OMNIVLA_LANGUAGE_POSE_MODALITY_ID,
    OMNIVLA_POSE_DIM,
    OMNIVLA_SOURCE_REPOSITORY,
    OMNIVLA_WAYPOINT_INDEX,
    OMNIVLA_WAYPOINT_SPACING_M,
)


class OmniVLALoading:
    def _load_model(self) -> None:
        source = str(self.config.source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)
        import torch
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.models.projectors import ProprioProjector
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoProcessor,
        )

        if (NUM_ACTIONS_CHUNK, ACTION_DIM, POSE_DIM) != (
            OMNIVLA_ACTION_HORIZON,
            OMNIVLA_ACTION_DIM,
            OMNIVLA_POSE_DIM,
        ):
            raise ValueError("Official OmniVLA constants do not match the frozen PSS contract")
        device = torch.device(self.config.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The official 8B BF16 OmniVLA checkpoint requires a CUDA device")
        _safe_register(AutoConfig, "openvla", OpenVLAConfig)
        _safe_register(AutoImageProcessor, OpenVLAConfig, PrismaticImageProcessor)
        _safe_register(AutoProcessor, OpenVLAConfig, PrismaticProcessor)
        _safe_register(AutoModelForVision2Seq, OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
        checkpoint = str(self.config.checkpoint_dir)
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
        ).to(device)
        vla.vision_backbone.set_num_images_in_input(2)
        vla.to(dtype=torch.bfloat16, device=device).eval()
        pose_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=OMNIVLA_POSE_DIM)
        pose_projector.load_state_dict(
            _load_state_dict(
                self.config.checkpoint_dir
                / f"proprio_projector--{OMNIVLA_CHECKPOINT_STEP}_checkpoint.pt",
                torch,
            )
        )
        pose_projector.to(device=device).eval()
        action_head = L1RegressionActionHead_idcat(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=OMNIVLA_ACTION_DIM
        )
        action_head.load_state_dict(
            _load_state_dict(
                self.config.checkpoint_dir
                / f"action_head--{OMNIVLA_CHECKPOINT_STEP}_checkpoint.pt",
                torch,
            )
        )
        action_head.to(dtype=torch.bfloat16, device=device).eval()
        self._torch = torch
        self._device = device
        self._processor = processor
        self._vla = vla
        self._pose_projector = pose_projector
        self._action_head = action_head
        self._action_tokenizer = ActionTokenizer(processor.tokenizer)
        self._prompt_builder_class = PurePromptBuilder
        self._get_current_action_mask = get_current_action_mask
        self._get_next_actions_mask = get_next_actions_mask
        self._num_patches = (
            vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
            + 1
        )

    def diagnostics(self) -> dict[str, Any]:
        latencies = [item.wall_latency_s for item in self._inferences]
        return {
            "schema": "pss-official-omnivla-policy-v1",
            "method": "omnivla",
            "official_source_repository": OMNIVLA_SOURCE_REPOSITORY,
            "official_source_revision": self._source_revision,
            "official_checkpoint_id": OMNIVLA_CHECKPOINT_ID,
            "official_checkpoint_revision": self._checkpoint_revision,
            "official_checkpoint_step": OMNIVLA_CHECKPOINT_STEP,
            "checkpoint_modified": False,
            "checkpoint_file_sha256": dict(self._checkpoint_hashes),
            "config": self.config.report(),
            "goal_modality": "language_and_2d_pose",
            "modality_id": OMNIVLA_LANGUAGE_POSE_MODALITY_ID,
            "language_instruction": OMNIVLA_GOAL_PROMPT,
            "waypoint_index": OMNIVLA_WAYPOINT_INDEX,
            "waypoint_spacing_m": OMNIVLA_WAYPOINT_SPACING_M,
            "dummy_action_rng": "seeded numpy RandomState.random_sample (deterministic equivalent of released np.random.rand placeholder generation)",
            "input_contract": {
                "ego_rgb": True,
                "body_goal_delta": True,
                "language": True,
                "goal_image": False,
                "pss_prediction": False,
                "calibration_record": False,
                "hazard_tube": False,
                "cbf_state": False,
                "simulator_stack_state": False,
                "future_frames": False,
                "visible_goal_beacon": False,
            },
            "command_decoder": "official waypoint[4] PD and coupled 0.3 m/s, 0.3 rad/s limits",
            "inference_count": len(self._inferences),
            "mean_inference_wall_latency_s": float(np.mean(latencies)) if latencies else None,
            "maximum_inference_wall_latency_s": max(latencies, default=None),
            "inferences": [item.summary() for item in self._inferences],
        }
