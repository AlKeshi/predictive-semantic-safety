from __future__ import annotations

import time
from typing import Any

import numpy as np

from pss.vla.omnivla_support import (
    _official_waypoint_pd_command,
    _require_git_revision,
    _validate_checkpoint,
)

from .omnivla_loading import OmniVLALoading
from .omnivla_types import (
    OMNIVLA_ACTION_DIM,
    OMNIVLA_ACTION_HORIZON,
    OMNIVLA_ACTION_PERIOD_S,
    OMNIVLA_CHECKPOINT_ID,
    OMNIVLA_CHECKPOINT_REVISION,
    OMNIVLA_GOAL_DISTANCE_CAP_M,
    OMNIVLA_GOAL_PROMPT,
    OMNIVLA_LANGUAGE_POSE_MODALITY_ID,
    OMNIVLA_SOURCE_REPOSITORY,
    OMNIVLA_SOURCE_REVISION,
    OMNIVLA_WAYPOINT_INDEX,
    OMNIVLA_WAYPOINT_SPACING_M,
    OmniVLAInference,
    OmniVLAPolicyConfig,
)


class OmniVLAPolicy(OmniVLALoading):
    def __init__(self, config: OmniVLAPolicyConfig):
        self.config = config
        self._source_revision = _require_git_revision(
            config.source_dir, OMNIVLA_SOURCE_REVISION, label="OmniVLA source"
        )
        self._checkpoint_revision = OMNIVLA_CHECKPOINT_REVISION
        self._checkpoint_hashes = _validate_checkpoint(config.checkpoint_dir)
        self._load_model()
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.RandomState(0)
        self._inferences: list[OmniVLAInference] = []

    @property
    def last_inference(self) -> OmniVLAInference:
        if not self._inferences:
            raise RuntimeError("OmniVLA has not produced an inference yet")
        return self._inferences[-1]

    def command(
        self, rgb: np.ndarray, *, body_goal_delta: np.ndarray, body_goal_yaw: float = 0.0
    ) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("OmniVLA RGB must be a uint8 H x W x 3 array")
        goal = np.asarray(body_goal_delta, dtype=float)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("body_goal_delta must be a finite length-2 vector")
        if not np.isfinite(body_goal_yaw):
            raise ValueError("body_goal_yaw must be finite")
        radius = float(np.linalg.norm(goal))
        if radius > OMNIVLA_GOAL_DISTANCE_CAP_M:
            goal = goal * (OMNIVLA_GOAL_DISTANCE_CAP_M / radius)
        forward, left = goal
        goal_pose = np.array(
            [
                forward / OMNIVLA_WAYPOINT_SPACING_M,
                left / OMNIVLA_WAYPOINT_SPACING_M,
                np.cos(body_goal_yaw),
                np.sin(body_goal_yaw),
            ],
            dtype=np.float32,
        )
        started = time.perf_counter()
        batch = self._make_batch(rgb, goal_pose, OMNIVLA_GOAL_PROMPT)
        waypoints = self._forward(batch)
        normalized = np.asarray(waypoints, dtype=np.float32)
        if normalized.shape != (OMNIVLA_ACTION_HORIZON, OMNIVLA_ACTION_DIM):
            raise ValueError(f"OmniVLA returned invalid waypoint shape {normalized.shape}")
        metric = normalized.copy()
        metric[:, :2] *= OMNIVLA_WAYPOINT_SPACING_M
        selected = metric[OMNIVLA_WAYPOINT_INDEX].copy()
        command = _official_waypoint_pd_command(selected)
        latency = float(time.perf_counter() - started)
        inference = OmniVLAInference(
            inference_index=len(self._inferences),
            goal_pose=goal_pose.copy(),
            normalized_waypoints=normalized.copy(),
            metric_waypoints=metric.copy(),
            selected_waypoint=selected,
            command=command.copy(),
            wall_latency_s=latency,
        )
        self._inferences.append(inference)
        return command.copy()

    def _make_batch(
        self, rgb: np.ndarray, goal_pose: np.ndarray, instruction: str
    ) -> dict[str, Any]:
        from PIL import Image

        torch = self._torch
        dummy_actions = self._rng.random_sample((OMNIVLA_ACTION_HORIZON, OMNIVLA_ACTION_DIM))
        current_action = dummy_actions[0]
        future_actions = dummy_actions[1:]
        action_string = self._action_tokenizer(current_action) + "".join(
            self._action_tokenizer(future_actions)
        )
        prompt_builder = self._prompt_builder_class("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {instruction}?")
        prompt_builder.add_turn("gpt", action_string)
        input_ids = torch.tensor(
            self._processor.tokenizer(
                prompt_builder.get_prompt(), add_special_tokens=True
            ).input_ids
        ).unsqueeze(0)
        labels = input_ids.clone()
        labels[:, : -(len(action_string) + 1)] = -100
        attention_mask = input_ids.ne(self._processor.tokenizer.pad_token_id)
        current_image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
        goal_image = Image.new("RGB", current_image.size, color=(0, 0, 0))
        transform = self._processor.image_processor.apply_transform
        current_pixels = transform(current_image)
        goal_pixels = transform(goal_image)
        pixel_values = torch.cat((current_pixels.unsqueeze(0), goal_pixels.unsqueeze(0)), dim=1)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "goal_pose": torch.from_numpy(goal_pose.copy()).unsqueeze(0),
        }

    def _forward(self, batch: dict[str, Any]) -> np.ndarray:
        torch = self._torch
        modality = torch.tensor(
            [OMNIVLA_LANGUAGE_POSE_MODALITY_ID], dtype=torch.bfloat16, device=self._device
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self._vla(
                input_ids=batch["input_ids"].to(self._device),
                attention_mask=batch["attention_mask"].to(self._device),
                pixel_values=batch["pixel_values"].to(dtype=torch.bfloat16, device=self._device),
                modality_id=modality,
                labels=batch["labels"].to(self._device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(dtype=torch.bfloat16, device=self._device),
                proprio_projector=self._pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )
            token_ids = batch["labels"][:, 1:].to(self._device)
            action_mask = self._get_current_action_mask(token_ids) | self._get_next_actions_mask(
                token_ids
            )
            text_hidden = output.hidden_states[-1][:, self._num_patches : -1]
            selected = text_hidden[action_mask]
            expected = OMNIVLA_ACTION_HORIZON * OMNIVLA_ACTION_DIM
            if selected.shape[0] != expected:
                raise ValueError(
                    f"OmniVLA action-token mask selected {selected.shape[0]} states, expected {expected}"
                )
            action_hidden = selected.reshape(1, expected, -1).to(torch.bfloat16)
            predicted = self._action_head.predict_action(action_hidden, modality)
        return predicted[0].float().cpu().numpy()


__all__ = [
    "OMNIVLA_ACTION_PERIOD_S",
    "OMNIVLA_CHECKPOINT_ID",
    "OMNIVLA_CHECKPOINT_REVISION",
    "OMNIVLA_SOURCE_REPOSITORY",
    "OMNIVLA_SOURCE_REVISION",
    "OmniVLAInference",
    "OmniVLAPolicy",
    "OmniVLAPolicyConfig",
]
