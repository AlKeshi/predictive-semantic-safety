from __future__ import annotations

import time
from copy import copy

import numpy as np

VIDEO_PREPROCESSING_SCHEMA = "qwen3vl_explicit_per_frame_pixel_limit_v1"
SYSTEM_PROMPT = "You analyze physical videos for robot navigation. Return one valid JSON object only, with no markdown or thinking blocks. Infer from the visible frames. Report visible objects even when their future motion is uncertain."
FAST_INFERENCE_SCHEMA = "caw_fast_xgrammar_bounded_numbers_v3"


def prepare_video_inputs(
    processor, rgb, times, prompt, forecast_space, max_pixels, video_cache=None
):
    if video_cache is not None:
        if video_cache.processor is not processor.video_processor:
            raise ValueError("Video cache belongs to another processor")
        processor = copy(processor)
        processor.video_processor = video_cache
    relative_times = times - times[0]
    coordinate_instruction = (
        "Keep the camera fixed at that last observation when predicting future pixel locations."
        if forecast_space == "pixel"
        else "Predict world-frame metric displacements as specified below; camera motion does not move world coordinates."
    )
    prompt_text = (
        f"This RGB video is {rgb.shape[2]} pixels wide and {rgb.shape[1]} pixels high. Frame timestamps in seconds relative to the first frame are {relative_times.tolist()}. Forecast times dt are relative to the LAST observation. "
        + coordinate_instruction
        + "\n"
        + str(prompt)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt_text}]},
    ]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ticks = np.rint(relative_times * 1000000).astype(int).tolist()
    metadata = {
        "total_num_frames": ticks[-1] + 1,
        "fps": 1000000.0,
        "frames_indices": ticks,
        "duration": float(relative_times[-1]),
        "height": int(rgb.shape[1]),
        "width": int(rgb.shape[2]),
    }
    inputs = processor(
        text=[rendered],
        videos=[rgb],
        videos_kwargs={
            "video_metadata": [metadata],
            "do_sample_frames": False,
            "size": {
                "shortest_edge": len(rgb) * min(128 * 128, max_pixels),
                "longest_edge": len(rgb) * max_pixels,
            },
        },
        return_tensors="pt",
    )
    return (rendered, metadata, inputs)


class _TokenTiming:
    def __init__(self):
        self.prompt_seen = False
        self.first_token_s = None
        self.started = time.perf_counter()

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_token_s is None:
            self.first_token_s = time.perf_counter() - self.started

    def end(self):
        pass
