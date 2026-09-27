import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from pss.perception.code_world import Detection, detect_objects
from pss.perception.rgbd_lift import pixel_to_world
from pss.predictors.checkpoint import MODEL_ID, MODEL_REVISION, resolve_checkpoint
from pss.predictors.code_as_world import CodeAsWorldPredictor, parse_json_response, prepare_clip
from pss.predictors.fixture_events import compile_event, parse_event
from pss.predictors.physical_forecast import lift_metric_response, metric_trajectory_prompt

FENCE = chr(96) * 3


@pytest.mark.parametrize(
    "text",
    [
        '{"x":1}',
        '<think>reason</think>{"x":1}',
        FENCE + 'json\n{"x":1}\n' + FENCE,
        FENCE + 'json\n<think>reason</think>{"x":1}\n' + FENCE,
    ],
)
def test_complete_json_wrappers(text):
    assert parse_json_response(text) == {"x": 1}


@pytest.mark.parametrize(
    "text",
    [
        '{"x":1,"x":2}',
        '{"x":NaN}',
        '{"x":1e999}',
        '{"x":1',
        'prose {"x":1}',
        "[]",
        '{"x":1} tail',
        '<think>unfinished{"x":1}',
    ],
)
def test_invalid_json_is_rejected_without_repair(text):
    with pytest.raises(ValueError):
        parse_json_response(text)


def test_clip_keeps_exact_nonuniform_observation_timestamps():
    rgb = np.arange(20 * 32 * 48 * 3, dtype=np.uint8).reshape(20, 32, 48, 3)
    times = np.cumsum(np.linspace(0.01, 0.05, 20))
    selected, actual = prepare_clip(rgb, times, max_frames=12)
    indices = np.linspace(0, 19, 12, dtype=int)
    np.testing.assert_array_equal(selected, rgb[indices])
    np.testing.assert_array_equal(actual, times[indices])
    with pytest.raises(ValueError):
        prepare_clip(rgb, times[::-1])


def detection(name="block"):
    return Detection(name, np.array([16.0, 16.0]), np.array([1.0, 0.0, 1.0]), 1.0, 0.2, 64)


def test_metric_forecast_retains_measured_origin_and_total_displacements():
    objects = [{"id": "block"}]
    d = detection()
    rows = [[t, 0.1 * t, -0.05 * t, 0] for t in [0.5, 1.0, 1.5, 2.0, 2.5]]
    payload = {"objects": [{"id": "block", "trajectory": rows}]}
    f = lift_metric_response(payload, {"block": d}, objects, origin=3.0, horizon=2.5)[0]
    np.testing.assert_array_equal(f.times, [3, 3.5, 4, 4.5, 5, 5.5])
    np.testing.assert_allclose(
        f.centers, [[1 + 0.1 * t, -0.05 * t] for t in [0, 0.5, 1, 1.5, 2, 2.5]]
    )
    assert f.radii == d.radius
    payload["objects"][0]["trajectory"][0][0] = 0.6
    with pytest.raises(ValueError):
        lift_metric_response(payload, {"block": d}, objects, origin=3.0)


def test_metric_prompt_rejects_future_measurements_and_missing_inventory():
    d = detection()
    prompt = metric_trajectory_prompt(
        [{"id": "block"}], [(1.0, {"block": d}), (2.0, {"block": d})], origin=2.0
    )
    assert "WORLD METRES" in prompt and "[0.5, 1.0, 1.5, 2.0, 2.5]" in prompt
    with pytest.raises(ValueError):
        metric_trajectory_prompt([{"id": "block"}], [(3.0, {"block": d})], origin=2.0)
    with pytest.raises(ValueError):
        lift_metric_response({"objects": []}, {}, [{"id": "block"}], origin=2.0)


def test_release_hypotheses_are_enclosed_without_reducing_prediction_error():
    d = detection("ceiling_fixture")
    d.center = np.array([1.0, 0.0, 2.0])
    d.radius = 0.636
    event = parse_event(
        {
            "id": "ceiling_fixture",
            "attachment_state": "loose",
            "evolution": "free_fall",
            "release_interval_s": [1.0, 2.0],
        },
        8.0,
    )
    f, detail = compile_event(event, d, [0.2, 0, 0], 3.0)
    trajectories = np.asarray(detail["trajectories_xyz_m"])
    centers = trajectories[:, :, :2].mean(0)
    spread = np.linalg.norm(trajectories[:, :, :2] - centers, axis=2).max(0)
    np.testing.assert_array_equal(f.centers, centers)
    np.testing.assert_array_equal(f.radii, d.radius + spread)
    assert trajectories[:, :, 2].min() >= 0.05
    np.testing.assert_array_equal(f.times, 3.0 + np.asarray(detail["offsets_s"]))


def test_supported_fixture_has_complete_stationary_forecast():
    d = detection("ceiling_fixture")
    d.center[2] = 2.0
    event = parse_event(
        dict(
            id=d.object_id,
            attachment_state="secure",
            evolution="supported",
            release_interval_s=None,
        ),
        8.0,
    )
    forecast, detail = compile_event(event, d, [0, 0, 0], 0.75)
    assert len(forecast.times) == 81 and forecast.object_id == d.object_id
    np.testing.assert_array_equal(forecast.centers, np.tile(d.center[:2], (81, 1)))
    assert detail["evolution"] == "supported"


def test_rgbd_detection_and_pixel_lifting_use_measured_camera_geometry():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24, 0] = 255
    k = np.array([[50.0, 0, 15.5], [0, 50.0, 15.5], [0, 0, 1.0]])
    observation = SimpleNamespace(
        rgb=rgb, depth=np.full((32, 32), 2.0), cam_intrinsics=k, cam_pose=np.eye(4)
    )
    objects = [{"id": "block", "color": [1, 0, 0], "radius": 0.1, "shape": "box"}]
    found = detect_objects(observation, objects)["block"]
    np.testing.assert_allclose(found.center, [0, 0, 2.1], atol=1e-12)
    np.testing.assert_array_equal(pixel_to_world(15.5, 15.5, 2.0, k, np.eye(4)), [0, 0, 2.0])
    with pytest.raises(ValueError):
        pixel_to_world(15.5, 15.5, -1.0, k, np.eye(4))


def test_local_checkpoint_requires_pinned_content(tmp_path, monkeypatch):
    from pss.predictors import checkpoint

    content = b"known checkpoint content"
    (tmp_path / "model.safetensors").write_bytes(content)
    monkeypatch.setattr(
        checkpoint,
        "CHECKPOINT_FILES",
        {"model.safetensors": ("sha256", len(content), hashlib.sha256(content).hexdigest())},
    )
    path, identity = resolve_checkpoint(tmp_path)
    assert path == tmp_path and identity == {"model_id": MODEL_ID, "revision": MODEL_REVISION}
    (tmp_path / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        resolve_checkpoint(tmp_path)
    with pytest.raises(ValueError):
        resolve_checkpoint(tmp_path, revision="other")


class FakeInputs(dict):
    def to(self, _):
        return self


class FakeProcessor:
    def __init__(self):
        self.video_processor = SimpleNamespace(temporal_patch_size=2, patch_size=16)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **_):
        self.messages = messages
        return "rendered"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        count = len(kwargs["videos"][0])
        temporal = (count + 1) // 2
        return FakeInputs(
            input_ids=np.zeros((1, 20), dtype=int),
            video_grid_thw=np.array([[temporal, 2, 2]]),
            pixel_values_videos=np.zeros((temporal * 4, 3)),
        )

    def decode(self, *_, **__):
        return '{"objects":[{"id":"block"}]}'


def test_inference_preserves_video_budget_metadata_and_explicit_result():
    p = object.__new__(CodeAsWorldPredictor)
    p.max_frames, p.max_pixels, p.max_new_tokens = (16, 262144, 768)
    p.provenance = {}
    p.processor = FakeProcessor()
    p.model = SimpleNamespace(
        device=SimpleNamespace(type="cpu"), generate=lambda **_: np.zeros((1, 23), dtype=int)
    )
    p.torch = SimpleNamespace(inference_mode=nullcontext)
    frames = np.zeros((17, 32, 48, 3), dtype=np.uint8)
    times = np.cumsum(np.linspace(0.03, 0.1, 17))
    result = p.infer(frames, "task", times, forecast_space="metric")
    assert result["success"] and result["generated_tokens"] == 3
    assert result["forecast_origin_s"] == times[-1]
    assert (
        result["input_sha256"]
        == hashlib.sha256(prepare_clip(frames, times)[0].tobytes()).hexdigest()
    )
    kwargs = p.processor.kwargs["videos_kwargs"]
    assert kwargs["size"]["longest_edge"] == 16 * 262144
    assert not kwargs["do_sample_frames"]
    assert result["video_preprocessing"]["represented_frame_count"] == 16
