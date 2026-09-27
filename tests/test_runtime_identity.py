import builtins
import json
import time

import pytest

from benchmarks.identity import describe
from pss import load_case


def test_router_identity_uses_worker_environment(tmp_path, monkeypatch):
    from pss.predictors import checkpoint

    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"identity-fixture")
    monkeypatch.setattr(checkpoint, "CHECKPOINT_FILES", {"model.safetensors": None})
    state = dict(
        status="ready",
        heartbeat_unix=time.time(),
        model=dict(
            model_path=str(model),
            acceleration_versions={"vllm": "worker-version"},
            python="3.11.0",
            accelerator={"name": "worker-gpu"},
        ),
    )
    (tmp_path / "ready.json").write_text(json.dumps(state))
    monkeypatch.setenv("PSS_INFERENCE_BACKEND", "vllm-router")
    monkeypatch.setenv("PSS_INFERENCE_QUEUE", str(tmp_path))
    original = builtins.__import__

    def no_local_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("The simulator client must not import Torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_local_torch)
    result = describe("stack", model, case=load_case("stack_00_r00"))
    assert result["accelerator"] == {"name": "worker-gpu"}
    assert result["dependencies"]["resident_inference"] == {"vllm": "worker-version"}
    assert "torch" not in result["dependencies"]
    for name in ("stack_stage", "dominos_stage", "stack_primitives"):
        assert f"datagen/assets/{name}.json" in result["source_files"]
    with pytest.raises(ValueError, match="checkpoint"):
        describe("stack", tmp_path / "different-model", case=load_case("stack_00_r00"))
