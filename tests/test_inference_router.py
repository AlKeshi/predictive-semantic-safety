import json
import threading
import time

import numpy as np
import pytest

from pss.predictors.inference_router import RoutedCodeAsWorldPredictor, atomic_json


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv("PSS_INFERENCE_QUEUE", str(tmp_path))
    ready = dict(
        status="ready",
        worker_id="worker-A",
        heartbeat_unix=time.time(),
        model=dict(model_path=str(tmp_path / "checkpoint")),
    )
    atomic_json(tmp_path / "ready.json", ready)
    return (tmp_path, ready)


def test_stopped_worker_rejects_before_submitting(queue):
    path, ready = queue
    ready["heartbeat_unix"] -= 30
    atomic_json(path / "ready.json", ready)
    with pytest.raises(RuntimeError, match="not ready"):
        RoutedCodeAsWorldPredictor()
    assert not (path / "requests").exists()


def test_worker_restart_fails_request_and_preserves_inputs(queue):
    path, ready = queue
    predictor = RoutedCodeAsWorldPredictor()
    ready["worker_id"] = "worker-B"
    atomic_json(path / "ready.json", ready)
    result = predictor.infer(
        np.zeros((2, 32, 32, 3), dtype=np.uint8),
        "visible evidence",
        [0, 1],
        response_schema={"type": "object"},
    )
    assert not result["success"]
    assert "changed" in result["error"]["message"]
    assert len(list((path / "requests").glob("*/input.npz"))) == 1


def test_distinct_requests_keep_full_video_timestamps_and_limits(queue):
    path, _ = queue
    seen = []
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            for request in path.glob("requests/*/request.json"):
                response = request.parent / "response.json"
                if response.exists():
                    continue
                payload = json.loads(request.read_text())
                with np.load(request.parent / "input.npz", allow_pickle=False) as clip:
                    seen.append((clip["rgb"].copy(), clip["times"].copy(), payload))
                atomic_json(
                    response,
                    dict(
                        success=True, parsed_json={"value": len(seen)}, error=None, latency_s=0.001
                    ),
                )
            stop.wait(0.005)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        predictor = RoutedCodeAsWorldPredictor(max_frames=12, max_new_tokens=3000)
        results = [
            predictor.infer(
                np.full((3, 32, 32, 3), i, dtype=np.uint8),
                "evidence",
                [0, 0.4, 1],
                response_schema={"type": "object"},
            )
            for i in (1, 2)
        ]
    finally:
        stop.set()
        thread.join()
    assert all((item["success"] for item in results))
    assert results[0]["router"]["request_id"] != results[1]["router"]["request_id"]
    assert [item[0].mean() for item in seen] == [1, 2]
    for _, times, payload in seen:
        assert times.tolist() == [0, 0.4, 1]
        assert payload["limits"] == dict(max_frames=12, max_new_tokens=3000, max_pixels=262144)
