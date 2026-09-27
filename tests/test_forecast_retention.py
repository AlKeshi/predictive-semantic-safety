from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from pss.control import CodeWorldController, Forecast


@pytest.mark.parametrize("family", ["route", "ceiling"])
@pytest.mark.parametrize("contradicted", [False, True])
@pytest.mark.parametrize("outcome", ["rejected", "failed", "accepted"])
@pytest.mark.parametrize("matched", [False, True])
def test_runner_retains_only_observation_consistent_geometry(
    tmp_path, monkeypatch, family, contradicted, outcome, matched
):
    module = import_module(f"benchmarks.{family}_forecast")
    state = np.array([0.0, 0.0, 0.2, 0.0])
    controller = CodeWorldController()
    old = Forecast("block", 0, np.array([0, 0.5, 1.5, 3]), np.tile([3, 0.8], (4, 1)), 0.13)
    assert controller.update([old], 0, state)["accepted"]
    center = [1, 0] if contradicted else [3, 0.8]
    centers = (
        np.tile(center, (4, 1)) if outcome == "accepted" else [center, [0.2, 0], [-1, 0], [-2, 0]]
    )
    forecast = Forecast("block", 0.1, old.times + 0.1, np.asarray(centers), 0.13)
    detection = SimpleNamespace(center=np.r_[center, 0.3], radius=0.13)
    result = dict(
        success=outcome != "failed", error="inference_failed", latency_s=0, parsed_json={}
    )

    def transform(values):
        return values

    transform.next_origin_after = lambda now: now + 1
    transform.invalidate = lambda error: setattr(transform, "invalid_reason", str(error))
    episode = SimpleNamespace(
        now=0.1,
        state=state,
        controller=controller,
        transform=transform if matched else None,
        config=SimpleNamespace(
            method="code_world" if family == "route" else "cw", horizon=3, update_period=1
        ),
        predictor=SimpleNamespace(inference_profile="baseline", infer=lambda *a, **kw: result),
        demo_navigation=False,
        out=tmp_path,
        sensor_detections={"block": detection},
        objects=[{"id": "block"}],
        observation_frames=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        observation_times=[0, 0.1],
        track_history=[],
        observation=SimpleNamespace(cam_pose=np.eye(4)),
        model_updates=[],
        next_update=0.1,
        detection=detection,
        history=[],
        diagnostic={},
        frames=[],
        timestamps=[],
        depth=np.zeros((32, 32)),
        camera=SimpleNamespace(k=np.eye(3)),
        pose=np.eye(4),
        horizon=3,
        active=True,
        last_committed_forecast=old,
        proposals=[],
    )
    if family == "route":
        monkeypatch.setattr(module, "metric_trajectory_prompt", lambda *a, **kw: "synthetic")
        monkeypatch.setattr(module, "lift_metric_response", lambda *a, **kw: [forecast])
        module.RouteForecast.forecast(episode)
        proposal = episode.model_updates[-1]
    else:
        monkeypatch.setattr(
            module, "causal_facts", lambda *a: {"measured_world_velocity_m_s": [0, 0, 0]}
        )
        monkeypatch.setattr(module, "analysis_prompt", lambda *a: "synthetic")
        monkeypatch.setattr(module, "refinement_prompt", lambda *a: "synthetic")
        monkeypatch.setattr(
            module,
            "parse_event",
            lambda *a: {"evolution": "falling", "attachment_state": "detached"},
        )
        monkeypatch.setattr(module, "compile_event", lambda *a: (forecast, {}))
        module.CeilingForecast.forecast(episode)
        proposal = episode.proposals[-1]
    _, decision = controller.command(state, [0.3, 0], 0.1)
    usable = outcome == "accepted" or (not contradicted and not (matched and outcome == "failed"))
    assert decision["feasible"] == usable
    assert bool(decision["enforced_rows"]) == usable
    assert bool(proposal.get("accepted")) == (outcome == "accepted")
    if contradicted and outcome == "rejected":
        assert "contradicts" in controller.snapshot()["emergency_reason"]
