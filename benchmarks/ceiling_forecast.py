from __future__ import annotations

import numpy as np

from pss.io import save_json
from pss.predictors.fixture_events import (
    analysis_prompt,
    causal_facts,
    compile_event,
    parse_event,
    refinement_prompt,
)

from .demo_margin import pad_forecasts
from .route_geometry import retained_forecast_receipt


class CeilingForecast:
    def forecast(self):
        self.next_forecast = (
            self.now + 1.0 if self.transform is None else self.transform.next_origin_after(self.now)
        )
        proposal = {"origin": self.now}
        if self.detection is None:
            if self.transform is not None:
                self.transform.invalidate("Missing required RGB-D prediction")
                self.active = True
                self.controller.invalidate(self.transform.invalid_reason)
            proposal["status"] = "missing_current_rgbd; retain committed finite support"
        else:
            try:
                facts = causal_facts(self.history, self.diagnostic)
                if self.config.method == "cw":
                    prompt = analysis_prompt(facts)
                    (self.out / f"prompt_{self.now:06.2f}.txt").write_text(prompt)
                    np.savez_compressed(
                        self.out / f"inputs_{self.now:06.2f}.npz",
                        rgb=np.asarray(self.frames),
                        times=np.asarray(self.timestamps),
                        depth=self.depth,
                        intrinsics=self.camera.k,
                        camera_to_world=self.pose,
                    )
                    self.predictor.schema_stage = "analysis"
                    initial = self.predictor.infer(
                        self.frames, prompt, self.timestamps, forecast_space="metric"
                    )
                    save_json(self.out / f"analysis_{self.now:06.2f}.json", initial)
                    if not initial["success"]:
                        raise ValueError(initial["error"])
                    refined_prompt = refinement_prompt(facts, initial["parsed_json"])
                    (self.out / f"event_prompt_{self.now:06.2f}.txt").write_text(refined_prompt)
                    self.predictor.schema_stage = "event"
                    result = self.predictor.infer(
                        self.frames, refined_prompt, self.timestamps, forecast_space="metric"
                    )
                    save_json(self.out / f"model_{self.now:06.2f}.json", result)
                    if not result["success"]:
                        raise ValueError(result["error"])
                    event = parse_event(result["parsed_json"], self.horizon)
                    forecast, rollout = compile_event(
                        event,
                        self.detection,
                        facts["measured_world_velocity_m_s"],
                        self.now,
                        self.horizon,
                    )
                    save_json(self.out / f"physical_rollout_{self.now:06.2f}.json", rollout)
                if self.demo_navigation and event["evolution"] == "supported":
                    forecast = None
                self.secure_observation = forecast is None and (not self.active)
                if self.secure_observation:
                    self.last_secure = self.now
                    proposal.update(status="visually_secure_overhead", accepted=False)
                elif forecast is None:
                    proposal.update(
                        status="supported_label_does_not_clear_active_family", accepted=False
                    )
                else:
                    if not self.active:
                        self.first_hazard = self.now
                    self.active = True
                    if self.transform is not None:
                        forecast = self.transform([forecast])[0]
                    elif self.demo_navigation:
                        forecast = pad_forecasts([forecast], self.demo_margin)[0]
                        proposal["heuristic_margin_m"] = self.demo_margin
                    proposal.update(self.controller.update([forecast], self.now, self.state))
                    if proposal.get("accepted"):
                        self.last_valid_support = self.now + self.horizon
                        self.last_committed_forecast = forecast
                proposal["event"] = event
                proposal["attachment_state"] = event["attachment_state"]
                proposal["motion_description"] = event.get("motion_description")
            except Exception as exc:
                if self.transform is not None:
                    self.transform.invalidate(exc)
                    self.active = True
                    self.controller.invalidate(self.transform.invalid_reason)
                proposal["status"] = f"forecast_error:{exc}"
                self.secure_observation = False
        if (
            not proposal.get("accepted")
            and self.last_committed_forecast is not None
            and self.controller.snapshot()["emergency_reason"] is None
        ):
            name = self.last_committed_forecast.object_id
            detections = {} if self.detection is None else {name: self.detection}
            try:
                proposal["retention"] = retained_forecast_receipt(
                    self.controller, detections, [{"id": name}], self.now
                )
            except ValueError as error:
                proposal.update(self.controller.invalidate(error))
        self.proposals.append(proposal)
        save_json(self.out / "proposals.json", self.proposals)
