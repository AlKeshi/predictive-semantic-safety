from __future__ import annotations

from pathlib import Path

import numpy as np

from pss.io import save_json
from pss.predictors.fast_json import metric_response_schema
from pss.predictors.physical_forecast import lift_metric_response, metric_trajectory_prompt

from .demo_margin import pad_forecasts
from .route_geometry import retained_forecast_receipt


class RouteForecast:
    def forecast(self):
        detections = self.sensor_detections
        update = {
            "origin_time": self.now,
            "horizon_s": self.config.horizon,
            "detected": list(detections),
            "method": self.config.method,
            "latency_s": 0.0,
            "accepted": False,
        }
        missing_inventory = set(detections) != {item["id"] for item in self.objects}
        update["refresh_missing"] = missing_inventory
        try:
            if missing_inventory:
                if self.transform is not None:
                    self.transform([])
                update.update(
                    retained_forecast_receipt(self.controller, detections, self.objects, self.now)
                )
            elif self.config.method == "code_world":
                frames = np.asarray(self.observation_frames)
                timestamps = np.asarray(self.observation_times)
                infer_options = {}
                prompt = metric_trajectory_prompt(
                    self.objects,
                    list(self.track_history),
                    origin=float(timestamps[-1]),
                    horizon=self.config.horizon,
                    camera_pose=self.observation.cam_pose,
                )
                infer_options["forecast_space"] = "metric"
                if getattr(self.predictor, "inference_profile", "baseline") == "fast":
                    infer_options["response_schema"] = metric_response_schema(
                        [obj["id"] for obj in self.objects]
                    )
                clip_path = Path("inference_inputs") / f"clip_{len(self.model_updates):04d}.npz"
                (self.out / clip_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self.out / clip_path, frames=frames, timestamps=timestamps)
                update["input_clip"] = clip_path.as_posix()
                update["prompt"] = prompt
                update["forecast_space"] = "metric"
                result = self.predictor.infer(frames, prompt, timestamps, **infer_options)
                update["inference"] = result
                update["latency_s"] = result["latency_s"]
                if not result.get("success", False):
                    raise ValueError(result.get("error") or "Model inference failed")
                forecasts = lift_metric_response(
                    result.get("parsed_json"),
                    detections,
                    self.objects,
                    origin=float(timestamps[-1]),
                    horizon=self.config.horizon,
                    camera_pose=self.observation.cam_pose,
                )
            if not missing_inventory:
                if self.transform is not None:
                    forecasts = self.transform(forecasts)
                elif self.demo_navigation:
                    forecasts = pad_forecasts(forecasts, self.demo_margin)
                    update["heuristic_margin_m"] = self.demo_margin
                decision = self.controller.update(forecasts, self.now, self.state)
                update.update(decision)
                update["proposed_forecasts"] = [
                    {
                        "object_id": f.object_id,
                        "origin_time": f.origin_time,
                        "times": f.times,
                        "centers": f.centers,
                        "radii": f.radii,
                    }
                    for f in forecasts
                ]
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            update["error"] = str(error)
            if self.transform is not None:
                self.transform.invalidate(error)
            if self.transform is not None or missing_inventory:
                update.update(self.controller.invalidate(error))
            else:
                update.update(self.controller.update([], self.now, self.state))
        if not update["accepted"] and self.controller.snapshot()["emergency_reason"] is None:
            try:
                update["retention"] = retained_forecast_receipt(
                    self.controller, detections, self.objects, self.now
                )
            except ValueError as error:
                update.update(self.controller.invalidate(error))
        self.model_updates.append(update)
        if len(self.model_updates) == 1:
            np.savez_compressed(
                self.out / "observation_clip.npz",
                frames=np.asarray(self.observation_frames),
                timestamps=np.asarray(self.observation_times),
            )
        save_json(self.out / "updates.json", self.model_updates)
        self.next_update = (
            self.next_update + self.config.update_period
            if self.transform is None
            else self.transform.next_origin_after(self.now)
        )
