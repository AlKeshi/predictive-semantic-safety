from dataclasses import asdict

import numpy as np
from PIL import Image

from pss.io import save_json
from pss.predictors.fast_json import metric_response_schema
from pss.predictors.physical_forecast import lift_metric_response, metric_trajectory_prompt


class Forecast:
    def forecast(self):
        folder = self.out / "inferences" / f"{len(self.updates):03d}"
        folder.mkdir(parents=True)
        indices = np.linspace(0, len(self.frames) - 1, min(8, len(self.frames)), dtype=int)
        rgb = np.asarray(self.frames)[indices]
        stamp = np.asarray(self.times)[indices]
        np.savez_compressed(
            folder / "observation.npz",
            frames=rgb,
            timestamps=stamp,
            depth=np.asarray(self.depths)[indices],
            cam_pose=np.asarray(self.poses)[indices],
            intrinsics=self.observation.cam_intrinsics,
        )
        Image.fromarray(rgb[-1]).save(folder / "last.png")
        save_json(
            folder / "tracks.json",
            [
                dict(time=t, detections={n: asdict(d) for n, d in ds.items()})
                for t, ds in self.tracks
            ],
        )
        needed = [spec for spec in self.objects if spec["id"] not in self.ground.grounded]
        needed_names = {spec["id"] for spec in needed}
        semantic_detections = {n: d for n, d in self.detections.items() if n in needed_names}
        semantic_tracks = [
            (t, {n: d for n, d in ds.items() if n in needed_names}) for t, ds in self.tracks
        ]
        update = dict(
            time=self.now,
            origin=float(stamp[-1]),
            visible_ids=sorted(self.detections),
            required_semantic_ids=sorted(needed_names),
            grounded_ids=sorted(self.ground.grounded),
        )
        try:
            if set(semantic_detections) != needed_names:
                raise ValueError(
                    "incomplete airborne RGB-D inventory: "
                    + str(sorted(needed_names - set(semantic_detections)))
                )
            prompt = metric_trajectory_prompt(
                needed,
                semantic_tracks,
                origin=float(stamp[-1]),
                horizon=self.horizon,
                camera_pose=self.observation.cam_pose,
            )
            options = {"forecast_space": "metric"}
            if getattr(self.predictor, "inference_profile", "baseline") == "fast":
                options["response_schema"] = metric_response_schema([obj["id"] for obj in needed])
            result = self.predictor.infer(rgb, prompt, stamp, **options)
            save_json(folder / "response.json", result)
            save_json(folder / "request.json", dict(prompt=prompt, options=options))
            update.update(request_id=folder.name, latency_s=result["latency_s"])
            if not result["success"]:
                raise ValueError(str(result["error"]))
            forecasts = lift_metric_response(
                result["parsed_json"],
                semantic_detections,
                needed,
                origin=float(stamp[-1]),
                horizon=self.horizon,
                camera_pose=self.observation.cam_pose,
            )
            update["proposed_forecasts"] = [asdict(f) for f in forecasts]
            self.semantic.update({f.object_id: f for f in forecasts})
            update["parsed"] = True
            self.semantic_failed = False
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            update["error"] = str(error)
            update["parsed"] = False
            self.semantic_failed = True
        self.updates.append(update)
        save_json(self.out / "updates.json", self.updates)
        self.next_update += self.period
