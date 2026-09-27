from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps


def visible_forecasts(diagnostic, snapshot, now):
    status = str(diagnostic.get("status", diagnostic.get("mode", ""))).lower()
    if (
        not diagnostic.get("feasible", False)
        or snapshot.get("emergency_reason")
        or any((word in status for word in ("brake", "emergency", "invalid", "expired")))
    ):
        return []
    rows = diagnostic.get("enforced_rows", [])
    result = []
    for forecast in snapshot.get("forecasts", []):
        name = forecast["object_id"]
        times = np.asarray(forecast["times"], dtype=float)
        centers = np.asarray(forecast["centers"], dtype=float)
        radii = np.asarray(forecast["radii"], dtype=float)
        if radii.ndim == 0:
            radii = np.full(len(times), radii)
        if (
            times.ndim != 1
            or len(times) < 2
            or centers.shape != (len(times), 2)
            or (radii.shape != times.shape)
            or (not np.all(np.isfinite(times)))
            or (not np.all(np.isfinite(centers)))
            or (not np.all(np.isfinite(radii)))
            or np.any(np.diff(times) <= 0)
            or np.any(radii < 0)
            or (now < times[0] - 1e-08)
            or (now >= times[-1] - 1e-08)
        ):
            continue
        enforced_times = np.unique(
            [
                float(row["time"])
                for row in rows
                if row.get("kind") == "predictive"
                and row.get("object_id") == name
                and np.isfinite(row.get("time", np.nan))
                and (max(float(now), times[0]) - 1e-08 <= float(row["time"]) <= times[-1] + 1e-08)
            ]
        )
        if not len(enforced_times):
            continue
        xy = np.column_stack(
            [np.interp(enforced_times, times, centers[:, axis]) for axis in range(2)]
        )
        result.append(
            {
                "object_id": name,
                "origin_time": float(forecast["origin_time"]),
                "times": enforced_times,
                "centers": xy,
            }
        )
    return result


class EpisodeRenderer:
    def __init__(self, world, out, title, fps=20, *, predictor_label=None):
        self.world, self.out, self.title = (world, Path(out), str(title))
        self.out.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.panel_width, self.panel_height = (960, 540)
        self.gutter = 8
        self.hero = world.mj.Renderer(world.model, height=self.panel_height, width=self.panel_width)
        self.camera = world.mj.MjvCamera()
        config = world.scenario.meta.get("mujoco_showcase_camera", {})
        lookat = np.asarray(config.get("lookat", (0.9, 0.0, 0.32)), dtype=float)
        if lookat.shape != (3,) or not np.all(np.isfinite(lookat)):
            raise ValueError("Overview lookat must be finite XYZ")
        self.camera.lookat[:] = np.clip(lookat, (-3.0, -3.0, 0.0), (4.0, 4.0, 2.0))
        self.camera.distance = float(np.clip(config.get("distance", 6.4), 3.0, 9.0))
        self.camera.azimuth = float(np.clip(config.get("azimuth", 125.0), -180.0, 180.0))
        self.camera.elevation = float(np.clip(config.get("elevation", -32.0), -70.0, -10.0))
        if not np.all(
            np.isfinite([self.camera.distance, self.camera.azimuth, self.camera.elevation])
        ):
            raise ValueError("Overview camera values must be finite")
        self.writer = imageio.get_writer(
            str(self.out / "demo.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=None
        )
        self.frame_count, self.gifs = (0, [])

    def capture(
        self, observation, diagnostic, snapshot, clearance, *, contact=False, goal_reached=False
    ):
        now = float(self.world.data.time)
        forecasts = visible_forecasts(diagnostic, snapshot, now)
        self.hero.update_scene(self.world.data, camera=self.camera)
        overview = Image.fromarray(np.asarray(self.hero.render()).copy()).convert("RGB")
        ego = Image.fromarray(np.asarray(observation.rgb).copy()).convert("RGB")
        canvas = Image.new(
            "RGB", (2 * self.panel_width + self.gutter, self.panel_height), (44, 48, 51)
        )
        canvas.paste(overview, (0, 0))
        fitted = ImageOps.contain(ego, (self.panel_width, self.panel_height))
        canvas.paste(
            fitted,
            (
                self.panel_width + self.gutter + (self.panel_width - fitted.width) // 2,
                (self.panel_height - fitted.height) // 2,
            ),
        )
        frame = np.asarray(canvas)
        self.writer.append_data(frame)
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            self.gifs.append(np.asarray(canvas.resize((1280, 358))))
        if self.frame_count == 1:
            canvas.save(self.out / "preview.png")
        if forecasts and (not (self.out / "forecast_preview.png").exists()):
            canvas.save(self.out / "forecast_preview.png")
        return frame

    def close(self):
        self.writer.close()
        self.hero.close()
        if self.gifs:
            imageio.mimsave(self.out / "demo.gif", self.gifs, duration=2 / self.fps, loop=0)
