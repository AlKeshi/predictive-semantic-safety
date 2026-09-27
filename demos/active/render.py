import json

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def render_video(out, semantic_enabled):
    model = mujoco.MjModel.from_xml_string((out / "scene.xml").read_text())
    data = mujoco.MjData(model)
    states = np.load(out / "states.npz")
    summary = json.loads((out / "summary.json").read_text())
    trace = json.loads((out / "trace.json").read_text())
    control_times = [row["time"] for row in trace]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.4, -0.65, 0.4]
    camera.distance, camera.azimuth, camera.elevation = (6.0, 100, -35)
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    end = summary["duration"]
    if summary["successful_recorded_demo"]:
        end = min(end, summary["goal_time"] + 3)
    elif not semantic_enabled and summary["first_contact"]:
        end = min(end, summary["first_contact"]["time"] + 0.05)
    with (
        mujoco.Renderer(model, height=720, width=1280) as renderer,
        imageio.get_reader(out / "actual-ego.mp4") as ego,
        imageio.get_writer(out / "demo.mp4", fps=20, codec="libx264", quality=8) as writer,
    ):
        for i, now in enumerate(states["times"]):
            if now > end + 1e-08:
                break
            data.qpos[:], data.qvel[:], data.time = (states["qpos"][i], states["qvel"][i], now)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = Image.fromarray(renderer.render())
            frame.paste(Image.fromarray(ego.get_data(i)).resize((320, 240)), (950, 50))
            draw = ImageDraw.Draw(frame)
            title = (
                "PSS: experimental retreat | oracle ground state | uncalibrated"
                if semantic_enabled
                else "Semantic off: direct goal"
            )
            draw.rectangle((0, 0, 1280, 40), fill="white")
            draw.text((15, 12), title, fill="black")
            control = trace[max(0, np.searchsorted(control_times, now, side="right") - 1)]
            status = control["diagnostic"]["status"]
            draw.rectangle((0, 685, 1280, 720), fill="white")
            draw.text((15, 696), f"t={now:.2f} s | {status}", fill="black")
            writer.append_data(np.asarray(frame))
