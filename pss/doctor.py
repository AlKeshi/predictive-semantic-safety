import argparse
import importlib.metadata
import platform


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss doctor", description="Check the simulator installation and assets."
    )
    parser.add_argument("--camera", action="store_true", help="Also create a MuJoCo renderer")
    args = parser.parse_args(argv)
    import mujoco

    from demos import scene_path

    model = mujoco.MjModel.from_xml_path(str(scene_path("stack-a")))
    if args.camera:
        with mujoco.Renderer(model, height=64, width=64) as renderer:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            assert renderer.render().shape == (64, 64, 3)
    print(
        f"Python {platform.python_version()}; assets OK; camera {('OK' if args.camera else 'not checked')}"
    )
    print(
        "; ".join(
            (
                f"{name} {importlib.metadata.version(name)}"
                for name in ("numpy", "mujoco", "onnxruntime", "osqp")
            )
        )
    )
