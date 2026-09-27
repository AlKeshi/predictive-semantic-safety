from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "go1"
_HOME_ANGLES = np.array(
    [0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8], dtype=np.float32
)


@functools.lru_cache(maxsize=4)
def _go1_onnx_session(policy_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "The Go1 MuJoCo runtime needs ONNX Runtime. Install `pip install -e .[mujoco]`."
        ) from exc
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(
        policy_path, sess_options=session_options, providers=["CPUExecutionProvider"]
    )


class Go1OnnxPolicy:
    def __init__(self, policy_path: Path | None = None):
        path = Path(policy_path or _ASSET_ROOT / "go1_policy.onnx").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing vendored Go1 locomotion policy: {path}")
        self._session = _go1_onnx_session(str(path))
        self.last_action = np.zeros(12, dtype=np.float32)

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def apply(self, model, data, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError("Go1 command must be a finite [vx, vy, yaw_rate] vector")
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
        joint_angles = data.qpos[7:19] - _HOME_ANGLES
        joint_velocities = data.qvel[6:18]
        observation = np.hstack(
            [linvel, gyro, gravity, joint_angles, joint_velocities, self.last_action, command]
        ).astype(np.float32)
        if observation.shape != (48,):
            raise RuntimeError(f"Unexpected Go1 policy observation shape {observation.shape}")
        action = self._session.run(["continuous_actions"], {"obs": observation.reshape(1, -1)})[0][
            0
        ]
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        data.ctrl[:] = self.last_action * 0.5 + _HOME_ANGLES
        return data.ctrl.copy()
