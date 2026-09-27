import json
import subprocess
import sys
import time

import numpy as np
import pytest

from pss import Forecast, Predictor, SafetyFilter, load_case, run_benchmark, run_demo
from pss.__main__ import COMMANDS, main
from pss.control.qp import project_velocity
from pss.predictors.service import wait_ready


def test_import_does_not_load_model():
    result = subprocess.run(
        [sys.executable, "-c", "import pss, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert all(
        callable(item) for item in (Predictor, Forecast, SafetyFilter, run_benchmark, run_demo)
    )
    assert load_case("stack_00_r00")["family"] == "stack"


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_keeps_host_arguments(command, monkeypatch, capsys):
    arguments = ["notebook", "--unrelated-host-option"]
    monkeypatch.setattr(sys, "argv", arguments.copy())
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert f"usage: pss {command}" in capsys.readouterr().out
    assert sys.argv == arguments


def test_cli_can_be_called_twice(capsys):
    main(["demo", "--list"])
    main(["demo", "--list"])
    assert capsys.readouterr().out.count("active/off") == 2


def test_demo_fails_before_creating_output(tmp_path):
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="checkpoint"):
        run_demo("pss/stack-a", destination)
    with pytest.raises(ValueError, match="Duration"):
        run_demo("active/off", destination, duration=float("nan"))
    assert not destination.exists()


@pytest.mark.parametrize("age", [30, -60])
def test_stale_or_future_worker_is_not_ready(tmp_path, age):
    (tmp_path / "ready.json").write_text(
        json.dumps(dict(status="ready", heartbeat_unix=time.time() - age))
    )
    with pytest.raises(TimeoutError):
        wait_ready(tmp_path, timeout=0)


def test_ready_and_stopped_workers(tmp_path):
    record = dict(status="ready", heartbeat_unix=time.time(), worker_id="test")
    (tmp_path / "ready.json").write_text(json.dumps(record))
    assert wait_ready(tmp_path, timeout=0) == record
    (tmp_path / "ready.json").write_text(json.dumps(dict(status="stopped")))
    with pytest.raises(RuntimeError, match="stopped"):
        wait_ready(tmp_path, timeout=0)


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_wait_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="Timeout"):
        wait_ready(tmp_path, timeout=timeout)


def test_missing_worker(tmp_path):
    with pytest.raises(TimeoutError):
        wait_ready(tmp_path, timeout=0)


def test_moving_obstacle_velocity_projection():
    rows = [{"coefficients": [1, 0], "lower_bound": 0.2}]
    np.testing.assert_allclose(project_velocity([-0.4, 0.3], rows, 0.5), [0.2, 0.3])
    rows.append({"coefficients": [-1, 0], "lower_bound": 0.1})
    with pytest.raises(ValueError, match="infeasible"):
        project_velocity([0, 0], rows, 0.5)
