import csv
import json
import sys

import pytest

from benchmarks.summarize import main, summarize


def outcome(trial="trial-1", *, contact=False):
    return {
        "method": "pss",
        "trial": trial,
        "contact_free": not contact,
        "hazard_contact": contact,
        "environment_contact": False,
        "status": "complete",
    }


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_csv_and_run_directory_agree(tmp_path):
    rows = [outcome("a"), outcome("b", contact=True)]
    for row in rows:
        save_json(tmp_path / "runs" / row["trial"] / "result.json", row)
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert summarize(path) == summarize(tmp_path / "runs")
    assert summarize(path)[0] == {
        "method": "pss",
        "label": "PSS",
        "trials": 2,
        "contact_free": 1,
        "rate": 0.5,
    }


def test_single_result_and_cli(tmp_path, monkeypatch, capsys):
    path = save_json(tmp_path / "result.json", outcome())
    monkeypatch.setattr(sys, "argv", ["summarize", str(path)])
    main()
    assert json.loads(capsys.readouterr().out) == summarize(path)


def test_cli_requires_input(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["summarize"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "source" in capsys.readouterr().err


def test_cli_passes_explicit_plot_output(tmp_path, monkeypatch, capsys):
    from benchmarks import summarize as module

    path = save_json(tmp_path / "result.json", outcome())
    output = tmp_path / "plot.svg"
    calls = []
    monkeypatch.setattr(module, "plot", lambda rows, target: calls.append((rows, target)))
    monkeypatch.setattr(sys, "argv", ["summarize", str(path), "--plot", str(output)])
    main()
    assert calls == [(json.loads(capsys.readouterr().out), output)]


def test_missing_input_and_empty_directory_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path / "absent.csv")
    with pytest.raises(ValueError, match="No completed"):
        summarize(tmp_path)


def test_auxiliary_outputs_are_not_counted(tmp_path):
    save_json(tmp_path / "run" / "result.json", outcome())
    save_json(tmp_path / "run" / "calibration_record.json", {})
    save_json(tmp_path / "unfinished" / "case.json", {})
    assert summarize(tmp_path)[0]["trials"] == 1


@pytest.mark.parametrize("value", [{}, [], {"status": "failed"}, {"status": "running"}])
def test_incomplete_or_invalid_json_is_rejected(tmp_path, value):
    path = save_json(tmp_path / "result.json", value)
    with pytest.raises(ValueError, match="completed"):
        summarize(path)


@pytest.mark.parametrize("value", [1, 0, None, "yes", [], {}])
def test_non_boolean_contact_flags_are_rejected(tmp_path, value):
    row = outcome()
    row["contact_free"] = value
    path = save_json(tmp_path / "result.json", row)
    with pytest.raises(ValueError, match="Invalid contact_free"):
        summarize(path)


def test_inconsistent_contact_flags_are_rejected(tmp_path):
    row = outcome()
    row["hazard_contact"] = True
    path = save_json(tmp_path / "result.json", row)
    with pytest.raises(ValueError, match="Inconsistent"):
        summarize(path)


def test_duplicate_trial_is_rejected(tmp_path):
    save_json(tmp_path / "a" / "result.json", outcome())
    save_json(tmp_path / "b" / "result.json", outcome())
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(tmp_path)


@pytest.mark.parametrize("field,value", [("method", "other"), ("method", []), ("trial", "")])
def test_invalid_identity_is_rejected(tmp_path, field, value):
    row = outcome()
    row[field] = value
    path = save_json(tmp_path / "result.json", row)
    with pytest.raises(ValueError):
        summarize(path)


def test_incomplete_csv_rows_are_rejected(tmp_path):
    row = outcome()
    row["status"] = "running"
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="completed"):
        summarize(path)


def test_demo_and_benchmark_cannot_be_pooled(tmp_path):
    demo = outcome("demo")
    demo["paper_comparable"] = False
    save_json(tmp_path / "demo" / "result.json", demo)
    save_json(tmp_path / "benchmark" / "result.json", outcome())
    with pytest.raises(ValueError, match="cannot be pooled"):
        summarize(tmp_path)


def test_paper_protocol_checks_denominator_pairing_and_grouped_intervals():
    import hashlib

    from benchmarks.protocol import paper_intervals
    from benchmarks.summarize import LABELS
    from datagen.manifest import list_cases

    cases = list_cases()
    groups = sorted({case["base_case_id"] for case in cases})
    rows = [
        dict(
            outcome(case["id"], contact=groups.index(case["base_case_id"]) >= len(groups) // 2),
            method=method,
            base_case_id=case["base_case_id"],
            paper_comparable=True,
            case_sha256=hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
            calibration_record_digest="same-record",
            calibration_pipeline_digest="same-pipeline",
            residual_margin_applied=method == "pss",
        )
        for method in LABELS
        for case in cases
    ]
    intervals = paper_intervals(rows, LABELS)
    for low, high in intervals.values():
        assert 0.2 < low < 0.4 and 0.6 < high < 0.8
    with pytest.raises(ValueError, match="1 missing/unknown"):
        paper_intervals(rows[:-1], LABELS)
    rows[-1]["calibration_record_digest"] = "different-record"
    with pytest.raises(ValueError, match="same record"):
        paper_intervals(rows, LABELS)
    rows[-1]["calibration_record_digest"] = "same-record"
    rows[-1]["paper_comparable"] = False
    with pytest.raises(ValueError, match="benchmark-only"):
        paper_intervals(rows, LABELS)


def test_csv_demo_and_benchmark_cannot_be_pooled(tmp_path):
    rows = [
        dict(outcome("demo"), paper_comparable=False),
        dict(outcome("benchmark"), paper_comparable=True),
    ]
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="cannot be pooled"):
        summarize(path)
