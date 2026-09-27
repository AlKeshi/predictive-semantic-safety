import json
from types import SimpleNamespace

import pytest


def accepts(text, schema=None):
    xgr = pytest.importorskip("xgrammar")
    from pss.predictors.fast_json import bounded_grammar

    info = xgr.TokenizerInfo(["<eos>"], stop_token_ids=[0])
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    grammar = bounded_grammar(json.dumps(schema or {"type": "object"}))
    matcher = xgr.GrammarMatcher(compiler.compile_grammar(grammar))
    return matcher.accept_string(text) and matcher.accept_token(0) and matcher.is_terminated()


@pytest.mark.parametrize("number", ["0", "-3.141592", "0.000001", "12345678", "1e-99", "-1.2E+12"])
def test_finite_numeric_spellings(number):
    assert accepts('{"value":' + number + "}")


@pytest.mark.parametrize("number", ["1.0000000", "1e000", "123456789", "1e", "NaN", "01"])
def test_runaway_or_invalid_numbers_are_rejected(number):
    assert not accepts('{"value":' + number + "}")


def test_escaped_quotes_and_digits_in_text_are_not_numeric_fields():
    text = json.dumps({"text": 'quoted "1.000000000000000000" and backslash', "value": -1e-06})
    assert accepts(text)


@pytest.mark.parametrize("rows,valid", [(3, False), (4, False), (5, True), (6, False)])
def test_fixture_schema_enforces_exact_trajectory_count(rows, valid):
    from pss.predictors.structured_json import ANALYSIS

    text = json.dumps(
        {
            "objects": [
                {
                    "id": "ceiling_fixture",
                    "attachment_state": "loose",
                    "motion_description": "Visible motion.",
                    "trajectory": [[i + 1, 0, 0, -1] for i in range(rows)],
                }
            ]
        }
    )
    assert accepts(text, ANALYSIS) == valid


def test_inventory_grammar_rejects_missing_or_duplicate_objects():
    from pss.predictors.fast_json import metric_response_schema

    schema = metric_response_schema(["a", "b"])
    entries = [
        {
            "id": name,
            "motion_description": "Motion.",
            "trajectory": [[i + 1, 0, 0, 0] for i in range(5)],
        }
        for name in ["a", "b"]
    ]
    assert accepts(json.dumps({"objects": entries}), schema)
    assert not accepts(json.dumps({"objects": entries[:1]}), schema)
    assert not accepts(json.dumps({"objects": [entries[0], entries[0]]}), schema)


def test_refinement_prompt_retains_analysis_evidence():
    from pss.predictors.fixture_events import refinement_prompt

    facts = {"current_center_m": [0, 0, 2]}
    assert '"anything": "A"' in refinement_prompt(facts, {"anything": "A"})


def test_schema_installers_isolate_stages_without_mutating_shared_model(monkeypatch):
    from pss.predictors import structured_json

    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    monkeypatch.setattr(structured_json, "generation_constraints", lambda predictor, schema: schema)

    def generate():
        return None

    shared = SimpleNamespace(generate=generate)
    first = SimpleNamespace(model=shared, provenance={}, inference_profile="baseline")
    second = SimpleNamespace(model=shared, provenance={}, inference_profile="baseline")
    structured_json.install_schema_generation(first)
    structured_json.install_schema_generation(second)
    first.schema_stage = "event"
    assert first._generation_constraints() is structured_json.EVENT
    assert second._generation_constraints() is structured_json.ANALYSIS
    assert shared.generate is generate


def test_resident_schema_receipt_uses_engine_version_not_client_version(monkeypatch):
    from pss.predictors import structured_json

    monkeypatch.setattr("importlib.metadata.version", lambda name: "client-version")
    predictor = SimpleNamespace(
        inference_profile="fast",
        provenance={"acceleration_versions": {"xgrammar": "engine-version"}},
    )
    structured_json.install_schema_generation(predictor)
    assert predictor.provenance["xgrammar_version"] == "engine-version"
