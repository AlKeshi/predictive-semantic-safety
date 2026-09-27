from functools import lru_cache

STATES = ["secure", "loose", "detached", "grounded", "uncertain"]
ENTRY = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "enum": ["ceiling_fixture"]},
        "attachment_state": {"type": "string", "enum": STATES},
        "motion_description": {"type": "string", "maxLength": 600},
        "trajectory": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
        },
    },
    "required": ["id", "attachment_state", "motion_description", "trajectory"],
    "additionalProperties": False,
}
ANALYSIS = {
    "type": "object",
    "properties": {"objects": {"type": "array", "minItems": 1, "maxItems": 1, "items": ENTRY}},
    "required": ["objects"],
    "additionalProperties": False,
}
EVENT = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "enum": ["ceiling_fixture"]},
        "attachment_state": {"type": "string", "enum": STATES},
        "evolution": {"type": "string", "enum": ["supported", "free_fall", "rest"]},
        "release_interval_s": {
            "anyOf": [
                {"type": "null"},
                {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
            ]
        },
        "motion_description": {"type": "string", "maxLength": 600},
    },
    "required": ["id", "attachment_state", "evolution", "release_interval_s", "motion_description"],
    "additionalProperties": False,
}


@lru_cache(maxsize=2)
def _tokenizer_data(tokenizer):
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    special_ids = set(tokenizer.all_special_ids)
    zero = tokenizer.encode("0", add_special_tokens=False)[-1]
    regular = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        after_zero = tokenizer.decode([zero, token_id])[1:]
        isolated = tokenizer.decode([token_id])
        regular.append((token_id, after_zero, len(after_zero) > len(isolated)))

    def decode(tokens):
        return tokenizer.decode(tokens).rstrip(chr(65533))

    return TokenEnforcerTokenizerData(
        regular, decode, tokenizer.eos_token_id, False, len(tokenizer)
    )


def generation_constraints(predictor, schema):
    if getattr(predictor, "inference_profile", "baseline") == "fast":
        from .fast_json import generation_constraints as fast_constraints

        return fast_constraints(predictor, schema)
    import transformers.tokenization_utils as legacy_tokenization
    from lmformatenforcer import JsonSchemaParser
    from transformers import PreTrainedTokenizerBase

    if not hasattr(legacy_tokenization, "PreTrainedTokenizerBase"):
        legacy_tokenization.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    parser = JsonSchemaParser(schema)
    prefix = build_transformers_prefix_allowed_tokens_fn(
        _tokenizer_data(predictor.processor.tokenizer), parser
    )
    return {"prefix_allowed_tokens_fn": prefix}


def install_schema_generation(predictor):
    import importlib.metadata

    predictor.schema_stage = "analysis"

    def constraints():
        schema = ANALYSIS if predictor.schema_stage == "analysis" else EVENT
        predictor._active_response_schema = schema
        return generation_constraints(predictor, schema)

    predictor._generation_constraints = constraints
    predictor.provenance["output_format"] = (
        "JSON Schema constrained token generation; no output repair"
    )
    if getattr(predictor, "inference_profile", "baseline") == "fast":
        predictor.provenance["output_format"] = "XGrammar schema and bounded numeric spelling"
        predictor.provenance["xgrammar_version"] = predictor.provenance.get(
            "acceleration_versions", {}
        ).get("xgrammar") or importlib.metadata.version("xgrammar")
    else:
        predictor.provenance["lm_format_enforcer_version"] = importlib.metadata.version(
            "lm-format-enforcer"
        )
