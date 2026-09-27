import json
import re
from functools import lru_cache

NUMBER_RULE = (
    'basic_number ::= "-"? ("0" | [1-9] [0-9]{0,7}) ("." [0-9]{1,6})? ([eE] [+-]? [0-9]{1,2})?'
)


@lru_cache(maxsize=32)
def bounded_grammar(schema_json):
    import xgrammar as xgr

    original = str(xgr.Grammar.from_json_schema(schema_json, max_whitespace_cnt=1))
    grammar, count = re.subn(
        "^basic_number ::=.*$", lambda _: NUMBER_RULE, original, flags=re.MULTILINE
    )
    if count != 1:
        raise RuntimeError("Unsupported XGrammar numeric rule; refusing unbounded generation")
    return str(xgr.Grammar.from_ebnf(grammar))


@lru_cache(maxsize=2)
def _compiler(tokenizer, vocab_size):
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    return xgr.GrammarCompiler(info, max_threads=2)


def generation_constraints(predictor, schema):
    from xgrammar.contrib.hf import LogitsProcessor

    if schema is None:
        raise ValueError("Fast inference requires an explicit response schema")
    config = predictor.model.config.text_config
    grammar = bounded_grammar(json.dumps(schema))
    compiled = _compiler(predictor.processor.tokenizer, config.vocab_size).compile_grammar(grammar)
    return {"logits_processor": [LogitsProcessor(compiled)]}


def metric_response_schema(names):
    if not names or any((not isinstance(name, str) or not name for name in names)):
        raise ValueError("Nonempty object IDs are required")
    if len(set(names)) != len(names):
        raise ValueError("Object IDs must be unique")
    entries = []
    for name in names:
        entries.append(
            {
                "type": "object",
                "properties": {
                    "id": {"const": name},
                    "motion_description": {"type": "string", "maxLength": 600},
                    "trajectory": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "items": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "number"},
                        },
                    },
                },
                "required": ["id", "motion_description", "trajectory"],
                "additionalProperties": False,
            }
        )
    return {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "prefixItems": entries,
                "minItems": len(entries),
                "maxItems": len(entries),
                "items": False,
            }
        },
        "required": ["objects"],
        "additionalProperties": False,
    }
