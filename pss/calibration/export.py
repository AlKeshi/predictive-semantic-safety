import argparse
import json
from pathlib import Path

from .data import pipeline_digest, validate_corpus


def export_corpus(paths):
    traces = [json.loads(Path(path).read_text()) for path in paths]
    if not traces:
        raise ValueError("At least one complete physical trace is required")
    first = traces[0]
    corpus = {key: first[key] for key in ("pipeline", "indices", "shape_radii")}
    corpus["sessions"] = []
    for trace in traces:
        if trace.get("schema") != "pss.calibration-session-trace.v1":
            raise ValueError("Unsupported trace schema")
        if trace["pipeline"].get("acquisition_mode") != "calibrated_closed_loop":
            raise ValueError("Raw-policy traces cannot be relabeled as calibrated acquisition")
        if trace.get("residual_margin_applied") is False:
            raise ValueError("Margin ablations cannot be relabeled as calibrated acquisition")
        if pipeline_digest(trace["pipeline"]) != trace["pipeline_digest"]:
            raise ValueError("Trace pipeline fingerprint is invalid")
        for key in ("pipeline", "indices", "shape_radii"):
            if trace[key] != corpus[key]:
                raise ValueError(f"Trace has different frozen {key}")
        if not trace.get("acquisition_record_digest"):
            raise ValueError("Trace must identify the actual acquisition controller record")
        if trace["acquisition_record_digest"] != first["acquisition_record_digest"]:
            raise ValueError("Traces use different acquisition controller records")
        corpus["sessions"].append(
            {
                key: trace[key]
                for key in ("session_id", "prediction", "truth", "acquisition_record_digest")
            }
        )
    validate_corpus(corpus)
    return corpus


def main():
    parser = argparse.ArgumentParser(description="Export")
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    corpus = export_corpus(args.traces)
    args.output.write_text(json.dumps(corpus, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
