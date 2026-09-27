import hashlib
import json

import numpy as np

from datagen.manifest import list_cases


def paper_intervals(rows, methods):
    cases = {case["id"]: case for case in list_cases()}
    expected = {(method, trial) for method in methods for trial in cases}
    actual = {(row["method"], row["trial"]): row for row in rows}
    missing, unexpected = expected - actual.keys(), actual.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"Incomplete paper cohort: {len(actual)} present, {len(missing)} missing/unknown, "
            f"{len(unexpected)} unexpected; no final rates computed"
        )
    for row in rows:
        case = cases[row["trial"]]
        digest = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
        if row.get("case_sha256") != digest or row.get("base_case_id") != case["base_case_id"]:
            raise ValueError("Result differs from the frozen physical case")
        if row.get("paper_comparable") not in (True, "True"):
            raise ValueError("Paper aggregation requires benchmark-only results")
    for trial in cases:
        full, raw = (actual[(method, trial)] for method in ("pss", "pss_uncalibrated"))
        for field in ("calibration_record_digest", "calibration_pipeline_digest"):
            if not full.get(field) or full[field] != raw.get(field):
                raise ValueError("PSS ablations must use the same record and acquisition schedule")
        if full.get("residual_margin_applied") not in (True, "True") or raw.get(
            "residual_margin_applied"
        ) not in (False, "False"):
            raise ValueError("PSS ablations must differ only in residual margin")
    groups = sorted({case["base_case_id"] for case in cases.values()})
    samples = np.random.default_rng(0).integers(len(groups), size=(10000, len(groups)))
    intervals = {}
    for method in methods:
        means = np.array(
            [
                np.mean(
                    [
                        actual[(method, trial)]["contact_free"] in (True, "True")
                        for trial, case in cases.items()
                        if case["base_case_id"] == group
                    ]
                )
                for group in groups
            ]
        )
        intervals[method] = np.quantile(means[samples].mean(axis=1), [0.025, 0.975]).tolist()
    return intervals
