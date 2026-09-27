from __future__ import annotations

import numpy as np


class ClearanceAudit:
    cutoff_m = 10.0

    def __init__(self):
        self.samples = 0
        self.pairs = 0
        self.invalid_samples = 0
        self.invalid_pairs = 0
        self.censored_pairs = 0
        self.raw_minimum = None
        self.valid_minimum = None
        self.examples = []

    def sample(self, world, robot, hazards):
        values = []
        invalid = False
        count = 0
        for a in robot:
            for b in hazards:
                count += 1
                self.pairs += 1
                witness = np.zeros(6)
                value = float(
                    world.mj.mj_geomDistance(
                        world.model, world.data, int(a), int(b), self.cutoff_m, witness
                    )
                )
                if np.isfinite(value):
                    self.raw_minimum = (
                        value if self.raw_minimum is None else min(self.raw_minimum, value)
                    )
                length = float(np.linalg.norm(witness[3:] - witness[:3]))
                tolerance = 1e-05 * max(1.0, abs(value)) if np.isfinite(value) else 1e-05
                reason = None
                if not np.isfinite(value) or not np.all(np.isfinite(witness)):
                    reason = "nonfinite_query"
                elif value > self.cutoff_m + tolerance:
                    reason = "distance_exceeds_query_cutoff"
                elif value >= self.cutoff_m - tolerance:
                    self.censored_pairs += 1
                elif abs(length - abs(value)) > tolerance:
                    reason = "distance_and_witness_disagree"
                lower_bound = None
                if int(world.model.geom_type[a]) in range(2, 7) and int(
                    world.model.geom_type[b]
                ) in range(2, 7):
                    lower_bound = float(
                        np.linalg.norm(world.data.geom_xpos[a] - world.data.geom_xpos[b])
                        - world.model.geom_rbound[a]
                        - world.model.geom_rbound[b]
                    )
                    if not np.isfinite(lower_bound):
                        lower_bound = None
                        reason = reason or "nonfinite_primitive_geometry"
                    elif reason is None and value < min(lower_bound, self.cutoff_m) - tolerance:
                        reason = "distance_below_enclosing_sphere_lower_bound"
                if reason:
                    invalid = True
                    self.invalid_pairs += 1
                    if len(self.examples) < 8:
                        self.examples.append(
                            {
                                "time_s": float(world.data.time),
                                "geom_ids": [int(a), int(b)],
                                "reason": reason,
                                "returned_distance_m": value if np.isfinite(value) else None,
                                "witness_length_m": length if np.isfinite(length) else None,
                                "enclosing_sphere_lower_bound_m": lower_bound,
                            }
                        )
                else:
                    values.append(value)
        if not count:
            return None
        self.samples += 1
        if invalid:
            self.invalid_samples += 1
            return None
        result = min(values)
        self.valid_minimum = (
            result if self.valid_minimum is None else min(self.valid_minimum, result)
        )
        return result

    def report(self):
        complete = bool(self.samples and (not self.invalid_samples))
        all_censored = complete and self.valid_minimum >= self.cutoff_m
        return {
            "minimum_geom_clearance_m": self.valid_minimum
            if complete and (not all_censored)
            else None,
            "geom_clearance_diagnostic": {
                "schema": "mujoco_distance_consistency_v1",
                "status": "above_query_cutoff"
                if all_censored
                else "complete"
                if complete
                else "inconsistent_queries"
                if self.samples
                else "no_pairs",
                "scope": "sampled MuJoCo geometry queries with numerical consistency checks; separate from native contact labels",
                "cutoff_m": self.cutoff_m,
                "samples_n": self.samples,
                "pairs_n": self.pairs,
                "invalid_samples_n": self.invalid_samples,
                "invalid_pairs_n": self.invalid_pairs,
                "above_cutoff_pairs_n": self.censored_pairs,
                "episode_minimum_lower_bound_m": self.cutoff_m if all_censored else None,
                "raw_returned_minimum_m": self.raw_minimum,
                "minimum_over_valid_samples_m": self.valid_minimum,
                "invalid_examples": self.examples,
                "replacement_distance_used": False,
            },
        }
