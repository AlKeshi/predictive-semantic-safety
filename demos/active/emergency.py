import numpy as np

from pss.control.constraints import static_rows
from pss.control.forecast import freeze_forecast


def fresh_ground_fallback(
    controller, state, now, nominal, acceleration, diagnostic, family, sources
):
    if (
        diagnostic["feasible"]
        or not family
        or (
            not any(
                (s["source"] == "oracle_current_ground_position_velocity" for s in sources.values())
            )
        )
    ):
        return (acceleration, diagnostic)
    try:
        frozen = tuple((freeze_forecast(f, now, controller.sample_dt)[0] for f in family))
        if len({f.object_id for f in frozen}) != len(frozen):
            raise ValueError("Duplicate emergency geometry")
        proposed = controller.backup_input(state, now, frozen)
        rows = static_rows(controller, np.asarray(state, float), now)
        checked, status = controller._solve(proposed, rows)
        if checked is None or status != "solved":
            return (acceleration, diagnostic)
    except (ValueError, FloatingPointError):
        return (acceleration, diagnostic)
    result = dict(diagnostic)
    result.update(
        status="uncertified_retreat:fresh_ground_geometry",
        feasible=False,
        certificate_valid=False,
        execute_backup=True,
        fallback_source="fresh_ground_public_retreat_static_filtered",
        acceleration=checked.tolist(),
        enforced_rows=rows,
        num_constraints=len(rows),
        intervened=bool(np.linalg.norm(checked - nominal) > 0.001),
        emergency_sources=sources,
        predictive_family_committed=False,
    )
    return (checked, result)
