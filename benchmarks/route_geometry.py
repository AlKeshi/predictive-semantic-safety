import numpy as np


def _objects(scenario):
    result = []
    for obstacle in scenario.dynamic_obstacles:
        radius = float(obstacle.radius)
        if obstacle.size is not None and obstacle.shape != "sphere":
            radius = float(np.linalg.norm(np.asarray(obstacle.size) / 2))
        center_height = float(
            obstacle.radius if obstacle.shape == "sphere" else obstacle.size[0] / 2
        )
        result.append(
            {
                "id": obstacle.name,
                "shape": obstacle.shape,
                "size": list(obstacle.size) if obstacle.size is not None else None,
                "radius": radius,
                "center_height": center_height,
                "surface_offset": center_height,
                "color": list(obstacle.color),
                "grounded": scenario.meta.get("code_world_grounded", False),
            }
        )
    return result


def _actual_hazards(world):
    return {
        ob.name: world.data.xpos[body].copy()
        for ob, body in zip(
            world.scenario.dynamic_obstacles, world.dynamic_obstacle_body_ids, strict=True
        )
    }


def _geometry_groups(world):
    robot = [
        i
        for i in range(world.model.ngeom)
        if int(world.model.geom_bodyid[i]) in world.robot_body_ids
        and (world.model.geom_contype[i] or world.model.geom_conaffinity[i])
    ]
    hazards = [
        i
        for i in range(world.model.ngeom)
        if int(world.model.geom_bodyid[i]) in world.dynamic_obstacle_body_ids
    ]
    environment = [
        i
        for i in range(world.model.ngeom)
        if int(world.model.geom_bodyid[i]) in world.scene_prop_body_ids + world.table_body_ids
        and (world.model.geom_contype[i] or world.model.geom_conaffinity[i])
    ]
    return (robot, hazards, environment)


def _contacts(world, robot, hazards, environment):
    robot, hazards, environment = (set(robot), set(hazards), set(environment))
    result = []
    for con in world.data.contact[: world.data.ncon]:
        a, b = (int(con.geom1), int(con.geom2))
        if b in robot:
            a, b = (b, a)
        if a not in robot or b not in hazards | environment:
            continue
        result.append(
            {
                "time": float(world.data.time),
                "kind": "hazard" if b in hazards else "environment",
                "robot_geom": world.model.geom(a).name,
                "robot_geom_id": a,
                "robot_body": world.model.body(int(world.model.geom_bodyid[a])).name,
                "other_geom": world.model.geom(b).name,
                "other_geom_id": b,
                "signed_distance_m": float(con.dist),
            }
        )
    return result


def retained_forecast_receipt(controller, detections, objects, now):
    expected = {item["id"] for item in objects}
    if not np.isfinite(now) or not set(detections) <= expected:
        raise ValueError("Observed identities differ from the declared inventory")
    snapshot = controller.snapshot()
    family = snapshot["forecasts"]
    if snapshot["emergency_reason"] is not None:
        raise ValueError("Cannot carry a latched invalid active family")
    if {item["object_id"] for item in family} != expected:
        raise ValueError("No complete previously committed inventory")
    for item in family:
        times = np.asarray(item["times"], dtype=float)
        if now < times[0] - 1e-09 or now >= times[-1] - 1e-08:
            raise ValueError("Carried forecast expired or unsupported")
        detection = detections.get(item["object_id"])
        if detection is not None:
            centers = np.asarray(item["centers"], dtype=float)
            predicted = np.array([np.interp(now, times, centers[:, axis]) for axis in range(2)])
            measured = np.asarray(detection.center[:2], dtype=float)
            radius = float(np.interp(now, times, np.asarray(item["radii"], dtype=float)))
            observed_radius = float(detection.radius)
            if (
                measured.shape != (2,)
                or not np.all(np.isfinite(measured))
                or (not np.isfinite(radius))
                or (not np.isfinite(observed_radius))
                or (radius < 0)
                or (observed_radius < 0)
                or (np.linalg.norm(measured - predicted) + observed_radius > radius + 1e-09)
            ):
                raise ValueError("Current observation contradicts carried forecast padding")
    return {
        "status": "observation_unavailable:retained_original_support"
        if set(detections) != expected
        else "retained_consistent_forecast",
        "refresh_missing": set(detections) != expected,
        "missing_object_ids": sorted(expected - set(detections)),
        "retained_committed_family": True,
        "carried_forecasts": family,
        "carry_until_time": min((item["times"][-1] for item in family)),
        "carry_requires_current_command_checks": True,
    }
