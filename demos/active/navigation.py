import heapq

import numpy as np


def waypoint(position, goal, forecasts, static, robot_radius=0.403):
    position, goal = (np.asarray(position, float), np.asarray(goal, float))
    step = 0.15
    lower = np.minimum(position, goal) - 2.5
    upper = np.maximum(position, goal) + 2.5
    xs = np.arange(lower[0], upper[0] + step, step)
    ys = np.arange(lower[1], upper[1] + step, step)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([xx, yy], axis=-1)
    margin = np.full(xx.shape, np.inf)
    radius_extra = robot_radius + 0.12 + step / np.sqrt(2)
    for f in forecasts:
        radii = np.broadcast_to(np.asarray(f.radii, float), np.asarray(f.times).shape)
        for p, r in zip(f.centers, radii, strict=True):
            margin = np.minimum(margin, np.linalg.norm(points - p, axis=-1) - r - radius_extra)
    for ob in static:
        margin = np.minimum(
            margin,
            np.linalg.norm(points - np.asarray(ob["center"]), axis=-1)
            - ob["radius"]
            - radius_extra,
        )

    def cell(p):
        return (int(np.argmin(abs(xs - p[0]))), int(np.argmin(abs(ys - p[1]))))

    free = margin > 0.0
    start, finish = (cell(position), cell(goal))
    if not free[finish]:
        return (None, dict(status="goal_occupied", resolution_m=step))
    queue = [(0.0, start)]
    distance, parent = ({start: 0.0}, {})
    moves = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    visited = set()
    while queue:
        _, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == finish:
            path = [node]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            points = np.array([[xs[i], ys[j]] for i, j in path])
            selected = goal
            for p in points[1:]:
                if np.linalg.norm(p - position) >= 0.65:
                    selected = p
                    break
            return (
                selected,
                dict(
                    status="planned" if free[start] else "planned_egress",
                    resolution_m=step,
                    path=points.tolist(),
                    waypoint=selected.tolist(),
                ),
            )
        for di, dj in moves:
            ni, nj = (node[0] + di, node[1] + dj)
            if ni < 0 or nj < 0 or ni >= len(xs) or (nj >= len(ys)):
                continue
            if not free[ni, nj] and (free[node] or margin[ni, nj] <= margin[node] + 1e-10):
                continue
            if (
                di
                and dj
                and (
                    min(margin[node[0] + di, node[1]], margin[node[0], node[1] + dj])
                    < min(0.0, margin[node])
                )
            ):
                continue
            nxt = (ni, nj)
            cost = distance[node] + float(np.hypot(di, dj)) * (1.0 if free[ni, nj] else 3.0)
            if cost + 1e-09 < distance.get(nxt, float("inf")):
                distance[nxt], parent[nxt] = (cost, node)
                heapq.heappush(queue, (cost + float(np.hypot(ni - finish[0], nj - finish[1])), nxt))
    return (None, dict(status="no_path", resolution_m=step))
