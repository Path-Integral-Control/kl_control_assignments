"""Procedural obstacle generation (forest, random fields).

Generates circle obstacles using rejection sampling
with clearance from start/goal positions.
"""

import numpy as np


def generate_forest(forest_config, start_pos, goal_pos):
    """Generate circle obstacles using rejection sampling.

    Parameters
    ----------
    forest_config : dict
        Keys: position, size, density, tree_radius, seed
    start_pos : array-like
        Start position [x, y] — clearance enforced.
    goal_pos : array-like
        Goal position [x, y] — clearance enforced.

    Returns
    -------
    obstacles : list of dict
        Each with keys: type, position, radius, cost
    """
    cx, cy = forest_config['position']
    w, h = forest_config['size']
    density = forest_config.get('density', 0.5)
    radius = forest_config.get('tree_radius', 0.2)
    count = int(w * h * density)
    seed = forest_config.get('seed', 42)
    cost = forest_config.get('cost', 4e12)
    clearance = forest_config.get('clearance', 1.0)
    min_dist = radius * 2.5

    rng = np.random.default_rng(seed)
    trees = []

    for _ in range(count * 5):
        cand_x = rng.uniform(cx - w / 2.0, cx + w / 2.0)
        cand_y = rng.uniform(cy - h / 2.0, cy + h / 2.0)

        if np.linalg.norm([cand_x - start_pos[0], cand_y - start_pos[1]]) < clearance:
            continue
        if np.linalg.norm([cand_x - goal_pos[0], cand_y - goal_pos[1]]) < clearance:
            continue

        if all(np.linalg.norm([cand_x - tx, cand_y - ty]) > min_dist for tx, ty in trees):
            trees.append((cand_x, cand_y))

        if len(trees) >= count:
            break

    obstacles = []
    for tx, ty in trees:
        obstacles.append({
            'type': 'circle',
            'position': [tx, ty],
            'radius': radius,
            'cost': cost,
        })

    return obstacles
