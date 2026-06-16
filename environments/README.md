# environments/ — Obstacle Fields and Cost Computation

Loads 2D obstacle environments from YAML configuration files and provides cost evaluation for both continuous and discrete (gridworld) settings.

Each environment defines a start position, goal position, and a set of obstacles. The `Environment` class computes:

- **State cost** (`get_state_cost`): quadratic goal-tracking cost plus obstacle penalties. This is the cost used by MPPI and the continuous-time SDE examples.
- **Obstacle-only cost** (`get_obstacle_cost`): obstacle penalties without the goal term. Used internally when discretizing to a gridworld, where goal information is encoded separately in the terminal cost.
- **Cost field** (`cost_field`): evaluates the state cost on a meshgrid, used for plotting and for the variational examples (Chapter 6).

## Obstacle types

- **Gaussian**: soft bump, C = amplitude * exp(-dist^2 / (2 * spread^2)). Used by `three_mountains`, `landing_site`, `simple_goal`.
- **Circle**: hard boundary with constant cost inside a radius. Used by `forest` (procedurally generated).
- **Box**: hard rectangle with constant cost inside and an optional exponential buffer zone. Used by `u_trap`, `double_slit`, `drunken_bridge`.

## Gridworld discretization

`to_gridworld(n_rows, n_cols, alpha)` maps the continuous environment onto a finite-state MDP for the Chapter 7 examples. The discretization uses obstacle-only running costs (scaled relative to alpha so that the M matrix entries span a useful sub-stochastic range), a flat terminal cost for non-goal states, and an absorbing goal state. See the `lmdp/` README for details on the resulting MDP structure.

## Environment YAML format

Each YAML file in `config/environments/` specifies `start_pos`, `goal_pos`, weights (`goal_weight`, `velocity_weight`, `control_weight`), optional `boundary` dimensions, and a list of obstacles. A `compatible_with` field declares which example types the environment supports. Environments with thin box walls or many small circles are not suitable for coarse gridworld discretization.
