"""Environment: loads obstacle fields from YAML configs, computes costs.

Obstacle types supported:
  - gaussian: soft bump, C = amplitude * exp(-dist^2 / (2*spread^2))
  - circle:   hard boundary, C = cost if inside radius
  - box:      hard rectangle, C = cost if inside bounds
  - forest:   procedural circle obstacles via rejection sampling

Key methods:
  from_yaml(path)              — load environment from a YAML config file
  get_state_cost(states)       — vectorized cost over obstacles + goal
  get_cost(states, actions)    — state cost + control cost
  cost_field(xx, yy)           — evaluate C(x,y) on a meshgrid (for plotting / Ch.6)
  to_gridworld(n_rows, n_cols, alpha) — discretize into a GridWorld for Ch.7
"""

import os
import numpy as np
import yaml

from environments.procedural import generate_forest


class Environment:

    def __init__(self, config):
        """Initialize from a config dict (typically loaded from YAML).

        Parameters
        ----------
        config : dict
            Keys: start_pos, goal_pos, goal_weight, velocity_weight,
                  control_weight, boundary, obstacles
        """
        self.compatible_with = config.get('compatible_with', [])
        self.start_pos = np.array(config['start_pos'], dtype=float)
        self.goal_pos = np.array(config['goal_pos'], dtype=float)
        self.goal_weight = config.get('goal_weight', 20.0)
        self.control_weight = config.get('control_weight', 0.5)
        self.velocity_weight = config.get('velocity_weight', 5.0)

        self.boundary = config.get('boundary', {'enabled': False})
        self.y_center = (self.start_pos[1] + self.goal_pos[1]) / 2.0
        self.x_center = (self.start_pos[0] + self.goal_pos[0]) / 2.0

        raw_obstacles = config.get('obstacles', [])
        self.obstacles = []

        for obs in raw_obstacles:
            if obs.get('type') == 'forest':
                self.obstacles.extend(
                    generate_forest(obs, self.start_pos, self.goal_pos))
            else:
                self.obstacles.append(obs)

    @classmethod
    def from_yaml(cls, path):
        """Load environment from a YAML config file."""
        if not os.path.isfile(path):
            candidate = os.path.join('config', 'environments', path if path.endswith('.yaml') else path + '.yaml')
            if os.path.isfile(candidate):
                path = candidate
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def check_compatibility(self, mode):
        """Check if this environment is compatible with a given mode.

        Parameters
        ----------
        mode : str
            One of: 'variational', 'gridworld', 'mppi_navigation', 'spine'

        Raises
        ------
        ValueError
            If the environment YAML declares incompatibility.
        """
        if self.compatible_with and mode not in self.compatible_with:
            raise ValueError(
                f"Environment is not compatible with '{mode}'. "
                f"Compatible modes: {self.compatible_with}. "
                f"Use an environment with gaussian obstacles for gridworld/spine.")

    def get_state_cost(self, states):
        """Compute state-dependent cost. Vectorized over arbitrary batch dims.

        Parameters
        ----------
        states : array of shape (..., state_dim)
            First 2 dims are always [x, y]. If state_dim >= 4,
            dims 2-3 are [vx, vy].

        Returns
        -------
        cost : array of shape (...)
        """
        x = states[..., 0]
        y = states[..., 1]

        dist_to_goal_sq = (x - self.goal_pos[0])**2 + (y - self.goal_pos[1])**2
        cost = self.goal_weight * dist_to_goal_sq

        if states.shape[-1] >= 4:
            vx = states[..., 2]
            vy = states[..., 3]
            cost += self.velocity_weight * (vx**2 + vy**2)

        for obs in self.obstacles:
            obs_type = obs.get('type', 'gaussian')

            if obs_type == 'gaussian':
                mx, my = obs['position']
                amp = obs['amplitude']
                spread = obs['spread']
                cost += amp * np.exp(-((x - mx)**2 + (y - my)**2) / (2.0 * spread**2))

            elif obs_type == 'circle':
                mx, my = obs['position']
                r = obs['radius']
                circle_cost = obs['cost']
                dist_sq = (x - mx)**2 + (y - my)**2
                cost += np.where(dist_sq <= r**2, circle_cost, 0.0)

            elif obs_type == 'box':
                mx, my = obs['position']
                w, h = obs['size']
                box_cost = obs['cost']
                in_box = (
                    (x >= mx - w / 2.0) & (x <= mx + w / 2.0) &
                    (y >= my - h / 2.0) & (y <= my + h / 2.0)
                )
                cost += np.where(in_box, box_cost, 0.0)

                buffer_dist = obs.get('buffer', 0.1)
                if buffer_dist > 0:
                    dist_x = np.maximum(0, np.abs(x - mx) - w / 2.0)
                    dist_y = np.maximum(0, np.abs(y - my) - h / 2.0)
                    dist_to_box = np.sqrt(dist_x**2 + dist_y**2)
                    margin_cost = (box_cost * 0.1) * np.exp(-3.0 * dist_to_box / buffer_dist)
                    cost += np.where(~in_box, margin_cost, 0.0)

        if self.boundary.get('enabled', False):
            width = self.boundary.get('width', 10.0)
            height = self.boundary.get('height', 10.0)
            outside = (
                (np.abs(y - self.y_center) > height / 2.0) |
                (np.abs(x - self.x_center) > width / 2.0)
            )
            cost += np.where(outside, 1e12, 0.0)

        return cost

    def get_obstacle_cost(self, states):
        """Compute obstacle-only cost (no goal, no velocity). Same shape as get_state_cost."""
        x = states[..., 0]
        y = states[..., 1]
        cost = np.zeros_like(x, dtype=float)

        for obs in self.obstacles:
            obs_type = obs.get('type', 'gaussian')

            if obs_type == 'gaussian':
                mx, my = obs['position']
                amp = obs['amplitude']
                spread = obs['spread']
                cost += amp * np.exp(-((x - mx)**2 + (y - my)**2) / (2.0 * spread**2))

            elif obs_type == 'circle':
                mx, my = obs['position']
                r = obs['radius']
                circle_cost = obs['cost']
                dist_sq = (x - mx)**2 + (y - my)**2
                cost += np.where(dist_sq <= r**2, circle_cost, 0.0)

            elif obs_type == 'box':
                mx, my = obs['position']
                w, h = obs['size']
                box_cost = obs['cost']
                in_box = (
                    (x >= mx - w / 2.0) & (x <= mx + w / 2.0) &
                    (y >= my - h / 2.0) & (y <= my + h / 2.0)
                )
                cost += np.where(in_box, box_cost, 0.0)

                buffer_dist = obs.get('buffer', 0.1)
                if buffer_dist > 0:
                    dist_x = np.maximum(0, np.abs(x - mx) - w / 2.0)
                    dist_y = np.maximum(0, np.abs(y - my) - h / 2.0)
                    dist_to_box = np.sqrt(dist_x**2 + dist_y**2)
                    margin_cost = (box_cost * 0.1) * np.exp(-3.0 * dist_to_box / buffer_dist)
                    cost += np.where(~in_box, margin_cost, 0.0)

        if self.boundary.get('enabled', False):
            width = self.boundary.get('width', 10.0)
            height = self.boundary.get('height', 10.0)
            outside = (
                (np.abs(y - self.y_center) > height / 2.0) |
                (np.abs(x - self.x_center) > width / 2.0)
            )
            cost += np.where(outside, 1e12, 0.0)

        return cost

    def clamp_to_obstacles(self, states):
        """Project states out of obstacles. Modifies x,y in-place.

        For circles: push to surface along radial direction.
        For boxes: push to nearest edge.
        """
        x = states[..., 0]
        y = states[..., 1]

        for obs in self.obstacles:
            obs_type = obs.get('type', 'gaussian')

            if obs_type == 'circle':
                mx, my = obs['position']
                r = obs['radius']
                dx = x - mx
                dy = y - my
                dist = np.sqrt(dx**2 + dy**2)
                inside = dist < r
                if np.any(inside):
                    scale = np.where(inside & (dist > 1e-8), r / dist, 1.0)
                    states[..., 0] = np.where(inside, mx + dx * scale, x)
                    states[..., 1] = np.where(inside, my + dy * scale, y)
                    x = states[..., 0]
                    y = states[..., 1]

            elif obs_type == 'box':
                mx, my = obs['position']
                w, h = obs['size']
                x0, x1 = mx - w / 2.0, mx + w / 2.0
                y0, y1 = my - h / 2.0, my + h / 2.0
                in_box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                if np.any(in_box):
                    dl = x - x0
                    dr = x1 - x
                    db = y - y0
                    dt_ = y1 - y
                    min_pen = np.minimum(np.minimum(dl, dr), np.minimum(db, dt_))
                    push_left = in_box & (dl == min_pen)
                    push_right = in_box & (dr == min_pen) & ~push_left
                    push_down = in_box & (db == min_pen) & ~push_left & ~push_right
                    push_up = in_box & ~push_left & ~push_right & ~push_down
                    states[..., 0] = np.where(push_left, x0, x)
                    states[..., 0] = np.where(push_right, x1, states[..., 0])
                    states[..., 1] = np.where(push_down, y0, y)
                    states[..., 1] = np.where(push_up, y1, states[..., 1])
                    x = states[..., 0]
                    y = states[..., 1]

        if self.boundary.get('enabled', False):
            bw = self.boundary.get('width', 10.0)
            bh = self.boundary.get('height', 10.0)
            states[..., 0] = np.clip(x, self.x_center - bw/2, self.x_center + bw/2)
            states[..., 1] = np.clip(y, self.y_center - bh/2, self.y_center + bh/2)

        return states

    def get_control_cost(self, actions):
        """Control cost: quadratic penalty on action magnitude."""
        return self.control_weight * np.sum(actions**2, axis=-1)

    def get_cost(self, states, actions):
        """Total step cost = state cost + control cost."""
        return self.get_state_cost(states) + self.get_control_cost(actions)

    def draw_cost_contours(self, ax, bounds=None, res=200, levels=12,
                           colors='black', alpha=0.3, linewidths=0.5):
        """Overlay cost field contour lines on an axis."""
        if bounds is None:
            bw = self.boundary.get('width', 14.0)
            bh = self.boundary.get('height', 14.0)
            bounds = (self.x_center - bw/2, self.x_center + bw/2,
                      self.y_center - bh/2, self.y_center + bh/2)
        xx, yy = np.meshgrid(
            np.linspace(bounds[0], bounds[1], res),
            np.linspace(bounds[2], bounds[3], res))
        C = self.cost_field(xx, yy)
        ax.contour(xx, yy, C, levels=levels, colors=colors,
                   linewidths=linewidths, alpha=alpha)

    def cost_field(self, xx, yy):
        """Evaluate C(x,y) on a meshgrid. For plotting and Ch.6 desirability.

        Parameters
        ----------
        xx, yy : 2D arrays from np.meshgrid

        Returns
        -------
        C : 2D array, same shape as xx
        """
        states = np.stack([xx, yy], axis=-1)
        return self.get_state_cost(states)

    def to_gridworld(self, n_rows, n_cols, alpha=5.0):
        """Discretize this environment into a GridWorld for Ch.7.

        Maps the continuous cost field onto a finite grid. Each cell's
        running cost is proportional to the continuous cost at its center.

        Parameters
        ----------
        n_rows, n_cols : int
            Grid dimensions.
        alpha : float
            Temperature (used to scale costs appropriately).

        Returns
        -------
        grid : GridWorld
        start_state : int
            Grid state corresponding to start_pos.
        """
        from lmdp.gridworld import GridWorld

        x_min = self.x_center - self.boundary.get('width', 14.0) / 2.0
        x_max = self.x_center + self.boundary.get('width', 14.0) / 2.0
        y_min = self.y_center - self.boundary.get('height', 14.0) / 2.0
        y_max = self.y_center + self.boundary.get('height', 14.0) / 2.0

        cell_w = (x_max - x_min) / n_cols
        cell_h = (y_max - y_min) / n_rows

        obstacles = set()
        obs_costs = np.zeros((n_rows, n_cols))

        for r in range(n_rows):
            for c in range(n_cols):
                cx = x_min + (c + 0.5) * cell_w
                cy = y_max - (r + 0.5) * cell_h
                states = np.array([[[cx, cy]]])
                obs_costs[r, c] = float(self.get_obstacle_cost(states[0])[0])
                if obs_costs[r, c] > 1e6:
                    obstacles.add((r, c))

        # Scale costs so M = exp(-C/alpha) spans a useful range:
        # Empty cells:   cost = alpha*0.2 → M ≈ 0.82 (sub-stochastic diffusion)
        # Mountain peaks: cost = alpha*5.0 → M ≈ 0.007 (effectively blocked)
        valid = obs_costs[obs_costs < 1e6]
        C_max = valid.max() if len(valid) > 0 else 1.0
        cost_floor = alpha * 0.2
        cost_ceil = alpha * 5.0
        cost_scale = max(C_max / (cost_ceil - cost_floor), 1.0)

        goal_r = int((y_max - self.goal_pos[1]) / cell_h)
        goal_c = int((self.goal_pos[0] - x_min) / cell_w)
        goal_r = np.clip(goal_r, 0, n_rows - 1)
        goal_c = np.clip(goal_c, 0, n_cols - 1)

        start_r = int((y_max - self.start_pos[1]) / cell_h)
        start_c = int((self.start_pos[0] - x_min) / cell_w)
        start_r = np.clip(start_r, 0, n_rows - 1)
        start_c = np.clip(start_c, 0, n_cols - 1)

        grid = GridWorld(
            n_rows=n_rows, n_cols=n_cols,
            obstacles=obstacles,
            goal=(goal_r, goal_c),
            step_cost=1.0,
            obstacle_cost=100.0,
            default_exit_cost=200.0,
        )

        goal_state = grid.rc_to_state(goal_r, goal_c)

        def field_cost(state, action):
            if state == goal_state:
                return 0.0
            next_s = grid.step(state, action)
            ri, ci = grid.state_to_rc(next_s)
            return obs_costs[ri, ci] / cost_scale + cost_floor
        grid.cost = field_cost

        orig_step = grid.step
        def absorbing_step(state, action):
            if state == goal_state:
                return goal_state
            return orig_step(state, action)
        grid.step = absorbing_step

        exit_cost = 20.0 * alpha
        def field_terminal_cost(state):
            r, c = grid.state_to_rc(state)
            if (r, c) == grid.goal:
                return 0.0
            return exit_cost
        grid.terminal_cost = field_terminal_cost

        # Store metadata for plotting
        grid._env_bounds = (x_min, x_max, y_min, y_max)
        grid._cell_w = cell_w
        grid._cell_h = cell_h
        grid._cell_costs = obs_costs

        start_state = grid.rc_to_state(start_r, start_c)
        return grid, start_state
