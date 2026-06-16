"""Finite-state gridworld environment for KL control (Chapter 7).

States are cells on an N x N grid, indexed 0..N*N-1 (row-major).
Actions are {up, down, left, right, stay}.
Transitions are deterministic: x_{t+1} = F(x_t, u_t).

The KL control problem (Eq. 7.9):
  min_Q  E^Q [ sum C_t(x,u) + alpha * D(Q||R) ] + E^Q [ C_exit(x_T) ]
"""

import numpy as np


ACTIONS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
    4: (-1, -1),  # up-left
    5: (-1, 1),   # up-right
    6: (1, -1),   # down-left
    7: (1, 1),    # down-right
    8: (0, 0),    # stay
}
ACTION_NAMES = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right', 'stay']
N_ACTIONS = len(ACTIONS)


class GridWorld:
    """Discrete-state environment for KL control examples.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    obstacles : set of (row, col)
        Cells with high traversal cost.
    goal : (row, col)
        Goal cell with C_exit = 0.
    step_cost : float
        Running cost for each action in a free cell.
    obstacle_cost : float
        Running cost for actions entering an obstacle cell.
    goal_exit_cost : float
        Terminal cost at the goal (typically 0).
    default_exit_cost : float
        Terminal cost for non-goal cells.
    """

    def __init__(self, n_rows, n_cols, obstacles, goal,
                 step_cost=1.0, obstacle_cost=100.0,
                 goal_exit_cost=0.0, default_exit_cost=200.0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols
        self.n_actions = N_ACTIONS
        self.obstacles = set(obstacles)
        self.goal = goal
        self.step_cost = step_cost
        self.obstacle_cost = obstacle_cost
        self.goal_exit_cost = goal_exit_cost
        self.default_exit_cost = default_exit_cost

    def rc_to_state(self, row, col):
        return row * self.n_cols + col

    def state_to_rc(self, state):
        return divmod(state, self.n_cols)

    def step(self, state, action):
        """Deterministic transition F(x, u). (Assumption 7.1)"""
        r, c = self.state_to_rc(state)
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
            return self.rc_to_state(nr, nc)
        return state

    def step_stochastic(self, state, action, slip_prob=0.2, rng=None):
        """Stochastic transition: intended action with prob (1-slip),
        uniform random action with prob slip. For Exercise 7.6."""
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < slip_prob:
            action = rng.integers(0, self.n_actions)
        return self.step(state, action)

    def cost(self, state, action):
        """Running cost C(x, u). Diagonal moves cost sqrt(2) times more."""
        next_s = self.step(state, action)
        r, c = self.state_to_rc(next_s)
        dr, dc = ACTIONS[action]
        dist = np.sqrt(dr**2 + dc**2) if (dr != 0 or dc != 0) else 1.0
        if (r, c) in self.obstacles:
            return self.obstacle_cost * dist
        return self.step_cost * dist

    def terminal_cost(self, state):
        """Terminal cost C_exit(x). Zero at goal, high elsewhere."""
        r, c = self.state_to_rc(state)
        if (r, c) == self.goal:
            return self.goal_exit_cost
        return self.default_exit_cost

    def reference_policy(self, state):
        """Reference policy R(u|x) — uniform over all actions."""
        return np.ones(self.n_actions) / self.n_actions

    def is_obstacle(self, row, col):
        return (row, col) in self.obstacles


def obstacle_grid():
    """10x10 grid with scattered obstacles. Start=(9,0), Goal=(0,9).

    Layout:
      . . . # . . . . . G
      . . . . . . # . . .
      . # . . . . . . . .
      . . . . # . . . # .
      . . . . . . . . . .
      . . # . . . . . . .
      . . . . . . # . . .
      . . . . # . . . . .
      . . . . . . . . . .
      S . . . . . . . . .
    """
    obstacles = {
        (0, 3), (1, 6), (2, 1), (3, 4), (3, 8),
        (5, 2), (6, 6), (7, 4),
    }
    return GridWorld(
        n_rows=10, n_cols=10,
        obstacles=obstacles,
        goal=(0, 9),
        step_cost=1.0,
        obstacle_cost=100.0,
        default_exit_cost=200.0,
    )


def small_grid():
    """5x5 grid for the determinism-breaking example. Start=(4,0), Goal=(0,4).

    Layout:
      . . . . G
      . # . . .
      . . . . .
      . . . # .
      S . . . .
    """
    obstacles = {(1, 1), (3, 3)}
    return GridWorld(
        n_rows=5, n_cols=5,
        obstacles=obstacles,
        goal=(0, 4),
        step_cost=1.0,
        obstacle_cost=50.0,
        default_exit_cost=100.0,
    )
