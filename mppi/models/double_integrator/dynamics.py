"""Double integrator: 2D point mass with acceleration control.

State: [x, y, vx, vy] (4D)
Control: [ax, ay] (2D, acceleration commands)

A minimal model for testing MPPI before moving to more complex dynamics.
"""

import numpy as np

from mppi.models.base import DynamicsModel


def _rk4(f, x, u, dt, xp):
    """Standard fourth-order Runge-Kutta integrator."""
    k1 = f(x, u, xp)
    k2 = f(x + 0.5 * dt * k1, u, xp)
    k3 = f(x + 0.5 * dt * k2, u, xp)
    k4 = f(x + dt * k3, u, xp)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class DoubleIntegrator(DynamicsModel):
    """2D double integrator with obstacle avoidance."""

    def __init__(self, goal=None, obs=None):
        self.goal = np.array(goal if goal is not None else [5.0, 5.0], dtype=np.float32)
        self.obstacles = obs  # list of (cx, cy, radius) or None

        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_control = 0.01
        self.w_terminal = 50.0
        self.w_obstacle = 1000.0

        self._sigma_v = 0.5

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def control_bounds(self) -> tuple:
        lo = np.array([-5.0, -5.0], dtype=np.float32)
        hi = np.array([5.0, 5.0], dtype=np.float32)
        return lo, hi

    def dynamics(self, x, u, xp):
        K = x.shape[0]
        xdot = xp.zeros((K, 4), dtype=xp.float32)
        xdot[:, 0] = x[:, 2]  # dx/dt = vx
        xdot[:, 1] = x[:, 3]  # dy/dt = vy
        xdot[:, 2] = u[:, 0]  # dvx/dt = ax
        xdot[:, 3] = u[:, 1]  # dvy/dt = ay
        return xdot

    def step(self, x, u, dt, xp):
        return _rk4(self.dynamics, x, u, dt, xp)

    def _obstacle_cost(self, x, xp):
        """Soft penalty for penetrating obstacle interiors."""
        if self.obstacles is None:
            return xp.zeros(x.shape[0], dtype=xp.float32)
        cost = xp.zeros(x.shape[0], dtype=xp.float32)
        for cx, cy, radius in self.obstacles:
            dx = x[:, 0] - cx
            dy = x[:, 1] - cy
            dist = xp.sqrt(dx**2 + dy**2)
            penetration = xp.maximum(radius - dist, 0.0)
            cost = cost + self.w_obstacle * penetration**2
        return cost

    def running_cost(self, x, u, t, xp):
        goal = xp.asarray(self.goal, dtype=xp.float32)
        dx = x[:, 0] - goal[0]
        dy = x[:, 1] - goal[1]
        dist_sq = dx**2 + dy**2
        speed_sq = x[:, 2] ** 2 + x[:, 3] ** 2
        ctrl_sq = u[:, 0] ** 2 + u[:, 1] ** 2

        cost = (
            self.w_pos * dist_sq
            + self.w_vel * speed_sq
            + self.w_control * ctrl_sq
            + self._obstacle_cost(x, xp)
        )
        return cost

    def terminal_cost(self, x, xp):
        goal = xp.asarray(self.goal, dtype=xp.float32)
        dx = x[:, 0] - goal[0]
        dy = x[:, 1] - goal[1]
        return self.w_terminal * (dx**2 + dy**2)

    # -- Continuous-time SDE interface --

    @property
    def noise_dim(self) -> int:
        return 2

    def diffusion(self, x, xp):
        """Process noise enters through the velocity channels."""
        K = x.shape[0]
        G = xp.zeros((K, self.state_dim, self.noise_dim), dtype=xp.float32)
        G[:, 2, 0] = self._sigma_v  # noise in vx
        G[:, 3, 1] = self._sigma_v  # noise in vy
        return G
