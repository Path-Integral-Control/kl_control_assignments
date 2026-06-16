"""Unicycle model: nonholonomic 2D vehicle.

State: [x, y, theta] (3D)
Control: [v, omega] (2D, forward speed and yaw rate)
"""

import numpy as np

from mppi.models.base import DynamicsModel


def _rk4(f, x, u, dt, xp):
    k1 = f(x, u, xp)
    k2 = f(x + 0.5 * dt * k1, u, xp)
    k3 = f(x + 0.5 * dt * k2, u, xp)
    k4 = f(x + dt * k3, u, xp)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _wrap_angle(theta, xp):
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


class Unicycle(DynamicsModel):
    """Unicycle with goal-seeking and obstacle avoidance.

    Nonholonomic dynamics produce richer MPPI behavior than the double
    integrator because the vehicle cannot strafe sideways.
    """

    def __init__(self, goal=None, obs=None):
        self.goal = np.array(goal if goal is not None else [5.0, 5.0], dtype=np.float32)
        self.obstacles = obs  # list of (cx, cy, radius) or None

        self.w_pos = 2.0
        self.w_heading = 0.5
        self.w_control = 0.01
        self.w_terminal = 100.0
        self.w_obstacle = 2000.0

        self._sigma_v = 0.3
        self._sigma_w = 0.5

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def control_bounds(self) -> tuple:
        lo = np.array([0.0, -4.0], dtype=np.float32)
        hi = np.array([2.0, 4.0], dtype=np.float32)
        return lo, hi

    def dynamics(self, x, u, xp):
        K = x.shape[0]
        theta = x[:, 2]
        v = u[:, 0]
        omega = u[:, 1]

        xdot = xp.zeros((K, 3), dtype=xp.float32)
        xdot[:, 0] = v * xp.cos(theta)
        xdot[:, 1] = v * xp.sin(theta)
        xdot[:, 2] = omega
        return xdot

    def step(self, x, u, dt, xp):
        x_next = _rk4(self.dynamics, x, u, dt, xp)
        x_next[:, 2] = _wrap_angle(x_next[:, 2], xp)
        return x_next

    def _obstacle_cost(self, x, xp):
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
        dist = xp.sqrt(dx**2 + dy**2)

        # Heading error: angle between current heading and bearing to goal
        bearing = xp.arctan2(goal[1] - x[:, 1], goal[0] - x[:, 0])
        heading_err = _wrap_angle(x[:, 2] - bearing, xp)

        ctrl_sq = u[:, 0] ** 2 + u[:, 1] ** 2

        cost = (
            self.w_pos * dist
            + self.w_heading * heading_err**2
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
    # Noncommutative noise structure matching the MATLAB unicycle example.
    # Velocity noise enters along the heading direction, yaw noise is additive.

    @property
    def noise_dim(self) -> int:
        return 2

    def diffusion(self, x, xp):
        K = x.shape[0]
        theta = x[:, 2]
        G = xp.zeros((K, self.state_dim, self.noise_dim), dtype=xp.float32)
        G[:, 0, 0] = self._sigma_v * xp.cos(theta)
        G[:, 1, 0] = self._sigma_v * xp.sin(theta)
        G[:, 2, 1] = self._sigma_w
        return G
