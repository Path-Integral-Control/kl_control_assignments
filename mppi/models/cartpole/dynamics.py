"""Cart-pole swing-up model for MPPI.

State:  [x, x_dot, theta, theta_dot]
  - x:         cart position (m)
  - x_dot:     cart velocity (m/s)
  - theta:     pole angle from upright (rad), 0 = upright, pi = hanging
  - theta_dot: pole angular velocity (rad/s)

Control: [F]
  - F: horizontal force on cart (N), bounds [-10, 10]

The swing-up is nonlinear and non-convex: from theta=pi (hanging),
the controller must pump energy before catching the pole upright.
LQR from the hanging position fails. Sampling finds the swing-up.
"""

import numpy as np
from mppi.models.base import DynamicsModel


class CartPole(DynamicsModel):

    def __init__(self, goal_theta=0.0, M=1.0, m=0.1, l=0.5, g=9.81,
                 w_angle=10.0, w_pos=1.0, w_vel=0.1, w_ctrl=0.01,
                 w_terminal=50.0, x_bounds=(-3.0, 3.0)):
        """
        Parameters
        ----------
        goal_theta : float
            Target pole angle (0 = upright).
        M : float
            Cart mass (kg).
        m : float
            Pole mass (kg).
        l : float
            Half pole length (m).
        g : float
            Gravitational acceleration (m/s^2).
        w_angle, w_pos, w_vel, w_ctrl, w_terminal : float
            Cost weights.
        x_bounds : tuple
            Cart position limits for boundary penalty.
        """
        self.goal_theta = goal_theta
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.w_angle = w_angle
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_ctrl = w_ctrl
        self.w_terminal = w_terminal
        self.x_bounds = x_bounds

    @property
    def state_dim(self):
        return 4

    @property
    def control_dim(self):
        return 1

    @property
    def control_bounds(self):
        return ([-10.0], [10.0])

    @property
    def noise_dim(self):
        return 1

    def dynamics(self, x, u, xp):
        """Cart-pole equations of motion (Lagrangian mechanics).

        x: (K, 4), u: (K, 1)
        Returns: xdot (K, 4)
        """
        pos = x[:, 0]
        vel = x[:, 1]
        theta = x[:, 2]
        omega = x[:, 3]
        F = u[:, 0]

        sin_th = xp.sin(theta)
        cos_th = xp.cos(theta)

        total_mass = self.M + self.m
        ml = self.m * self.l

        # Standard cart-pole dynamics
        temp = (F + ml * omega**2 * sin_th) / total_mass
        alpha_num = self.g * sin_th - cos_th * temp
        alpha_den = self.l * (4.0/3.0 - self.m * cos_th**2 / total_mass)
        theta_ddot = alpha_num / alpha_den

        x_ddot = temp - ml * theta_ddot * cos_th / total_mass

        xdot = xp.zeros_like(x)
        xdot[:, 0] = vel
        xdot[:, 1] = x_ddot
        xdot[:, 2] = omega
        xdot[:, 3] = theta_ddot

        return xdot

    def step(self, x, u, dt, xp):
        """RK4 integration."""
        k1 = self.dynamics(x, u, xp)
        k2 = self.dynamics(x + 0.5 * dt * k1, u, xp)
        k3 = self.dynamics(x + 0.5 * dt * k2, u, xp)
        k4 = self.dynamics(x + dt * k3, u, xp)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _angle_error(self, theta, xp):
        """Wrapped angle error from upright (goal_theta)."""
        err = theta - self.goal_theta
        return xp.arctan2(xp.sin(err), xp.cos(err))

    def running_cost(self, x, u, t, xp):
        """Cost per timestep.

        x: (K, 4), u: (K, 1)
        Returns: (K,)
        """
        angle_err = self._angle_error(x[:, 2], xp)
        cost = self.w_angle * angle_err**2
        cost += self.w_pos * x[:, 0]**2
        cost += self.w_vel * (x[:, 1]**2 + x[:, 3]**2)
        cost += self.w_ctrl * u[:, 0]**2

        # Boundary penalty on cart position
        lo, hi = self.x_bounds
        out_lo = xp.clip(lo - x[:, 0], 0, None)
        out_hi = xp.clip(x[:, 0] - hi, 0, None)
        cost += 100.0 * (out_lo**2 + out_hi**2)

        return cost

    def terminal_cost(self, x, xp):
        """Terminal cost emphasizing upright + centered.

        x: (K, 4)
        Returns: (K,)
        """
        angle_err = self._angle_error(x[:, 2], xp)
        cost = self.w_terminal * (self.w_angle * angle_err**2 + self.w_pos * x[:, 0]**2)
        return cost

    def drift(self, x, u, xp):
        """For SDE formulation: drift = f(x) + g(x)*u."""
        return self.dynamics(x, u, xp)

    def diffusion(self, x, xp):
        """Diffusion matrix G(x) for SDE: noise enters through force channel.

        Returns: (K, 4, 1)
        """
        K = x.shape[0]
        G = xp.zeros((K, 4, 1), dtype=x.dtype)

        total_mass = self.M + self.m
        cos_th = xp.cos(x[:, 2])
        den = self.l * (4.0/3.0 - self.m * cos_th**2 / total_mass)

        # Force noise maps through the same dynamics as control
        G[:, 1, 0] = 1.0 / total_mass
        G[:, 3, 0] = -cos_th / (total_mass * den)

        return G
