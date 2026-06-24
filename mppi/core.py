"""MPPI solver supporting discrete-time and continuous-time (SDE) formulations.

Implements the Model Predictive Path Integral algorithm from Williams et al.
(2017). The solver generates K noisy rollouts around a nominal control
sequence, weights them by exponentiated negative cost, and produces a
weighted-average optimal control.
"""

import numpy as np

from mppi.backend import get_backend, set_backend


class MPPI:
    """Sampling-based MPC using path integral control theory.

    Parameters
    ----------
    model : DynamicsModel
        Dynamics model with step(), running_cost(), and terminal_cost().
    K : int
        Number of rollout samples.
    T : int
        Planning horizon in timesteps.
    dt : float
        Timestep duration.
    lambda_ : float
        Temperature parameter. Lower values make the controller more greedy
        (concentrating weight on low-cost trajectories).
    sigma : array-like of shape (control_dim,)
        Standard deviation of control noise for each control channel.
    use_gpu : bool
        If True, use CuPy for GPU-accelerated rollouts.
    warm_start : bool
        If True, shift the previous solution forward by one step at each call.
    """

    def __init__(self, model, K, T, dt, lambda_, sigma, use_gpu=False, warm_start=True, dtype=None):
        set_backend(use_gpu)
        self.xp = get_backend()
        xp = self.xp

        self.model = model
        self.K = K
        self.T = T
        self.dt = dt
        self.lambda_ = lambda_
        self.dtype = dtype or xp.float32
        self.sigma = xp.array(sigma, dtype=self.dtype)
        self.warm_start = warm_start

        self.U = xp.zeros((T, model.control_dim), dtype=self.dtype)

        self.trajectories = None
        self.weights = None
        self.costs = None

    def _compute_weights(self, costs):
        """Importance sampling weights from trajectory costs."""
        xp = self.xp

        # ##########################################################
        # TODO: Compute importance sampling weights from
        # trajectory costs. Subtract the minimum cost for
        # numerical stability, then exponentiate the negative
        # costs scaled by self.lambda_. Normalize weights to
        # sum to 1. If the total weight is near zero, return
        # uniform weights.
        #
        # ##########################################################
        # raise NotImplementedError("TODO: _compute_weights")

        xp = self.xp
        beta = xp.min(costs)
        weights = xp.exp(-(costs - beta) / self.lambda_)
        weights_sum = xp.sum(weights)
        if weights_sum < 1e-10:
            return xp.ones(self.K, dtype=self.dtype) / self.K
        return weights / weights_sum

    def _warm_start_shift(self):
        """Shift the nominal control sequence left by one step."""
        if not self.warm_start:
            return
        self.U = self.xp.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0

    def _generate_perturbed_controls(self):
        """Sample K perturbed control sequences around the nominal."""
        xp = self.xp
        epsilon = xp.random.randn(self.K, self.T, self.model.control_dim).astype(self.dtype)
        U_perturbed = self.U[None, :, :] + self.sigma[None, None, :] * epsilon

        lo, hi = self.model.control_bounds
        lo = xp.array(lo, dtype=self.dtype)
        hi = xp.array(hi, dtype=self.dtype)
        return xp.clip(U_perturbed, lo, hi)

    def solve(self, x0):
        """Discrete-time MPPI. Returns optimal control sequence of shape (T, control_dim).

        Rolls out K trajectories using model.step(), computes importance
        sampling weights from the costs, and returns the weighted-average
        control sequence.
        """
        xp = self.xp
        K, T = self.K, self.T

        self._warm_start_shift()
        U_perturbed = self._generate_perturbed_controls()

        states = xp.zeros((K, T + 1, self.model.state_dim), dtype=self.dtype)
        x0_arr = xp.array(x0, dtype=self.dtype)
        states[:, 0] = xp.tile(x0_arr, (K, 1))

        costs = xp.zeros(K, dtype=self.dtype)

        for t in range(T):
            # ##########################################################
            # TODO: Propagate states forward using self.model.step()
            # and accumulate running costs via self.model.running_cost().
            #
            # ##########################################################
            # raise NotImplementedError("TODO: solve forward step")
            states[:, t + 1] = self.model.step(states[:, t], U_perturbed[:, t], self.dt, xp)
            costs += self.model.running_cost(states[:, t + 1], U_perturbed[:, t], t, xp)
            if hasattr(self.model, 'clamp_state'):
                states[:, t + 1] = self.model.clamp_state(states[:, t + 1], xp)
        
        costs += self.model.terminal_cost(states[:, -1], xp)

        # ##########################################################
        # TODO: Add terminal cost, compute importance weights via
        # self._compute_weights(), and update self.U as the
        # weighted average of U_perturbed.
        #
        # ##########################################################
        # raise NotImplementedError("TODO: solve update")

        weights = self._compute_weights(costs)
        self.U = xp.sum(weights[:, None, None] * U_perturbed, axis=0)

        self.trajectories = states
        self.weights = weights
        self.costs = costs

        return self.U

    def solve_continuous(self, x0):
        """Continuous-time MPPI using Euler-Maruyama integration.

        For models with an SDE formulation dx = f(x,u)dt + G(x)dW, this
        variant injects process noise through the diffusion matrix rather
        than through additive control noise alone.
        """
        xp = self.xp
        K, T = self.K, self.T

        self._warm_start_shift()
        U_perturbed = self._generate_perturbed_controls()

        states = xp.zeros((K, T + 1, self.model.state_dim), dtype=self.dtype)
        x0_arr = xp.array(x0, dtype=self.dtype)
        states[:, 0] = xp.tile(x0_arr, (K, 1))

        costs = xp.zeros(K, dtype=self.dtype)
        noise_dim = self.model.noise_dim

        for t in range(T):
            x_t = states[:, t]
            u_t = U_perturbed[:, t]

            # ##########################################################
            # TODO: Propagate states forward using self.model.step(),
            # inject process noise through the diffusion matrix G(x)
            # via self.model.diffusion(x_t, xp), and accumulate
            # running costs via self.model.running_cost().
            #
            # ##########################################################
            # raise NotImplementedError("TODO: solve_continuous forward step")

            x_next = self.model.step(x_t, u_t, self.dt, xp)

            if noise_dim > 0:
                B = self.model.diffusion(x_t, xp)
                dW = xp.sqrt(self.dt) * xp.random.randn(K, noise_dim, 1).astype(self.dtype)
                x_next = x_next + (B @ dW).squeeze(-1)

            costs += self.model.running_cost(x_next, u_t, t, xp)

            if hasattr(self.model, 'clamp_state'):
                x_next = self.model.clamp_state(x_next, xp)
            states[:, t + 1] = x_next

        # ##########################################################
        # TODO: Add terminal cost, compute importance weights via
        # self._compute_weights(), and update self.U as the
        # weighted average of U_perturbed.
        #
        # ##########################################################
        # raise NotImplementedError("TODO: solve_continuous update")

        costs += self.model.terminal_cost(states[:, -1], xp)

        weights = self._compute_weights(costs)
        self.U = xp.sum(weights[:, None, None] * U_perturbed, axis=0)
        
        self.trajectories = states
        self.weights = weights
        self.costs = costs

        return self.U

    def get_optimal_trajectory(self):
        """Return the highest-weight trajectory from the last solve call."""
        if self.trajectories is None:
            return None
        xp = self.xp
        best_idx = xp.argmax(self.weights)
        if hasattr(best_idx, "item"):
            best_idx = best_idx.item()
        return self.trajectories[best_idx]

    def get_planned_trajectory(self, x0):
        """Roll out the optimal control sequence from x0 through the dynamics.

        Returns the trajectory of shape (T+1, state_dim) that the controller
        would actually follow if it applied the full horizon of controls.
        """
        xp = self.xp
        traj = xp.zeros((self.T + 1, self.model.state_dim), dtype=self.dtype)
        traj[0] = xp.array(x0, dtype=self.dtype)
        for t in range(self.T):
            x_t = traj[t][None, :]
            u_t = self.U[t][None, :]
            traj[t + 1] = self.model.step(x_t, u_t, self.dt, xp).squeeze(0)
        return traj

    def get_sampled_trajectories(self):
        """Return all K sampled trajectories from the last solve. Shape (K, T+1, state_dim)."""
        return self.trajectories

    def get_weights(self):
        """Return importance sampling weights from the last solve. Shape (K,)."""
        return self.weights

    def get_costs(self):
        """Return per-sample costs from the last solve. Shape (K,)."""
        return self.costs
