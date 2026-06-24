"""Integration utilities for MPPI trajectory rollouts.

All functions accept an `xp` parameter (numpy or cupy) and operate
on batched states of shape (K, state_dim).
"""

import numpy as np


def euler_step(dynamics_fn, x, u, dt, xp):
    """Forward Euler: x_next = x + f(x, u) * dt."""
    # ##########################################################
    # TODO: Implement forward Euler integration
    #
    # Evaluate the dynamics f(x, u) and take one step of size dt.
    # ##########################################################
    # raise NotImplementedError("TODO: euler_step")

    return x + dynamics_fn(x, u, xp) * dt


def rk4_step(dynamics_fn, x, u, dt, xp):
    """Fourth-order Runge-Kutta with fixed control over the step."""
    k1 = dynamics_fn(x, u, xp)
    k2 = dynamics_fn(x + 0.5 * dt * k1, u, xp)
    k3 = dynamics_fn(x + 0.5 * dt * k2, u, xp)
    k4 = dynamics_fn(x + dt * k3, u, xp)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_substep(
    dynamics_fn,
    x,
    u,
    dt,
    n_substeps,
    xp,
    clamp_state_fn=None,
    clamp_deriv_fn=None,
    normalize_fn=None,
):
    """RK4 with multiple substeps for stiff dynamics.

    Each intermediate state is normalized (quaternions) and clamped
    (angle wrapping, velocity limits) between substeps to keep the
    integration stable.
    """
    dt_sub = dt / n_substeps

    if normalize_fn is not None:
        x = normalize_fn(x, xp)
    if clamp_state_fn is not None:
        x = clamp_state_fn(x, xp)

    for _ in range(n_substeps):
        k1 = dynamics_fn(x, u, xp)
        if clamp_deriv_fn is not None:
            k1 = clamp_deriv_fn(k1, xp)

        x_tmp = x + 0.5 * dt_sub * k1
        if normalize_fn is not None:
            x_tmp = normalize_fn(x_tmp, xp)
        if clamp_state_fn is not None:
            x_tmp = clamp_state_fn(x_tmp, xp)

        k2 = dynamics_fn(x_tmp, u, xp)
        if clamp_deriv_fn is not None:
            k2 = clamp_deriv_fn(k2, xp)

        x_tmp = x + 0.5 * dt_sub * k2
        if normalize_fn is not None:
            x_tmp = normalize_fn(x_tmp, xp)
        if clamp_state_fn is not None:
            x_tmp = clamp_state_fn(x_tmp, xp)

        k3 = dynamics_fn(x_tmp, u, xp)
        if clamp_deriv_fn is not None:
            k3 = clamp_deriv_fn(k3, xp)

        x_tmp = x + dt_sub * k3
        if normalize_fn is not None:
            x_tmp = normalize_fn(x_tmp, xp)
        if clamp_state_fn is not None:
            x_tmp = clamp_state_fn(x_tmp, xp)

        k4 = dynamics_fn(x_tmp, u, xp)
        if clamp_deriv_fn is not None:
            k4 = clamp_deriv_fn(k4, xp)

        x = x + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if normalize_fn is not None:
            x = normalize_fn(x, xp)
        if clamp_state_fn is not None:
            x = clamp_state_fn(x, xp)

    return x


def euler_maruyama_step(drift_fn, diffusion_fn, x, u, dt, xp):
    """Euler-Maruyama integrator for SDEs: dx = f(x,u)dt + G(x)dW.

    Used in continuous-time MPPI where process noise models disturbances.
    The noise dimension is inferred from the diffusion matrix shape.
    """
    # ##########################################################
    # TODO: Implement Euler-Maruyama integration
    #
    # Discretize the SDE dx = f(x,u)dt + G(x)dW.
    # Use drift_fn(x, u, xp) for f, diffusion_fn(x, xp) for G.
    # Sample Brownian increments of the correct variance.
    # ##########################################################
    #raise NotImplementedError("TODO: euler_maruyama_step")

    drift = drift_fn(x, u, xp)
    B = diffusion_fn(x, xp)
    noise_dim = B.shape[2]

    dW = xp.sqrt(dt) * xp.random.randn(x.shape[0], noise_dim, 1).astype(xp.float32)
    return x + drift * dt + (B @ dW).squeeze(-1)


def normalize_quaternion(x, quat_indices, xp):
    """Normalize the quaternion stored at quat_indices within each state vector.

    If the quaternion norm is degenerate (< 1e-8), resets to identity [1, 0, 0, 0].
    Operates in-place on x and returns the modified array.
    """
    idx = list(quat_indices)
    q = x[:, idx]
    norm = xp.linalg.norm(q, axis=1, keepdims=True)

    identity = xp.zeros_like(q)
    identity[:, 0] = 1.0

    valid = (norm > 1e-8).squeeze()
    q_normed = q / xp.maximum(norm, 1e-8)
    x[:, idx] = xp.where(valid[:, None], q_normed, identity)
    return x


def wrap_angle(angle, xp):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
