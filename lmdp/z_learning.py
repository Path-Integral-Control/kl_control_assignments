"""Infinite-horizon KL control via Z-iteration (Chapter 7).

For the infinite-horizon linearly solvable MDP, the optimal desirability
Z* satisfies the fixed-point equation Z = M @ Z with an absorbing goal.

Key equations:
  (I - M_red) Z_red = m_goal                                (linear solve)
  Z^{k+1} = M @ Z^{k}, Z(goal) = 1                        (power iteration)
  Q*(u|x) = R(u|x) exp(-C(x,u)/alpha) Z*(F(x,u)) / Z*(x)  (Eq. 7.29)
"""

import numpy as np


def z_linear_solve(M, goal_state):
    """Solve Z = M @ Z with Z(goal) = 1 via matrix inversion.

    Rearranges to (I - M_reduced) @ Z_reduced = M[:, goal] for non-goal states.
    """
    n = M.shape[0]
    Z = np.zeros(n)
    Z[goal_state] = 1.0

    others = [i for i in range(n) if i != goal_state]
    M_red = M[np.ix_(others, others)]
    m_goal = M[others, goal_state]

    Z_red = np.linalg.solve(np.eye(len(others)) - M_red, m_goal)
    for i, idx in enumerate(others):
        Z[idx] = Z_red[i]

    return Z


def z_power_iteration(M, goal_state, n_iters):
    """Solve Z = M @ Z with Z(goal) = 1 via power iteration."""
    n = M.shape[0]
    Z = np.ones(n)
    history = [Z.copy()]

    # ##########################################################
    # TODO: Power iteration to solve the fixed-point Z = M Z.
    # Repeatedly multiply M by Z, reset Z(goal) = 1, and
    # normalize. Store each iterate in history.
    #
    # ##########################################################
    raise NotImplementedError("TODO: z_power_iteration")

    return Z, history


def policy_from_Z(grid, Z, alpha):
    """Recover stationary policy from Z (Eq. 7.29)."""
    n = grid.n_states
    policy = np.zeros((n, grid.n_actions))

    for x in range(n):
        R_u = grid.reference_policy(x)
        for u in range(grid.n_actions):
            x_next = grid.step(x, u)
            c = grid.cost(x, u)
            policy[x, u] = R_u[u] * np.exp(-c / alpha) * Z[x_next]

        total = policy[x].sum()
        if total > 0:
            policy[x] /= total
        else:
            policy[x] = 1.0 / grid.n_actions

    return policy
