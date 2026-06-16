"""Backward dynamic programming for linearly solvable KL control (Chapter 7).

Under deterministic transitions (Assumption 7.1), the KL control problem
reduces to a sequence of linear operations:  Z_k = M_k @ Z_{k+1}.

Key equations:
  Z_k(x) = sum_u R(u|x) exp(-C(x,u)/alpha) * Z_{k+1}(F(x,u))       (Eq. 7.25)
  Z_k = M_k @ Z_{k+1}   where M is the desirability matrix            (Eq. 7.26)
  Q*(u|x) = R(u|x) exp(-C(x,u)/alpha) Z_{k+1}(F(x,u)) / Z_k(x)     (Eq. 7.29)
  V_k(x) = -alpha * log Z_k(x)                                        (Eq. 7.23)
"""

import numpy as np
from lmdp.gridworld import N_ACTIONS, ACTIONS


def build_M_matrix(grid, alpha):
    """Build the desirability transition matrix M.  (Eq. 7.27)

    M(x, x') = sum_u  1_{x'=F(x,u)}  R(u|x)  exp(-C(x,u)/alpha)

    Parameters
    ----------
    grid : GridWorld
    alpha : float

    Returns
    -------
    M : array of shape (n_states, n_states)
    """
    n = grid.n_states
    M = np.zeros((n, n))

    # ##########################################################
    # TODO: Fill in M(x, x') by looping over all states and
    # actions (Eq. 7.27). For each state x and action u, get
    # R(u|x) from grid.reference_policy(), next state x' from
    # grid.step(), and running cost C(x,u) from grid.cost().
    # Accumulate into M[x, x'] the product of R(u|x) and the
    # exponentiated negative cost scaled by alpha.
    #
    # ##########################################################
    raise NotImplementedError("TODO: build_M_matrix")

    return M


def build_M_matrix_stochastic(grid, alpha, slip_prob=0.2):
    """Generalized M for stochastic transitions.  (Eq. 7.30)

    M(x, x') = sum_u  P(x'|x,u)  R(u|x)  exp(-C(x,u)/alpha)

    When transitions are stochastic, Z_k <= M_k Z_{k+1} (Jensen's inequality).
    """
    M_det = build_M_matrix(grid, alpha)

    n = grid.n_states
    M_slip = np.zeros((n, n))
    for x in range(n):
        row_sum = M_det[x].sum()
        for u_rand in range(grid.n_actions):
            x_rand = grid.step(x, u_rand)
            M_slip[x, x_rand] += row_sum / grid.n_actions

    return (1.0 - slip_prob) * M_det + slip_prob * M_slip


def backward_recursion(Z_T, M, T):
    """Algorithm 12: Z_k = M @ Z_{k+1} for k = T-1, ..., 0.

    Parameters
    ----------
    Z_T : array of shape (n_states,)
        Terminal desirability: Z_T(x) = exp(-C_exit(x)/alpha).  (Eq. 7.28)
    M : array of shape (n_states, n_states)
        Desirability matrix (time-invariant for stationary problems).
    T : int
        Number of time steps (horizon).

    Returns
    -------
    Z : array of shape (T+1, n_states)
        Z[k] is the desirability vector at time step k.
    """
    n = len(Z_T)
    Z = np.zeros((T + 1, n))
    Z[T] = Z_T

    # ##########################################################
    # TODO: Backward pass from k = T-1 down to 0
    # (Algorithm 12). Each step is a single matrix-vector
    # product: Z_k = M * Z_{k+1}.
    #
    # ##########################################################
    raise NotImplementedError("TODO: backward_recursion")

    return Z


def reconstruct_policy(grid, Z, alpha):
    """Reconstruct optimal policy Q*(u|x) from desirability.  (Eq. 7.29)

    Q*(u|x) = R(u|x) exp(-C(x,u)/alpha) * Z_{k+1}(F(x,u)) / Z_k(x)

    Parameters
    ----------
    grid : GridWorld
    Z : array of shape (T+1, n_states)
    alpha : float

    Returns
    -------
    policy : array of shape (T, n_states, n_actions)
        policy[k, x, u] = Q*(u_k | x_k = x)
    """
    T = Z.shape[0] - 1
    n = grid.n_states
    policy = np.zeros((T, n, grid.n_actions))

    # ##########################################################
    # TODO: Reconstruct Q*(u|x) at each time step from Z
    # (Eq. 7.29). Loop over time steps k and states x. For
    # each action u, multiply the reference policy R(u|x) by
    # the exponentiated negative cost and the ratio
    # Z_{k+1}(x') / Z_k(x). Normalize policy[k, x, :] to
    # sum to 1. If Z[k, x] is near zero, fall back to uniform.
    #
    # ##########################################################
    raise NotImplementedError("TODO: reconstruct_policy")

    return policy


def value_from_desirability(Z, alpha):
    """Convert desirability to value function.  (Eq. 7.23)

    V_k(x) = -alpha * log(Z_k(x))
    """
    Z_safe = np.clip(Z, 1e-300, None)
    return -alpha * np.log(Z_safe)


def bellman_recursion_general(grid, alpha, T):
    """Algorithm 11: general Bellman recursion (handles stochastic transitions).

    This is the nonlinear recursion that works without Assumption 7.1.
    Used in ch7_determinism_breaking.py to compute the TRUE value function
    under stochastic transitions.

    V_k(x) = min_Q sum_u Q(u|x) { rho_k(x,u) + alpha log Q(u|x)/R(u|x) }

    where rho_k(x,u) = C(x,u) + E[V_{k+1}(x') | x, u]

    Under stochastic transitions with slip_prob, the expectation is over
    the stochastic next-state distribution.
    """
    n = grid.n_states
    V = np.zeros((T + 1, n))

    for x in range(n):
        V[T, x] = grid.terminal_cost(x)

    for k in range(T - 1, -1, -1):
        for x in range(n):
            R_u = grid.reference_policy(x)
            log_sum = -np.inf
            terms = np.zeros(grid.n_actions)
            for u in range(grid.n_actions):
                rho = grid.cost(x, u) + V[k + 1, grid.step(x, u)]
                terms[u] = np.log(R_u[u] + 1e-300) - rho / alpha

            max_term = np.max(terms)
            V[k, x] = -alpha * (max_term + np.log(np.sum(np.exp(terms - max_term))))

    return V


def bellman_recursion_stochastic(grid, alpha, T, slip_prob=0.2):
    """Nonlinear Bellman recursion with stochastic transitions.

    Same as bellman_recursion_general but uses E[V_{k+1}(x')] under
    the stochastic transition model P(x'|x,u) instead of V_{k+1}(F(x,u)).
    """
    n = grid.n_states
    V = np.zeros((T + 1, n))

    for x in range(n):
        V[T, x] = grid.terminal_cost(x)

    for k in range(T - 1, -1, -1):
        for x in range(n):
            R_u = grid.reference_policy(x)
            terms = np.zeros(grid.n_actions)
            for u in range(grid.n_actions):
                c = grid.cost(x, u)

                x_intended = grid.step(x, u)
                EV = (1.0 - slip_prob) * V[k + 1, x_intended]
                p_slip = slip_prob / grid.n_actions
                for u_rand in range(grid.n_actions):
                    x_rand = grid.step(x, u_rand)
                    EV += p_slip * V[k + 1, x_rand]

                rho = c + EV
                terms[u] = np.log(R_u[u] + 1e-300) - rho / alpha

            max_term = np.max(terms)
            V[k, x] = -alpha * (max_term + np.log(np.sum(np.exp(terms - max_term))))

    return V
