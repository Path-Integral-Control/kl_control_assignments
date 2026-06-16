# lmdp/ — Linearly Solvable MDPs (Chapter 7)

Implements KL-regularized control on finite-state gridworlds. The KL control problem (Eq. 7.9) replaces the standard Bellman recursion with a linear one by penalizing deviation from a reference policy R:

```
min_Q  E^Q [ sum C_k(x,u) + alpha * D_KL(Q_k || R) ] + E^Q [ C_exit(x_T) ]
```

Under deterministic transitions (Assumption 7.1), the optimal desirability Z_k(x) satisfies the linear recursion (Eq. 7.26):

```
Z_k = M @ Z_{k+1}
```

where M is the desirability transition matrix (Eq. 7.27) and the optimal policy is recovered as (Eq. 7.29):

```
Q*(u|x) = R(u|x) * exp(-C(x,u)/alpha) * Z_{k+1}(F(x,u)) / Z_k(x)
```

This module provides three approaches to computing Z and the resulting policy:

## Backward recursion (`backward.py`, Algorithm 12)

Computes Z_k for all states and timesteps by iterating Z_k = M @ Z_{k+1} backward from the terminal condition Z_T(x) = exp(-C_exit(x)/alpha). This is exact and runs in O(T * n_states^2) time. It gives a time-varying policy Q*_k(u|x) for each timestep k.

## Forward Monte Carlo (`forward_mc.py`, Algorithms 13/14)

Estimates Z_k(x_k) and Q*(u_k|x_k) by sampling N paths forward under the reference policy R and computing importance weights. The key insight is that the optimal policy can be recovered from the ratio of per-action rewards (Eq. 7.43):

```
Q*(u|x) ≈ r(x,u) / r(x)
```

This ratio cancels the absolute scale of the desirability, so the policy estimate is meaningful even when Z is extremely small. The cost is that performance depends on the horizon T being long enough for random walks under R to reach the goal. Since a random walk's displacement scales as sqrt(T) rather than T, the required horizon can be significantly longer than the path length under the optimal policy.

## Z-learning (`z_learning.py`)

Solves the infinite-horizon version of the problem. With an absorbing goal state (Z(goal) = 1, zero cost, self-transitions), the fixed-point equation Z = M @ Z can be solved directly via (I - M_red)^{-1} m_goal, or iteratively via power iteration Z^{k+1} = M @ Z^{k}. The resulting policy is stationary (time-independent).

## Gridworld (`gridworld.py`)

Defines the finite-state environment. States are cells on an N x N grid (row-major indexing). There are 9 actions: 4 cardinal, 4 diagonal, and stay. Transitions are deterministic and clip at boundaries. The gridworld is typically constructed from a continuous environment via `Environment.to_gridworld()`, which maps obstacle costs onto the grid and sets up an absorbing goal state.
