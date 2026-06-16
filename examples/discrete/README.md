# examples/discrete/ — Gridworld KL Control and Discrete-Time MPPI

## Gridworld KL Control (Chapter 7)

Three examples that solve the same KL control problem on a gridworld. The default environment is `three_mountains` (three Gaussian peaks between a start position and a goal). The agent must find an optimal stochastic policy Q\*(u|x) that reaches the goal while avoiding the mountains and staying close to the uniform reference policy R.

All three produce 2x2 plots showing the environment layout, value/desirability heatmaps, policy arrows, and sample trajectories.

### gridworld_backward.py — Backward Linear Recursion (Algorithm 12)

Computes exact solution via Z_k = M @ Z_{k+1} backward from the terminal condition Z_T(x) = exp(-C_exit(x)/alpha). The desirability transition matrix M encodes both the transition dynamics and the running cost. The result is a time-varying policy Q\*_k(u|x) for each timestep k = 0, ..., T-1.

```bash
python3 examples/discrete/gridworld_backward.py
python3 examples/discrete/gridworld_backward.py --env config/environments/landing_site.yaml
python3 examples/discrete/gridworld_backward.py --alpha 2.0 --T 30 --grid_size 20
```

### gridworld_forward_mc.py — Forward Monte Carlo (Algorithms 13/14)

Estimates the policy by sampling N paths forward under the uniform reference R at each state and timestep. Rather than computing Z directly (which is numerically tiny, on the order of 1e-8 for this problem), the policy is recovered from the ratio of per-action desirability scores: Q\*(u|x) = r(x,u) / r(x). This ratio cancels the absolute scale, so the policy estimate is meaningful regardless of the magnitude of Z.

The convergence plot (panel b) shows that trajectory cost under the MC policy decreases as the sample budget N increases, approaching the exact backward solution.

Note that the horizon T must be significantly longer than the physical path length. This is because the reference policy is a random walk, and a random walk's displacement scales as sqrt(T), not T. With T too short, sampled paths almost never reach the goal, producing no useful signal for the estimator.

```bash
python3 examples/discrete/gridworld_forward_mc.py
python3 examples/discrete/gridworld_forward_mc.py --N 200 --T 80
```

### gridworld_z_learning.py — Infinite-Horizon Z-Iteration

Solves the infinite-horizon version of the problem, where the goal is absorbing (all actions at the goal return to the goal with zero cost). The desirability Z\* satisfies the fixed-point equation Z = M @ Z, which can be solved either by direct linear algebra ((I - M_red)^{-1} m_goal) or by power iteration Z^{k+1} = M @ Z^{k} with Z(goal) = 1 fixed.

The convergence plot shows exponential convergence of power iteration to the direct solve. The resulting policy is stationary (time-independent), unlike the finite-horizon backward solution.

```bash
python3 examples/discrete/gridworld_z_learning.py
python3 examples/discrete/gridworld_z_learning.py --env config/environments/landing_site.yaml
python3 examples/discrete/gridworld_z_learning.py --iters 500
```

### Differences between the three

All three compute the same object, an optimal policy Q\*(u|x) for a KL-regularized control problem, but they differ in assumptions and computational tradeoffs:

- **Backward** is exact and efficient for finite-horizon problems, but requires the full state space and transition model.
- **Forward MC** requires only a simulator (no model of M), but needs many samples and a long horizon for the random walk to explore the state space.
- **Z-learning** handles infinite-horizon problems and gives a stationary policy, but requires solving a linear system or waiting for power iteration to converge.

## Discrete-Time MPPI (Chapters 8-9)

### double_integrator_navigation.py — 2D Point Mass Navigation

A 4D double integrator [x, y, vx, vy] navigating through obstacle fields. This is the simplest MPPI example and the best one for exploring how MPPI parameters affect behavior:

- **Covariance (--sigma)**: controls exploration. Low sigma produces conservative, local plans. High sigma enables finding paths around distant obstacles but increases variance.
- **Horizon (--T)**: how far ahead the planner looks. Short horizons miss long-range structure (e.g., walking into a U-trap). Long horizons find globally better paths but cost more computation.
- **Temperature (--lambda_)**: sharpness of the importance weighting. Low lambda is greedy (concentrates on the best rollout); high lambda averages more broadly.

```bash
python3 examples/discrete/double_integrator_navigation.py --env config/environments/three_mountains.yaml --animate
python3 examples/discrete/double_integrator_navigation.py --env config/environments/forest.yaml --T 200 --sigma 5.0 --animate
python3 examples/discrete/double_integrator_navigation.py --env config/environments/u_trap.yaml --T 200 --animate
python3 examples/discrete/double_integrator_navigation.py --env config/environments/double_slit.yaml --sigma 5.0 --animate
```

### unicycle.py — Nonholonomic Vehicle

A 3D unicycle [x, y, theta] that can only move along its heading direction. Demonstrates how MPPI handles orientation-dependent dynamics. The planner must coordinate steering and speed to navigate around obstacles.

```bash
python3 examples/discrete/unicycle.py --animate
python3 examples/discrete/unicycle.py --env config/environments/forest.yaml --animate
python3 examples/discrete/unicycle.py --env config/environments/u_trap.yaml --T 200 --sigma 2.0 --animate
```

### cartpole.py — Cart-Pole Swing-Up

A 4D cart-pole [x, xdot, theta, thetadot] with 1D force control. The pole starts hanging down (theta = pi) and must be swung up to vertical (theta = 0). The system is underactuated, so the planner must discover the pump-and-catch strategy through sampling.

```bash
python3 examples/discrete/cartpole.py --animate
python3 examples/discrete/cartpole.py --gpu --samples 4096
```

### fixed_wing/ — 6DOF Fixed-Wing Aircraft

A 13D fixed-wing aircraft with full aerodynamic modeling (lift, drag, moments, stall). Two examples:

- `nominal.py`: flies a racetrack pattern using a TECS-based PID controller (no MPPI). Useful as a baseline.
- `mppi.py`: MPPI with the nominal controller as a warm-start. Sampled rollouts are visualized in 3D.

```bash
python3 examples/discrete/fixed_wing/nominal.py
python3 examples/discrete/fixed_wing/mppi.py --K 256 --jit      # CPU with Numba JIT
python3 examples/discrete/fixed_wing/mppi.py --K 1024 --gpu     # GPU with fused CUDA kernel
python3 examples/discrete/fixed_wing/mppi.py --K 1024 --gpu --noise
```
