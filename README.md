# KL Control Assignments

Course materials for Stochastic Optimal Control course. Three assignments cover the variational free energy formula, linearly solvable MDPs, and path integral control.

## Assignments

| Assignment | Topic | Chapter |
|------------|-------|---------|
| [Assignment 1](assignments/assignment1_ch6.pdf) | Variational free energy and Boltzmann sampling | Ch. 6 |
| [Assignment 2](assignments/assignment2_ch7.pdf) | Linearly solvable MDPs | Ch. 7 |
| [Assignment 3](assignments/assignment3_ch8.pdf) | Path integral control | Ch. 8 |

## Setup

```bash
git clone https://github.com/Path-Integral-Control/kl_control_assignments.git
# or with SSH: git clone git@github.com:Path-Integral-Control/kl_control_assignments.git

cd kl_control_assignments
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
```

For GPU acceleration (optional), install CuPy matching your CUDA version and the corresponding cuBLAS runtime:
```bash
pip install cupy-cuda12x nvidia-cublas-cu12    # CUDA 12.x
pip install cupy-cuda11x nvidia-cublas-cu11    # CUDA 11.x
```
If CuPy's CCCL headers fail to compile, pin to a known working version (e.g. `cupy-cuda12x==14.0.0`).

Numba (included in requirements) compiles GPU kernels used in fixed wing examples automatically for GPU acceleration.

If you run into compatibility issues, install the pinned versions:
```bash
pip install -r requirements.txt
```

## Overview

All output (plots, videos) goes to `examples/results/`.

### Assignment 1: Variational Formula (Ch. 6)

KL-regularized objectives of the form $J(Q) = \mathbb{E}_Q[C] + \alpha \, \mathrm{KL}(Q \| Q_0)$ are lower-bounded by the free energy for any probability measure $Q$. The variational formula gives the optimal solution as the Boltzmann distribution $Q^* \propto Q_0 \exp(-C/\alpha)$. Assignment 1 explores this through Boltzmann sampling from 1D and 2D cost landscapes, and examines how the temperature parameter $\alpha$ controls the tradeoff between exploration and exploitation.

```bash
python examples/variational/boltzmann_sampling.py
python examples/variational/alpha_sweep.py
python examples/variational/boltzmann_sampling.py --benchmark
python examples/variational/boltzmann_sampling.py --benchmark --gpu
```

### Assignment 2: Linearly Solvable MDPs (Ch. 7)

Linearly solvable MDPs exploit the log-transform $Z = \exp(-V/\alpha)$ to convert the nonlinear Bellman equation into a linear recursion $Z_k = M \, Z_{k+1}$. Assignment 2 implements three solution methods on gridworld environments: backward recursion, forward Monte Carlo estimation, and Z-learning. A determinism-breaking exercise (Exercise 7.6) demonstrates the Jensen gap that arises when transitions are stochastic.

https://github.com/user-attachments/assets/676d86e2-3a60-4515-8940-98cac6b2ed4a

```bash
# Backward recursion (Algorithm 12)
python examples/discrete/gridworld_backward.py
python examples/discrete/gridworld_backward.py --env config/environments/landing_site.yaml

# Forward Monte Carlo (Algorithms 13/14)
python examples/discrete/gridworld_forward_mc.py
python examples/discrete/gridworld_forward_mc.py --env config/environments/landing_site.yaml

# Z-learning (infinite horizon)
python examples/discrete/gridworld_z_learning.py
python examples/discrete/gridworld_z_learning.py --env config/environments/landing_site.yaml

# Determinism breaking (Exercise 7.6)
python examples/discrete/determinism_breaking.py --slip 0.0
python examples/discrete/determinism_breaking.py --slip 0.2
```

### Assignment 3: Path Integral Control (Ch. 8)

Path integral control generalizes KL control from linearly solvable MDPs (Ch. 7) to continuous state spaces. Assignment 3 implements both the discrete-time and the continuous-time SDE versions, and test them on four systems: a double integrator, a nonholonomic unicycle, a cart-pole swing-up, and a 13D fixed-wing aircraft.

Cart-pole swing-up:

https://github.com/user-attachments/assets/46e13171-c41f-481d-8b6d-c9f5916c89c1

```bash
python examples/discrete/cartpole.py --animate
```

2D double integrator navigation across obstacle environments (use `--env` to switch):

https://github.com/user-attachments/assets/f59958b6-3fc6-45a2-acda-b741fb139cc6

```bash
python examples/discrete/double_integrator_navigation.py --env config/environments/forest.yaml --animate
python examples/discrete/double_integrator_navigation.py --env config/environments/u_trap.yaml --animate
python examples/discrete/double_integrator_navigation.py --env config/environments/three_mountains.yaml --animate
```

The effect of horizon length $T$ and noise covariance $\sigma$ on planning behavior:

https://github.com/user-attachments/assets/30bf64ac-d31a-4cdd-a896-c94dcd41f1f1

```bash
# Forest: default parameters vs increased horizon and covariance
python examples/discrete/double_integrator_navigation.py --env config/environments/forest.yaml --animate
python examples/discrete/double_integrator_navigation.py --env config/environments/forest.yaml --T 200 --sigma 5.0 --animate

# Double slit: low covariance gets stuck at the barrier, high covariance breaks symmetry
python examples/discrete/double_integrator_navigation.py --env config/environments/double_slit.yaml --sigma 1.0 --animate
python examples/discrete/double_integrator_navigation.py --env config/environments/double_slit.yaml --sigma 5.0 --animate

# U-trap: short horizon walks into the trap, long horizon plans around the barrier
python examples/discrete/double_integrator_navigation.py --env config/environments/u_trap.yaml --T 50 --animate
python examples/discrete/double_integrator_navigation.py --env config/environments/u_trap.yaml --T 200 --animate
```

Nonholonomic unicycle through the same environments:

```bash
python examples/discrete/unicycle.py --animate
python examples/discrete/unicycle.py --env config/environments/forest.yaml --animate
python examples/discrete/unicycle.py --env config/environments/u_trap.yaml --T 200 --sigma 2.0 --animate
```

Continuous-time SDE formulation:

```bash
python examples/continuous/double_integrator_sde.py --animate
python examples/continuous/double_integrator_sde.py --env config/environments/forest.yaml --animate
python examples/continuous/unicycle_sde.py --animate
python examples/continuous/unicycle_sde.py --env forest --animate
python examples/continuous/unicycle_sde.py --env u_trap --animate --T 400 --sigma 2.0 --lambda_ 100
python examples/continuous/cartpole_sde.py --animate
```

Fixed-wing 13D 6DOF aircraft with nominal TECS controller and MPPI:

https://github.com/user-attachments/assets/38f90f4c-b796-4238-85b5-64bf12e1b7ac

```bash
python examples/discrete/fixed_wing/nominal.py
python examples/discrete/fixed_wing/nominal.py --noise
python examples/discrete/fixed_wing/mppi.py --K 256 --jit
python examples/discrete/fixed_wing/mppi.py --K 1024 --gpu
python examples/discrete/fixed_wing/mppi.py --K 1024 --gpu --noise
```

## Environments

Available environments: `three_mountains`, `forest`, `drunken_bridge`, `u_trap`, `double_slit`, `landing_site`, `simple_goal`.

Switch with `--env config/environments/<name>.yaml`.

## References

- T. Tanaka, "Stochastic Optimal Control," lecture notes
- Todorov, "Linearly-solvable Markov decision problems," NeurIPS 2007
- Williams et al., "Model Predictive Path Integral Control: From Theory to Parallel Computation," JGCD 2017
- Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning," ICRA 2017
- Fixed-wing aerodynamic model derived from [CyECCA](https://github.com/CogniPilot/cyecca)
- Environment and obstacle design: M. Robinson ([@Ban-Ironic-Ohms](https://github.com/Ban-Ironic-Ohms))
