# examples/ — Running the Examples

All examples save output to `examples/results/`. Most MPPI examples support `--animate` for real-time visualization (requires a display) and run headless by default (saves a static PNG).

## Variational formula — `variational/` (Chapter 6)

Demonstrates the variational formula on simple cost landscapes. No dynamics. These are static optimization problems over distributions. See `variational/README.md`.

## KL control on gridworlds — `discrete/gridworld_*.py` (Chapter 7)

Three examples solving the same finite-horizon KL control problem on a 15x15 gridworld, each using a different algorithm. The default environment is `three_mountains` (three Gaussian peaks between start and goal). See `discrete/README.md` for details on each example and how they compare.

```bash
python3 examples/discrete/gridworld_backward.py
python3 examples/discrete/gridworld_forward_mc.py
python3 examples/discrete/gridworld_z_learning.py
```

## Discrete-time MPPI — `discrete/` (Chapters 8-9)

Sampling-based MPC on four dynamical systems with increasing complexity: double integrator (linear), unicycle (nonholonomic), cart-pole (underactuated), and fixed-wing aircraft (6DOF). See `discrete/README.md`.

```bash
python3 examples/discrete/double_integrator_navigation.py --animate
python3 examples/discrete/unicycle.py --animate
python3 examples/discrete/cartpole.py --animate
python3 examples/discrete/fixed_wing/mppi.py --K 256 --jit
```

## Continuous-time SDE — `continuous/` (Chapters 8-9)

The same dynamical systems solved with continuous-time MPPI using Euler-Maruyama integration. Process noise enters through a diffusion matrix G(x). See `continuous/README.md`.

```bash
python3 examples/continuous/double_integrator_sde.py --animate
python3 examples/continuous/unicycle_sde.py --animate
python3 examples/continuous/cartpole_sde.py --animate
```