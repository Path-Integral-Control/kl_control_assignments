# mppi/ — Model Predictive Path Integral Control (Chapters 8-9)

Implements the MPPI algorithm (Williams et al., 2017) for sampling-based model predictive control. MPPI generates K noisy rollouts around a nominal control sequence, weights them by exponentiated negative cost, and returns the weighted-average control. This is the trajectory-level application of the same Boltzmann-weighted importance sampling from the variational formula (Chapter 6).

## Solver (`core.py`)

The `MPPI` class supports both discrete-time and continuous-time (SDE) formulations:

- `solve(x0)`: discrete-time rollouts with `model.step()`. Noise is added to the control sequence and trajectories are integrated forward. Returns the optimal first control.
- `solve_continuous(x0)`: continuous-time rollouts with Euler-Maruyama integration. Process noise enters through the diffusion matrix G(x). Requires models that implement `diffusion_matrix()`.

The temperature parameter `lambda_` controls the sharpness of the weighting: small lambda concentrates weight on the lowest-cost rollouts; large lambda averages more broadly.

## Integrators (`integrators.py`)

Provides `euler_step`, `rk4_step`, and `rk4_substep` (RK4 with multiple substeps for stiff dynamics, including optional state clamping and quaternion normalization between substeps). All functions accept an `xp` parameter for NumPy/CuPy backend switching.

## Backend (`backend.py`)

Transparent NumPy/CuPy switching. Call `set_backend(True)` before constructing an MPPI instance to use GPU-accelerated rollouts.

## Dynamics models (`models/`)

Each model subclasses `DynamicsModel` (defined in `models/base.py`) and implements `dynamics()`, `step()`, `running_cost()`, and `terminal_cost()`. Available models:

- **Double integrator** (`models/double_integrator/`): 4D state [x, y, vx, vy], 2D control [ax, ay]. Linear dynamics, used for 2D navigation examples.
- **Unicycle** (`models/unicycle/`): 3D state [x, y, theta], 2D control [v, omega]. Nonholonomic. The vehicle can only move along its heading direction.
- **Cart-pole** (`models/cartpole/`): 4D state [x, xdot, theta, thetadot], 1D control [F]. Underactuated swing-up problem.
- **Fixed-wing** (`models/fixed_wing/`): 13D state [position, body velocity, quaternion, angular rates], 4D control [aileron, elevator, throttle, rudder]. Full 6DOF aerodynamic model with stall modeling, quaternion kinematics, and a TECS-based nominal controller for warm-starting MPPI.
