# examples/continuous/ — Continuous-Time SDE Examples (Chapters 8-9)

Continuous-time formulation of MPPI using Euler-Maruyama integration of the stochastic differential equation:

```
dx = f(x, u) dt + G(x) dW
```

where G(x) is the diffusion matrix that determines how process noise enters the dynamics. The key difference from discrete-time MPPI is that noise acts through the dynamics (additive or state-dependent), rather than being added directly to the control sequence.

Each example uses `MPPI.solve_continuous()`, which requires the model to implement a `diffusion_matrix(x)` method in addition to the standard dynamics.

## double_integrator_sde.py

Double integrator with force noise entering through the velocity channels. The diffusion matrix maps Brownian motion into acceleration disturbances. Supports all `--env` environments.

```bash
python3 examples/continuous/double_integrator_sde.py --animate
python3 examples/continuous/double_integrator_sde.py --env config/environments/forest.yaml --animate
```

## unicycle_sde.py

Unicycle with noncommutative diffusion: velocity noise is injected along the heading direction, and yaw noise is additive. The state-dependent structure of G(x) means noise in different channels interacts with the vehicle's orientation.

```bash
python3 examples/continuous/unicycle_sde.py --animate
python3 examples/continuous/unicycle_sde.py --env config/environments/u_trap.yaml --animate --T 400 --sigma 2.0 --K 2048 --lambda 100
```

## cartpole_sde.py

Cart-pole swing-up with force noise entering through the mass matrix. The diffusion matrix maps Brownian noise into the acceleration channels of both cart and pole, coupling through the system's inertia.

```bash
python3 examples/continuous/cartpole_sde.py --animate
python3 examples/continuous/cartpole_sde.py --gpu --samples 4096
```

## Discrete vs continuous

The discrete-time and continuous-time examples solve the same control problems. The main differences are:

- **Noise model**: discrete adds noise to the control input; continuous adds noise through a diffusion matrix G(x).
- **Sufficient condition** (Theorem 8.1): the continuous formulation makes explicit the condition sigma * sigma^T = alpha * g * R^{-1} * g^T under which KL control and standard SOC are equivalent. See `examples/sufficient_condition.py`.
- **Integration**: continuous uses Euler-Maruyama (or higher-order stochastic integrators) instead of deterministic RK4/Euler.
