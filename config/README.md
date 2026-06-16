# config/ — Configuration Files

## Environment configs (`environments/`)

YAML files defining 2D obstacle fields. Each file specifies start/goal positions, cost weights, boundary dimensions, and a list of obstacles (Gaussian, circle, or box). Switch any navigation example to a different environment with `--env config/environments/<name>.yaml`.

Environments compatible with gridworld discretization (Chapter 7): `three_mountains`, `landing_site`, `simple_goal`. These use Gaussian obstacles that resolve well on a coarse 15x15 grid.

Environments requiring continuous-state solvers only (MPPI, SDE): `forest` (200 procedural circles too fine for a coarse grid), `u_trap`, `double_slit`, `drunken_bridge` (thin box walls that vanish at coarse discretization).

## Fixed-wing configs

- `fixedwing_aircraft_params.yaml`: aerodynamic coefficients, mass/inertia, and physical limits for the 6DOF fixed-wing model.
- `fixedwing_controller_gains.yaml`: PID gains for the TECS-based nominal controller used to warm-start MPPI in the fixed-wing example.
