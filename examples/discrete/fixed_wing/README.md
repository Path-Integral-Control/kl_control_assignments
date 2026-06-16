# Fixed-Wing MPPI

6DOF rigid body aircraft controlled by the same MPPI solver used in the other examples.

## Model

State (13D): position in world frame (ENU), velocity in body frame (FLU), quaternion (scalar-first, body-to-world), angular rates in body frame.

Control (4D): aileron, elevator, throttle, rudder.

Aerodynamic forces and moments are computed from angle of attack and sideslip using coefficients derived from CyECCA.

## Integration

The fixed-wing model uses RK4 with 50 substeps per MPPI timestep. After every substep, three steps are run:

1. Quaternion normalization, enforcing unit norm and resetting to identity if degenerate.
2. State clamping of velocity, angular rate, angle of attack, and sideslip to physical limits.
3. Derivative clamping of linear and angular accelerations to prevent numerical blowup.

A second integration path (`simulate_single`) uses scipy's implicit Radau solver with Baumgarte quaternion constraint stabilization for high-accuracy single-trajectory validation.

## Nominal Controller

A TECS (Total Energy Control System) controller provides the warm-start control sequence for MPPI. It computes thrust from total energy error, pitch from flight path angle, and inner-loop PD on pitch/roll/yaw with rate damping.

## Usage

```bash
python examples/discrete/fixed_wing/nominal.py
python examples/discrete/fixed_wing/nominal.py --noise
python examples/discrete/fixed_wing/mppi.py --K 256 --jit
python examples/discrete/fixed_wing/mppi.py --K 1024 --gpu
python examples/discrete/fixed_wing/mppi.py --K 1024 --gpu --noise
```