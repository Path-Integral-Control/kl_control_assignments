"""Double Integrator via Continuous-Time MPPI (SDE formulation).

Uses solve_continuous() with Euler-Maruyama integration:
  dx = f(x,u)dt + G(x)dW

The diffusion matrix G(x) injects process noise through the velocity
channels, modeling force disturbances.

Usage:
    python examples/continuous/double_integrator_sde.py
    python examples/continuous/double_integrator_sde.py --animate
    python examples/continuous/double_integrator_sde.py --env config/environments/forest.yaml --animate
"""

import sys
import os
import argparse
import time

import numpy as np
import matplotlib
if '--animate' in sys.argv:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import apply_style, label_panel, raise_window, COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from environments import Environment
from mppi import MPPI
from mppi.models import DoubleIntegrator


DEFAULT_ENV = 'config/environments/three_mountains.yaml'
DEFAULT_K = 1024
DEFAULT_T = 80
DEFAULT_DT = 0.025
DEFAULT_LAMBDA = 100.0
SIM_STEPS = 200
N_SHOW = 50
SEED = 42


def draw_environment(ax, env):
    xx, yy = np.meshgrid(np.linspace(-2, 12, 200), np.linspace(-2, 12, 200))
    C = env.cost_field(xx, yy)
    ax.contourf(xx, yy, np.clip(C, 0, 5000), levels=30, cmap='YlOrRd', alpha=0.5)
    for o in env.obstacles:
        if o['type'] == 'circle':
            c = mpatches.Circle(o['position'], o['radius'], edgecolor='white',
                                facecolor='none', linewidth=0.5, alpha=0.6)
            ax.add_patch(c)
    ax.plot(*env.start_pos, 'o', color=COLORS['tertiary'], markersize=10, zorder=10)
    ax.plot(*env.goal_pos, '*', color=COLORS['secondary'], markersize=14, zorder=10)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')


def main():
    parser = argparse.ArgumentParser(description='Double integrator SDE (continuous-time)')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV,
                        help='Environment YAML (three_mountains, forest, u_trap, double_slit, drunken_bridge, landing_site, simple_goal)')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--samples', '--K', type=int, default=DEFAULT_K)
    parser.add_argument('--T', type=int, default=DEFAULT_T)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--lambda_', '--lambda', type=float, default=DEFAULT_LAMBDA)
    parser.add_argument('--animate', action='store_true', help='Live animation')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(SEED)
    env = Environment.from_yaml(args.env)
    env_name = os.path.splitext(os.path.basename(args.env))[0]

    model = DoubleIntegrator(goal=env.goal_pos, obs=None)
    _original_cost = model.running_cost
    def _env_running_cost(x, u, t, xp):
        cost = _original_cost(x, u, t, xp)
        x_np = x.get() if hasattr(x, 'get') else x
        env_cost = env.get_obstacle_cost(x_np)
        cost += xp.array(env_cost, dtype=cost.dtype)
        return cost
    model.running_cost = _env_running_cost
    def _clamp(x, xp):
        if hasattr(x, 'get'):
            x_np = x.get()
            env.clamp_to_obstacles(x_np)
            x[:] = xp.asarray(x_np)
        else:
            env.clamp_to_obstacles(x)
        return x
    model.clamp_state = _clamp
    solver = MPPI(model, K=args.samples, T=args.T, dt=DEFAULT_DT,
                  lambda_=args.lambda_, sigma=[args.sigma, args.sigma], use_gpu=args.gpu)
    xp = solver.xp

    x = xp.array([env.start_pos[0], env.start_pos[1], 0.0, 0.0], dtype=xp.float32)

    print(f'Double Integrator SDE | env={env_name}, K={args.samples}')

    if args.animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_environment(ax, env)
        ax.set_title(f'SDE continuous-time — {env_name}')

        trail_line, = ax.plot([], [], color=COLORS['primary'], linewidth=2)
        pos_dot, = ax.plot([], [], 'ko', markersize=8, zorder=10)
        sample_lines = [ax.plot([], [], color='steelblue', lw=0.3, alpha=0.2)[0] for _ in range(N_SHOW)]

        plt.tight_layout()
        raise_window(fig)
        plt.show()

        hx, hy = [], []
        crash_count = 0
        for step in range(SIM_STEPS):
            U = solver.solve_continuous(x)
            u = U[0]
            x = model.step(x[None, :], u[None, :], DEFAULT_DT, xp).squeeze(0)
            x_np = x.get() if hasattr(x, 'get') else x
            pos_before = x_np[:2].copy()
            env.clamp_to_obstacles(x[None, :])
            x_np = x.get() if hasattr(x, 'get') else x
            if not np.allclose(pos_before, x_np[:2]):
                crash_count += 1
            hx.append(float(x_np[0])); hy.append(float(x_np[1]))

            if step % 2 == 0:
                trail_line.set_data(hx, hy)
                pos_dot.set_data([hx[-1]], [hy[-1]])

                samples = solver.get_sampled_trajectories()
                weights = solver.get_weights()
                if hasattr(samples, 'get'):
                    samples, weights = samples.get(), weights.get()
                top_idx = np.argsort(weights)[-N_SHOW:]
                for j, idx in enumerate(top_idx):
                    sample_lines[j].set_data(samples[idx, :, 0], samples[idx, :, 1])

                dist = np.linalg.norm(x_np[:2] - env.goal_pos)
                ax.set_title(f'SDE — {env_name}  |  step {step}  |  dist={dist:.1f}m  |  crashes={crash_count}')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            dist = np.linalg.norm(x_np[:2] - env.goal_pos)
            if step % 50 == 0:
                print(f'  step {step:3d}: dist={dist:.2f}, crashes={crash_count}')
            if dist < 0.3:
                print(f'  REACHED GOAL in {step} steps (crashes={crash_count})')
                ax.set_title(f'SDE — {env_name}  |  GOAL in {step} steps  |  crashes={crash_count}')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                break

        plt.ioff()
        input("Press Enter to close...")
        plt.close('all')

    else:
        trajectory = [x.copy()]
        solve_times = []

        crash_count = 0
        for step in range(SIM_STEPS):
            t0 = time.time()
            U = solver.solve_continuous(x)
            solve_times.append(time.time() - t0)
            u = U[0]
            x = model.step(x[None, :], u[None, :], DEFAULT_DT, xp).squeeze(0)
            x_np = x.get() if hasattr(x, 'get') else x
            pos_before = x_np[:2].copy()
            env.clamp_to_obstacles(x[None, :])
            x_np = x.get() if hasattr(x, 'get') else x
            if not np.allclose(pos_before, x_np[:2]):
                crash_count += 1
            trajectory.append(x.copy())

            dist = np.linalg.norm(x_np[:2] - env.goal_pos)
            if step % 50 == 0:
                print(f'  step {step:3d}: dist={dist:.2f}, solve={solve_times[-1]*1000:.1f}ms, crashes={crash_count}')
            if dist < 0.3:
                print(f'  REACHED GOAL in {step} steps (crashes={crash_count})')
                break

        trajectory = np.array([t.get() if hasattr(t, 'get') else t for t in trajectory])
        last_samples = solver.get_sampled_trajectories()
        if hasattr(last_samples, 'get'):
            last_samples = last_samples.get()

        apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        for ax in axes:
            draw_environment(ax, env)

        axes[0].plot(trajectory[:, 0], trajectory[:, 1], color=COLORS['primary'], linewidth=2)
        axes[0].set_title('SDE trajectory')
        label_panel(axes[0], 'a')

        if last_samples is not None:
            weights = solver.get_weights()
            if hasattr(weights, 'get'):
                weights = weights.get()
            n_show = min(80, args.samples)
            top_idx = np.argsort(weights)[-n_show:]
            for idx in top_idx:
                axes[1].plot(last_samples[idx, :, 0], last_samples[idx, :, 1],
                             color=COLORS['light'], linewidth=0.3, alpha=0.3)
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], color=COLORS['primary'], linewidth=2, label='Executed')
        axes[1].set_title(rf'Sampled rollouts ($K={args.samples}$)')
        axes[1].legend()
        label_panel(axes[1], 'b')

        fig.suptitle(f'Double integrator -- continuous-time SDE ({env_name})')
        plt.tight_layout()

        outpath = args.save or f'examples/results/continuous/double_integrator/sde_{env_name}.png'
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Mean solve time: {np.mean(solve_times)*1000:.1f}ms')
        print(f'Saved to {outpath}')
        plt.close(fig)


if __name__ == '__main__':
    main()
