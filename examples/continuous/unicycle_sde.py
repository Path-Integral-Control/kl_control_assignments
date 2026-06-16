"""Unicycle via Continuous-Time MPPI (SDE formulation).

Uses solve_continuous() with noncommutative diffusion:
  dx = [v*cos(theta), v*sin(theta), omega]^T dt + G(x) dW

The diffusion matrix G(x) injects velocity noise along the heading
direction and additive yaw noise.

Usage:
    python examples/continuous/unicycle_sde.py
    python examples/continuous/unicycle_sde.py --animate
    python examples/continuous/unicycle_sde.py --env config/environments/u_trap.yaml --animate
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
from mppi.models import Unicycle


DEFAULT_ENV = 'config/environments/three_mountains.yaml'
DEFAULT_K = 2048
DEFAULT_T = 80
DEFAULT_DT = 0.025
DEFAULT_LAMBDA = 100.0
SIM_STEPS = 600
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
    parser = argparse.ArgumentParser(description='Unicycle SDE (continuous-time)')
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

    model = Unicycle(goal=env.goal_pos, obs=None)
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
                  lambda_=args.lambda_, sigma=[args.sigma, args.sigma * 2], use_gpu=args.gpu)
    xp = solver.xp

    theta0 = np.arctan2(env.goal_pos[1] - env.start_pos[1],
                        env.goal_pos[0] - env.start_pos[0])
    x = xp.array([env.start_pos[0], env.start_pos[1], theta0], dtype=xp.float32)

    print(f'Unicycle SDE | env={env_name}, K={args.samples}')

    if args.animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_environment(ax, env)
        ax.set_title(f'Unicycle SDE — {env_name}')

        trail_line, = ax.plot([], [], color=COLORS['primary'], linewidth=2)
        pos_dot, = ax.plot([], [], 'ko', markersize=8, zorder=10)
        heading_arrow = [None]
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

                if heading_arrow[0]:
                    heading_arrow[0].remove()
                dx = 0.5 * np.cos(float(x_np[2]))
                dy = 0.5 * np.sin(float(x_np[2]))
                heading_arrow[0] = ax.annotate('', xy=(hx[-1] + dx, hy[-1] + dy),
                    xytext=(hx[-1], hy[-1]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

                samples = solver.get_sampled_trajectories()
                weights = solver.get_weights()
                if hasattr(samples, 'get'):
                    samples, weights = samples.get(), weights.get()
                top_idx = np.argsort(weights)[-N_SHOW:]
                for j, idx in enumerate(top_idx):
                    sample_lines[j].set_data(samples[idx, :, 0], samples[idx, :, 1])

                dist = np.linalg.norm(x_np[:2] - env.goal_pos)
                ax.set_title(f'Unicycle SDE — {env_name}  |  step {step}  |  dist={dist:.1f}m  |  crashes={crash_count}')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            dist = np.linalg.norm(x_np[:2] - env.goal_pos)
            if step % 50 == 0:
                print(f'  step {step:3d}: dist={dist:.2f}, theta={np.degrees(float(x_np[2])):.1f}deg, crashes={crash_count}')
            if dist < 0.3:
                print(f'  REACHED GOAL in {step} steps (crashes={crash_count})')
                ax.set_title(f'Unicycle SDE — {env_name}  |  GOAL in {step} steps  |  crashes={crash_count}')
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
                print(f'  step {step:3d}: dist={dist:.2f}, theta={np.degrees(x_np[2]):.1f}deg, crashes={crash_count}')
            if dist < 0.3:
                print(f'  REACHED GOAL in {step} steps (crashes={crash_count})')
                break

        trajectory = np.array([t.get() if hasattr(t, 'get') else t for t in trajectory])

        apply_style()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        draw_environment(ax, env)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=COLORS['primary'], linewidth=2, label='SDE trajectory')

        for i in range(0, len(trajectory), 20):
            dx = 0.4 * np.cos(trajectory[i, 2])
            dy = 0.4 * np.sin(trajectory[i, 2])
            ax.annotate('', xy=(trajectory[i, 0] + dx, trajectory[i, 1] + dy),
                        xytext=(trajectory[i, 0], trajectory[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))

        ax.set_title(f'Unicycle -- continuous-time SDE ({env_name})')
        ax.legend()
        plt.tight_layout()

        outpath = args.save or f'examples/results/continuous/unicycle/sde_{env_name}.png'
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Mean solve: {np.mean(solve_times)*1000:.1f}ms | Saved to {outpath}')
        plt.close(fig)


if __name__ == '__main__':
    main()
