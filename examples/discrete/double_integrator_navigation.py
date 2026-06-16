"""Double integrator navigation via MPPI.

Loads any environment from YAML and navigates through it.

Usage:
    python examples/discrete/double_integrator_forest.py
    python examples/discrete/double_integrator_forest.py --env config/environments/three_mountains.yaml
    python examples/discrete/double_integrator_forest.py --env config/environments/forest.yaml --animate
    python examples/discrete/double_integrator_forest.py --env config/environments/u_trap.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Double integrator MPPI navigation')
parser.add_argument('--env', type=str, default='config/environments/forest.yaml',
                    help='Environment YAML (three_mountains, forest, u_trap, double_slit, drunken_bridge, landing_site, simple_goal)')
parser.add_argument('--animate', action='store_true', help='Real-time animation')
parser.add_argument('--K', type=int, default=1024, help='Number of samples')
parser.add_argument('--T', type=int, default=50, help='Planning horizon (timesteps)')
parser.add_argument('--sigma', type=float, default=1.5, help='Control noise std dev')
parser.add_argument('--lambda_', '--lambda', type=float, default=10.0, help='Temperature parameter')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--steps', type=int, default=300)
args = parser.parse_args()

import matplotlib
if args.animate:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import apply_style, raise_window, COLORS

from environments import Environment
from mppi import MPPI
from mppi.models import DoubleIntegrator

# === Setup ===
env = Environment.from_yaml(args.env)
env_name = os.path.splitext(os.path.basename(args.env))[0]

model = DoubleIntegrator(goal=env.goal_pos, obs=None)

# Override cost to use the Environment (handles all obstacle types: gaussian, circle, box)
_original_running_cost = model.running_cost
def _env_running_cost(x, u, t, xp):
    cost = _original_running_cost(x, u, t, xp)
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

DT = 0.05

np.random.seed(42)
solver = MPPI(model, K=args.K, T=args.T, dt=DT, lambda_=args.lambda_,
              sigma=[args.sigma, args.sigma], use_gpu=args.gpu)
xp = solver.xp

x = xp.array([env.start_pos[0], env.start_pos[1], 0.0, 0.0], dtype=xp.float32)
print(f'Navigation ({env_name}) | K={args.K}, T={args.T}, sigma={args.sigma}, lambda={args.lambda_}')


def draw_environment(ax):
    """Draw obstacles, start, goal."""
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


if args.animate:
    # === Real-time animation: simulate step by step ===
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    draw_environment(ax)

    trail_line, = ax.plot([], [], color=COLORS['primary'], linewidth=2.5)
    pos_dot, = ax.plot([], [], 'o', color=COLORS['primary'], markersize=8, zorder=10)
    sample_lines = [ax.plot([], [], color=COLORS['light'], linewidth=0.3, alpha=0.2)[0]
                    for _ in range(30)]

    history = []
    crash_count = [0]
    done = [False]

    def update(frame):
        global x
        if done[0]:
            return trail_line, pos_dot

        U = solver.solve(x)
        u = U[0]
        x = model.step(x[None, :], u[None, :], 0.05, xp).squeeze(0)
        x_np = x.get() if hasattr(x, 'get') else x
        pos_before = x_np[:2].copy()
        env.clamp_to_obstacles(x[None, :])
        x_np = x.get() if hasattr(x, 'get') else x
        if not np.allclose(pos_before, x_np[:2]):
            crash_count[0] += 1
        history.append(x_np.copy())

        h = np.array(history)
        trail_line.set_data(h[:, 0], h[:, 1])
        pos_dot.set_data([x_np[0]], [x_np[1]])

        samples = solver.get_sampled_trajectories()
        weights = solver.get_weights()
        if hasattr(samples, 'get'):
            samples = samples.get()
            weights = weights.get()
        top_idx = np.argsort(weights)[-30:]
        for j, idx in enumerate(top_idx):
            sample_lines[j].set_data(samples[idx, :, 0], samples[idx, :, 1])

        dist = np.linalg.norm(x_np[:2] - env.goal_pos)
        ax.set_title(f'MPPI {env_name}  |  step {frame}  |  dist={dist:.1f}m  |  crashes={crash_count[0]}')

        if frame % 50 == 0:
            print(f'  step {frame}: dist={dist:.1f}m, crashes={crash_count[0]}')

        if dist < 0.3:
            print(f'  REACHED GOAL in {frame} steps (crashes={crash_count[0]})')
            ax.set_title(f'MPPI {env_name}  |  GOAL in {frame} steps  |  crashes={crash_count[0]}')
            done[0] = True

        return trail_line, pos_dot

    anim = FuncAnimation(fig, update, frames=args.steps, interval=30,
                         blit=False, repeat=False, cache_frame_data=False)
    raise_window(fig)
    plt.show()

    # After window closes, save final state as static plot
    os.makedirs('examples/results', exist_ok=True)
    if len(history) > 0:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        draw_environment(ax2)
        h = np.array(history)
        ax2.plot(h[:, 0], h[:, 1], color=COLORS['primary'], linewidth=2.5, label='MPPI trajectory')
        ax2.set_title(f'MPPI navigation — {env_name}')
        ax2.legend(loc='upper left')
        outpath = f'examples/results/discrete/double_integrator/{env_name}.png'
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath)
        print(f'Saved to {outpath}')
        plt.close(fig2)

else:
    # === Static: run full simulation, then plot ===
    history = [x.copy()]
    for step in range(args.steps):
        U = solver.solve(x)
        u = U[0]
        x = model.step(x[None, :], u[None, :], 0.05, xp).squeeze(0)
        history.append(x.copy())
        if step % 50 == 0:
            x_np = x.get() if hasattr(x, 'get') else x
            dist = np.linalg.norm(x_np[:2] - env.goal_pos)
            print(f'  step {step}: dist={dist:.1f}m')

    history = np.array([h.get() if hasattr(h, 'get') else h for h in history])
    last_samples = solver.get_sampled_trajectories()
    if hasattr(last_samples, 'get'):
        last_samples = last_samples.get()
    last_weights = solver.get_weights()
    if hasattr(last_weights, 'get'):
        last_weights = last_weights.get()

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    draw_environment(ax)

    if last_samples is not None:
        top_idx = np.argsort(last_weights)[-30:]
        for idx in top_idx:
            ax.plot(last_samples[idx, :, 0], last_samples[idx, :, 1],
                    color=COLORS['light'], linewidth=0.3, alpha=0.3)

    ax.plot(history[:, 0], history[:, 1], color=COLORS['primary'],
            linewidth=2.5, label='MPPI trajectory')
    ax.set_title(f'MPPI navigation — {env_name}')
    ax.legend(loc='upper left')

    os.makedirs('examples/results', exist_ok=True)
    outpath = f'examples/results/discrete/double_integrator/{env_name}.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath)
    print(f'Saved to {outpath}')
    plt.close(fig)
