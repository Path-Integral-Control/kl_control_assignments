"""Chapter 7 — Backward Linear Recursion on a Gridworld (Algorithm 12).

Demonstrates linearly solvable KL control on a gridworld loaded from
a YAML environment config. Under deterministic transitions (Assumption 7.1),
the KL control problem reduces to:

  Z_k = M_k @ Z_{k+1}      (Eq. 7.26)

where M is the desirability transition matrix (Eq. 7.27).

Plots (2x2):
  1. Grid layout (obstacles, start, goal)
  2. Desirability heatmap Z_0(x)
  3. Value function V_0(x) with policy arrows
  4. Sample trajectories under Q*

Usage:
    python examples/discrete/gridworld_backward.py
    python examples/discrete/gridworld_backward.py --animate
    python examples/discrete/gridworld_backward.py --env config/environments/landing_site.yaml
    python examples/discrete/gridworld_backward.py --alpha 2.0 --T 30 --grid_size 20
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
if '--animate' in sys.argv:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import apply_style, label_panel, raise_window, COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from environments import Environment
from lmdp.gridworld import ACTIONS
from lmdp.backward import (
    build_M_matrix,
    backward_recursion,
    reconstruct_policy,
    value_from_desirability,
)


# === Default Parameters ===
DEFAULT_ENV = 'config/environments/three_mountains.yaml'
DEFAULT_ALPHA = 10.0
DEFAULT_T = 40
DEFAULT_GRID = 15
N_TRAJECTORIES = 30
SEED = 42


def draw_grid(ax, grid, env, start_state, title=''):
    nr, nc = grid.n_rows, grid.n_cols
    bounds = grid._env_bounds
    cell_w, cell_h = grid._cell_w, grid._cell_h

    xx, yy = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 200),
        np.linspace(bounds[2], bounds[3], 200))
    C = env.cost_field(xx, yy)
    ax.contourf(xx, yy, C, levels=30, cmap='YlOrRd', alpha=0.5)

    for r in range(nr + 1):
        y = bounds[3] - r * cell_h
        ax.axhline(y, color='gray', linewidth=0.3, alpha=0.3)
    for c in range(nc + 1):
        x = bounds[0] + c * cell_w
        ax.axvline(x, color='gray', linewidth=0.3, alpha=0.3)

    for r, c_idx in grid.obstacles:
        x = bounds[0] + c_idx * cell_w
        y = bounds[3] - (r + 1) * cell_h
        ax.add_patch(plt.Rectangle((x, y), cell_w, cell_h, color='black', alpha=0.7))

    sx, sy = state_to_xy(grid, start_state)
    gx, gy = state_to_xy(grid, grid.rc_to_state(*grid.goal))
    ax.plot(sx, sy, 'go', markersize=10, zorder=10)
    ax.plot(gx, gy, 'r*', markersize=14, zorder=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')


def draw_heatmap(ax, grid, values, env, start_state, title='', cmap='YlOrRd'):
    nr, nc = grid.n_rows, grid.n_cols
    bounds = grid._env_bounds
    grid_vals = values.reshape(nr, nc)

    im = ax.imshow(grid_vals, cmap=cmap, origin='upper',
                   extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                   aspect='equal')
    env.draw_cost_contours(ax, bounds=bounds)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for r, c_idx in grid.obstacles:
        x = bounds[0] + c_idx * grid._cell_w
        y = bounds[3] - (r + 1) * grid._cell_h
        ax.add_patch(plt.Rectangle((x, y), grid._cell_w, grid._cell_h,
                     color='black', alpha=0.7))

    sx, sy = state_to_xy(grid, start_state)
    gx, gy = state_to_xy(grid, grid.rc_to_state(*grid.goal))
    ax.plot(sx, sy, 'go', markersize=8, zorder=10)
    ax.plot(gx, gy, 'r*', markersize=12, zorder=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(title)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')


def draw_policy_arrows(ax, grid, policy_k, color='black', alpha=0.7, size=40):
    bounds = grid._env_bounds
    arrow_map = {
        0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
        4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1),
    }
    angle_map = {
        0: 0, 1: 180, 2: 270, 3: 90,
        4: 315, 5: 45, 6: 225, 7: 135,
    }

    from matplotlib.markers import MarkerStyle
    for x in range(grid.n_states):
        r, c = grid.state_to_rc(x)
        if (r, c) in grid.obstacles or (r, c) == grid.goal:
            continue
        best_u = np.argmax(policy_k[x])
        if best_u in arrow_map:
            cx = bounds[0] + (c + 0.5) * grid._cell_w
            cy = bounds[3] - (r + 0.5) * grid._cell_h
            marker = MarkerStyle('^').rotated(deg=-angle_map[best_u])
            ax.scatter(cx, cy, marker=marker, s=size, c=color,
                       alpha=alpha, zorder=5, linewidths=0)


def state_to_xy(grid, state):
    r, c = grid.state_to_rc(state)
    bounds = grid._env_bounds
    x = bounds[0] + (c + 0.5) * grid._cell_w
    y = bounds[3] - (r + 0.5) * grid._cell_h
    return x, y


def sample_trajectory(grid, policy, start_state, T, rng):
    traj = [start_state]
    x = start_state
    for k in range(min(T, policy.shape[0])):
        probs = policy[k, x]
        if probs.sum() < 1e-10:
            break
        a = rng.choice(grid.n_actions, p=probs)
        x = grid.step(x, a)
        traj.append(x)
        if grid.state_to_rc(x) == grid.goal:
            break
    return traj


def run_animate(grid, env, policy, trajectories, start, env_name, alpha, args):
    bounds = grid._env_bounds
    cell_w, cell_h = grid._cell_w, grid._cell_h
    cell_costs = grid._cell_costs
    cost_max = cell_costs[cell_costs < 1e6].max()
    max_len = max(len(t) for t in trajectories)

    fig, ax = plt.subplots(figsize=(8, 8))
    xx, yy = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 200),
        np.linspace(bounds[2], bounds[3], 200))
    C = env.cost_field(xx, yy)
    ax.contourf(xx, yy, C, levels=30, cmap='YlOrRd', alpha=0.5)

    for r in range(grid.n_rows + 1):
        ax.axhline(bounds[3] - r * cell_h, color='gray', lw=0.3, alpha=0.3)
    for c in range(grid.n_cols + 1):
        ax.axvline(bounds[0] + c * cell_w, color='gray', lw=0.3, alpha=0.3)

    for r, c_idx in grid.obstacles:
        x = bounds[0] + c_idx * cell_w
        y = bounds[3] - (r + 1) * cell_h
        ax.add_patch(plt.Rectangle((x, y), cell_w, cell_h, color='black', alpha=0.7))

    sx, sy = state_to_xy(grid, start)
    gx, gy = state_to_xy(grid, grid.rc_to_state(*grid.goal))
    ax.plot(sx, sy, 'go', ms=10, zorder=10)
    ax.plot(gx, gy, 'r*', ms=14, zorder=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    trail_lines = [ax.plot([], [], '-', lw=1.0, alpha=0.6, color=COLORS['primary'])[0]
                   for _ in range(len(trajectories))]
    dots = [ax.plot([], [], 'o', ms=5, color=COLORS['primary'], zorder=8)[0]
            for _ in range(len(trajectories))]
    flash_patches = []

    finished = [False] * len(trajectories)
    reached_count = [0]

    def update(frame):
        for p in flash_patches:
            p.remove()
        flash_patches.clear()

        for i, traj in enumerate(trajectories):
            step = frame
            if step >= len(traj):
                step = len(traj) - 1
                if not finished[i]:
                    finished[i] = True
                    if grid.state_to_rc(traj[-1]) == grid.goal:
                        reached_count[0] += 1
                    else:
                        trail_lines[i].set_color(COLORS['gray'])
                        trail_lines[i].set_alpha(0.3)

            xs, ys = zip(*[state_to_xy(grid, s) for s in traj[:step+1]])
            trail_lines[i].set_data(xs, ys)

            cx, cy = state_to_xy(grid, traj[step])
            if not finished[i]:
                dots[i].set_data([cx], [cy])
            else:
                dots[i].set_data([], [])

            if not finished[i] and step < len(traj):
                r_idx, c_idx = grid.state_to_rc(traj[step])
                cost_here = cell_costs[r_idx, c_idx]
                if cost_here > cost_max * 0.15:
                    intensity = min(cost_here / cost_max, 1.0)
                    rx = bounds[0] + c_idx * cell_w
                    ry = bounds[3] - (r_idx + 1) * cell_h
                    p = ax.add_patch(plt.Rectangle(
                        (rx, ry), cell_w, cell_h,
                        color='red', alpha=0.4 * intensity, zorder=6))
                    flash_patches.append(p)

        ax.set_title(f'Backward KL Control ({env_name})  |  step {frame}  |  '
                     f'{reached_count[0]}/{len(trajectories)} reached goal')
        return trail_lines + dots + flash_patches

    anim = FuncAnimation(fig, update, frames=max_len + 5, interval=150, blit=False)

    if args.save and args.save.endswith(('.mp4', '.gif')):
        anim.save(args.save, fps=10, dpi=120)
        print(f'Saved animation to {args.save}')
    else:
        raise_window(fig)
        plt.show()


def run_static(grid, env, policy, Z, V, trajectories, start, env_name, alpha, args):
    apply_style()
    reached = sum(1 for t in trajectories if grid.state_to_rc(t[-1]) == grid.goal)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    draw_grid(axes[0, 0], grid, env, start, title='Gridworld layout')
    label_panel(axes[0, 0], 'a')
    draw_heatmap(axes[0, 1], grid, Z[0], env, start, title=rf'Desirability $Z_0(\mathbf{{x}})$, $\alpha={alpha}$')
    label_panel(axes[0, 1], 'b')
    draw_heatmap(axes[1, 0], grid, V[0], env, start, title=r'Value $V_0(\mathbf{x})$ + policy arrows', cmap='RdYlBu_r')
    draw_policy_arrows(axes[1, 0], grid, policy[0])
    label_panel(axes[1, 0], 'c')

    draw_grid(axes[1, 1], grid, env, start,
              title=rf'Sample trajectories under $Q^*$ ({reached}/{N_TRAJECTORIES} reach goal)')
    for traj in trajectories:
        coords = [state_to_xy(grid, s) for s in traj]
        xs, ys = zip(*coords)
        axes[1, 1].plot(xs, ys, '-', linewidth=0.8, alpha=0.5, color=COLORS['primary'])
    label_panel(axes[1, 1], 'd')

    plt.tight_layout()

    outpath = args.save or f'examples/results/discrete/gridworld/backward_{env_name}.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Ch.7: Backward recursion on gridworld')
    parser.add_argument('--animate', action='store_true', help='Real-time animated visualization')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV,
                        help='Environment YAML (three_mountains, landing_site, simple_goal)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--T', type=int, default=DEFAULT_T)
    parser.add_argument('--grid_size', type=int, default=DEFAULT_GRID)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    alpha = args.alpha
    T = args.T
    rng = np.random.default_rng(SEED)

    # --- Load environment from YAML ---
    env = Environment.from_yaml(args.env)
    env.check_compatibility('gridworld')
    grid, start = env.to_gridworld(args.grid_size, args.grid_size, alpha)

    env_name = os.path.splitext(os.path.basename(args.env))[0]
    print(f'Environment: {env_name}')
    print(f'Grid: {grid.n_rows}x{grid.n_cols}, {len(grid.obstacles)} obstacles')
    print(f'alpha={alpha}, T={T}')

    # --- Terminal desirability Z_T(x) = exp(-C_exit(x) / alpha)  (Eq. 7.28) ---
    Z_T = np.array([np.exp(-grid.terminal_cost(x) / alpha) for x in range(grid.n_states)])

    # --- Build M matrix  (Eq. 7.27) ---
    M = build_M_matrix(grid, alpha)

    # --- Backward recursion: Z_k = M @ Z_{k+1}  (Algorithm 12) ---
    Z = backward_recursion(Z_T, M, T)
    V = value_from_desirability(Z, alpha)
    policy = reconstruct_policy(grid, Z, alpha)

    print(f'V_0 at start: {V[0, start]:.2f}')
    print(f'V_0 at goal:  {V[0, grid.rc_to_state(*grid.goal)]:.2f}')

    # --- Sample trajectories under Q* ---
    trajectories = []
    for _ in range(N_TRAJECTORIES):
        traj = sample_trajectory(grid, policy, start, T, rng)
        trajectories.append(traj)
    reached = sum(1 for t in trajectories if grid.state_to_rc(t[-1]) == grid.goal)
    print(f'{reached}/{N_TRAJECTORIES} trajectories reached the goal')

    if args.animate:
        run_animate(grid, env, policy, trajectories, start, env_name, alpha, args)
    else:
        run_static(grid, env, policy, Z, V, trajectories, start, env_name, alpha, args)


if __name__ == '__main__':
    main()
