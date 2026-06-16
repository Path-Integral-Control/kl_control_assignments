"""Chapter 7 — Infinite-horizon KL control via Z-iteration.

For the infinite-horizon linearly solvable MDP, the optimal desirability
Z* satisfies the fixed-point equation:

  Z = M @ Z

Z* is the principal eigenvector of M (Perron-Frobenius theorem).

This example compares:
  1. Power iteration: Z <- M @ Z, normalize, repeat
  2. Direct eigenvalue solve: Z* = principal eigenvector of M
  3. Finite-horizon backward recursion (Algorithm 12) for reference

Usage:
    python examples/discrete/gridworld_z_learning.py
    python examples/discrete/gridworld_z_learning.py --animate
    python examples/discrete/gridworld_z_learning.py --env config/environments/three_mountains.yaml
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
from lmdp.gridworld import GridWorld
from lmdp.backward import build_M_matrix
from lmdp.z_learning import z_linear_solve, z_power_iteration, policy_from_Z


DEFAULT_ENV = 'config/environments/three_mountains.yaml'
DEFAULT_ALPHA = 10.0
DEFAULT_T = 100
DEFAULT_GRID = 15
DEFAULT_ITERS = 200
SEED = 42


def state_to_xy(grid, s):
    r, c = grid.state_to_rc(s)
    bounds = grid._env_bounds
    cx = bounds[0] + (c + 0.5) * grid._cell_w
    cy = bounds[3] - (r + 0.5) * grid._cell_h
    return cx, cy


def sample_trajectory(grid, policy, start, T, rng):
    traj = [start]
    x = start
    goal = grid.rc_to_state(*grid.goal)
    for t in range(T):
        if x == goal:
            break
        probs = policy[x]
        u = rng.choice(grid.n_actions, p=probs)
        x = grid.step(x, u)
        traj.append(x)
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

    trail_lines = [ax.plot([], [], '-', lw=1.0, alpha=0.6, color=COLORS['quaternary'])[0]
                   for _ in range(len(trajectories))]
    dots = [ax.plot([], [], 'o', ms=5, color=COLORS['quaternary'], zorder=8)[0]
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

        ax.set_title(f'Z-learning ({env_name})  |  step {frame}  |  '
                     f'{reached_count[0]}/{len(trajectories)} reached goal')
        return trail_lines + dots + flash_patches

    anim = FuncAnimation(fig, update, frames=max_len + 5, interval=150, blit=False)

    if args.save and args.save.endswith(('.mp4', '.gif')):
        anim.save(args.save, fps=10, dpi=120)
        print(f'Saved animation to {args.save}')
    else:
        raise_window(fig)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Ch.7: Infinite-horizon Z-iteration')
    parser.add_argument('--animate', action='store_true', help='Real-time animated visualization')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV,
                        help='Environment YAML (three_mountains, landing_site, simple_goal)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--T', type=int, default=DEFAULT_T)
    parser.add_argument('--grid_size', type=int, default=DEFAULT_GRID)
    parser.add_argument('--iters', type=int, default=DEFAULT_ITERS)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    alpha = args.alpha
    rng = np.random.default_rng(SEED)

    env = Environment.from_yaml(args.env)
    env.check_compatibility('gridworld')
    grid, start = env.to_gridworld(args.grid_size, args.grid_size, alpha)

    env_name = os.path.splitext(os.path.basename(args.env))[0]
    print(f'Z-iteration | env={env_name}, grid={args.grid_size}x{args.grid_size}, alpha={alpha}')

    M = build_M_matrix(grid, alpha)
    goal_state = grid.rc_to_state(*grid.goal)

    # --- Direct linear solve: (I - M_red) Z_red = m_goal ---
    Z_solve = z_linear_solve(M, goal_state)
    V_solve = -alpha * np.log(np.clip(Z_solve, 1e-300, None))
    print(f'Z(start) = {Z_solve[start]:.6f}, Z(goal) = {Z_solve[goal_state]:.6f}')

    # --- Policy from solved Z ---
    policy = policy_from_Z(grid, Z_solve, alpha)
    trajectories = []
    for _ in range(30):
        traj = sample_trajectory(grid, policy, start, args.T, rng)
        trajectories.append(traj)
    reached = sum(1 for t in trajectories if grid.state_to_rc(t[-1]) == grid.goal)
    print(f'{reached}/30 trajectories reached goal')

    if args.animate:
        run_animate(grid, env, policy, trajectories, start, env_name, alpha, args)
        return

    # --- Power iteration with Z(goal) = 1 fixed ---
    Z_power, Z_history = z_power_iteration(M, goal_state, args.iters)
    V_power = -alpha * np.log(np.clip(Z_power, 1e-300, None))

    # --- Convergence: error between power iteration and direct solve ---
    errors = []
    for Z_snap in Z_history:
        err = np.max(np.abs(Z_snap - Z_solve))
        errors.append(err)

    # === Plotting (2x2) ===
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    n_r, n_c = grid.n_rows, grid.n_cols

    bounds = grid._env_bounds

    # (a) Direct solve V(x)
    V_grid = V_solve.reshape(n_r, n_c)
    im = axes[0, 0].imshow(V_grid, cmap='viridis', origin='upper',
                           extent=[bounds[0], bounds[1], bounds[2], bounds[3]])
    env.draw_cost_contours(axes[0, 0], bounds=bounds, colors='white', alpha=0.4)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[0, 0].set_title(r'$V(x)$ from $(I - M)^{-1}$ (infinite horizon)')
    label_panel(axes[0, 0], 'a')

    # (b) V(x) + policy arrows
    im2 = axes[0, 1].imshow(V_grid, cmap='viridis', origin='upper',
                            extent=[bounds[0], bounds[1], bounds[2], bounds[3]])
    env.draw_cost_contours(axes[0, 1], bounds=bounds, colors='white', alpha=0.4)
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    angle_map = {
        0: 0, 1: 180, 2: 270, 3: 90,
        4: 315, 5: 45, 6: 225, 7: 135,
    }
    from matplotlib.markers import MarkerStyle
    for x in range(grid.n_states):
        r, c = grid.state_to_rc(x)
        if (r, c) in grid.obstacles or (r, c) == grid.goal:
            continue
        best_u = np.argmax(policy[x])
        if best_u in angle_map:
            cx, cy = state_to_xy(grid, x)
            marker = MarkerStyle('^').rotated(deg=-angle_map[best_u])
            axes[0, 1].scatter(cx, cy, marker=marker, s=40, c='white',
                               alpha=0.7, zorder=5, linewidths=0)
    axes[0, 1].set_title(r'$V(x)$ + stationary policy $Q^*(u|x)$')
    label_panel(axes[0, 1], 'b')

    # (c) Power iteration convergence
    axes[1, 0].semilogy(errors, color=COLORS['primary'], linewidth=1.5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel(r'$\|Z^{(k)} - Z^*\|_\infty$')
    axes[1, 0].set_title('Power iteration convergence')
    axes[1, 0].grid(True, alpha=0.3)
    label_panel(axes[1, 0], 'c')

    # (d) Trajectories under stationary policy
    xx, yy = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 200),
        np.linspace(bounds[2], bounds[3], 200))
    C_field = env.cost_field(xx, yy)
    axes[1, 1].contourf(xx, yy, C_field, levels=30, cmap='YlOrRd', alpha=0.5)
    gx, gy = state_to_xy(grid, grid.rc_to_state(*grid.goal))
    sx, sy = state_to_xy(grid, start)
    axes[1, 1].plot(gx, gy, 'r*', markersize=14, zorder=10)
    axes[1, 1].plot(sx, sy, 'go', markersize=10, zorder=10)

    for traj in trajectories:
        coords = [state_to_xy(grid, s) for s in traj]
        xs, ys = zip(*coords)
        reached_goal = grid.state_to_rc(traj[-1]) == grid.goal
        color = COLORS['primary'] if reached_goal else COLORS['gray']
        axes[1, 1].plot(xs, ys, '-', color=color, alpha=0.4, linewidth=0.8)

    axes[1, 1].set_xlim(bounds[0], bounds[1])
    axes[1, 1].set_ylim(bounds[2], bounds[3])
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_title(f'Optimal policy ({reached}/30 reach goal)')
    label_panel(axes[1, 1], 'd')

    fig.suptitle(f'Infinite-horizon KL control ({env_name})', fontsize=13)
    plt.tight_layout()

    outpath = args.save or f'examples/results/discrete/gridworld/z_learning_{env_name}.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
