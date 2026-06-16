"""Chapter 7 — Forward Monte Carlo KL Control (Algorithms 13/14).

Instead of backward recursion, estimate the desirability Z_k(x_k) and
the optimal policy Q*(u_k|x_k) by sampling N paths forward under the
reference policy R and computing path rewards.

Compares MC-estimated Z against the exact backward-recursion Z.

Plots (2x2):
  1. Closed-loop MC trajectory on the grid
  2. Z convergence: MC estimate vs exact as N grows
  3. Policy arrows: MC vs exact backward
  4. Trajectory cost distribution

Usage:
    python examples/discrete/gridworld_forward_mc.py
    python examples/discrete/gridworld_forward_mc.py --animate
    python examples/discrete/gridworld_forward_mc.py --env config/environments/landing_site.yaml
    python examples/discrete/gridworld_forward_mc.py --N 500 --alpha 5.0
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
from lmdp.backward import (
    build_M_matrix,
    backward_recursion,
    reconstruct_policy,
    value_from_desirability,
)
from lmdp.forward_mc import (
    generate_sample_paths,
    compute_path_rewards,
    empirical_reward_per_action,
    approximate_policy,
    forward_mc_control,
)


DEFAULT_ENV = 'config/environments/three_mountains.yaml'
DEFAULT_ALPHA = 10.0
DEFAULT_T = 100
DEFAULT_GRID = 15
DEFAULT_N = 500
N_MC_EPISODES = 30
SEED = 42


def draw_grid(ax, grid, env, start_state, title=''):
    nr, nc = grid.n_rows, grid.n_cols
    bounds = grid._env_bounds

    xx, yy = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 200),
        np.linspace(bounds[2], bounds[3], 200))
    C = env.cost_field(xx, yy)
    ax.contourf(xx, yy, C, levels=30, cmap='YlOrRd', alpha=0.5)

    for r in range(nr + 1):
        y = bounds[3] - r * grid._cell_h
        ax.axhline(y, color='gray', linewidth=0.3, alpha=0.3)
    for c in range(nc + 1):
        x = bounds[0] + c * grid._cell_w
        ax.axvline(x, color='gray', linewidth=0.3, alpha=0.3)

    for r, c_idx in grid.obstacles:
        x = bounds[0] + c_idx * grid._cell_w
        y = bounds[3] - (r + 1) * grid._cell_h
        ax.add_patch(plt.Rectangle((x, y), grid._cell_w, grid._cell_h, color='black', alpha=0.7))

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


def state_to_xy(grid, state):
    r, c = grid.state_to_rc(state)
    bounds = grid._env_bounds
    x = bounds[0] + (c + 0.5) * grid._cell_w
    y = bounds[3] - (r + 0.5) * grid._cell_h
    return x, y


def sample_trajectory_exact(grid, policy, start, T, rng):
    traj = [start]
    actions = []
    x = start
    for k in range(min(T, policy.shape[0])):
        probs = policy[k, x]
        if probs.sum() < 1e-10:
            break
        a = rng.choice(grid.n_actions, p=probs)
        actions.append(a)
        x = grid.step(x, a)
        traj.append(x)
        if grid.state_to_rc(x) == grid.goal:
            break
    return traj, actions


def run_animate(grid, env, mc_trajs, start, env_name, args):
    bounds = grid._env_bounds
    cell_w, cell_h = grid._cell_w, grid._cell_h
    cell_costs = grid._cell_costs
    cost_max = cell_costs[cell_costs < 1e6].max()
    max_len = max(len(t) for t in mc_trajs)

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

    trail_lines = [ax.plot([], [], '-', lw=1.0, alpha=0.6, color=COLORS['secondary'])[0]
                   for _ in range(len(mc_trajs))]
    dots = [ax.plot([], [], 'o', ms=5, color=COLORS['secondary'], zorder=8)[0]
            for _ in range(len(mc_trajs))]
    flash_patches = []

    finished = [False] * len(mc_trajs)
    reached_count = [0]

    def update(frame):
        for p in flash_patches:
            p.remove()
        flash_patches.clear()

        for i, traj in enumerate(mc_trajs):
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

        ax.set_title(f'Forward MC ({env_name})  |  step {frame}  |  '
                     f'{reached_count[0]}/{len(mc_trajs)} reached goal')
        return trail_lines + dots + flash_patches

    anim = FuncAnimation(fig, update, frames=max_len + 5, interval=150, blit=False)

    if args.save and args.save.endswith(('.mp4', '.gif')):
        anim.save(args.save, fps=10, dpi=120)
        print(f'Saved animation to {args.save}')
    else:
        raise_window(fig)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Ch.7: Forward MC vs backward recursion')
    parser.add_argument('--animate', action='store_true', help='Real-time animated visualization')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV,
                        help='Environment YAML (three_mountains, landing_site, simple_goal)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--T', type=int, default=DEFAULT_T)
    parser.add_argument('--grid_size', type=int, default=DEFAULT_GRID)
    parser.add_argument('--N', type=int, default=DEFAULT_N, help='MC samples per step')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    alpha = args.alpha
    T = args.T
    N = args.N
    rng = np.random.default_rng(SEED)

    env = Environment.from_yaml(args.env)
    env.check_compatibility('gridworld')
    grid, start = env.to_gridworld(args.grid_size, args.grid_size, alpha)

    env_name = os.path.splitext(os.path.basename(args.env))[0]
    print(f'Forward MC | env={env_name}, grid={args.grid_size}x{args.grid_size}, alpha={alpha}')

    # --- Exact backward recursion (ground truth) ---
    Z_T = np.array([np.exp(-grid.terminal_cost(x) / alpha) for x in range(grid.n_states)])
    M = build_M_matrix(grid, alpha)
    Z_exact = backward_recursion(Z_T, M, T)
    policy_exact = reconstruct_policy(grid, Z_exact, alpha)
    Z0_exact_at_start = Z_exact[0, start]

    # --- Forward MC episodes ---
    mc_trajs = []
    mc_costs_list = []
    for ep in range(N_MC_EPISODES):
        traj, acts, Z_est = forward_mc_control(
            grid, start, T, N, alpha, rng=np.random.default_rng(SEED + ep))
        mc_trajs.append(traj)
        cost = sum(grid.cost(traj[i], acts[i]) for i in range(len(acts)))
        cost += grid.terminal_cost(traj[-1])
        mc_costs_list.append(cost)

    mc_reached = sum(1 for t in mc_trajs if grid.state_to_rc(t[-1]) == grid.goal)
    print(f'Z_0(start) exact: {Z0_exact_at_start:.2e}')
    print(f'MC episodes: {mc_reached}/{N_MC_EPISODES} reached goal')

    if args.animate:
        run_animate(grid, env, mc_trajs, start, env_name, args)
        return

    # --- Closed-loop performance: trajectory cost vs N samples per step ---
    N_values = [10, 25, 50, 100, 200, 500]
    traj_cost_means = []
    traj_cost_stds = []
    for n_samples in N_values:
        costs = []
        for trial in range(10):
            traj, acts, _ = forward_mc_control(
                grid, start, T, n_samples, alpha,
                rng=np.random.default_rng(SEED + 2000 + trial))
            c = sum(grid.cost(traj[i], acts[i]) for i in range(len(acts)))
            c += grid.terminal_cost(traj[-1])
            costs.append(c)
        traj_cost_means.append(np.mean(costs))
        traj_cost_stds.append(np.std(costs))

    # --- Exact policy costs ---
    exact_costs_list = []
    for ep in range(N_MC_EPISODES):
        traj, acts = sample_trajectory_exact(
            grid, policy_exact, start, T, np.random.default_rng(SEED + 500 + ep))
        cost = sum(grid.cost(traj[i], acts[i]) for i in range(len(acts)))
        cost += grid.terminal_cost(traj[-1])
        exact_costs_list.append(cost)

    # === Plotting (2x2) ===
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: MC trajectories
    draw_grid(axes[0, 0], grid, env, start,
              title=f'Forward MC trajectories ({mc_reached}/{N_MC_EPISODES} reach goal)')
    for traj in mc_trajs:
        coords = [state_to_xy(grid, s) for s in traj]
        xs, ys = zip(*coords)
        reached = grid.state_to_rc(traj[-1]) == grid.goal
        color = COLORS['primary'] if reached else COLORS['gray']
        axes[0, 0].plot(xs, ys, '-', linewidth=0.8, alpha=0.5, color=color)
    label_panel(axes[0, 0], 'a')

    # Panel 2: Trajectory cost vs N (closed-loop performance)
    exact_mean = np.mean(exact_costs_list)
    axes[0, 1].errorbar(N_values, traj_cost_means, yerr=traj_cost_stds,
                        fmt='o-', color=COLORS['secondary'], capsize=4,
                        label='Forward MC')
    axes[0, 1].axhline(exact_mean, color=COLORS['gray'], linestyle='--', linewidth=1.5,
                       label=f'Exact backward (mean={exact_mean:.0f})')
    axes[0, 1].set_xlabel(r'MC samples per step $N$')
    axes[0, 1].set_ylabel('Trajectory cost')
    axes[0, 1].set_title('Closed-loop cost vs sample budget')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    label_panel(axes[0, 1], 'b')

    # Panel 3: MC vs exact policy arrows on value function heatmap
    V_exact = value_from_desirability(Z_exact, alpha)
    V_grid = V_exact[0].reshape(grid.n_rows, grid.n_cols)
    bounds = grid._env_bounds
    im3 = axes[1, 0].imshow(V_grid, cmap='RdYlBu_r', origin='upper',
                            extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                            aspect='equal')
    env.draw_cost_contours(axes[1, 0], bounds=bounds)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title('Policy: MC (blue) vs exact (black)')

    angle_map = {
        0: 0, 1: 180, 2: 270, 3: 90,
        4: 315, 5: 45, 6: 225, 7: 135,
    }
    from matplotlib.markers import MarkerStyle
    goal_state = grid.rc_to_state(*grid.goal)
    mc_policy_0 = np.zeros((grid.n_states, grid.n_actions))
    for x in range(grid.n_states):
        r_idx, c_idx = grid.state_to_rc(x)
        if (r_idx, c_idx) in grid.obstacles or x == goal_state:
            continue
        paths, acts = generate_sample_paths(grid, x, 0, T, N, np.random.default_rng(SEED + x))
        rewards, _ = compute_path_rewards(grid, paths, acts, alpha)
        r_per_a = empirical_reward_per_action(rewards, acts[:, 0], grid.n_actions)
        mc_policy_0[x] = approximate_policy(r_per_a)

    for x in range(grid.n_states):
        r_idx, c_idx = grid.state_to_rc(x)
        if (r_idx, c_idx) in grid.obstacles or x == goal_state:
            continue
        cx, cy = state_to_xy(grid, x)
        best_exact = np.argmax(policy_exact[0, x])
        if best_exact in angle_map:
            m = MarkerStyle('^').rotated(deg=-angle_map[best_exact])
            axes[1, 0].scatter(cx, cy, marker=m, s=40, c='black',
                               alpha=0.7, zorder=5, linewidths=0)
        best_mc = np.argmax(mc_policy_0[x])
        if best_mc in angle_map:
            m = MarkerStyle('^').rotated(deg=-angle_map[best_mc])
            axes[1, 0].scatter(cx, cy, marker=m, s=25, c=COLORS['primary'],
                               alpha=0.5, zorder=4, linewidths=0)
    label_panel(axes[1, 0], 'c')

    # Panel 4: Cost distributions
    axes[1, 1].hist(exact_costs_list, bins=15, alpha=0.5, color=COLORS['gray'],
                    label='Exact backward policy', density=True)
    axes[1, 1].hist(mc_costs_list, bins=15, alpha=0.5, color=COLORS['secondary'],
                    label='Forward MC policy', density=True)
    axes[1, 1].set_xlabel('Trajectory cost')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Trajectory cost distribution')
    axes[1, 1].legend()
    label_panel(axes[1, 1], 'd')

    fig.suptitle(f'Forward MC KL control ({env_name})', fontsize=13)
    plt.tight_layout()

    outpath = args.save or f'examples/results/discrete/gridworld/forward_mc_{env_name}.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
