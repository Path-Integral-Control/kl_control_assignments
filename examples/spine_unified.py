"""Navigation Spine — Same obstacle field across three formulations.

One script, three panels. The SAME 2D navigation problem (start on left,
goal on right, obstacles from a YAML config) solved at each level:

  Panel 1 (Variational): Static desirability field — where does Q* concentrate?
  Panel 2 (Discrete):    LMDP — gridworld policy arrows + trajectories
  Panel 3 (Continuous):  MPPI — sampled rollouts + closed-loop trajectory

The through-line: all three use the same Boltzmann-weighted logic at
different levels of abstraction.

Usage:
    python examples/spine_unified.py
    python examples/spine_unified.py --env config/environments/landing_site.yaml
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from plot_style import apply_style, label_panel, COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from environments import Environment
from variational.sampling import desirability_scores, inverse_cdf_resample
from lmdp.backward import build_M_matrix, backward_recursion, reconstruct_policy
from mppi import MPPI
from mppi.models import DoubleIntegrator


DEFAULT_ENV = 'config/environments/three_mountains.yaml'
DEFAULT_ALPHA = 3.0
GRID_N = 20
GRID_T = 30
MPPI_K = 1024
MPPI_T = 40
MPPI_DT = 0.05
MPPI_STEPS = 200
SEED = 42


def main():
    parser = argparse.ArgumentParser(description='Navigation spine: variational → discrete → continuous')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV,
                        help='Environment YAML (three_mountains, landing_site, simple_goal)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    alpha = args.alpha
    rng = np.random.default_rng(SEED)

    # --- Load environment from YAML ---
    env = Environment.from_yaml(args.env)
    env.check_compatibility('spine')
    env_name = os.path.splitext(os.path.basename(args.env))[0]
    print(f'Spine | env={env_name}, alpha={alpha}')

    # === Fine grid for cost field visualization ===
    bw = env.boundary.get('width', 14.0)
    bh = env.boundary.get('height', 14.0)
    x_min, x_max = env.x_center - bw/2, env.x_center + bw/2
    y_min, y_max = env.y_center - bh/2, env.y_center + bh/2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    C_field = env.cost_field(xx, yy)

    # ================================================================
    # PANEL 1: Variational — Static desirability field
    # ================================================================
    N_samples = 2000
    X_ref = np.column_stack([
        rng.uniform(x_min, x_max, N_samples),
        rng.uniform(y_min, y_max, N_samples),
    ])
    C_ref = env.cost_field(X_ref[:, 0:1], X_ref[:, 1:2]).flatten()
    r, _ = desirability_scores(C_ref, alpha)
    X_qstar, _ = inverse_cdf_resample(X_ref, r, n_resample=500, rng=rng)

    # ================================================================
    # PANEL 2: Discrete — LMDP on gridworld
    # ================================================================
    grid, start_state = env.to_gridworld(GRID_N, GRID_N, alpha)
    bounds = grid._env_bounds
    cell_w, cell_h = grid._cell_w, grid._cell_h

    Z_T = np.array([np.exp(-grid.terminal_cost(x) / alpha) for x in range(grid.n_states)])
    M = build_M_matrix(grid, alpha)
    Z = backward_recursion(Z_T, M, GRID_T)
    policy = reconstruct_policy(grid, Z, alpha)

    grid_trajs = []
    for _ in range(15):
        traj = [start_state]
        x = start_state
        for k in range(min(GRID_T, policy.shape[0])):
            probs = policy[k, x]
            if probs.sum() < 1e-10:
                break
            a = rng.choice(grid.n_actions, p=probs)
            x = grid.step(x, a)
            traj.append(x)
            if grid.state_to_rc(x) == grid.goal:
                break
        grid_trajs.append(traj)

    # ================================================================
    # PANEL 3: Continuous — MPPI
    # ================================================================
    obs_mppi = []
    for o in env.obstacles:
        if o['type'] == 'gaussian':
            obs_mppi.append((o['position'][0], o['position'][1], o['spread']))
        elif o['type'] == 'circle':
            obs_mppi.append((o['position'][0], o['position'][1], o['radius']))

    model = DoubleIntegrator(goal=env.goal_pos, obs=obs_mppi if obs_mppi else None)
    model.w_obstacle = 500.0

    np.random.seed(SEED)
    solver = MPPI(model, K=MPPI_K, T=MPPI_T, dt=MPPI_DT, lambda_=alpha * 10,
                  sigma=[1.0, 1.0])
    xp = solver.xp

    x = xp.array([env.start_pos[0], env.start_pos[1], 0.0, 0.0], dtype=xp.float32)
    mppi_traj = [x.copy()]

    for step in range(MPPI_STEPS):
        U = solver.solve(x)
        u = U[0]
        x = model.step(x[None, :], u[None, :], MPPI_DT, xp).squeeze(0)
        mppi_traj.append(x.copy())

    mppi_traj = np.array(mppi_traj)
    last_samples = solver.get_sampled_trajectories()
    last_weights = solver.get_weights()

    # ================================================================
    # PLOTTING — 1x3
    # ================================================================
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax in axes:
        ax.contourf(xx, yy, np.clip(C_field, 0, np.percentile(C_field, 95)),
                    levels=30, cmap='YlOrRd', alpha=0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.plot(*env.start_pos, 'o', color=COLORS['tertiary'], markersize=10, zorder=10)
        ax.plot(*env.goal_pos, '*', color=COLORS['secondary'], markersize=14, zorder=10)

    # Panel 1: Variational — Q* samples
    axes[0].scatter(X_qstar[:, 0], X_qstar[:, 1], s=5, c=COLORS['primary'], alpha=0.5, zorder=5)
    axes[0].set_title(rf'Variational: Boltzmann $Q^*$ ($\alpha={alpha}$)')
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$y$')
    label_panel(axes[0], 'a')

    # Panel 2: Discrete — gridworld policy + trajectories
    arrow_map = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}
    for s in range(grid.n_states):
        ri, ci = grid.state_to_rc(s)
        if (ri, ci) in grid.obstacles:
            continue
        cx_cell = bounds[0] + (ci + 0.5) * cell_w
        cy_cell = bounds[3] - (ri + 0.5) * cell_h
        best_a = np.argmax(policy[0, s])
        if best_a < 4:
            dx, dy = arrow_map[best_a]
            axes[1].annotate('', xy=(cx_cell + dx*cell_w*0.3, cy_cell + dy*cell_h*0.3),
                             xytext=(cx_cell, cy_cell),
                             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.8, alpha=0.6))

    for traj in grid_trajs:
        coords = []
        for s in traj:
            ri, ci = grid.state_to_rc(s)
            coords.append((bounds[0] + (ci + 0.5)*cell_w, bounds[3] - (ri + 0.5)*cell_h))
        xs, ys = zip(*coords)
        axes[1].plot(xs, ys, color=COLORS['primary'], linewidth=0.8, alpha=0.4)

    axes[1].set_title(rf'Discrete: LMDP policy (${GRID_N}\times{GRID_N}$)')
    axes[1].set_xlabel(r'$x$')
    label_panel(axes[1], 'b')

    # Panel 3: Continuous — MPPI trajectory + samples
    if last_samples is not None:
        n_show = min(50, MPPI_K)
        top_idx = np.argsort(last_weights)[-n_show:]
        for idx in top_idx:
            axes[2].plot(last_samples[idx, :, 0], last_samples[idx, :, 1],
                         color=COLORS['light'], linewidth=0.3, alpha=0.3)

    axes[2].plot(mppi_traj[:, 0], mppi_traj[:, 1], color=COLORS['primary'], linewidth=2.0, label='MPPI trajectory')
    axes[2].set_title(rf'Continuous: MPPI ($K={MPPI_K}$)')
    axes[2].set_xlabel(r'$x$')
    axes[2].legend(fontsize=8)
    label_panel(axes[2], 'c')

    plt.tight_layout()

    outpath = args.save or f'examples/results/spine_unified_{env_name}.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
