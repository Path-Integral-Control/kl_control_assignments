"""Chapter 8 — The Sufficient Condition: sigma*sigma^T = alpha*g*R^{-1}*g^T.

Theorem 8.1 (Eq. 8.35/8.50): A standard SOC problem can be reformulated
as a KL control problem (and solved by path integral / MPPI) if and only if
the noise covariance, input coupling, and control cost satisfy:

  sigma(x) sigma(x)^T = alpha * g(x) * R(x)^{-1} * g(x)^T

When this condition holds, the MPPI temperature lambda should equal alpha.
When it's violated (mismatched lambda), the controller degrades.

Demonstrates on the double integrator:
  - Matched alpha: clean trajectories, low cost
  - Mismatched alpha: degraded performance

Plots (2x3):
  Row 1 (matched):  trajectory, cost over time, matrix equality check
  Row 2 (mismatched): trajectory, cost over time, matrix mismatch

Usage:
    python examples/ch8_sufficient_condition.py
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
from mppi import MPPI
from mppi.models import DoubleIntegrator


# === Parameters ===
K = 1024
T = 40
DT = 0.05
SIGMA_CTRL = np.array([1.0, 1.0])
SIM_STEPS = 150
SEED = 42

GOAL = np.array([5.0, 5.0])
OBSTACLES = [(2.5, 3.0, 0.8), (4.0, 1.5, 0.6)]


def run_mppi(model, lambda_, label, seed):
    """Run a full MPPI simulation and return trajectory + costs."""
    np.random.seed(seed)
    solver = MPPI(model, K=K, T=T, dt=DT, lambda_=lambda_, sigma=SIGMA_CTRL)
    xp = solver.xp

    x = xp.array([0.0, 0.0, 0.0, 0.0], dtype=xp.float32)
    trajectory = [x.copy()]
    step_costs = []

    for step in range(SIM_STEPS):
        U = solver.solve(x)
        u = U[0]
        x = model.step(x[None, :], u[None, :], DT, xp).squeeze(0)
        trajectory.append(x.copy())
        step_costs.append(float(xp.min(solver.get_costs())))

    trajectory = np.array(trajectory)
    return trajectory, step_costs


def main():
    parser = argparse.ArgumentParser(description='Ch.8: Sufficient condition demo')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    model = DoubleIntegrator(goal=GOAL, obs=OBSTACLES)

    # --- Construct the matrices from Theorem 8.1 ---
    # For the double integrator:
    #   dx = [vx, vy, ax, ay]^T dt + G dW
    #   Control enters through acceleration: g = [[0,0],[0,0],[1,0],[0,1]]
    #   Diffusion (from model): G maps noise into velocity channels
    #   Control cost weight: R = w_ctrl * I  (from running_cost)

    g = np.zeros((4, 2))
    g[2, 0] = 1.0
    g[3, 1] = 1.0

    R_ctrl = model.w_control * np.eye(2)

    # The model's diffusion matrix at rest
    x_test = np.zeros((1, 4), dtype=np.float32)
    G = model.diffusion(x_test, np).squeeze(0)
    sigma_mat = G

    sigma_sigmaT = sigma_mat @ sigma_mat.T

    # --- Find the alpha that satisfies the condition ---
    # sigma*sigma^T = alpha * g * R^{-1} * g^T
    R_inv = np.linalg.inv(R_ctrl)
    g_Rinv_gT = g @ R_inv @ g.T

    nonzero_mask = np.abs(g_Rinv_gT) > 1e-10
    if np.any(nonzero_mask):
        ratios = sigma_sigmaT[nonzero_mask] / g_Rinv_gT[nonzero_mask]
        alpha_matched = float(np.mean(ratios))
    else:
        alpha_matched = 1.0

    alpha_mismatched = alpha_matched * 20.0

    print(f'sigma * sigma^T:\n{sigma_sigmaT}')
    print(f'g * R^{{-1}} * g^T:\n{g_Rinv_gT}')
    print(f'Matched alpha: {alpha_matched:.4f}')
    print(f'Mismatched alpha: {alpha_mismatched:.4f}')

    # --- Run MPPI with matched and mismatched lambda ---
    print('\nRunning matched...')
    traj_match, costs_match = run_mppi(model, alpha_matched, 'matched', SEED)
    print('Running mismatched...')
    traj_mismatch, costs_mismatch = run_mppi(model, alpha_mismatched, 'mismatched', SEED + 100)

    # --- Plotting (2x3) ---
    apply_style()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    panel_labels = [['a', 'b', 'c'], ['d', 'e', 'f']]

    for row, (traj, costs, lam, label) in enumerate([
        (traj_match, costs_match, alpha_matched, 'Matched'),
        (traj_mismatch, costs_mismatch, alpha_mismatched, 'Mismatched'),
    ]):
        # Panel 1: Trajectory
        ax = axes[row, 0]
        ax.plot(traj[:, 0], traj[:, 1], color=COLORS['primary'], linewidth=1.5)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=COLORS['tertiary'], markersize=8, label='Start')
        ax.plot(GOAL[0], GOAL[1], '*', color=COLORS['secondary'], markersize=12, label='Goal')
        for ox, oy, r in OBSTACLES:
            circle = plt.Circle((ox, oy), r, color=COLORS['lightgray'], alpha=0.5)
            ax.add_patch(circle)
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 7)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(rf'{label}: $\lambda = {lam:.2f}$')
        ax.legend(fontsize=8)
        label_panel(ax, panel_labels[row][0])

        # Panel 2: Cost over time
        ax = axes[row, 1]
        ax.plot(np.arange(len(costs)) * DT, costs, color=COLORS['gray'], linewidth=0.8)
        ax.set_xlabel(r'Time (s)')
        ax.set_ylabel('Best trajectory cost')
        ax.set_title(f'{label}: cost convergence')
        ax.set_yscale('log')
        label_panel(ax, panel_labels[row][1])

        # Panel 3: Matrix check
        ax = axes[row, 2]
        lhs = sigma_sigmaT
        rhs = lam * g_Rinv_gT
        diff = np.abs(lhs - rhs)
        im = ax.imshow(diff, cmap='Reds', vmin=0)
        ax.set_title(rf'{label}: $|\sigma\sigma^\top - \lambda \, g R^{{-1}} g^\top|$')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{diff[i,j]:.3f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        label_panel(ax, panel_labels[row][2])

    fig.suptitle(r'Sufficient condition: $\sigma\sigma^\top = \alpha \, g \, R^{-1} g^\top$')
    plt.tight_layout()

    outpath = args.save or 'examples/results/sufficient_condition.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
