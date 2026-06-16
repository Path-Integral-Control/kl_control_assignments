"""Chapter 7 — What Breaks Without Deterministic Transitions (Exercise 7.6).

Under deterministic transitions (Assumption 7.1):
  Z_k = M_k @ Z_{k+1}      (exact equality, Eq. 7.26)

Under stochastic transitions:
  Z_k <= M_k @ Z_{k+1}      (Jensen's inequality, Eq. 7.31)

The backward linear recursion OVERESTIMATES desirability when transitions
are stochastic. Forward Monte Carlo still converges to the TRUE value.

Plots (3 panels):
  1. Deterministic: Z_backward matches Z_mc (equality)
  2. Stochastic: Z_linear_overestimate > Z_true, Z_mc matches Z_true
  3. Jensen gap: Z_tilde / Z_true > 1

Usage:
    python examples/discrete/determinism_breaking.py
    python examples/discrete/determinism_breaking.py --slip 0.3
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import apply_style, label_panel, COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lmdp.gridworld import small_grid
from lmdp.backward import (
    build_M_matrix,
    build_M_matrix_stochastic,
    backward_recursion,
    value_from_desirability,
    bellman_recursion_stochastic,
)
from lmdp.forward_mc import generate_sample_paths, compute_path_rewards


# === Parameters ===
DEFAULT_ALPHA = 5.0
DEFAULT_T = 15
DEFAULT_SLIP = 0.2
N_MC_TRIALS = 50
N_MC_SAMPLES = 1000
SEED = 42


def mc_Z_estimate_stochastic(grid, x0, T, N, alpha, slip_prob, rng):
    """Estimate Z_0(x0) via forward MC with stochastic transitions."""
    paths = np.zeros((N, T + 1), dtype=int)
    actions = np.zeros((N, T), dtype=int)
    paths[:, 0] = x0

    for i in range(N):
        for t in range(T):
            R_u = grid.reference_policy(paths[i, t])
            a = rng.choice(grid.n_actions, p=R_u)
            actions[i, t] = a
            paths[i, t + 1] = grid.step_stochastic(paths[i, t], a, slip_prob, rng)

    rewards, C_min = compute_path_rewards(grid, paths, actions, alpha)
    return np.mean(rewards) * np.exp(-C_min / alpha)


def main():
    parser = argparse.ArgumentParser(description='Ch.7: Determinism breaking (Exercise 7.6)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--T', type=int, default=DEFAULT_T)
    parser.add_argument('--slip', type=float, default=DEFAULT_SLIP, help='Slip probability')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    alpha = args.alpha
    T = args.T
    slip_prob = args.slip
    grid = small_grid()
    n = grid.n_states

    # --- Terminal desirability ---
    Z_T = np.array([np.exp(-grid.terminal_cost(x) / alpha) for x in range(n)])

    # === Case 1: Deterministic transitions ===
    M_det = build_M_matrix(grid, alpha)
    Z_det = backward_recursion(Z_T, M_det, T)

    Z_mc_det = np.zeros(n)
    for x in range(n):
        estimates = []
        for trial in range(N_MC_TRIALS):
            rng = np.random.default_rng(SEED + x * 1000 + trial)
            paths, acts = generate_sample_paths(grid, x, 0, T, N_MC_SAMPLES, rng)
            rewards, C_min = compute_path_rewards(grid, paths, acts, alpha)
            estimates.append(np.mean(rewards) * np.exp(-C_min / alpha))
        Z_mc_det[x] = np.mean(estimates)

    # === Case 2: Stochastic transitions ===
    M_stoch = build_M_matrix_stochastic(grid, alpha, slip_prob)
    Z_tilde = backward_recursion(Z_T, M_stoch, T)

    V_true_stoch = bellman_recursion_stochastic(grid, alpha, T, slip_prob)
    Z_true_stoch = np.exp(-V_true_stoch / alpha)

    Z_mc_stoch = np.zeros(n)
    for x in range(n):
        estimates = []
        for trial in range(N_MC_TRIALS):
            rng = np.random.default_rng(SEED + 50000 + x * 1000 + trial)
            Z_mc_stoch[x] = mc_Z_estimate_stochastic(
                grid, x, T, N_MC_SAMPLES, alpha, slip_prob, rng)
        Z_mc_stoch[x] = np.mean([mc_Z_estimate_stochastic(
            grid, x, T, N_MC_SAMPLES, alpha, slip_prob,
            np.random.default_rng(SEED + 50000 + x * 1000 + t))
            for t in range(N_MC_TRIALS)])

    print(f'alpha={alpha}, T={T}, slip_prob={slip_prob}')
    print(f'Deterministic: max |Z_backward - Z_mc| = {np.max(np.abs(Z_det[0] - Z_mc_det)):.6f}')
    jensen_gap = Z_tilde[0] / np.clip(Z_true_stoch[0], 1e-300, None)
    print(f'Stochastic: mean Jensen gap Z_tilde/Z_true = {np.mean(jensen_gap[Z_true_stoch[0] > 1e-10]):.3f}')

    # === Plotting (3 panels) ===
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    states = np.arange(n)
    labels = [f'({r},{c})' for r in range(grid.n_rows) for c in range(grid.n_cols)]

    # Panel 1: Deterministic — backward = MC
    ax1 = axes[0]
    w = 0.35
    ax1.bar(states - w/2, Z_det[0], w, label='Backward (exact)', color=COLORS['gray'], alpha=0.7)
    ax1.bar(states + w/2, Z_mc_det, w, label='Forward MC', color=COLORS['secondary'], alpha=0.7)
    ax1.set_xlabel('State')
    ax1.set_ylabel(r'$Z_0(x)$')
    ax1.set_title(r'Deterministic: backward $=$ MC')
    ax1.legend(fontsize=8)
    ax1.set_xticks(states[::5])
    ax1.set_xticklabels([labels[i] for i in states[::5]], fontsize=7)
    label_panel(ax1, 'a')

    # Panel 2: Stochastic — linear overestimates, MC matches true
    ax2 = axes[1]
    w = 0.25
    ax2.bar(states - w, Z_true_stoch[0], w, label='True (nonlinear Bellman)', color=COLORS['gray'], alpha=0.7)
    ax2.bar(states, Z_tilde[0], w, label='Linear recursion (overestimate)', color=COLORS['accent'], alpha=0.7)
    ax2.bar(states + w, Z_mc_stoch, w, label='Forward MC', color=COLORS['secondary'], alpha=0.7)
    ax2.set_xlabel('State')
    ax2.set_ylabel(r'$Z_0(x)$')
    ax2.set_title(rf'Stochastic (slip$={slip_prob}$): linear overestimates')
    ax2.legend(fontsize=7)
    ax2.set_xticks(states[::5])
    ax2.set_xticklabels([labels[i] for i in states[::5]], fontsize=7)
    label_panel(ax2, 'b')

    # Panel 3: Jensen gap
    ax3 = axes[2]
    mask = Z_true_stoch[0] > 1e-10
    gap = np.ones(n)
    gap[mask] = Z_tilde[0, mask] / Z_true_stoch[0, mask]
    ax3.bar(states, gap, color=COLORS['quaternary'], alpha=0.7)
    ax3.axhline(1.0, color=COLORS['gray'], linestyle='--', linewidth=1.0, label='Equality (deterministic)')
    ax3.set_xlabel('State')
    ax3.set_ylabel(r'$\tilde{Z} / Z_{\mathrm{true}}$')
    ax3.set_title(r'Jensen gap: $\tilde{Z}_0 / Z_{0,\mathrm{true}} \geq 1$')
    ax3.legend()
    ax3.set_xticks(states[::5])
    ax3.set_xticklabels([labels[i] for i in states[::5]], fontsize=7)
    label_panel(ax3, 'c')

    plt.tight_layout()

    outpath = args.save or f'examples/results/discrete/gridworld/determinism_breaking_slip{slip_prob:.1f}.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
