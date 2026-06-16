"""Chapter 6 — Temperature Sweep on 1D Double-Well.

Shows how the temperature parameter alpha controls the Boltzmann
distribution Q*(x). At small alpha, Q* concentrates on the global
minima. At large alpha, Q* spreads across the domain.

Cost function: C(x) = (x^2 - 1)^2   (double-well, minima at x = +/-1)
Reference R:  uniform on [-2, 2]

Top row:  5 panels, one per alpha. Each shows C(x), exact Q*(x), and
          a histogram of Algorithm 10 resampled points.
Bottom:   F(Q*) vs alpha — analytic (line) and Monte Carlo (dots).

Usage:
    python examples/variational/alpha_sweep.py
    python examples/variational/alpha_sweep.py --N 5000
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
from variational.costs import double_well_1d
from variational.sampling import (
    desirability_scores,
    free_energy_estimate,
    inverse_cdf_resample,
)


# === Parameters ===
ALPHA_VALUES = [0.05, 0.1, 0.5, 1.0, 5.0]
DEFAULT_N = 10000
X_MIN, X_MAX = -2.0, 2.0
GRID_PTS = 500
SEED = 42


def analytic_Q_star(x, alpha):
    """Exact Boltzmann distribution Q*(x) for uniform R on [X_MIN, X_MAX].

    Q*(x) = exp(-C(x)/alpha) / Z   where Z = integral of exp(-C/alpha) over domain.
    """
    C = double_well_1d(x)
    log_unnorm = -C / alpha
    log_unnorm -= np.max(log_unnorm)
    unnorm = np.exp(log_unnorm)
    dx = x[1] - x[0]
    Z = np.sum(unnorm) * dx
    return unnorm / Z


def analytic_free_energy(x_grid, alpha):
    """Exact free energy F(Q*) = -alpha * log( integral R(x) exp(-C(x)/alpha) dx ).

    For uniform R on [X_MIN, X_MAX], R(x) = 1/(X_MAX - X_MIN).
    """
    C = double_well_1d(x_grid)
    dx = x_grid[1] - x_grid[0]
    R_density = 1.0 / (X_MAX - X_MIN)
    log_max = np.max(-C / alpha)
    integrand = np.exp(-C / alpha - log_max)
    log_integral = np.log(np.sum(integrand) * dx * R_density) + log_max
    return -alpha * log_integral


def main():
    parser = argparse.ArgumentParser(description='Ch.6: Alpha sweep on 1D double-well')
    parser.add_argument('--N', type=int, default=DEFAULT_N, help='Number of reference samples')
    parser.add_argument('--save', type=str, default=None, help='Save figure to file')
    args = parser.parse_args()

    N = args.N
    rng = np.random.default_rng(SEED)

    x_grid = np.linspace(X_MIN, X_MAX, GRID_PTS)
    C_grid = double_well_1d(x_grid)

    X_ref = rng.uniform(X_MIN, X_MAX, size=N)
    C_ref = double_well_1d(X_ref)

    apply_style()

    n_alpha = len(ALPHA_VALUES)
    fig = plt.figure(figsize=(3.5 * n_alpha, 6))
    gs = fig.add_gridspec(2, n_alpha, height_ratios=[3, 2], hspace=0.35)

    # --- Top row: Q* distribution per alpha ---
    for i, alpha in enumerate(ALPHA_VALUES):
        ax = fig.add_subplot(gs[0, i])

        Q_exact = analytic_Q_star(x_grid, alpha)

        r, _ = desirability_scores(C_ref, alpha)
        X_resampled, _ = inverse_cdf_resample(X_ref, r, n_resample=2000, rng=rng)

        ax_cost = ax.twinx()
        ax_cost.plot(x_grid, C_grid, color=COLORS['gray'], linewidth=0.8, alpha=0.3)
        ax_cost.set_ylabel(r'$C(x)$', color=COLORS['gray'], fontsize=8)
        ax_cost.tick_params(axis='y', labelcolor=COLORS['gray'], labelsize=7)

        ax.fill_between(x_grid, Q_exact, alpha=0.3, color=COLORS['primary'], label=r'$Q^*(x)$ exact')
        ax.hist(X_resampled, bins=60, range=(X_MIN, X_MAX), density=True,
                alpha=0.5, color=COLORS['secondary'], label='Resampled')

        ax.set_title(rf'$\alpha = {alpha}$')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xlabel(r'$x$')
        if i == 0:
            ax.set_ylabel(r'$Q^*(x)$')
            ax.legend(fontsize=7, loc='upper left')

    # --- Bottom row: F(Q*) vs alpha (span all columns) ---
    ax_bottom = fig.add_subplot(gs[1, :])

    alpha_fine = np.linspace(0.02, 6.0, 200)
    F_analytic = [analytic_free_energy(x_grid, a) for a in alpha_fine]

    F_mc = []
    F_mc_std = []
    for alpha in ALPHA_VALUES:
        estimates = []
        for trial in range(50):
            trial_rng = np.random.default_rng(SEED + trial)
            X_trial = trial_rng.uniform(X_MIN, X_MAX, size=N)
            C_trial = double_well_1d(X_trial)
            estimates.append(free_energy_estimate(C_trial, alpha))
        F_mc.append(np.mean(estimates))
        F_mc_std.append(np.std(estimates))

    ax_bottom.plot(alpha_fine, F_analytic, color=COLORS['gray'], linewidth=1.5, label=r'$F(Q^*)$ analytic')
    ax_bottom.errorbar(ALPHA_VALUES, F_mc, yerr=F_mc_std, fmt='o', color=COLORS['secondary'],
                       markersize=6, capsize=4, label=rf'$F(Q^*)$ MC ($N={N}$)')
    ax_bottom.set_xlabel(r'$\alpha$')
    ax_bottom.set_ylabel(r'$F(Q^*)$')
    ax_bottom.set_title(r'Free energy vs temperature')
    label_panel(ax_bottom, 'f')
    ax_bottom.legend()

    plt.tight_layout()

    outpath = args.save or 'examples/results/variational/alpha_sweep.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
