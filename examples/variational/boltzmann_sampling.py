"""Chapter 6 — Variational Formula & Boltzmann Sampling.

Demonstrates Algorithm 10 (inverse-CDF resampling from Q*):
  - 3D surface of the cost function C(x) on [0,1]^2
  - Q* concentrates in low-cost regions
  - Uniform samples from R + free energy convergence

The variational formula (Eq. 6.1):
  -alpha * log E^R[exp(-C(X)/alpha)] = inf_Q { E^Q[C(X)] + alpha * D(Q||R) }

The optimal Q* is the Boltzmann distribution (Eq. 6.3):
  Q*(x) = R(x) * exp(-C(x)/alpha) / Z

Usage:
    python examples/variational/boltzmann_sampling.py
    python examples/variational/boltzmann_sampling.py --alpha 0.5 --N 1000
"""

import sys
import os
import argparse

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mppi.backend import get_backend, set_backend
from variational.costs import landscape_2d
from variational.sampling import (
    desirability_scores,
    free_energy_estimate,
    free_energy_convergence,
    inverse_cdf_resample,
)


# === Parameters ===
DEFAULT_ALPHA = 0.5
DEFAULT_N = 500
DEFAULT_N_RESAMPLE = 500
GRID_RES = 200
SEED = 42


def main():
    parser = argparse.ArgumentParser(description='Ch.6: Boltzmann sampling on 2D potential')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help='Temperature parameter')
    parser.add_argument('--N', type=int, default=DEFAULT_N, help='Number of reference samples')
    parser.add_argument('--n_resample', type=int, default=DEFAULT_N_RESAMPLE, help='Number of Q* samples')
    parser.add_argument('--gpu', action='store_true', help='Use GPU via CuPy')
    parser.add_argument('--benchmark', action='store_true', help='Run timing comparison across sample sizes')
    parser.add_argument('--save', type=str, default=None, help='Save figure to file')
    args = parser.parse_args()

    set_backend(args.gpu)
    xp = get_backend()

    alpha = args.alpha
    N = args.N
    n_resample = args.n_resample
    rng = np.random.default_rng(SEED)

    if args.benchmark:
        sizes = [1000, 10000, 100000, 1000000]
        print(f'{"N":>10s}  {"Time (ms)":>10s}  {"F(Q*)":>10s}  {"Backend":>8s}')
        print('-' * 45)
        for n in sizes:
            X = rng.uniform(0, 1, size=(n, 2))
            C = landscape_2d(X[:, 0], X[:, 1], xp=np)
            C_xp = xp.asarray(C)
            # warmup
            desirability_scores(C_xp, alpha, xp=xp)
            t0 = time.time()
            r, _ = desirability_scores(C_xp, alpha, xp=xp)
            F = free_energy_estimate(C_xp, alpha, xp=xp)
            inverse_cdf_resample(X, r, n_resample=1000, rng=rng, xp=xp)
            if hasattr(xp, 'cuda'):
                xp.cuda.Stream.null.synchronize()
            ms = (time.time() - t0) * 1000
            print(f'{n:>10d}  {ms:>10.1f}  {F:>10.4f}  {xp.__name__:>8s}')
        return

    # --- Cost landscape on a fine grid (for plotting) ---
    xx, yy = np.meshgrid(
        np.linspace(0, 1, GRID_RES),
        np.linspace(0, 1, GRID_RES),
    )
    C_grid = landscape_2d(xx, yy)

    # --- Sample N points uniformly from R on [0,1]^2 ---
    X_ref = rng.uniform(0, 1, size=(N, 2))
    C_ref = landscape_2d(X_ref[:, 0], X_ref[:, 1])
    C_ref_xp = xp.asarray(C_ref)

    # --- Compute desirability scores r(i) = exp(-C(X_i)/alpha)  (Eq. 6.25) ---
    r, C_min = desirability_scores(C_ref_xp, alpha, xp=xp)

    # --- Free energy estimate F(Q*) = -alpha * log(mean(r)) + C_min  (Eq. 6.27) ---
    F_est = free_energy_estimate(C_ref_xp, alpha, xp=xp)
    F_running = free_energy_convergence(C_ref_xp, alpha, xp=xp)
    if hasattr(F_running, 'get'):
        F_running = F_running.get()

    # --- Resample from Q* via Algorithm 10 (inverse-CDF) ---
    r_np = r.get() if hasattr(r, 'get') else r
    X_qstar, _ = inverse_cdf_resample(X_ref, r_np, n_resample=n_resample, rng=rng)

    print(f'alpha = {alpha}')
    print(f'N = {N} reference samples from R (uniform)')
    print(f'Cost C(x) range: [{C_ref.min():.3f}, {C_ref.max():.3f}]')
    print(f'Free energy F(Q*) estimate: {F_est:.4f}')
    print(f'Resampled {n_resample} points from Q*')
    print(f'Backend: {xp.__name__}')

    # === Plotting (4 panels) ===
    # --- Apply academic plot style ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from plot_style import apply_style, label_panel, COLORS
    apply_style()

    fig = plt.figure(figsize=(12, 10))

    # (a) 3D surface of C(x)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_surface(xx, yy, C_grid, cmap='coolwarm', alpha=0.9,
                     rstride=4, cstride=4, linewidth=0.1, edgecolor='none')
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_zlabel(r'$C(\mathbf{x})$')
    ax1.view_init(elev=30, azim=-60)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.set_title(r'(a) Cost function $C(\mathbf{x})$')

    # (b) Uniform samples from R
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.contour(xx, yy, C_grid, levels=12, colors=COLORS['gray'], linewidths=0.4)
    ax2.scatter(X_ref[:, 0], X_ref[:, 1], s=3, c=COLORS['gray'], alpha=0.5, edgecolors='none')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$x_1$'); ax2.set_ylabel(r'$x_2$')
    ax2.set_title(rf'$N={N}$ samples from $R$ (uniform)')
    label_panel(ax2, 'b')

    # (c) Free energy convergence
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(np.arange(1, N + 1), F_running, color=COLORS['primary'], linewidth=1.0,
             label=r'MC estimate')
    ax3.axhline(F_est, color=COLORS['secondary'], linestyle='--', linewidth=1.0,
                label=rf'$F(Q^*) = {F_est:.1f}$')
    ax3.set_xlabel(r'Sample size $N$')
    ax3.set_ylabel(r'$F(Q^*)$')
    ax3.set_title('Free energy convergence (Eq. 6.27)')
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, N)
    label_panel(ax3, 'c')

    # (d) Resampled points from Q*
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.contour(xx, yy, C_grid, levels=12, colors=COLORS['gray'], linewidths=0.4)
    ax4.scatter(X_qstar[:, 0], X_qstar[:, 1], s=5, c=COLORS['secondary'],
                alpha=0.6, edgecolors='none')
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')
    ax4.set_xlabel(r'$x_1$'); ax4.set_ylabel(r'$x_2$')
    ax4.set_title(rf'Resampled $Q^*$ ($\alpha={alpha}$, Algorithm 10)')
    label_panel(ax4, 'd')

    plt.tight_layout()

    outpath = args.save or 'examples/results/variational/boltzmann_sampling.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
