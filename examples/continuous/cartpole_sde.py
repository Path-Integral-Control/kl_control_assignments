"""Cart-pole swing-up via Continuous-Time MPPI (SDE formulation).

Uses solve_continuous() with force noise entering through the
dynamics: the diffusion matrix maps Brownian noise into the
acceleration channels via the cart-pole's mass matrix.

Usage:
    python examples/continuous/cartpole_sde.py
    python examples/continuous/cartpole_sde.py --animate
    python examples/continuous/cartpole_sde.py --gpu --samples 4096
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import apply_style, label_panel, raise_window, COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mppi import MPPI
from mppi.models import CartPole


DEFAULT_K = 1024
DEFAULT_T = 50
DEFAULT_DT = 0.02
DEFAULT_LAMBDA = 10.0
DEFAULT_SIGMA = 3.0
SIM_STEPS = 300
SEED = 42

CART_W = 0.4
CART_H = 0.2
POLE_L = 1.0


def main():
    parser = argparse.ArgumentParser(description='Cartpole SDE (continuous-time)')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--samples', '--K', type=int, default=DEFAULT_K)
    parser.add_argument('--animate', action='store_true', help='Live animation')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(SEED)
    model = CartPole()
    solver = MPPI(model, K=args.samples, T=DEFAULT_T, dt=DEFAULT_DT,
                  lambda_=DEFAULT_LAMBDA, sigma=[DEFAULT_SIGMA], use_gpu=args.gpu)
    xp = solver.xp

    x = xp.array([0.0, 0.0, np.pi, 0.0], dtype=xp.float32)

    print(f'Cartpole SDE swing-up | K={args.samples}')

    if args.animate:
        plt.ion()
        fig, (ax_cart, ax_phase) = plt.subplots(1, 2, figsize=(14, 5))

        # Cart-pole visual
        ax_cart.set_xlim(-3, 3)
        ax_cart.set_ylim(-1.5, 1.5)
        ax_cart.set_aspect('equal')
        ax_cart.grid(True, alpha=0.3)
        ax_cart.axhline(0, color='k', lw=0.5)
        ax_cart.set_title('Cart-Pole SDE')

        cart_patch = plt.Rectangle((-CART_W/2, -CART_H/2), CART_W, CART_H,
                                   fc='steelblue', ec='black', zorder=5)
        ax_cart.add_patch(cart_patch)
        pole_line, = ax_cart.plot([], [], 'o-', color='firebrick', lw=3,
                                  markersize=8, markevery=[1], zorder=6)

        # Phase portrait
        ax_phase.set_xlim(-np.pi, np.pi)
        ax_phase.set_ylim(-10, 10)
        ax_phase.set_xlabel(r'$\theta$ (rad)')
        ax_phase.set_ylabel(r'$\dot{\theta}$ (rad/s)')
        ax_phase.set_title('Phase Portrait')
        ax_phase.axvline(0, color=COLORS['tertiary'], ls='--', alpha=0.5)
        ax_phase.axhline(0, color=COLORS['tertiary'], ls='--', alpha=0.5)
        ax_phase.grid(True, alpha=0.3)
        phase_line, = ax_phase.plot([], [], color=COLORS['primary'], lw=0.8, alpha=0.7)
        phase_dot, = ax_phase.plot([], [], 'ro', markersize=6, zorder=10)

        plt.tight_layout()
        raise_window(fig)
        plt.show()

        h_theta, h_thetadot = [], []
        for step in range(SIM_STEPS):
            U = solver.solve_continuous(x)
            u = U[0]
            x = model.step(x[None, :], u[None, :], DEFAULT_DT, xp).squeeze(0)
            x_np = x.get() if hasattr(x, 'get') else x

            theta_w = np.arctan2(np.sin(x_np[2]), np.cos(x_np[2]))
            h_theta.append(float(theta_w))
            h_thetadot.append(float(x_np[3]))

            if step % 2 == 0:
                cx = float(x_np[0])
                cart_patch.set_xy((cx - CART_W/2, -CART_H/2))
                pole_x = cx + POLE_L * np.sin(float(x_np[2]))
                pole_y = POLE_L * np.cos(float(x_np[2]))
                pole_line.set_data([cx, pole_x], [0, pole_y])

                phase_line.set_data(h_theta, h_thetadot)
                phase_dot.set_data([h_theta[-1]], [h_thetadot[-1]])

                angle_deg = np.degrees(theta_w)
                fig.suptitle(f'SDE  |  step {step}  |  x={cx:.2f}  |  theta={angle_deg:.1f}°')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            if step % 50 == 0:
                angle = np.degrees(np.arctan2(np.sin(x_np[2]), np.cos(x_np[2])))
                print(f'  step {step:3d}: x={x_np[0]:.2f}, theta={angle:.1f}deg')

        plt.ioff()
        input("Press Enter to close...")
        plt.close('all')

    else:
        trajectory = [x.copy()]
        controls = []
        solve_times = []

        for step in range(SIM_STEPS):
            t0 = time.time()
            U = solver.solve_continuous(x)
            solve_times.append(time.time() - t0)
            u = U[0]
            x = model.step(x[None, :], u[None, :], DEFAULT_DT, xp).squeeze(0)
            trajectory.append(x.copy())
            controls.append(u.copy())

            if step % 50 == 0:
                x_np = x.get() if hasattr(x, 'get') else x
                angle = np.degrees(np.arctan2(np.sin(x_np[2]), np.cos(x_np[2])))
                print(f'  step {step:3d}: x={x_np[0]:.2f}, theta={angle:.1f}deg, '
                      f'solve={solve_times[-1]*1000:.1f}ms')

        trajectory = np.array([t.get() if hasattr(t, 'get') else t for t in trajectory])
        controls = np.array([c.get() if hasattr(c, 'get') else c for c in controls])
        theta_wrapped = np.arctan2(np.sin(trajectory[:, 2]), np.cos(trajectory[:, 2]))

        apply_style()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        t_axis = np.arange(len(trajectory)) * DEFAULT_DT

        sc = axes[0, 0].scatter(theta_wrapped[:-1], trajectory[:-1, 3],
                                c=np.arange(len(theta_wrapped) - 1), cmap='coolwarm', s=3)
        axes[0, 0].set_xlabel(r'$\theta$ (rad)'); axes[0, 0].set_ylabel(r'$\dot{\theta}$ (rad/s)')
        axes[0, 0].set_title(r'Phase portrait'); axes[0, 0].axvline(0, color=COLORS['tertiary'], ls='--', alpha=0.5)
        plt.colorbar(sc, ax=axes[0, 0], label='Step')
        label_panel(axes[0, 0], 'a')

        axes[0, 1].plot(t_axis, np.degrees(theta_wrapped), color=COLORS['primary'], lw=1, label=r'$\theta$')
        axes[0, 1].axhline(0, color=COLORS['tertiary'], ls='--', alpha=0.5)
        axes[0, 1].set_xlabel(r'Time (s)'); axes[0, 1].set_ylabel(r'Angle (deg)')
        axes[0, 1].set_title(r'Pole angle and cart position')
        ax_pos = axes[0, 1].twinx()
        ax_pos.plot(t_axis, trajectory[:, 0], color=COLORS['accent'], lw=1, label=r'$x$')
        ax_pos.set_ylabel(r'$x$ (m)')
        label_panel(axes[0, 1], 'b')

        t_ctrl = np.arange(len(controls)) * DEFAULT_DT
        axes[1, 0].plot(t_ctrl, controls[:, 0], color=COLORS['gray'], lw=0.8)
        axes[1, 0].set_xlabel(r'Time (s)'); axes[1, 0].set_ylabel(r'Force $F$ (N)')
        axes[1, 0].set_title('Control input (SDE)')
        lo, hi = model.control_bounds
        axes[1, 0].axhline(lo[0], color=COLORS['secondary'], ls='--', alpha=0.3)
        axes[1, 0].axhline(hi[0], color=COLORS['secondary'], ls='--', alpha=0.3)
        label_panel(axes[1, 0], 'c')

        axes[1, 1].plot(np.arange(len(solve_times)) * DEFAULT_DT,
                        np.array(solve_times) * 1000, color=COLORS['gray'], lw=0.5)
        axes[1, 1].set_xlabel(r'Time (s)'); axes[1, 1].set_ylabel('Solve time (ms)')
        axes[1, 1].set_title('MPPI solve time')
        label_panel(axes[1, 1], 'd')

        fig.suptitle('Cart-pole swing-up -- continuous-time SDE')
        plt.tight_layout()

        outpath = args.save or 'examples/results/continuous/cartpole/cartpole_sde.png'
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Mean solve: {np.mean(solve_times)*1000:.1f}ms | Saved to {outpath}')
        plt.close(fig)


if __name__ == '__main__':
    main()
