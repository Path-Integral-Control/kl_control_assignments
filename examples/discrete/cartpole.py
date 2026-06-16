"""Cart-pole swing-up via MPPI.

Demonstrates MPPI on a nonlinear, underactuated system. The controller
must discover the pump-and-catch strategy to swing the pole from
hanging (theta=pi) to upright (theta=0).

Usage:
    python examples/discrete/cartpole.py                    # Static plot
    python examples/discrete/cartpole.py --animate          # Real-time cart+pole animation
    python examples/discrete/cartpole.py --gpu --samples 4096
"""

import sys
import os
import argparse
import time

import numpy as np

parser = argparse.ArgumentParser(description='Cart-pole swing-up via MPPI')
parser.add_argument('--animate', action='store_true', help='Real-time animation')
parser.add_argument('--continuous', action='store_true', help='Use SDE formulation')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--samples', type=int, default=1024)
parser.add_argument('--save', type=str, default=None)
args = parser.parse_args()

import matplotlib
if args.animate:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import apply_style, label_panel, raise_window, COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mppi import MPPI
from mppi.models import CartPole

# === Parameters ===
DT = 0.02
SIM_STEPS = 300
POLE_LEN = 0.5
CART_W, CART_H = 0.3, 0.15

model = CartPole()
np.random.seed(42)
solver = MPPI(model, K=args.samples, T=50, dt=DT, lambda_=10.0,
              sigma=[3.0], use_gpu=args.gpu)
xp = solver.xp
x = xp.array([0.0, 0.0, np.pi, 0.0], dtype=xp.float32)

mode = 'continuous (SDE)' if args.continuous else 'discrete'
print(f'Cart-pole swing-up | K={args.samples}, {mode}')


if args.animate:
    # === Real-time animation: cart+pole + phase portrait ===
    apply_style()
    fig, (ax_cart, ax_phase) = plt.subplots(1, 2, figsize=(14, 5),
                                             gridspec_kw={'width_ratios': [2, 1]})

    # Cart-pole visual
    ax_cart.set_xlim(-3, 3)
    ax_cart.set_ylim(-0.8, 0.8)
    ax_cart.set_aspect('equal')
    ax_cart.axhline(0, color=COLORS['lightgray'], linewidth=0.5)
    ax_cart.set_xlabel(r'$x$ (m)')
    ax_cart.set_title('Cart-Pole Swing-Up')

    cart_patch = mpatches.FancyBboxPatch((-CART_W/2, -CART_H/2), CART_W, CART_H,
                                         boxstyle="round,pad=0.02",
                                         facecolor=COLORS['primary'], edgecolor='black')
    ax_cart.add_patch(cart_patch)
    pole_line, = ax_cart.plot([], [], color=COLORS['secondary'], linewidth=4, solid_capstyle='round')
    pole_tip, = ax_cart.plot([], [], 'o', color=COLORS['secondary'], markersize=8)

    # Phase portrait
    ax_phase.set_xlabel(r'$\theta$ (rad)')
    ax_phase.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    ax_phase.set_title('Phase portrait')
    ax_phase.axvline(0, color=COLORS['tertiary'], linestyle='--', alpha=0.5)
    ax_phase.axhline(0, color=COLORS['lightgray'], linestyle=':', alpha=0.3)
    ax_phase.set_xlim(-np.pi - 0.5, np.pi + 0.5)
    ax_phase.set_ylim(-8, 8)
    phase_trail, = ax_phase.plot([], [], color=COLORS['primary'], linewidth=1, alpha=0.5)
    phase_dot, = ax_phase.plot([], [], 'o', color=COLORS['secondary'], markersize=6)

    history_theta = []
    history_thetadot = []

    def update(frame):
        global x
        if args.continuous:
            U = solver.solve_continuous(x)
        else:
            U = solver.solve(x)
        u = U[0]
        x = model.step(x[None, :], u[None, :], DT, xp).squeeze(0)
        x_np = x.get() if hasattr(x, 'get') else x

        cart_x = x_np[0]
        theta = x_np[2]
        theta_w = np.arctan2(np.sin(theta), np.cos(theta))

        # Update cart
        cart_patch.set_x(cart_x - CART_W / 2)
        cart_patch.set_y(-CART_H / 2)

        # Update pole (theta=0 is UP)
        pole_end_x = cart_x + POLE_LEN * np.sin(theta)
        pole_end_y = POLE_LEN * np.cos(theta)
        pole_line.set_data([cart_x, pole_end_x], [0, pole_end_y])
        pole_tip.set_data([pole_end_x], [pole_end_y])

        # Auto-scroll cart view
        ax_cart.set_xlim(cart_x - 2, cart_x + 2)

        # Phase portrait
        history_theta.append(theta_w)
        history_thetadot.append(x_np[3])
        phase_trail.set_data(history_theta, history_thetadot)
        phase_dot.set_data([theta_w], [x_np[3]])

        angle_deg = np.degrees(theta_w)
        u_val = float(u[0].get() if hasattr(u[0], 'get') else u[0])
        fig.suptitle(f'Step {frame}  |  x={cart_x:.2f}m  |  '
                     rf'$\theta$={angle_deg:.0f}°  |  F={u_val:.1f}N')

        if frame % 50 == 0:
            print(f'  step {frame}: x={cart_x:.2f}, theta={angle_deg:.1f} deg')

        return cart_patch, pole_line, pole_tip, phase_trail, phase_dot

    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=SIM_STEPS, interval=20,
                         blit=False, repeat=False, cache_frame_data=False)
    raise_window(fig)
    plt.show()

    # Save final state
    os.makedirs('examples/results/discrete/cartpole', exist_ok=True)
    if len(history_theta) > 0:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        ax2.plot(np.arange(len(history_theta)) * DT, np.degrees(history_theta),
                 color=COLORS['primary'])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel(r'$\theta$ (deg)')
        ax2.set_title('Cart-pole swing-up')
        ax2.axhline(0, color=COLORS['tertiary'], linestyle='--', alpha=0.5)
        plt.savefig('examples/results/discrete/cartpole/cartpole.png')
        print('Saved to examples/results/discrete/cartpole/cartpole.png')
        plt.close(fig2)

else:
    # === Static: run simulation, then plot ===
    trajectory = [x.copy()]
    controls = []
    costs_per_step = []

    for step in range(SIM_STEPS):
        if args.continuous:
            U = solver.solve_continuous(x)
        else:
            U = solver.solve(x)
        u = U[0]
        x = model.step(x[None, :], u[None, :], DT, xp).squeeze(0)
        trajectory.append(x.copy())
        controls.append(u.copy())
        costs_per_step.append(float(xp.min(solver.get_costs())))

        if step % 50 == 0:
            x_np = x.get() if hasattr(x, 'get') else x
            angle = np.degrees(np.arctan2(np.sin(x_np[2]), np.cos(x_np[2])))
            print(f'  step {step:3d}: x={x_np[0]:.2f}, theta={angle:.1f} deg')

    if hasattr(trajectory[0], 'get'):
        trajectory = [t.get() for t in trajectory]
        controls = [c.get() for c in controls]
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    theta_wrapped = np.arctan2(np.sin(trajectory[:, 2]), np.cos(trajectory[:, 2]))

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t_axis = np.arange(len(trajectory)) * DT

    sc = axes[0, 0].scatter(theta_wrapped[:-1], trajectory[:-1, 3],
                             c=np.arange(len(theta_wrapped) - 1), cmap='coolwarm', s=3)
    axes[0, 0].set_xlabel(r'$\theta$ (rad)')
    axes[0, 0].set_ylabel(r'$\dot{\theta}$ (rad/s)')
    axes[0, 0].set_title(r'Phase portrait')
    axes[0, 0].axvline(0, color=COLORS['tertiary'], linestyle='--', alpha=0.5)
    plt.colorbar(sc, ax=axes[0, 0], label='Step')
    label_panel(axes[0, 0], 'a')

    axes[0, 1].plot(t_axis, np.degrees(theta_wrapped), color=COLORS['primary'], linewidth=1.0)
    axes[0, 1].axhline(0, color=COLORS['tertiary'], linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel(r'$\theta$ (deg)')
    axes[0, 1].set_title('Pole angle')
    ax2b = axes[0, 1].twinx()
    ax2b.plot(t_axis, trajectory[:, 0], color=COLORS['accent'], linewidth=1.0)
    ax2b.set_ylabel('Cart $x$ (m)')
    label_panel(axes[0, 1], 'b')

    t_ctrl = np.arange(len(controls)) * DT
    axes[1, 0].plot(t_ctrl, controls[:, 0], color=COLORS['gray'], linewidth=0.8)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Force $F$ (N)')
    axes[1, 0].set_title('Control input')
    lo, hi = model.control_bounds
    axes[1, 0].axhline(lo[0], color=COLORS['secondary'], linestyle='--', alpha=0.3)
    axes[1, 0].axhline(hi[0], color=COLORS['secondary'], linestyle='--', alpha=0.3)
    label_panel(axes[1, 0], 'c')

    axes[1, 1].plot(np.arange(len(costs_per_step)) * DT, costs_per_step,
                    color=COLORS['gray'], linewidth=0.8)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Min trajectory cost')
    axes[1, 1].set_title('Cost per step')
    axes[1, 1].set_yscale('log')
    label_panel(axes[1, 1], 'd')

    plt.tight_layout()

    outpath = args.save or 'examples/results/discrete/cartpole/cartpole.png'
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Saved to {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    pass
