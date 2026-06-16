#!/usr/bin/env python3
"""MPPI example: Double Integrator with obstacle avoidance.

A 2D point mass must reach a goal while avoiding circular obstacles.
Demonstrates both discrete-time and continuous-time (SDE) MPPI.

Usage:
    python examples/double_integrator.py --animate
    python examples/double_integrator.py --animate --continuous
    python examples/double_integrator.py --animate --gpu
    python examples/double_integrator.py --save trajectory.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
if '--animate' in sys.argv:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mppi import MPPI
from mppi.backend import get_backend, set_backend
from mppi.models.double_integrator import DoubleIntegrator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import raise_window

OBSTACLES = [(3.0, 2.0, 1.0), (6.0, 5.0, 1.2), (2.0, 6.0, 0.8)]
GOAL = [8.0, 8.0]
N_SHOW_SAMPLES = 40


def make_solver(args):
    set_backend(args.gpu)
    model = DoubleIntegrator(goal=GOAL, obs=OBSTACLES)
    solver = MPPI(model, K=1024, T=40, dt=0.05, lambda_=10.0, sigma=[1.0, 1.0], use_gpu=args.gpu)
    return model, solver


def draw_static_elements(ax):
    for cx, cy, r in OBSTACLES:
        ax.add_patch(plt.Circle((cx, cy), r, color="red", alpha=0.25))
    ax.plot(*GOAL, "r*", markersize=18, zorder=5)
    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-1.5, 10.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def run_animate(args):
    xp = get_backend()
    model, solver = make_solver(args)
    solve_fn = solver.solve_continuous if args.continuous else solver.solve

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_static_elements(ax)
    mode = "continuous" if args.continuous else "discrete"
    ax.set_title(f"MPPI Double Integrator [{mode}]")

    sample_lines = [ax.plot([], [], color="steelblue", alpha=0.2, lw=0.8)[0] for _ in range(N_SHOW_SAMPLES)]
    planned_line, = ax.plot([], [], color="orangered", lw=3.5, solid_capstyle="round", label="planned trajectory", zorder=8)
    trail_line, = ax.plot([], [], "b-", lw=2.0, label="closed-loop trajectory")
    pos_dot, = ax.plot([], [], "ko", markersize=8, zorder=10)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
                        fontfamily="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=9)

    state = dict(
        x=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        traj_x=[0.0], traj_y=[0.0], step=0, done=False,
    )

    def update(frame):
        if state["done"]:
            return []

        x = state["x"]
        U = solve_fn(x)
        sampled = solver.get_sampled_trajectories()
        weights = solver.get_weights()
        if hasattr(U, "get"):
            U, sampled, weights = U.get(), sampled.get(), weights.get()

        top_idx = np.argsort(weights)[-N_SHOW_SAMPLES:]
        for i, idx in enumerate(top_idx):
            sample_lines[i].set_data(sampled[idx, :, 0], sampled[idx, :, 1])

        planned = solver.get_planned_trajectory(x)
        if hasattr(planned, "get"):
            planned = planned.get()
        planned_line.set_data(planned[:, 0], planned[:, 1])

        u0 = U[0]
        x_batch = xp.array(x[None, :], dtype=xp.float32)
        u_batch = xp.array(u0[None, :], dtype=xp.float32)
        x_next = model.step(x_batch, u_batch, 0.05, xp)
        if hasattr(x_next, "get"):
            x_next = x_next.get()
        state["x"] = x_next.squeeze(0)

        state["traj_x"].append(float(state["x"][0]))
        state["traj_y"].append(float(state["x"][1]))
        state["step"] += 1

        trail_line.set_data(state["traj_x"], state["traj_y"])
        pos_dot.set_data([state["x"][0]], [state["x"][1]])

        dist = np.linalg.norm(state["x"][:2] - np.array(GOAL))
        speed = np.linalg.norm(state["x"][2:4])
        info_text.set_text(f"step {state['step']:3d}  dist={dist:.2f}  speed={speed:.2f}")

        if dist < 0.3:
            info_text.set_text(f"REACHED GOAL in {state['step']} steps!")
            state["done"] = True

        return sample_lines + [planned_line, trail_line, pos_dot, info_text]

    anim = FuncAnimation(fig, update, frames=300, interval=30, blit=True, repeat=False)

    if args.save and args.save.endswith((".mp4", ".gif")):
        anim.save(args.save, fps=20, dpi=120)
        print(f"Saved animation to {args.save}")
    else:
        raise_window(fig)
        plt.show()


def run_static(args):
    xp = get_backend()
    model, solver = make_solver(args)
    solve_fn = solver.solve_continuous if args.continuous else solver.solve

    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    traj = [x[:2].copy()]

    for step in range(200):
        U = solve_fn(x)
        if hasattr(U, "get"):
            U = U.get()
        x_batch = xp.array(x[None, :], dtype=xp.float32)
        u_batch = xp.array(U[0][None, :], dtype=xp.float32)
        x_next = model.step(x_batch, u_batch, 0.05, xp)
        if hasattr(x_next, "get"):
            x_next = x_next.get()
        x = x_next.squeeze(0)
        traj.append(x[:2].copy())
        if np.linalg.norm(x[:2] - np.array(GOAL)) < 0.3:
            print(f"Reached goal in {step + 1} steps.")
            break

    traj = np.array(traj)
    sampled = solver.get_sampled_trajectories()
    weights = solver.get_weights()
    if hasattr(sampled, "get"):
        sampled, weights = sampled.get(), weights.get()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mode = "continuous" if args.continuous else "discrete"
    backend = "GPU" if args.gpu else "CPU"
    fig.suptitle(f"MPPI Double Integrator [{mode}, {backend}]", fontsize=14)

    for ax_idx, ax in enumerate(axes):
        draw_static_elements(ax)

    axes[0].set_title("Trajectory")
    axes[0].plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="trajectory")
    axes[0].plot(traj[0, 0], traj[0, 1], "go", markersize=10, label="start")
    axes[0].legend()

    axes[1].set_title("Samples and planned trajectory (last solve)")
    n_show = min(50, sampled.shape[0])
    top_idx = np.argsort(weights)[-n_show:]
    for idx in top_idx:
        axes[1].plot(sampled[idx, :, 0], sampled[idx, :, 1], color="steelblue", alpha=0.2, lw=0.8)
    planned = solver.get_planned_trajectory(x)
    if hasattr(planned, "get"):
        planned = planned.get()
    axes[1].plot(planned[:, 0], planned[:, 1], color="orangered", lw=3.5, label="planned trajectory")
    axes[1].plot(traj[-1, 0], traj[-1, 1], "bs", markersize=8, label="current pos")
    axes[1].legend()

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    if args.animate:
        raise_window(fig)
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI Double Integrator Example")
    parser.add_argument("--animate", action="store_true", help="Real-time animated visualization")
    parser.add_argument("--continuous", action="store_true", help="Use continuous-time SDE formulation")
    parser.add_argument("--gpu", action="store_true", help="Use CuPy for GPU acceleration")
    parser.add_argument("--save", type=str, default=None, help="Save figure/animation to file")
    args = parser.parse_args()
    if args.animate:
        run_animate(args)
    else:
        run_static(args)
