#!/usr/bin/env python3
"""MPPI example: Unicycle with nonholonomic constraints.

A 2D vehicle that can only drive forward (no strafing). Demonstrates how MPPI
handles orientation-dependent dynamics. Supports discrete-time and
continuous-time (SDE with noncommutative noise) formulations.

Usage:
    python examples/discrete/unicycle.py --animate
    python examples/discrete/unicycle.py --env config/environments/forest.yaml --animate
    python examples/discrete/unicycle.py --animate --continuous
    python examples/discrete/unicycle.py --gpu
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
if '--animate' in sys.argv:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mppi import MPPI
from mppi.backend import get_backend, set_backend
from mppi.models.unicycle import Unicycle
from environments import Environment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import raise_window

DEFAULT_OBSTACLES = [(3.0, 3.0, 0.8), (5.0, 2.0, 0.6), (2.0, 5.5, 0.7)]
DEFAULT_GOAL = [7.0, 7.0]
DEFAULT_START = [0.0, 0.0]
N_SHOW_SAMPLES = 50
DT = 0.05


def make_solver(args, env=None):
    set_backend(args.gpu)
    if env is not None:
        goal = list(env.goal_pos)
        model = Unicycle(goal=goal, obs=None)
        _original_cost = model.running_cost
        def _env_running_cost(x, u, t, xp):
            cost = _original_cost(x, u, t, xp)
            x_np = x.get() if hasattr(x, 'get') else x
            env_cost = env.get_obstacle_cost(x_np)
            cost += xp.array(env_cost, dtype=cost.dtype)
            return cost
        model.running_cost = _env_running_cost
        def _clamp(x, xp):
            if hasattr(x, 'get'):
                x_np = x.get()
                env.clamp_to_obstacles(x_np)
                x[:] = xp.asarray(x_np)
            else:
                env.clamp_to_obstacles(x)
            return x
        model.clamp_state = _clamp
    else:
        model = Unicycle(goal=DEFAULT_GOAL, obs=DEFAULT_OBSTACLES)
    solver = MPPI(model, K=args.K, T=args.T, dt=DT, lambda_=args.lambda_,
                  sigma=[args.sigma, args.sigma * 2], use_gpu=args.gpu)
    return model, solver


def draw_static_elements(ax, env=None):
    if env is not None:
        xx, yy = np.meshgrid(np.linspace(-2, 12, 200), np.linspace(-2, 12, 200))
        C = env.cost_field(xx, yy)
        ax.contourf(xx, yy, np.clip(C, 0, 5000), levels=30, cmap='YlOrRd', alpha=0.5)
        for o in env.obstacles:
            if o['type'] == 'circle':
                c = mpatches.Circle(o['position'], o['radius'], edgecolor='white',
                                    facecolor='none', linewidth=0.5, alpha=0.6)
                ax.add_patch(c)
        ax.plot(*env.start_pos, 'o', color='green', markersize=10, zorder=10)
        ax.plot(*env.goal_pos, '*', color='gold', markersize=14, zorder=10)
    else:
        for cx, cy, r in DEFAULT_OBSTACLES:
            ax.add_patch(plt.Circle((cx, cy), r, color="red", alpha=0.25))
        ax.plot(*DEFAULT_GOAL, "r*", markersize=18, zorder=5)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def run_animate(args, env=None):
    xp = get_backend()
    model, solver = make_solver(args, env)
    solve_fn = solver.solve_continuous if args.continuous else solver.solve

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_static_elements(ax, env)
    mode = "continuous" if args.continuous else "discrete"
    env_label = f" — {os.path.splitext(os.path.basename(args.env))[0]}" if args.env else ""
    ax.set_title(f"MPPI Unicycle [{mode}]{env_label}")

    sample_lines = [ax.plot([], [], color="steelblue", alpha=0.2, lw=0.8)[0] for _ in range(N_SHOW_SAMPLES)]
    planned_line, = ax.plot([], [], color="orangered", lw=3.5, solid_capstyle="round", label="planned trajectory", zorder=8)
    trail_line, = ax.plot([], [], "b-", lw=2.0, label="closed-loop trajectory")
    pos_dot, = ax.plot([], [], "ko", markersize=6, zorder=10)
    heading_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="->", color="black", lw=2.0))
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
                        fontfamily="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=9)

    start = env.start_pos if env else DEFAULT_START
    goal = list(env.goal_pos) if env else DEFAULT_GOAL
    heading0 = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    state = dict(
        x=np.array([start[0], start[1], heading0], dtype=np.float32),
        traj_x=[start[0]], traj_y=[start[1]], step=0, done=False,
        crash_count=0,
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
        x_next = model.step(x_batch, u_batch, DT, xp)
        if hasattr(x_next, "get"):
            x_next = x_next.get()
        state["x"] = x_next.squeeze(0)

        if env is not None:
            pos_before = state["x"][:2].copy()
            env.clamp_to_obstacles(state["x"][None, :])
            if not np.allclose(pos_before, state["x"][:2]):
                state["crash_count"] += 1

        state["traj_x"].append(float(state["x"][0]))
        state["traj_y"].append(float(state["x"][1]))
        state["step"] += 1

        trail_line.set_data(state["traj_x"], state["traj_y"])
        pos_dot.set_data([state["x"][0]], [state["x"][1]])

        arrow_len = 0.5
        dx = arrow_len * np.cos(state["x"][2])
        dy = arrow_len * np.sin(state["x"][2])
        heading_arrow.set_position((state["x"][0], state["x"][1]))
        heading_arrow.xy = (state["x"][0] + dx, state["x"][1] + dy)

        dist = np.linalg.norm(state["x"][:2] - np.array(goal))
        info_text.set_text(f"step {state['step']:3d}  dist={dist:.2f}  crashes={state['crash_count']}")

        if dist < 0.3:
            info_text.set_text(f"REACHED GOAL in {state['step']} steps!  crashes={state['crash_count']}")
            state["done"] = True

        return sample_lines + [planned_line, trail_line, pos_dot, heading_arrow, info_text]

    anim = FuncAnimation(fig, update, frames=400, interval=30, blit=True, repeat=False)

    if args.save and args.save.endswith((".mp4", ".gif")):
        anim.save(args.save, fps=20, dpi=120)
        print(f"Saved animation to {args.save}")
    else:
        raise_window(fig)
        plt.show()


def run_static(args, env=None):
    xp = get_backend()
    model, solver = make_solver(args, env)
    solve_fn = solver.solve_continuous if args.continuous else solver.solve

    start = env.start_pos if env else DEFAULT_START
    goal = list(env.goal_pos) if env else DEFAULT_GOAL
    heading0 = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    x = np.array([start[0], start[1], heading0], dtype=np.float32)
    traj = [x.copy()]
    controls = []

    crash_count = 0
    for step in range(300):
        U = solve_fn(x)
        if hasattr(U, "get"):
            U = U.get()
        controls.append(U[0].copy())
        x_batch = xp.array(x[None, :], dtype=xp.float32)
        u_batch = xp.array(U[0][None, :], dtype=xp.float32)
        x_next = model.step(x_batch, u_batch, DT, xp)
        if hasattr(x_next, "get"):
            x_next = x_next.get()
        x = x_next.squeeze(0)
        if env is not None:
            pos_before = x[:2].copy()
            env.clamp_to_obstacles(x[None, :])
            if not np.allclose(pos_before, x[:2]):
                crash_count += 1
        traj.append(x.copy())
        if np.linalg.norm(x[:2] - np.array(goal)) < 0.3:
            print(f"Reached goal in {step + 1} steps. (obstacle crashes: {crash_count})")
            break

    traj = np.array(traj)
    controls = np.array(controls)
    sampled = solver.get_sampled_trajectories()
    weights = solver.get_weights()
    if hasattr(sampled, "get"):
        sampled, weights = sampled.get(), weights.get()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    mode = "continuous" if args.continuous else "discrete"
    backend = "GPU" if args.gpu else "CPU"
    env_label = f" — {os.path.splitext(os.path.basename(args.env))[0]}" if args.env else ""
    fig.suptitle(f"MPPI Unicycle [{mode}, {backend}]{env_label}", fontsize=14)

    # Trajectory with heading
    ax = axes[0]
    ax.set_title("Trajectory with heading")
    draw_static_elements(ax, env)
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2)
    ax.plot(traj[0, 0], traj[0, 1], "go", markersize=10, label="start")
    arrow_step = max(1, len(traj) // 20)
    for i in range(0, len(traj), arrow_step):
        dx = 0.3 * np.cos(traj[i, 2])
        dy = 0.3 * np.sin(traj[i, 2])
        ax.arrow(traj[i, 0], traj[i, 1], dx, dy, head_width=0.12, head_length=0.06, fc="navy", ec="navy")
    ax.legend()

    # Sampled trajectories + planned
    ax = axes[1]
    ax.set_title("Samples and planned trajectory (last solve)")
    draw_static_elements(ax, env)
    n_show = min(60, sampled.shape[0])
    top_idx = np.argsort(weights)[-n_show:]
    for idx in top_idx:
        ax.plot(sampled[idx, :, 0], sampled[idx, :, 1], color="steelblue", alpha=0.2, lw=0.8)
    planned = solver.get_planned_trajectory(x)
    if hasattr(planned, "get"):
        planned = planned.get()
    ax.plot(planned[:, 0], planned[:, 1], color="orangered", lw=3.5, label="planned trajectory")
    ax.plot(traj[-1, 0], traj[-1, 1], "bs", markersize=8, label="current pos")
    ax.legend()

    # Controls
    ax = axes[2]
    ax.set_title("Control inputs")
    t = np.arange(len(controls)) * DT
    ax.plot(t, controls[:, 0], label="v (speed)")
    ax.plot(t, controls[:, 1], label="omega (yaw rate)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("control")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    if args.animate:
        raise_window(fig)
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI Unicycle Example")
    parser.add_argument("--animate", action="store_true", help="Real-time animated visualization")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment YAML (three_mountains, forest, u_trap, double_slit, drunken_bridge, landing_site, simple_goal)")
    parser.add_argument("--continuous", action="store_true", help="Use continuous-time SDE formulation")
    parser.add_argument("--gpu", action="store_true", help="Use CuPy for GPU acceleration")
    parser.add_argument("--save", type=str, default=None, help="Save figure/animation to file")
    parser.add_argument("--K", type=int, default=2048, help="Number of samples")
    parser.add_argument("--T", type=int, default=50, help="Planning horizon")
    parser.add_argument("--sigma", type=float, default=0.5, help="Control noise std dev")
    parser.add_argument("--lambda_", "--lambda", type=float, default=5.0, help="Temperature parameter")
    args = parser.parse_args()

    env = Environment.from_yaml(args.env) if args.env else None
    if args.animate:
        run_animate(args, env)
    else:
        run_static(args, env)
