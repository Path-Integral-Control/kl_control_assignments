"""Fixed-wing MPPI with nominal controller — live visualization.

Shows MPPI sampled rollouts in 3D as the aircraft flies.

Usage:
    python examples/discrete/fixed_wing/mppi.py              # Live
    python examples/discrete/fixed_wing/mppi.py --headless   # Save PNG only
    python examples/discrete/fixed_wing/mppi.py --noise
    python examples/discrete/fixed_wing/mppi.py --K 256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import argparse
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore', message='.*Grid size.*')
warnings.filterwarnings('ignore', message='.*tight_layout.*')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from plot_style import raise_window

parser = argparse.ArgumentParser(description='Fixed-wing MPPI')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--no-nominal', action='store_true')
parser.add_argument('--noise', action='store_true')
parser.add_argument('--gpu', action='store_true', help='Use CuPy GPU backend')
parser.add_argument('--jit', action='store_true', help='Use Numba JIT-compiled dynamics (~3x faster)')
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--K', type=int, default=512)
args = parser.parse_args()

import matplotlib
if args.headless:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mppi import MPPI
from mppi.models.fixed_wing import FixedWing
from mppi.models.fixed_wing.dynamics_jit import FixedWingJIT
from mppi.models.fixed_wing.controller import NominalController, _quat_to_euler, _quat_rotate
from mppi.models.fixed_wing.path_follower import PathFollower

# === Setup ===
if args.gpu:
    from mppi.models.fixed_wing.dynamics_gpu import FixedWingGPU
    ModelClass = FixedWingGPU
elif args.jit:
    ModelClass = FixedWingJIT
else:
    ModelClass = FixedWing
model = ModelClass(n_substeps=3)
model.target_speed = 6.5
model_exec = ModelClass(n_substeps=50)
model_exec.target_speed = 6.5
wp = model.waypoints
dt = 0.04

controller = NominalController(gains_path='config/fixedwing_controller_gains.yaml')
controller.trim_elev = 0.30
controller.phi_lim = np.radians(40.0)
guidance = PathFollower(params={'v_cruise_nominal': 6.5})

MPPI_T = 64
MPPI_LAMBDA = 10000.0
MPPI_SIGMA = [0.4, 0.3, 0.5, 0.3]

np.random.seed(42)
solver_dtype = np.float64 if args.jit else np.float32
solver = MPPI(model, K=args.K, T=MPPI_T, dt=dt,
              lambda_=MPPI_LAMBDA, sigma=MPPI_SIGMA, use_gpu=args.gpu,
              warm_start=True, dtype=solver_dtype)

# Initial state
x0 = model.trim_state()
x0[0], x0[1], x0[2] = wp[0, 0], wp[0, 1], wp[0, 2]
dx, dy = wp[1, 0] - wp[0, 0], wp[1, 1] - wp[0, 1]
yaw0 = np.arctan2(dy, dx)
x0[6], x0[9] = np.cos(yaw0 / 2), np.sin(yaw0 / 2)

# Initialize nominal with controller rollout
if not args.no_nominal:
    xx = x0.copy()
    U_nom = np.zeros((MPPI_T, 4))
    for t in range(MPPI_T):
        vw = _quat_rotate(xx[6], xx[7], xx[8], xx[9], xx[3], xx[4], xx[5])
        ref = guidance.compute_reference(xx[0], xx[1], xx[2], vw[0], vw[1], vw[2], xx[3], wp, dt)
        u = controller.compute(xx, ref, dt)
        u[1] = -u[1]
        U_nom[t] = u
        xx = model_exec.step(xx.reshape(1, -1), u.reshape(1, -1), dt, np).flatten()
    xp = solver.xp
    solver.U = xp.array(U_nom, dtype=solver_dtype)

x = xp.array(x0, dtype=solver_dtype)

# === Figure ===
if not args.headless:
    plt.ion()

fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

from scipy.interpolate import splprep, splev
hw = model.track_half_width
wp_closed = np.vstack([wp[:, :2], wp[:1, :2]])
tck, _ = splprep([wp_closed[:, 0], wp_closed[:, 1]], s=0, per=True)
u_fine = np.linspace(0, 1, 300)
cx, cy = splev(u_fine, tck)
from shapely.geometry import LineString
centerline = LineString(np.column_stack([cx, cy]))
left_line = centerline.offset_curve(hw)
right_line = centerline.offset_curve(-hw)
left_edge = np.array(left_line.coords)
right_edge = np.array(right_line.coords)

ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.plot(cx, cy, np.full_like(cx, wp[0, 2]), 'k--', linewidth=0.5, alpha=0.3)
ax3d.plot(left_edge[:, 0], left_edge[:, 1], np.full(len(left_edge), wp[0, 2]),
          color='gray', linewidth=0.8, alpha=0.4)
ax3d.plot(right_edge[:, 0], right_edge[:, 1], np.full(len(right_edge), wp[0, 2]),
          color='gray', linewidth=0.8, alpha=0.4)
ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
ax3d.set_title('3D Flight + Rollouts')
trail_3d, = ax3d.plot([], [], [], 'dodgerblue', linewidth=2)
AC_S = 1.5
ac_body = np.array([
    [AC_S, 0, 0], [-AC_S, 0, 0],
    [0.15*AC_S, -AC_S, 0], [0.15*AC_S, AC_S, 0],
    [-0.85*AC_S, -0.4*AC_S, 0], [-0.85*AC_S, 0.4*AC_S, 0],
    [-0.85*AC_S, 0, 0], [-0.85*AC_S, 0, 0.35*AC_S],
])
ac_fuse, = ax3d.plot([], [], [], 'r-', linewidth=2.5, zorder=10)
ac_wing, = ax3d.plot([], [], [], 'r-', linewidth=2.5, zorder=10)
ac_htail, = ax3d.plot([], [], [], 'r-', linewidth=2, zorder=10)
ac_vtail, = ax3d.plot([], [], [], 'r-', linewidth=2, zorder=10)
N_SHOW = min(128, args.K)
sample_lines = [ax3d.plot([], [], [], 'c-', linewidth=0.2, alpha=0.15)[0] for _ in range(N_SHOW)]

ax_top = fig.add_subplot(gs[0, 1])
track_poly = np.vstack([left_edge, right_edge[::-1]])
ax_top.fill(track_poly[:, 0], track_poly[:, 1], color='#e8e8e8', zorder=0)
ax_top.plot(left_edge[:, 0], left_edge[:, 1], 'k-', linewidth=1.2, alpha=0.6)
ax_top.plot(right_edge[:, 0], right_edge[:, 1], 'k-', linewidth=1.2, alpha=0.6)
ax_top.plot(cx, cy, 'k--', linewidth=0.5, alpha=0.3)
ax_top.set_aspect('equal')
ax_top.set_title('Top-Down + Rollouts')
trail_top, = ax_top.plot([], [], 'dodgerblue', linewidth=1.5)
sample_lines_top = [ax_top.plot([], [], 'c-', linewidth=0.2, alpha=0.15)[0] for _ in range(N_SHOW)]
ac_fuse_top, = ax_top.plot([], [], 'r-', linewidth=2, zorder=10)
ac_wing_top, = ax_top.plot([], [], 'r-', linewidth=2, zorder=10)
ac_htail_top, = ax_top.plot([], [], 'r-', linewidth=1.5, zorder=10)

ax_alt = fig.add_subplot(gs[0, 2])
ax_alt.axhline(3.0, color='green', linestyle='--', alpha=0.4)
ax_alt.set_title('Altitude'); ax_alt.set_ylabel('z (m)')
alt_line, = ax_alt.plot([], [], 'dodgerblue')

ax_cost = fig.add_subplot(gs[1, 1])
cost_line, = ax_cost.plot([], [], 'k-', linewidth=0.8)
ax_cost.set_title('Best Trajectory Cost'); ax_cost.set_ylabel('Cost')
ax_cost.set_yscale('symlog', linthresh=100)

ax_ctrl = fig.add_subplot(gs[1, 2])
ax_ctrl.set_ylim(-1.2, 1.2)
ax_ctrl.set_title('Controls'); ax_ctrl.set_ylabel('Command')
ail_l, = ax_ctrl.plot([], [], color='#d62728', lw=1.2, linestyle='-', label='Ail')
elev_l, = ax_ctrl.plot([], [], color='#1f77b4', lw=1.2, linestyle='--', label='Elev')
thr_l, = ax_ctrl.plot([], [], color='#2ca02c', lw=1.2, linestyle='-.', label='Thr')
rud_l, = ax_ctrl.plot([], [], color='#ff7f0e', lw=1.2, linestyle=':', label='Rud')
ax_ctrl.legend(fontsize=7, ncol=4, loc='upper right')

plt.tight_layout()
if not args.headless:
    raise_window(fig)
    plt.show()

# === Simulate live ===
hx, hy, hz = [], [], []
hcost = []
cail, celev, cthr, crud = [], [], [], []
CAM_R = 12

print(f'MPPI Fixed-Wing | K={args.K}, T={MPPI_T}, noise={args.noise}')

for step in range(args.steps):
    t0 = time.time()
    U_opt = solver.solve(x)
    solve_ms = (time.time() - t0) * 1000

    u = U_opt[0]
    u_np = u.get() if hasattr(u, 'get') else np.array(u)
    if args.noise:
        u_np += np.array(MPPI_SIGMA) * np.random.randn(4)
        u_np = np.clip(u_np, *model.control_bounds)

    x_cpu = x.get() if hasattr(x, 'get') else np.array(x)
    x_next = model_exec.step(x_cpu.reshape(1, -1), u_np.reshape(1, -1), dt, np).flatten()
    x = xp.array(x_next, dtype=solver_dtype)

    x_np = x.get() if hasattr(x, 'get') else np.array(x)
    hx.append(float(x_np[0])); hy.append(float(x_np[1])); hz.append(float(x_np[2]))
    costs = solver.get_costs()
    hcost.append(float(np.min(costs.get() if hasattr(costs, 'get') else costs)))
    cail.append(u_np[0]); celev.append(u_np[1])
    cthr.append(u_np[2]); crud.append(u_np[3])

    # Update display
    if not args.headless and step % 2 == 0:
        n = len(hx); i0 = max(0, n - 150)
        t_arr = np.arange(i0, n) * dt
        x_d = x.get() if hasattr(x, 'get') else np.array(x)
        roll, _, yaw = _quat_to_euler(x_d[6], x_d[7], x_d[8], x_d[9])
        px, py, pz = float(x_d[0]), float(x_d[1]), float(x_d[2])
        spd = float(np.linalg.norm(x_d[3:6]))

        # Aircraft wireframe rotated by quaternion
        ac_w = np.zeros((8, 3))
        for i in range(8):
            rx, ry, rz = _quat_rotate(x_d[6], x_d[7], x_d[8], x_d[9], ac_body[i, 0], ac_body[i, 1], ac_body[i, 2])
            ac_w[i] = [px + rx, py + ry, pz + rz]

        # 3D trail + aircraft
        trail_3d.set_data(hx[i0:], hy[i0:])
        trail_3d.set_3d_properties(hz[i0:])
        ac_fuse.set_data(ac_w[0:2, 0], ac_w[0:2, 1]); ac_fuse.set_3d_properties(ac_w[0:2, 2])
        ac_wing.set_data(ac_w[2:4, 0], ac_w[2:4, 1]); ac_wing.set_3d_properties(ac_w[2:4, 2])
        ac_htail.set_data(ac_w[4:6, 0], ac_w[4:6, 1]); ac_htail.set_3d_properties(ac_w[4:6, 2])
        ac_vtail.set_data(ac_w[6:8, 0], ac_w[6:8, 1]); ac_vtail.set_3d_properties(ac_w[6:8, 2])
        ax3d.view_init(elev=35, azim=np.degrees(yaw) + 210)
        ax3d.set_xlim(px - CAM_R, px + CAM_R)
        ax3d.set_ylim(py - CAM_R, py + CAM_R)
        ax3d.set_zlim(max(0, pz - 4), pz + 4)

        # Sampled rollouts
        samples = solver.get_sampled_trajectories()
        weights = solver.get_weights()
        if hasattr(samples, 'get'):
            samples, weights = samples.get(), weights.get()
        top_idx = np.argsort(weights)[-N_SHOW:]
        for j, idx in enumerate(top_idx):
            s = samples[idx]
            sample_lines[j].set_data(s[:, 0], s[:, 1])
            sample_lines[j].set_3d_properties(s[:, 2])
            sample_lines_top[j].set_data(s[:, 0], s[:, 1])

        # Top-down
        trail_top.set_data(hx[i0:], hy[i0:])
        ac_fuse_top.set_data(ac_w[0:2, 0], ac_w[0:2, 1])
        ac_wing_top.set_data(ac_w[2:4, 0], ac_w[2:4, 1])
        ac_htail_top.set_data(ac_w[4:6, 0], ac_w[4:6, 1])
        fr = CAM_R * 0.8
        ax_top.set_xlim(px - fr, px + fr); ax_top.set_ylim(py - fr, py + fr)

        # Strips
        t_now = step * dt; tlo = max(0, t_now - 3); thi = t_now + 0.5
        alt_line.set_data(t_arr, hz[i0:])
        ax_alt.set_xlim(tlo, thi)
        ax_alt.set_ylim(max(0, min(hz[i0:]) - 0.5), max(hz[i0:]) + 0.5)

        cost_line.set_data(np.arange(len(hcost)) * dt, hcost)
        ax_cost.set_xlim(tlo, thi)
        cost_slice = np.array(hcost[i0:])
        if len(cost_slice) > 0:
            cmin, cmax = cost_slice.min(), cost_slice.max()
            pad = max(abs(cmin), abs(cmax), 100) * 0.2
            ax_cost.set_ylim(cmin - pad, cmax + pad)

        ail_l.set_data(t_arr, cail[i0:]); elev_l.set_data(t_arr, celev[i0:])
        thr_l.set_data(t_arr, cthr[i0:]); rud_l.set_data(t_arr, crud[i0:])
        ax_ctrl.set_xlim(tlo, thi)

        fig.suptitle(f'MPPI  |  t={t_now:.1f}s  |  alt={pz:.1f}m  |  V={spd:.1f}m/s  |  solve={solve_ms:.0f}ms')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    if step % 50 == 0:
        print(f'  step {step}: alt={float(x_np[2]):.1f} spd={float(np.linalg.norm(x_np[3:6])):.1f} solve={solve_ms:.0f}ms')

# === Save ===
os.makedirs('examples/results/discrete/fixed_wing', exist_ok=True)
suffix = '_noisy' if args.noise else ''
fig_s, ax_s = plt.subplots(figsize=(10, 8))
ax_s.plot(wp[:, 0], wp[:, 1], 'k--', linewidth=0.5)
ax_s.plot(hx, hy, 'dodgerblue', linewidth=2)
ax_s.set_aspect('equal')
ax_s.set_title('MPPI fixed-wing flight')
outpath = f'examples/results/discrete/fixed_wing/mppi{suffix}.png'
plt.savefig(outpath)
print(f'Saved to {outpath}')
plt.close(fig_s)

if not args.headless:
    plt.ioff()
    input("Press Enter to close...")
plt.close('all')
