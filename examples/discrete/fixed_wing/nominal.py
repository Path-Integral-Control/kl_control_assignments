"""Fixed-wing flight with nominal controller — live visualization.

Usage:
    python examples/discrete/fixed_wing/nominal.py
    python examples/discrete/fixed_wing/nominal.py --headless
    python examples/discrete/fixed_wing/nominal.py --noise
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Fixed-wing nominal controller')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--steps', type=int, default=750)
parser.add_argument('--noise', action='store_true')
args = parser.parse_args()

import matplotlib
if args.headless:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from plot_style import raise_window

from mppi.models.fixed_wing import FixedWing
from mppi.models.fixed_wing.controller import NominalController, _quat_to_euler, _quat_rotate
from mppi.models.fixed_wing.path_follower import PathFollower

# === Setup ===
model = FixedWing()
model.target_speed = 6.5
controller = NominalController(gains_path='config/fixedwing_controller_gains.yaml')
controller.trim_elev = 0.30
controller.phi_lim = np.radians(40.0)
guidance = PathFollower(params={'v_cruise_nominal': 6.5})
wp = model.waypoints
dt = 0.04
NOISE_SIGMA = np.array([0.2, 0.1, 0.4, 0.2])

x = model.trim_state()
x[0], x[1], x[2] = wp[0, 0], wp[0, 1], wp[0, 2]
dx, dy = wp[1, 0] - wp[0, 0], wp[1, 1] - wp[0, 1]
yaw0 = np.arctan2(dy, dx)
x[6], x[9] = np.cos(yaw0 / 2), np.sin(yaw0 / 2)

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

# 3D view
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.plot(cx, cy, np.full_like(cx, wp[0, 2]), 'k--', linewidth=0.5, alpha=0.3)
ax3d.plot(left_edge[:, 0], left_edge[:, 1], np.full(len(left_edge), wp[0, 2]),
          color='gray', linewidth=0.8, alpha=0.4)
ax3d.plot(right_edge[:, 0], right_edge[:, 1], np.full(len(right_edge), wp[0, 2]),
          color='gray', linewidth=0.8, alpha=0.4)
ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
ax3d.set_title('3D Flight Path')
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

# 2D panels
ax_top = fig.add_subplot(gs[0, 1])
track_poly = np.vstack([left_edge, right_edge[::-1]])
ax_top.fill(track_poly[:, 0], track_poly[:, 1], color='#e8e8e8', zorder=0)
ax_top.plot(left_edge[:, 0], left_edge[:, 1], 'k-', linewidth=1.2, alpha=0.6)
ax_top.plot(right_edge[:, 0], right_edge[:, 1], 'k-', linewidth=1.2, alpha=0.6)
ax_top.plot(cx, cy, 'k--', linewidth=0.5, alpha=0.3)
ax_top.set_aspect('equal')
ax_top.set_title('Top-Down')
trail_top, = ax_top.plot([], [], 'dodgerblue', linewidth=1.5)
ac_fuse_top, = ax_top.plot([], [], 'r-', linewidth=2, zorder=10)
ac_wing_top, = ax_top.plot([], [], 'r-', linewidth=2, zorder=10)
ac_htail_top, = ax_top.plot([], [], 'r-', linewidth=1.5, zorder=10)

ax_alt = fig.add_subplot(gs[0, 2])
ax_alt.axhline(3.0, color='green', linestyle='--', alpha=0.4)
ax_alt.set_title('Altitude'); ax_alt.set_ylabel('z (m)')
alt_line, = ax_alt.plot([], [], 'dodgerblue')

ax_spd = fig.add_subplot(gs[1, 1])
ax_spd.axhline(7.0, color='green', linestyle='--', alpha=0.4)
ax_spd.set_title('Airspeed'); ax_spd.set_ylabel('V (m/s)')
spd_line, = ax_spd.plot([], [], 'dodgerblue')

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
hx, hy, hz, hspd = [], [], [], []
cail, celev, cthr, crud = [], [], [], []
CAM_R = 12

print(f'Fixed-wing nominal | steps={args.steps}, noise={args.noise}')

for step in range(args.steps):
    vx_w, vy_w, vz_w = _quat_rotate(x[6], x[7], x[8], x[9], x[3], x[4], x[5])
    ref = guidance.compute_reference(x[0], x[1], x[2], vx_w, vy_w, vz_w, x[3], wp, dt)
    u = controller.compute(x, ref, dt)
    u_dyn = u.copy()
    u_dyn[1] = -u_dyn[1]
    if args.noise:
        u_dyn += NOISE_SIGMA * np.random.randn(4)
        u_dyn = np.clip(u_dyn, *model.control_bounds)
    x = model.step(x.reshape(1, -1), u_dyn.reshape(1, -1), dt, np).flatten()

    hx.append(x[0]); hy.append(x[1]); hz.append(x[2])
    hspd.append(np.linalg.norm(x[3:6]))
    cail.append(u[0]); celev.append(u[1]); cthr.append(u[2]); crud.append(u[3])

    # Update display
    if not args.headless and step % 3 == 0:
        n = len(hx); i0 = max(0, n - 150)
        t = np.arange(i0, n) * dt
        roll, _, yaw = _quat_to_euler(x[6], x[7], x[8], x[9])
        px, py, pz = x[0], x[1], x[2]

        # Aircraft wireframe rotated by quaternion
        ac_w = np.zeros((8, 3))
        for i in range(8):
            rx, ry, rz = _quat_rotate(x[6], x[7], x[8], x[9], ac_body[i, 0], ac_body[i, 1], ac_body[i, 2])
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

        # Top-down
        trail_top.set_data(hx[i0:], hy[i0:])
        ac_fuse_top.set_data(ac_w[0:2, 0], ac_w[0:2, 1])
        ac_wing_top.set_data(ac_w[2:4, 0], ac_w[2:4, 1])
        ac_htail_top.set_data(ac_w[4:6, 0], ac_w[4:6, 1])
        ax_top.set_xlim(px - CAM_R * 0.8, px + CAM_R * 0.8)
        ax_top.set_ylim(py - CAM_R * 0.8, py + CAM_R * 0.8)

        # Strips
        t_now = step * dt
        tlo, thi = max(0, t_now - 3), t_now + 0.5

        alt_line.set_data(t, hz[i0:])
        ax_alt.set_xlim(tlo, thi)
        ax_alt.set_ylim(max(0, min(hz[i0:]) - 0.5), max(hz[i0:]) + 0.5)

        spd_line.set_data(t, hspd[i0:])
        ax_spd.set_xlim(tlo, thi)
        ax_spd.set_ylim(max(0, min(hspd[i0:]) - 1), max(hspd[i0:]) + 1)

        ail_l.set_data(t, cail[i0:]); elev_l.set_data(t, celev[i0:])
        thr_l.set_data(t, cthr[i0:]); rud_l.set_data(t, crud[i0:])
        ax_ctrl.set_xlim(tlo, thi)

        fig.suptitle(f'Nominal  |  t={t_now:.1f}s  |  alt={pz:.1f}m  |  V={hspd[-1]:.1f}m/s  |  roll={np.degrees(roll):.0f}°')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    if step % 100 == 0:
        print(f'  step {step}: alt={x[2]:.1f} spd={np.linalg.norm(x[3:6]):.1f}')

# === Save PNG ===
os.makedirs('examples/results/discrete/fixed_wing', exist_ok=True)
suffix = '_noisy' if args.noise else ''
fig_save, ax_save = plt.subplots(figsize=(10, 8))
ax_save.plot(wp[:, 0], wp[:, 1], 'k--', linewidth=0.5)
ax_save.plot(hx, hy, 'dodgerblue', linewidth=2)
ax_save.set_aspect('equal')
ax_save.set_title('Fixed-wing nominal flight')
outpath = f'examples/results/discrete/fixed_wing/nominal{suffix}.png'
plt.savefig(outpath)
print(f'Saved to {outpath}')
plt.close(fig_save)

if not args.headless:
    plt.ioff()
    input("Press Enter to close...")
plt.close('all')
