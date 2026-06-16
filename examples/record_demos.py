"""Record demo videos for README.

Usage:
    python examples/record_demos.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

OUT_DIR = 'examples/results/videos'
os.makedirs(OUT_DIR, exist_ok=True)
FPS = 30
DPI = 120


def record_forest():
    from environments import Environment
    from mppi import MPPI
    from mppi.models import DoubleIntegrator
    from examples.plot_style import COLORS

    env = Environment.from_yaml('config/environments/forest.yaml')
    model = DoubleIntegrator(goal=env.goal_pos, obs=None)
    _orig = model.running_cost
    def _cost(x, u, t, xp):
        c = _orig(x, u, t, xp)
        c += xp.array(env.get_obstacle_cost(x), dtype=c.dtype)
        return c
    model.running_cost = _cost
    def _clamp(x, xp):
        env.clamp_to_obstacles(x)
        return x
    model.clamp_state = _clamp

    np.random.seed(42)
    solver = MPPI(model, K=1024, T=50, dt=0.05, lambda_=10.0, sigma=[1.5, 1.5])

    x = np.array([env.start_pos[0], env.start_pos[1], 0.0, 0.0], dtype=np.float32)
    N_SHOW = 30

    fig, ax = plt.subplots(figsize=(8, 8))
    xx, yy = np.meshgrid(np.linspace(-2, 12, 200), np.linspace(-2, 12, 200))
    C = env.cost_field(xx, yy)
    ax.contourf(xx, yy, np.clip(C, 0, 5000), levels=30, cmap='YlOrRd', alpha=0.5)
    for o in env.obstacles:
        if o['type'] == 'circle':
            ax.add_patch(mpatches.Circle(o['position'], o['radius'], ec='white', fc='none', lw=0.5, alpha=0.6))
    ax.plot(*env.start_pos, 'o', color='green', markersize=10, zorder=10)
    ax.plot(*env.goal_pos, '*', color='gold', markersize=14, zorder=10)
    ax.set_xlim(-2, 12); ax.set_ylim(-2, 12); ax.set_aspect('equal')
    ax.set_title('MPPI Forest Navigation')

    trail, = ax.plot([], [], color=COLORS['primary'], lw=2)
    dot, = ax.plot([], [], 'ko', ms=8, zorder=10)
    sample_lines = [ax.plot([], [], 'c-', lw=0.3, alpha=0.2)[0] for _ in range(N_SHOW)]

    hx, hy = [], []
    state = {'x': x}

    def update(frame):
        U = solver.solve(state['x'])
        u = U[0]
        state['x'] = model.step(state['x'][None, :], u[None, :], 0.05, np).squeeze(0)
        hx.append(float(state['x'][0])); hy.append(float(state['x'][1]))
        trail.set_data(hx, hy)
        dot.set_data([hx[-1]], [hy[-1]])
        samples = solver.get_sampled_trajectories()
        weights = solver.get_weights()
        top = np.argsort(weights)[-N_SHOW:]
        for j, idx in enumerate(top):
            sample_lines[j].set_data(samples[idx, :, 0], samples[idx, :, 1])
        dist = np.linalg.norm(state['x'][:2] - env.goal_pos)
        ax.set_title(f'MPPI Forest  |  step {frame}  |  dist={dist:.1f}m')
        return [trail, dot] + sample_lines

    anim = FuncAnimation(fig, update, frames=200, interval=33, blit=False)
    path = os.path.join(OUT_DIR, 'forest_navigation.mp4')
    anim.save(path, fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f'Saved {path}')


def record_u_trap():
    from environments import Environment
    from mppi import MPPI
    from mppi.models import Unicycle

    env = Environment.from_yaml('config/environments/u_trap.yaml')
    model = Unicycle(goal=list(env.goal_pos), obs=None)
    _orig = model.running_cost
    def _cost(x, u, t, xp):
        c = _orig(x, u, t, xp)
        c += xp.array(env.get_obstacle_cost(x), dtype=c.dtype)
        return c
    model.running_cost = _cost
    def _clamp(x, xp):
        env.clamp_to_obstacles(x)
        return x
    model.clamp_state = _clamp

    np.random.seed(42)
    solver = MPPI(model, K=2048, T=400, dt=0.025, lambda_=100.0, sigma=[2.0, 4.0])

    theta0 = np.arctan2(env.goal_pos[1] - env.start_pos[1], env.goal_pos[0] - env.start_pos[0])
    x = np.array([env.start_pos[0], env.start_pos[1], theta0], dtype=np.float32)
    N_SHOW = 30

    fig, ax = plt.subplots(figsize=(8, 8))
    xx, yy = np.meshgrid(np.linspace(-2, 12, 200), np.linspace(-2, 12, 200))
    C = env.cost_field(xx, yy)
    ax.contourf(xx, yy, np.clip(C, 0, 5000), levels=30, cmap='YlOrRd', alpha=0.5)
    ax.plot(*env.start_pos, 'o', color='green', markersize=10, zorder=10)
    ax.plot(*env.goal_pos, '*', color='gold', markersize=14, zorder=10)
    ax.set_xlim(-2, 12); ax.set_ylim(-1, 11); ax.set_aspect('equal')
    ax.set_title('Unicycle SDE — U-Trap')

    trail, = ax.plot([], [], color='dodgerblue', lw=2)
    dot, = ax.plot([], [], 'ko', ms=8, zorder=10)
    arrow = [None]
    sample_lines = [ax.plot([], [], 'c-', lw=0.3, alpha=0.2)[0] for _ in range(N_SHOW)]

    hx, hy = [], []
    state = {'x': x}

    def update(frame):
        U = solver.solve_continuous(state['x'])
        u = U[0]
        state['x'] = model.step(state['x'][None, :], u[None, :], 0.025, np).squeeze(0)
        hx.append(float(state['x'][0])); hy.append(float(state['x'][1]))
        trail.set_data(hx, hy)
        dot.set_data([hx[-1]], [hy[-1]])
        if arrow[0]:
            arrow[0].remove()
        dx = 0.5 * np.cos(float(state['x'][2]))
        dy = 0.5 * np.sin(float(state['x'][2]))
        arrow[0] = ax.annotate('', xy=(hx[-1]+dx, hy[-1]+dy), xytext=(hx[-1], hy[-1]),
                                arrowprops=dict(arrowstyle='->', color='black', lw=2))
        samples = solver.get_sampled_trajectories()
        weights = solver.get_weights()
        top = np.argsort(weights)[-N_SHOW:]
        for j, idx in enumerate(top):
            sample_lines[j].set_data(samples[idx, :, 0], samples[idx, :, 1])
        dist = np.linalg.norm(state['x'][:2] - env.goal_pos)
        ax.set_title(f'Unicycle SDE — U-Trap  |  step {frame}  |  dist={dist:.1f}m')
        return [trail, dot] + sample_lines

    anim = FuncAnimation(fig, update, frames=600, interval=33, blit=False)
    path = os.path.join(OUT_DIR, 'u_trap_unicycle.mp4')
    anim.save(path, fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f'Saved {path}')


def record_cartpole():
    from mppi import MPPI
    from mppi.models import CartPole

    POLE_L = 1.0
    CART_W, CART_H = 0.4, 0.2

    np.random.seed(42)
    model = CartPole()
    solver = MPPI(model, K=1024, T=50, dt=0.02, lambda_=10.0, sigma=[3.0])

    x = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)

    fig, (ax_cart, ax_phase) = plt.subplots(1, 2, figsize=(14, 5))
    ax_cart.set_xlim(-3, 3); ax_cart.set_ylim(-1.5, 1.5)
    ax_cart.set_aspect('equal'); ax_cart.grid(True, alpha=0.3)
    ax_cart.axhline(0, color='k', lw=0.5)
    ax_cart.set_title('Cart-Pole Swing-Up')

    cart = plt.Rectangle((-CART_W/2, -CART_H/2), CART_W, CART_H, fc='steelblue', ec='black', zorder=5)
    ax_cart.add_patch(cart)
    pole, = ax_cart.plot([], [], 'o-', color='firebrick', lw=3, ms=8, markevery=[1], zorder=6)

    ax_phase.set_xlim(-np.pi, np.pi); ax_phase.set_ylim(-10, 10)
    ax_phase.set_xlabel(r'$\theta$'); ax_phase.set_ylabel(r'$\dot{\theta}$')
    ax_phase.set_title('Phase Portrait')
    ax_phase.axvline(0, color='gray', ls='--', alpha=0.5)
    ax_phase.axhline(0, color='gray', ls='--', alpha=0.5)
    ax_phase.grid(True, alpha=0.3)
    phase_line, = ax_phase.plot([], [], 'b-', lw=0.8, alpha=0.7)
    phase_dot, = ax_phase.plot([], [], 'ro', ms=6, zorder=10)

    h_theta, h_thetadot = [], []
    state = {'x': x}

    def update(frame):
        U = solver.solve(state['x'])
        u = U[0]
        state['x'] = model.step(state['x'][None, :], u[None, :], 0.02, np).squeeze(0)
        cx = float(state['x'][0])
        theta = float(state['x'][2])
        tw = np.arctan2(np.sin(theta), np.cos(theta))
        h_theta.append(tw); h_thetadot.append(float(state['x'][3]))

        cart.set_xy((cx - CART_W/2, -CART_H/2))
        pole.set_data([cx, cx + POLE_L*np.sin(theta)], [0, POLE_L*np.cos(theta)])
        phase_line.set_data(h_theta, h_thetadot)
        phase_dot.set_data([h_theta[-1]], [h_thetadot[-1]])
        fig.suptitle(f'step {frame}  |  x={cx:.2f}  |  θ={np.degrees(tw):.1f}°')
        return [cart, pole, phase_line, phase_dot]

    anim = FuncAnimation(fig, update, frames=300, interval=33, blit=False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'cartpole_swingup.mp4')
    anim.save(path, fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f'Saved {path}')


def record_fixed_wing():
    from mppi import MPPI
    from mppi.models.fixed_wing import FixedWing
    from mppi.models.fixed_wing.controller import NominalController, _quat_to_euler, _quat_rotate
    from mppi.models.fixed_wing.path_follower import PathFollower

    try:
        from mppi.models.fixed_wing.dynamics_jit import FixedWingJIT
        ModelClass = FixedWingJIT
    except ImportError:
        from mppi.models.fixed_wing import FixedWing
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

    np.random.seed(42)
    solver_dtype = np.float64
    solver = MPPI(model, K=256, T=64, dt=dt, lambda_=10000.0,
                  sigma=[0.4, 0.3, 0.5, 0.3], warm_start=True, dtype=solver_dtype)

    x0 = model.trim_state()
    x0[0], x0[1], x0[2] = wp[0, 0], wp[0, 1], wp[0, 2]
    dx, dy = wp[1, 0] - wp[0, 0], wp[1, 1] - wp[0, 1]
    yaw0 = np.arctan2(dy, dx)
    x0[6], x0[9] = np.cos(yaw0 / 2), np.sin(yaw0 / 2)

    xx = x0.copy()
    U_nom = np.zeros((64, 4))
    for t in range(64):
        vw = _quat_rotate(xx[6], xx[7], xx[8], xx[9], xx[3], xx[4], xx[5])
        ref = guidance.compute_reference(xx[0], xx[1], xx[2], vw[0], vw[1], vw[2], xx[3], wp, dt)
        u = controller.compute(xx, ref, dt)
        u[1] = -u[1]
        U_nom[t] = u
        xx = model_exec.step(xx.reshape(1, -1), u.reshape(1, -1), dt, np).flatten()
    solver.U = np.array(U_nom, dtype=solver_dtype)

    x = np.array(x0, dtype=solver_dtype)
    N_SHOW = 40
    CAM_R = 12
    AC_S = 1.5
    ac_body = np.array([
        [AC_S, 0, 0], [-AC_S, 0, 0],
        [0.15*AC_S, -AC_S, 0], [0.15*AC_S, AC_S, 0],
        [-0.85*AC_S, -0.4*AC_S, 0], [-0.85*AC_S, 0.4*AC_S, 0],
        [-0.85*AC_S, 0, 0], [-0.85*AC_S, 0, 0.35*AC_S],
    ])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.05)
    from scipy.interpolate import splprep, splev
    from shapely.geometry import LineString
    hw = model.track_half_width
    wp_closed = np.vstack([wp[:, :2], wp[:1, :2]])
    tck_t, _ = splprep([wp_closed[:, 0], wp_closed[:, 1]], s=0, per=True)
    u_fine = np.linspace(0, 1, 300)
    tcx, tcy = splev(u_fine, tck_t)
    cl = LineString(np.column_stack([tcx, tcy]))
    le = np.array(cl.offset_curve(hw).coords)
    re = np.array(cl.offset_curve(-hw).coords)

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.plot(tcx, tcy, np.full_like(tcx, wp[0, 2]), 'k--', lw=0.5, alpha=0.3)
    ax3d.plot(le[:, 0], le[:, 1], np.full(len(le), wp[0, 2]), color='gray', lw=0.8, alpha=0.4)
    ax3d.plot(re[:, 0], re[:, 1], np.full(len(re), wp[0, 2]), color='gray', lw=0.8, alpha=0.4)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title('3D Flight + Rollouts')
    trail_3d, = ax3d.plot([], [], [], 'dodgerblue', lw=2)
    ac_fuse, = ax3d.plot([], [], [], 'r-', lw=2.5, zorder=10)
    ac_wing, = ax3d.plot([], [], [], 'r-', lw=2.5, zorder=10)
    ac_htail, = ax3d.plot([], [], [], 'r-', lw=2, zorder=10)
    ac_vtail, = ax3d.plot([], [], [], 'r-', lw=2, zorder=10)
    sample_lines = [ax3d.plot([], [], [], 'c-', lw=0.4, alpha=0.3)[0] for _ in range(N_SHOW)]

    ax_top = fig.add_subplot(gs[0, 1])
    track_poly = np.vstack([le, re[::-1]])
    ax_top.fill(track_poly[:, 0], track_poly[:, 1], color='#e8e8e8', zorder=0)
    ax_top.plot(le[:, 0], le[:, 1], 'k-', lw=1.2, alpha=0.6)
    ax_top.plot(re[:, 0], re[:, 1], 'k-', lw=1.2, alpha=0.6)
    ax_top.plot(tcx, tcy, 'k--', lw=0.5, alpha=0.3)
    ax_top.set_aspect('equal')
    ax_top.set_title('Top-Down + Rollouts')
    trail_top, = ax_top.plot([], [], 'dodgerblue', lw=1.5)
    sample_top = [ax_top.plot([], [], 'c-', lw=0.4, alpha=0.3)[0] for _ in range(N_SHOW)]
    ac_fuse_top, = ax_top.plot([], [], 'r-', lw=2, zorder=10)
    ac_wing_top, = ax_top.plot([], [], 'r-', lw=2, zorder=10)
    ac_htail_top, = ax_top.plot([], [], 'r-', lw=1.5, zorder=10)

    plt.tight_layout()
    hx, hy, hz = [], [], []
    state = {'x': x}

    def update(frame):
        U_opt = solver.solve(state['x'])
        u = U_opt[0]
        state['x'] = model_exec.step(state['x'].reshape(1,-1), u.reshape(1,-1), dt, np).flatten()
        state['x'] = np.array(state['x'], dtype=solver_dtype)
        px, py, pz = state['x'][0], state['x'][1], state['x'][2]
        hx.append(px); hy.append(py); hz.append(pz)
        roll, _, yaw = _quat_to_euler(state['x'][6], state['x'][7], state['x'][8], state['x'][9])

        ac_w = np.zeros((8, 3))
        for i in range(8):
            rx, ry, rz = _quat_rotate(state['x'][6], state['x'][7], state['x'][8], state['x'][9],
                                       ac_body[i,0], ac_body[i,1], ac_body[i,2])
            ac_w[i] = [px+rx, py+ry, pz+rz]

        i0 = max(0, len(hx)-150)
        trail_3d.set_data(hx[i0:], hy[i0:]); trail_3d.set_3d_properties(hz[i0:])
        ac_fuse.set_data(ac_w[0:2,0], ac_w[0:2,1]); ac_fuse.set_3d_properties(ac_w[0:2,2])
        ac_wing.set_data(ac_w[2:4,0], ac_w[2:4,1]); ac_wing.set_3d_properties(ac_w[2:4,2])
        ac_htail.set_data(ac_w[4:6,0], ac_w[4:6,1]); ac_htail.set_3d_properties(ac_w[4:6,2])
        ac_vtail.set_data(ac_w[6:8,0], ac_w[6:8,1]); ac_vtail.set_3d_properties(ac_w[6:8,2])
        ax3d.view_init(elev=35, azim=np.degrees(yaw)+210)
        ax3d.set_xlim(px-CAM_R, px+CAM_R); ax3d.set_ylim(py-CAM_R, py+CAM_R)
        ax3d.set_zlim(max(0, pz-4), pz+4)

        samples = solver.get_sampled_trajectories()
        weights = solver.get_weights()
        top_idx = np.argsort(weights)[-N_SHOW:]
        for j, idx in enumerate(top_idx):
            s = samples[idx]
            sample_lines[j].set_data(s[:,0], s[:,1]); sample_lines[j].set_3d_properties(s[:,2])
            sample_top[j].set_data(s[:,0], s[:,1])

        trail_top.set_data(hx[i0:], hy[i0:])
        ac_fuse_top.set_data(ac_w[0:2,0], ac_w[0:2,1])
        ac_wing_top.set_data(ac_w[2:4,0], ac_w[2:4,1])
        ac_htail_top.set_data(ac_w[4:6,0], ac_w[4:6,1])
        fr = CAM_R * 0.8
        ax_top.set_xlim(px-fr, px+fr); ax_top.set_ylim(py-fr, py+fr)

        spd = np.linalg.norm(state['x'][3:6])
        fig.suptitle(f'Fixed-Wing MPPI  |  t={frame*dt:.1f}s  |  alt={pz:.1f}m  |  V={spd:.1f}m/s')
        return []

    anim = FuncAnimation(fig, update, frames=500, interval=33, blit=False)
    path = os.path.join(OUT_DIR, 'fixed_wing_mppi.mp4')
    anim.save(path, fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f'Saved {path}')


def record_gridworld():
    from environments import Environment
    from lmdp.backward import build_M_matrix, backward_recursion, reconstruct_policy, value_from_desirability

    ALPHA = 10.0
    T = 40
    GRID_SIZE = 15
    N_TRAJ = 30
    rng = np.random.default_rng(42)

    env = Environment.from_yaml('config/environments/three_mountains.yaml')
    grid, start = env.to_gridworld(GRID_SIZE, GRID_SIZE, ALPHA)

    Z_T = np.array([np.exp(-grid.terminal_cost(x) / ALPHA) for x in range(grid.n_states)])
    M = build_M_matrix(grid, ALPHA)
    Z = backward_recursion(Z_T, M, T)
    V = value_from_desirability(Z, ALPHA)
    policy = reconstruct_policy(grid, Z, ALPHA)

    bounds = grid._env_bounds
    cell_w, cell_h = grid._cell_w, grid._cell_h

    def s2xy(s):
        r, c = grid.state_to_rc(s)
        return bounds[0] + (c + 0.5) * cell_w, bounds[3] - (r + 0.5) * cell_h

    # Pre-roll all trajectories
    trajectories = []
    for _ in range(N_TRAJ):
        traj = [start]
        x = start
        for k in range(T):
            if k >= policy.shape[0]:
                break
            probs = policy[k, x]
            if probs.sum() < 1e-10:
                break
            a = rng.choice(grid.n_actions, p=probs)
            x = grid.step(x, a)
            traj.append(x)
            if grid.state_to_rc(x) == grid.goal:
                break
        trajectories.append(traj)

    max_len = max(len(t) for t in trajectories)

    # Cell costs for flash intensity
    cell_costs = grid._cell_costs
    cost_max = cell_costs[cell_costs < 1e6].max()

    # --- Static image ---
    fig_static, ax_s = plt.subplots(figsize=(8, 8))
    xx, yy = np.meshgrid(np.linspace(bounds[0], bounds[1], 200),
                         np.linspace(bounds[2], bounds[3], 200))
    C = env.cost_field(xx, yy)
    ax_s.contourf(xx, yy, C, levels=30, cmap='YlOrRd', alpha=0.5)
    for r in range(grid.n_rows + 1):
        ax_s.axhline(bounds[3] - r * cell_h, color='gray', lw=0.3, alpha=0.3)
    for c in range(grid.n_cols + 1):
        ax_s.axvline(bounds[0] + c * cell_w, color='gray', lw=0.3, alpha=0.3)
    sx, sy = s2xy(start)
    gx, gy = s2xy(grid.rc_to_state(*grid.goal))
    ax_s.plot(sx, sy, 'go', ms=10, zorder=10)
    ax_s.plot(gx, gy, 'r*', ms=14, zorder=10)
    for traj in trajectories:
        xs, ys = zip(*[s2xy(s) for s in traj])
        reached = grid.state_to_rc(traj[-1]) == grid.goal
        ax_s.plot(xs, ys, '-', lw=0.8, alpha=0.5,
                  color='dodgerblue' if reached else 'gray')
    ax_s.set_xlim(bounds[0], bounds[1])
    ax_s.set_ylim(bounds[2], bounds[3])
    ax_s.set_aspect('equal')
    ax_s.set_title(f'Backward KL Control — {sum(1 for t in trajectories if grid.state_to_rc(t[-1]) == grid.goal)}/{N_TRAJ} reach goal')
    ax_s.set_xlabel('$x$'); ax_s.set_ylabel('$y$')
    img_dir = os.path.join(OUT_DIR, '..', 'images')
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, 'gridworld_backward.png')
    fig_static.savefig(img_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig_static)
    print(f'  Saved {img_path}')

    # --- Video ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(xx, yy, C, levels=30, cmap='YlOrRd', alpha=0.5)
    for r in range(grid.n_rows + 1):
        ax.axhline(bounds[3] - r * cell_h, color='gray', lw=0.3, alpha=0.3)
    for c in range(grid.n_cols + 1):
        ax.axvline(bounds[0] + c * cell_w, color='gray', lw=0.3, alpha=0.3)
    ax.plot(sx, sy, 'go', ms=10, zorder=10)
    ax.plot(gx, gy, 'r*', ms=14, zorder=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')

    trail_lines = [ax.plot([], [], '-', lw=1.0, alpha=0.6, color='dodgerblue')[0]
                   for _ in range(N_TRAJ)]
    dots = [ax.plot([], [], 'o', ms=5, color='dodgerblue', zorder=8)[0]
            for _ in range(N_TRAJ)]
    flash_patches = []

    finished = [False] * N_TRAJ
    reached_count = [0]

    def update(frame):
        # Remove previous flash patches
        for p in flash_patches:
            p.remove()
        flash_patches.clear()

        for i, traj in enumerate(trajectories):
            step = frame
            if step >= len(traj):
                step = len(traj) - 1
                if not finished[i]:
                    finished[i] = True
                    if grid.state_to_rc(traj[-1]) == grid.goal:
                        reached_count[0] += 1
                        trail_lines[i].set_color('dodgerblue')
                    else:
                        trail_lines[i].set_color('gray')
                        trail_lines[i].set_alpha(0.3)

            xs, ys = zip(*[s2xy(s) for s in traj[:step+1]])
            trail_lines[i].set_data(xs, ys)

            cx, cy = s2xy(traj[step])
            if not finished[i]:
                dots[i].set_data([cx], [cy])
            else:
                dots[i].set_data([], [])

            # Flash red on high-cost cells
            if not finished[i] and step < len(traj):
                r_idx, c_idx = grid.state_to_rc(traj[step])
                cost_here = cell_costs[r_idx, c_idx]
                if cost_here > cost_max * 0.15:
                    intensity = min(cost_here / cost_max, 1.0)
                    rx = bounds[0] + c_idx * cell_w
                    ry = bounds[3] - (r_idx + 1) * cell_h
                    p = ax.add_patch(plt.Rectangle(
                        (rx, ry), cell_w, cell_h,
                        color='red', alpha=0.4 * intensity, zorder=6))
                    flash_patches.append(p)

        ax.set_title(f'Backward KL Control  |  step {frame}  |  '
                     f'{reached_count[0]}/{N_TRAJ} reached goal')
        return trail_lines + dots + flash_patches

    anim = FuncAnimation(fig, update, frames=max_len + 5, interval=150, blit=False)
    vid_path = os.path.join(OUT_DIR, 'gridworld_backward.mp4')
    anim.save(vid_path, fps=10, dpi=DPI)
    plt.close(fig)
    print(f'  Saved {vid_path}')


if __name__ == '__main__':
    print('Recording gridworld backward...')
    record_gridworld()
    print('Recording forest navigation...')
    record_forest()
    print('Recording u-trap unicycle...')
    record_u_trap()
    print('Recording cartpole swing-up...')
    record_cartpole()
    print('Recording fixed-wing MPPI...')
    record_fixed_wing()
    print('Done!')
