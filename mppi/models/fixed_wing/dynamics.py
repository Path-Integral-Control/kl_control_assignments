"""Fixed-wing 6DOF dynamics for MPPI.

6DOF rigid body with aerodynamic forces, stall model,
and quaternion attitude representation.

Aerodynamic model derived from CyECCA [1].

References:
  [1] CogniPilot CyECCA — https://github.com/CogniPilot/cyecca

State:  [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r]  (13D)
  - position in world frame (ENU)
  - velocity in body frame (FLU)
  - quaternion (scalar-first, body-to-world)
  - angular rates in body frame

Control: [aileron, elevator, throttle, rudder]  (4D)
  - aileron:  [-1, 1]  -> ±30 deg
  - elevator: [-1, 1]  -> ±24 deg
  - throttle: [0.2, 1] -> 0 to thr_max N
  - rudder:   [-0.8, 0.8] -> ±20 deg (sign flipped per HH Sport Cub mapping)

Integration:
  step()            — RK4 with substeps, quaternion normalization,
                      state/derivative clamping after each substep.
  simulate_single() — scipy Radau (adaptive implicit) with Baumgarte
                      quaternion constraint stabilization.
"""

import numpy as np
import yaml
from mppi.models.base import DynamicsModel


# === Parameter indices (35-element array) ===
P_THR_MAX = 0
P_M = 1
P_XCG = 2
P_XAC = 3
P_S = 4
P_RHO = 5
P_G = 6
P_JX = 7
P_JY = 8
P_JZ = 9
P_JXZ = 10
P_CBAR = 11
P_SPAN = 12
P_CM0 = 13
P_CLDA = 14
P_CLDR = 15
P_CMDE = 16
P_CNDR = 17
P_CNDA = 18
P_CYDA = 19
P_CYDR = 20
P_CL0 = 21
P_CLA = 22
P_CMA = 23
P_CMQ = 24
P_CD0 = 25
P_CDCLS = 26
P_CNB = 27
P_CLP = 28
P_CNR = 29
P_CNP = 30
P_CLR = 31
P_CYB = 32
P_CYR = 33
P_CYP = 34

PARAM_NAMES = [
    'thr_max', 'm', 'XCG', 'XAC', 'S', 'rho', 'g',
    'Jx', 'Jy', 'Jz', 'Jxz', 'cbar', 'span',
    'Cm0', 'Clda', 'Cldr', 'Cmde', 'Cndr', 'Cnda', 'CYda', 'CYdr',
    'CL0', 'CLa', 'Cma', 'Cmq', 'CD0', 'CDCLS',
    'Cnb', 'Clp', 'Cnr', 'Cnp', 'Clr', 'CYb', 'CYr', 'CYp',
]

# Default aircraft parameters
DEFAULT_PARAMS = {
    'thr_max': 0.35, 'm': 0.057, 'XCG': 0.25, 'XAC': 0.25,
    'S': 0.05553, 'rho': 1.225, 'g': 9.81,
    'Jx': 1.5e-4, 'Jy': 2.0e-4, 'Jz': 2.2e-4, 'Jxz': 0.0,
    'cbar': 0.09, 'span': 0.617,
    'Cm0': 0.102, 'Clda': 0.24, 'Cldr': 0.05, 'Cmde': 0.9,
    'Cndr': 0.22, 'Cnda': 0.03, 'CYda': 0.02, 'CYdr': -0.18,
    'CL0': 0.60, 'CLa': 8.0, 'Cma': 1.5, 'Cmq': -12.0,
    'CD0': 0.14, 'CDCLS': 0.055,
    'Cnb': 0.10, 'Clp': -1.30, 'Cnr': -0.12, 'Cnp': -0.10,
    'Clr': 0.10, 'CYb': -0.85, 'CYr': 0.25, 'CYp': 0.15,
}

DEG2RAD = np.pi / 180.0
MAX_DEFL_AIL = 30.0 * DEG2RAD
MAX_DEFL_ELEV = 24.0 * DEG2RAD
MAX_DEFL_RUD = 20.0 * DEG2RAD
ALPHA_STALL = 20.0 * DEG2RAD
TOL_V = 1e-3

# Clamping limits
MAX_VELOCITY = 15.0
MIN_VELOCITY = 1.0
MAX_LATERAL_VEL = 0.5
MAX_OMEGA = 1.57
MAX_ACCEL = 20.0
MAX_ANG_ACCEL = 10.0
MAX_ALPHA = 0.35
MAX_BETA = 0.0873
N_SUBSTEPS = 50


def _params_to_array(params_dict):
    arr = np.zeros(35)
    for i, name in enumerate(PARAM_NAMES):
        arr[i] = params_dict.get(name, DEFAULT_PARAMS[name])
    return arr


def _quat_rotate(qw, qx, qy, qz, vx, vy, vz, xp):
    rx = (qw*qw + qx*qx - qy*qy - qz*qz)*vx + 2*(qx*qy - qw*qz)*vy + 2*(qx*qz + qw*qy)*vz
    ry = 2*(qx*qy + qw*qz)*vx + (qw*qw - qx*qx + qy*qy - qz*qz)*vy + 2*(qy*qz - qw*qx)*vz
    rz = 2*(qx*qz - qw*qy)*vx + 2*(qy*qz + qw*qx)*vy + (qw*qw - qx*qx - qy*qy + qz*qz)*vz
    return rx, ry, rz


def _quat_rotate_inv(qw, qx, qy, qz, vx, vy, vz, xp):
    return _quat_rotate(qw, -qx, -qy, -qz, vx, vy, vz, xp)


class FixedWing(DynamicsModel):

    def __init__(self, waypoints=None, params=None, params_path=None, n_substeps=N_SUBSTEPS):
        if params_path is not None:
            with open(params_path, 'r') as f:
                raw = yaml.safe_load(f)
            params_dict = {k: v for k, v in raw.items() if k in PARAM_NAMES}
            self.params = _params_to_array(params_dict)
        elif params is not None:
            self.params = _params_to_array(params)
        else:
            self.params = _params_to_array(DEFAULT_PARAMS)

        if waypoints is None:
            pts = []
            # Top straight
            for x in np.linspace(-15, 10, 10, endpoint=False):
                pts.append([x, 8.0, 3.0])
            # Right chicane (S-curve with altitude bump)
            for s in np.linspace(0, np.pi, 20, endpoint=False):
                pts.append([10 + 3*np.sin(2*s), 8 - 16*(s/np.pi), 3.0 + 0.5*np.sin(s)])
            # Bezier curve: tangent to chicane exit, curves smoothly to bottom-left
            s_last = np.pi * 19/20
            ex = 10 + 3*np.sin(2*s_last)
            ey = 8 - 16*(s_last/np.pi)
            dx_ds = 6*np.cos(2*s_last)
            dy_ds = -16/np.pi
            P0 = np.array([ex, ey])
            P1 = P0 + 8.0 * np.array([dx_ds, dy_ds]) / np.sqrt(dx_ds**2 + dy_ds**2)
            P3 = np.array([-15.0, -8.0])
            P2 = np.array([-5.0, -10.0])
            for s in np.linspace(0, 1, 20, endpoint=True)[1:-1]:
                b = (1-s)**3*P0 + 3*(1-s)**2*s*P1 + 3*(1-s)*s**2*P2 + s**3*P3
                pts.append([b[0], b[1], 3.0])
            # Left semicircle turn
            for s in np.linspace(-np.pi/2, np.pi/2, 10, endpoint=False):
                pts.append([-15 - 8*np.cos(s), 8*np.sin(s), 3.0])
            self.waypoints = np.array(pts)
        else:
            self.waypoints = np.array(waypoints)

        self.n_substeps = n_substeps
        self.target_speed = 5.0
        self.track_half_width = 2.5
        self.w_lateral = 15000.0
        self.w_altitude = 2000.0
        self.w_progress = 5000.0
        self.w_velocity = 10.0
        self.w_angular_rate = 100.0
        self.w_stall = 10000.0
        self.w_boundary = 500000.0
        self.w_surface = 50.0
        self.w_smoothness = 200.0

    @property
    def state_dim(self):
        return 13

    @property
    def control_dim(self):
        return 4

    @property
    def control_bounds(self):
        return ([-1.0, -1.0, 0.2, -0.8], [1.0, 1.0, 1.0, 0.8])

    @property
    def noise_dim(self):
        return 0

    def dynamics(self, x, u, xp):
        K = x.shape[0]
        p = self.params

        vx_b, vy_b, vz_b = x[:, 3], x[:, 4], x[:, 5]
        qw, qx_q, qy_q, qz_q = x[:, 6], x[:, 7], x[:, 8], x[:, 9]
        P, Q, R = x[:, 10], x[:, 11], x[:, 12]

        V = xp.sqrt(vx_b**2 + vy_b**2 + vz_b**2)
        V = xp.clip(V, TOL_V, None)

        alpha = xp.arctan2(-vz_b, vx_b)
        beta = xp.arcsin(xp.clip(vy_b / V, -1.0, 1.0))

        ail_rad = MAX_DEFL_AIL * u[:, 0]
        elev_rad = MAX_DEFL_ELEV * u[:, 1]
        rud_rad = -MAX_DEFL_RUD * u[:, 3]

        qbar = 0.5 * p[P_RHO] * V**2

        CL = p[P_CL0] + p[P_CLA] * alpha
        CL = xp.where(xp.abs(alpha) >= ALPHA_STALL, 0.0, CL)
        CD = p[P_CD0] + p[P_CDCLS] * CL**2

        CC = (-p[P_CYB] * beta
              + p[P_CYDA] * ail_rad / MAX_DEFL_AIL
              + p[P_CYDR] * rud_rad / MAX_DEFL_RUD
              + p[P_CYP] * p[P_SPAN] / (2 * V) * P
              + p[P_CYR] * p[P_SPAN] / (2 * V) * R)

        Cl = p[P_CLDA] * ail_rad - p[P_CLDR] * rud_rad
        Cm = p[P_CM0] + p[P_CMA] * alpha + p[P_CMDE] * elev_rad + (p[P_XAC] - p[P_XCG]) * CL
        Cn = p[P_CNB] * beta + p[P_CNDR] * rud_rad - p[P_CNDA] * ail_rad

        D_mag = CD * qbar * p[P_S]
        L_mag = CL * qbar * p[P_S]
        Fs_mag = CC * qbar * p[P_S]

        v_hat_x, v_hat_y, v_hat_z = vx_b / V, vy_b / V, vz_b / V
        Dx_b, Dy_b, Dz_b = -D_mag * v_hat_x, -D_mag * v_hat_y, -D_mag * v_hat_z

        cos_a, sin_a = xp.cos(alpha), xp.sin(alpha)
        cos_b, sin_b = xp.cos(beta), xp.sin(beta)

        # Wind-to-body rotation R_bn = Rz(beta) * Ry(-alpha)
        # Wind-to-body rotation for lift and sideforce
        # R_bn column 2 (z): [-cos_b*sin_a, -sin_b*sin_a, cos_a]
        # R_bn column 1 (y): [-sin_b, cos_b, 0]
        Lx_b = -L_mag * cos_b * sin_a
        Ly_b = -L_mag * sin_b * sin_a
        Lz_b = L_mag * cos_a

        Sx_b = -Fs_mag * sin_b
        Sy_b = Fs_mag * cos_b
        Sz_b = xp.zeros_like(Fs_mag)

        FAx = Dx_b + Lx_b + Sx_b
        FAy = Dy_b + Ly_b + Sy_b
        FAz = Dz_b + Lz_b + Sz_b

        throttle = xp.clip(u[:, 2], 0.0, 1.0)
        FTx = p[P_THR_MAX] * throttle

        gx_b, gy_b, gz_b = _quat_rotate_inv(
            qw, qx_q, qy_q, qz_q,
            xp.zeros(K, dtype=x.dtype), xp.zeros(K, dtype=x.dtype),
            -p[P_M] * p[P_G] * xp.ones(K, dtype=x.dtype), xp)

        Fx = FAx + FTx + gx_b
        Fy = FAy + gy_b
        Fz = FAz + gz_b

        Mx = qbar * p[P_S] * p[P_SPAN] * Cl
        My = qbar * p[P_S] * p[P_CBAR] * Cm
        Mz = qbar * p[P_S] * p[P_SPAN] * Cn

        Mx += p[P_CLP] * p[P_SPAN] / (2 * V) * P + p[P_CLR] * p[P_SPAN] / (2 * V) * R
        My += p[P_CMQ] * p[P_CBAR] / (2 * V) * Q
        Mz += p[P_CNP] * p[P_SPAN] / (2 * V) * P + p[P_CNR] * p[P_SPAN] / (2 * V) * R

        xdot = xp.zeros_like(x)

        px_d, py_d, pz_d = _quat_rotate(qw, qx_q, qy_q, qz_q, vx_b, vy_b, vz_b, xp)
        xdot[:, 0], xdot[:, 1], xdot[:, 2] = px_d, py_d, pz_d

        xdot[:, 3] = Fx / p[P_M] - (Q * vz_b - R * vy_b)
        xdot[:, 4] = Fy / p[P_M] - (R * vx_b - P * vz_b)
        xdot[:, 5] = Fz / p[P_M] - (P * vy_b - Q * vx_b)

        xdot[:, 6] = 0.5 * (-qx_q * P - qy_q * Q - qz_q * R)
        xdot[:, 7] = 0.5 * (qw * P + qy_q * R - qz_q * Q)
        xdot[:, 8] = 0.5 * (qw * Q + qz_q * P - qx_q * R)
        xdot[:, 9] = 0.5 * (qw * R + qx_q * Q - qy_q * P)

        Jx, Jy, Jz, Jxz_val = p[P_JX], p[P_JY], p[P_JZ], p[P_JXZ]
        Jw_x = Jx * P + Jxz_val * R
        Jw_y = Jy * Q
        Jw_z = Jz * R + Jxz_val * P
        cross_x = Q * Jw_z - R * Jw_y
        cross_y = R * Jw_x - P * Jw_z
        cross_z = P * Jw_y - Q * Jw_x
        tau_x, tau_y, tau_z = Mx - cross_x, My - cross_y, Mz - cross_z
        det = Jx * Jz - Jxz_val**2
        det = max(det, 1e-20)
        xdot[:, 10] = (Jz * tau_x + Jxz_val * tau_z) / det
        xdot[:, 11] = tau_y / Jy
        xdot[:, 12] = (Jx * tau_z + Jxz_val * tau_x) / det

        return xdot

    def step(self, x, u, dt, xp):
        lo, hi = self.control_bounds
        u = xp.clip(u, xp.array(lo, dtype=u.dtype), xp.array(hi, dtype=u.dtype))

        x_cur = x.copy()
        dt_sub = dt / self.n_substeps

        x_cur = self._normalize_quat(x_cur, xp)
        x_cur = self._clamp_state(x_cur, xp)
        x_cur = self._clamp_alpha_beta(x_cur, xp)

        for _ in range(self.n_substeps):
            k1 = self.dynamics(x_cur, u, xp)
            k1 = self._clamp_derivatives(k1, xp)

            xtmp = x_cur + 0.5 * dt_sub * k1
            xtmp = self._normalize_quat(xtmp, xp)
            xtmp = self._clamp_state(xtmp, xp)
            xtmp = self._clamp_alpha_beta(xtmp, xp)

            k2 = self.dynamics(xtmp, u, xp)
            k2 = self._clamp_derivatives(k2, xp)

            xtmp = x_cur + 0.5 * dt_sub * k2
            xtmp = self._normalize_quat(xtmp, xp)
            xtmp = self._clamp_state(xtmp, xp)
            xtmp = self._clamp_alpha_beta(xtmp, xp)

            k3 = self.dynamics(xtmp, u, xp)
            k3 = self._clamp_derivatives(k3, xp)

            xtmp = x_cur + dt_sub * k3
            xtmp = self._normalize_quat(xtmp, xp)
            xtmp = self._clamp_state(xtmp, xp)
            xtmp = self._clamp_alpha_beta(xtmp, xp)

            k4 = self.dynamics(xtmp, u, xp)
            k4 = self._clamp_derivatives(k4, xp)

            x_cur = x_cur + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x_cur = self._normalize_quat(x_cur, xp)
            x_cur = self._clamp_state(x_cur, xp)
            x_cur = self._clamp_alpha_beta(x_cur, xp)

        return x_cur

    def simulate_single(self, x0, u_func, t_span, dt_eval=0.01):
        """Single trajectory via scipy Radau with quaternion constraint stabilization."""
        from scipy.integrate import solve_ivp

        BAUMGARTE_LAMBDA = 100.0

        def rhs(t, state):
            x_2d = state.reshape(1, -1)
            u_2d = np.array(u_func(t)).reshape(1, -1)
            lo, hi = self.control_bounds
            u_2d = np.clip(u_2d, lo, hi)
            xdot = self.dynamics(x_2d, u_2d, np).flatten()

            q = state[6:10]
            q_norm_sq = np.sum(q**2)
            xdot[6:10] += -BAUMGARTE_LAMBDA * (q_norm_sq - 1.0) * q

            return xdot

        t_eval = np.arange(t_span[0], t_span[1], dt_eval)
        sol = solve_ivp(rhs, t_span, x0, method='Radau',
                        t_eval=t_eval, rtol=1e-8, atol=1e-10,
                        max_step=dt_eval)

        for i in range(sol.y.shape[1]):
            q = sol.y[6:10, i]
            qn = np.linalg.norm(q)
            if qn > 1e-6:
                sol.y[6:10, i] = q / qn
            else:
                sol.y[6:10, i] = [1, 0, 0, 0]

        return sol.t, sol.y.T

    def _normalize_quat(self, x, xp):
        qw, qx_q, qy_q, qz_q = x[:, 6], x[:, 7], x[:, 8], x[:, 9]
        norm = xp.sqrt(qw**2 + qx_q**2 + qy_q**2 + qz_q**2)
        valid = norm > 1e-6
        x[:, 6] = xp.where(valid, qw / norm, 1.0)
        x[:, 7] = xp.where(valid, qx_q / norm, 0.0)
        x[:, 8] = xp.where(valid, qy_q / norm, 0.0)
        x[:, 9] = xp.where(valid, qz_q / norm, 0.0)
        return x

    def _clamp_state(self, x, xp):
        x[:, 3] = xp.clip(x[:, 3], -MAX_VELOCITY, MAX_VELOCITY)
        x[:, 4] = xp.clip(x[:, 4], -MAX_LATERAL_VEL, MAX_LATERAL_VEL)
        x[:, 5] = xp.clip(x[:, 5], -MAX_VELOCITY, MAX_VELOCITY)

        vx, vy, vz = x[:, 3], x[:, 4], x[:, 5]
        vel_mag = xp.sqrt(vx**2 + vy**2 + vz**2)
        too_slow = vel_mag < MIN_VELOCITY
        scale = xp.where((vel_mag > 1e-6) & too_slow, MIN_VELOCITY / vel_mag, 1.0)
        zero_vel = vel_mag <= 1e-6
        x[:, 3] = xp.where(zero_vel, MIN_VELOCITY, x[:, 3] * scale)
        x[:, 4] = xp.where(zero_vel, 0.0, x[:, 4] * scale)
        x[:, 5] = xp.where(zero_vel, 0.0, x[:, 5] * scale)

        x[:, 10] = xp.clip(x[:, 10], -MAX_OMEGA, MAX_OMEGA)
        x[:, 11] = xp.clip(x[:, 11], -MAX_OMEGA, MAX_OMEGA)
        x[:, 12] = xp.clip(x[:, 12], -MAX_OMEGA, MAX_OMEGA)
        return x

    def _clamp_alpha_beta(self, x, xp):
        vx, vy, vz = x[:, 3], x[:, 4], x[:, 5]
        V = xp.sqrt(vx**2 + vy**2 + vz**2)
        do_clamp = V > TOL_V

        alpha = xp.arctan2(-vz, vx)
        beta = xp.arcsin(xp.clip(vy / xp.clip(V, TOL_V, None), -1.0, 1.0))

        alpha_c = xp.clip(alpha, -MAX_ALPHA, MAX_ALPHA)
        beta_c = xp.clip(beta, -MAX_BETA, MAX_BETA)

        needs_clamp = ((alpha != alpha_c) | (beta != beta_c)) & do_clamp

        cos_a, sin_a = xp.cos(alpha_c), xp.sin(alpha_c)
        cos_b, sin_b = xp.cos(beta_c), xp.sin(beta_c)

        x[:, 3] = xp.where(needs_clamp, V * cos_a * cos_b, x[:, 3])
        x[:, 4] = xp.where(needs_clamp, V * sin_b, x[:, 4])
        x[:, 5] = xp.where(needs_clamp, -V * sin_a * cos_b, x[:, 5])
        return x

    def _clamp_derivatives(self, xdot, xp):
        xdot[:, 3:6] = xp.clip(xdot[:, 3:6], -MAX_ACCEL, MAX_ACCEL)
        xdot[:, 10:13] = xp.clip(xdot[:, 10:13], -MAX_ANG_ACCEL, MAX_ANG_ACCEL)
        return xdot

    def running_cost(self, x, u, t, xp):
        K = x.shape[0]
        pos = x[:, :3]
        wp = xp.array(self.waypoints, dtype=x.dtype)
        n_wp = wp.shape[0]

        best_cost = xp.full(K, 1e10, dtype=x.dtype)
        best_lateral = xp.zeros(K, dtype=x.dtype)
        best_seg_hat = xp.zeros((K, 3), dtype=x.dtype)

        for i in range(n_wp):
            j = (i + 1) % n_wp
            seg = wp[j] - wp[i]
            seg_len = float(xp.sqrt(xp.sum(seg**2)))
            if seg_len < 1e-6:
                continue
            seg_hat = seg / seg_len
            to_pos = pos - wp[i][None, :]
            proj = xp.clip(xp.sum(to_pos * seg_hat[None, :], axis=1), 0, seg_len)
            closest = wp[i][None, :] + proj[:, None] * seg_hat[None, :]
            diff = pos - closest
            lateral_err = xp.sqrt(xp.sum(diff[:, :2]**2, axis=1))
            alt_err = xp.abs(diff[:, 2])
            seg_cost = self.w_lateral * lateral_err**2 + self.w_altitude * alt_err**2
            better = seg_cost < best_cost
            best_cost = xp.where(better, seg_cost, best_cost)
            best_lateral = xp.where(better, lateral_err, best_lateral)
            best_seg_hat = xp.where(better[:, None], seg_hat[None, :], best_seg_hat)

        # Signed progress: velocity dot path tangent (world frame)
        qw = x[:, 6]; qx = x[:, 7]; qy = x[:, 8]; qz = x[:, 9]
        vxb = x[:, 3]; vyb = x[:, 4]; vzb = x[:, 5]
        vx_w = (qw*qw+qx*qx-qy*qy-qz*qz)*vxb + 2*(qx*qy-qw*qz)*vyb + 2*(qx*qz+qw*qy)*vzb
        vy_w = 2*(qx*qy+qw*qz)*vxb + (qw*qw-qx*qx+qy*qy-qz*qz)*vyb + 2*(qy*qz-qw*qx)*vzb
        vz_w = 2*(qx*qz-qw*qy)*vxb + 2*(qy*qz+qw*qx)*vyb + (qw*qw-qx*qx-qy*qy+qz*qz)*vzb
        forward_speed = vx_w*best_seg_hat[:, 0] + vy_w*best_seg_hat[:, 1] + vz_w*best_seg_hat[:, 2]

        track_violation = xp.clip(best_lateral - self.track_half_width, 0, None)
        cost = best_cost - self.w_progress * forward_speed + self.w_boundary * track_violation**2

        vx, vy, vz = x[:, 3], x[:, 4], x[:, 5]
        speed = xp.sqrt(vx**2 + vy**2 + vz**2)
        cost += self.w_velocity * (speed - self.target_speed)**2
        cost += self.w_angular_rate * (x[:, 10]**2 + x[:, 11]**2 + x[:, 12]**2)

        alpha = xp.arctan2(-vz, vx)
        cost += self.w_stall * xp.clip(xp.abs(alpha) - 0.3, 0, None)**2

        alt = x[:, 2]
        cost += self.w_boundary * (xp.clip(1.5 - alt, 0, None)**2 + xp.clip(alt - 10.0, 0, None)**2)
        cost += self.w_boundary * (xp.clip(4.0 - speed, 0, None)**2 + xp.clip(speed - 17.0, 0, None)**2)
        cost += self.w_surface * (u[:, 0]**2 + u[:, 3]**2)

        return cost

    def terminal_cost(self, x, xp):
        K = x.shape[0]
        pos = x[:, :3]
        wp = xp.array(self.waypoints, dtype=x.dtype)
        min_dist = xp.full(K, 1e10, dtype=x.dtype)
        for i in range(wp.shape[0]):
            dist = xp.sum((pos - wp[i][None, :])**2, axis=1)
            min_dist = xp.minimum(min_dist, dist)
        return 5.0 * (self.w_lateral * min_dist + self.w_altitude * xp.abs(x[:, 2] - 3.0)**2)

    def nominal_control(self, x, t, xp):
        K = x.shape[0]
        u = xp.zeros((K, 4), dtype=x.dtype)
        u[:, 1] = 0.1
        u[:, 2] = 0.37
        return u

    def trim_state(self):
        x = np.zeros(13)
        x[3] = self.target_speed
        x[6] = 1.0
        x[2] = 3.0
        return x
