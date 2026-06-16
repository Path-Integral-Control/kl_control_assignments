"""GPU-fused fixed-wing dynamics for MPPI.

Single CUDA kernel runs the full rollout: K samples x T horizon x n_substeps x RK4.
One kernel launch per solve() call. Compiled once, reused every solve.

Requires: numba with CUDA support, NVIDIA driver matching CUDA toolkit version.

Usage:
    from mppi.models.fixed_wing.dynamics_gpu import FixedWingGPU
    model = FixedWingGPU()
"""

import numpy as np
from numba import cuda, float64 as f64
import math

from mppi.models.fixed_wing.dynamics import FixedWing


@cuda.jit(device=True)
def _dynamics_dev(x, u, p, xdot):
    vx = x[3]; vy = x[4]; vz = x[5]
    qw = x[6]; qx = x[7]; qy = x[8]; qz = x[9]
    P = x[10]; Q = x[11]; R = x[12]

    V = math.sqrt(vx*vx + vy*vy + vz*vz)
    if V < 1e-3:
        V = 1e-3

    alpha = math.atan2(-vz, vx)
    sb = vy / V
    if sb > 1.0: sb = 1.0
    if sb < -1.0: sb = -1.0
    beta = math.asin(sb)

    ail = 0.5236 * u[0]
    elev = 0.4189 * u[1]
    rud = -0.3491 * u[3]

    qbar = 0.5 * p[5] * V * V

    CL = p[21] + p[22] * alpha
    if abs(alpha) >= 0.3491:
        CL = 0.0
    CD = p[25] + p[26] * CL * CL

    CC = (-p[32]*beta + p[19]*ail/0.5236 + p[20]*rud/0.3491
          + p[34]*p[12]/(2*V)*P + p[33]*p[12]/(2*V)*R)

    Cl = p[14]*ail - p[15]*rud
    Cm = p[13] + p[23]*alpha + p[16]*elev + (p[3]-p[2])*CL
    Cn = p[27]*beta + p[17]*rud - p[18]*ail

    D = CD*qbar*p[4]; L = CL*qbar*p[4]; Fs = CC*qbar*p[4]

    vhx = vx/V; vhy = vy/V; vhz = vz/V
    ca = math.cos(alpha); sa = math.sin(alpha)
    cb = math.cos(beta); sb2 = math.sin(beta)

    FAx = -D*vhx + (-L*cb*sa) + (-Fs*sb2)
    FAy = -D*vhy + (-L*sb2*sa) + (Fs*cb)
    FAz = -D*vhz + (L*ca)

    thr = u[2]
    if thr < 0: thr = 0.0
    if thr > 1: thr = 1.0
    FTx = p[0] * thr

    mg = p[1] * p[6]
    gx = 2*(qx*qz - qw*qy)*(-mg)
    gy = 2*(qy*qz + qw*qx)*(-mg)
    gz = (qw*qw - qx*qx - qy*qy + qz*qz)*(-mg)

    m = p[1]
    xdot[0] = (qw*qw+qx*qx-qy*qy-qz*qz)*vx + 2*(qx*qy-qw*qz)*vy + 2*(qx*qz+qw*qy)*vz
    xdot[1] = 2*(qx*qy+qw*qz)*vx + (qw*qw-qx*qx+qy*qy-qz*qz)*vy + 2*(qy*qz-qw*qx)*vz
    xdot[2] = 2*(qx*qz-qw*qy)*vx + 2*(qy*qz+qw*qx)*vy + (qw*qw-qx*qx-qy*qy+qz*qz)*vz
    xdot[3] = (FAx+FTx+gx)/m - (Q*vz - R*vy)
    xdot[4] = (FAy+gy)/m - (R*vx - P*vz)
    xdot[5] = (FAz+gz)/m - (P*vy - Q*vx)
    xdot[6] = 0.5*(-qx*P - qy*Q - qz*R)
    xdot[7] = 0.5*(qw*P + qy*R - qz*Q)
    xdot[8] = 0.5*(qw*Q + qz*P - qx*R)
    xdot[9] = 0.5*(qw*R + qx*Q - qy*P)

    Jx = p[7]; Jy = p[8]; Jz = p[9]; Jxz = p[10]
    Jwx = Jx*P+Jxz*R; Jwy = Jy*Q; Jwz = Jz*R+Jxz*P
    tx = (qbar*p[4]*p[12]*Cl + p[28]*p[12]/(2*V)*P + p[31]*p[12]/(2*V)*R) - (Q*Jwz-R*Jwy)
    ty = (qbar*p[4]*p[11]*Cm + p[24]*p[11]/(2*V)*Q) - (R*Jwx-P*Jwz)
    tz = (qbar*p[4]*p[12]*Cn + p[30]*p[12]/(2*V)*P + p[29]*p[12]/(2*V)*R) - (P*Jwy-Q*Jwx)
    det = Jx*Jz - Jxz*Jxz
    if abs(det) < 1e-20:
        det = 1e-20
    xdot[10] = (Jz*tx+Jxz*tz)/det
    xdot[11] = ty/Jy
    xdot[12] = (Jx*tz+Jxz*tx)/det


@cuda.jit(device=True)
def _normalize_quat_dev(x):
    n = math.sqrt(x[6]*x[6]+x[7]*x[7]+x[8]*x[8]+x[9]*x[9])
    if n > 1e-6:
        x[6] /= n; x[7] /= n; x[8] /= n; x[9] /= n
    else:
        x[6] = 1; x[7] = 0; x[8] = 0; x[9] = 0


@cuda.jit(device=True)
def _clamp_state_dev(x):
    for i in range(3, 6):
        if x[i] > 15: x[i] = 15
        if x[i] < -15: x[i] = -15
    if x[4] > 0.5: x[4] = 0.5
    if x[4] < -0.5: x[4] = -0.5
    vm = math.sqrt(x[3]*x[3]+x[4]*x[4]+x[5]*x[5])
    if vm < 1.0 and vm > 1e-6:
        s = 1.0/vm; x[3] *= s; x[4] *= s; x[5] *= s
    elif vm <= 1e-6:
        x[3] = 1; x[4] = 0; x[5] = 0
    for i in range(10, 13):
        if x[i] > 1.57: x[i] = 1.57
        if x[i] < -1.57: x[i] = -1.57


@cuda.jit(device=True)
def _clamp_deriv_dev(xdot):
    for i in range(3, 6):
        if xdot[i] > 20: xdot[i] = 20
        if xdot[i] < -20: xdot[i] = -20
    for i in range(10, 13):
        if xdot[i] > 10: xdot[i] = 10
        if xdot[i] < -10: xdot[i] = -10


@cuda.jit(device=True)
def _clamp_ab_dev(x):
    vx = x[3]; vy = x[4]; vz = x[5]
    V = math.sqrt(vx*vx+vy*vy+vz*vz)
    if V > 1e-3:
        alpha = math.atan2(-vz, vx)
        sv = vy/V
        if sv > 1: sv = 1
        if sv < -1: sv = -1
        beta = math.asin(sv)
        ac = alpha; bc = beta
        if ac > 0.35: ac = 0.35
        if ac < -0.35: ac = -0.35
        if bc > 0.0873: bc = 0.0873
        if bc < -0.0873: bc = -0.0873
        if alpha != ac or beta != bc:
            x[3] = V*math.cos(ac)*math.cos(bc)
            x[4] = V*math.sin(bc)
            x[5] = -V*math.sin(ac)*math.cos(bc)


@cuda.jit(device=True)
def _rk4_substep_dev(x, u, p, dt_sub, k1, k2, k3, k4, tmp):
    _dynamics_dev(x, u, p, k1); _clamp_deriv_dev(k1)
    for i in range(13): tmp[i] = x[i] + 0.5*dt_sub*k1[i]
    _normalize_quat_dev(tmp); _clamp_state_dev(tmp); _clamp_ab_dev(tmp)

    _dynamics_dev(tmp, u, p, k2); _clamp_deriv_dev(k2)
    for i in range(13): tmp[i] = x[i] + 0.5*dt_sub*k2[i]
    _normalize_quat_dev(tmp); _clamp_state_dev(tmp); _clamp_ab_dev(tmp)

    _dynamics_dev(tmp, u, p, k3); _clamp_deriv_dev(k3)
    for i in range(13): tmp[i] = x[i] + dt_sub*k3[i]
    _normalize_quat_dev(tmp); _clamp_state_dev(tmp); _clamp_ab_dev(tmp)

    _dynamics_dev(tmp, u, p, k4); _clamp_deriv_dev(k4)

    for i in range(13):
        x[i] += (dt_sub/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
    _normalize_quat_dev(x); _clamp_state_dev(x); _clamp_ab_dev(x)


@cuda.jit(device=True)
def _path_cost_dev(x, u, wp, n_wp, cw):
    """Path tracking cost. cw = [w_lat, w_alt, w_prog, w_vel, w_ang, w_stall, w_bnd, w_srf, tgt_spd, track_hw]"""
    px = x[0]; py = x[1]; pz = x[2]

    best_cost = 1e10
    best_lat = 0.0
    best_shx = 0.0; best_shy = 0.0; best_shz = 0.0

    for i in range(n_wp):
        j = (i + 1) % n_wp
        sx = wp[j, 0] - wp[i, 0]
        sy = wp[j, 1] - wp[i, 1]
        sz = wp[j, 2] - wp[i, 2]
        seg_len = math.sqrt(sx*sx + sy*sy + sz*sz)
        if seg_len < 1e-6:
            continue
        shx = sx/seg_len; shy = sy/seg_len; shz = sz/seg_len
        tpx = px - wp[i, 0]; tpy = py - wp[i, 1]; tpz = pz - wp[i, 2]
        proj = tpx*shx + tpy*shy + tpz*shz
        if proj < 0: proj = 0.0
        if proj > seg_len: proj = seg_len
        cx = wp[i, 0] + proj*shx; cy = wp[i, 1] + proj*shy; cz = wp[i, 2] + proj*shz
        dx = px - cx; dy = py - cy; dz = pz - cz
        lat = math.sqrt(dx*dx + dy*dy)
        alt = abs(dz)
        sc = cw[0]*lat*lat + cw[1]*alt*alt
        if sc < best_cost:
            best_cost = sc
            best_lat = lat
            best_shx = shx; best_shy = shy; best_shz = shz

    # Signed progress: world-frame velocity dot path tangent
    qw = x[6]; qx = x[7]; qy = x[8]; qz = x[9]
    vxb = x[3]; vyb = x[4]; vzb = x[5]
    vx_w = (qw*qw+qx*qx-qy*qy-qz*qz)*vxb + 2*(qx*qy-qw*qz)*vyb + 2*(qx*qz+qw*qy)*vzb
    vy_w = 2*(qx*qy+qw*qz)*vxb + (qw*qw-qx*qx+qy*qy-qz*qz)*vyb + 2*(qy*qz-qw*qx)*vzb
    vz_w = 2*(qx*qz-qw*qy)*vxb + 2*(qy*qz+qw*qx)*vyb + (qw*qw-qx*qx-qy*qy+qz*qz)*vzb
    fwd_speed = vx_w*best_shx + vy_w*best_shy + vz_w*best_shz

    cost = best_cost - cw[2]*fwd_speed

    # Track boundary violation
    track_hw = cw[9]
    if best_lat > track_hw:
        violation = best_lat - track_hw
        cost += cw[6] * violation * violation

    # Speed
    spd = math.sqrt(x[3]*x[3] + x[4]*x[4] + x[5]*x[5])
    cost += cw[3] * (spd - cw[8])**2

    # Angular rate
    cost += cw[4] * (x[10]*x[10] + x[11]*x[11] + x[12]*x[12])

    # Stall
    alpha = math.atan2(-x[5], x[3])
    aa = abs(alpha) - 0.3
    if aa > 0:
        cost += cw[5] * aa * aa

    # Altitude/speed bounds
    if x[2] < 1.5:
        cost += cw[6] * (1.5 - x[2])**2
    if x[2] > 10.0:
        cost += cw[6] * (x[2] - 10.0)**2
    if spd < 4.0:
        cost += cw[6] * (4.0 - spd)**2
    if spd > 17.0:
        cost += cw[6] * (spd - 17.0)**2

    # Surface deflection
    cost += cw[7] * (u[0]*u[0] + u[3]*u[3])

    return cost


@cuda.jit
def _rollout_kernel(states, controls, params, costs, wp, n_wp, cost_weights,
                    dt, n_sub, K, T):
    k = cuda.grid(1)
    if k >= K:
        return

    x = cuda.local.array(13, dtype=f64)
    tmp = cuda.local.array(13, dtype=f64)
    k1 = cuda.local.array(13, dtype=f64)
    k2 = cuda.local.array(13, dtype=f64)
    k3 = cuda.local.array(13, dtype=f64)
    k4 = cuda.local.array(13, dtype=f64)
    u = cuda.local.array(4, dtype=f64)
    p = cuda.local.array(35, dtype=f64)
    cw = cuda.local.array(10, dtype=f64)

    for i in range(35):
        p[i] = params[i]
    for i in range(10):
        cw[i] = cost_weights[i]
    for i in range(13):
        x[i] = states[k, 0, i]

    dt_sub = dt / n_sub
    total_cost = 0.0

    for t in range(T):
        for i in range(4):
            u[i] = controls[k, t, i]

        _normalize_quat_dev(x)
        _clamp_state_dev(x)
        _clamp_ab_dev(x)

        for sub in range(n_sub):
            _rk4_substep_dev(x, u, p, dt_sub, k1, k2, k3, k4, tmp)

        for i in range(13):
            states[k, t+1, i] = x[i]

        total_cost += _path_cost_dev(x, u, wp, n_wp, cw)

    # Terminal cost
    best_dist = 1e10
    for i in range(n_wp):
        dx = x[0] - wp[i, 0]; dy = x[1] - wp[i, 1]
        d = dx*dx + dy*dy
        if d < best_dist:
            best_dist = d
    total_cost += 5.0 * (cw[0] * best_dist + cw[1] * (x[2] - 3.0)**2)

    costs[k] = total_cost


@cuda.jit
def _step_kernel(states_in, states_out, controls, params, dt, n_sub, K):
    """Single-step kernel: step K states by one timestep. For execution loop."""
    k = cuda.grid(1)
    if k >= K:
        return

    x = cuda.local.array(13, dtype=f64)
    tmp = cuda.local.array(13, dtype=f64)
    k1 = cuda.local.array(13, dtype=f64)
    k2 = cuda.local.array(13, dtype=f64)
    k3 = cuda.local.array(13, dtype=f64)
    k4 = cuda.local.array(13, dtype=f64)
    u = cuda.local.array(4, dtype=f64)
    p = cuda.local.array(35, dtype=f64)

    for i in range(35):
        p[i] = params[i]
    for i in range(13):
        x[i] = states_in[k, i]
    for i in range(4):
        u[i] = controls[k, i]

    dt_sub = dt / n_sub
    _normalize_quat_dev(x)
    _clamp_state_dev(x)
    _clamp_ab_dev(x)

    for sub in range(n_sub):
        _rk4_substep_dev(x, u, p, dt_sub, k1, k2, k3, k4, tmp)

    for i in range(13):
        states_out[k, i] = x[i]


class FixedWingGPU(FixedWing):
    """Fixed-wing with GPU-fused rollout kernel.

    full_rollout(): single kernel launch for K x T x substeps x RK4.
    step(): single kernel launch for K x substeps x RK4.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('n_substeps', 3)
        super().__init__(**kwargs)
        self._d_params = None
        self._d_wp = None
        self._d_cw = None
        self._threads = 128

    def _ensure_device_arrays(self):
        if self._d_params is None:
            self._d_params = cuda.to_device(self.params.astype(np.float64))
        if self._d_wp is None:
            self._d_wp = cuda.to_device(self.waypoints.astype(np.float64))
        if self._d_cw is None:
            cw = np.array([
                self.w_lateral, self.w_altitude, self.w_progress,
                self.w_velocity, self.w_angular_rate, self.w_stall,
                self.w_boundary, self.w_surface, self.target_speed,
                self.track_half_width,
            ], dtype=np.float64)
            self._d_cw = cuda.to_device(cw)

    def step(self, x, u, dt, xp):
        self._ensure_device_arrays()
        K = x.shape[0]
        x64 = np.asarray(x, dtype=np.float64)
        u64 = np.asarray(u, dtype=np.float64)
        lo, hi = self.control_bounds
        u64 = np.clip(u64, lo, hi)

        d_in = cuda.to_device(x64)
        d_out = cuda.device_array((K, 13), dtype=np.float64)
        d_u = cuda.to_device(u64)

        blocks = (K + self._threads - 1) // self._threads
        _step_kernel[blocks, self._threads](
            d_in, d_out, d_u, self._d_params, dt, self.n_substeps, K)

        return d_out.copy_to_host()

    def full_rollout(self, x0, U_perturbed, dt):
        """Full fused rollout. Returns (states, costs) as numpy arrays.

        Parameters
        ----------
        x0 : array (13,)
        U_perturbed : array (K, T, 4)
        dt : float

        Returns
        -------
        states : array (K, T+1, 13)
        costs : array (K,)
        """
        self._ensure_device_arrays()
        K, T, _ = U_perturbed.shape
        n_wp = self.waypoints.shape[0]

        states = np.zeros((K, T+1, 13), dtype=np.float64)
        x0_64 = np.asarray(x0, dtype=np.float64)
        states[:, 0] = x0_64[None, :]
        costs = np.zeros(K, dtype=np.float64)

        d_states = cuda.to_device(states)
        d_controls = cuda.to_device(np.asarray(U_perturbed, dtype=np.float64))
        d_costs = cuda.to_device(costs)

        blocks = (K + self._threads - 1) // self._threads
        _rollout_kernel[blocks, self._threads](
            d_states, d_controls, self._d_params, d_costs,
            self._d_wp, n_wp, self._d_cw,
            dt, self.n_substeps, K, T)

        cuda.synchronize()
        return d_states.copy_to_host(), d_costs.copy_to_host()
