"""JIT-compiled fixed-wing dynamics for fast MPPI rollouts.

Uses Numba @njit to compile the RK4 + substep loop into native code.
Eliminates Python loop overhead — runs ~10x faster than interpreted.

Usage:
    from mppi.models.fixed_wing.dynamics_jit import FixedWingJIT
    model = FixedWingJIT(n_substeps=3)
    # Use with MPPI solver same as FixedWing
"""

import numpy as np
from numba import njit, prange
import math

from mppi.models.fixed_wing.dynamics import (
    FixedWing, DEFAULT_PARAMS, _params_to_array, N_SUBSTEPS
)


@njit(cache=True)
def _dynamics_jit(x, u, p, xdot):
    """Fixed-wing dynamics — JIT compiled."""
    vx = x[3]; vy = x[4]; vz = x[5]
    qw = x[6]; qx = x[7]; qy = x[8]; qz = x[9]
    P = x[10]; Q = x[11]; R = x[12]

    V = math.sqrt(vx*vx + vy*vy + vz*vz)
    if V < 1e-3: V = 1e-3

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
    if abs(alpha) >= 0.3491: CL = 0.0
    CD = p[25] + p[26] * CL * CL

    CC = (-p[32]*beta + p[19]*ail/0.5236 + p[20]*rud/0.3491
          + p[34]*p[12]/(2*V)*P + p[33]*p[12]/(2*V)*R)

    Cl = p[14]*ail - p[15]*rud
    Cm = p[13] + p[23]*alpha + p[16]*elev + (p[3]-p[2])*CL
    Cn = p[27]*beta + p[17]*rud - p[18]*ail

    D = CD*qbar*p[4]; L = CL*qbar*p[4]; Fs = CC*qbar*p[4]

    vh_x = vx/V; vh_y = vy/V; vh_z = vz/V
    ca = math.cos(alpha); sa = math.sin(alpha)
    cb = math.cos(beta); sb2 = math.sin(beta)

    FAx = -D*vh_x + (-L*cb*sa) + (-Fs*sb2)
    FAy = -D*vh_y + (-L*sb2*sa) + (Fs*cb)
    FAz = -D*vh_z + (L*ca)

    thr = u[2]
    if thr < 0: thr = 0.0
    if thr > 1: thr = 1.0
    FTx = p[0] * thr

    # Gravity in body frame: rotate [0,0,-mg] by inverse quaternion
    # _quat_rotate_inv: conjugate q then rotate
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
    Mx = qbar*p[4]*p[12]*Cl + p[28]*p[12]/(2*V)*P + p[31]*p[12]/(2*V)*R
    My = qbar*p[4]*p[11]*Cm + p[24]*p[11]/(2*V)*Q
    Mz = qbar*p[4]*p[12]*Cn + p[30]*p[12]/(2*V)*P + p[29]*p[12]/(2*V)*R
    Jwx = Jx*P+Jxz*R; Jwy = Jy*Q; Jwz = Jz*R+Jxz*P
    tx = Mx-(Q*Jwz-R*Jwy); ty = My-(R*Jwx-P*Jwz); tz = Mz-(P*Jwy-Q*Jwx)
    det = Jx*Jz - Jxz*Jxz
    if abs(det) < 1e-20: det = 1e-20
    xdot[10] = (Jz*tx+Jxz*tz)/det
    xdot[11] = ty/Jy
    xdot[12] = (Jx*tz+Jxz*tx)/det


@njit(cache=True)
def _normalize_quat_jit(x):
    n = math.sqrt(x[6]*x[6]+x[7]*x[7]+x[8]*x[8]+x[9]*x[9])
    if n > 1e-6:
        x[6]/=n; x[7]/=n; x[8]/=n; x[9]/=n
    else:
        x[6]=1; x[7]=0; x[8]=0; x[9]=0


@njit(cache=True)
def _clamp_state_jit(x):
    for i in range(3,6):
        if x[i]>15: x[i]=15
        if x[i]<-15: x[i]=-15
    if x[4]>0.5: x[4]=0.5
    if x[4]<-0.5: x[4]=-0.5
    vm = math.sqrt(x[3]*x[3]+x[4]*x[4]+x[5]*x[5])
    if vm < 1.0 and vm > 1e-6:
        s = 1.0/vm; x[3]*=s; x[4]*=s; x[5]*=s
    elif vm <= 1e-6:
        x[3]=1; x[4]=0; x[5]=0
    for i in range(10,13):
        if x[i]>1.57: x[i]=1.57
        if x[i]<-1.57: x[i]=-1.57


@njit(cache=True)
def _clamp_deriv_jit(xdot):
    for i in range(3,6):
        if xdot[i]>20: xdot[i]=20
        if xdot[i]<-20: xdot[i]=-20
    for i in range(10,13):
        if xdot[i]>10: xdot[i]=10
        if xdot[i]<-10: xdot[i]=-10


@njit(cache=True)
def _clamp_ab_jit(x):
    vx=x[3]; vy=x[4]; vz=x[5]
    V = math.sqrt(vx*vx+vy*vy+vz*vz)
    if V > 1e-3:
        alpha = math.atan2(-vz, vx)
        sv = vy/V
        if sv>1: sv=1
        if sv<-1: sv=-1
        beta = math.asin(sv)
        ac = alpha; bc = beta
        if ac>0.35: ac=0.35
        if ac<-0.35: ac=-0.35
        if bc>0.0873: bc=0.0873
        if bc<-0.0873: bc=-0.0873
        if alpha!=ac or beta!=bc:
            x[3]=V*math.cos(ac)*math.cos(bc)
            x[4]=V*math.sin(bc)
            x[5]=-V*math.sin(ac)*math.cos(bc)


@njit(cache=True)
def _step_single_jit(x, u, p, dt, n_sub):
    """RK4 with substeps for a single state. Returns new state."""
    out = x.copy()
    dt_sub = dt / n_sub
    k1 = np.empty(13)
    k2 = np.empty(13)
    k3 = np.empty(13)
    k4 = np.empty(13)
    tmp = np.empty(13)

    _normalize_quat_jit(out)
    _clamp_state_jit(out)
    _clamp_ab_jit(out)

    for _ in range(n_sub):
        _dynamics_jit(out, u, p, k1); _clamp_deriv_jit(k1)
        for i in range(13): tmp[i] = out[i]+0.5*dt_sub*k1[i]
        _normalize_quat_jit(tmp); _clamp_state_jit(tmp); _clamp_ab_jit(tmp)

        _dynamics_jit(tmp, u, p, k2); _clamp_deriv_jit(k2)
        for i in range(13): tmp[i] = out[i]+0.5*dt_sub*k2[i]
        _normalize_quat_jit(tmp); _clamp_state_jit(tmp); _clamp_ab_jit(tmp)

        _dynamics_jit(tmp, u, p, k3); _clamp_deriv_jit(k3)
        for i in range(13): tmp[i] = out[i]+dt_sub*k3[i]
        _normalize_quat_jit(tmp); _clamp_state_jit(tmp); _clamp_ab_jit(tmp)

        _dynamics_jit(tmp, u, p, k4); _clamp_deriv_jit(k4)

        for i in range(13):
            out[i] += (dt_sub/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
        _normalize_quat_jit(out); _clamp_state_jit(out); _clamp_ab_jit(out)

    return out


@njit(parallel=True, cache=True)
def _step_batch_jit(states, controls, params, dt, n_sub):
    """Step K states in parallel using Numba prange."""
    K = states.shape[0]
    out = np.empty_like(states)
    for k in prange(K):
        out[k] = _step_single_jit(states[k], controls[k], params, dt, n_sub)
    return out


class FixedWingJIT(FixedWing):
    """Fixed-wing with JIT-compiled dynamics (~3x faster). Uses float64."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._params_f64 = self.params.astype(np.float64)
        # Warm up JIT on first call
        _x = self.trim_state().astype(np.float64)
        _u = np.zeros(4, dtype=np.float64)
        _step_single_jit(_x, _u, self._params_f64, 0.04, self.n_substeps)

    def step(self, x, u, dt, xp):
        """Batched step using JIT-compiled dynamics (float64)."""
        x64 = np.asarray(x, dtype=np.float64)
        u64 = np.asarray(u, dtype=np.float64)
        lo, hi = self.control_bounds
        u64 = np.clip(u64, lo, hi)
        return _step_batch_jit(x64, u64, self._params_f64, dt, self.n_substeps)
