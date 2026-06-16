"""TECS nominal controller for fixed-wing MPPI.

Ported from MPPI-PURT:
  Python source: auav_pylon_2026/tecs_controller_xtrack_sample.py
  Gains: param/sim_stateless.yaml (all integrals zeroed, MATCHED CUDA)
  Sign conventions: SIGN_CONVENTIONS.md

This is the stateless (no integrators) version used as the nominal
policy during MPPI rollouts. It includes alpha-beta filters and
slew limiting which are necessary for convergence.

Control output: [aileron, elevator, throttle, rudder]
  - elevator output is NOT inverted here — the sim-side flip
    (u[1] = -elev_cmd) must be applied by the caller or dynamics.
    See SIGN_CONVENTIONS.md §3 for why.

Frame conventions:
  - Pitch state from quaternion: positive = nose DOWN (cyecca)
  - TECS internal: flips to nose-up-positive
  - Gamma: arcsin(vz_global / V) — Python convention
  - Chi: uses yaw (body heading), not ground track
  - Chi error: (chi_ref - chi) * -1
"""

import numpy as np
import yaml
from pathlib import Path


def _wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class TECSController:
    """Stateless TECS PD controller for fixed-wing MPPI nominal policy.

    Ported line-by-line from tecs_controller_xtrack_sample.py with
    integral gains set to zero (sim_stateless.yaml).
    """

    def __init__(self, gains_path=None):
        """Load gains from YAML or use sim_stateless defaults."""

        if gains_path is not None:
            with open(gains_path, 'r') as f:
                p = yaml.safe_load(f)
        else:
            p = {}

        self.g = 9.81
        self.thr_max = 7.5  # TECS internal scaling (tuned, not physical thrust)

        # --- Outer loop: energy → thrust ---
        self.K_thrustp = p.get('K_thrustp', 0.3)
        self.K_energy_p = p.get('K_energy_p', 10.0)
        self.K_energy_d = p.get('K_energy_d', 0.6)
        self.K_pitchp = p.get('K_pitchp', 0.85)
        self.K_gamma_p = p.get('K_gamma_p', 2.8)
        self.K_alt = p.get('K_alt', 1.0)
        self.trim_thrust = p.get('trim_thrust', 3.5)
        self.trim_thrust_low = p.get('trim_thrust_low', 2.8)
        self.alt_trim_low = p.get('alt_trim_low', 2.5)
        self.alt_trim_high = p.get('alt_trim_high', 7.0)

        # --- Inner loop: pitch → elevator ---
        self.K_elevp = p.get('K_elevp', 1.2)
        self.K_q = p.get('K_q', 0.7)
        self.K_phi_elev = p.get('K_phi_elev', 1.0)
        self.trim_elev = p.get('trim_elev', 0.0)

        # --- Lateral: heading → roll → aileron ---
        self.k_chi = p.get('k_chi', 1.5)
        self.k_chi_d = p.get('k_chi_d', 0.1)
        self.K_phi_p = p.get('K_phi_p', 3.5)
        self.K_phi_d = p.get('K_phi_d', 0.2)
        self.K_phi_gravity_ff = p.get('K_phi_gravity_ff', 0.15)
        self.K_phi_ff = p.get('K_phi_ff', 0.3)
        self.phi_lim = np.deg2rad(p.get('phi_lim_deg', 55.0))
        self.phi_dot_lim = np.deg2rad(p.get('phi_dot_lim_deg_s', 150.0))
        self.phi_dot_lim_cross_solve = np.deg2rad(p.get('phi_dot_lim_cross_solve_deg', 900.0))
        self.chi_deadband = np.deg2rad(p.get('chi_deadband_deg', 0.5))
        self.curvature_blend_gain = p.get('curvature_blend_gain', 4.0)
        self.da_max = p.get('da_max', 1.0)
        self.trim_ail = p.get('trim_ail', 0.0)

        # --- Rudder ---
        self.Cnda = p.get('Cnda', 0.03)
        self.Cndr = p.get('Cndr', 0.12)

        # --- Vehicle ---
        self.mass = p.get('mass', 0.057)
        self.weight = self.mass * self.g

        # --- Alpha-beta filter states (MATCHED CUDA) ---
        self.chi_dot_ab_alpha = 0.6
        self.chi_dot_ab_beta = 0.08
        self.phi_des_ab_alpha = 0.6
        self.phi_des_ab_beta = 0.3

        self._chi_dot_filtered = 0.0
        self._chi_dot_rate = 0.0
        self._chi_dot_init = False

        self._phi_des_filtered = 0.0
        self._phi_des_rate = 0.0
        self._phi_des_init = False
        self._phi_cmd = 0.0

    def compute_thrust_pitch(self, z, ref_gamma, ref_v, ref_accel,
                              V_est, gamma_est, vdot_est):
        """Outer loop: energy balance → thrust and pitch commands.

        Ported from tecs_controller_xtrack_sample.py:compute_thrust_pitch()
        """
        r_gamma = float(ref_gamma)
        r_V_dot = float(ref_accel)

        # Envelope protection
        drag = 1.0
        r_V_dot = np.clip(r_V_dot, -drag / self.weight,
                          (self.thr_max - drag) / self.weight)

        # --- Thrust ---
        error_norm_Es_dot = (r_gamma - gamma_est) + (r_V_dot - vdot_est) / self.g

        thrust_stability_term = self.K_thrustp * (gamma_est + vdot_est / self.g)
        thrust_direct_term = self.K_energy_p * error_norm_Es_dot
        thrust_damping_term = self.K_energy_d * gamma_est

        # Altitude-interpolated trim
        alt_range = max(self.alt_trim_high - self.alt_trim_low, 0.1)
        alt_frac = np.clip((z - self.alt_trim_low) / alt_range, 0.0, 1.0)
        effective_trim = self.trim_thrust_low + alt_frac * (self.trim_thrust - self.trim_thrust_low)

        thrust_unsat = (effective_trim
                        + self.weight * (thrust_stability_term
                                         + thrust_direct_term
                                         - thrust_damping_term))

        thr_min = 0.80 * effective_trim
        thrust = float(np.clip(thrust_unsat, thr_min, self.thr_max))

        # --- Pitch ---
        gamma_error = r_gamma - gamma_est
        pitch_direct_term = self.K_gamma_p * gamma_error
        pitch_stability_term = self.K_pitchp * (gamma_est - vdot_est / self.g)

        pitch_unsat = pitch_direct_term - pitch_stability_term
        ref_pitch = float(np.clip(pitch_unsat, np.deg2rad(-20), np.deg2rad(20)))

        return thrust, ref_pitch

    def compute(self, state, ref_data, dt=0.04):
        """Full TECS control: outer loop + inner loop surfaces.

        Parameters
        ----------
        state : array (13,)
            [px,py,pz, vx_b,vy_b,vz_b, qw,qx,qy,qz, p,q,r]
        ref_data : dict
            Must contain: des_heading, des_gamma, des_v, des_a
            Optional: curvature (default 0)
        dt : float

        Returns
        -------
        controls : array (4,)
            [aileron, elevator, throttle, rudder]
        """
        # --- Unpack state ---
        px, py, pz = state[0], state[1], state[2]
        vx_b, vy_b, vz_b = state[3], state[4], state[5]
        qw, qx, qy, qz = state[6], state[7], state[8], state[9]
        p_rate, q_rate, r_rate = state[10], state[11], state[12]

        # --- Derived quantities ---
        roll, pitch, yaw = _quat_to_euler(qw, qx, qy, qz)
        vx_w, vy_w, vz_w = _quat_rotate(qw, qx, qy, qz, vx_b, vy_b, vz_b)

        # Airspeed (body frame) — for TECS energy equations
        V_airspeed = np.sqrt(vx_b**2 + vy_b**2 + vz_b**2)
        V_airspeed = max(V_airspeed, 1e-3)

        # Ground speed (global frame) — for turn rate, curvature ff, phi_des
        V_ground = np.sqrt(vx_w**2 + vy_w**2 + vz_w**2)
        V_ground = max(V_ground, 1e-3)

        # Gamma: Python convention — arcsin(vz_global / V)
        gamma_est = np.arcsin(np.clip(vz_w / V_airspeed, -1, 1))
        vdot_est = 0.0  # stateless: no velocity history

        # --- Outer loop ---
        ref_gamma = ref_data.get('des_gamma', 0.0)
        ref_v = ref_data.get('des_v', 5.0)
        ref_accel = ref_data.get('des_a', 0.0)

        ref_thrust, ref_pitch = self.compute_thrust_pitch(
            pz, ref_gamma, ref_v, ref_accel, V_airspeed, gamma_est, vdot_est)

        ref_heading = ref_data.get('des_heading', 0.0)

        # ==========================================
        # ELEVATOR (pitch PD + turn coordination)
        # ==========================================
        # SIGN CONVENTION: pitch from quaternion is nose-up-NEGATIVE
        # TECS internal uses nose-up-POSITIVE
        pitch_actual = -1.0 * pitch  # §2 of SIGN_CONVENTIONS.md

        error_pitch = _wrap_pi(ref_pitch - pitch_actual)

        V_floor = max(V_ground, 5.0)  # q_turn uses ground speed
        roll_clamp = np.clip(roll, -1.2, 1.2)
        q_turn = (np.sin(roll) * np.cos(pitch_actual)
                  * np.tan(roll_clamp) * self.g / V_floor)
        error_q = np.clip(q_turn - q_rate, -1.0, 1.0)

        cos_roll_safe = max(np.cos(roll), 0.1)
        nz_excess = (1.0 / cos_roll_safe) - 1.0
        ele_ff_phi = self.K_phi_elev * nz_excess

        elev_cmd = (self.trim_elev
                    + self.K_elevp * error_pitch
                    + self.K_q * error_q
                    + ele_ff_phi)
        elev_cmd = np.clip(elev_cmd, -1.0, 1.0)

        # ==========================================
        # THROTTLE
        # ==========================================
        throttle_cmd = np.clip(ref_thrust / self.thr_max, 0.0, 1.0)

        # ==========================================
        # AILERON (heading → roll → PD)
        # ==========================================
        chi = _wrap_pi(yaw)  # body heading, not ground track (MATCHED CUDA)
        chi_ref = float(ref_heading)
        chi_err = _wrap_pi(chi_ref - chi) * -1.0  # §9 of SIGN_CONVENTIONS

        if abs(chi_err) < self.chi_deadband:
            chi_err = 0.0

        path_curvature = ref_data.get('curvature', 0.0)
        Vg = max(V_ground, 0.05)  # Curvature ff and phi_des use ground speed

        # Kinematic yaw rate from body rates (MATCHED CUDA)
        cos_pitch_safe = max(np.cos(pitch), 0.1)
        chi_dot_raw = (q_rate * np.sin(roll) + r_rate * np.cos(roll)) / cos_pitch_safe

        # Alpha-beta filter on chi_dot (MATCHED CUDA)
        if self._chi_dot_init:
            predicted = self._chi_dot_filtered + self._chi_dot_rate * dt
            residual = chi_dot_raw - predicted
            chi_dot_measured = predicted + self.chi_dot_ab_alpha * residual
            self._chi_dot_rate += (self.chi_dot_ab_beta / dt) * residual
            self._chi_dot_rate = np.clip(self._chi_dot_rate, -50.0, 50.0)
            self._chi_dot_filtered = chi_dot_measured
        else:
            chi_dot_measured = chi_dot_raw
            self._chi_dot_filtered = chi_dot_raw
            self._chi_dot_rate = 0.0
            self._chi_dot_init = True

        # Feedforward from path curvature (MATCHED CUDA: negated)
        chi_dot_ff = -Vg * path_curvature

        # D-term on heading error rate
        chi_err_dot = chi_dot_measured - chi_dot_ff
        chi_err_dot = np.clip(chi_err_dot, -2.0, 2.0)

        chi_dot_fb = self.k_chi * chi_err + self.k_chi_d * chi_err_dot

        # Non-convex alpha blending (MATCHED CUDA: alpha_max > 1)
        alpha_max = 1.5
        alpha = np.clip(abs(path_curvature) * self.curvature_blend_gain, 0.3, 1.0)
        chi_dot_des = (alpha_max - alpha) * chi_dot_fb + alpha * chi_dot_ff

        # Desired bank angle
        phi_des = np.arctan2(Vg * chi_dot_des, self.g)
        phi_des = float(np.clip(phi_des, -self.phi_lim, self.phi_lim))

        # Slew rate limit (MATCHED CUDA: cross-solve rate for real-time)
        dphi_max = self.phi_dot_lim_cross_solve * dt
        phi_des = np.clip(phi_des - self._phi_cmd, -dphi_max, dphi_max) + self._phi_cmd
        phi_des = float(np.clip(phi_des, -self.phi_lim, self.phi_lim))

        # Alpha-beta filter on phi_des (MATCHED CUDA)
        phi_des_rate = 0.0
        if self._phi_des_init:
            predicted = self._phi_des_filtered + self._phi_des_rate * dt
            residual = phi_des - predicted
            phi_des = predicted + self.phi_des_ab_alpha * residual
            phi_des_rate = self._phi_des_rate + (self.phi_des_ab_beta / dt) * residual
            phi_des_rate = np.clip(phi_des_rate, -self.phi_dot_lim, self.phi_dot_lim)
            phi_des = float(np.clip(phi_des, -self.phi_lim, self.phi_lim))
        else:
            self._phi_des_init = True

        self._phi_des_filtered = phi_des
        self._phi_des_rate = phi_des_rate
        self._phi_cmd = phi_des

        # Roll PD with feedforward (MATCHED CUDA)
        e_phi = _wrap_pi(self._phi_cmd - roll)
        p_clamped = np.clip(p_rate, -2.0, 2.0)

        ail_gravity_ff = self.K_phi_gravity_ff * np.sin(self._phi_cmd)
        ail_rate_ff = self.K_phi_ff * phi_des_rate

        ail_cmd = (self.trim_ail
                   + self.K_phi_p * e_phi
                   - self.K_phi_d * p_clamped
                   + ail_gravity_ff
                   + ail_rate_ff)
        ail_cmd = float(np.clip(ail_cmd, -self.da_max, self.da_max))

        # ==========================================
        # RUDDER (adverse yaw compensation)
        # ==========================================
        if abs(self.Cndr) > 1e-6:
            rud_cmd = -ail_cmd * (self.Cnda / self.Cndr)
        else:
            rud_cmd = 0.0
        rud_cmd = float(np.clip(rud_cmd, -1.0, 1.0))

        return np.array([ail_cmd, elev_cmd, throttle_cmd, rud_cmd])

    def reset(self):
        """Reset filter states (call when starting a new trajectory)."""
        self._chi_dot_filtered = 0.0
        self._chi_dot_rate = 0.0
        self._chi_dot_init = False
        self._phi_des_filtered = 0.0
        self._phi_des_rate = 0.0
        self._phi_des_init = False
        self._phi_cmd = 0.0


# ============================================================
# Helpers
# ============================================================

def _quat_to_euler(qw, qx, qy, qz):
    """Quaternion (scalar-first) → Euler angles (roll, pitch, yaw). ZYX."""
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    sinp = np.clip(2*(qw*qy - qz*qx), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return roll, pitch, yaw


def _quat_rotate(qw, qx, qy, qz, vx, vy, vz):
    """Rotate vector by quaternion (body → world)."""
    rx = (qw*qw + qx*qx - qy*qy - qz*qz)*vx + 2*(qx*qy - qw*qz)*vy + 2*(qx*qz + qw*qy)*vz
    ry = 2*(qx*qy + qw*qz)*vx + (qw*qw - qx*qx + qy*qy - qz*qz)*vy + 2*(qy*qz - qw*qx)*vz
    rz = 2*(qx*qz - qw*qy)*vx + 2*(qy*qz + qw*qx)*vy + (qw*qw - qx*qx - qy*qy + qz*qz)*vz
    return rx, ry, rz
