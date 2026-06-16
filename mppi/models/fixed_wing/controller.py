"""Energy-based nominal controller for fixed-wing MPPI.

Stateless PD controller based on Total Energy Control principles:
  - Outer loop: energy balance → thrust command, flight path angle → pitch command
  - Inner loop: pitch PD → elevator, heading-to-roll PD → aileron, adverse yaw → rudder
  - Curvature-aware heading rate blending for coordinated path following

Used as the nominal policy for MPPI rollouts — provides a baseline
trajectory that MPPI improves upon via sampling.

Control output: [aileron, elevator, throttle, rudder]
  - Elevator output is NOT inverted here — the caller must apply
    u[1] = -elev before passing to the dynamics model.

Frame conventions:
  - Pitch from quaternion: positive = nose down
  - Controller internal: flips to nose-up-positive
  - Gamma: arcsin(vz_global / V)
  - Heading: uses yaw (body heading), not ground track
"""

import numpy as np
import yaml
from pathlib import Path


def _wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class NominalController:
    """Stateless PD controller for fixed-wing MPPI nominal policy."""

    def __init__(self, gains_path=None):
        """Load gains from YAML or use defaults."""

        if gains_path is not None:
            with open(gains_path, 'r') as f:
                p = yaml.safe_load(f)
        else:
            p = {}

        self.g = 9.81
        self.thr_max = 7.5  # Internal scaling (not physical thrust)

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
        self.phi_lim = np.deg2rad(p.get('phi_lim_deg', 55.0))
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

        self._phi_cmd = 0.0

    def compute_thrust_pitch(self, z, ref_gamma, ref_v, ref_accel,
                              V_est, gamma_est, vdot_est):
        """Outer loop: energy balance → thrust and pitch commands.

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
        """Compute control: outer loop + inner loop surfaces.

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

        # Airspeed (body frame) — for energy equations
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
        # Controller internal uses nose-up-POSITIVE
        pitch_actual = -1.0 * pitch

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
        chi = _wrap_pi(yaw)  # body heading, not ground track
        chi_ref = float(ref_heading)
        chi_err = _wrap_pi(chi_ref - chi) * -1.0

        if abs(chi_err) < self.chi_deadband:
            chi_err = 0.0

        path_curvature = ref_data.get('curvature', 0.0)
        Vg = max(V_ground, 0.05)  # Curvature ff and phi_des use ground speed

        # Kinematic yaw rate from body rates
        cos_pitch_safe = max(np.cos(pitch), 0.1)
        chi_dot_raw = (q_rate * np.sin(roll) + r_rate * np.cos(roll)) / cos_pitch_safe

        chi_dot_measured = chi_dot_raw

        # Feedforward from path curvature
        chi_dot_ff = -Vg * path_curvature

        # D-term on heading error rate
        chi_err_dot = chi_dot_measured - chi_dot_ff
        chi_err_dot = np.clip(chi_err_dot, -2.0, 2.0)

        chi_dot_fb = self.k_chi * chi_err + self.k_chi_d * chi_err_dot

        # Non-convex alpha blending
        alpha_max = 1.5
        alpha = np.clip(abs(path_curvature) * self.curvature_blend_gain, 0.3, 1.0)
        chi_dot_des = (alpha_max - alpha) * chi_dot_fb + alpha * chi_dot_ff

        # Desired bank angle
        phi_des = np.arctan2(Vg * chi_dot_des, self.g)
        phi_des = float(np.clip(phi_des, -self.phi_lim, self.phi_lim))

        self._phi_cmd = phi_des

        # Roll PD
        e_phi = _wrap_pi(self._phi_cmd - roll)
        p_clamped = np.clip(p_rate, -2.0, 2.0)

        ail_cmd = (self.trim_ail
                   + self.K_phi_p * e_phi
                   - self.K_phi_d * p_clamped)
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
