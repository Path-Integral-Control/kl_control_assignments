"""Path-following guidance for fixed-wing nominal controller.

Computes heading, flight path angle, speed, and curvature references
from a waypoint path using Frenet frame projection:
  - Nearest-segment projection with signed lateral error
  - Dynamic lookahead point (velocity-dependent L1 distance)
  - Smooth tangent estimation via 4-point Lagrange interpolation
  - Smooth curvature estimation for turn anticipation
  - Altitude tracking with derivative damping
  - Bearing/tangent heading blend (near path → follow tangent, far → steer back)

Produces a ref_data dict for the nominal controller:
  des_heading, des_gamma, des_v, des_a, curvature
"""

import numpy as np


def _wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ============================================================
# Path Sampling Helpers
# ============================================================

def sample_point_on_path(seg_idx, t_in_segment, dist, waypoints):
    """Walk forward along path by arc-length distance from a segment position.


    Parameters
    ----------
    seg_idx : int
        Current segment index.
    t_in_segment : float
        Progress within segment [0, 1].
    dist : float
        Arc-length distance to walk forward (meters).
    waypoints : array (N, 3+)
        Waypoint positions.

    Returns
    -------
    x, y : float
        Position at the sampled point.
    """
    n = len(waypoints)
    wp0 = waypoints[seg_idx]
    wp1 = waypoints[(seg_idx + 1) % n]
    sx, sy = wp1[0] - wp0[0], wp1[1] - wp0[1]
    slen = np.sqrt(sx * sx + sy * sy)

    remaining = (1.0 - t_in_segment) * slen
    if remaining >= dist:
        t = t_in_segment + dist / max(slen, 1e-6)
        return wp0[0] + t * sx, wp0[1] + t * sy

    accumulated = remaining
    cur = (seg_idx + 1) % n

    for _ in range(min(10, n)):
        nxt = (cur + 1) % n
        wc, wn = waypoints[cur], waypoints[nxt]
        dx, dy = wn[0] - wc[0], wn[1] - wc[1]
        dl = np.sqrt(dx * dx + dy * dy)
        if dl < 0.01:
            cur = nxt
            continue
        if accumulated + dl >= dist:
            t = (dist - accumulated) / dl
            return wc[0] + t * dx, wc[1] + t * dy
        accumulated += dl
        cur = nxt
        if cur == seg_idx:
            break

    wf = waypoints[cur]
    return wf[0], wf[1]


def compute_smooth_tangent(seg_idx, t_in_segment, Ld, waypoints):
    """4-point cubic tangent at lookahead distance.

    Samples 4 points at s = {0, Ld/3, 2Ld/3, Ld}, fits cubic,
    returns normalized tangent at s=Ld.

    Lagrange derivative at s=3h:
      f'(3h) = (-2*P0 + 9*P1 - 18*P2 + 11*P3) / (6h)
    """
    h = Ld / 3.0
    px, py = np.zeros(4), np.zeros(4)

    px[0], py[0] = sample_point_on_path(seg_idx, t_in_segment, 0.0, waypoints)
    px[1], py[1] = sample_point_on_path(seg_idx, t_in_segment, h, waypoints)
    px[2], py[2] = sample_point_on_path(seg_idx, t_in_segment, 2.0 * h, waypoints)
    px[3], py[3] = sample_point_on_path(seg_idx, t_in_segment, Ld, waypoints)

    inv_6h = 1.0 / max(6.0 * h, 1e-6)
    tx = (-2.0 * px[0] + 9.0 * px[1] - 18.0 * px[2] + 11.0 * px[3]) * inv_6h
    ty = (-2.0 * py[0] + 9.0 * py[1] - 18.0 * py[2] + 11.0 * py[3]) * inv_6h

    tlen = np.sqrt(tx * tx + ty * ty)
    if tlen > 1e-6:
        return tx / tlen, ty / tlen
    else:
        fdx = px[3] - px[2]
        fdy = py[3] - py[2]
        fdlen = np.sqrt(fdx * fdx + fdy * fdy)
        if fdlen > 1e-6:
            return fdx / fdlen, fdy / fdlen
        return 1.0, 0.0


def compute_smooth_curvature(seg_idx, t_in_segment, sample_distance, waypoints):
    """4-point cubic curvature estimation.

    kappa = (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)
    """
    h = sample_distance / 3.0
    if h < 0.1:
        return 0.0

    px, py = np.zeros(4), np.zeros(4)
    px[0], py[0] = sample_point_on_path(seg_idx, t_in_segment, 0.0, waypoints)
    px[1], py[1] = sample_point_on_path(seg_idx, t_in_segment, h, waypoints)
    px[2], py[2] = sample_point_on_path(seg_idx, t_in_segment, 2.0 * h, waypoints)
    px[3], py[3] = sample_point_on_path(seg_idx, t_in_segment, 3.0 * h, waypoints)

    inv_6h = 1.0 / (6.0 * h)
    dx = (-11.0 * px[0] + 18.0 * px[1] - 9.0 * px[2] + 2.0 * px[3]) * inv_6h
    dy = (-11.0 * py[0] + 18.0 * py[1] - 9.0 * py[2] + 2.0 * py[3]) * inv_6h

    inv_h2 = 1.0 / (h * h)
    ddx = (2.0 * px[0] - 5.0 * px[1] + 4.0 * px[2] - px[3]) * inv_h2
    ddy = (2.0 * py[0] - 5.0 * py[1] + 4.0 * py[2] - py[3]) * inv_h2

    speed_sq = dx * dx + dy * dy
    speed = np.sqrt(speed_sq)
    if speed < 1e-6:
        return 0.0

    return (dx * ddy - dy * ddx) / (speed_sq * speed)


# ============================================================
# Frenet Projection
# ============================================================

def compute_frenet_projection(pos_x, pos_y, pos_z, waypoints):
    """Find nearest segment and compute Frenet state.


    Returns
    -------
    dict with: segment_idx, lateral_error, path_progress,
               tangent_x, tangent_y, projection_x, projection_y
    """
    n = len(waypoints)
    best_dist_sq = 1e9
    best_idx = 0
    best_t = 0.0

    for step in range(n):
        i = step
        j = (i + 1) % n
        wx1, wy1, wz1 = waypoints[i, 0], waypoints[i, 1], waypoints[i, 2]
        wx2, wy2, wz2 = waypoints[j, 0], waypoints[j, 1], waypoints[j, 2]

        sx, sy, sz = wx2 - wx1, wy2 - wy1, wz2 - wz1
        seg_len_sq = sx * sx + sy * sy + sz * sz
        if seg_len_sq < 1e-12:
            continue

        to_x, to_y, to_z = pos_x - wx1, pos_y - wy1, pos_z - wz1
        t_unrestricted = (to_x * sx + to_y * sy + to_z * sz) / seg_len_sq

        # Clamped for segment selection
        t_clamp = _clamp(t_unrestricted, 0.0, 1.0)
        cx = wx1 + t_clamp * sx
        cy = wy1 + t_clamp * sy
        cz = wz1 + t_clamp * sz

        dx, dy, dz = pos_x - cx, pos_y - cy, pos_z - cz
        dist_sq = dx * dx + dy * dy + dz * dz

        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = i
            best_t = t_unrestricted

    # Compute projection
    i = best_idx
    j = (i + 1) % n
    wp1, wp2 = waypoints[i], waypoints[j]
    sx, sy, sz = wp2[0] - wp1[0], wp2[1] - wp1[1], wp2[2] - wp1[2]
    seg_len = np.sqrt(sx * sx + sy * sy + sz * sz)

    proj_x = wp1[0] + best_t * sx
    proj_y = wp1[1] + best_t * sy
    proj_z = wp1[2] + best_t * sz

    # Tangent
    if seg_len > 1e-6:
        tx, ty, tz = sx / seg_len, sy / seg_len, sz / seg_len
    else:
        tx, ty, tz = 1.0, 0.0, 0.0

    # Signed lateral error
    disp_x = pos_x - proj_x
    disp_y = pos_y - proj_y
    cross_z = sx * disp_y - sy * disp_x
    lat_dist = np.sqrt(disp_x ** 2 + disp_y ** 2)
    sign = 1.0 if cross_z >= 0 else -1.0

    return {
        'segment_idx': best_idx,
        'lateral_error': sign * lat_dist,
        'path_progress': float(best_idx) + _clamp(best_t, 0.0, 1.0),
        'tangent_x': tx,
        'tangent_y': ty,
        'tangent_z': tz,
        'projection_x': proj_x,
        'projection_y': proj_y,
        'projection_z': proj_z,
    }


# ============================================================
# Reference Computation (the core guidance logic)
# ============================================================

class PathFollower:
    """Computes TECS reference data from path and vehicle state.

    + computeDesiredGamma() + computeAdaptiveSpeed()

    This is the guidance logic that was essential for convergence.
    """

    def __init__(self, params=None):
        p = params or {}

        # Lookahead
        self.lookahead_time_s = p.get('lookahead_time_s', 0.25)
        self.lookahead_min_m = p.get('lookahead_min_m', 2.0)
        self.lookahead_max_m = p.get('lookahead_max_m', 3.0)
        self.tangent_lookahead_blend = p.get('tangent_lookahead_blend', 0.22)

        # Altitude
        self.K_h = p.get('K_h', 1.0)
        self.K_gamma_d = p.get('K_gamma_d', 1.0)

        # Speed
        self.v_cruise_nominal = p.get('v_cruise_nominal', 6.5)

        # Curvature blending
        self.curvature_blend_distance = p.get('curvature_blend_distance', 3.0)

    def compute_reference(self, pos_x, pos_y, pos_z,
                           vx_w, vy_w, vz_w, forward_speed,
                           waypoints, dt=0.04):
        """Compute TECS ref_data dict from vehicle state and waypoint path.


        Parameters
        ----------
        pos_x, pos_y, pos_z : float
            Vehicle position (world frame).
        vx_w, vy_w, vz_w : float
            Vehicle velocity (world/global frame).
        forward_speed : float
            Body-frame forward velocity (u component).
        waypoints : array (N, 3+)
        dt : float

        Returns
        -------
        ref_data : dict
            Keys: des_heading, des_gamma, des_v, des_a, curvature
        """
        n_wp = len(waypoints)

        # --- Frenet projection ---
        frenet = compute_frenet_projection(pos_x, pos_y, pos_z, waypoints)
        seg_idx = frenet['segment_idx']
        lateral_error = frenet['lateral_error']

        t_in_seg = frenet['path_progress'] - float(seg_idx)
        t_in_seg = _clamp(t_in_seg, 0.0, 1.0)

        V_horz = max(np.sqrt(vx_w ** 2 + vy_w ** 2), 1e-3)

        # --- Step 1: Lookahead point (dynamic, adaptive) ---
        base_Ld = _clamp(V_horz * self.lookahead_time_s,
                         self.lookahead_min_m, self.lookahead_max_m)
        abs_xtrack = abs(lateral_error)

        # Adaptive Ld: larger when on-path, smaller when off-path
        adaptive_Ld = base_Ld * 1.5 / (abs_xtrack + 0.5)
        Ld = _clamp(adaptive_Ld, base_Ld, 3.0 * base_Ld)

        # Sample lookahead point on path
        la_x, la_y = sample_point_on_path(seg_idx, t_in_seg, Ld, waypoints)

        # Path-following bearing: toward lookahead point
        chi_bearing_path = np.arctan2(la_y - pos_y, la_x - pos_x)

        # --- Step 2: Smooth tangent at lookahead ---
        tangent_sample_dist = max(Ld, base_Ld)
        la_tx, la_ty = compute_smooth_tangent(seg_idx, t_in_seg,
                                               tangent_sample_dist, waypoints)

        # Tangent-based bearing: project Ld along tangent
        tangent_la_x = frenet['projection_x'] + Ld * la_tx
        tangent_la_y = frenet['projection_y'] + Ld * la_ty
        chi_bearing_tangent = np.arctan2(tangent_la_y - pos_y, tangent_la_x - pos_x)

        # Blend path-following and tangent bearings
        chi_bearing = _wrap_pi(chi_bearing_path
                               + self.tangent_lookahead_blend
                               * _wrap_pi(chi_bearing_tangent - chi_bearing_path))

        # Tangent heading at lookahead
        chi_tangent = np.arctan2(la_ty, la_tx)

        # Blend: near path → tangent, far from path → bearing
        blend_alpha = 1.0 / (1.0 + abs_xtrack * abs_xtrack * 0.06)
        chi_ref_raw = _wrap_pi(chi_bearing
                               + blend_alpha * _wrap_pi(chi_tangent - chi_bearing))

        # Heading reference from bearing/tangent blend
        des_heading = chi_ref_raw

        # --- Step 4: Desired gamma (altitude tracking with derivative damping) ---
        next_idx = (seg_idx + 1) % n_wp
        next_wp = waypoints[next_idx]
        horz_dist = np.sqrt((next_wp[0] - pos_x) ** 2 + (next_wp[1] - pos_y) ** 2)
        z_err = next_wp[2] - pos_z
        des_gamma = _clamp(self.K_h * z_err / max(horz_dist, 2.0), -0.349, 0.349)

        # Derivative damping: reduce gamma_ref when already climbing/descending
        V_approx = np.sqrt(vx_w ** 2 + vy_w ** 2 + vz_w ** 2)
        if V_approx > 1.0:
            gamma_est = np.arcsin(_clamp(vz_w / V_approx, -1.0, 1.0))
            des_gamma -= self.K_gamma_d * gamma_est
            des_gamma = _clamp(des_gamma, -0.349, 0.349)

        # --- Step 5: Smooth curvature ---
        curvature = compute_smooth_curvature(seg_idx, t_in_seg,
                                              self.curvature_blend_distance,
                                              waypoints)

        # Constant cruise speed
        des_v = self.v_cruise_nominal
        des_a = 0.0

        return {
            'des_heading': des_heading,
            'des_gamma': des_gamma,
            'des_v': des_v,
            'des_a': des_a,
            'curvature': curvature,
        }

    def reset(self):
        """Reset filter states."""
        self._prev_xtrack_dot_filtered = 0.0
        self._xtrack_dot_init = False
