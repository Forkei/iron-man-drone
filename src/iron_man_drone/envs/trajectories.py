"""
Reference trajectory generators for SimpleFlight M1.

Lazy design: store polynomial coefficients per env, evaluate on demand.
This avoids the 1050-point vmap on every env reset that caused 17x throughput loss.

Training: random C2-continuous quintic polynomial + random infeasible zigzag (50/50 mix).
Eval:     figure-eight (slow/normal/fast), pentagram (slow/fast) — OOD held-out.

All functions are pure JAX — safe to vmap over environments.

Polynomial trajectory (M1.3+):
  Each segment is a true 5th-degree polynomial in time for x and y, with 6 boundary
  conditions (pos, vel, acc at start and end). Interior waypoints have nonzero velocity
  and acceleration (C2 continuity) — the drone never stops mid-trajectory.

  This replaces the broken M1/M1.2 implementation that applied a quintic scalar h(τ)
  to straight-line segments, giving κ=0 everywhere (stop-pivot-go).
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import NamedTuple

# Trajectory type codes
TRAJ_POLY           = 0
TRAJ_ZIGZAG         = 1
TRAJ_FIGURE8_SLOW   = 2
TRAJ_FIGURE8_NORMAL = 3
TRAJ_FIGURE8_FAST   = 4
TRAJ_PENTAGRAM_SLOW = 5
TRAJ_PENTAGRAM_FAST = 6

# Max piecewise segments: poly ≤ 9 segs, zigzag ≤ 12 segs at 10.5s episode.
MAX_SEGS = 14

DT = 0.01  # 100 Hz — must match SIM_FREQ in quadrotor_env.py


class Trajectory(NamedTuple):
    """
    Lazy trajectory — stores parameters, evaluates position on demand.

    TRAJ_POLY:
      poly_coeffs[i] = (6, 3) polynomial coefficients for segment i.
      p(tau) = sum_k coeffs[k] * tau^k, where tau = t - cum_times[i].
      cum_times[i] = start time of segment i; +inf-padded beyond last segment.
      waypoints unused (zeros).

    TRAJ_ZIGZAG:
      waypoints[i], waypoints[i+1] = start/end of segment i.
      cum_times[i] = start time of segment i.
      poly_coeffs unused (zeros).

    Analytic types (figure-eight, pentagram):
      waypoints, cum_times, poly_coeffs all zeros (unused).
    """
    waypoints:   jnp.ndarray   # (MAX_SEGS + 1, 3)  — zigzag waypoints
    cum_times:   jnp.ndarray   # (MAX_SEGS + 1,)    — segment start times, +inf-padded
    traj_type:   jnp.ndarray   # ()  int32
    total_time:  jnp.ndarray   # ()  float32
    poly_coeffs: jnp.ndarray   # (MAX_SEGS, 6, 3)   — quintic coefficients per segment


# ---------------------------------------------------------------------------
# Quintic polynomial helper
# ---------------------------------------------------------------------------

def _solve_quintic_coeffs(
    p0: jnp.ndarray, v0: jnp.ndarray, a0: jnp.ndarray,
    p1: jnp.ndarray, v1: jnp.ndarray, a1: jnp.ndarray,
    T:  jnp.ndarray,
) -> jnp.ndarray:
    """
    Closed-form coefficients for p(tau) = c[0] + c[1]*tau + ... + c[5]*tau^5
    satisfying the 6 boundary conditions:
      p(0) = p0, p'(0) = v0, p''(0) = a0
      p(T) = p1, p'(T) = v1, p''(T) = a1

    Works on scalars or batched arrays (broadcast applies).
    Returns shape (..., 6) where ... matches the input shape.
    """
    T  = jnp.maximum(T, 1e-6)
    c0 = p0
    c1 = v0
    c2 = a0 * 0.5

    d1 = p1 - p0 - v0 * T - a0 * T**2 * 0.5
    d2 = v1 - v0 - a0 * T
    d3 = a1 - a0

    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    c3 = 10.0 * d1 / T3 - 4.0 * d2 / T2 + d3 / (2.0 * T)
    c4 = -15.0 * d1 / T4 + 7.0 * d2 / T3 - d3 / T2
    c5 = 6.0 * d1 / T5 - 3.0 * d2 / T4 + d3 / (2.0 * T3)

    return jnp.stack([c0, c1, c2, c3, c4, c5], axis=0)  # (6, ...) → caller transposes if needed


# ---------------------------------------------------------------------------
# Core: evaluate position at arbitrary time t
# ---------------------------------------------------------------------------

def eval_trajectory_position(traj: Trajectory, t: jnp.ndarray) -> jnp.ndarray:
    """Return world-frame (x, y, z) position at time t (seconds). Shape: (3,)."""
    t = jnp.clip(jnp.asarray(t, dtype=jnp.float32), 0.0, traj.total_time)

    def _seg(cum_times, t):
        idx = jnp.searchsorted(cum_times, t, side="right") - 1
        return jnp.clip(idx, 0, MAX_SEGS - 1)

    def eval_poly(args):
        traj, t = args
        i   = _seg(traj.cum_times, t)
        t0  = traj.cum_times[i]
        tau = t - t0                           # local time within segment
        c   = traj.poly_coeffs[i]             # (6, 3)
        # Horner's method: p = c[0] + tau*(c[1] + tau*(c[2] + ...))
        p = c[5]
        p = c[4] + tau * p
        p = c[3] + tau * p
        p = c[2] + tau * p
        p = c[1] + tau * p
        p = c[0] + tau * p
        return p                               # (3,)

    def eval_zigzag(args):
        traj, t = args
        i  = _seg(traj.cum_times, t)
        t0 = traj.cum_times[i]
        t1 = traj.cum_times[i + 1]
        alpha = jnp.clip((t - t0) / jnp.maximum(t1 - t0, 1e-9), 0.0, 1.0)
        p0, p1 = traj.waypoints[i], traj.waypoints[i + 1]
        return p0 + alpha * (p1 - p0)

    def eval_f8_slow(args):
        _, t = args
        T = 15.0
        return jnp.array([jnp.cos(2*jnp.pi*t/T), jnp.sin(4*jnp.pi*t/T)/2.0, 1.0])

    def eval_f8_normal(args):
        _, t = args
        T = 5.5
        return jnp.array([jnp.cos(2*jnp.pi*t/T), jnp.sin(4*jnp.pi*t/T)/2.0, 1.0])

    def eval_f8_fast(args):
        _, t = args
        T = 3.5
        return jnp.array([jnp.cos(2*jnp.pi*t/T), jnp.sin(4*jnp.pi*t/T)/2.0, 1.0])

    def eval_penta_slow(args):
        _, t = args
        return _pentagram_at_t(t, 0.5)

    def eval_penta_fast(args):
        _, t = args
        return _pentagram_at_t(t, 1.0)

    return jax.lax.switch(
        traj.traj_type,
        [eval_poly, eval_zigzag,
         eval_f8_slow, eval_f8_normal, eval_f8_fast,
         eval_penta_slow, eval_penta_fast],
        (traj, t),
    )


def _pentagram_at_t(t: jnp.ndarray, speed_mps: float) -> jnp.ndarray:
    """Star-pentagon position at time t; loops periodically."""
    vertex_order = jnp.array([0, 2, 4, 1, 3], dtype=jnp.int32)
    angles = 2.0 * jnp.pi * vertex_order / 5.0 - jnp.pi / 2.0
    vx = jnp.cos(angles)
    vy = jnp.sin(angles)
    vertices  = jnp.stack([vx, vy, jnp.ones(5)], axis=-1)      # (5, 3)
    seg_vecs  = jnp.roll(vertices, -1, axis=0) - vertices        # (5, 3)
    seg_lens  = jnp.linalg.norm(seg_vecs[:, :2], axis=-1)        # (5,)
    total_len = seg_lens.sum()
    T         = total_len / speed_mps

    t_frac   = (t % T) / T
    cum_frac = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lens / total_len)])  # (6,)

    seg_idx  = jnp.searchsorted(cum_frac[1:], t_frac, side="right")
    seg_idx  = jnp.clip(seg_idx, 0, 4)
    seg_s    = cum_frac[seg_idx]
    seg_e    = cum_frac[seg_idx + 1]
    alpha    = jnp.clip((t_frac - seg_s) / jnp.maximum(seg_e - seg_s, 1e-9), 0.0, 1.0)
    return vertices[seg_idx] + alpha * seg_vecs[seg_idx]


# ---------------------------------------------------------------------------
# Reference accessors (used by quadrotor_env._build_obs and _compute_reward)
# ---------------------------------------------------------------------------

def get_reference_window(
    traj: Trajectory,
    step: jnp.ndarray,
    lookahead_n: int = 10,
    lookahead_steps_per_point: int = 5,
) -> jnp.ndarray:
    """
    Returns the next `lookahead_n` reference positions, each `lookahead_steps_per_point`
    steps (50 ms) ahead of the previous one, starting one spacing ahead of `step`.
    Shape: (lookahead_n, 3).
    """
    t_base  = jnp.asarray(step, dtype=jnp.float32) * DT
    offsets = jnp.arange(1, lookahead_n + 1) * (lookahead_steps_per_point * DT)
    times   = t_base + offsets
    return jax.vmap(lambda t: eval_trajectory_position(traj, t))(times)


def get_reference_pos(traj: Trajectory, step: jnp.ndarray) -> jnp.ndarray:
    """Current reference position (3,) at episode step `step`."""
    t = jnp.asarray(step, dtype=jnp.float32) * DT
    return eval_trajectory_position(traj, t)


# ---------------------------------------------------------------------------
# Random training trajectory constructors
# ---------------------------------------------------------------------------

def sample_polynomial_trajectory(
    key: jnp.ndarray,
    dt: float,
    total_steps: int,
    lookahead_steps: int = 50,
    seg_duration: tuple = (1.5, 4.0),
    height: float = 1.0,
    max_vel: float = 0.8,
    max_acc: float = 2.0,
) -> Trajectory:
    """
    C2-continuous random quintic polynomial trajectory.

    Each segment is a 5th-degree polynomial in time, with random nonzero velocities
    and accelerations at interior waypoints (C2 continuity). The drone moves through
    waypoints without stopping, generating meaningful curvature throughout.

    max_vel: max speed at interior waypoints [m/s] (paper states 0–1 m/s)
    max_acc: max acceleration at interior waypoints [m/s²]
    """
    total_time = (total_steps + lookahead_steps) * dt
    n_seg = int(total_time / seg_duration[0]) + 2

    assert n_seg <= MAX_SEGS, (
        f"sample_polynomial_trajectory: n_seg={n_seg} exceeds MAX_SEGS={MAX_SEGS}. "
        f"Reduce seg_duration[0] or increase MAX_SEGS."
    )

    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    wp_xy   = jax.random.uniform(k1, (n_seg + 1, 2), minval=-1.5, maxval=1.5)
    durs    = jax.random.uniform(k2, (n_seg,), minval=seg_duration[0], maxval=seg_duration[1])
    vels_xy = jax.random.uniform(k3, (n_seg + 1, 2), minval=-max_vel, maxval=max_vel)
    accs_xy = jax.random.uniform(k4, (n_seg + 1, 2), minval=-max_acc, maxval=max_acc)

    # Hover at episode start and end
    vels_xy = vels_xy.at[0].set(0.0)
    vels_xy = vels_xy.at[-1].set(0.0)
    accs_xy = accs_xy.at[0].set(0.0)
    accs_xy = accs_xy.at[-1].set(0.0)

    # Solve per-segment quintic coefficients for x and y via closed-form formula.
    # _solve_quintic_coeffs with (2,) inputs returns (6, 2).
    # _solve_quintic_coeffs with (2,) inputs: stack([c0..c5], axis=0) → (6, 2).
    # vmap over n_seg segments → (n_seg, 6, 2).
    coeffs_xy = jax.vmap(
        lambda p0, v0, a0, p1, v1, a1, T:
            _solve_quintic_coeffs(p0, v0, a0, p1, v1, a1, T)
    )(
        wp_xy[:n_seg], vels_xy[:n_seg], accs_xy[:n_seg],
        wp_xy[1:n_seg+1], vels_xy[1:n_seg+1], accs_xy[1:n_seg+1],
        durs,
    )

    # z: constant altitude — [height, 0, 0, 0, 0, 0] per segment
    coeffs_z = jnp.zeros((n_seg, 6, 1)).at[:, 0, 0].set(height)

    # Concatenate to (n_seg, 6, 3) and pad to (MAX_SEGS, 6, 3)
    coeffs_xyz = jnp.concatenate([coeffs_xy, coeffs_z], axis=2)
    n_pad = MAX_SEGS - n_seg
    coeffs_pad = jnp.concatenate([coeffs_xyz, jnp.zeros((n_pad, 6, 3))], axis=0)

    # Cumulative segment start times, +inf-padded
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durs)])
    n_cum_pad = MAX_SEGS + 1 - (n_seg + 1)
    cum_pad = jnp.concatenate([cum, jnp.full((n_cum_pad,), jnp.inf)])

    return Trajectory(
        waypoints=jnp.zeros((MAX_SEGS + 1, 3)),
        cum_times=cum_pad,
        traj_type=jnp.array(TRAJ_POLY, dtype=jnp.int32),
        total_time=jnp.array(total_steps * dt, dtype=jnp.float32),
        poly_coeffs=coeffs_pad,
    )


def sample_zigzag_trajectory(
    key: jnp.ndarray,
    dt: float,
    total_steps: int,
    lookahead_steps: int = 50,
    seg_duration: tuple = (1.0, 1.5),
    height: float = 1.0,
) -> Trajectory:
    """
    Random infeasible zigzag: waypoints in [-1,1]^2, connected by straight lines.
    Infeasible because waypoint corners require infinite acceleration.
    """
    total_time = (total_steps + lookahead_steps) * dt
    n_seg = int(total_time / seg_duration[0]) + 2

    assert n_seg <= MAX_SEGS, (
        f"sample_zigzag_trajectory: n_seg={n_seg} exceeds MAX_SEGS={MAX_SEGS}. "
        f"Increase MAX_SEGS or seg_duration[0]."
    )

    key, k1, k2 = jax.random.split(key, 3)
    wp_xy = jax.random.uniform(k1, (n_seg + 1, 2), minval=-1.0, maxval=1.0)
    durs  = jax.random.uniform(k2, (n_seg,), minval=seg_duration[0], maxval=seg_duration[1])
    cum   = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durs)])
    wp    = jnp.concatenate([wp_xy, jnp.full((n_seg + 1, 1), height)], axis=1)

    n_pad   = MAX_SEGS + 1 - (n_seg + 1)
    wp_pad  = jnp.concatenate([wp,  jnp.zeros((n_pad, 3))],       axis=0)
    cum_pad = jnp.concatenate([cum, jnp.full((n_pad,), jnp.inf)], axis=0)

    return Trajectory(
        waypoints=wp_pad,
        cum_times=cum_pad,
        traj_type=jnp.array(TRAJ_ZIGZAG, dtype=jnp.int32),
        total_time=jnp.array(total_steps * dt, dtype=jnp.float32),
        poly_coeffs=jnp.zeros((MAX_SEGS, 6, 3)),
    )


# ---------------------------------------------------------------------------
# Analytic eval trajectory constructors
# ---------------------------------------------------------------------------

def make_figure_eight_trajectory(
    dt: float,
    total_steps: int,
    lookahead_steps: int = 50,
    speed: str = "normal",
) -> Trajectory:
    """Analytic figure-eight for evaluation (OOD)."""
    _type = {"slow": TRAJ_FIGURE8_SLOW, "normal": TRAJ_FIGURE8_NORMAL, "fast": TRAJ_FIGURE8_FAST}
    return Trajectory(
        waypoints=jnp.zeros((MAX_SEGS + 1, 3)),
        cum_times=jnp.zeros((MAX_SEGS + 1,)),
        traj_type=jnp.array(_type[speed], dtype=jnp.int32),
        total_time=jnp.array(total_steps * dt, dtype=jnp.float32),
        poly_coeffs=jnp.zeros((MAX_SEGS, 6, 3)),
    )


def make_pentagram_trajectory(
    dt: float,
    total_steps: int,
    lookahead_steps: int = 50,
    speed: str = "slow",
) -> Trajectory:
    """Analytic pentagram for evaluation (OOD)."""
    _type = {"slow": TRAJ_PENTAGRAM_SLOW, "fast": TRAJ_PENTAGRAM_FAST}
    return Trajectory(
        waypoints=jnp.zeros((MAX_SEGS + 1, 3)),
        cum_times=jnp.zeros((MAX_SEGS + 1,)),
        traj_type=jnp.array(_type[speed], dtype=jnp.int32),
        total_time=jnp.array(total_steps * dt, dtype=jnp.float32),
        poly_coeffs=jnp.zeros((MAX_SEGS, 6, 3)),
    )
