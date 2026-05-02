"""
Reference trajectory generators for SimpleFlight M1.

Training uses: random smooth polynomial + random infeasible zigzag (50/50 mix).
Eval uses:     figure-eight (slow/normal/fast), pentagram (slow/fast),
               held-out polynomial, held-out zigzag.

Figure-eight and pentagram are explicitly held-out OOD test cases.
All functions are pure JAX — safe to vmap over environments.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import NamedTuple


class Trajectory(NamedTuple):
    """A reference trajectory callable via position(t) and velocity(t)."""
    # Precomputed waypoints: (max_steps + lookahead, 3)
    positions: jnp.ndarray
    # Total duration in seconds
    duration: float


# ---------------------------------------------------------------------------
# Analytic trajectories (eval / OOD)
# ---------------------------------------------------------------------------

def figure_eight_positions(t: jnp.ndarray, speed: str = "normal") -> jnp.ndarray:
    """
    Figure-eight trajectory: p(t) = [cos(2π t/T), sin(4π t/T)/2, 1.0]
    Speed: slow T=15s, normal T=5.5s, fast T=3.5s
    Returns position (3,) at time t.
    """
    T = {"slow": 15.0, "normal": 5.5, "fast": 3.5}[speed]
    x = jnp.cos(2.0 * jnp.pi * t / T)
    y = jnp.sin(4.0 * jnp.pi * t / T) / 2.0
    z = jnp.ones_like(t)
    return jnp.stack([x, y, z], axis=-1)


def pentagram_positions(t: jnp.ndarray, speed: str = "slow") -> jnp.ndarray:
    """
    Pentagram trajectory: 5-vertex star polygon, constant velocity.
    Vertices at angles 2π*k/5 for k=0..4, radius=1.0m, height=1.0m.
    Speed: slow 0.5 m/s, fast 1.0 m/s (determines period T).
    """
    # Pentagram vertices (order: 0 -> 2 -> 4 -> 1 -> 3 -> 0 for star pattern)
    vertex_order = jnp.array([0, 2, 4, 1, 3], dtype=jnp.int32)
    angles = 2.0 * jnp.pi * vertex_order / 5.0 - jnp.pi / 2.0  # start at top
    radius = 1.0
    vertices = jnp.stack([
        radius * jnp.cos(angles),
        radius * jnp.sin(angles),
        jnp.ones(5),
    ], axis=-1)  # (5, 3)

    # Segment lengths
    seg_vecs = jnp.roll(vertices, -1, axis=0) - vertices  # (5, 3)
    seg_lens = jnp.linalg.norm(seg_vecs[:, :2], axis=-1)  # horizontal length
    total_len = seg_lens.sum()

    vel = {"slow": 0.5, "fast": 1.0}[speed]
    T = total_len / vel

    # Find which segment we're on at time t
    t_frac = (t % T) / T  # [0, 1)
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lens / total_len)])

    def interp_segment(seg_idx):
        seg_start = cumulative[seg_idx]
        seg_end = cumulative[seg_idx + 1]
        alpha = (t_frac - seg_start) / (seg_end - seg_start + 1e-9)
        alpha = jnp.clip(alpha, 0.0, 1.0)
        return vertices[seg_idx] + alpha * seg_vecs[seg_idx]

    seg_idx = jnp.searchsorted(cumulative[1:], t_frac, side="right")
    seg_idx = jnp.clip(seg_idx, 0, 4)
    # Use vmap-friendly conditional
    pos = jax.lax.switch(seg_idx, [interp_segment(i) for i in range(5)], seg_idx)
    return pos


def precompute_trajectory(
    traj_fn,
    dt: float,
    total_steps: int,
    lookahead_steps: int = 50,
) -> Trajectory:
    """
    Precompute a trajectory as an array of positions.
    Returns (total_steps + lookahead_steps, 3) array.
    """
    t = jnp.arange(total_steps + lookahead_steps) * dt
    positions = jax.vmap(traj_fn)(t)
    return Trajectory(positions=positions, duration=total_steps * dt)


# ---------------------------------------------------------------------------
# Random trajectory generators (training distribution)
# ---------------------------------------------------------------------------

def sample_polynomial_trajectory(
    key: jnp.ndarray,
    dt: float,
    total_steps: int,
    lookahead_steps: int = 50,
    max_vel: float = 1.0,
    seg_duration: tuple = (1.5, 4.0),
    height: float = 1.0,
) -> Trajectory:
    """
    Random smooth polynomial trajectory (5th-degree).
    Segments of random duration in seg_duration range.
    Continuity of pos/vel/acc at junctions.
    Stays within 2m of origin, height=1m.
    """
    total_time = (total_steps + lookahead_steps) * dt
    n_seg_max = int(total_time / seg_duration[0]) + 2

    key, k1, k2, k3 = jax.random.split(key, 4)

    # Sample random waypoints and durations
    waypoints = jax.random.uniform(k1, (n_seg_max + 1, 2), minval=-1.5, maxval=1.5)
    durations = jax.random.uniform(k2, (n_seg_max,), minval=seg_duration[0], maxval=seg_duration[1])

    # Cumulative times
    cum_times = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durations)])

    def poly5(tau, p0, p1, v0=None, v1=None, a0=None, a1=None):
        """5th-degree polynomial from p0 to p1 over tau in [0, 1]."""
        if v0 is None: v0 = jnp.zeros_like(p0)
        if v1 is None: v1 = jnp.zeros_like(p0)
        if a0 is None: a0 = jnp.zeros_like(p0)
        if a1 is None: a1 = jnp.zeros_like(p0)

        c0 = p0
        c1 = v0
        c2 = a0 / 2.0
        c3 = 10*(p1-p0) - 6*v0 - 4*v1 - 3*a0 + 1.5*a1
        c4 = -15*(p1-p0) + 8*v0 + 7*v1 + 3*a0 - 2*a1
        c5 = 6*(p1-p0) - 3*v0 - 3*v1 - a0 + a1
        return c0 + c1*tau + c2*tau**2 + c3*tau**3 + c4*tau**4 + c5*tau**5

    # Evaluate at each timestep
    t_eval = jnp.arange(total_steps + lookahead_steps) * dt

    def eval_at_t(t):
        seg_idx = jnp.searchsorted(cum_times[1:], t, side="right")
        seg_idx = jnp.clip(seg_idx, 0, n_seg_max - 1)
        T = durations[seg_idx]
        t0 = cum_times[seg_idx]
        tau = (t - t0) / (T + 1e-9)
        tau = jnp.clip(tau, 0.0, 1.0)
        p0 = waypoints[seg_idx]
        p1 = waypoints[seg_idx + 1]
        xy = poly5(tau, p0, p1)
        return jnp.array([xy[0], xy[1], height])

    positions = jax.vmap(eval_at_t)(t_eval)
    return Trajectory(positions=positions, duration=total_steps * dt)


def sample_zigzag_trajectory(
    key: jnp.ndarray,
    dt: float,
    total_steps: int,
    lookahead_steps: int = 50,
    max_vel: float = 2.0,
    seg_duration: tuple = (1.0, 1.5),
    height: float = 1.0,
) -> Trajectory:
    """
    Random infeasible zigzag trajectory.
    Waypoints in [-1, 1]^2, connected by straight lines.
    Max velocity ~2 m/s, time intervals 1-1.5s.
    Infeasible because direction changes require infinite acceleration at waypoints.
    """
    total_time = (total_steps + lookahead_steps) * dt
    n_wp = int(total_time / seg_duration[0]) + 2

    key, k1, k2 = jax.random.split(key, 3)
    waypoints = jax.random.uniform(k1, (n_wp + 1, 2), minval=-1.0, maxval=1.0)
    durations = jax.random.uniform(k2, (n_wp,), minval=seg_duration[0], maxval=seg_duration[1])
    cum_times = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durations)])

    t_eval = jnp.arange(total_steps + lookahead_steps) * dt

    def eval_at_t(t):
        seg_idx = jnp.searchsorted(cum_times[1:], t, side="right")
        seg_idx = jnp.clip(seg_idx, 0, n_wp - 1)
        T = durations[seg_idx]
        t0 = cum_times[seg_idx]
        alpha = (t - t0) / (T + 1e-9)
        alpha = jnp.clip(alpha, 0.0, 1.0)
        xy = waypoints[seg_idx] * (1 - alpha) + waypoints[seg_idx + 1] * alpha
        return jnp.array([xy[0], xy[1], height])

    positions = jax.vmap(eval_at_t)(t_eval)
    return Trajectory(positions=positions, duration=total_steps * dt)


def get_reference_window(
    traj: Trajectory,
    step: int,
    lookahead_n: int = 10,
    lookahead_steps_per_point: int = 5,  # 5 steps = 50ms at 100Hz
) -> jnp.ndarray:
    """
    Returns the next `lookahead_n` reference positions spaced `lookahead_steps_per_point`
    apart, starting from current step.
    Shape: (lookahead_n, 3)
    """
    indices = step + jnp.arange(1, lookahead_n + 1) * lookahead_steps_per_point
    indices = jnp.clip(indices, 0, traj.positions.shape[0] - 1)
    return traj.positions[indices]


def get_reference_pos(traj: Trajectory, step: int) -> jnp.ndarray:
    """Current reference position (3,)."""
    idx = jnp.clip(step, 0, traj.positions.shape[0] - 1)
    return traj.positions[idx]
