"""
Obstacle randomization for M2.5 depth env.

sample_obstacle_configs — pure numpy, called at Python level before JIT.
min_distance_to_obstacle — JAX-compatible box SDF, available to M3 for collision reward.
"""

from __future__ import annotations
import warnings
import numpy as np
import jax.numpy as jnp

N_OBSTACLE_SLOTS: int = 16   # must match MJCF body count in crazyflie_depth.xml

_ARENA_XY    = 3.0            # obstacle xy drawn from U(-3, 3)
_SPAWN_EXCL  = 0.5            # min xy distance from (0,0) to any active obstacle center
_INTER_EXCL  = 0.4            # min center-to-center xy distance between any two obstacles
_MAX_RETRIES = 50
_DEFAULT_HE  = np.array([0.05, 0.05, 0.5], dtype=np.float32)   # pillar half-extents


def sample_obstacle_configs(
    rng: np.random.Generator,
    n_obstacles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample one episode's obstacle layout.

    Returns (centers, half_extents), both shape (N_OBSTACLE_SLOTS=16, 3) float32.
    Active rows [0..n_obstacles-1] are placed inside the arena.
    Inactive rows are parked at (100, 100, 100) / zeros so they stay outside the
    scene and frustum without being removed from the static MJCF structure.
    """
    if not (0 <= n_obstacles <= N_OBSTACLE_SLOTS):
        raise ValueError(f"n_obstacles must be in [0, {N_OBSTACLE_SLOTS}], got {n_obstacles}")

    centers      = np.empty((N_OBSTACLE_SLOTS, 3), dtype=np.float32)
    half_extents = np.zeros((N_OBSTACLE_SLOTS, 3), dtype=np.float32)

    # Park inactive slots out of scene
    centers[n_obstacles:] = [100.0, 100.0, 100.0]

    placed_xy: list[np.ndarray] = []

    for i in range(n_obstacles):
        xy = _sample_xy(rng, placed_xy, i)
        placed_xy.append(xy)
        centers[i]      = [xy[0], xy[1], 1.0]
        half_extents[i] = _DEFAULT_HE

    return centers, half_extents


def _sample_xy(
    rng: np.random.Generator,
    placed_xy: list[np.ndarray],
    slot_idx: int,
) -> np.ndarray:
    """Rejection-sample an obstacle xy position satisfying spawn + inter-obstacle exclusions."""
    # Fast path: try satisfying both constraints together
    for _ in range(_MAX_RETRIES):
        xy = rng.uniform(-_ARENA_XY, _ARENA_XY, size=2).astype(np.float32)
        if np.linalg.norm(xy) < _SPAWN_EXCL:
            continue
        if any(np.linalg.norm(xy - p) < _INTER_EXCL for p in placed_xy):
            continue
        return xy

    # Fallback: spawn exclusion is hard (safety); inter-obstacle is soft (warn + accept)
    warnings.warn(
        f"obstacle_{slot_idx}: inter-obstacle exclusion ({_INTER_EXCL} m) "
        f"unmet after {_MAX_RETRIES} retries; accepting without it. "
        f"This is expected only when n_obstacles is large relative to arena size.",
        stacklevel=3,
    )
    for _ in range(_MAX_RETRIES * 10):
        xy = rng.uniform(-_ARENA_XY, _ARENA_XY, size=2).astype(np.float32)
        if np.linalg.norm(xy) >= _SPAWN_EXCL:
            return xy

    raise RuntimeError(
        f"obstacle_{slot_idx}: failed to satisfy spawn exclusion ({_SPAWN_EXCL} m) "
        f"after {_MAX_RETRIES * 10} retries — arena geometry may be degenerate."
    )


def min_distance_to_obstacle(
    drone_pos: jnp.ndarray,
    centers: jnp.ndarray,
    half_extents: jnp.ndarray,
    n_obstacles: int,
) -> jnp.ndarray:
    """
    Minimum L∞ surface-to-surface distance from drone_pos to the nearest active obstacle.

    Box SDF per obstacle: max(0, max_axis(|drone_pos - center| - half_extents)).
    Returns jnp.inf when n_obstacles == 0.

    n_obstacles is a Python int (static in JIT). centers/half_extents may be numpy or JAX
    arrays; converted internally.
    """
    if n_obstacles == 0:
        return jnp.inf

    pos = jnp.asarray(drone_pos)                      # (3,)
    c   = jnp.asarray(centers[:n_obstacles])           # (n, 3)
    he  = jnp.asarray(half_extents[:n_obstacles])      # (n, 3)

    diff  = jnp.abs(pos - c) - he                     # (n, 3)
    dists = jnp.max(jnp.maximum(diff, 0.0), axis=1)   # (n,)  L∞ SDF per obstacle
    return jnp.min(dists)
