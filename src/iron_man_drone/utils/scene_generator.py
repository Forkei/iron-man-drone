"""
Multi-mode procedural obstacle generator for M3.

Five modes: forest / urban / random / slalom / hallway
Training distribution: forest, urban, random, slalom (equal weight by default)
OOD holdout: hallway (not used in training)

All modes return (centers, half_extents) of shape (N_OBSTACLE_SLOTS=16, 3)
compatible with DepthVecEnv.batch_reset.

Geom sizes in the MJCF are fixed (0.05 × 0.05 × 0.5 m pillars). half_extents
returned here match the MJCF geometry so the L∞ SDF in min_distance_to_obstacle
matches the rendered depth. Modes differ in placement pattern, not geom size.

Curriculum: pass density_mult ∈ [0, 1.5] to scale n_obstacles within each
mode's range. Typical schedule: 0.5 → 1.0 over first 50M env-steps.
"""

from __future__ import annotations
import warnings
import numpy as np
from .obstacle_randomization import N_OBSTACLE_SLOTS, _SPAWN_EXCL, _MAX_RETRIES

# Half-extents matching crazyflie_depth.xml geom size="0.05 0.05 0.5"
_PILLAR_HE = np.array([0.05, 0.05, 0.5], dtype=np.float32)

# Arena bounds
_ARENA_XY   = 4.0   # M3 arena is larger than M2.5's 3 m (spec: ±5 m)
_INTER_EXCL = 0.3   # min center-to-center xy distance between obstacles

TRAINING_MODES = ("forest", "urban", "random", "slalom")
HOLDOUT_MODES  = ("hallway",)
TRAINING_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

# Per-mode obstacle count range at density_mult=1.0
_MODE_N_RANGE = {
    "forest":  (6, 12),
    "urban":   (4, 8),
    "random":  (4, 8),
    "slalom":  (4, 8),
    "hallway": (8, 16),  # 4-8 segments per wall, 2 walls
}


def sample_mode(rng: np.random.Generator, modes=TRAINING_MODES, weights=TRAINING_WEIGHTS) -> str:
    """Sample a scene mode from the training distribution."""
    return rng.choice(list(modes), p=weights / weights.sum())


def sample_scene(
    rng: np.random.Generator,
    mode: str,
    density_mult: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample one episode's obstacle layout for the given mode.

    Returns (centers, half_extents), both shape (N_OBSTACLE_SLOTS=16, 3) float32.
    Active rows [0..n-1] are placed; inactive rows parked at (100, 100, 100).
    density_mult scales n_obstacles: 0.5 = sparse, 1.0 = normal, 1.5 = stress.
    """
    if mode == "forest":
        return _sample_forest(rng, density_mult)
    elif mode == "urban":
        return _sample_urban(rng, density_mult)
    elif mode == "random":
        return _sample_random(rng, density_mult)
    elif mode == "slalom":
        return _sample_slalom(rng, density_mult)
    elif mode == "hallway":
        return _sample_hallway(rng, density_mult)
    else:
        raise ValueError(f"Unknown scene mode: {mode!r}. Valid: {list(_MODE_N_RANGE)}")


# ── Mode implementations ───────────────────────────────────────────────────────

def _sample_forest(rng, density_mult):
    """
    Dense thin pillars — random xy placement throughout arena.
    Visual character: obstacle field drone must thread through.
    """
    lo, hi = _MODE_N_RANGE["forest"]
    n = _scale_n(rng, lo, hi, density_mult)

    centers, half_extents = _init_slots()
    placed_xy = []
    for i in range(n):
        xy = _sample_xy_free(rng, placed_xy, _SPAWN_EXCL, _INTER_EXCL, _ARENA_XY, i)
        placed_xy.append(xy)
        z = rng.uniform(0.8, 1.5)   # pillar center height
        centers[i]      = [xy[0], xy[1], z]
        half_extents[i] = _PILLAR_HE

    return centers, half_extents


def _sample_urban(rng, density_mult):
    """
    Grid of pillars with random jitter — simulates building-grid layout.
    Visual character: regular grid structure with navigable corridors.
    """
    lo, hi = _MODE_N_RANGE["urban"]
    n = _scale_n(rng, lo, hi, density_mult)

    # Build a soft grid: place on grid nodes with ±0.8m jitter
    grid_spacing = _ARENA_XY * 2 / 3.0   # ~2.67 m spacing for a 3x3 grid
    grid_coords = []
    for gx in [-1, 0, 1]:
        for gy in [-1, 0, 1]:
            grid_coords.append((gx * grid_spacing, gy * grid_spacing))
    rng.shuffle(grid_coords)

    centers, half_extents = _init_slots()
    placed_xy = []
    for i in range(min(n, len(grid_coords))):
        gx, gy = grid_coords[i]
        for _ in range(20):
            jitter = rng.uniform(-0.8, 0.8, size=2)
            xy = np.array([gx + jitter[0], gy + jitter[1]], dtype=np.float32)
            xy = np.clip(xy, -_ARENA_XY + 0.3, _ARENA_XY - 0.3)
            if np.linalg.norm(xy) < _SPAWN_EXCL:
                continue
            if any(np.linalg.norm(xy - p) < _INTER_EXCL for p in placed_xy):
                continue
            placed_xy.append(xy)
            z = rng.uniform(0.8, 1.5)
            centers[i]      = [xy[0], xy[1], z]
            half_extents[i] = _PILLAR_HE
            break

    return centers, half_extents


def _sample_random(rng, density_mult):
    """
    Fully random placement — training-distribution baseline.
    Same logic as M2.5's sample_obstacle_configs, with M3's larger arena.
    """
    lo, hi = _MODE_N_RANGE["random"]
    n = _scale_n(rng, lo, hi, density_mult)

    centers, half_extents = _init_slots()
    placed_xy = []
    for i in range(n):
        xy = _sample_xy_free(rng, placed_xy, _SPAWN_EXCL, _INTER_EXCL, _ARENA_XY, i)
        placed_xy.append(xy)
        z = rng.uniform(0.7, 1.6)
        centers[i]      = [xy[0], xy[1], z]
        half_extents[i] = _PILLAR_HE

    return centers, half_extents


def _sample_slalom(rng, density_mult):
    """
    Alternating left/right gate posts along a random corridor direction.
    Visual character: structured zigzag pattern.
    """
    lo, hi = _MODE_N_RANGE["slalom"]
    n = _scale_n(rng, lo, hi, density_mult)

    # Corridor direction: random angle in [-45°, 45°] from y-axis
    angle = rng.uniform(-np.pi / 4, np.pi / 4)
    fwd   = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
    side  = np.array([np.cos(angle), -np.sin(angle)], dtype=np.float32)

    spacing  = rng.uniform(2.0, 3.5)   # along-corridor spacing between gates
    offset_m = rng.uniform(0.8, 1.5)   # lateral offset (how far left/right the post sits)
    start    = -spacing * (n - 1) / 2.0

    centers, half_extents = _init_slots()
    placed_xy = []
    slot = 0
    for i in range(n):
        sign = 1.0 if i % 2 == 0 else -1.0
        along_dist = start + i * spacing
        xy = fwd * along_dist + side * (sign * offset_m)
        xy = np.clip(xy, -_ARENA_XY + 0.3, _ARENA_XY - 0.3).astype(np.float32)
        if np.linalg.norm(xy) < 0.3:   # relax spawn exclusion for structured modes
            xy = xy + fwd * 0.5
        # Skip if clipping collapsed this post onto an existing one
        if any(np.linalg.norm(xy - p) < _INTER_EXCL for p in placed_xy):
            continue
        placed_xy.append(xy)
        z = rng.uniform(0.8, 1.5)
        centers[slot]      = [xy[0], xy[1], z]
        half_extents[slot] = _PILLAR_HE
        slot += 1

    return centers, half_extents


def _sample_hallway(rng, density_mult):
    """
    Two parallel rows of closely-spaced pillars forming a navigable corridor.
    Visual character: strong bilateral symmetry, clear depth wall on each side.
    OOD eval mode — not in training distribution by default.

    Uses all N_OBSTACLE_SLOTS=16 (8 per wall) at full density.
    """
    lo, hi = _MODE_N_RANGE["hallway"]
    n_total = _scale_n(rng, lo, hi, density_mult)
    n_per_side = n_total // 2   # equal split

    # Corridor: random orientation, ±corridor_half_width from center
    angle          = rng.uniform(-np.pi / 6, np.pi / 6)   # mostly axis-aligned
    fwd            = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
    side           = np.array([np.cos(angle), -np.sin(angle)], dtype=np.float32)

    corridor_hw    = rng.uniform(0.9, 1.4)    # half-width of gap drone must fly through
    pillar_spacing = rng.uniform(1.0, 1.8)    # along-corridor spacing between pillars
    start          = -pillar_spacing * (n_per_side - 1) / 2.0

    centers, half_extents = _init_slots()
    slot = 0
    for side_sign in (+1.0, -1.0):
        for i in range(n_per_side):
            along = start + i * pillar_spacing
            xy = fwd * along + side * (side_sign * corridor_hw)
            xy = np.clip(xy, -_ARENA_XY + 0.2, _ARENA_XY - 0.2).astype(np.float32)
            z  = rng.uniform(0.8, 1.5)
            centers[slot]      = [xy[0], xy[1], z]
            half_extents[slot] = _PILLAR_HE
            slot += 1
            if slot >= N_OBSTACLE_SLOTS:
                break
        if slot >= N_OBSTACLE_SLOTS:
            break

    return centers, half_extents


# ── Helpers ────────────────────────────────────────────────────────────────────

def _init_slots() -> tuple[np.ndarray, np.ndarray]:
    centers      = np.empty((N_OBSTACLE_SLOTS, 3), dtype=np.float32)
    half_extents = np.zeros((N_OBSTACLE_SLOTS, 3), dtype=np.float32)
    centers[:]   = [100.0, 100.0, 100.0]
    return centers, half_extents


def _scale_n(rng: np.random.Generator, lo: int, hi: int, density_mult: float) -> int:
    if density_mult <= 0.0:
        return 0
    lo_f = lo * density_mult
    hi_f = hi * density_mult
    lo_i = max(1, int(np.floor(lo_f)))
    hi_i = min(N_OBSTACLE_SLOTS, int(np.ceil(hi_f)))
    if lo_i >= hi_i:
        return lo_i
    return int(rng.integers(lo_i, hi_i + 1))


def _sample_xy_free(
    rng: np.random.Generator,
    placed_xy: list,
    spawn_excl: float,
    inter_excl: float,
    arena_xy: float,
    slot_idx: int,
) -> np.ndarray:
    """Rejection-sample xy satisfying spawn + inter-obstacle exclusions."""
    for _ in range(_MAX_RETRIES):
        xy = rng.uniform(-arena_xy, arena_xy, size=2).astype(np.float32)
        if np.linalg.norm(xy) < spawn_excl:
            continue
        if any(np.linalg.norm(xy - p) < inter_excl for p in placed_xy):
            continue
        return xy

    warnings.warn(
        f"obstacle_{slot_idx}: inter-obstacle exclusion unmet after {_MAX_RETRIES} retries; "
        f"accepting without it.",
        stacklevel=3,
    )
    for _ in range(_MAX_RETRIES * 10):
        xy = rng.uniform(-arena_xy, arena_xy, size=2).astype(np.float32)
        if np.linalg.norm(xy) >= spawn_excl:
            return xy

    raise RuntimeError(f"obstacle_{slot_idx}: failed to satisfy spawn exclusion.")
