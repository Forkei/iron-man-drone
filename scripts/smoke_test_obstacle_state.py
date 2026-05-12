"""
Task 2 gate — obstacle_randomization.py sanity check (SC-7 subset).

Tests:
  T1: shape is (N_OBSTACLE_SLOTS=16, 3) for both centers and half_extents
  T2: active rows are placed inside the arena with correct z and half-extents
  T3: inactive rows are parked correctly (centers=[100,100,100], he=zeros)
  T4: spawn exclusion — no active obstacle within 0.5 m xy of (0,0)
  T5: inter-obstacle exclusion — no two active obstacles within 0.4 m of each other
  T6: min_distance_to_obstacle returns positive float for drone at origin
  T7: n_obstacles=0 returns all-park centers, zero half_extents, min_dist=inf
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax.numpy as jnp
from iron_man_drone.utils.obstacle_randomization import (
    N_OBSTACLE_SLOTS, sample_obstacle_configs, min_distance_to_obstacle
)

N_SAMPLES   = 100
N_ACTIVE    = 4
SPAWN_EXCL  = 0.5
INTER_EXCL  = 0.4
HE_EXPECTED = np.array([0.05, 0.05, 0.5], dtype=np.float32)


def check(name, passed, detail=""):
    mark = "✓" if passed else "✗"
    print(f"  {mark}  {name}")
    if detail:
        print(f"      {detail}")
    if not passed:
        raise AssertionError(f"FAILED: {name}  {detail}")


def main():
    print(f"\n{'='*60}")
    print(f"  Task 2 obstacle randomization gate")
    print(f"  N_OBSTACLE_SLOTS={N_OBSTACLE_SLOTS}  samples={N_SAMPLES}  n_active={N_ACTIVE}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(seed=0)

    spawn_violations   = 0
    inter_violations   = 0
    z_violations       = 0
    he_violations      = 0
    park_violations    = 0

    for trial in range(N_SAMPLES):
        c, he = sample_obstacle_configs(rng, N_ACTIVE)

        # T1: shape
        assert c.shape  == (N_OBSTACLE_SLOTS, 3), f"centers shape {c.shape}"
        assert he.shape == (N_OBSTACLE_SLOTS, 3), f"half_extents shape {he.shape}"

        # T2/T3: active rows
        for i in range(N_ACTIVE):
            if np.linalg.norm(c[i, :2]) < SPAWN_EXCL:
                spawn_violations += 1
            if not np.isclose(c[i, 2], 1.0):
                z_violations += 1
            if not np.allclose(he[i], HE_EXPECTED):
                he_violations += 1
        for i in range(N_ACTIVE, N_OBSTACLE_SLOTS):
            if not np.allclose(c[i], [100.0, 100.0, 100.0]):
                park_violations += 1
            if not np.allclose(he[i], 0.0):
                park_violations += 1

        # T5: inter-obstacle
        for i in range(N_ACTIVE):
            for j in range(i + 1, N_ACTIVE):
                d = np.linalg.norm(c[i, :2] - c[j, :2])
                if d < INTER_EXCL:
                    inter_violations += 1

    check("T1 shape (N_OBSTACLE_SLOTS=16, 3)",
          spawn_violations == 0 or True,   # shape checked inside loop via assert
          f"checked {N_SAMPLES} samples")

    check("T2 spawn exclusion ≥ 0.5 m from (0,0)",
          spawn_violations == 0,
          f"violations: {spawn_violations}/{N_SAMPLES * N_ACTIVE}")

    check("T3 active z = 1.0",
          z_violations == 0,
          f"violations: {z_violations}/{N_SAMPLES * N_ACTIVE}")

    check("T4 active half_extents = [0.05, 0.05, 0.5]",
          he_violations == 0,
          f"violations: {he_violations}/{N_SAMPLES * N_ACTIVE}")

    check("T5 inter-obstacle ≥ 0.4 m center-to-center",
          inter_violations == 0,
          f"violations: {inter_violations}")

    check("T6 inactive rows parked correctly",
          park_violations == 0,
          f"violations: {park_violations}")

    # T6: min_distance_to_obstacle > 0 for drone at origin
    rng2 = np.random.default_rng(seed=42)
    c4, he4 = sample_obstacle_configs(rng2, N_ACTIVE)
    d = min_distance_to_obstacle(jnp.zeros(3), c4, he4, N_ACTIVE)
    check("T6 min_distance_to_obstacle > 0.4 for drone at origin",
          float(d) > 0.4,
          f"min_dist = {float(d):.4f} m")

    # T7: n_obstacles=0
    c0, he0 = sample_obstacle_configs(rng2, 0)
    check("T7 n_obstacles=0 centers all parked",
          np.all(c0 == 100.0),
          f"non-park rows: {np.sum(c0 != 100.0)}")
    check("T7 n_obstacles=0 half_extents all zero",
          np.all(he0 == 0.0))
    d0 = min_distance_to_obstacle(jnp.zeros(3), c0, he0, 0)
    check("T7 min_distance_to_obstacle returns inf for n_obstacles=0",
          jnp.isinf(d0))

    print(f"\n{'='*60}")
    print(f"  ALL PASS — Task 2 complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
