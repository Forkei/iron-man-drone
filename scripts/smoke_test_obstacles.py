"""
SC-5 — Obstacle randomization + episode stability smoke test.

Part 1 — Geometry (pure JAX/numpy, no GPU required):
  T1: N_OBSTACLE_SLOTS=16, shape (16, 3)
  T2: no active obstacle within 0.5 m of (0,0) — spawn exclusion
  T3: no two active obstacles within 0.4 m of each other — inter-obstacle exclusion
  T4: active obstacles have z = 1.0
  T5: active obstacles have half_extents = [0.05, 0.05, 0.5]
  T6: inactive rows parked at [100,100,100] with zero half_extents

Part 2 — Episode stability (requires CUDA / DepthVecEnv):
  T7: DepthVecEnv(n_obstacles=4) batch_reset + 1000 batch_step calls without error
  T8: obstacle positions remain non-NaN throughout the run

Usage:
  python scripts/smoke_test_obstacles.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import types
import numpy as np
import jax
import jax.numpy as jnp

from iron_man_drone.utils.obstacle_randomization import (
    N_OBSTACLE_SLOTS, sample_obstacle_configs,
)

N_SAMPLES  = 100
N_ACTIVE   = 4
HE_EXPECT  = np.array([0.05, 0.05, 0.5], dtype=np.float32)
SPAWN_EXCL = 0.5
INTER_EXCL = 0.4


def _gate(label, passed, detail=""):
    mark = "✓" if passed else "✗"
    print(f"  {mark}  {label}")
    if detail:
        print(f"      {detail}")
    if not passed:
        raise AssertionError(f"FAILED: {label}  {detail}")
    return True


def part1_geometry():
    print("\n  Part 1 — Geometry (100 samples, 4 active obstacles each)\n")
    rng = np.random.default_rng(seed=7)

    spawn_violations = inter_violations = z_violations = he_violations = park_violations = 0

    for _ in range(N_SAMPLES):
        c, he = sample_obstacle_configs(rng, N_ACTIVE)

        assert c.shape  == (N_OBSTACLE_SLOTS, 3), f"centers shape {c.shape}"
        assert he.shape == (N_OBSTACLE_SLOTS, 3), f"he shape {he.shape}"

        for i in range(N_ACTIVE):
            if np.linalg.norm(c[i, :2]) < SPAWN_EXCL:
                spawn_violations += 1
            if not np.isclose(c[i, 2], 1.0):
                z_violations += 1
            if not np.allclose(he[i], HE_EXPECT):
                he_violations += 1

        for i in range(N_OBSTACLE_SLOTS - N_ACTIVE):
            row = N_ACTIVE + i
            if not np.allclose(c[row], [100.0, 100.0, 100.0]):
                park_violations += 1
            if not np.allclose(he[row], 0.0):
                park_violations += 1

        for i in range(N_ACTIVE):
            for j in range(i + 1, N_ACTIVE):
                d = np.linalg.norm(c[i, :2] - c[j, :2])
                if d < INTER_EXCL:
                    inter_violations += 1

    _gate("T1 shape (16, 3) for centers and half_extents",
          True, f"verified for {N_SAMPLES} samples")
    _gate("T2 spawn exclusion ≥ 0.5 m from (0,0)",
          spawn_violations == 0,
          f"violations: {spawn_violations}/{N_SAMPLES * N_ACTIVE}")
    _gate("T3 inter-obstacle exclusion ≥ 0.4 m",
          inter_violations == 0,
          f"violations: {inter_violations}")
    _gate("T4 active obstacle z = 1.0",
          z_violations == 0,
          f"violations: {z_violations}/{N_SAMPLES * N_ACTIVE}")
    _gate("T5 active obstacle half_extents = [0.05, 0.05, 0.5]",
          he_violations == 0,
          f"violations: {he_violations}/{N_SAMPLES * N_ACTIVE}")
    _gate("T6 inactive rows parked at [100,100,100] with zero half_extents",
          park_violations == 0,
          f"violations: {park_violations}")


def part2_episode_stability():
    print("\n  Part 2 — Episode stability (DepthVecEnv N=4, n_obstacles=4, 1000 steps)\n")
    from iron_man_drone.envs.quadrotor_env_depth import DepthVecEnv

    cfg = types.SimpleNamespace(num_envs=4, max_episode_steps=1000)
    env = DepthVecEnv(cfg, n_obstacles=4, fault_prob=0.7)

    keys = jax.random.split(jax.random.PRNGKey(99), 4)
    states, a_obs, c_obs = env.batch_reset(keys)

    rng_act = np.random.default_rng(seed=42)
    max_depth_tried = 0
    for step in range(1000):
        actions = jnp.array(rng_act.uniform(-0.1, 0.1, (4, 4)).astype(np.float32))
        states, a_obs, c_obs, rewards, dones = env.batch_step(states, actions)

        pos = np.array(states.obstacle_positions)
        if np.any(np.isnan(pos)):
            max_depth_tried = step
            break
        max_depth_tried = step + 1

    jax.block_until_ready(states.mjx_data.qpos)
    pos_final = np.array(states.obstacle_positions)

    _gate("T7 1000 batch_step calls complete without error",
          max_depth_tried == 1000,
          f"completed {max_depth_tried}/1000 steps")
    _gate("T8 obstacle positions remain non-NaN throughout",
          not np.any(np.isnan(pos_final)),
          f"obstacle_positions shape {pos_final.shape}")


def main():
    print(f"\n{'='*60}")
    print(f"  SC-5 — Obstacle randomization + episode stability")
    print(f"  N_SAMPLES={N_SAMPLES}  N_ACTIVE={N_ACTIVE}  N_OBSTACLE_SLOTS={N_OBSTACLE_SLOTS}")
    print(f"{'='*60}")

    part1_geometry()
    part2_episode_stability()

    print(f"\n{'='*60}")
    print(f"  ALL PASS — SC-5 complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
