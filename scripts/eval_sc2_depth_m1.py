"""
SC-2 — M1.3 actor with DepthVecEnv regression check.

Confirms crazyflie_depth.xml produces identical dynamics to crazyflie.xml by
evaluating the M1.3 actor (42-dim obs) on DepthVecEnv with n_obstacles=0.

Gate: figure_eight_normal nominal MED ≤ 0.042 m
      (M1.3 epoch_013000 baseline on VecEnv: 0.0402 m — gate is 1.045×)

Usage:
  python scripts/eval_sc2_depth_m1.py
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
from iron_man_drone.evaluation.eval_suite import (
    load_actor_from_checkpoint, make_m1_traj_suite,
    nominal_condition, run_eval_suite, print_suite_summary,
)
from iron_man_drone.envs.quadrotor_env_depth import DepthVecEnv

REPO_ROOT = Path(__file__).parent.parent
CKPT = (
    REPO_ROOT
    / "experiments/m1_3_polynomial_fix"
    / "m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000"
)
GATE_MED = 0.042  # canonical VecEnv baseline = 0.0402m; 1.045× headroom


def main():
    print(f"\nSC-2 — M1.3 + DepthVecEnv regression check")
    print(f"JAX devices : {jax.devices()}")
    print(f"Checkpoint  : {CKPT}\n")

    cfg = types.SimpleNamespace(num_envs=1)
    env = DepthVecEnv(cfg, n_obstacles=0, fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)

    actor_state, actor_apply_fn = load_actor_from_checkpoint(str(CKPT), obs_dim=42)

    traj_configs = make_m1_traj_suite()
    conditions   = [nominal_condition()]

    results = run_eval_suite(
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_state.params,
        env=env,
        traj_configs=traj_configs,
        conditions=conditions,
        obs_dim=42,
        verbose=True,
    )

    print_suite_summary(results, "SC-2 — M1.3 actor on DepthVecEnv")

    med = results["figure_eight_normal"]["nominal"].mean_med
    passed = med <= GATE_MED
    mark = "✓" if passed else "✗"
    print(f"\n  {mark}  SC-2 gate: figure_eight_normal nominal MED ≤ {GATE_MED:.3f} m")
    print(f"      got {med:.4f} m  (M1.3 VecEnv baseline: 0.0402 m)")

    if not passed:
        raise AssertionError(
            f"SC-2 FAILED: figure_eight_normal nominal MED {med:.4f} > {GATE_MED:.3f} m\n"
            f"DepthVecEnv dynamics may differ from VecEnv. Check crazyflie_depth.xml."
        )

    print(f"\n  ALL PASS — SC-2 complete\n")


if __name__ == "__main__":
    main()
