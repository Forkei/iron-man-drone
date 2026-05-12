"""
SC-1 — M2 Phase 1 (privileged) actor with DepthVecEnv regression check.

Confirms crazyflie_depth.xml produces identical dynamics to crazyflie.xml by
evaluating the M2 Phase 1 actor (50-dim privileged obs) on DepthVecEnv with
n_obstacles=0.  Renders one depth frame per trajectory when --render_depth is set.

Gate: figure_eight_normal nominal MED ≤ 0.065 m
      (Phase 1 privileged baseline on VecEnv: 0.0574 m — gate is 1.15×)

Usage:
  python scripts/eval_sc1_depth_m2.py
  python scripts/eval_sc1_depth_m2.py --render_depth
"""

import sys
import argparse
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
from iron_man_drone.evaluation.eval_suite import (
    load_actor_from_checkpoint, make_m2_traj_suite,
    nominal_condition, run_eval_suite, print_suite_summary,
)
from iron_man_drone.envs.quadrotor_env_depth import DepthVecEnv

REPO_ROOT = Path(__file__).parent.parent
CKPT = (
    REPO_ROOT
    / "experiments/m2_phase1_baseline"
    / "m2_phase1_baseline_1778244202/checkpoints/final"
)
FIGURES_DIR = REPO_ROOT / "notes" / "figures"
GATE_MED    = 0.065


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_depth", action="store_true",
                        help="Render depth frame at each trajectory start and save PNG")
    args = parser.parse_args()

    print(f"\nSC-1 — M2 + DepthVecEnv regression check")
    print(f"JAX devices : {jax.devices()}")
    print(f"Checkpoint  : {CKPT}")
    print(f"render_depth: {args.render_depth}\n")

    cfg = types.SimpleNamespace(num_envs=1)
    env = DepthVecEnv(cfg, n_obstacles=0, fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)

    actor_state, actor_apply_fn = load_actor_from_checkpoint(str(CKPT), obs_dim=50)

    traj_configs = make_m2_traj_suite()
    conditions   = [nominal_condition()]

    results = run_eval_suite(
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_state.params,
        env=env,
        traj_configs=traj_configs,
        conditions=conditions,
        obs_dim=50,
        verbose=True,
        render_depth=args.render_depth,
        figures_dir=FIGURES_DIR if args.render_depth else None,
    )

    print_suite_summary(results, "SC-1 — M2 Phase 1 actor on DepthVecEnv")

    med = results["figure_eight_normal"]["nominal"].mean_med
    passed = med <= GATE_MED
    mark = "✓" if passed else "✗"
    print(f"\n  {mark}  SC-1 gate: figure_eight_normal nominal MED ≤ {GATE_MED:.3f} m")
    print(f"      got {med:.4f} m  (Phase 1 VecEnv baseline: 0.0574 m)")

    if not passed:
        raise AssertionError(
            f"SC-1 FAILED: figure_eight_normal nominal MED {med:.4f} > {GATE_MED:.3f} m\n"
            f"DepthVecEnv dynamics may differ from VecEnv. Check crazyflie_depth.xml."
        )

    print(f"\n  ALL PASS — SC-1 complete\n")


if __name__ == "__main__":
    main()
