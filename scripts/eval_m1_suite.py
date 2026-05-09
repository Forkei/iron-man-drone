"""
Run M1.3 epoch_013000 through the unified eval_suite.py pipeline.

Produces canonical M1 numbers from the same GPU-MJX backend used for M2 eval.
Output is printed and saved to experiments/m1_suite_results.json.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
from iron_man_drone.evaluation.eval_suite import (
    load_actor_from_checkpoint, make_m1_traj_suite,
    nominal_condition, run_eval_suite, print_suite_summary,
)
from iron_man_drone.envs.quadrotor_env import VecEnv

REPO_ROOT = Path(__file__).parent.parent
CKPT = (
    REPO_ROOT
    / "experiments/m1_3_polynomial_fix"
    / "m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000"
)

print(f"JAX devices : {jax.devices()}")
print(f"Checkpoint  : {CKPT}")
print()

class _Cfg:
    num_envs = 1

env = VecEnv(_Cfg(), fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)
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

print_suite_summary(results, "M1.3 epoch_013000 — eval_suite.py (GPU MJX, 2026-05-09)")

out = {
    traj: {
        cond: {
            "mean_med": cr.mean_med,
            "meds": cr.meds,
            "any_crashed": cr.any_crashed,
        }
        for cond, cr in conds.items()
    }
    for traj, conds in results.items()
}
out_path = REPO_ROOT / "experiments/m1_suite_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
