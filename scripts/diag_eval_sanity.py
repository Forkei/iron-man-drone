"""
Task 1 + 2 diagnostic — M1.3 resume investigation.

Minimal version: 2 checkpoints, 200 steps each, print every 20 steps.
Goal: is the post-resume drone flying badly, or is the MED metric broken?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

REPO_ROOT = Path(__file__).parent.parent
CKPT_BASE = REPO_ROOT / "experiments/m1_3_polynomial_fix/m1_3_polynomial_fix_1777826991/checkpoints"

N_STEPS = 200  # enough to see figure-eight behavior without leaking memory


def rollout(actor_apply, params, env, eval_reset, eval_step, eval_traj, drone_id, label):
    from iron_man_drone.envs.quadrotor_env import _build_obs, EPISODE_STEPS
    from iron_man_drone.envs.trajectories import get_reference_pos

    key = jax.random.PRNGKey(0)
    state, a_obs, _ = eval_reset(key, jnp.ones(()))
    state = state._replace(traj=eval_traj)
    a_obs, _ = _build_obs(state.mjx_data, eval_traj, state.step, drone_id)

    print(f"\n--- {label} ---")
    errors = []
    for si in range(min(N_STEPS, EPISODE_STEPS)):
        mean, _ = actor_apply(params, a_obs[None])
        action = mean[0]
        state, a_obs, _, _, done = eval_step(state, action, jnp.ones(()))
        state = state._replace(traj=eval_traj)

        pos = np.array(state.mjx_data.xpos[drone_id])     # (3,)
        ref = np.array(get_reference_pos(eval_traj, jnp.int32(si)))  # (3,)
        err_xy = float(np.linalg.norm(pos[:2] - ref[:2]))
        errors.append(err_xy)

        if si % 20 == 0 or bool(done):
            print(f"  step {si:3d}  drone=({pos[0]:+.3f},{pos[1]:+.3f},z={pos[2]:.2f})"
                  f"  ref=({ref[0]:+.3f},{ref[1]:+.3f})  err={err_xy:.3f}m"
                  + ("  *** CRASHED ***" if bool(done) else ""))
        if bool(done):
            print(f"  Episode terminated at step {si}")
            break

    med = float(np.mean(errors))
    print(f"  MED over {len(errors)} steps: {med:.4f}m  "
          f"(max={max(errors):.3f}m, final={errors[-1]:.3f}m)")
    return med


def main():
    from iron_man_drone.envs.quadrotor_env import VecEnv, DT, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
    from iron_man_drone.envs.trajectories import make_figure_eight_trajectory
    from iron_man_drone.policy.networks import Actor

    class Cfg:
        num_envs = 1

    print("Building env (once)...")
    env = VecEnv(Cfg())
    drone_id = env.mj_model.body("drone").id
    actor = Actor(hidden_dim=256, num_layers=3)
    eval_traj = make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal")

    eval_reset = jax.jit(env._reset_fn)
    eval_step  = jax.jit(env._step_fn)

    # Warm up JIT with dummy call
    _s, _o, _ = eval_reset(jax.random.PRNGKey(999), jnp.ones(()))
    print("JIT warmed.")

    checkpointer = ocp.PyTreeCheckpointer()

    print("\n" + "=" * 70)
    print("TASK 1 — Compare pre-resume (epoch_002000) vs post-resume (epoch_014000)")
    print(f"Running {N_STEPS} steps each on figure_eight_normal")
    print("=" * 70)

    ckpt_2000 = checkpointer.restore(str(CKPT_BASE / "epoch_002000"))
    params_2000 = ckpt_2000["actor"]["params"]
    med_2000 = rollout(actor.apply, params_2000, env, eval_reset, eval_step, eval_traj, drone_id,
                       "epoch_002000 (pre-resume, known MED=0.0850m)")
    del ckpt_2000, params_2000

    ckpt_14000 = checkpointer.restore(str(CKPT_BASE / "epoch_014000"))
    params_14000 = ckpt_14000["actor"]["params"]
    med_14000 = rollout(actor.apply, params_14000, env, eval_reset, eval_step, eval_traj, drone_id,
                        "epoch_014000 (post-resume, inline eval reported 0.9711m)")
    del ckpt_14000, params_14000

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  epoch_002000  manual MED ({N_STEPS} steps): {med_2000:.4f}m  (inline reported: 0.0850m)")
    print(f"  epoch_014000  manual MED ({N_STEPS} steps): {med_14000:.4f}m  (inline reported: 0.9711m)")
    print()
    if med_14000 > 0.5:
        print("  FINDING: Post-resume drone is genuinely flying badly (>0.5m error).")
        print("  Inline eval metric is CORRECT. Policy was damaged by resume.")
    elif med_14000 < 0.15:
        print("  FINDING: Post-resume drone is flying OK (<0.15m error).")
        print("  Inline eval metric is BROKEN (reports 0.97m but drone flies fine).")
    else:
        print(f"  FINDING: Post-resume drone is degraded ({med_14000:.3f}m) but not random.")
        print("  Intermediate case — some policy damage but not total collapse.")


if __name__ == "__main__":
    main()
