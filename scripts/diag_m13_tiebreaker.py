"""
Tiebreaker eval: MJX single-env, same code path as inline training eval.
Reports both mean and median to isolate aggregation difference from simulator difference.
Also reports the per-step error curve to characterize acquisition vs steady-state.

Checkpoint: epoch_013000 (best inline MED during M1.3 run 2).
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import yaml

REPO_ROOT = Path(__file__).parent.parent
CHECKPOINT = REPO_ROOT / "experiments/m1_3_polynomial_fix/m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000"


def main():
    # ── Load actor (same as train_m1.py) ──────────────────────────────────────
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    config_path = CHECKPOINT.parent.parent / "config_frozen.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = PPOConfig(
        actor_obs_dim=cfg["observation"]["actor_dim"],
        critic_obs_dim=cfg["observation"]["critic_dim"],
        action_dim=cfg["action"]["dim"],
        hidden_dim=cfg["network"]["hidden_dim"],
        num_layers=cfg["network"]["num_layers"],
    )

    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(CHECKPOINT.resolve()))
    actor_state = actor_state.replace(params=ckpt["actor"]["params"])
    print("Actor loaded from epoch_013000.")

    # ── MJX env — single-env step (NOT vmapped), same as inline training eval ─
    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, EPISODE_STEPS, _build_obs, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT,
    )
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, get_reference_pos,
    )

    class _Cfg:
        num_envs = 1

    env = VecEnv(_Cfg())
    eval_reset = jax.jit(env._reset_fn)   # single-env, no vmap
    eval_step  = jax.jit(env._step_fn)    # single-env, no vmap
    drone_id   = env.mj_model.body("drone").id

    eval_traj = make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal")

    # Warm up JIT (same shapes as inline eval)
    print("Pre-warming MJX single-env JIT ...")
    pw_key = jax.random.PRNGKey(999)
    pw_state, pw_obs, _ = eval_reset(pw_key, jnp.ones(()))
    pw_state = pw_state._replace(traj=eval_traj)
    pw_mean, _ = actor_state.apply_fn(actor_state.params, pw_obs[None])
    eval_step(pw_state, pw_mean[0], jnp.ones(()))
    print("JIT warm-up done.")

    # ── Run 3 episodes (different random seeds) ───────────────────────────────
    all_results = []
    seeds = [0, 1, 2]

    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        key, rk = jax.random.split(key)

        state, a_obs, _ = eval_reset(rk, jnp.ones(()))   # kf_mult=1.0 — DR disabled
        state = state._replace(traj=eval_traj)
        a_obs, _ = _build_obs(state.mjx_data, eval_traj, state.step, drone_id)

        positions, refs_inline, refs_diag = [], [], []

        for si in range(EPISODE_STEPS):
            mean, _ = actor_state.apply_fn(actor_state.params, a_obs[None])
            action = mean[0]   # deterministic — policy mean, no sampling

            state, a_obs, _, _, done = eval_step(state, action, jnp.ones(()))  # kf_mult=1.0
            state = state._replace(traj=eval_traj)

            pos_xy = np.array(state.mjx_data.xpos[drone_id, :2])
            # Inline eval convention: ref at step si (1 behind drone's state.step=si+1)
            ref_inline = np.array(get_reference_pos(eval_traj, jnp.int32(si))[:2])
            # Diagnostic convention: ref at step si+1 (matches drone state.step)
            ref_diag   = np.array(get_reference_pos(eval_traj, jnp.int32(si + 1))[:2])

            positions.append(pos_xy)
            refs_inline.append(ref_inline)
            refs_diag.append(ref_diag)

            if bool(done):
                print(f"  seed={seed}: episode terminated at step {si}")
                break

        positions    = np.array(positions)
        refs_inline  = np.array(refs_inline)
        refs_diag    = np.array(refs_diag)

        err_inline = np.linalg.norm(positions - refs_inline, axis=1)
        err_diag   = np.linalg.norm(positions - refs_diag,   axis=1)

        # Initial position of drone (from reset)
        init_pos = np.array(eval_reset(rk, jnp.ones(()))[0].mjx_data.xpos[drone_id, :2])

        result = {
            "seed": seed,
            "n_steps": len(positions),
            "init_xy_err": float(np.linalg.norm(init_pos - refs_inline[0])),
            # Inline convention (matches what training logged)
            "mean_inline":   float(err_inline.mean()),
            "median_inline": float(np.median(err_inline)),
            # Diagnostic convention (ref at si+1)
            "mean_diag":     float(err_diag.mean()),
            "median_diag":   float(np.median(err_diag)),
            # Acquisition characterization
            "mean_first50":  float(err_inline[:50].mean()),
            "mean_steps50+": float(err_inline[50:].mean()),
            "median_steps50+": float(np.median(err_inline[50:])),
            "crashed": len(positions) < EPISODE_STEPS,
        }
        all_results.append(result)

        print(f"\n  seed={seed} | steps={result['n_steps']}")
        print(f"    inline  mean={result['mean_inline']:.4f}  median={result['median_inline']:.4f}")
        print(f"    diag    mean={result['mean_diag']:.4f}    median={result['median_diag']:.4f}")
        print(f"    first 50 steps mean_err: {result['mean_first50']:.4f}m")
        print(f"    steps 50+ mean_err:      {result['mean_steps50+']:.4f}m")
        print(f"    steps 50+ median_err:    {result['median_steps50+']:.4f}m")

    # ── Summary across all seeds ───────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  TIEBREAKER SUMMARY — epoch_013000 figure_eight_normal (MJX)")
    print(f"{'='*62}")
    print(f"  {'Metric':<35} {'seed0':>8} {'seed1':>8} {'seed2':>8}")
    print(f"  {'-'*62}")

    for key_name, label in [
        ("mean_inline",    "Mean (inline convention)"),
        ("median_inline",  "Median (inline convention)"),
        ("mean_first50",   "Mean error, steps 0-49 (acquisition)"),
        ("mean_steps50+",  "Mean error, steps 50+ (steady-state)"),
        ("median_steps50+","Median error, steps 50+ (steady-state)"),
    ]:
        vals = [r[key_name] for r in all_results]
        print(f"  {label:<35} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f}")

    print(f"\n  Training inline eval reported: ~0.066–0.084m  (= mean_inline, this table)")
    print(f"  CPU mujoco diagnostic reported: 0.026m        (= median, different simulator)")
    print(f"  Target: 0.056m")
    print(f"{'='*62}\n")

    # Print structured output for diagnostic doc
    mean_inline_avg  = float(np.mean([r["mean_inline"]    for r in all_results]))
    med_inline_avg   = float(np.mean([r["median_inline"]  for r in all_results]))
    mean_ss_avg      = float(np.mean([r["mean_steps50+"]  for r in all_results]))
    med_ss_avg       = float(np.mean([r["median_steps50+"] for r in all_results]))
    mean_acq_avg     = float(np.mean([r["mean_first50"]   for r in all_results]))

    print(f"SUMMARY:")
    print(f"  MJX mean (inline convention, full episode):   {mean_inline_avg:.4f}m")
    print(f"  MJX median (full episode):                    {med_inline_avg:.4f}m")
    print(f"  MJX mean error, acquisition (steps 0-49):    {mean_acq_avg:.4f}m")
    print(f"  MJX mean error, steady-state (steps 50+):    {mean_ss_avg:.4f}m")
    print(f"  MJX median error, steady-state (steps 50+):  {med_ss_avg:.4f}m")
    print(f"")
    print(f"  If MJX steady-state median ≈ CPU mujoco median (0.026m):")
    print(f"    → Discrepancy is purely mean vs median + acquisition tail")
    print(f"    → No simulator difference. Both show ~0.026m true tracking quality.")
    print(f"  If MJX steady-state median >> 0.026m:")
    print(f"    → Real simulator difference. MJX dynamics give worse tracking.")
    print(f"    → Inline eval (MJX) is ground truth for training-env performance.")


if __name__ == "__main__":
    main()
