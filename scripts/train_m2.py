"""
M2 Phase 1 training — RMA fault-tolerant policy.

Usage:
  python scripts/train_m2.py                          # full Phase 1 (15k epochs)
  python scripts/train_m2.py --nominal_only           # validation gate 4: nominal-only, 1k epochs
  python scripts/train_m2.py --total_epochs 1000      # override epoch count
  python scripts/train_m2.py --num_envs 512           # smaller GPU

GATE: notes/M2_hypothesis.md must exist and be non-trivial before this runs.

M2 interface changes from train_m1.py:
  - No kf_multipliers arg in env calls — all DR sampled inside reset()
  - collect_rollout has no kf_multipliers arg
  - VecEnv takes fault_prob, eta_min, mass_lo, mass_hi constructor args
  - Nominal-only mode sets fault_prob=0.0, mass_lo=mass_hi=1.0 for gate 4
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent


def check_hypothesis_gate():
    hyp = REPO_ROOT / "notes" / "M2_hypothesis.md"
    if not hyp.exists() or hyp.stat().st_size < 200:
        print("ERROR: notes/M2_hypothesis.md is missing or too short.")
        print("Fill in the hypothesis doc before training. This is a hard gate.")
        sys.exit(1)
    # Crude check: the template has [fill in] placeholders
    content = hyp.read_text()
    if "[fill in before training]" in content:
        print("ERROR: notes/M2_hypothesis.md still has unfilled placeholders.")
        print("Complete the hypothesis doc (date, predictions) before training.")
        sys.exit(1)
    print("[GATE PASSED] M2 hypothesis doc found and filled.")


def sanity_check_entropy_ratio(actor_state, critic_state, env, flat_cfg, key):
    import distrax
    keys = jax.random.split(key, flat_cfg["num_envs"])
    states, a_obs, c_obs = env.batch_reset(keys)

    total_reward = 0.0
    total_entropy = 0.0
    n_steps = 10

    for _ in range(n_steps):
        key, act_key = jax.random.split(key)
        mean, log_std = actor_state.apply_fn(actor_state.params, a_obs)
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        actions = dist.sample(seed=act_key)
        entropy = float(dist.entropy().mean())
        states, a_obs, c_obs, rewards, dones = env.batch_step(states, actions)
        total_reward += float(rewards.mean())
        total_entropy += entropy

    mean_reward = total_reward / n_steps
    mean_entropy = total_entropy / n_steps
    entropy_contrib = flat_cfg["entropy_coeff"] * mean_entropy
    ratio = abs(entropy_contrib) / (abs(mean_reward) + 1e-8)

    print()
    print("=" * 62)
    print("   PRE-TRAINING ENTROPY / REWARD SANITY CHECK")
    print("=" * 62)
    print(f"  Mean reward per step:       {mean_reward:8.4f}")
    print(f"  Policy entropy:             {mean_entropy:8.4f} nats")
    print(f"  Entropy contribution:       {entropy_contrib:8.4f}")
    print(f"  Entropy / reward ratio:     {ratio*100:7.2f}%   (target: < 10%)")
    print()
    if ratio > 0.10:
        print("  !! WARNING: entropy ratio ABOVE 10% !!")
    else:
        print(f"  OK — entropy ratio is {ratio*100:.1f}% (< 10%)")
    print("=" * 62)
    print()


def main():
    check_hypothesis_gate()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/m2_phase1_baseline/config.yaml"))
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--total_epochs", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--nominal_only", action="store_true",
                        help="Validation gate 4: train with nominal DR only (fault_prob=0, mass=1)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.num_envs:
        cfg["env"]["num_envs"] = args.num_envs
    if args.total_epochs:
        cfg.setdefault("experiment", {})["total_epochs"] = args.total_epochs

    flat_cfg = {
        "num_envs":       cfg["env"]["num_envs"],
        "horizon":        cfg["ppo"]["horizon"],
        "total_epochs":   cfg.get("experiment", {}).get("total_epochs", 15000),
        "actor_lr":       cfg["ppo"]["actor_lr"],
        "critic_lr":      cfg["ppo"]["critic_lr"],
        "entropy_coeff":  cfg["ppo"]["entropy_coeff"],
        "gamma":          cfg["ppo"]["gamma"],
        "gae_lambda":     cfg["ppo"]["gae_lambda"],
        "clip_eps":       cfg["ppo"]["clip_eps"],
        "critic_updates": cfg["ppo"]["critic_updates"],
        "max_grad_norm":  cfg["ppo"]["max_grad_norm"],
        "num_minibatches":cfg["ppo"]["num_minibatches"],
        "ppo_epochs":     cfg["ppo"].get("ppo_epochs", 5),
        "actor_obs_dim":  cfg["observation"]["actor_dim"],
        "critic_obs_dim": cfg["observation"]["critic_dim"],
        "action_dim":     cfg["action"]["dim"],
        "hidden_dim":     cfg["network"]["hidden_dim"],
        "num_layers":     cfg["network"]["num_layers"],
    }

    dr_cfg = cfg["env"]["dr"]

    # DR parameters — overridden in nominal_only mode
    if args.nominal_only:
        fault_prob = 0.0
        mass_lo = mass_hi = 1.0
        print("[NOMINAL ONLY] fault_prob=0, mass=1.0 — validation gate 4 mode")
    else:
        fault_prob = dr_cfg.get("fault_prob", 0.7)
        mass_lo, mass_hi = dr_cfg.get("mass_range", [0.8, 1.2])

    eta_min = dr_cfg.get("eta_min", 0.5)

    exp_name = cfg.get("experiment", {}).get("name", "m2_phase1_baseline")
    if args.nominal_only:
        exp_name = exp_name + "_nominal_only"
    run_name = args.run_name or f"{exp_name}_{int(time.time())}"
    run_dir = REPO_ROOT / "experiments" / exp_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    import shutil
    shutil.copy(args.config, run_dir / "config_frozen.yaml")

    print(f"\nJAX devices: {jax.devices()}")
    assert len(jax.devices()) > 0
    key = jax.random.PRNGKey(cfg.get("experiment", {}).get("seed", 42))

    from iron_man_drone.envs.quadrotor_env import VecEnv

    class SimpleConfig:
        def __init__(self, d):
            self.__dict__.update(d)

    env_cfg = SimpleConfig({"num_envs": flat_cfg["num_envs"]})
    env = VecEnv(
        env_cfg,
        fault_prob=fault_prob,
        eta_min=eta_min,
        mass_lo=mass_lo,
        mass_hi=mass_hi,
    )
    print(f"Environment loaded. {flat_cfg['num_envs']} parallel envs.")
    print(f"DR: fault_prob={fault_prob}, eta_min={eta_min}, mass=[{mass_lo},{mass_hi}]")

    from iron_man_drone.policy.ppo import PPOConfig, create_train_states, collect_rollout, ppo_update

    ppo_cfg = PPOConfig(
        gamma=flat_cfg["gamma"],
        gae_lambda=flat_cfg["gae_lambda"],
        clip_eps=flat_cfg["clip_eps"],
        actor_lr=flat_cfg["actor_lr"],
        critic_lr=flat_cfg["critic_lr"],
        critic_updates=flat_cfg["critic_updates"],
        entropy_coeff=flat_cfg["entropy_coeff"],
        max_grad_norm=flat_cfg["max_grad_norm"],
        num_minibatches=flat_cfg["num_minibatches"],
        ppo_epochs=flat_cfg["ppo_epochs"],
        horizon=flat_cfg["horizon"],
        num_envs=flat_cfg["num_envs"],
        actor_obs_dim=flat_cfg["actor_obs_dim"],
        critic_obs_dim=flat_cfg["critic_obs_dim"],
        action_dim=flat_cfg["action_dim"],
        hidden_dim=flat_cfg["hidden_dim"],
        num_layers=flat_cfg["num_layers"],
    )

    key, init_key = jax.random.split(key)
    actor, critic, actor_state, critic_state = create_train_states(init_key, ppo_cfg)

    sanity_check_entropy_ratio(actor_state, critic_state, env, flat_cfg, key)

    # Initial reset — no kf_mults arg (M2 interface)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, flat_cfg["num_envs"])
    env_states, actor_obs, critic_obs = env.batch_reset(reset_keys)

    # Logging
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project="iron-man-drone-m2",
                name=run_name,
                config=flat_cfg,
                dir=str(run_dir),
                mode="offline",
            )
            use_wandb = True
        except Exception:
            use_wandb = False
    else:
        use_wandb = False

    import csv as _csv

    class _CSVWriter:
        def __init__(self, path):
            self._file = open(path, "w", newline="", buffering=1)
            self._writer = None
        def add_scalar(self, tag, value, step):
            row = {"step": step, "tag": tag, "value": value}
            if self._writer is None:
                self._writer = _csv.DictWriter(self._file, fieldnames=row.keys())
                self._writer.writeheader()
            self._writer.writerow(row)
        def close(self):
            self._file.close()

    (run_dir / "logs").mkdir(exist_ok=True)
    tb_writer = _CSVWriter(run_dir / "logs" / "metrics.csv")

    # In-training eval: figure-eight-normal, nominal conditions
    from iron_man_drone.envs.quadrotor_env import (
        EPISODE_STEPS, _build_obs, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT,
    )
    from iron_man_drone.envs.trajectories import make_figure_eight_trajectory, get_reference_pos

    _eval_traj  = make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal")
    _eval_reset = jax.jit(env._reset_fn)
    _eval_step  = jax.jit(env._step_fn)
    _drone_id   = env.mj_model.body("drone").id

    # Nominal priv_state for eval (no fault, nominal mass)
    _nominal_priv = jnp.concatenate([jnp.ones(4), jnp.ones(1), jnp.zeros(3)])

    print("Pre-warming eval JIT...")
    _pw_key = jax.random.PRNGKey(999)
    _pw_state, _pw_obs, _ = _eval_reset(_pw_key)
    _pw_state = _pw_state._replace(traj=_eval_traj, priv_state=_nominal_priv)
    _pw_a_obs, _ = _build_obs(_pw_state.mjx_data, _eval_traj, _pw_state.step, _drone_id, _nominal_priv)
    _pw_mean, _ = actor_state.apply_fn(actor_state.params, _pw_a_obs[None])
    _pw_state, _, _, _, _ = _eval_step(_pw_state, _pw_mean[0])
    del _pw_key, _pw_state, _pw_obs, _pw_a_obs, _pw_mean
    print("Eval JIT pre-warmed.")

    def _run_med_eval(actor_state, key, priv_override=None):
        """
        One deterministic episode on figure-eight-normal.
        priv_override: if provided (8-dim array), override the priv_state
            (e.g., to evaluate under a specific fault condition).
        """
        key, rk = jax.random.split(key)
        state, a_obs, _ = _eval_reset(rk)
        state = state._replace(traj=_eval_traj)

        if priv_override is not None:
            state = state._replace(priv_state=priv_override)

        a_obs, _ = _build_obs(
            state.mjx_data, _eval_traj, state.step, _drone_id, state.priv_state
        )

        positions, refs = [], []
        for si in range(EPISODE_STEPS):
            mean, _ = actor_state.apply_fn(actor_state.params, a_obs[None])
            action = mean[0]
            state, a_obs, _, _, done = _eval_step(state, action)
            state = state._replace(traj=_eval_traj)
            positions.append(np.array(state.mjx_data.xpos[_drone_id, :2]))
            refs.append(np.array(get_reference_pos(_eval_traj, jnp.int32(si))[:2]))
            if bool(done):
                break

        if not positions:
            return float("nan")
        positions = np.array(positions)
        refs = np.array(refs[:len(positions)])
        return float(np.linalg.norm(positions - refs, axis=1).mean())

    # JIT-compiled rollout and update (no kf_mults arg in M2)
    collect_rollout_jit = jax.jit(
        lambda as_, cs_, es_, ao_, co_, k_: collect_rollout(
            as_, cs_, es_, ao_, co_, env.step, env.reset, k_, ppo_cfg
        )
    )
    ppo_update_jit = jax.jit(
        lambda as_, cs_, tr_, lv_, k_: ppo_update(as_, cs_, tr_, lv_, k_, ppo_cfg)
    )

    start_time = time.time()
    total_steps = 0
    checkpoint_interval = cfg.get("experiment", {}).get("checkpoint_every", 1000)
    log_interval        = cfg.get("experiment", {}).get("log_every", 10)
    med_eval_interval   = cfg.get("experiment", {}).get("eval_every", 500)

    print(f"\nStarting training: 0–{flat_cfg['total_epochs']} epochs")
    print(f"Steps per epoch: {flat_cfg['num_envs'] * flat_cfg['horizon']}")

    for epoch in range(flat_cfg["total_epochs"]):
        key, rollout_key, update_key = jax.random.split(key, 3)

        transitions, env_states, actor_obs, critic_obs = collect_rollout_jit(
            actor_state, critic_state,
            env_states, actor_obs, critic_obs,
            rollout_key,
        )

        last_value = critic_state.apply_fn(critic_state.params, critic_obs)
        actor_state, critic_state, metrics = ppo_update_jit(
            actor_state, critic_state, transitions, last_value, update_key
        )
        total_steps += flat_cfg["num_envs"] * flat_cfg["horizon"]

        if epoch % log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            mean_reward = float(transitions.reward.mean())
            metrics_cpu = jax.tree_util.tree_map(float, metrics)
            done_rate = float(transitions.done.mean())
            mean_ep_len = 1.0 / (done_rate + 1e-6)

            print(
                f"Epoch {epoch:5d} | "
                f"reward {mean_reward:6.4f} | "
                f"vloss {metrics_cpu['critic_loss']:7.4f} | "
                f"entropy {metrics_cpu['entropy']:5.3f} | "
                f"ep_len {mean_ep_len:5.0f}s | "
                f"fps {fps:7.0f}"
            )
            tb_writer.add_scalar("train/reward", mean_reward, epoch)
            tb_writer.add_scalar("train/actor_loss", metrics_cpu["actor_loss"], epoch)
            tb_writer.add_scalar("train/critic_loss", metrics_cpu["critic_loss"], epoch)
            tb_writer.add_scalar("train/entropy", metrics_cpu["entropy"], epoch)
            tb_writer.add_scalar("train/done_rate", done_rate, epoch)
            tb_writer.add_scalar("train/fps", fps, epoch)

            if use_wandb:
                import wandb
                wandb.log({"epoch": epoch, "reward": mean_reward,
                           "done_rate": done_rate, **metrics_cpu, "fps": fps})

        if epoch > 0 and epoch % med_eval_interval == 0:
            key, eval_key = jax.random.split(key)
            med_nominal = _run_med_eval(actor_state, eval_key, priv_override=_nominal_priv)
            print(f"  [EVAL] epoch {epoch:5d} | nominal MED = {med_nominal:.4f} m "
                  f"(target ≤0.037m)")
            tb_writer.add_scalar("eval/med_nominal", med_nominal, epoch)

            # Gate check
            if epoch == 5000 and med_nominal > 0.055:
                print(f"\n  [ABORT GATE] epoch 5000: nominal MED {med_nominal:.4f} > 0.055m")
                print("  Re-read M2_spec.md §F1 before next run. Stopping.")
                break
            if epoch == 10000 and med_nominal > 0.045:
                print(f"\n  [ABORT GATE] epoch 10000: nominal MED {med_nominal:.4f} > 0.045m")
                print("  Re-read M2_spec.md §F1 before next run. Stopping.")
                break

        if epoch % checkpoint_interval == 0 and epoch > 0:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(
                str(ckpt_dir / f"epoch_{epoch:06d}"),
                {"actor": actor_state, "critic": critic_state},
            )
            print(f"  Checkpoint saved: epoch_{epoch:06d}")

    # Final checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        str(ckpt_dir / "final"),
        {"actor": actor_state, "critic": critic_state},
    )
    print(f"\nTraining complete. Final checkpoint: {ckpt_dir}/final")
    print(f"Total time: {(time.time() - start_time) / 3600:.1f} hours")
    tb_writer.close()


if __name__ == "__main__":
    main()
