"""
M1 training entry point — SimpleFlight recipe on MuJoCo MJX.

Usage:
  python scripts/train_m1.py
  python scripts/train_m1.py --config experiments/m1_baseline/config.yaml
  python scripts/train_m1.py --num_envs 512  # override for smaller GPU

GATE: notes/M1_hypothesis.md must exist before this runs.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent


def check_hypothesis_gate():
    hyp = REPO_ROOT / "notes" / "M1_hypothesis.md"
    if not hyp.exists() or hyp.stat().st_size < 100:
        print("ERROR: notes/M1_hypothesis.md is missing or too short.")
        print("Write the hypothesis doc before training. This is a hard gate.")
        sys.exit(1)
    print("[GATE PASSED] Hypothesis doc found.")


def sanity_check_entropy_ratio(actor_state, critic_state, env, cfg, key):
    """
    Pre-training sanity: entropy contribution must be < 10% of reward.
    Run 10 steps with the initial random policy, print numbers loudly.
    Does NOT abort — the user reads the numbers and decides.
    """
    import distrax
    from iron_man_drone.utils.domain_randomization import sample_kf_multiplier

    keys     = jax.random.split(key, cfg["num_envs"])
    kf_mults = jax.jit(jax.vmap(sample_kf_multiplier))(jax.random.split(key, cfg["num_envs"]))
    states, a_obs, c_obs = env.batch_reset(keys, kf_mults)

    total_reward = 0.0
    total_entropy_nats = 0.0
    n_steps = 10

    for _ in range(n_steps):
        key, act_key = jax.random.split(key)
        mean, log_std = actor_state.apply_fn(actor_state.params, a_obs)
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        actions = dist.sample(seed=act_key)
        entropy_nats = float(dist.entropy().mean())

        states, a_obs, c_obs, rewards, dones = env.batch_step(states, actions, kf_mults)
        total_reward += float(rewards.mean())
        total_entropy_nats += entropy_nats

    mean_reward = total_reward / n_steps
    mean_entropy_nats = total_entropy_nats / n_steps
    entropy_contrib = cfg["entropy_coeff"] * mean_entropy_nats
    ratio = abs(entropy_contrib) / (abs(mean_reward) + 1e-8)

    print()
    print("=" * 62)
    print("   PRE-TRAINING ENTROPY / REWARD SANITY CHECK")
    print("=" * 62)
    print(f"  Mean reward per step:       {mean_reward:8.4f}")
    print(f"  Policy entropy:             {mean_entropy_nats:8.4f} nats")
    print(f"  Entropy coeff:              {cfg['entropy_coeff']:8.6f}")
    print(f"  Entropy contribution:       {entropy_contrib:8.4f}  (coeff × entropy)")
    print(f"  Entropy / reward ratio:     {ratio*100:7.2f}%   (target: < 10%)")
    print()
    if ratio > 0.10:
        print("  !! WARNING: entropy contribution is ABOVE 10% of reward !!")
        print("  !! Check entropy_coeff and reward scaling before training. !!")
        print("  !! Continuing anyway — your call. See notes/M1_hypothesis.md !!")
    else:
        print(f"  OK — entropy contribution is {ratio*100:.1f}% of reward (< 10%)")
    print("=" * 62)
    print()


def main():
    check_hypothesis_gate()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/m1_baseline/config.yaml"))
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--total_epochs", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint dir to resume from (e.g. .../checkpoints/epoch_002500)")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch number to resume from (must match --resume checkpoint)")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.num_envs:
        cfg["env"]["num_envs"] = args.num_envs
    if args.total_epochs:
        cfg.setdefault("experiment", {})["total_epochs"] = args.total_epochs

    # Flatten nested config for convenience
    flat_cfg = {
        "num_envs": cfg["env"]["num_envs"],
        "horizon": cfg["ppo"]["horizon"],
        "total_epochs": cfg.get("experiment", {}).get("total_epochs", 15000),
        "actor_lr": cfg["ppo"]["actor_lr"],
        "critic_lr": cfg["ppo"]["critic_lr"],
        "entropy_coeff": cfg["ppo"]["entropy_coeff"],
        "gamma": cfg["ppo"]["gamma"],
        "gae_lambda": cfg["ppo"]["gae_lambda"],
        "clip_eps": cfg["ppo"]["clip_eps"],
        "critic_updates": cfg["ppo"]["critic_updates"],
        "max_grad_norm": cfg["ppo"]["max_grad_norm"],
        "num_minibatches": cfg["ppo"]["num_minibatches"],
        "ppo_epochs": 5,
        "actor_obs_dim": cfg["observation"]["actor_dim"],
        "critic_obs_dim": cfg["observation"]["critic_dim"],
        "action_dim": cfg["action"]["dim"],
        "hidden_dim": cfg["network"]["hidden_dim"],
        "num_layers": cfg["network"]["num_layers"],
    }

    # Run directory
    exp_name = cfg.get("experiment", {}).get("name", "m1_baseline")
    if args.resume:
        # Resume: run_dir is grandparent of checkpoint (checkpoints/epoch_XXXXXX → run_dir)
        args.resume = str(Path(args.resume).resolve())
        run_dir = Path(args.resume).parent.parent
        run_name = run_dir.name
        print(f"Resuming from checkpoint: {args.resume}")
        print(f"Run directory: {run_dir}")
    else:
        run_name = args.run_name or f"{exp_name}_{int(time.time())}"
        run_dir = REPO_ROOT / "experiments" / exp_name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run directory: {run_dir}")

    # Freeze config (skip on resume — already frozen)
    import shutil
    if not args.resume:
        shutil.copy(args.config, run_dir / "config_frozen.yaml")

    # JAX setup
    print(f"\nJAX devices: {jax.devices()}")
    assert len(jax.devices()) > 0, "No JAX devices found"
    key = jax.random.PRNGKey(cfg.get("experiment", {}).get("seed", 42))

    # Build environment and policy
    from iron_man_drone.envs.quadrotor_env import VecEnv

    class SimpleConfig:
        def __init__(self, d):
            self.__dict__.update(d)

    env_cfg = SimpleConfig({"num_envs": flat_cfg["num_envs"]})
    env = VecEnv(env_cfg)
    print(f"Environment loaded. {flat_cfg['num_envs']} parallel envs.")

    from iron_man_drone.policy.ppo import PPOConfig, create_train_states, collect_rollout, ppo_update
    from iron_man_drone.utils.domain_randomization import sample_env_params, sample_kf_multiplier

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

    if args.resume:
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt = checkpointer.restore(args.resume)
        # Restore params only — opt_state deserializes as plain dicts (optax namedtuples
        # can't round-trip through Orbax without schema), so we let Adam restart fresh.
        # Policy weights are what matter; momentum loss over ~500 epochs is negligible.
        actor_state  = actor_state.replace(params=ckpt["actor"]["params"])
        critic_state = critic_state.replace(params=ckpt["critic"]["params"])
        print(f"Checkpoint restored (epoch {args.start_epoch}).")
    else:
        # Sanity check before fresh training
        sanity_check_entropy_ratio(actor_state, critic_state, env, flat_cfg, key)

    # Vectorized kf_multiplier sampling (vmapped, not Python for-loop)
    _sample_kf_batch = jax.jit(jax.vmap(sample_kf_multiplier))

    # Initial reset
    key, reset_key, kf_key0 = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_key, flat_cfg["num_envs"])
    kf_mults   = _sample_kf_batch(jax.random.split(kf_key0, flat_cfg["num_envs"]))
    env_states, actor_obs, critic_obs = env.batch_reset(reset_keys, kf_mults)

    # Logging setup
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project="iron-man-drone-m1",
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
        """Minimal drop-in for SummaryWriter that writes to a CSV."""
        def __init__(self, path, append=False):
            self._path = path
            mode = "a" if append else "w"
            self._file = open(path, mode, newline="", buffering=1)
            self._writer = None
            self._has_header = append and path.exists() and path.stat().st_size > 0
        def add_scalar(self, tag, value, step):
            row = {"step": step, "tag": tag, "value": value}
            if self._writer is None:
                self._writer = _csv.DictWriter(self._file, fieldnames=row.keys())
                if not self._has_header:
                    self._writer.writeheader()
            self._writer.writerow(row)
        def close(self):
            self._file.close()

    (run_dir / "logs").mkdir(exist_ok=True)
    csv_path = run_dir / "logs" / "metrics.csv"
    tb_writer = _CSVWriter(csv_path, append=bool(args.resume))

    # In-training MED eval setup (figure-eight normal, single env, deterministic)
    from iron_man_drone.envs.quadrotor_env import (
        EPISODE_STEPS, _build_obs, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT, EnvState,
    )
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, get_reference_pos,
    )
    _eval_traj    = make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal")
    _eval_reset   = jax.jit(env._reset_fn)
    _eval_step    = jax.jit(env._step_fn)
    _drone_id     = env.mj_model.body("drone").id

    # Pre-warm eval JIT kernels before training starts so their compilation
    # doesn't evict the training vmapped kernel from the XLA cache mid-run.
    print("Pre-warming eval JIT...")
    _pw_key = jax.random.PRNGKey(999)
    _pw_state, _pw_obs, _ = _eval_reset(_pw_key, jnp.ones(()))
    _pw_state = _pw_state._replace(traj=_eval_traj)
    _pw_mean, _ = actor_state.apply_fn(actor_state.params, _pw_obs[None])
    _pw_state, _, _, _, _ = _eval_step(_pw_state, _pw_mean[0], jnp.ones(()))
    del _pw_key, _pw_state, _pw_obs, _pw_mean
    print("Eval JIT pre-warmed.")

    def _run_med_eval(actor_state, key):
        """One deterministic episode on figure-eight-normal. Returns xy MED (m)."""
        key, rk = jax.random.split(key)
        state, a_obs, _ = _eval_reset(rk, jnp.ones(()))
        state = state._replace(traj=_eval_traj)
        a_obs, _ = _build_obs(state.mjx_data, _eval_traj, state.step, _drone_id)

        positions, refs = [], []
        for si in range(EPISODE_STEPS):
            mean, _ = actor_state.apply_fn(actor_state.params, a_obs[None])
            action   = mean[0]
            state, a_obs, _, _, done = _eval_step(state, action, jnp.ones(()))
            state    = state._replace(traj=_eval_traj)
            positions.append(np.array(state.mjx_data.xpos[_drone_id, :2]))
            refs.append(np.array(get_reference_pos(_eval_traj, jnp.int32(si))[:2]))
            if bool(done): break

        positions = np.array(positions)
        refs      = np.array(refs)
        return float(np.linalg.norm(positions - refs, axis=1).mean())

    # Training loop
    start_epoch = args.start_epoch
    print(f"\nStarting training: epochs {start_epoch}–{flat_cfg['total_epochs']}")
    print(f"Steps per epoch: {flat_cfg['num_envs'] * flat_cfg['horizon']}")
    checkpoint_interval = cfg.get("experiment", {}).get("checkpoint_every", 1000)
    log_interval        = cfg.get("experiment", {}).get("log_every", 10)
    med_eval_interval   = cfg.get("experiment", {}).get("eval_every", 1000)

    # JIT-compile both rollout and update
    collect_rollout_jit = jax.jit(
        lambda as_, cs_, es_, ao_, co_, kf_, k_: collect_rollout(
            as_, cs_, es_, ao_, co_, kf_, env.step, env.reset, k_, ppo_cfg
        )
    )
    ppo_update_jit = jax.jit(
        lambda as_, cs_, tr_, lv_, k_: ppo_update(as_, cs_, tr_, lv_, k_, ppo_cfg)
    )

    start_time = time.time()
    total_steps = 0  # fps tracks throughput since (re)start, not cumulative

    for epoch in range(start_epoch, flat_cfg["total_epochs"]):
        key, rollout_key, update_key, kf_key = jax.random.split(key, 4)

        # Re-sample DR params periodically (vectorized)
        if epoch % 10 == 0:
            kf_mults = _sample_kf_batch(jax.random.split(kf_key, flat_cfg["num_envs"]))

        # Collect rollout (JIT-compiled)
        transitions, env_states, actor_obs, critic_obs = collect_rollout_jit(
            actor_state, critic_state,
            env_states, actor_obs, critic_obs,
            kf_mults, rollout_key,
        )

        # Get last value for GAE bootstrap
        last_value = critic_state.apply_fn(critic_state.params, critic_obs)

        # PPO update
        actor_state, critic_state, metrics = ppo_update_jit(
            actor_state, critic_state, transitions, last_value, update_key
        )
        total_steps += flat_cfg["num_envs"] * flat_cfg["horizon"]

        # Logging
        if epoch % log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            mean_reward = float(transitions.reward.mean())
            metrics_cpu = jax.tree_util.tree_map(float, metrics)
            done_rate = float(transitions.done.mean())
            # mean steps between episode ends; clamp to avoid div-by-zero early in training
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
            tb_writer.add_scalar("train/episode_length_steps", mean_ep_len, epoch)
            tb_writer.add_scalar("train/fps", fps, epoch)

            if use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch, "reward": mean_reward,
                    "episode_length_steps": mean_ep_len,
                    **metrics_cpu, "fps": fps,
                })

        # MED eval on figure-eight-normal (every 1000 epochs)
        if epoch > 0 and epoch % med_eval_interval == 0:
            key, eval_key = jax.random.split(key)
            med = _run_med_eval(actor_state, eval_key)
            print(f"  [EVAL] epoch {epoch:5d} | figure_eight_normal MED = {med:.4f} m")
            tb_writer.add_scalar("eval/med_figure8_normal", med, epoch)

        # Checkpointing (every 500 epochs, skip the resume epoch — already on disk)
        if epoch % checkpoint_interval == 0 and epoch > start_epoch:
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
    print(f"Run eval: python scripts/eval_m1.py --checkpoint {ckpt_dir}/final")
    tb_writer.close()


if __name__ == "__main__":
    main()
