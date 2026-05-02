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
    Pre-training sanity: entropy must be < 10% of total reward magnitude.
    Run with a random policy for a few steps and log the ratio.
    """
    print("\n--- Pre-training sanity check ---")
    import distrax
    from iron_man_drone.utils.domain_randomization import sample_env_params

    keys = jax.random.split(key, cfg["num_envs"])
    kf_mults = jnp.array([
        sample_env_params(k)["kf_multiplier"]
        for k in jax.random.split(key, cfg["num_envs"])
    ])
    states, a_obs, c_obs = env.batch_reset(keys, kf_mults)

    # Collect 10 steps with random policy
    total_reward = 0.0
    total_entropy = 0.0
    n_steps = 10

    for _ in range(n_steps):
        key, act_key, step_key = jax.random.split(key, 3)
        mean, log_std = actor_state.apply_fn(actor_state.params, a_obs)
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        actions = dist.sample(seed=act_key)
        entropy = dist.entropy().mean()

        states, a_obs, c_obs, rewards, dones = env.batch_step(states, actions, kf_mults)
        total_reward += float(rewards.mean())
        total_entropy += float(entropy) * cfg["entropy_coeff"]

    mean_reward = total_reward / n_steps
    mean_entropy_contrib = total_entropy / n_steps
    ratio = abs(mean_entropy_contrib) / (abs(mean_reward) + 1e-8)

    print(f"  Mean reward:             {mean_reward:.4f}")
    print(f"  Mean entropy contribution: {mean_entropy_contrib:.4f}")
    print(f"  Entropy / reward ratio:  {ratio:.3f}")

    if ratio > 0.10:
        print(f"\nWARNING: entropy contribution is {ratio*100:.1f}% of reward.")
        print("This violates the <10% constraint from notes/M1_hypothesis.md.")
        print("Reduce entropy_coeff before training.")
        print("Continuing anyway — monitor carefully.")
    else:
        print(f"  [OK] Entropy < 10% of reward.")
    print()


def main():
    check_hypothesis_gate()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/m1_baseline/config.yaml"))
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--total_epochs", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.num_envs:
        cfg["env"]["num_envs"] = args.num_envs
    if args.total_epochs:
        cfg["ppo"]["total_epochs"] = args.total_epochs

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
        "num_minibatches": cfg["ppo"]["minibatch_size"],
        "ppo_epochs": 5,
        "actor_obs_dim": cfg["observation"]["actor_dim"],
        "critic_obs_dim": cfg["observation"]["critic_dim"],
        "action_dim": cfg["action"]["dim"],
        "hidden_dim": cfg["network"]["hidden_dim"],
        "num_layers": cfg["network"]["num_layers"],
    }

    # Run directory
    run_name = args.run_name or f"m1_{int(time.time())}"
    run_dir = REPO_ROOT / "experiments" / "m1_baseline" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Freeze config
    import shutil
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
    from iron_man_drone.utils.domain_randomization import sample_env_params

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

    # Sanity check before training
    sanity_check_entropy_ratio(actor_state, critic_state, env, flat_cfg, key)

    # Initial reset
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, flat_cfg["num_envs"])
    kf_mults = jnp.array([
        sample_env_params(k)["kf_multiplier"]
        for k in jax.random.split(key, flat_cfg["num_envs"])
    ])
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

    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=str(run_dir / "logs"))

    # Training loop
    print(f"\nStarting training: {flat_cfg['total_epochs']} epochs")
    print(f"Steps per epoch: {flat_cfg['num_envs'] * flat_cfg['horizon']}")
    total_steps = 0
    eval_interval = cfg.get("experiment", {}).get("eval_every", 500)
    checkpoint_interval = cfg.get("experiment", {}).get("checkpoint_every", 1000)
    log_interval = cfg.get("experiment", {}).get("log_every", 10)

    # JIT-compile the update function
    ppo_update_jit = jax.jit(
        lambda as_, cs_, tr_, lv_, k_: ppo_update(as_, cs_, tr_, lv_, k_, ppo_cfg)
    )

    start_time = time.time()

    for epoch in range(flat_cfg["total_epochs"]):
        key, rollout_key, update_key, kf_key = jax.random.split(key, 4)

        # Re-sample DR params periodically
        if epoch % 10 == 0:
            kf_mults = jnp.array([
                sample_env_params(k)["kf_multiplier"]
                for k in jax.random.split(kf_key, flat_cfg["num_envs"])
            ])

        # Collect rollout
        t0 = time.time()
        transitions, env_states, actor_obs, critic_obs = collect_rollout(
            actor_state, critic_state,
            env_states, actor_obs, critic_obs,
            kf_mults,
            env.step, env.reset,
            rollout_key, ppo_cfg,
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

            print(
                f"Epoch {epoch:5d} | "
                f"reward {mean_reward:7.4f} | "
                f"actor_loss {metrics_cpu['actor_loss']:7.4f} | "
                f"critic_loss {metrics_cpu['critic_loss']:7.4f} | "
                f"entropy {metrics_cpu['entropy']:6.4f} | "
                f"fps {fps:8.0f}"
            )

            tb_writer.add_scalar("train/reward", mean_reward, epoch)
            tb_writer.add_scalar("train/actor_loss", metrics_cpu["actor_loss"], epoch)
            tb_writer.add_scalar("train/critic_loss", metrics_cpu["critic_loss"], epoch)
            tb_writer.add_scalar("train/entropy", metrics_cpu["entropy"], epoch)
            tb_writer.add_scalar("train/fps", fps, epoch)

            if use_wandb:
                import wandb
                wandb.log({"epoch": epoch, "reward": mean_reward, **metrics_cpu, "fps": fps})

        # Checkpointing
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
    print(f"Run eval: python scripts/eval_m1.py --checkpoint {ckpt_dir}/final")
    tb_writer.close()


if __name__ == "__main__":
    main()
