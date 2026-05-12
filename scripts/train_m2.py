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
    parser.add_argument("--resume_from", default=None,
                        help="Checkpoint path to resume from (full optimizer state required)")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch number the resume checkpoint corresponds to")
    parser.add_argument("--resume_med_nominal", type=float, default=None,
                        help="Nominal MED at start_epoch, used as trend-gate reference on resume")
    parser.add_argument("--trend_gate_improvement", type=float, default=0.010,
                        help="Min MED improvement required over the 5k-epoch trend window "
                             "(default 0.010 for fresh runs; use 0.005 for DR-resumed runs)")
    parser.add_argument("--env", choices=["m2", "depth"], default="m2",
                        help="Environment backend: m2 = standard VecEnv, "
                             "depth = DepthVecEnv with MJWarp rendering (SC-6)")
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

    class SimpleConfig:
        def __init__(self, d):
            self.__dict__.update(d)

    env_cfg = SimpleConfig({"num_envs": flat_cfg["num_envs"]})

    if args.env == "depth":
        from iron_man_drone.envs.quadrotor_env_depth import DepthVecEnv
        n_obstacles = cfg["env"].get("n_obstacles", 4)
        env = DepthVecEnv(
            env_cfg,
            n_obstacles=n_obstacles,
            fault_prob=fault_prob,
            eta_min=eta_min,
            mass_lo=mass_lo,
            mass_hi=mass_hi,
        )
        print(f"Environment loaded: DepthVecEnv. {flat_cfg['num_envs']} parallel envs.")
        print(f"n_obstacles={n_obstacles}  DR: fault_prob={fault_prob}, "
              f"eta_min={eta_min}, mass=[{mass_lo},{mass_hi}]")
    else:
        from iron_man_drone.envs.quadrotor_env import VecEnv
        env = VecEnv(
            env_cfg,
            fault_prob=fault_prob,
            eta_min=eta_min,
            mass_lo=mass_lo,
            mass_hi=mass_hi,
        )
        print(f"Environment loaded: VecEnv. {flat_cfg['num_envs']} parallel envs.")
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

    if args.resume_from:
        import orbax.checkpoint as ocp
        _checkpointer = ocp.PyTreeCheckpointer()
        _ckpt = _checkpointer.restore(
            args.resume_from,
            item={"actor": actor_state, "critic": critic_state},
        )
        actor_state, critic_state = _ckpt["actor"], _ckpt["critic"]
        print(f"Resumed from: {args.resume_from}")
        print(f"  start_epoch={args.start_epoch}, "
              f"resume_med_nominal={args.resume_med_nominal}, "
              f"trend_gate_improvement={args.trend_gate_improvement}")

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

    _eval_traj = make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal")
    _drone_id  = env.mj_model.body("drone").id

    # Eval priv_states — physics (rotor_efficiency, mass_scale) must match
    # what the actor observes so there is no mismatch between sim and policy input.
    _nominal_priv = jnp.concatenate([jnp.ones(4), jnp.ones(1), jnp.zeros(3)])
    _fault70_priv = jnp.concatenate([jnp.array([0.7, 1.0, 1.0, 1.0]),
                                     jnp.ones(1), jnp.zeros(3)])  # rotor 0 at η=0.70

    # Precompute all EPISODE_STEPS reference XY positions once — avoids creating
    # 1000 separate JIT traces (one per unique integer step) inside the eval loop.
    _eval_ref_xy = jnp.array(
        jax.vmap(lambda s: get_reference_pos(_eval_traj, s)[:2])(
            jnp.arange(EPISODE_STEPS, dtype=jnp.int32)
        )
    )  # (EPISODE_STEPS, 2), constant

    @jax.jit
    def _eval_episode_jit(actor_params, reset_key, priv_override):
        """Full 1000-step episode compiled as a single XLA program via lax.scan.
        Steps after done=True are masked out of the MED calculation."""
        state, _, _ = env._reset_fn(reset_key)
        state = state._replace(
            traj=_eval_traj,
            priv_state=priv_override,
            rotor_efficiency=priv_override[:4],
            mass_scale=priv_override[4],
        )
        a_obs, _ = _build_obs(state.mjx_data, _eval_traj, state.step, _drone_id, priv_override)

        def scan_step(carry, ref_xy):
            state, a_obs, already_done = carry
            mean, _ = actor_state.apply_fn(actor_params, a_obs[None])
            action = mean[0]
            new_state, new_a_obs, _, _, done = env._step_fn(state, action)
            new_state = new_state._replace(traj=_eval_traj)
            pos_xy = new_state.mjx_data.xpos[_drone_id, :2]
            error = jnp.linalg.norm(pos_xy - ref_xy)
            active = ~already_done
            masked_error = jnp.where(active, error, 0.0)
            return (new_state, new_a_obs, already_done | done), (masked_error, active)

        _, (errors, active_mask) = jax.lax.scan(
            scan_step, (state, a_obs, jnp.bool_(False)), _eval_ref_xy
        )
        n_active = jnp.sum(active_mask)
        med = jnp.where(n_active > 0,
                        jnp.sum(errors) / n_active.astype(jnp.float32),
                        jnp.nan)
        return med

    def _run_med_eval(a_state, key, priv_override=None):
        key, rk = jax.random.split(key)
        priv = _nominal_priv if priv_override is None else priv_override
        return float(_eval_episode_jit(a_state.params, rk, priv))

    print("Pre-warming eval JIT (compiles full 1000-step scan — takes ~1 min)...")
    _ = _eval_episode_jit(actor_state.params, jax.random.PRNGKey(999), _nominal_priv)
    print("Eval JIT pre-warmed.")

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
    _med_at_5k = None      # set at epoch 5k for 5k→10k trend gate (fresh runs only)
    _last_med_nominal = None  # most recent nominal eval (for post-loop trend gate)
    start_epoch = args.start_epoch
    _trend_gate_improvement = args.trend_gate_improvement
    _trend_ref_med = args.resume_med_nominal   # None on fresh run, 0.100 on resume

    print(f"\nStarting training: {start_epoch}–{flat_cfg['total_epochs']} epochs")
    print(f"Steps per epoch: {flat_cfg['num_envs'] * flat_cfg['horizon']}")
    if start_epoch > 0:
        print(f"Trend gate: improvement ≥ {_trend_gate_improvement:.3f} m "
              f"from epoch {start_epoch} (ref MED={_trend_ref_med})")

    for epoch in range(start_epoch, flat_cfg["total_epochs"]):
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
            key, eval_key1, eval_key2 = jax.random.split(key, 3)

            med_nominal = _run_med_eval(actor_state, eval_key1, priv_override=_nominal_priv)
            med_fault70 = _run_med_eval(actor_state, eval_key2, priv_override=_fault70_priv)

            print(f"  [EVAL] epoch {epoch:5d} | "
                  f"nominal MED = {med_nominal:.4f} m (≤0.037) | "
                  f"fault η=0.70 MED = {med_fault70:.4f} m (≤0.060)")
            tb_writer.add_scalar("eval/med_nominal",  med_nominal,  epoch)
            tb_writer.add_scalar("eval/med_fault_eta70", med_fault70, epoch)
            _last_med_nominal = med_nominal

            # ── Gates only active on fresh runs (start_epoch == 0) ────────────
            # Resumed runs skip these; their trend gate is checked post-loop.
            if start_epoch == 0:
                # Absolute gates (calibrated vs M1.3 clean run + 1.5× DR margin):
                #   M1.3 epoch 5000  = 0.091 m → 1.5× = 0.137 m → gate 0.130 m
                #   M1.3 epoch 10000 = 0.081 m → 1.5× = 0.122 m → gate 0.115 m
                if epoch == 5000:
                    _med_at_5k = med_nominal
                    if med_nominal > 0.130:
                        print(f"\n  [ABORT GATE] epoch 5000: nominal MED {med_nominal:.4f} > 0.130 m")
                        print("  (M1.3 was 0.091 m at epoch 5k; 1.5× ceiling = 0.137 m)")
                        print("  Re-read M2_spec.md §F1 before next run. Stopping.")
                        break
                if epoch == 10000:
                    if med_nominal > 0.115:
                        print(f"\n  [ABORT GATE] epoch 10000: nominal MED {med_nominal:.4f} > 0.115 m")
                        print("  (M1.3 was 0.081 m at epoch 10k; 1.5× ceiling = 0.122 m)")
                        print("  Re-read M2_spec.md §F1 before next run. Stopping.")
                        break
                    # Trend gate: epochs 5k→10k must improve by ≥ trend_gate_improvement.
                    # Default 0.010 m matches M1.3's measured improvement in this window.
                    # See notes/M2_trend_gate_recalibration.md for DR-run recalibration.
                    if _med_at_5k is not None:
                        improvement = _med_at_5k - med_nominal
                        if improvement < _trend_gate_improvement:
                            print(f"\n  [ABORT GATE] epoch 5k→10k trend: MED improved only "
                                  f"{improvement:.4f} m "
                                  f"({_med_at_5k:.4f} → {med_nominal:.4f}, "
                                  f"need ≥ {_trend_gate_improvement:.3f} m)")
                            print("  Policy has plateaued. M1.3 improved 0.010 m in this window.")
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

    # ── Final eval (always, regardless of whether loop ran to completion) ───────
    key, eval_key1, eval_key2 = jax.random.split(key, 3)
    med_nominal_final  = _run_med_eval(actor_state, eval_key1, priv_override=_nominal_priv)
    med_fault70_final  = _run_med_eval(actor_state, eval_key2, priv_override=_fault70_priv)
    final_epoch = epoch + 1  # range() is 0-indexed; last completed epoch + 1
    print(f"\n  [FINAL EVAL] epoch {final_epoch} | "
          f"nominal MED = {med_nominal_final:.4f} m | "
          f"fault η=0.70 MED = {med_fault70_final:.4f} m")
    tb_writer.add_scalar("eval/med_nominal",     med_nominal_final,  final_epoch)
    tb_writer.add_scalar("eval/med_fault_eta70", med_fault70_final,  final_epoch)
    _last_med_nominal = med_nominal_final

    # ── Post-loop trend gate (resumed runs only) ──────────────────────────────
    # Fresh runs had inline gates at epochs 5k and 10k.
    # Resumed runs check here: improvement over the resumed window must be ≥ threshold.
    if start_epoch > 0 and _trend_ref_med is not None:
        improvement = _trend_ref_med - _last_med_nominal
        passed = improvement >= _trend_gate_improvement
        gate_status = "PASSED" if passed else "FIRED"
        print(f"\n  [TREND GATE] epoch {start_epoch}→{final_epoch}: "
              f"MED {_trend_ref_med:.4f} → {_last_med_nominal:.4f} m "
              f"(improved {improvement:.4f} m, need ≥ {_trend_gate_improvement:.3f} m) "
              f"— {gate_status}")
        if not passed:
            print("  Policy plateaued over the resumed window.")
            print("  See notes/M2_trend_gate_recalibration.md for context.")
            print("  Recommendation: ship Phase 2 on this checkpoint anyway (user pre-approved).")

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
