"""
M3 training script — joint fault+obstacle policy from scratch.

Epoch = 1 rollout update = horizon × num_envs env-steps (32 × 1024 = 32,768).
At 32.6k fps: 500M env-steps ≈ 15,259 epochs ≈ 4.3 hours.

H1 abort gate check at 30M env-steps (~916 epochs).
1-hour checkpoint pause at ~117M env-steps (~3,576 epochs).

Hypothesis doc gate: notes/M3_hypothesis.md must exist and be non-trivial.
Run name: set M3_RUN_NAME env var or pass --run_name flag.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import distrax

# Persistent XLA compilation cache — survives WSL restarts, saves 20-30 min
# compilation overhead on subsequent runs.
_JAX_CACHE = Path.home() / ".cache" / "jax_m3_xla"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iron_man_drone.envs.quadrotor_env_m3 import (
    M3VecEnv, M3EnvState,
    ACTOR_OBS_DIM, CRITIC_OBS_DIM, ENCODER_DIM, OBS_BASE_DIM, K_NEAREST,
    compute_depth_bins_batch,
    build_full_obs,
    D_CRASH,
    EPISODE_STEPS,
)
from iron_man_drone.envs.trajectories import (
    make_figure_eight_trajectory, DT,
)
from iron_man_drone.policy.networks import Actor, Critic
from iron_man_drone.policy.encoder import (
    AdaptationEncoder, build_history_window,
    denormalize_e_hat, normalize_e_t, E_T_DIM,
    H as ENC_HISTORY, OBS_DIM as ENC_OBS_DIM, ACTION_DIM, WINDOW_DIM,
)
from iron_man_drone.policy.ppo import Transition, PPOConfig, ppo_update
from iron_man_drone.utils.scene_generator import (
    sample_mode, sample_scene, TRAINING_MODES, TRAINING_WEIGHTS,
)
from iron_man_drone.envs.quadrotor_env import LOOKAHEAD_N, LOOKAHEAD_DT_STEPS


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class M3Config:
    num_envs:       int   = 1024
    horizon:        int   = 32
    total_env_steps: int  = 500_000_000

    # PPO
    actor_lr:       float = 3e-4
    critic_lr:      float = 1e-4
    encoder_lr:     float = 1e-4
    gamma:          float = 0.99
    gae_lambda:     float = 0.95
    clip_eps:       float = 0.2
    entropy_coeff:  float = 1e-3
    max_grad_norm:  float = 0.5
    critic_updates: int   = 16
    ppo_epochs:     int   = 5
    num_minibatches: int  = 8

    # DR
    fault_prob:     float = 0.7
    eta_min:        float = 0.5
    mass_lo:        float = 0.8
    mass_hi:        float = 1.2

    # Curriculum density_mult schedule (env-steps)
    curriculum_sparse_until: int   = 10_000_000   # density_mult=0.5 below
    curriculum_full_until:   int   = 50_000_000   # density_mult ramps 0.5→1.0
    # density_mult=1.0 after; 20% of episodes at 1.5x handled in batch_reset

    # Logging
    log_every_epochs:        int   = 100          # ~3.3M env-steps
    eval_every_epochs_early: int   = 99999         # disabled — eval after training via eval_m3.py
    eval_every_epochs_late:  int   = 99999         # disabled
    ckpt_every_epochs:       int   = 50             # ~1.6M env-steps (~5 min at 6000 fps)
    eval_switch_epoch:       int   = 7629          # epoch where early→late eval switches

    # Gates
    h1_check_epoch:      int   = 916    # 30M env-steps
    h1_crash_threshold:  float = 0.70   # abort if crash rate > this at h1_check_epoch
    h1_trend_window:     int   = 150    # epochs for trend check
    one_hour_epoch:      int   = 3576   # ~117M env-steps ≈ 1h at 32.6k fps

    # Seeds
    seed: int = 42

    @property
    def env_steps_per_epoch(self) -> int:
        return self.horizon * self.num_envs   # 32,768


def density_mult_at(env_steps: int, cfg: M3Config) -> float:
    if env_steps < cfg.curriculum_sparse_until:
        return 0.5
    elif env_steps < cfg.curriculum_full_until:
        frac = (env_steps - cfg.curriculum_sparse_until) / (
            cfg.curriculum_full_until - cfg.curriculum_sparse_until
        )
        return 0.5 + 0.5 * frac
    return 1.0


# ── Encoder supervised update ──────────────────────────────────────────────────

def _encoder_loss_fn(enc_params, enc_apply_fn, windows, priv_targets):
    """MSE on normalized privileged state prediction."""
    z_hat_norm = jax.vmap(lambda w: enc_apply_fn(enc_params, w))(windows)
    targets_norm = jax.vmap(normalize_e_t)(priv_targets)
    return jnp.mean((z_hat_norm - targets_norm) ** 2)


@jax.jit
def encoder_update(enc_state, windows, priv_targets):
    def loss_fn(params):
        return _encoder_loss_fn(params, enc_state.apply_fn, windows, priv_targets)
    loss, grads = jax.value_and_grad(loss_fn)(enc_state.params)
    enc_state = enc_state.apply_gradients(grads=grads)
    return enc_state, loss


# ── Encoder inference helpers ──────────────────────────────────────────────────

_build_window_vmap = jax.jit(
    jax.vmap(build_history_window)
)

@jax.jit
def apply_encoder_vmap(enc_params, encoder, windows):
    z_norm = jax.vmap(lambda w: encoder.apply(enc_params, w))(windows)
    return jax.vmap(denormalize_e_hat)(z_norm)

@jax.jit
def build_full_obs_vmap(base_obs, z_hat, depth_bins, k, obs_dists):
    return jax.vmap(build_full_obs)(base_obs, z_hat, depth_bins, k, obs_dists)


# ── Eval ───────────────────────────────────────────────────────────────────────

def run_quick_eval(
    env: M3VecEnv,
    actor_state, enc_state,
    cfg: M3Config,
    key: jnp.ndarray,
    n_episodes: int = 32,
    include_fault: bool = True,
) -> dict:
    """
    Quick crash-rate + MED eval on a fixed figure-eight trajectory.
    Returns dict: crash_rate, med_nominal, med_fault70 (if include_fault).
    """
    results = {}
    encoder = enc_state.apply_fn

    for fault_label, eta in ([("nominal", 1.0)] + ([("fault70", 0.70)] if include_fault else [])):
        rng = np.random.default_rng(seed=0)
        keys = jax.random.split(key, n_episodes)

        # Fixed figure-eight trajectory, no obstacles for regression eval
        traj = make_figure_eight_trajectory(DT, EPISODE_STEPS, speed="normal")

        crash_count = 0
        total_pos_err = 0.0
        total_steps = 0

        for ep_key in keys:
            # Reset: no obstacles
            centers = np.full((16, 3), 100.0, dtype=np.float32)
            he      = np.zeros((16, 3), dtype=np.float32)
            n_obs   = jnp.zeros((), dtype=jnp.int32)
            ep_key2 = ep_key[None]
            states, base_obs, priv_state = env._reset_jit(
                ep_key2,
                jnp.array(centers)[None],
                jnp.array(he)[None],
                n_obs[None],
            )
            states  = jax.tree_util.tree_map(lambda x: x[0], states)
            base_obs = base_obs[0]

            # Override fault state for fault eval
            if abs(eta - 1.0) > 0.01:
                # Build priv_state with single rotor at eta
                ps = np.array(priv_state[0])
                ps[0] = eta
                states = states._replace(priv_state=jnp.array(ps))

            crashed = False
            ep_err  = []

            for _ in range(EPISODE_STEPS):
                # Encoder
                window = build_window_single(states.obs_base_buf, states.action_buf)
                z_hat_norm = encoder(enc_state.params, window)
                z_hat = denormalize_e_hat(z_hat_norm)

                # Depth (no obstacles → all bins = 1.0)
                depth_bins = jnp.ones(16, dtype=jnp.float32)
                obs_dists  = jnp.full(K_NEAREST, jnp.inf)
                k_norm     = states.step.astype(jnp.float32) / EPISODE_STEPS

                actor_obs, _ = build_full_obs(base_obs, z_hat, depth_bins, k_norm, obs_dists)
                actor_obs = actor_obs[None]

                mean, log_std = actor_state.apply_fn(actor_state.params, actor_obs)
                action = mean[0]   # deterministic greedy

                states, base_obs, _, _, done = env._raw_step_fn(states, action)

                from iron_man_drone.envs.trajectories import get_reference_pos
                ref = get_reference_pos(states.traj, states.step)
                pos = states.mjx_data.xpos[env.drone_body_id]
                ep_err.append(float(jnp.linalg.norm(pos - ref)))

                if bool(done):
                    # Crashed if below episode timeout
                    if int(states.step) < EPISODE_STEPS - 5:
                        crashed = True
                    break

            crash_count += int(crashed)
            if ep_err:
                total_pos_err += float(np.mean(ep_err))
                total_steps   += 1

        results[f"crash_rate_{fault_label}"] = crash_count / n_episodes
        results[f"med_{fault_label}"]        = total_pos_err / max(total_steps, 1)

    return results


def build_window_single(obs_base_buf, action_buf):
    return build_history_window(obs_base_buf, action_buf)


# ── SIGTERM handler — checkpoint on kill, then exit cleanly ───────────────────

import signal as _signal
_shutdown_requested = False

def _sigterm_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[SIGTERM] Shutdown signal received — will save checkpoint at end of current epoch.")

_signal.signal(_signal.SIGTERM, _sigterm_handler)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: M3Config, run_name: str, exp_dir: Path):
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Hypothesis gate ───────────────────────────────────────────────────────
    hyp_path = ROOT / "notes" / "M3_hypothesis.md"
    if not hyp_path.exists():
        print("BLOCKED: notes/M3_hypothesis.md not found. Write hypothesis doc first.")
        sys.exit(1)
    hyp_text = hyp_path.read_text()
    if hyp_text.count("[fill]") > 2:
        print("BLOCKED: notes/M3_hypothesis.md still has unfilled [fill] placeholders.")
        sys.exit(1)
    print("Hypothesis doc: OK")

    key = jax.random.PRNGKey(cfg.seed)

    # ── Environment ────────────────────────────────────────────────────────────
    print("Initializing M3VecEnv (MJWarp init may take ~30 s)...")

    class _Cfg:
        num_envs = cfg.num_envs

    env = M3VecEnv(
        _Cfg(),
        fault_prob=cfg.fault_prob, eta_min=cfg.eta_min,
        mass_lo=cfg.mass_lo,       mass_hi=cfg.mass_hi,
    )
    print(f"M3VecEnv ready — {cfg.num_envs} envs, drone_body_id={env.drone_body_id}")

    # ── Networks ────────────────────────────────────────────────────────────────
    key, k_act, k_crit, k_enc = jax.random.split(key, 4)

    actor   = Actor()
    critic  = Critic()
    encoder = AdaptationEncoder()

    actor_params  = actor.init(k_act,  jnp.zeros((1, ACTOR_OBS_DIM)))
    critic_params = critic.init(k_crit, jnp.zeros((1, CRITIC_OBS_DIM)))
    enc_params    = encoder.init(k_enc,  jnp.zeros((1, WINDOW_DIM)))

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.actor_lr)),
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.critic_lr)),
    )
    enc_state = TrainState.create(
        apply_fn=encoder.apply,
        params=enc_params,
        tx=optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.encoder_lr)),
    )

    # ── Entropy sanity check ──────────────────────────────────────────────────
    dummy_obs = jnp.zeros((1, ACTOR_OBS_DIM))
    mean, log_std = actor.apply(actor_params, dummy_obs)
    dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    entropy_at_init = float(dist.entropy()[0])
    # Expected reward ≈ W_TRACK * 1.0 + W_SMOOTH * 1.0 + W_SURVIVE = 2.11
    expected_reward = 2.11
    entropy_reward_ratio = (cfg.entropy_coeff * entropy_at_init) / expected_reward
    print(f"Entropy ratio at init: {entropy_reward_ratio:.3f} (must be < 0.10)")
    if entropy_reward_ratio >= 0.10:
        print("WARNING: entropy ratio >= 0.10 — check entropy_coeff or reward scale")

    gae_gamma   = cfg.gamma
    gae_lambda  = cfg.gae_lambda
    clip_eps    = cfg.clip_eps
    entropy_c   = cfg.entropy_coeff

    # ── JIT-compiled helpers ──────────────────────────────────────────────────
    @jax.jit
    def _sample_actions(actor_params, actor_obs, key):
        mean, log_std = actor.apply(actor_params, actor_obs)
        d = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        acts = d.sample(seed=key)
        lps  = d.log_prob(acts)
        return acts, lps

    @jax.jit
    def _get_values(critic_params, critic_obs):
        return critic.apply(critic_params, critic_obs)

    @jax.jit
    def _build_encoder_windows(obs_base_buf, action_buf):
        return jax.vmap(build_history_window)(obs_base_buf, action_buf)

    @jax.jit
    def _apply_encoder(enc_params, windows):
        z_norm = jax.vmap(lambda w: encoder.apply(enc_params, w))(windows)
        return jax.vmap(denormalize_e_hat)(z_norm)

    @jax.jit
    def _build_obs_batch(base_obs, z_hat, depth_bins, k, obs_dists):
        return jax.vmap(build_full_obs)(base_obs, z_hat, depth_bins, k, obs_dists)

    # PPO update — single-step JIT functions called from Python loops.
    # Avoids lax.scan over hundreds of gradient steps which OOMs XLA compiler.
    @jax.jit
    def _compute_gae(transitions, last_value):
        from iron_man_drone.policy.ppo import compute_gae
        return compute_gae(transitions, last_value, gae_gamma, gae_lambda)

    @jax.jit
    def _actor_step(actor_state, mb_obs, mb_acts, mb_log_probs, mb_adv):
        def loss_fn(params):
            mean, log_std = actor.apply(params, mb_obs)
            dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
            new_lp  = dist.log_prob(mb_acts)
            entropy = dist.entropy()
            ratio   = jnp.exp(new_lp - mb_log_probs)
            adv_n   = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg1 = -ratio * adv_n
            pg2 = -jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_n
            pg_loss  = jnp.maximum(pg1, pg2).mean()
            ent_loss = -entropy_c * entropy.mean()
            return pg_loss + ent_loss, (pg_loss, entropy.mean())
        (loss, (pg, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
        return actor_state.apply_gradients(grads=grads), loss, ent

    @jax.jit
    def _critic_step(critic_state, mb_critic_obs, mb_ret):
        def loss_fn(params):
            values = critic.apply(params, mb_critic_obs)
            return jnp.mean((values - mb_ret) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(critic_state.params)
        return critic_state.apply_gradients(grads=grads), loss

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_path  = exp_dir / "train_log.csv"
    log_fields = [
        "epoch", "env_steps", "r_total", "r_track", "r_obs",
        "crash_rate", "entropy", "actor_loss", "critic_loss", "enc_loss",
        "density_mult",
    ]
    log_file_mode = "a" if log_path.exists() else "w"
    log_file = open(log_path, log_file_mode, newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if log_file_mode == "w":
        log_writer.writeheader()

    # WSL-native filesystem — avoids 30-40 min fsync stalls on /mnt/c writes.
    ckpt_dir = Path("/home/forke/m3_checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from latest checkpoint if available ────────────────────────────
    start_epoch = 0
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is None:
        # Also check legacy /mnt/c checkpoints from earlier runs
        legacy_dir = exp_dir / "checkpoints"
        latest_ckpt = find_latest_checkpoint(legacy_dir)
    if latest_ckpt is not None:
        print(f"  [RESUME] Found checkpoint: {latest_ckpt}")
        actor_state, critic_state, enc_state, ckpt_epoch = load_checkpoint(
            latest_ckpt, actor_state, critic_state, enc_state)
        start_epoch = ckpt_epoch + 1
        print(f"  [RESUME] Resuming from epoch {start_epoch} ({start_epoch * cfg.env_steps_per_epoch / 1e6:.1f}M env-steps)\n")
    else:
        print("  [RESUME] No checkpoint found — training from scratch.\n")

    # ── Initial reset ─────────────────────────────────────────────────────────
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, cfg.num_envs)
    states, base_obs, priv_state = env.batch_reset(reset_keys, density_mult=0.5)

    # Initial obs (encoder hasn't warmed up yet; use ground truth z for first obs)
    depth_imgs  = env.batch_render(states)
    depth_bins  = env.compute_depth_bins(depth_imgs)
    z_hat       = priv_state   # ground truth for episode 0 obs
    k_norm      = states.step.astype(jnp.float32) / EPISODE_STEPS
    obs_dists   = env.compute_k_nearest_batch(states)
    actor_obs, critic_obs = _build_obs_batch(base_obs, z_hat, depth_bins, k_norm, obs_dists)

    # ── Tracking state ────────────────────────────────────────────────────────
    t_start           = time.time()
    recent_crash_hist = []   # for H1 trend check
    one_hour_reported = False

    print(f"\nBeginning M3 training — run: {run_name}")
    print(f"Target: {cfg.total_env_steps:,} env-steps | {cfg.total_env_steps // cfg.env_steps_per_epoch:,} epochs")
    print(f"H1 abort check at epoch {cfg.h1_check_epoch} ({cfg.h1_check_epoch * cfg.env_steps_per_epoch / 1e6:.1f}M env-steps)\n")

    total_epochs = cfg.total_env_steps // cfg.env_steps_per_epoch

    for epoch in range(start_epoch, total_epochs):
        _epoch_t0 = time.time()
        env_steps = epoch * cfg.env_steps_per_epoch
        d_mult    = density_mult_at(env_steps, cfg)

        # ── Rollout collection ────────────────────────────────────────────────
        transitions_list = []
        encoder_windows_list  = []
        encoder_targets_list  = []

        crash_steps = 0
        total_steps = 0
        r_track_sum = 0.0
        r_obs_sum   = 0.0

        key, rollout_key = jax.random.split(key)

        for t in range(cfg.horizon):
            key, act_key, reset_key = jax.random.split(key, 3)

            # Collect encoder windows BEFORE stepping (current history state)
            enc_windows = _build_encoder_windows(states.obs_base_buf, states.action_buf)
            encoder_windows_list.append(enc_windows)
            encoder_targets_list.append(priv_state)

            # Sample actions
            actions, log_probs = _sample_actions(actor_state.params, actor_obs, act_key)

            # Values
            values = _get_values(critic_state.params, critic_obs)

            # Step environment
            new_states, new_base_obs, new_priv_state, rewards, dones = env.batch_step(states, actions)

            # Detect crash vs timeout for logging
            step_arr = np.array(new_states.step)
            done_arr = np.array(dones)
            crash_steps += int(np.sum(done_arr & (step_arr < EPISODE_STEPS - 5)))
            total_steps += cfg.num_envs

            # Render depth
            depth_imgs  = env.batch_render(new_states)
            depth_bins  = env.compute_depth_bins(depth_imgs)

            # Encoder → z_hat
            new_windows = _build_encoder_windows(new_states.obs_base_buf, new_states.action_buf)
            z_hat_new   = _apply_encoder(enc_state.params, new_windows)

            # Build next obs
            k_new       = new_states.step.astype(jnp.float32) / EPISODE_STEPS
            obs_dists   = env.compute_k_nearest_batch(new_states)
            new_actor_obs, new_critic_obs = _build_obs_batch(
                new_base_obs, z_hat_new, depth_bins, k_new, obs_dists
            )

            transitions_list.append(Transition(
                actor_obs=actor_obs,
                critic_obs=critic_obs,
                action=actions,
                log_prob=log_probs,
                reward=rewards,
                value=values,
                done=dones,
            ))

            # Auto-reset done envs with fresh obstacle configs.
            # Always use full N-env batch to keep shapes fixed — avoids XLA
            # recompilation that occurs when variable-size subsets are passed.
            if bool(jnp.any(dones)):
                stress = d_mult >= 1.0 and (epoch % 5 == 0)
                dm     = 1.5 if stress else d_mult
                reset_keys_full = jax.random.split(reset_key, cfg.num_envs)
                new_s, new_bo, new_ps = env.batch_reset(
                    reset_keys_full, density_mult=dm
                )

                # Apply resets via where — always (N, ...) shape, no recompile
                dones_mask = dones.astype(jnp.bool_)

                def _where_reset(r, c):
                    if r.ndim == 0:
                        return c
                    mask = dones_mask.reshape((-1,) + (1,) * (r.ndim - 1))
                    return jnp.where(mask, r, c)

                new_states     = jax.tree_util.tree_map(_where_reset, new_s, new_states)
                new_base_obs   = jnp.where(dones_mask[:, None], new_bo, new_base_obs)
                new_priv_state = jnp.where(dones_mask[:, None], new_ps, new_priv_state)

                # Rebuild obs for all envs (full-batch render — fixed shape)
                depth_imgs_r = env.batch_render(new_states)
                depth_bins_r = env.compute_depth_bins(depth_imgs_r)
                enc_win_r    = _build_encoder_windows(new_states.obs_base_buf, new_states.action_buf)
                z_hat_r      = _apply_encoder(enc_state.params, enc_win_r)
                k_r          = new_states.step.astype(jnp.float32) / EPISODE_STEPS
                od_r         = env.compute_k_nearest_batch(new_states)
                new_actor_obs, new_critic_obs = _build_obs_batch(
                    new_base_obs, z_hat_r, depth_bins_r, k_r, od_r
                )

            states      = new_states
            base_obs    = new_base_obs
            priv_state  = new_priv_state
            actor_obs   = new_actor_obs
            critic_obs  = new_critic_obs

        # ── Stack transitions ─────────────────────────────────────────────────
        transitions = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *transitions_list
        )  # (horizon, num_envs, ...)

        last_values = _get_values(critic_state.params, critic_obs)
        if epoch < 3:
            print(f"  [T{epoch}] rollout done  +{time.time()-_epoch_t0:.1f}s", flush=True)

        # ── PPO update — Python loops, single-step JITs, no lax.scan ─────────
        advantages, returns = _compute_gae(transitions, last_values)
        T, N = transitions.reward.shape
        flat = jax.tree_util.tree_map(lambda x: x.reshape(T * N, *x.shape[2:]), transitions)
        flat_adv = advantages.reshape(T * N)
        flat_ret = returns.reshape(T * N)
        mb_size  = T * N // cfg.num_minibatches
        ppo_metrics = {"actor_loss": 0.0, "entropy": 0.0, "critic_loss": 0.0}
        for _ in range(cfg.ppo_epochs):
            key, subkey = jax.random.split(key)
            perm = np.array(jax.random.permutation(subkey, T * N))
            for mb_i in range(cfg.num_minibatches):
                idx = perm[mb_i * mb_size : (mb_i + 1) * mb_size]
                mb_trans    = jax.tree_util.tree_map(lambda x: x[idx], flat)
                mb_adv      = flat_adv[idx]
                mb_ret      = flat_ret[idx]
                actor_state, a_loss, ent = _actor_step(
                    actor_state, mb_trans.actor_obs, mb_trans.action,
                    mb_trans.log_prob, mb_adv
                )
                for _ in range(cfg.critic_updates):
                    critic_state, c_loss = _critic_step(
                        critic_state, mb_trans.critic_obs, mb_ret
                    )
                ppo_metrics = {"actor_loss": float(a_loss), "entropy": float(ent),
                               "critic_loss": float(c_loss)}
        if epoch < 3:
            print(f"  [T{epoch}] PPO done     +{time.time()-_epoch_t0:.1f}s", flush=True)

        # ── Encoder update ────────────────────────────────────────────────────
        enc_windows_all = jnp.concatenate(encoder_windows_list, axis=0)  # (T*N, 2300)
        enc_targets_all = jnp.concatenate(encoder_targets_list, axis=0)  # (T*N, 8)
        enc_state, enc_loss = encoder_update(enc_state, enc_windows_all, enc_targets_all)
        if epoch < 3:
            print(f"  [T{epoch}] enc done     +{time.time()-_epoch_t0:.1f}s", flush=True)

        # ── Crash rate tracking ───────────────────────────────────────────────
        crash_rate = crash_steps / max(total_steps, 1)
        recent_crash_hist.append(crash_rate)
        if len(recent_crash_hist) > cfg.h1_trend_window:
            recent_crash_hist.pop(0)

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch % cfg.log_every_epochs == 0:
            r_total_mean = float(jnp.mean(transitions.reward))
            entropy      = float(ppo_metrics.get("entropy", 0.0))
            actor_loss   = float(ppo_metrics.get("actor_loss", 0.0))
            critic_loss  = float(ppo_metrics.get("critic_loss", 0.0))

            elapsed = time.time() - t_start
            fps     = env_steps / max(elapsed, 1)
            print(
                f"[{epoch:6d} | {env_steps/1e6:5.1f}M steps | {elapsed/3600:.2f}h | {fps:.0f} fps] "
                f"r={r_total_mean:.3f}  crash={crash_rate:.2%}  enc_loss={float(enc_loss):.4f}  "
                f"d_mult={d_mult:.2f}"
            )
            log_writer.writerow({
                "epoch": epoch, "env_steps": env_steps,
                "r_total": r_total_mean, "r_track": 0, "r_obs": 0,
                "crash_rate": crash_rate, "entropy": entropy,
                "actor_loss": actor_loss, "critic_loss": critic_loss,
                "enc_loss": float(enc_loss), "density_mult": d_mult,
            })
            log_file.flush()

            # NaN guard — abort early rather than run 500M steps on dead weights
            if epoch > 0 and (jnp.isnan(r_total_mean) or jnp.isnan(float(enc_loss))):
                print(
                    f"\n[ABORT NaN] epoch {epoch}: r={r_total_mean:.4f}  enc_loss={float(enc_loss):.4f}\n"
                    "Weights have gone NaN. Common causes: critic_updates too high, LR too large.\n"
                    "Stopping to avoid wasting compute."
                )
                save_checkpoint(ckpt_dir / f"abort_nan_e{epoch:06d}", actor_state, critic_state, enc_state, epoch)
                sys.exit(3)

        # ── H1 abort gate ─────────────────────────────────────────────────────
        if epoch == cfg.h1_check_epoch:
            trend_down = (
                len(recent_crash_hist) >= 2 and
                recent_crash_hist[-1] < recent_crash_hist[len(recent_crash_hist) // 2]
            )
            if crash_rate > cfg.h1_crash_threshold and not trend_down:
                print(
                    f"\n[ABORT H1] Epoch {epoch}: crash_rate={crash_rate:.2%} > {cfg.h1_crash_threshold:.0%} "
                    f"and not trending down.\n"
                    "Candidates: (a) reward shape sparse, (b) obstacle density too high, "
                    "(c) depth bin pipeline broken.\n"
                    "See hypothesis doc FM1/FM2. Stopping training."
                )
                save_checkpoint(exp_dir / "checkpoints" / "abort_h1", actor_state, critic_state, enc_state, epoch)
                sys.exit(2)
            else:
                print(f"[H1 PASS] epoch {epoch}: crash_rate={crash_rate:.2%}, trending={'down' if trend_down else 'flat-ok'}")

        # ── 1-hour checkpoint ─────────────────────────────────────────────────
        if epoch == cfg.one_hour_epoch and not one_hour_reported:
            one_hour_reported = True
            eval_results = {"crash_rate": crash_rate, "med_nominal": -1, "med_fault70": -1}
            report_path  = exp_dir / "checkpoint_1h_report.md"
            write_1h_report(report_path, epoch, env_steps, eval_results, crash_rate, enc_loss, cfg)
            print(f"\n{'='*70}")
            print(f"1-HOUR CHECKPOINT — epoch {epoch} ({env_steps/1e6:.0f}M env-steps)")
            print(f"crash_rate={crash_rate:.2%}  enc_loss={float(enc_loss):.4f}")
            print(f"Report: {report_path}")
            print("Press Enter to continue overnight, or Ctrl+C to abort.")
            print('='*70)
            try:
                input()
            except KeyboardInterrupt:
                print("Training aborted by user at 1-hour checkpoint.")
                save_checkpoint(ckpt_dir / f"epoch_{epoch:06d}", actor_state, critic_state, enc_state, epoch)
                sys.exit(0)
            except EOFError:
                print("Stdin closed (background run) — continuing automatically.")

        # ── Periodic eval ─────────────────────────────────────────────────────
        eval_every = (
            cfg.eval_every_epochs_early if epoch < cfg.eval_switch_epoch
            else cfg.eval_every_epochs_late
        )
        if epoch > 0 and epoch % eval_every == 0:
            key, eval_key = jax.random.split(key)
            eval_results = run_quick_eval(env, actor_state, enc_state, cfg, eval_key, n_episodes=16)
            print(
                f"  [EVAL] nominal crash={eval_results.get('crash_rate_nominal', 0):.2%}  "
                f"MED={eval_results.get('med_nominal', 0):.4f}  "
                f"fault70 crash={eval_results.get('crash_rate_fault70', 0):.2%}  "
                f"MED={eval_results.get('med_fault70', 0):.4f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch > 0 and epoch % cfg.ckpt_every_epochs == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:06d}", actor_state, critic_state, enc_state, epoch)

        # ── SIGTERM graceful exit ─────────────────────────────────────────────
        if _shutdown_requested:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:06d}", actor_state, critic_state, enc_state, epoch)
            print(f"[SIGTERM] Checkpoint saved at epoch {epoch}. Exiting cleanly.")
            log_file.close()
            sys.exit(0)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(ckpt_dir / "final", actor_state, critic_state, enc_state, total_epochs)
    log_file.close()
    print(f"\nTraining complete. Checkpoints in {ckpt_dir}")


# ── Utilities ──────────────────────────────────────────────────────────────────

def scatter_reset(states: M3EnvState, new_states: M3EnvState, idxs: np.ndarray) -> M3EnvState:
    """Scatter new_states into states at positions idxs (for auto-reset)."""
    def _scatter(full, new):
        if full.ndim == 0:
            return full
        return full.at[idxs].set(new)
    return jax.tree_util.tree_map(_scatter, states, new_states)


def save_checkpoint(path: Path, actor_state, critic_state, enc_state, epoch: int):
    import orbax.checkpoint as ocp
    import shutil
    path.mkdir(parents=True, exist_ok=True)
    ckptr = ocp.PyTreeCheckpointer()
    for sub in ("actor", "critic", "encoder"):
        sub_path = path / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)
        ckptr.save(str(sub_path), {"actor": actor_state, "critic": critic_state, "encoder": enc_state}[sub])
    (path / "epoch.txt").write_text(str(epoch))
    print(f"  [CKPT] Saved to {path}")


def find_latest_checkpoint(ckpt_dir: Path):
    """Return path to the highest epoch_XXXXXX checkpoint that has epoch.txt, or None."""
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("epoch_*"))
    for c in reversed(candidates):
        if (c / "epoch.txt").exists() and (c / "actor").exists():
            return c
    return None


def load_checkpoint(path: Path, actor_state, critic_state, enc_state):
    """Restore network states from checkpoint. Returns (actor_state, critic_state, enc_state, epoch)."""
    import orbax.checkpoint as ocp
    ckptr = ocp.PyTreeCheckpointer()
    actor_state  = ckptr.restore(str(path / "actor"),   item=actor_state)
    critic_state = ckptr.restore(str(path / "critic"),  item=critic_state)
    enc_state    = ckptr.restore(str(path / "encoder"), item=enc_state)
    epoch = int((path / "epoch.txt").read_text().strip())
    return actor_state, critic_state, enc_state, epoch


def write_1h_report(path: Path, epoch: int, env_steps: int, eval_results: dict,
                     crash_rate: float, enc_loss, cfg: M3Config):
    lines = [
        f"# M3 1-Hour Checkpoint Report",
        f"",
        f"**Epoch**: {epoch} | **Env-steps**: {env_steps:,} ({env_steps/1e6:.1f}M)",
        f"**Wall time**: ~1 hour",
        f"",
        f"## Current metrics",
        f"",
        f"| Metric | Value | H1 prediction |",
        f"|---|---|---|",
        f"| Training crash rate | {crash_rate:.1%} | < 60% by 20M env-steps |",
        f"| Encoder loss | {float(enc_loss):.5f} | — |",
        f"",
        f"## Quick eval (no obstacles, figure-eight-normal)",
        f"",
        f"| Condition | Crash rate | MED |",
        f"|---|---|---|",
    ]
    for k, v in eval_results.items():
        if k.startswith("crash_rate"):
            label = k.replace("crash_rate_", "")
            med   = eval_results.get(f"med_{label}", float("nan"))
            lines.append(f"| {label} | {v:.1%} | {med:.4f} m |")
    lines += [
        f"",
        f"## Decision",
        f"",
        f"- [ ] Continue overnight (press Enter in terminal)",
        f"- [ ] Abort (Ctrl+C in terminal)",
        f"",
        f"Review H1–H7 in notes/M3_hypothesis.md against these numbers before deciding.",
    ]
    path.write_text("\n".join(lines))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=os.environ.get("M3_RUN_NAME", ""))
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    args = parser.parse_args()

    import time as _time
    ts = int(_time.time())
    run_name = args.run_name or f"m3_run1_joint_4mode_3ms_{ts}"

    cfg = M3Config()
    if args.num_envs:
        cfg.num_envs = args.num_envs
    if args.total_steps:
        cfg.total_env_steps = args.total_steps

    exp_dir = ROOT / "experiments" / run_name
    print(f"Run: {run_name}")
    print(f"Experiment dir: {exp_dir}")

    train(cfg, run_name, exp_dir)
