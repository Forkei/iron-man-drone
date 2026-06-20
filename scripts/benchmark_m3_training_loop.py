"""
Benchmark the M3 training loop components to estimate actual training fps.
Runs N_EPOCHS × HORIZON env steps including batch_step, batch_render,
encoder inference, and obs building — the same operations as train_m3.py.
Does NOT do PPO updates (those are < 5% of wall time).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from iron_man_drone.envs.quadrotor_env_m3 import (
    M3VecEnv, ACTOR_OBS_DIM, CRITIC_OBS_DIM, ENCODER_DIM,
    OBS_BASE_DIM, K_NEAREST, build_full_obs, EPISODE_STEPS,
)
from iron_man_drone.policy.encoder import (
    AdaptationEncoder, build_history_window,
    H as ENC_H, OBS_DIM as ENC_OBS, ACTION_DIM, WINDOW_DIM,
)
from iron_man_drone.policy.networks import Actor, Critic

N_ENVS   = 1024
HORIZON  = 32
N_EPOCHS = 20    # 20 × 32 × 1024 = 655k steps — enough for stable measurement

print(f"M3 training loop benchmark: N={N_ENVS}, horizon={HORIZON}, epochs={N_EPOCHS}")
print(f"Total env-steps: {N_EPOCHS * HORIZON * N_ENVS:,}\n")

key = jax.random.PRNGKey(0)

# ── Build env ─────────────────────────────────────────────────────────────────
env = M3VecEnv(SimpleNamespace(num_envs=N_ENVS), fault_prob=0.7,
               mass_lo=0.8, mass_hi=1.2)
reset_keys = jax.random.split(key, N_ENVS)
states, base_obs, priv_state = env.batch_reset(reset_keys, density_mult=0.5)
print(f"Env ready — drone_body_id={env.drone_body_id}")

# ── Dummy policy (random weights) ────────────────────────────────────────────
actor  = Actor(hidden_dim=256, num_layers=3, action_dim=4)
critic = Critic(hidden_dim=256, num_layers=3)
encoder = AdaptationEncoder()

key, ak, ck, ek = jax.random.split(key, 4)
dummy_actor_obs  = jnp.zeros((ACTOR_OBS_DIM,))
dummy_critic_obs = jnp.zeros((CRITIC_OBS_DIM,))
dummy_window     = jnp.zeros((WINDOW_DIM,))
ap = actor.init(ak, dummy_actor_obs)
cp = critic.init(ck, dummy_critic_obs)
ep = encoder.init(ek, dummy_window)

@jax.jit
def _sample_actions(params, obs, key):
    mean, log_std = jax.vmap(lambda o: actor.apply(params, o))(obs)
    dist = jax.vmap(lambda m, ls: jax.random.normal(key, m.shape) * jnp.exp(ls) + m)(mean, log_std)
    return dist

@jax.jit
def _apply_encoder(params, windows):
    return jax.vmap(lambda w: encoder.apply(params, w))(windows)

def _build_encoder_windows(obs_buf, act_buf):
    return jax.vmap(lambda ob, ab: build_history_window(ob, ab))(obs_buf, act_buf)

def _get_full_obs(base_obs, z_hat, depth_bins, step, obs_dists):
    k = step.astype(jnp.float32) / EPISODE_STEPS
    return jax.vmap(lambda b, z, d, ki, od: build_full_obs(b, z, d, ki, od))(
        base_obs, z_hat, depth_bins, k, obs_dists
    )

# JIT warmup
print("JIT warmup (compiling kernels)...")
t_warmup = time.time()
enc_windows = _build_encoder_windows(states.obs_base_buf, states.action_buf)
z_hat = _apply_encoder(ep, enc_windows)
obs_dists = env.compute_k_nearest_batch(states)
depth_imgs = env.batch_render(states)
depth_bins = env.compute_depth_bins(depth_imgs)
actor_obs, critic_obs = _get_full_obs(base_obs, z_hat, depth_bins, states.step, obs_dists)
actions = _sample_actions(ap, actor_obs, key)
states2, base_obs2, priv2, _, _ = env.batch_step(states, actions)
jax.block_until_ready(states2.mjx_data.qpos)
print(f"  Warmup: {time.time()-t_warmup:.1f}s\n")

# ── Benchmark loop ────────────────────────────────────────────────────────────
print("Running benchmark...")
t_start = time.time()

for epoch in range(N_EPOCHS):
    for t in range(HORIZON):
        key, act_key = jax.random.split(key)

        enc_windows = _build_encoder_windows(states.obs_base_buf, states.action_buf)
        z_hat       = _apply_encoder(ep, enc_windows)
        obs_dists   = env.compute_k_nearest_batch(states)
        depth_imgs  = env.batch_render(states)
        depth_bins  = env.compute_depth_bins(depth_imgs)
        actor_obs, critic_obs = _get_full_obs(
            base_obs, z_hat, depth_bins, states.step, obs_dists
        )
        actions = _sample_actions(ap, actor_obs, act_key)
        states, base_obs, priv_state, _, _ = env.batch_step(states, actions)

    # Sync GPU after each epoch for accurate timing
    jax.block_until_ready(states.mjx_data.qpos)
    elapsed = time.time() - t_start
    done_steps = (epoch + 1) * HORIZON * N_ENVS
    fps = done_steps / elapsed
    print(f"  epoch {epoch+1:3d}/{N_EPOCHS}  {done_steps/1e6:.2f}M steps  {fps:,.0f} fps")

elapsed = time.time() - t_start
total_steps = N_EPOCHS * HORIZON * N_ENVS
fps = total_steps / elapsed

print(f"\n─────────────────────────────────────────────")
print(f"Training loop fps (step+render+encoder+obs): {fps:,.0f}")
print(f"")
TARGET = 500_000_000
epoch_size = HORIZON * N_ENVS
total_epochs = TARGET // epoch_size
est_hours = TARGET / fps / 3600
print(f"500M env-step training estimate:")
print(f"  {total_epochs:,} epochs × {epoch_size:,} env-steps/epoch")
print(f"  At {fps:,.0f} fps:  {est_hours:.1f} hours  ({est_hours*60:.0f} min)")
print(f"  H1 gate at 30M steps:  {30e6/fps/3600:.2f}h  ({30e6/fps/60:.0f} min)")
print(f"  1-hour pause at ~117M: {117e6/fps/3600:.2f}h  ({117e6/fps/60:.0f} min)")
