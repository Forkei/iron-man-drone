"""
Phase 2 data collection.

Rolls out the frozen Phase 1 actor under full DR. Records (obs_base, action, priv_state)
at each of 1000 steps per episode across 20,000 episodes. Data is saved in chunks of
N_ENVS episodes each.

Output dir: experiments/phase2_data/  (one NPZ per chunk)
  chunk_{i:04d}.npz:
    obs_base   : (N_ENVS, 1000, 42)  float32 — observable obs without priv_state
    actions    : (N_ENVS, 1000, 4)   float32 — CTBR actions taken by Phase 1 actor
    priv_states: (N_ENVS, 8)         float32 — raw e_t, constant per episode

DR matching Phase 1: fault_prob=0.70, η∈[0.5,1.0), mass∈[0.8,1.2], kf∈[0.7,1.3], wind=0.

Usage:
  python scripts/collect_phase2_data.py
  python scripts/collect_phase2_data.py --n_episodes 5000 --num_envs 512
  python scripts/collect_phase2_data.py --checkpoint experiments/.../checkpoints/final
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "experiments/m2_phase1_baseline"
    / "m2_phase1_baseline_1778244202/checkpoints/final"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/phase2_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out_dir",    default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n_episodes", type=int, default=20000)
    parser.add_argument("--num_envs",   type=int, default=2048)
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices : {jax.devices()}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Output dir  : {out_dir}")
    print(f"Target episodes: {args.n_episodes}")
    print(f"Envs per batch : {args.num_envs}")
    print()

    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, EPISODE_STEPS,
    )
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    # ── Environment — DR matching Phase 1 training ───────────────────────────
    class _Cfg:
        num_envs = args.num_envs

    env = VecEnv(
        _Cfg(),
        fault_prob=0.70,
        eta_min=0.50,
        mass_lo=0.80,
        mass_hi=1.20,
    )
    print(f"Env: {args.num_envs} parallel envs. DR: fault_prob=0.70, η_min=0.50, mass=[0.8,1.2]")

    # ── Phase 1 actor (frozen) ───────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        actor_obs_dim=50, critic_obs_dim=51,
        action_dim=4, hidden_dim=256, num_layers=3,
    )
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)

    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    restored     = checkpointer.restore(str(ckpt_path))
    actor_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored["actor"]["params"]
    )
    actor_state = actor_state.replace(params=actor_params)
    print("Phase 1 actor loaded (frozen).")

    @jax.jit
    def actor_apply(params, obs):
        return actor_state.apply_fn(params, obs)

    # Warmup
    _ = actor_apply(actor_params, jnp.zeros((args.num_envs, 50)))
    print("Actor JIT warmed up.")

    # ── Batch collection (lax.scan over EPISODE_STEPS) ───────────────────────
    @jax.jit
    def collect_batch(reset_keys: jnp.ndarray):
        """
        Collect one batch of N_ENVS episodes.

        Returns:
          obs_base   : (EPISODE_STEPS, N_ENVS, 42) float32
          actions    : (EPISODE_STEPS, N_ENVS, 4)  float32
          priv_states: (N_ENVS, 8)                  float32
        """
        states, a_obs, _ = env.batch_reset(reset_keys)
        priv_states = states.priv_state   # (N_ENVS, 8), constant per episode

        def step(carry, _):
            states, a_obs = carry
            mean, _ = actor_apply(actor_params, a_obs)   # (N_ENVS, 4)
            obs_base = a_obs[:, :42]                     # (N_ENVS, 42) — strip priv_state
            new_states, new_a_obs, _, _, _ = env.batch_step(states, mean)
            return (new_states, new_a_obs), (obs_base, mean)

        _, (obs_bases, actions) = jax.lax.scan(
            step, (states, a_obs), None, length=EPISODE_STEPS
        )
        # obs_bases : (EPISODE_STEPS, N_ENVS, 42)
        # actions   : (EPISODE_STEPS, N_ENVS, 4)
        return obs_bases, actions, priv_states

    # ── Warmup scan ──────────────────────────────────────────────────────────
    print("\nWarming up lax.scan (~2 min for first JIT)...")
    t_warmup = time.time()
    _keys = jax.random.split(jax.random.PRNGKey(999), args.num_envs)
    _obs, _act, _priv = collect_batch(_keys)
    _ = jax.block_until_ready((_obs, _act, _priv))
    print(f"Scan JIT warmed up in {time.time() - t_warmup:.1f} s.")

    # ── Collection loop ───────────────────────────────────────────────────────
    key = jax.random.PRNGKey(args.seed)
    n_batches  = int(np.ceil(args.n_episodes / args.num_envs))
    total_eps  = n_batches * args.num_envs
    n_fault    = 0
    n_episodes = 0
    t_start    = time.time()

    print(f"\nCollecting {n_batches} batches ({total_eps} episodes total) ...")
    print()

    for batch_i in range(n_batches):
        key, batch_key = jax.random.split(key)
        reset_keys = jax.random.split(batch_key, args.num_envs)

        t_batch = time.time()
        obs_b, act_b, priv_b = collect_batch(reset_keys)
        _ = jax.block_until_ready((obs_b, act_b, priv_b))
        dt = time.time() - t_batch

        # Convert to numpy and transpose: (T, N, dim) → (N, T, dim)
        obs_np  = np.array(obs_b).transpose(1, 0, 2)   # (N_ENVS, 1000, 42)
        act_np  = np.array(act_b).transpose(1, 0, 2)   # (N_ENVS, 1000, 4)
        priv_np = np.array(priv_b)                     # (N_ENVS, 8)

        # Fault episode stats: episode is fault if any η < 0.99
        is_fault = (priv_np[:, :4].min(axis=1) < 0.99)
        n_fault    += int(is_fault.sum())
        n_episodes += args.num_envs

        chunk_path = out_dir / f"chunk_{batch_i:04d}.npz"
        np.savez_compressed(chunk_path,
                            obs_base=obs_np.astype(np.float32),
                            actions=act_np.astype(np.float32),
                            priv_states=priv_np.astype(np.float32))

        fps = args.num_envs * EPISODE_STEPS / dt
        print(f"  Batch {batch_i+1:2d}/{n_batches}: {args.num_envs} episodes, "
              f"fault={is_fault.sum()}/{args.num_envs} ({100*is_fault.mean():.0f}%), "
              f"{fps/1e3:.0f}k fps, {dt:.1f}s → {chunk_path.name}")

    total_time = time.time() - t_start

    print()
    print("=" * 62)
    print(f"  Data collection complete")
    print(f"  Total episodes     : {n_episodes}")
    print(f"  Fault episodes     : {n_fault} ({100*n_fault/n_episodes:.1f}%)")
    print(f"  Nominal episodes   : {n_episodes-n_fault} ({100*(n_episodes-n_fault)/n_episodes:.1f}%)")
    print(f"  Output dir         : {out_dir}")
    print(f"  Chunks saved       : {n_batches}")
    print(f"  Pairs per episode  : {EPISODE_STEPS} steps (window valid at t≥49)")
    print(f"  Total pairs (t≥49) : ~{n_episodes * 951 // 1_000_000}M")
    print(f"  Wall-clock time    : {total_time/60:.1f} min")

    # Disk usage
    chunk_size_mb = sum(p.stat().st_size for p in out_dir.glob("chunk_*.npz")) / 1e6
    print(f"  Disk usage         : {chunk_size_mb:.0f} MB ({chunk_size_mb/1000:.2f} GB)")
    print("=" * 62)

    # Sanity: sample priv_state stats
    _eta = priv_np[:, :4]
    _mass = priv_np[:, 4]
    _wind = priv_np[:, 5:8]
    print(f"\n  Last batch priv_state stats (sanity):")
    print(f"    η mean (fault only): {_eta[is_fault].min(axis=1).mean():.3f} "
          f"(expected ≈ 0.75 = midpoint of [0.5, 1.0))")
    print(f"    mass mean:           {_mass.mean():.3f} ± {_mass.std():.3f} "
          f"(expected ≈ 1.00 ± 0.12)")
    print(f"    wind (all):          {np.abs(_wind).max():.6f} (expected 0.0)")
    print()
    print("Step 1 complete. Run scripts/train_phase2_encoder.py next.")


if __name__ == "__main__":
    main()
