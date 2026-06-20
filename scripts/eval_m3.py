"""
M3 evaluation — collision-free / success / tracking MED, stratified by episode difficulty.

This is the eval M3 never had. Beyond raw rates, it answers the fairness question:
a collision-free episode is meaningless if no obstacle was ever near the path, and
a crash is unfair if the *reference trajectory itself* passes through an obstacle
(perfect tracking would still crash). The scene generator places obstacles
independent of the trajectory, so both cases occur.

For every episode we compute IDEAL-PATH CLEARANCE: the minimum L-inf surface
distance from the reference trajectory (over the whole horizon) to any obstacle.
Episodes are bucketed:
  unfair    c < D_CRASH(0.15)   ideal line enters crash zone -> guaranteed crash
  tight     0.15 <= c < 0.35    real avoidance required (tracking error ~0.15 m)
  moderate  0.35 <= c < 0.70    some proximity
  clear     c >= 0.70           effectively a freebie
The honest avoidance metric is the crash/CF rate on FAIR episodes (c >= D_CRASH),
especially the `tight` bucket.

Per condition we also break down by scene mode.

Run inside WSL with jax_env:
  /home/forke/jax_env/bin/python scripts/eval_m3.py \
      --ckpt /home/forke/m3_checkpoints/m3_run1/epoch_005899 --n 96
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iron_man_drone.envs.quadrotor_env_m3 import (
    M3VecEnv, ACTOR_OBS_DIM, CRITIC_OBS_DIM,
    build_full_obs, D_CRASH, D_SAFE, EPISODE_STEPS, N_OBSTACLE_SLOTS,
)
from iron_man_drone.envs.trajectories import get_reference_pos
from iron_man_drone.policy.networks import Actor, Critic
from iron_man_drone.policy.encoder import (
    AdaptationEncoder, build_history_window, denormalize_e_hat, WINDOW_DIM,
)
from iron_man_drone.utils.scene_generator import TRAINING_MODES, TRAINING_WEIGHTS

sys.path.insert(0, str(ROOT / "scripts"))
from train_m3 import load_checkpoint, find_latest_checkpoint

SUCCESS_MED = 0.10   # m — Tier 1 success threshold


def build_states():
    key = jax.random.PRNGKey(0)
    k_a, k_c, k_e = jax.random.split(key, 3)
    actor, critic, encoder = Actor(), Critic(), AdaptationEncoder()
    actor_p  = actor.init(k_a,  jnp.zeros((1, ACTOR_OBS_DIM)))
    critic_p = critic.init(k_c, jnp.zeros((1, CRITIC_OBS_DIM)))
    enc_p    = encoder.init(k_e, jnp.zeros((1, WINDOW_DIM)))
    mk = lambda p, lr: TrainState.create(
        apply_fn=None, params=p,
        tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr)))
    return (actor, critic, encoder,
            mk(actor_p, 3e-4), mk(critic_p, 1e-4), mk(enc_p, 1e-4))


def make_eval_fns(actor, encoder):
    @jax.jit
    def encode(enc_params, obs_base_buf, action_buf):
        windows = jax.vmap(build_history_window)(obs_base_buf, action_buf)
        z_norm  = jax.vmap(lambda w: encoder.apply(enc_params, w))(windows)
        return jax.vmap(denormalize_e_hat)(z_norm)

    @jax.jit
    def assemble(base_obs, z_hat, depth_bins, k, obs_dists):
        actor_obs, _ = jax.vmap(build_full_obs)(base_obs, z_hat, depth_bins, k, obs_dists)
        return actor_obs

    @jax.jit
    def greedy_action(actor_params, actor_obs):
        mean, _ = actor.apply(actor_params, actor_obs)
        return mean

    @jax.jit
    def ref_pos_batch(traj, step):
        return jax.vmap(get_reference_pos)(traj, step)

    @jax.jit
    def ref_clearance(traj, centers, he, n_obs):
        """Per-env min L-inf surface distance from the reference path to any obstacle."""
        def single(tr, c, h, n):
            steps = jnp.arange(EPISODE_STEPS)
            refs  = jax.vmap(lambda s: get_reference_pos(tr, s))(steps)   # (T,3)
            diff  = jnp.abs(refs[:, None, :] - c[None, :, :]) - h[None, :, :]  # (T,N,3)
            d     = jnp.max(jnp.maximum(diff, 0.0), axis=2)               # (T,N)
            active = jnp.arange(N_OBSTACLE_SLOTS) < n
            d = jnp.where(active[None, :], d, jnp.inf)
            return jnp.min(d)
        return jax.vmap(single)(traj, centers, he, n_obs)

    return encode, assemble, greedy_action, ref_pos_batch, ref_clearance


def eval_condition(env, actor_state, enc_state, fns, key, n,
                   modes=None, density_mult=1.0, eta=1.0, no_obstacles=False):
    encode, assemble, greedy_action, ref_pos_batch, ref_clearance = fns
    drone_id = env.drone_body_id
    keys = jax.random.split(key, n)

    if no_obstacles:
        centers = np.full((n, N_OBSTACLE_SLOTS, 3), 100.0, dtype=np.float32)
        he      = np.zeros((n, N_OBSTACLE_SLOTS, 3), dtype=np.float32)
        n_obs   = np.zeros((n,), dtype=np.int32)
        states, base_obs, priv = env._reset_jit(
            keys, jnp.array(centers), jnp.array(he), jnp.array(n_obs))
        mode_arr = np.array(["clear"] * n)
    else:
        if modes is not None and len(modes) > 1:
            rng = np.random.default_rng(0)
            p = (TRAINING_WEIGHTS / TRAINING_WEIGHTS.sum()
                 if len(modes) == len(TRAINING_WEIGHTS) else None)
            mode_arr = rng.choice(modes, size=n, p=p)
        elif modes is not None:
            mode_arr = np.array([modes[0]] * n)
        else:
            mode_arr = np.array(["mixed"] * n)
        states, base_obs, priv = env.batch_reset(
            keys, modes=list(mode_arr), density_mult=density_mult)

    if abs(eta - 1.0) > 1e-3:
        re = np.array(states.rotor_efficiency); re[:, 0] = eta
        ps = np.array(states.priv_state);       ps[:, 0] = eta
        states = states._replace(rotor_efficiency=jnp.array(re), priv_state=jnp.array(ps))

    clearance = np.array(ref_clearance(
        states.traj, states.obstacle_positions,
        states.obstacle_half_extents, states.n_obstacles))   # (n,) inf if no obstacles

    depth_bins = env.compute_depth_bins(env.batch_render(states))
    z_hat      = encode(enc_state.params, states.obs_base_buf, states.action_buf)
    k_norm     = states.step.astype(jnp.float32) / EPISODE_STEPS
    obs_dists  = env.compute_k_nearest_batch(states)
    actor_obs  = assemble(base_obs, z_hat, depth_bins, k_norm, obs_dists)

    ever_done    = np.zeros(n, dtype=bool)
    ever_crashed = np.zeros(n, dtype=bool)
    ever_oob     = np.zeros(n, dtype=bool)
    err_sum      = np.zeros(n, dtype=np.float64)
    err_cnt      = np.zeros(n, dtype=np.int64)

    for _ in range(EPISODE_STEPS):
        action = greedy_action(actor_state.params, actor_obs)
        new_states, new_base, new_priv, reward, done = env.batch_step(states, action)

        od   = env.compute_k_nearest_batch(new_states)
        dmin = np.array(jnp.min(od, axis=1))
        crashed_now = dmin < D_CRASH

        pos = np.array(new_states.mjx_data.xpos[:, drone_id])
        ref = np.array(ref_pos_batch(new_states.traj, new_states.step))
        err = np.linalg.norm(pos - ref, axis=1)
        step_arr = np.array(new_states.step)

        active  = ~ever_done
        done_np = np.array(done)
        newly   = done_np & active
        early   = step_arr < (EPISODE_STEPS - 5)
        ever_crashed |= newly & crashed_now
        ever_oob     |= newly & ~crashed_now & early
        err_sum[active] += err[active]
        err_cnt[active] += 1
        ever_done |= done_np

        depth_bins = env.compute_depth_bins(env.batch_render(new_states))
        z_hat      = encode(enc_state.params, new_states.obs_base_buf, new_states.action_buf)
        k_norm     = new_states.step.astype(jnp.float32) / EPISODE_STEPS
        actor_obs  = assemble(new_base, z_hat, depth_bins, k_norm, od)
        states = new_states
        if ever_done.all():
            break

    collision_free = ~ever_crashed & ~ever_oob
    med = err_sum / np.maximum(err_cnt, 1)
    return dict(
        clearance=clearance, collision_free=collision_free,
        crashed=ever_crashed, oob=ever_oob, med=med, mode=mode_arr,
    )


# Difficulty buckets by ideal-path clearance (m)
BUCKETS = [
    ("unfair",   -np.inf, D_CRASH),    # < 0.15  line through crash zone
    ("tight",    D_CRASH, 0.35),       # real avoidance required
    ("moderate", 0.35,    D_SAFE),     # 0.35-0.5
    ("clear",    D_SAFE,  np.inf),     # >= 0.5  freebie
]


def report(name, r):
    cf, cr, oob, med, cl = (r["collision_free"], r["crashed"], r["oob"],
                            r["med"], r["clearance"])
    n = len(cf)
    fair = cl >= D_CRASH
    cf_fair = cf[fair]
    cf_meds = med[cf]
    print(f"\n[{name}]  n={n}")
    print(f"  overall : CF={cf.mean():.1%}  crash={cr.mean():.1%}  oob={oob.mean():.1%}  "
          f"MED(cf)={cf_meds.mean() if cf_meds.size else float('nan'):.3f}m")
    if np.isfinite(cl).any():
        print(f"  FAIR-only (clearance>={D_CRASH}m, n={fair.sum()}): "
              f"CF={cf_fair.mean() if cf_fair.size else float('nan'):.1%}   "
              f"[{(~fair).sum()} unfair episodes excluded]")
        print("  by ideal-path clearance:")
        for label, lo, hi in BUCKETS:
            m = (cl >= lo) & (cl < hi)
            if m.sum() == 0:
                continue
            print(f"    {label:9s} n={int(m.sum()):3d}  "
                  f"crash={cr[m].mean():.1%}  CF={cf[m].mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--only", default=None,
                    help="comma list of condition names to run (e.g. random,fault70,clear)")
    ap.add_argument("--out", default=None, help="optional JSON dump of per-episode arrays")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(
        Path("/home/forke/m3_checkpoints/m3_run1"))
    if ckpt is None:
        print("No checkpoint found."); sys.exit(1)
    print(f"Checkpoint: {ckpt}")

    actor, critic, encoder, actor_state, critic_state, enc_state = build_states()
    actor_state, critic_state, enc_state, epoch = load_checkpoint(
        ckpt, actor_state, critic_state, enc_state)
    print(f"Loaded epoch {epoch} ({epoch * 32768 / 1e6:.0f}M env-steps)")

    class _Cfg:
        num_envs = args.n
    print("Initializing M3VecEnv (MJWarp init ~30s)...")
    env = M3VecEnv(_Cfg(), fault_prob=0.0)
    fns = make_eval_fns(actor, encoder)
    key = jax.random.PRNGKey(12345)
    print(f"Eval: {args.n} episodes/condition, greedy, {EPISODE_STEPS} steps")

    conditions = (
        [(m, dict(modes=[m], density_mult=1.0, eta=1.0)) for m in TRAINING_MODES]
        + [("fault70", dict(modes=TRAINING_MODES, density_mult=1.0, eta=0.70)),
           ("clear",   dict(no_obstacles=True, eta=1.0))]
    )
    if args.only:
        want = set(args.only.split(","))
        conditions = [(n, kw) for n, kw in conditions if n in want]

    print("\n" + "=" * 64)
    print("M3 EVAL — stratified by ideal-path difficulty")
    print("=" * 64)
    dump = {}
    for name, kw in conditions:
        key, sub = jax.random.split(key)
        r = eval_condition(env, actor_state, enc_state, fns, sub, args.n, **kw)
        report(name, r)
        if args.out:
            dump[name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in r.items()}

    print("\n" + "-" * 64)
    print("Gates: CF(fair)>=80% nominal, MED(cf)<=0.10m, CF(fair)>=50% fault70.")
    print("'unfair' episodes have the ideal path inside the crash zone — not the policy's fault.")
    if args.out:
        Path(args.out).write_text(json.dumps(dump))
        print(f"Per-episode arrays -> {args.out}")


if __name__ == "__main__":
    main()
