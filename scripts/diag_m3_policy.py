"""
M3 policy behavior diagnostics — WHY does the final policy track loosely (~0.17m)?

Separates the candidate causes instead of assuming reward saturation:

  Q1 TRAJECTORY: is 0.17m mostly because M3 evals on hard poly/zigzag @3m/s?
     -> run the policy on figure-eight (M2's eval trajectory + T/4 methodology).
        If MED ~0.06 there, the gap is trajectory difficulty, not the policy.
  Q2 ENCODER: is the fault-encoder z adding noise?
     -> compare encoder-z vs oracle-z (true priv_state) on the same rollouts.
        If oracle tracks much tighter, fix the encoder, not the tracking reward.
  Q3 SIGNATURE: oscillation / lag / offset?
     -> per-step error over the episode, action chatter (mean |du|), and
        error-vs-reference-speed. Offset+flat => reward saturation;
        error rising with speed => control authority; high chatter => smoothness.

No obstacles in any test (depth bins = 1.0), so this isolates the tracking path.

Run: /home/forke/jax_env/bin/python scripts/diag_m3_policy.py \
       --ckpt /home/forke/m3_checkpoints/m3_run1/final
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iron_man_drone.envs.quadrotor_env_m3 import (
    M3VecEnv, DEPTH_N_BINS, EPISODE_STEPS, N_OBSTACLE_SLOTS, build_m3_base_obs,
)
from iron_man_drone.envs.trajectories import (
    make_figure_eight_trajectory, eval_trajectory_position, get_reference_pos,
    DT as TRAJ_DT,
)
from iron_man_drone.envs.quadrotor_env import LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
from eval_m3 import build_states, make_eval_fns
from train_m3 import load_checkpoint

DEPTH_ONES = jnp.ones(DEPTH_N_BINS)
NOMINAL_PRIV = jnp.array([1., 1., 1., 1., 1., 0., 0., 0.])


def _clear_reset(env, keys):
    """Reset with NO obstacles (parked), default poly/zigzag trajectory."""
    n = keys.shape[0]
    centers = jnp.array(np.full((n, N_OBSTACLE_SLOTS, 3), 100.0, np.float32))
    he      = jnp.zeros((n, N_OBSTACLE_SLOTS, 3))
    n_obs   = jnp.zeros((n,), jnp.int32)
    return env._reset_jit(keys, centers, he, n_obs)


def rollout(env, actor_state, enc_state, fns, keys, z_mode="encoder", steps=EPISODE_STEPS):
    """Clear-air rollout. Returns dict of per-step arrays (T, N, ...)."""
    encode, assemble, greedy_action, ref_pos_batch, _ = fns
    drone_id = env.drone_body_id
    states, base_obs, priv = _clear_reset(env, keys)
    n = keys.shape[0]
    depth = jnp.tile(DEPTH_ONES, (n, 1))

    def make_obs(states, base_obs):
        if z_mode == "oracle":
            z = jnp.tile(NOMINAL_PRIV, (n, 1))
        else:
            z = encode(enc_state.params, states.obs_base_buf, states.action_buf)
        k = states.step.astype(jnp.float32) / EPISODE_STEPS
        od = env.compute_k_nearest_batch(states)
        return assemble(base_obs, z, depth, k, od)

    actor_obs = make_obs(states, base_obs)
    P, R, A, D = [], [], [], []
    for _ in range(steps):
        action = greedy_action(actor_state.params, actor_obs)
        states, base_obs, _, _, done = env.batch_step(states, action)
        step_i = np.array(states.step)
        early = step_i < (EPISODE_STEPS - 5)          # done before timeout = divergence/crash
        P.append(np.array(states.mjx_data.xpos[:, drone_id]))
        R.append(np.array(ref_pos_batch(states.traj, states.step)))
        A.append(np.array(action))
        D.append(np.array(done) & early)
        actor_obs = make_obs(states, base_obs)
    return {"pos": np.array(P), "ref": np.array(R), "act": np.array(A), "done": np.array(D)}


def _alive_mask(roll):
    """(T,N) bool: True for steps before the env first diverges (early done)."""
    return np.cumsum(roll["done"], axis=0) == 0


def med_of(roll, warmup=0):
    """Done-masked MED (3D and XY): error only over steps where the env is still alive."""
    alive = _alive_mask(roll)[warmup:]
    err   = np.linalg.norm(roll["pos"] - roll["ref"], axis=2)[warmup:]
    errxy = np.linalg.norm(roll["pos"][..., :2] - roll["ref"][..., :2], axis=2)[warmup:]
    m = alive.sum()
    return float((err * alive).sum() / max(m, 1)), float((errxy * alive).sum() / max(m, 1))


def diverge_rate(roll):
    """Fraction of envs that diverged (went done before timeout)."""
    return float((roll["done"].sum(axis=0) > 0).mean())


def fig8_med(actor_state, enc_state, fns, n=16, z_mode="encoder"):
    """Figure-eight, T/4 init, M2 methodology -> XY MED (comparable to M2's 0.057m)."""
    encode, assemble, greedy_action, ref_pos_batch, _ = fns
    OFFSET = round(5.5 / 4 / TRAJ_DT)                       # 138
    LB = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS                   # 50
    total = EPISODE_STEPS + OFFSET + LB + 10               # 1198
    fig8 = make_figure_eight_trajectory(TRAJ_DT, total, LB, speed="normal")

    class _Cfg: num_envs = n
    env = M3VecEnv(_Cfg(), override_traj=fig8, fault_prob=0.0, mass_lo=1.0, mass_hi=1.0)
    drone_id = env.drone_body_id
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    states, base_obs, _ = env.batch_reset(keys, ["random"] * n, density_mult=0.0)
    states = states._replace(
        step=jnp.full((n,), OFFSET, jnp.int32), kf_multiplier=jnp.ones(n),
        rotor_efficiency=jnp.ones((n, 4)), mass_scale=jnp.ones(n),
        priv_state=jnp.tile(NOMINAL_PRIV, (n, 1)))
    base_obs = jax.vmap(lambda s: build_m3_base_obs(s.mjx_data, s.traj, s.step, drone_id))(states)
    depth = jnp.tile(DEPTH_ONES, (n, 1))

    def make_obs(states, base_obs):
        z = jnp.tile(NOMINAL_PRIV, (n, 1)) if z_mode == "oracle" else \
            encode(enc_state.params, states.obs_base_buf, states.action_buf)
        k = states.step.astype(jnp.float32) / EPISODE_STEPS
        od = env.compute_k_nearest_batch(states)
        return assemble(base_obs, z, depth, k, od)

    actor_obs = make_obs(states, base_obs)
    errs, dones = [], []
    for t in range(EPISODE_STEPS):
        action = greedy_action(actor_state.params, actor_obs)
        states, base_obs, _, _, done = env.batch_step(states, action)
        ref = np.array(eval_trajectory_position(fig8, jnp.array(float(OFFSET + t + 1) * TRAJ_DT)))
        pos = np.array(states.mjx_data.xpos[:, drone_id])
        errs.append(np.linalg.norm(pos[:, :2] - ref[:2], axis=1))
        dones.append(np.array(done) & (np.array(states.step) < EPISODE_STEPS - 5))
        actor_obs = make_obs(states, base_obs)
    errs = np.array(errs); alive = np.cumsum(np.array(dones), axis=0) == 0       # (T,N)
    per_env = (errs * alive).sum(axis=0) / np.maximum(alive.sum(axis=0), 1)
    div = float((np.array(dones).sum(axis=0) > 0).mean())
    return float(np.median(per_env)), div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/forke/m3_checkpoints/m3_run1/final")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    actor, critic, encoder, a_s, c_s, e_s = build_states()
    a_s, c_s, e_s, epoch = load_checkpoint(Path(args.ckpt), a_s, c_s, e_s)
    print(f"Loaded epoch {epoch} ({epoch*32768/1e6:.0f}M steps)\n")
    fns = make_eval_fns(actor, encoder)

    class _Cfg: num_envs = args.n
    print("Init M3VecEnv...")
    env = M3VecEnv(_Cfg(), fault_prob=0.0, mass_lo=1.0, mass_hi=1.0)
    keys = jax.random.split(jax.random.PRNGKey(7), args.n)

    print("\n=== M3 POLICY DIAGNOSTICS (final, clear air) ===\n")

    # Q2 + signature: poly/zigzag, encoder vs oracle z
    roll_enc = rollout(env, a_s, e_s, fns, keys, z_mode="encoder")
    roll_ora = rollout(env, a_s, e_s, fns, keys, z_mode="oracle")
    med_enc, medxy_enc = med_of(roll_enc)
    med_ora, medxy_ora = med_of(roll_ora)
    med_enc_ss, _ = med_of(roll_enc, warmup=200)   # steady-state (skip acquisition)

    dr_enc, dr_ora = diverge_rate(roll_enc), diverge_rate(roll_ora)
    print("Q2 ENCODER (poly/zigzag @3m/s, clear air; done-masked):")
    print(f"  encoder-z : MED3D={med_enc:.3f}m  XY={medxy_enc:.3f}m  (steady>200: {med_enc_ss:.3f}m)  diverged={dr_enc:.0%}")
    print(f"  oracle-z  : MED3D={med_ora:.3f}m  XY={medxy_ora:.3f}m  diverged={dr_ora:.0%}")
    print(f"  -> encoder adds {med_enc-med_ora:+.3f}m MED, {dr_enc-dr_ora:+.0%} divergence vs oracle "
          f"({'ENCODER IS A FACTOR' if (med_enc-med_ora>0.02 or dr_enc-dr_ora>0.05) else 'encoder ~neutral'})")

    # Q3 signature — alive-masked
    alive = _alive_mask(roll_enc)                                                   # (T,N)
    err_all = np.linalg.norm(roll_enc["pos"] - roll_enc["ref"], axis=2)             # (T,N)
    def masked_mean(a, m):
        return float((a * m).sum() / max(m.sum(), 1))
    err_t = np.array([masked_mean(err_all[t], alive[t]) for t in range(err_all.shape[0])])  # (T,)
    # chatter only while alive
    dact = np.abs(np.diff(roll_enc["act"], axis=0))                                 # (T-1,N,4)
    chatter = masked_mean(dact.mean(axis=2), alive[1:])
    refspeed = np.linalg.norm(np.diff(roll_enc["ref"], axis=0), axis=2) / TRAJ_DT   # (T-1,N)
    err_step = err_all[1:]; al = alive[1:]
    sp = refspeed[al]; er = err_step[al]
    lo = er[sp < np.percentile(sp, 33)].mean()
    hi = er[sp > np.percentile(sp, 67)].mean()
    print("\nQ3 SIGNATURE (encoder-z, alive only):")
    print(f"  error @ acquisition (t<50): {err_t[:50].mean():.3f}m   steady (t>200): {err_t[200:].mean():.3f}m")
    print(f"  action chatter (mean |du| per step): {chatter:.4f}  (0=smooth, >0.05=chattering)")
    print(f"  error at LOW ref-speed: {lo:.3f}m   at HIGH ref-speed: {hi:.3f}m   "
          f"(ratio {hi/max(lo,1e-6):.2f}x => {'control-authority limited' if hi/max(lo,1e-6)>1.5 else 'uniform offset'})")

    # Q1 trajectory: figure-eight (M2 methodology)
    print("\nQ1 TRAJECTORY (figure-eight, T/4 init, M2 methodology):")
    fe_enc, fe_enc_div = fig8_med(a_s, e_s, fns, n=16, z_mode="encoder")
    fe_ora, fe_ora_div = fig8_med(a_s, e_s, fns, n=16, z_mode="oracle")
    print(f"  figure-eight XY MED: encoder-z={fe_enc:.3f}m (div {fe_enc_div:.0%})  "
          f"oracle-z={fe_ora:.3f}m (div {fe_ora_div:.0%})   (M2 ref: 0.057m)")
    print(f"  poly/zigzag XY MED (encoder): {medxy_enc:.3f}m  (div {dr_enc:.0%})")
    print(f"  -> figure-eight is {'MUCH easier' if fe_enc < medxy_enc*0.7 else 'similar'} => "
          f"{'trajectory difficulty is a big factor' if fe_enc < medxy_enc*0.7 else 'policy/reward limited, not trajectory'}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(err_t); ax[0].axhline(0.10, color="r", ls="--", label="gate 0.10m")
        ax[0].set_title("Tracking error over episode"); ax[0].set_xlabel("step"); ax[0].set_ylabel("3D err (m)"); ax[0].legend()
        ax[1].plot(roll_enc["act"][:300, 0]); ax[1].set_title("Action[0] (first 300 steps) — chatter check"); ax[1].set_xlabel("step")
        ax[2].hist(er, bins=40); ax[2].axvline(0.10, color="r", ls="--")
        ax[2].set_title("Per-step error distribution"); ax[2].set_xlabel("err (m)")
        out = ROOT / "experiments" / "m3_run1" / "diag"; out.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out / "diag_final.png", dpi=90)
        print(f"\nplot -> {out/'diag_final.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
