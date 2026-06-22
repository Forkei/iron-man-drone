"""
Minimal standalone: does the FINAL M3 policy track the figure-eight tightly?

Disambiguates the two candidate causes of M3's loose tracking:
  - if fig8 XY MED ~0.05m  -> looseness is the (infeasible zigzag) TRAJECTORIES;
                              the policy CAN track tight, fix = feasible trajectories.
  - if fig8 XY MED ~0.3m   -> the policy itself tracks loose (fault-robustness tax /
                              reward balance); deeper fix needed.

Single env build (the full diag hung building multiple MJWarp envs in one process).
M2 reference on this trajectory/methodology: 0.057m.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from iron_man_drone.envs.quadrotor_env_m3 import M3VecEnv, DEPTH_N_BINS, EPISODE_STEPS, build_m3_base_obs
from iron_man_drone.envs.trajectories import make_figure_eight_trajectory, eval_trajectory_position, DT as TRAJ_DT
from iron_man_drone.envs.quadrotor_env import LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
from eval_m3 import build_states, make_eval_fns
from train_m3 import load_checkpoint

NOMINAL = jnp.array([1., 1., 1., 1., 1., 0., 0., 0.])

def main():
    ck = "/home/forke/m3_checkpoints/m3_run1/final"
    actor, critic, encoder, a_s, c_s, e_s = build_states()
    a_s, c_s, e_s, epoch = load_checkpoint(Path(ck), a_s, c_s, e_s)
    print(f"Loaded epoch {epoch}", flush=True)
    encode, assemble, greedy, ref_pos_batch, _ = make_eval_fns(actor, encoder)

    n = 16
    OFFSET = round(5.5 / 4 / TRAJ_DT)
    LB = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    fig8 = make_figure_eight_trajectory(TRAJ_DT, EPISODE_STEPS + OFFSET + LB + 10, LB, speed="normal")

    class _Cfg: num_envs = n
    print("building env...", flush=True)
    env = M3VecEnv(_Cfg(), override_traj=fig8, fault_prob=0.0, mass_lo=1.0, mass_hi=1.0)
    did = env.drone_body_id
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    states, base_obs, _ = env.batch_reset(keys, ["random"] * n, density_mult=0.0)
    states = states._replace(step=jnp.full((n,), OFFSET, jnp.int32), kf_multiplier=jnp.ones(n),
                             rotor_efficiency=jnp.ones((n, 4)), mass_scale=jnp.ones(n),
                             priv_state=jnp.tile(NOMINAL, (n, 1)))
    base_obs = jax.vmap(lambda s: build_m3_base_obs(s.mjx_data, s.traj, s.step, did))(states)
    depth = jnp.tile(jnp.ones(DEPTH_N_BINS), (n, 1))
    print("rolling out...", flush=True)

    def obs(states, base_obs):
        z = encode(e_s.params, states.obs_base_buf, states.action_buf)
        k = states.step.astype(jnp.float32) / EPISODE_STEPS
        od = env.compute_k_nearest_batch(states)
        return assemble(base_obs, z, depth, k, od)

    ao = obs(states, base_obs)
    errs, dones = [], []
    for t in range(EPISODE_STEPS):
        states, base_obs, _, _, done = env.batch_step(states, greedy(a_s.params, ao))
        ref = np.array(eval_trajectory_position(fig8, jnp.array(float(OFFSET + t + 1) * TRAJ_DT)))
        pos = np.array(states.mjx_data.xpos[:, did])
        errs.append(np.linalg.norm(pos[:, :2] - ref[:2], axis=1))
        dones.append(np.array(done) & (np.array(states.step) < EPISODE_STEPS - 5))
        ao = obs(states, base_obs)
    errs = np.array(errs); alive = np.cumsum(np.array(dones), axis=0) == 0
    per_env = (errs * alive).sum(axis=0) / np.maximum(alive.sum(axis=0), 1)
    div = float((np.array(dones).sum(axis=0) > 0).mean())
    print(f"\nFIGURE-EIGHT (final M3 policy, encoder-z, done-masked):", flush=True)
    print(f"  XY MED = {float(np.median(per_env)):.3f} m   diverged = {div:.0%}   (M2 ref 0.057m)", flush=True)
    print(f"  per-env MED sorted: {np.sort(per_env).round(3).tolist()}", flush=True)

if __name__ == "__main__":
    main()
