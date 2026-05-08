"""
Standalone MED eval on a saved M2 Phase 1 checkpoint.

Runs figure_eight_normal in two conditions:
  1. Nominal  — η=[1,1,1,1], kf=1.0, mass=1.0  (target ≤ 0.037 m)
  2. Fault    — rotor 0 at η=0.70, kf=1.0, mass=1.0  (target ≤ 0.060 m)

Methodology matches M1.3's in-training eval (train_m1.py _run_med_eval):
  - kf_multiplier fixed to 1.0 (M1.3 always passed jnp.ones(()) for kf_mult)
  - Same trajectory: figure_eight_normal, speed="normal"
  - Same episode length: 1000 steps
  - No T/4 phase offset (M1.3 training eval did not use it; only eval_m1_full.py does)
  - Physics and actor priv_state both set to match eval condition

Usage:
  python scripts/eval_m2_checkpoint.py --checkpoint PATH/checkpoints/final
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent

EVAL_SEEDS = [42, 99, 7]   # three seeds; report mean, min, max


def run_single_eval(actor_state, env_reset, env_step, eval_traj, drone_id,
                    priv_state, seed: int,
                    ref_positions_xy: np.ndarray) -> tuple[float, int]:
    """
    One deterministic episode on figure_eight_normal.
    priv_state (8-dim): sets actor input, rotor_efficiency, mass_scale, AND kf_multiplier=1.0.
    ref_positions_xy: (EPISODE_STEPS, 2) numpy array pre-computed outside the loop
                      to avoid per-step JAX kernel accumulation.
    Returns (MED, steps_completed).
    """
    from iron_man_drone.envs.quadrotor_env import _build_obs, EPISODE_STEPS

    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)

    state, _, _ = env_reset(rk)
    state = state._replace(
        traj=eval_traj,
        priv_state=priv_state,
        rotor_efficiency=priv_state[:4],
        mass_scale=priv_state[4],
        kf_multiplier=jnp.ones(()),   # match M1.3: kf_mult=1.0, not random
    )
    a_obs, _ = _build_obs(state.mjx_data, eval_traj, state.step, drone_id, priv_state)

    positions = []
    n_steps = 0
    for si in range(EPISODE_STEPS):
        mean, _ = actor_state.apply_fn(actor_state.params, a_obs[None])
        action = mean[0]
        state, a_obs, _, _, done = env_step(state, action)
        state = state._replace(traj=eval_traj)
        positions.append(np.array(state.mjx_data.xpos[drone_id, :2]))
        n_steps = si + 1
        if bool(done):
            break

    if not positions:
        return float("nan"), 0
    positions = np.array(positions)
    refs      = ref_positions_xy[:n_steps]
    med       = float(np.linalg.norm(positions - refs, axis=1).mean())
    return med, n_steps


def run_condition(actor_state, env_reset, env_step, eval_traj, drone_id,
                  priv_state, label: str, target_str: str,
                  ref_positions_xy: np.ndarray) -> list[float]:
    meds = []
    steps_list = []
    for seed in EVAL_SEEDS:
        med, steps = run_single_eval(
            actor_state, env_reset, env_step, eval_traj, drone_id, priv_state, seed,
            ref_positions_xy
        )
        meds.append(med)
        steps_list.append(steps)
        print(f"    seed={seed:3d}: MED={med:.4f} m  ({steps} steps)")

    mean_med = float(np.mean(meds))
    print(f"  → {label}: mean={mean_med:.4f} m  "
          f"[{min(meds):.4f}, {max(meds):.4f}]  {target_str}")
    return meds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint dir (e.g. .../checkpoints/final)")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Eval seeds : {EVAL_SEEDS}")
    print()

    from iron_man_drone.envs.quadrotor_env import VecEnv, DT, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
    from iron_man_drone.envs.trajectories import make_figure_eight_trajectory
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    class _Cfg:
        num_envs = 1

    env      = VecEnv(_Cfg(), fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)
    drone_id = env.mj_model.body("drone").id

    ppo_cfg = PPOConfig(actor_obs_dim=50, critic_obs_dim=51,
                        action_dim=4, hidden_dim=256, num_layers=3)
    key = jax.random.PRNGKey(0)
    _, _, actor_state, critic_state = create_train_states(key, ppo_cfg)

    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    # Restore raw (no item= template) to avoid CPU→GPU sharding deadlock,
    # then manually move params to the default device.
    restored_raw = checkpointer.restore(args.checkpoint)
    actor_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored_raw["actor"]["params"]
    )
    actor_state = actor_state.replace(params=actor_params)
    print("Checkpoint loaded.\n")

    eval_traj  = make_figure_eight_trajectory(
        DT, 1000, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal"
    )
    env_reset  = jax.jit(env._reset_fn)
    env_step   = jax.jit(env._step_fn)

    # Pre-compute reference positions as numpy — avoids per-step JAX kernel
    # accumulation (jnp.int32(si) inside a Python loop creates 1000 unique traces).
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS
    from iron_man_drone.envs.trajectories import get_reference_pos
    _all_steps = jnp.arange(EPISODE_STEPS, dtype=jnp.int32)
    ref_positions_xy = np.array(
        jax.vmap(lambda s: get_reference_pos(eval_traj, s))(_all_steps)
    )[:, :2]  # (EPISODE_STEPS, 2)
    print(f"Reference positions precomputed: {ref_positions_xy.shape}")

    # JIT warm-up
    _k = jax.random.PRNGKey(999)
    _s, _, _ = env_reset(_k)
    _priv_w = jnp.concatenate([jnp.ones(4), jnp.ones(1), jnp.zeros(3)])
    _s = _s._replace(traj=eval_traj, priv_state=_priv_w,
                     rotor_efficiency=_priv_w[:4], mass_scale=_priv_w[4],
                     kf_multiplier=jnp.ones(()))
    from iron_man_drone.envs.quadrotor_env import _build_obs
    _ao, _ = _build_obs(_s.mjx_data, eval_traj, _s.step, drone_id, _priv_w)
    _m, _ = actor_state.apply_fn(actor_state.params, _ao[None])
    env_step(_s, _m[0])
    print("JIT warmed up.\n")

    # ── Condition 1: Nominal ──────────────────────────────────────────────────
    priv_nominal = jnp.concatenate([jnp.ones(4), jnp.ones(1), jnp.zeros(3)])
    print("Nominal (η=[1,1,1,1], kf=1.0, mass=1.0):")
    meds_nominal = run_condition(
        actor_state, env_reset, env_step, eval_traj, drone_id,
        priv_nominal, "Nominal", "target ≤ 0.037 m  |  M1.3 ep-1000 ref: 0.099 m",
        ref_positions_xy
    )

    print()

    # ── Condition 2: Rotor 0 at η=0.70 ───────────────────────────────────────
    priv_fault70 = jnp.concatenate([
        jnp.array([0.7, 1.0, 1.0, 1.0]),
        jnp.ones(1),
        jnp.zeros(3),
    ])
    print("Fault rotor 0 η=0.70 (kf=1.0, mass=1.0):")
    meds_fault70 = run_condition(
        actor_state, env_reset, env_step, eval_traj, drone_id,
        priv_fault70, "Fault η=0.70", "target ≤ 0.060 m (Phase 2 go/no-go)",
        ref_positions_xy
    )

    print()
    print("=" * 58)
    print("SUMMARY")
    print("=" * 58)
    mean_nom  = float(np.mean(meds_nominal))
    mean_f70  = float(np.mean(meds_fault70))
    ratio     = mean_nom / 0.099 if mean_nom < 9 else float("inf")
    nom_ok    = mean_nom <= 0.119   # within 20% of M1.3 0.099 m
    print(f"  Nominal  mean MED : {mean_nom:.4f} m  "
          f"({'within' if nom_ok else 'OUTSIDE'} 20% of M1.3 0.099 m, ratio={ratio:.2f}×)")
    print(f"  Fault70  mean MED : {mean_f70:.4f} m  "
          f"({'PASS' if mean_f70 <= 0.060 else 'MISS'} Phase 2 target)")
    print()
    print("Physics fix sanity (fault MED >> nominal MED → fix working):")
    if mean_f70 > mean_nom * 1.5:
        print(f"  PASS — fault MED is {mean_f70/mean_nom:.1f}× nominal (physics degradation applied)")
    else:
        print(f"  FAIL — fault MED only {mean_f70/mean_nom:.1f}× nominal (physics may not be applying)")
    print()
    print("Note: this is the 1k-epoch nominal-only checkpoint.")
    print("Phase 1 go/no-go is evaluated at 15k epochs with full DR.")


if __name__ == "__main__":
    main()
