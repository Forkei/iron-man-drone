"""
M2 Phase 1 full eval suite.

Trajectories  : figure-eight slow/normal/fast, pentagram slow/fast,
                polynomial (fixed seed), zigzag (fixed seed)
Conditions    : nominal  (η=[1,1,1,1], kf=1.0, mass=1.0)
                fault    (rotor 0 at η=0.70, kf=1.0, mass=1.0)
Seeds         : 3 starting positions each  →  7 trajs × 2 conds × 3 seeds = 42 episodes
Eval method   : lax.scan — full episode compiled as one XLA kernel (~1 s each after warmup)

Usage:
  python scripts/eval_m2_full.py --checkpoint PATH/checkpoints/final
  python scripts/eval_m2_full.py --checkpoint PATH/checkpoints/final --fault_eta 0.50
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT  = Path(__file__).parent.parent
EVAL_SEEDS = [42, 99, 7]

# M2 spec targets (from config eval section)
TARGETS = {
    "figure_eight_normal_nominal": 0.037,
    "figure_eight_normal_eta70":   0.060,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint dir (e.g. .../checkpoints/final)")
    parser.add_argument("--fault_eta", type=float, default=0.70,
                        help="Rotor-0 efficiency for fault condition (default 0.70)")
    args = parser.parse_args()

    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, EPISODE_STEPS, _build_obs, DT, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS,
    )
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, make_pentagram_trajectory,
        sample_polynomial_trajectory, sample_zigzag_trajectory,
        get_reference_pos,
    )
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    print(f"JAX devices : {jax.devices()}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Fault η     : {args.fault_eta}")
    print(f"Eval seeds  : {EVAL_SEEDS}")
    print()

    # ── Env (1 env, no DR for eval) ─────────────────────────────────────────
    class _Cfg:
        num_envs = 1

    env      = VecEnv(_Cfg(), fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)
    drone_id = env.mj_model.body("drone").id

    # ── Policy ──────────────────────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        actor_obs_dim=50, critic_obs_dim=51,
        action_dim=4, hidden_dim=256, num_layers=3,
    )
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)

    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    restored     = checkpointer.restore(args.checkpoint)
    actor_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored["actor"]["params"]
    )
    actor_state = actor_state.replace(params=actor_params)
    print("Checkpoint loaded.\n")

    # ── Priv states ──────────────────────────────────────────────────────────
    priv_nominal = jnp.concatenate([jnp.ones(4), jnp.ones(1), jnp.zeros(3)])
    priv_fault   = jnp.concatenate([
        jnp.array([args.fault_eta, 1.0, 1.0, 1.0]),
        jnp.ones(1), jnp.zeros(3),
    ])

    # ── Scan-based episode (JIT compiled once, all trajectories reuse kernel) ─
    @jax.jit
    def eval_episode(actor_params, reset_key, priv_override, eval_traj, ref_xy):
        """Full 1000-step episode via lax.scan. Steps after crash masked from MED."""
        state, _, _ = env._reset_fn(reset_key)
        state = state._replace(
            traj=eval_traj,
            priv_state=priv_override,
            rotor_efficiency=priv_override[:4],
            mass_scale=priv_override[4],
            kf_multiplier=jnp.ones(()),
        )
        a_obs, _ = _build_obs(
            state.mjx_data, eval_traj, state.step, drone_id, priv_override
        )

        def scan_step(carry, ref_xy_t):
            state, a_obs, already_done = carry
            mean, _ = actor_state.apply_fn(actor_params, a_obs[None])
            action   = mean[0]
            new_state, new_a_obs, _, _, done = env._step_fn(state, action)
            new_state = new_state._replace(traj=eval_traj)
            pos_xy  = new_state.mjx_data.xpos[drone_id, :2]
            error   = jnp.linalg.norm(pos_xy - ref_xy_t)
            active  = ~already_done
            return (new_state, new_a_obs, already_done | done), (
                jnp.where(active, error, 0.0), active
            )

        _, (errors, active_mask) = jax.lax.scan(
            scan_step, (state, a_obs, jnp.bool_(False)), ref_xy
        )
        n   = jnp.sum(active_mask)
        med = jnp.where(n > 0, jnp.sum(errors) / n.astype(jnp.float32), jnp.nan)
        return med, n

    def precompute_refs(traj):
        steps = jnp.arange(EPISODE_STEPS, dtype=jnp.int32)
        return jnp.array(jax.vmap(lambda s: get_reference_pos(traj, s)[:2])(steps))

    def run_condition(label, priv, traj, ref_xy):
        meds = []
        for seed in EVAL_SEEDS:
            _med, _steps = eval_episode(
                actor_params, jax.random.PRNGKey(seed), priv, traj, ref_xy
            )
            med, steps = float(_med), int(_steps)
            meds.append(med)
            flag = f"  ({steps} steps)"
            print(f"      seed {seed:3d}: {med:.4f} m{flag}")
        mean_med = float(np.mean(meds))
        print(f"    → {label}: {mean_med:.4f} m  [{min(meds):.4f}, {max(meds):.4f}]")
        return mean_med, meds

    # ── JIT warmup ───────────────────────────────────────────────────────────
    LOOKAHEAD = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    print("Warming up JIT (compiles full 1000-step scan — ~1 min)...")
    _wt  = make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD, speed="normal")
    _wr  = precompute_refs(_wt)
    eval_episode(actor_params, jax.random.PRNGKey(0), priv_nominal, _wt, _wr)
    print("JIT warmed up.\n")

    # ── Trajectory suite ─────────────────────────────────────────────────────
    # polynomial and zigzag use fixed trajectory seeds so results are reproducible.
    traj_suite = [
        ("figure_eight_slow",
         make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD, speed="slow")),
        ("figure_eight_normal",
         make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD, speed="normal")),
        ("figure_eight_fast",
         make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD, speed="fast")),
        ("pentagram_slow",
         make_pentagram_trajectory(DT, EPISODE_STEPS, LOOKAHEAD, speed="slow")),
        ("pentagram_fast",
         make_pentagram_trajectory(DT, EPISODE_STEPS, LOOKAHEAD, speed="fast")),
        ("polynomial",
         sample_polynomial_trajectory(jax.random.PRNGKey(42), DT, EPISODE_STEPS, LOOKAHEAD)),
        ("zigzag",
         sample_zigzag_trajectory(jax.random.PRNGKey(42), DT, EPISODE_STEPS, LOOKAHEAD)),
    ]

    results = {}
    for name, traj in traj_suite:
        print(f"── {name} " + "─" * max(1, 45 - len(name)))
        ref_xy = precompute_refs(traj)
        print("  Nominal:")
        nom_mean, nom_meds = run_condition("nominal", priv_nominal, traj, ref_xy)
        print(f"  Fault η={args.fault_eta}:")
        flt_mean, flt_meds = run_condition(f"fault η={args.fault_eta}", priv_fault, traj, ref_xy)
        results[name] = (nom_mean, flt_mean, nom_meds, flt_meds)
        print()

    # ── Summary table ────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  M2 Phase 1 Final Eval — 15 000 epochs — {Path(args.checkpoint).parent.parent.name}")
    print(f"  Fault condition: rotor 0 at η={args.fault_eta}")
    print("=" * 72)
    print(f"  {'Trajectory':<22} {'Nominal':>10} {'Fault':>10} {'F/N ratio':>10}  Status")
    print("  " + "-" * 68)
    for name, (nom, flt, _, _) in results.items():
        ratio = flt / nom if nom > 0 else float("inf")
        # Check against spec targets if available
        nom_key = f"{name}_nominal"
        flt_key = f"{name}_eta{int(args.fault_eta * 100)}"
        nom_tgt = TARGETS.get(nom_key)
        flt_tgt = TARGETS.get(flt_key)
        status_parts = []
        if nom_tgt:
            status_parts.append(f"nom {'✓' if nom <= nom_tgt else '✗'}{nom_tgt:.3f}")
        if flt_tgt:
            status_parts.append(f"flt {'✓' if flt <= flt_tgt else '✗'}{flt_tgt:.3f}")
        status = "  ".join(status_parts)
        print(f"  {name:<22} {nom:>10.4f} {flt:>10.4f} {ratio:>10.1f}×  {status}")
    print("=" * 72)

    nom_all = [v[0] for v in results.values()]
    flt_all = [v[1] for v in results.values()]
    print(f"\n  Overall nominal MED : mean={np.mean(nom_all):.4f}  "
          f"min={np.min(nom_all):.4f}  max={np.max(nom_all):.4f}")
    print(f"  Overall fault MED   : mean={np.mean(flt_all):.4f}  "
          f"min={np.min(flt_all):.4f}  max={np.max(flt_all):.4f}")

    # Physics fix sanity: fault MED should be > nominal on average
    mean_ratio = np.mean(flt_all) / np.mean(nom_all)
    sane = mean_ratio > 1.2
    print(f"\n  Physics fix sanity  : fault/nominal = {mean_ratio:.2f}×  "
          f"({'PASS' if sane else 'FAIL — fault physics may not be applying'})")


if __name__ == "__main__":
    main()
