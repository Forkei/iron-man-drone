"""
Phase 2 closed-loop evaluation.

Deploys Phase 1 actor + Phase 2 encoder in closed loop.
At each timestep the encoder ingests a 0.5 s history window and outputs ê_t,
which is denormalized and substituted for ground-truth e_t in the actor obs.

Ring buffer: (H=50, 46) float32, zeros at episode start.
  pair[t] = (obs_base[t], action[t-1]) = 46-dim
  Window at step t: ring_buf rolled so newest pair is at [-1].

Eval suite: figure-eight (slow/normal/fast), pentagram (slow/fast),
            polynomial, zigzag — nominal and fault η=0.70 conditions.
3 seeds per (traj, condition) pair.  Crash-only termination (no timeout).

Phase 2 pass criteria (figure_eight_normal):
  Nominal   MED ≤ 0.065 m  (Phase 1 privileged baseline: 0.057 m)
  Fault     MED ≤ 0.100 m  (Phase 1 privileged baseline: 0.079 m)

Usage:
  python scripts/eval_m2_phase2.py
  python scripts/eval_m2_phase2.py --actor_checkpoint experiments/.../final
  python scripts/eval_m2_phase2.py --encoder_checkpoint experiments/phase2_encoder/best_checkpoint
"""

import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

REPO_ROOT = Path(__file__).parent.parent

DEFAULT_ACTOR_CHECKPOINT = (
    REPO_ROOT / "experiments/m2_phase1_baseline"
    / "m2_phase1_baseline_1778244202/checkpoints/final"
)
DEFAULT_ENCODER_CHECKPOINT = REPO_ROOT / "experiments/phase2_encoder/best_checkpoint"

# Phase 1 privileged-state reference numbers (T/4-corrected, 3-seed mean)
PHASE1_REF = {"nominal": 0.0574, "fault_eta70": 0.0790}

GATE_NOMINAL = 0.065
GATE_FAULT   = 0.100

H          = 50
OBS_DIM    = 42
ACTION_DIM = 4
PAIR_DIM   = OBS_DIM + ACTION_DIM   # 46
WINDOW_DIM = H * PAIR_DIM           # 2300
EVAL_SEEDS = [42, 99, 7]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_checkpoint",   default=str(DEFAULT_ACTOR_CHECKPOINT))
    parser.add_argument("--encoder_checkpoint", default=str(DEFAULT_ENCODER_CHECKPOINT))
    parser.add_argument("--out_dir",            default=str(REPO_ROOT / "experiments/phase2_eval"))
    args = parser.parse_args()

    actor_ckpt_path   = Path(args.actor_checkpoint).resolve()
    encoder_ckpt_path = Path(args.encoder_checkpoint).resolve()
    out_dir           = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices        : {jax.devices()}")
    print(f"Actor checkpoint   : {actor_ckpt_path}")
    print(f"Encoder checkpoint : {encoder_ckpt_path}")
    print()

    # ── Phase 1 actor (frozen) ────────────────────────────────────────────────
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states
    ppo_cfg = PPOConfig(
        actor_obs_dim=50, critic_obs_dim=51,
        action_dim=4, hidden_dim=256, num_layers=3,
    )
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer = ocp.PyTreeCheckpointer()
    restored_actor = checkpointer.restore(str(actor_ckpt_path))
    actor_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored_actor["actor"]["params"]
    )
    actor_state = actor_state.replace(params=actor_params)
    print("Phase 1 actor loaded.")

    @jax.jit
    def actor_apply(params, obs):
        return actor_state.apply_fn(params, obs)

    # ── Phase 2 encoder ───────────────────────────────────────────────────────
    from iron_man_drone.policy.encoder import AdaptationEncoder, denormalize_e_hat
    encoder    = AdaptationEncoder()
    restored_enc = checkpointer.restore(str(encoder_ckpt_path))
    enc_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored_enc["params"]
    )
    print("Phase 2 encoder loaded.")

    @jax.jit
    def encoder_apply(params, window):
        return encoder.apply(params, window)

    # Warmup both networks
    _ = actor_apply(actor_params, jnp.zeros((1, 50)))
    _ = encoder_apply(enc_params, jnp.zeros((1, WINDOW_DIM)))
    print("Actor + encoder JIT warmed up.")
    print()

    # ── Environment ───────────────────────────────────────────────────────────
    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, EPISODE_STEPS, DT, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS,
        _build_obs, MIN_HEIGHT, MAX_HEIGHT_ABOVE_REF, MAX_TILT_RAD,
    )

    class _Cfg:
        num_envs = 1

    env      = VecEnv(_Cfg(), fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)
    drone_id = env.mj_model.body("drone").id
    _cos_max_tilt = float(jnp.cos(MAX_TILT_RAD))

    # ── Trajectory suite + conditions ─────────────────────────────────────────
    from iron_man_drone.evaluation.eval_suite import (
        make_m2_traj_suite, nominal_condition, fault_condition,
        EpisodeResult, ConditionResult, F8_OFFSETS,
    )
    from iron_man_drone.envs.trajectories import get_reference_pos

    traj_configs = make_m2_traj_suite()
    conditions   = [nominal_condition(), fault_condition(0.70)]

    # ── Per-episode eval with ring-buffer encoder ─────────────────────────────

    def precompute_refs(traj, offset_steps: int) -> jnp.ndarray:
        steps = jnp.arange(EPISODE_STEPS, dtype=jnp.int32) + offset_steps
        return jnp.array(jax.vmap(lambda s: get_reference_pos(traj, s)[:2])(steps))

    @jax.jit
    def eval_episode(
        actor_params,
        enc_params,
        reset_key: jnp.ndarray,
        priv_state_override: jnp.ndarray,  # (8,) — sets physics; encoder replaces in actor obs
        eval_traj,
        ref_xy: jnp.ndarray,               # (EPISODE_STEPS, 2)
        offset_steps: jnp.ndarray,
    ):
        state, _, _ = env._reset_fn(reset_key)
        state = state._replace(
            traj=eval_traj,
            priv_state=priv_state_override,
            rotor_efficiency=priv_state_override[:4],
            mass_scale=priv_state_override[4],
            kf_multiplier=jnp.ones(()),
            step=jnp.int32(offset_steps),
        )
        full_obs, _ = _build_obs(
            state.mjx_data, eval_traj, state.step, drone_id, priv_state_override
        )
        obs_base = full_obs[:OBS_DIM]  # (42,)

        ring_buf    = jnp.zeros((H, PAIR_DIM))   # oldest → [0], newest → [H-1]
        prev_action = jnp.zeros(ACTION_DIM)
        cos_tilt    = jnp.array(_cos_max_tilt)

        def scan_step(carry, ref_xy_t):
            state, obs_base, ring_buf, prev_action, already_done = carry

            # Build pair_t = (obs_base[t], action[t-1]) and push into ring buffer
            pair_t   = jnp.concatenate([obs_base, prev_action])          # (46,)
            new_ring = jnp.concatenate([ring_buf[1:], pair_t[None]], axis=0)  # (H, 46)

            # Encoder: window → ê_t_norm → ê_t_raw (physical units)
            window    = new_ring.reshape(1, -1)                   # (1, 2300)
            e_hat_n   = encoder_apply(enc_params, window)[0]      # (8,) in [-1, 1]
            e_hat_raw = denormalize_e_hat(e_hat_n)                # (8,) physical

            # Actor: [obs_base (42), ê_t_raw (8)] = 50-dim
            actor_obs = jnp.concatenate([obs_base, e_hat_raw])[None]   # (1, 50)
            mean, _   = actor_apply(actor_params, actor_obs)
            action    = mean[0]                                          # (4,)

            # Step environment
            new_state, new_full_obs, _, _, _ = env._step_fn(state, action)
            new_state    = new_state._replace(traj=eval_traj)
            new_obs_base = new_full_obs[:OBS_DIM]                       # (42,)

            # Crash-only termination (no step-timeout — T/4 offset would fire early)
            pos        = new_state.mjx_data.xpos[drone_id]
            body_z_z   = new_state.mjx_data.xmat[drone_id].reshape(-1)[8]
            horiz_dist = jnp.linalg.norm(pos[:2] - ref_xy_t)
            crash = (
                (pos[2] < MIN_HEIGHT)
                | (horiz_dist > MAX_HEIGHT_ABOVE_REF)
                | (jnp.abs(pos[2] - 1.0) > MAX_HEIGHT_ABOVE_REF)
                | (body_z_z < cos_tilt)
            )
            error  = jnp.linalg.norm(pos[:2] - ref_xy_t)
            active = ~already_done

            new_carry = (new_state, new_obs_base, new_ring, action, already_done | crash)
            return new_carry, (jnp.where(active, error, 0.0), active)

        init_carry = (state, obs_base, ring_buf, prev_action, jnp.bool_(False))
        _, (errors, active_mask) = jax.lax.scan(scan_step, init_carry, ref_xy)

        n   = jnp.sum(active_mask)
        med = jnp.where(n > 0, jnp.sum(errors) / n.astype(jnp.float32), jnp.nan)
        return med, n

    # ── JIT warmup ────────────────────────────────────────────────────────────
    print("Warming up JIT (first compile ~2 min)...")
    t_wup = time.time()
    from iron_man_drone.envs.trajectories import make_figure_eight_trajectory
    lookahead = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    _off = F8_OFFSETS["normal"]
    _dummy_traj = make_figure_eight_trajectory(
        DT, EPISODE_STEPS + _off + lookahead + 5, lookahead, speed="normal"
    )
    _dummy_ref  = precompute_refs(_dummy_traj, _off)
    _dummy_priv = nominal_condition().priv_state
    _m, _n = eval_episode(
        actor_params, enc_params,
        jax.random.PRNGKey(0), _dummy_priv,
        _dummy_traj, _dummy_ref, jnp.int32(_off),
    )
    jax.block_until_ready((_m, _n))
    print(f"JIT warmed up in {time.time()-t_wup:.1f} s.\n")

    # ── Eval loop ─────────────────────────────────────────────────────────────
    results: dict[str, dict[str, ConditionResult]] = {}
    t_eval = time.time()

    for tc in traj_configs:
        results[tc.name] = {}
        ref_xy     = precompute_refs(tc.traj, tc.offset_steps)
        offset_tag = f"[T/4 offset={tc.offset_steps}]" if tc.offset_steps > 0 else "[t=0]"
        print(f"── {tc.name} {offset_tag}")

        for cond in conditions:
            print(f"  {cond.name}:")
            seed_results: list[EpisodeResult] = []
            for seed in EVAL_SEEDS:
                _med, _n = eval_episode(
                    actor_params, enc_params,
                    jax.random.PRNGKey(seed), cond.priv_state,
                    tc.traj, ref_xy, jnp.int32(tc.offset_steps),
                )
                med     = float(_med)
                n_steps = int(_n)
                crashed = n_steps < EPISODE_STEPS
                seed_results.append(EpisodeResult(med=med, n_steps=n_steps, crashed=crashed))
                print(f"      seed {seed:3d}: {med:.4f} m  ({n_steps} steps"
                      f"{'  CRASH' if crashed else ''})")

            meds     = [r.med for r in seed_results]
            mean_med = float(np.mean(meds))
            results[tc.name][cond.name] = ConditionResult(
                mean_med=mean_med, per_seed=seed_results
            )
            print(f"    → {cond.name}: {mean_med:.4f} m  [{min(meds):.4f}, {max(meds):.4f}]")
        print()

    total_eval_time = time.time() - t_eval

    # ── Results table ─────────────────────────────────────────────────────────
    cond_names = [c.name for c in conditions]
    col_w      = 18
    divider_w  = 26 + col_w * len(cond_names)
    print("=" * divider_w)
    print("  Phase 2 Closed-Loop Evaluation — encoder deployed")
    print("=" * divider_w)
    print(f"  {'Trajectory':<24}" + "".join(f"{c:>{col_w}}" for c in cond_names))
    print("  " + "-" * (divider_w - 2))
    for traj_name, cond_res in results.items():
        row = f"  {traj_name:<24}"
        for cn in cond_names:
            cr = cond_res[cn]
            mark = "*" if cr.any_crashed else " "
            row += f"{cr.mean_med:>{col_w - 2}.4f} m{mark}"
        print(row)
    print("=" * divider_w)
    print("  * = at least one seed crashed")
    print()

    # ── Phase 1 comparison ────────────────────────────────────────────────────
    f8n      = results.get("figure_eight_normal", {})
    nom_cr   = f8n.get("nominal")
    fault_cr = f8n.get("fault_eta70")

    print("  figure_eight_normal — Phase 1 vs Phase 2:")
    print(f"  {'Condition':<16}  {'Phase 1':>10}  {'Phase 2':>10}  {'Δ%':>8}  {'Gate':>8}  Status")
    print("  " + "-" * 66)

    def _row(label, cr, ph1_ref, gate):
        if cr is None:
            return f"  {label:<16}  {ph1_ref:>10.4f}m  {'N/A':>10}  {'N/A':>8}  {gate:.3f}m  N/A"
        med = cr.mean_med
        pct = (med - ph1_ref) / ph1_ref * 100.0
        ok  = med <= gate
        return (
            f"  {label:<16}  {ph1_ref:>10.4f}m  {med:>10.4f}m  {pct:>+7.1f}%  "
            f"{gate:.3f}m  {'PASS' if ok else 'FAIL'}"
        )

    print(_row("nominal",      nom_cr,   PHASE1_REF["nominal"],    GATE_NOMINAL))
    print(_row("fault η=0.70", fault_cr, PHASE1_REF["fault_eta70"], GATE_FAULT))
    print()

    # ── Gate verdict ──────────────────────────────────────────────────────────
    nom_pass   = nom_cr   is not None and nom_cr.mean_med   <= GATE_NOMINAL
    fault_pass = fault_cr is not None and fault_cr.mean_med <= GATE_FAULT

    print(f"  Phase 2 gate (figure_eight_normal):")
    print(f"    Nominal ≤ {GATE_NOMINAL:.3f} m  : {'PASS' if nom_pass   else 'FAIL'}")
    print(f"    Fault   ≤ {GATE_FAULT:.3f} m  : {'PASS' if fault_pass else 'FAIL'}")
    print()

    if nom_pass and fault_pass:
        print("  ✓  PHASE 2 GATE PASSED — ready to proceed to M3.")
    else:
        print("  ✗  PHASE 2 GATE FAILED — diagnose before proceeding.")
        if not nom_pass:
            print("     Nominal fails → likely zero-padding instability (F5) or actor")
            print("     sensitivity to encoder noise (F7). Check training MSE first.")
        if not fault_pass:
            print("     Fault fails → encoder may not be identifying fault correctly.")
            print("     Check per-channel η MSE from train_phase2_encoder.py output.")
        print("     See notes/M2_phase2_spec.md §F5–F7 for diagnostics.")

    print(f"\n  Eval time : {total_eval_time / 60:.1f} min")
    print("=" * divider_w)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        traj: {
            cond: {
                "mean_med": float(cr.mean_med),
                "any_crashed": bool(cr.any_crashed),
                "meds": [float(m) for m in cr.meds],
            }
            for cond, cr in conds.items()
        }
        for traj, conds in results.items()
    }
    summary_path = out_dir / "eval_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {summary_path}")


if __name__ == "__main__":
    main()
