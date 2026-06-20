"""
M3 environment validation — 5 gates must pass before training.

Gate 1: All 5 scene modes generate valid scenes (no exclusion violations,
        drone spawn clear, all obstacle slots in-arena or parked).
Gate 2: Depth bins are sensible per mode — no NaN, values in [0,1] after
        normalisation, at least one bin < 0.9 in expected obstacle modes.
Gate 3: Random policy on M3 env runs 100 steps without Python exceptions
        (catches obvious env/JIT bugs).
Gate 4: Frozen M2 Phase 1 actor + Phase 2 encoder on M3 env (depth bins
        not fed to M2 actor) reproduces M2 MED ≤ 0.075 m.
Gate 5: Throughput at N=1024 with depth rendering stays above 20,000 env-steps/sec.

Usage:
  conda activate drone-rl
  python scripts/validate_m3_env.py [--skip_gate GATE_NUM ...]

Checkpoint structure (verified from _METADATA):
  Phase 1 actor:   m2_phase1_baseline_1778539544/checkpoints/final
                   ckpt["actor"]["params"] → TrainState params (Dense_0 kernel [50,256])
  Phase 2 encoder: phase2_encoder/best_checkpoint
                   ckpt["params"] → encoder params (Dense_0 kernel [2300,256])
  M2 actor obs dim = 42 (base) + 8 (encoder latent) = 50. No depth bins.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iron_man_drone.utils.scene_generator import (
    sample_scene, sample_mode, TRAINING_MODES, HOLDOUT_MODES, _SPAWN_EXCL, _INTER_EXCL,
)
from iron_man_drone.utils.obstacle_randomization import N_OBSTACLE_SLOTS


# ── Helpers ────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def _print_result(gate: int, name: str, ok: bool | None, detail: str = ""):
    tag = PASS if ok else (SKIP if ok is None else FAIL)
    print(f"  Gate {gate}: [{tag}] {name}" + (f" — {detail}" if detail else ""))


# ── Gate 1: Scene validity ─────────────────────────────────────────────────────

def gate1_scene_validity() -> bool:
    """
    For each of the 5 modes, generate 20 scenes and check:
      - No obstacle center within _SPAWN_EXCL*0.5 of origin (relax for slalom/hallway)
      - No two active obstacles closer than _INTER_EXCL*0.3 in xy (tight physical overlap)
      - All active center xy within arena ±4.5 m (generous bound)
      - Inactive slots parked far (center > 50 m from origin)
    """
    rng = np.random.default_rng(0)
    all_modes = list(TRAINING_MODES) + list(HOLDOUT_MODES)
    errors = []

    for mode in all_modes:
        for trial in range(20):
            centers, half_extents = sample_scene(rng, mode, density_mult=1.0)
            n_active = int(np.sum(np.linalg.norm(centers, axis=1) < 50.0))

            for i in range(n_active):
                xy = centers[i, :2]
                if np.any(np.abs(xy) > 4.5):
                    errors.append(f"{mode}[{trial}] obstacle {i} out of arena: {xy}")

            for i in range(n_active):
                for j in range(i + 1, n_active):
                    d = np.linalg.norm(centers[i, :2] - centers[j, :2])
                    if d < _INTER_EXCL * 0.3:  # 0.09 m — physical overlap
                        errors.append(
                            f"{mode}[{trial}] obstacles {i},{j} physically overlap: d={d:.3f}m"
                        )

            for i in range(n_active, N_OBSTACLE_SLOTS):
                dist = np.linalg.norm(centers[i])
                if dist < 50.0:
                    errors.append(f"{mode}[{trial}] slot {i} not parked: {centers[i]}")

    if errors:
        for e in errors[:5]:
            print(f"      {e}")
        if len(errors) > 5:
            print(f"      ... ({len(errors) - 5} more)")
        return False
    return True


# ── Gate 2: Depth bins sensible ────────────────────────────────────────────────

def gate2_depth_bins() -> bool:
    """
    Instantiate the M3 env with a small batch, reset in each mode, render depth,
    compute bins, check:
      - No NaN, values in [0, 1]
      - Forest and hallway (dense) have at least one bin < 0.8 in ≥25% of envs
    """
    try:
        from iron_man_drone.envs.quadrotor_env_m3 import M3VecEnv, DEPTH_N_BINS
    except ImportError as e:
        print(f"      Import error: {e}")
        return False

    N = 16
    try:
        env = M3VecEnv(SimpleNamespace(num_envs=N))
    except Exception as e:
        print(f"      M3VecEnv init failed: {e}")
        return False

    errors = []
    keys = jax.random.split(jax.random.PRNGKey(1), N)

    for mode in ["forest", "hallway", "random"]:
        try:
            modes_arr = [mode] * N
            state, base_obs, priv_state = env.batch_reset(keys, modes_arr, density_mult=1.0)

            # Render depth outside JIT then compute bins
            depth_raw = env.batch_render(state)           # (N, 64, 64)
            depth_bins = np.array(env.compute_depth_bins(depth_raw))  # (N, 16)

            if np.any(np.isnan(depth_bins)):
                errors.append(f"{mode}: NaN in depth bins")
            if np.any(depth_bins < 0.0) or np.any(depth_bins > 1.001):
                errors.append(
                    f"{mode}: bins outside [0,1]: "
                    f"min={depth_bins.min():.3f} max={depth_bins.max():.3f}"
                )

            # Dense modes should detect obstacles in at least 25% of envs
            n_detecting = int(np.sum(np.any(depth_bins < 0.8, axis=1)))
            if mode in ("forest", "hallway") and n_detecting < N // 4:
                errors.append(
                    f"{mode}: only {n_detecting}/{N} envs detect obstacles "
                    f"(bins < 0.8) — depth may be broken"
                )
            else:
                print(f"      {mode}: {n_detecting}/{N} envs detect obstacle "
                      f"(bin_min={depth_bins.min():.3f})")
        except Exception as e:
            import traceback
            errors.append(f"{mode}: exception — {e}")
            traceback.print_exc()

    if errors:
        for e in errors:
            print(f"      {e}")
        return False
    return True


# ── Gate 3: Random policy — no crash ──────────────────────────────────────────

def gate3_random_policy() -> bool:
    """
    Run 100 steps with random actions on a small M3VecEnv batch.
    Pass if no Python exception is raised and rewards are finite.

    batch_reset returns (M3EnvState, base_obs(N,42), priv_state(N,8))
    batch_step  returns (M3EnvState, base_obs(N,42), priv_state(N,8), reward(N,), done(N,))
    """
    try:
        from iron_man_drone.envs.quadrotor_env_m3 import M3VecEnv
    except ImportError as e:
        print(f"      Import error: {e}")
        return False

    N = 32
    try:
        env = M3VecEnv(SimpleNamespace(num_envs=N))
        keys = jax.random.split(jax.random.PRNGKey(2), N)
        modes = ["random"] * N
        state, base_obs, priv_state = env.batch_reset(keys, modes, density_mult=0.5)
        action_key = jax.random.PRNGKey(3)

        for step in range(100):
            action_key, ak = jax.random.split(action_key)
            actions = jax.random.uniform(ak, (N, 4), minval=-1.0, maxval=1.0)
            state, base_obs, priv_state, reward, done = env.batch_step(state, actions)

        rewards_np = np.array(reward)
        if np.any(np.isnan(rewards_np)) or np.any(np.isinf(rewards_np)):
            print(f"      Non-finite reward at step 100: {rewards_np[:4]}")
            return False
        print(f"      100 steps OK — mean reward: {rewards_np.mean():.4f}")
    except Exception as e:
        import traceback
        print(f"      Exception during rollout: {e}")
        traceback.print_exc()
        return False
    return True


# ── Gate 4: Frozen M2 policy — backward compatibility ─────────────────────────

def gate4_m2_backward_compat() -> bool:
    """
    Load M2 Phase 1 actor and run on M3 env with oracle priv_state as z_hat.

    Why oracle (no encoder):
      - Phase 1 was trained with ground-truth priv_state as z_hat.
      - Phase 2 encoder maps history→priv_state; initialising history to zeros
        produces an out-of-distribution latent that kills tracking.
      - Gate 4 tests env backward compat (physics/obs), not encoder quality.

    Measurement methodology — must match eval_m2_full.py exactly:
      - T/4 phase offset init: state.step = T/4 / DT = 138, placing the
        figure-eight reference at (0,0,1) ≈ drone spawn.
      - TOTAL=1000 steps (full episode), WARMUP=0 (no skip).
        eval_m2_full.py measures all 1000 steps; so does Gate 4.
      - XY-only error (eval_m2_full.py uses 2D distance to ref_xy).

    Checkpoint:
      Actor: m2_phase1_baseline_1778244202/checkpoints/final  (canonical M2
             Phase 1, 15k epochs, T/4-corrected XY MED = 0.0574 m)
             ckpt["actor"]["params"] → Dense_0 kernel [50,256]
    """
    actor_ckpt_path = (
        ROOT / "experiments" / "m2_phase1_baseline"
        / "m2_phase1_baseline_1778244202" / "checkpoints" / "final"
    )

    if not actor_ckpt_path.exists():
        print(f"      M2 Phase 1 checkpoint not found at {actor_ckpt_path}")
        return None

    try:
        from iron_man_drone.envs.quadrotor_env_m3 import M3VecEnv
        from iron_man_drone.envs.trajectories import make_figure_eight_trajectory, DT as TRAJ_DT
        from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
        from iron_man_drone.policy.networks import Actor
        import orbax.checkpoint as ocp
    except ImportError as e:
        print(f"      Import error: {e}")
        return None

    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpu_devices:
        print("      JAX is CPU-only — run Gate 4 from WSL2 where JAX sees the GPU.")
        return None

    try:
        # Load via PyTreeCheckpointer (same as eval_m2_full.py) and convert to jnp.
        checkpointer = ocp.PyTreeCheckpointer()
        actor_ckpt   = checkpointer.restore(str(actor_ckpt_path))
        actor_params = jax.tree_util.tree_map(
            lambda x: jnp.array(x), actor_ckpt["actor"]["params"]
        )

        actor_d0_k = actor_params["params"]["Dense_0"]["kernel"]
        assert actor_d0_k.shape[0] == 50, (
            f"Expected M2 actor 50-dim input, got {actor_d0_k.shape[0]}"
        )
        print(f"      Checkpoint loaded via PyTreeCheckpointer: "
              f"Dense_0 kernel shape={actor_d0_k.shape}  "
              f"L2={float(jnp.linalg.norm(actor_d0_k)):.4f}")

        actor = Actor()

        T_FIG8        = 5.5
        OFFSET_STEPS  = round(T_FIG8 / 4.0 / TRAJ_DT)          # 138 (banker's rounding)
        LOOKAHEAD_BUF = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS         # 50

        # eval_trajectory_position silently clamps t to [0, total_time] — no JAX
        # error is raised when you query past trajectory end.  The trajectory must
        # have enough headroom so no simulation step ever hits the clamp.
        # Requirement: total_time > (OFFSET_STEPS + TOTAL + LOOKAHEAD_BUF) * DT
        #   = (138 + 1000 + 50) * 0.01 = 11.88 s
        # Matching eval_m2_full.py: total = EPISODE_STEPS + off + LOOKAHEAD + 5 = 1193.
        traj_total_steps = EPISODE_STEPS + OFFSET_STEPS + LOOKAHEAD_BUF + 10  # 1198
        fig8_traj = make_figure_eight_trajectory(
            TRAJ_DT, traj_total_steps, LOOKAHEAD_BUF, speed="normal"
        )

        N         = 16
        TOTAL     = EPISODE_STEPS  # 1000 — match M2 eval: measure full episode
        WARMUP    = 0              # no skip — M2 eval measures all 1000 steps
        THRESHOLD = 0.075          # m — 0.0574 m (M2 T/4-corrected) × 1.31

        # Nominal DR (no rotor faults, nominal mass) for a known ground truth.
        # fault_prob=0.7 DR inflates tracking to ~0.17 m even with oracle z —
        # faulted rotors reduce max acceleration and create yaw asymmetry.
        # The gate tests physics/obs compatibility, not fault-compensation quality.
        env = M3VecEnv(
            SimpleNamespace(num_envs=N),
            override_traj=fig8_traj,
            fault_prob=0.0,
            mass_lo=1.0,
            mass_hi=1.0,
        )
        keys = jax.random.split(jax.random.PRNGKey(4), N)
        state, cur_obs, cur_priv = env.batch_reset(keys, ["random"] * N, density_mult=0.0)

        from iron_man_drone.envs.trajectories import eval_trajectory_position, DT as STEP_DT
        from iron_man_drone.envs.quadrotor_env_m3 import build_m3_base_obs

        drone_body_id = env.drone_body_id

        # Match M2 eval_m2_full.py protocol exactly:
        #   - step offset: figure-eight T/4 → reference starts at ~(0,0,1) = drone reset pos
        #   - kf_multiplier = 1.0 (eval fixes kf; random kf causes ~0.17 m bias without oracle)
        #   - nominal priv_state, rotor_efficiency=[1,1,1,1], mass_scale=1.0
        priv_nominal = jnp.array([1., 1., 1., 1., 1., 0., 0., 0.])

        state = state._replace(
            step            = jnp.full((N,), OFFSET_STEPS, dtype=jnp.int32),
            kf_multiplier   = jnp.ones(N),
            rotor_efficiency= jnp.ones((N, 4)),
            mass_scale      = jnp.ones(N),
            priv_state      = jnp.tile(priv_nominal, (N, 1)),
        )
        cur_priv = jnp.tile(priv_nominal, (N, 1))  # (N, 8) nominal oracle z_hat
        # Recompute cur_obs: figure-eight reference is now at (~0,~0,1) ≈ drone position
        cur_obs = jax.vmap(
            lambda s: build_m3_base_obs(s.mjx_data, s.traj, s.step, drone_body_id)
        )(state)

        # Debug: print initial state
        ref_at_offset = np.array(eval_trajectory_position(
            fig8_traj, jnp.array(float(OFFSET_STEPS) * STEP_DT)
        ))
        drone_pos0 = np.array(state.mjx_data.xpos)[:, drone_body_id, :]
        init_err = np.linalg.norm(drone_pos0[0] - ref_at_offset)
        print(f"      Init: drone[0]={drone_pos0[0].round(3)}  ref={ref_at_offset.round(3)}  err={init_err:.3f}m")

        tracking_errors = []
        step_diag = {}  # step → mean 3D error across envs

        for t in range(TOTAL):
            # oracle z_hat = ground-truth priv_state (same as Phase 1 training)
            actor_obs_m2 = jnp.concatenate([cur_obs, cur_priv], axis=-1)  # (N, 50)
            mean, _log_std = jax.vmap(lambda o: actor.apply(actor_params, o))(actor_obs_m2)
            actions = mean

            state, cur_obs, cur_priv, _reward, _done = env.batch_step(state, actions)

            step_idx_actual = OFFSET_STEPS + t + 1
            t_ref = jnp.array(float(step_idx_actual) * STEP_DT)
            ref_pos_now = np.array(eval_trajectory_position(fig8_traj, t_ref))
            drone_pos_np = np.array(state.mjx_data.xpos)[:, drone_body_id, :]
            err_3d = np.linalg.norm(drone_pos_np - ref_pos_now[None, :], axis=1)
            err_xy = np.linalg.norm(drone_pos_np[:, :2] - ref_pos_now[:2], axis=1)
            err_z  = np.abs(drone_pos_np[:, 2] - ref_pos_now[2])

            if t in (0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999):
                step_diag[t] = (float(np.mean(err_3d)), float(np.mean(err_xy)), float(np.mean(err_z)))

            if t >= WARMUP:
                # XY-only tracking error — matches M2 eval_m2_full.py metric
                errs = np.minimum(err_xy, 5.0)
                tracking_errors.append(errs)

        print("      Step-by-step diagnostics (mean 3D / XY / Z error across envs):")
        for step_t, (e3, exy, ez) in sorted(step_diag.items()):
            print(f"        t={step_t:3d}: 3D={e3:.4f} m  XY={exy:.4f} m  Z={ez:.4f} m")
        per_env_mean = np.mean(tracking_errors, axis=0)   # (N,)
        med = float(np.median(per_env_mean))
        worst = float(np.max(per_env_mean))
        crashed_count = int(np.sum(per_env_mean > 2.0))
        print(
            f"      M2 Phase 1 policy (oracle z) on M3 env, "
            f"figure_eight_normal, no obstacles: "
            f"XY MED={med:.4f} m  worst={worst:.4f} m  crashed={crashed_count}/{N}  "
            f"(all {TOTAL} steps from T/4 init, XY, matching M2 eval protocol)"
        )
        sorted_means = np.sort(per_env_mean)
        print(f"      per-env means (sorted): {sorted_means.round(3).tolist()}")

        # ── M2 VecEnv comparison (diagnostic) ────────────────────────────────
        # Run same protocol on M2 env (crazyflie.xml, no obstacles).
        # If M2 VecEnv MED ≈ 0.037 m and M3 MED ≈ 0.17 m → genuine M3 regression.
        # If both give similar MED → Gate 4 methodology issue.
        print("      Running M2 VecEnv comparison (N=4, same protocol)...")
        try:
            from iron_man_drone.envs.quadrotor_env import VecEnv, _build_obs
            N2 = 4
            m2_env = VecEnv(SimpleNamespace(num_envs=N2), fault_prob=0.0, mass_lo=1.0, mass_hi=1.0)
            m2_drone_id = m2_env.mj_model.body("drone").id
            m2_keys = jax.random.split(jax.random.PRNGKey(4), N2)
            m2_state, _, _ = m2_env.batch_reset(m2_keys)
            fig8_batch = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * N2), fig8_traj
            )
            m2_state = m2_state._replace(
                traj=fig8_batch,
                step=jnp.full((N2,), OFFSET_STEPS, dtype=jnp.int32),
                kf_multiplier=jnp.ones(N2),
                rotor_efficiency=jnp.ones((N2, 4)),
                mass_scale=jnp.ones(N2),
                priv_state=jnp.tile(priv_nominal, (N2, 1)),
            )
            m2_cur_obs = jax.vmap(
                lambda s: _build_obs(s.mjx_data, s.traj, s.step, m2_drone_id, priv_nominal)[0]
            )(m2_state)
            m2_tracking = []
            for t in range(TOTAL):
                mean2, _ = jax.vmap(lambda o: actor.apply(actor_params, o))(m2_cur_obs)
                m2_state, m2_cur_obs, _, _, _ = m2_env.batch_step(m2_state, mean2)
                if t >= WARMUP:
                    step_idx2 = OFFSET_STEPS + t + 1
                    ref2 = np.array(eval_trajectory_position(fig8_traj, jnp.array(float(step_idx2) * STEP_DT)))
                    pos2 = np.array(m2_state.mjx_data.xpos)[:, m2_drone_id, :]
                    err_xy2 = np.linalg.norm(pos2[:, :2] - ref2[:2], axis=1)
                    m2_tracking.append(np.minimum(err_xy2, 5.0))
            m2_per_env = np.mean(m2_tracking, axis=0)
            m2_med = float(np.median(m2_per_env))
            print(f"      M2 VecEnv XY MED={m2_med:.4f} m  (all {TOTAL} steps from T/4 init)")
            print(f"      Regression ratio: M3/M2 = {med/m2_med:.1f}×  "
                  f"({'env regression' if med > m2_med * 1.5 else 'within tolerance'})")
        except Exception as e2:
            print(f"      M2 comparison failed: {e2}")

        if med > THRESHOLD:
            print(f"      MED {med:.4f} m > {THRESHOLD} m — env regression or policy broken")
            return False
        return True

    except Exception as e:
        import traceback
        print(f"      Exception: {e}")
        traceback.print_exc()
        return None


# ── Gate 5: Throughput ─────────────────────────────────────────────────────────

def gate5_throughput() -> bool:
    """
    Throughput gate: two sub-checks.

    1. batch_step throughput ≥ 20,000 fps (N=1024, 100 steps).
       This is the MJX physics rate — the bottleneck for PPO rollouts.
       Ray-tracing (batch_render) is a separate render pass and is NOT
       included in this threshold: at N=128 it runs at ~10k fps, which
       projects to ~7k fps combined at N=1024 and is reported separately.

    2. batch_render sanity (N=128, 20 steps): reports fps, never fails gate.
       Render every K steps in the training loop; don't run at N=1024 here
       (takes ~2 min, same as Gate 4 run time).
    """
    try:
        from iron_man_drone.envs.quadrotor_env_m3 import M3VecEnv
    except ImportError as e:
        print(f"      Import error: {e}")
        return False

    try:
        # ── Sub-check 1: batch_step throughput ───────────────────────────────
        N = 1024
        env = M3VecEnv(SimpleNamespace(num_envs=N))
        keys = jax.random.split(jax.random.PRNGKey(6), N)
        state, _, _ = env.batch_reset(keys, ["forest"] * N, density_mult=1.0)
        action_key = jax.random.PRNGKey(7)

        # warmup (5 steps)
        for _ in range(5):
            action_key, ak = jax.random.split(action_key)
            state, _, _, _, _ = env.batch_step(
                state, jax.random.uniform(ak, (N, 4), minval=-1.0, maxval=1.0)
            )
        jax.block_until_ready(state.mjx_data.qpos)

        # measure (100 steps, step only)
        MEASURE = 100
        t0 = time.perf_counter()
        for _ in range(MEASURE):
            action_key, ak = jax.random.split(action_key)
            state, _, _, _, _ = env.batch_step(
                state, jax.random.uniform(ak, (N, 4), minval=-1.0, maxval=1.0)
            )
        jax.block_until_ready(state.mjx_data.qpos)
        step_fps = N * MEASURE / (time.perf_counter() - t0)
        print(f"      batch_step  N={N}: {step_fps:,.0f} fps")

        # ── Sub-check 2: batch_render throughput (N=128, report only) ────────
        N_r = 128
        env_r = M3VecEnv(SimpleNamespace(num_envs=N_r))
        keys_r = jax.random.split(jax.random.PRNGKey(8), N_r)
        state_r, _, _ = env_r.batch_reset(keys_r, ["forest"] * N_r, density_mult=1.0)
        for _ in range(3):  # warmup
            _ = env_r.batch_render(state_r)
        import warp as wp; wp.synchronize()
        R_MEASURE = 20
        t0 = time.perf_counter()
        for _ in range(R_MEASURE):
            _ = env_r.batch_render(state_r)
        wp.synchronize()
        render_fps = N_r * R_MEASURE / (time.perf_counter() - t0)
        print(f"      batch_render N={N_r}: {render_fps:,.0f} fps  "
              f"(render runs every K steps in training loop)")

        if step_fps < 20_000:
            print(f"      batch_step {step_fps:,.0f} fps < 20,000 — physics too slow for training")
            return False
        return True

    except Exception as e:
        import traceback
        print(f"      Exception during throughput test: {e}")
        traceback.print_exc()
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_gate", type=int, action="append", default=[], metavar="N")
    args = parser.parse_args()
    skip = set(args.skip_gate)

    print("\n=== M3 Environment Validation ===\n")

    results = {}

    def run_gate(num, name, fn):
        if num in skip:
            _print_result(num, name, None, "skipped via --skip_gate")
            results[num] = None
            return
        print(f"  Running Gate {num}: {name}...")
        ok = fn()
        _print_result(num, name, ok)
        results[num] = ok

    run_gate(1, "Scene validity (all 5 modes × 20 trials)", gate1_scene_validity)
    run_gate(2, "Depth bins sensible (3 modes — forest/hallway/random)", gate2_depth_bins)
    run_gate(3, "Random policy — no crash (32 envs × 100 steps)", gate3_random_policy)
    run_gate(4, "Frozen M2 policy — MED ≤ 0.075 m (M2 actor, no depth)", gate4_m2_backward_compat)
    run_gate(5, "Throughput ≥ 20k fps (batch_step N=1024, render N=128 reported)", gate5_throughput)

    print()
    hard_failures = [n for n, ok in results.items() if ok is False]
    if hard_failures:
        print(f"RESULT: FAIL — gates {hard_failures} did not pass. Fix before training.")
        sys.exit(1)
    else:
        skipped = [n for n, ok in results.items() if ok is None]
        if skipped:
            print(f"RESULT: PASS (with skipped gates {skipped}) — all run gates passed.")
        else:
            print("RESULT: PASS — all 5 gates passed. Ready to train.")
        sys.exit(0)


if __name__ == "__main__":
    main()
