"""
Full M1 eval suite — corrected initialization matching SimpleFlight's methodology.

Key corrections vs old _run_med_eval:
  1. Figure-eight (slow/normal/fast): apply traj_t0 = T/4 phase offset so
     the reference starts at (0,0,1) = drone spawn position. Zero cold-start.
     Source: SimpleFlight track.py, confirmed per arXiv 2412.11764.
  2. Pentagram (slow/fast): traj_t0 = 0, no correction (matches SimpleFlight).
  3. Polynomial held-out: first waypoint fixed to (0,0,1). SimpleFlight's
     ChainedPolynomial uses origin=(0,0,1) as the mandatory first waypoint.
  4. Zigzag held-out: same — first waypoint fixed to (0,0,1).

Aggregation: arithmetic mean over full 1000-step episode (matches SimpleFlight).
Dynamics: CPU mujoco.mj_step — confirmed identical results to MJX inline eval.

Usage:
  python scripts/eval_m1_full.py --checkpoint path/to/epoch_013000
  python scripts/eval_m1_full.py  # uses epoch_013000 by default
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import mujoco
import orbax.checkpoint as ocp
import yaml

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "experiments/m1_3_polynomial_fix"
    / "m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000"
)

# M1 thresholds (from notes/M1_hypothesis.md and paper comparison)
THRESHOLDS = {
    "figure_eight_slow":   0.050,
    "figure_eight_normal": 0.056,
    "figure_eight_fast":   0.150,
    "pentagram_slow":      None,   # paper: 0.024m — no hard threshold set in hypothesis
    "pentagram_fast":      None,   # paper: 0.045m
    "polynomial":          None,   # paper: 0.032m
    "zigzag":              None,   # paper: 0.052m
}

# Paper numbers (SimpleFlight 100Hz Crazyflie, Table III arXiv 2412.11764)
PAPER_NUMBERS = {
    "figure_eight_slow":   0.016,
    "figure_eight_normal": 0.028,
    "figure_eight_fast":   0.051,
    "pentagram_slow":      0.024,
    "pentagram_fast":      0.045,
    "polynomial":          0.032,
    "zigzag":              0.052,
}

N_POLY_SEEDS = 3    # number of random polynomial trajectories
N_ZIGZAG_SEEDS = 3  # number of random zigzag trajectories


# ---------------------------------------------------------------------------
# Pure-numpy trajectory evaluation — avoids JAX recompilation in the rollout
# loop. get_reference_window creates a new vmap lambda each call; wrapping 1000
# steps × 9 trajectories in JAX causes 9000 XLA recompilations (and OOM).
# Instead we precompute all reference data as numpy before each rollout.
# ---------------------------------------------------------------------------

def _f8_at_t_np(t_arr, T):
    t = np.asarray(t_arr)
    return np.stack([np.cos(2*np.pi*t/T), np.sin(4*np.pi*t/T)/2, np.ones_like(t)], axis=-1)


def _pentagram_at_t_np(t_arr, speed_mps):
    """Numpy port of _pentagram_at_t."""
    vertex_order = np.array([0, 2, 4, 1, 3])
    angles = 2.0*np.pi*vertex_order/5.0 - np.pi/2.0
    vertices = np.stack([np.cos(angles), np.sin(angles), np.ones(5)], axis=-1)   # (5, 3)
    seg_vecs = np.roll(vertices, -1, axis=0) - vertices                           # (5, 3)
    seg_lens = np.linalg.norm(seg_vecs[:, :2], axis=-1)                           # (5,)
    total_len = seg_lens.sum()
    T = total_len / speed_mps

    t = np.asarray(t_arr, dtype=float)
    t_frac = (t % T) / T
    cum_frac = np.concatenate([[0.0], np.cumsum(seg_lens / total_len)])           # (6,)

    # vectorised over t
    seg_idx = np.searchsorted(cum_frac[1:], t_frac, side="right").clip(0, 4)
    seg_s = cum_frac[seg_idx]
    seg_e = cum_frac[seg_idx + 1]
    alpha = np.clip((t_frac - seg_s) / np.maximum(seg_e - seg_s, 1e-9), 0.0, 1.0)
    return vertices[seg_idx] + alpha[:, None] * seg_vecs[seg_idx]                # (N, 3)


def _poly_at_t_np(traj, t_arr):
    """Numpy Horner evaluation for polynomial trajectory."""
    cum = np.array(traj.cum_times)
    coeffs = np.array(traj.poly_coeffs)   # (MAX_SEGS, 6, 3)
    total_time = float(traj.total_time)
    t = np.clip(np.asarray(t_arr, dtype=float), 0.0, total_time)
    out = np.zeros((len(t), 3))
    for j, tj in enumerate(t):
        idx = max(0, min(int(np.searchsorted(cum, tj, side="right")) - 1, coeffs.shape[0]-1))
        tau = tj - cum[idx]
        c = coeffs[idx]   # (6, 3)
        p = c[5]
        p = c[4] + tau*p; p = c[3] + tau*p; p = c[2] + tau*p
        p = c[1] + tau*p; p = c[0] + tau*p
        out[j] = p
    return out


def _zigzag_at_t_np(traj, t_arr):
    """Numpy linear interpolation for zigzag trajectory."""
    cum = np.array(traj.cum_times)
    wps = np.array(traj.waypoints)        # (MAX_SEGS+1, 3)
    total_time = float(traj.total_time)
    t = np.clip(np.asarray(t_arr, dtype=float), 0.0, total_time)
    out = np.zeros((len(t), 3))
    for j, tj in enumerate(t):
        idx = max(0, min(int(np.searchsorted(cum, tj, side="right")) - 1, wps.shape[0]-2))
        t0, t1 = cum[idx], cum[idx+1]
        alpha = float(np.clip((tj - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
        out[j] = wps[idx] + alpha*(wps[idx+1] - wps[idx])
    return out


# traj_type constants (must match trajectories.py)
_TRAJ_POLY, _TRAJ_ZIGZAG = 0, 1
_TRAJ_F8_SLOW, _TRAJ_F8_NORMAL, _TRAJ_F8_FAST = 2, 3, 4
_TRAJ_PENTA_SLOW, _TRAJ_PENTA_FAST = 5, 6
_F8_PERIODS = {_TRAJ_F8_SLOW: 15.0, _TRAJ_F8_NORMAL: 5.5, _TRAJ_F8_FAST: 3.5}
_PENTA_SPEEDS = {_TRAJ_PENTA_SLOW: 0.5, _TRAJ_PENTA_FAST: 1.0}


def eval_traj_np(traj, t_arr):
    """Evaluate trajectory at array of times; returns (N, 3) numpy array."""
    tt = int(traj.traj_type)
    if tt in _F8_PERIODS:
        return _f8_at_t_np(t_arr, _F8_PERIODS[tt])
    elif tt in _PENTA_SPEEDS:
        return _pentagram_at_t_np(t_arr, _PENTA_SPEEDS[tt])
    elif tt == _TRAJ_POLY:
        return _poly_at_t_np(traj, t_arr)
    else:  # zigzag
        return _zigzag_at_t_np(traj, t_arr)


def precompute_references(traj, offset_steps, episode_steps, lookahead_n, lookahead_dt_steps, dt):
    """
    Returns:
      ref_windows  (episode_steps, lookahead_n, 3) — actor observation window
      ref_pos_err  (episode_steps, 3)              — reference for error measurement (at step si)
    """
    ref_windows = np.zeros((episode_steps, lookahead_n, 3))
    ref_pos_err = np.zeros((episode_steps, 3))

    for si in range(episode_steps):
        t_base = (si + offset_steps) * dt
        for k in range(lookahead_n):
            t_look = t_base + (k + 1) * lookahead_dt_steps * dt
            ref_windows[si, k] = eval_traj_np(traj, np.array([t_look]))[0]
        ref_pos_err[si] = eval_traj_np(traj, np.array([t_base]))[0]

    return ref_windows, ref_pos_err


def load_actor(checkpoint_path):
    config_path = Path(checkpoint_path).parent.parent / "config_frozen.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from iron_man_drone.policy.ppo import PPOConfig, create_train_states
    ppo_cfg = PPOConfig(
        actor_obs_dim=cfg["observation"]["actor_dim"],
        critic_obs_dim=cfg["observation"]["critic_dim"],
        action_dim=cfg["action"]["dim"],
        hidden_dim=cfg["network"]["hidden_dim"],
        num_layers=cfg["network"]["num_layers"],
    )
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(Path(checkpoint_path).resolve()))
    actor_state = actor_state.replace(params=ckpt["actor"]["params"])

    @jax.jit
    def actor_forward(params, obs):
        return actor_state.apply_fn(params, obs)

    # Warm up JIT
    dummy = jnp.zeros((1, cfg["observation"]["actor_dim"]))
    actor_forward(actor_state.params, dummy)
    return actor_state, actor_forward


def run_rollout(actor_state, actor_forward, mj_model, mj_data, drone_body_id,
                eval_traj, offset_steps, label):
    """
    Run one 1000-step CPU mujoco rollout.
    Reference windows are precomputed as numpy before the loop — avoids
    the JAX vmap lambda recompilation trap (9000 XLA compilations otherwise).
    Returns (mean_xy_error, crashed, xy_errs_array).
    """
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT
    from iron_man_drone.control.ctbr_controller import (
        ctbr_to_rotor_speeds, compute_wrench, MASS, GRAVITY, KF,
    )

    # Precompute all reference data as plain numpy — one eval per step, no JAX
    ref_wins, ref_pos_err = precompute_references(
        eval_traj, offset_steps, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT
    )
    # ref_wins:    (1000, 10, 3)
    # ref_pos_err: (1000, 3)  — reference at step si (for error measurement)

    # Reset
    mj_data.qpos[:3] = [0.0, 0.0, 1.0]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    hover_omega = np.sqrt(float(MASS * GRAVITY) / (4.0 * float(KF))) * np.ones(4)
    rotor_speeds = hover_omega.copy()

    positions = []
    ref_positions = []
    crashed = False

    for step_i in range(EPISODE_STEPS):
        pos    = np.array(mj_data.xpos[drone_body_id])
        R_flat = np.array(mj_data.xmat[drone_body_id]).reshape(-1)
        vel    = np.array(mj_data.qvel[:3])

        # Use precomputed reference window (no JAX call here)
        ref_win = ref_wins[step_i]                          # (10, 3)
        e_W = (ref_win - pos[None, :]).reshape(-1)          # (30,)
        actor_obs = jnp.array(np.concatenate([e_W, vel, R_flat])[None])
        mean, _ = actor_forward(actor_state.params, actor_obs)
        action = np.array(mean[0])

        omega_current = np.array(mj_data.qvel[3:6])
        new_rs = np.array(ctbr_to_rotor_speeds(
            jnp.array(action), jnp.array(rotor_speeds),
            jnp.array(omega_current), DT,
        ))
        force_body_j, torque_body_j = compute_wrench(jnp.array(new_rs))
        R_mat = np.array(mj_data.xmat[drone_body_id]).reshape(3, 3)
        force_w  = R_mat @ np.array(force_body_j)
        torque_w = R_mat @ np.array(torque_body_j)

        mj_data.xfrc_applied[:] = 0.0
        mj_data.xfrc_applied[drone_body_id, :3] = force_w
        mj_data.xfrc_applied[drone_body_id, 3:]  = torque_w
        mujoco.mj_step(mj_model, mj_data)
        rotor_speeds = new_rs

        pos_after = np.array(mj_data.xpos[drone_body_id])
        ref_xy = ref_pos_err[step_i, :2]
        positions.append(pos_after[:2].copy())
        ref_positions.append(ref_xy.copy())

        if pos_after[2] < 0.05 or np.linalg.norm(pos_after) > 8.0:
            crashed = True
            for _ in range(EPISODE_STEPS - step_i - 1):
                positions.append(pos_after[:2].copy())
                ref_positions.append(ref_xy.copy())
            break

    positions     = np.array(positions)
    ref_positions = np.array(ref_positions)
    xy_errs = np.linalg.norm(positions - ref_positions, axis=1)
    return float(xy_errs.mean()), crashed, xy_errs


def make_poly_traj_fixed_origin(seed, dt, total_steps, lookahead_steps):
    """
    Polynomial trajectory with first waypoint fixed to (0,0) — matches SimpleFlight,
    where ChainedPolynomial uses origin=(0,0,1) as the mandatory starting point.

    The first segment is re-solved so all 6 quintic BCs are consistent:
      p0=(0,0), v0=0, a0=0  (hover at start, same as training)
      p1, v1, a1 recovered from the randomly-generated segment's end state.
    """
    from iron_man_drone.envs.trajectories import sample_polynomial_trajectory, _solve_quintic_coeffs
    key = jax.random.PRNGKey(seed)
    traj = sample_polynomial_trajectory(key, dt, total_steps, lookahead_steps)

    # Duration of the first segment (cum_times[0]=0, cum_times[1]=T1)
    T1 = float(traj.cum_times[1])
    c = np.array(traj.poly_coeffs[0, :, :2])  # (6, 2) — original first segment xy

    # Recover end BCs from the randomly-generated polynomial (Horner evaluation)
    p1 = c[0] + T1*(c[1] + T1*(c[2] + T1*(c[3] + T1*(c[4] + T1*c[5]))))
    v1 = c[1] + T1*(2*c[2] + T1*(3*c[3] + T1*(4*c[4] + T1*5*c[5])))
    a1 = 2*c[2] + T1*(6*c[3] + T1*(12*c[4] + T1*20*c[5]))

    # Re-solve segment 0 with p0=(0,0), v0=0, a0=0 and original end BCs
    new_c = np.array(_solve_quintic_coeffs(
        jnp.zeros(2), jnp.zeros(2), jnp.zeros(2),
        jnp.array(p1), jnp.array(v1), jnp.array(a1),
        jnp.array(T1, dtype=jnp.float32),
    ))  # (6, 2)

    new_poly = traj.poly_coeffs.at[0, :, :2].set(jnp.array(new_c))
    return traj._replace(poly_coeffs=new_poly)


def make_zigzag_traj_fixed_origin(seed, dt, total_steps, lookahead_steps):
    """Zigzag trajectory with first waypoint fixed to (0,0) — matches SimpleFlight."""
    from iron_man_drone.envs.trajectories import sample_zigzag_trajectory
    key = jax.random.PRNGKey(seed)
    traj = sample_zigzag_trajectory(key, dt, total_steps, lookahead_steps)
    # Fix first waypoint xy to (0,0) — SimpleFlight's RandomZigzag starts from origin
    new_waypoints = traj.waypoints.at[0, :2].set(jnp.zeros(2))
    return traj._replace(waypoints=new_waypoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    run_dir = Path(checkpoint).parent.parent
    output_path = args.output or str(run_dir / "M1_eval_results.md")

    print(f"Checkpoint: {checkpoint}")
    print(f"Output:     {output_path}")

    # Load actor
    print("\nLoading actor and warming JIT...")
    actor_state, actor_forward = load_actor(checkpoint)
    print("Actor ready.")

    # Load MuJoCo
    xml_path = str(REPO_ROOT / "src/iron_man_drone/envs/crazyflie.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)
    drone_body_id = mj_model.body("drone").id

    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, make_pentagram_trajectory,
    )
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT

    lookahead_steps = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS

    results = {}

    # ── Figure-eight (slow / normal / fast) — T/4 phase offset ────────────────
    for speed, T_period in [("slow", 15.0), ("normal", 5.5), ("fast", 3.5)]:
        label = f"figure_eight_{speed}"
        offset_steps = int(round(T_period / 4.0 / DT))
        # Create trajectory with enough total_time for the offset + lookahead
        total_steps_ext = EPISODE_STEPS + offset_steps + lookahead_steps + 5
        eval_traj = make_figure_eight_trajectory(DT, total_steps_ext, lookahead_steps, speed=speed)

        # Verify offset puts reference at ~(0,0,1)
        ref_at_offset = eval_traj_np(eval_traj, np.array([offset_steps * DT]))[0]
        print(f"\n[{label}] T={T_period}s, offset={offset_steps} steps ({offset_steps*DT:.3f}s)")
        print(f"  Reference at offset: ({ref_at_offset[0]:.4f}, {ref_at_offset[1]:.4f}, {ref_at_offset[2]:.4f})")
        print(f"  Initial XY error:    {np.linalg.norm(ref_at_offset[:2]):.4f} m")
        print(f"  Running 1000-step rollout...")

        med, crashed, xy_errs = run_rollout(
            actor_state, actor_forward, mj_model, mj_data, drone_body_id,
            eval_traj, offset_steps, label,
        )
        results[label] = {"med": med, "crashed": crashed, "xy_errs": xy_errs}
        status = "CRASHED" if crashed else "clean"
        threshold = THRESHOLDS[label]
        paper = PAPER_NUMBERS[label]
        pass_fail = ""
        if threshold is not None:
            pass_fail = "PASS" if med < threshold else "FAIL"
        print(f"  MED = {med:.4f} m | {status} | threshold={threshold} → {pass_fail} | paper={paper:.3f} m")

    # ── Pentagram (slow / fast) — traj_t0 = 0, no offset ─────────────────────
    for speed in ["slow", "fast"]:
        label = f"pentagram_{speed}"
        total_steps_ext = EPISODE_STEPS + lookahead_steps + 5
        eval_traj = make_pentagram_trajectory(DT, total_steps_ext, lookahead_steps, speed=speed)

        ref_at_0 = eval_traj_np(eval_traj, np.array([0.0]))[0]
        print(f"\n[{label}] offset=0 steps")
        print(f"  Reference at t=0: ({ref_at_0[0]:.4f}, {ref_at_0[1]:.4f}, {ref_at_0[2]:.4f})")
        print(f"  Initial XY error: {np.linalg.norm(ref_at_0[:2]):.4f} m (cold-start, same as SimpleFlight)")
        print(f"  Running 1000-step rollout...")

        med, crashed, xy_errs = run_rollout(
            actor_state, actor_forward, mj_model, mj_data, drone_body_id,
            eval_traj, 0, label,
        )
        results[label] = {"med": med, "crashed": crashed, "xy_errs": xy_errs}
        status = "CRASHED" if crashed else "clean"
        paper = PAPER_NUMBERS[label]
        print(f"  MED = {med:.4f} m | {status} | paper={paper:.3f} m")

    # ── Polynomial held-out — first waypoint fixed to origin ──────────────────
    print(f"\n[polynomial] offset=0, first waypoint fixed to (0,0,1)")
    poly_meds = []
    poly_crashed = 0
    for seed in range(42, 42 + N_POLY_SEEDS):
        eval_traj = make_poly_traj_fixed_origin(seed, DT, EPISODE_STEPS + lookahead_steps + 5, lookahead_steps)
        ref_at_0 = eval_traj_np(eval_traj, np.array([0.0]))[0]
        print(f"  seed={seed}: ref at t=0 = ({ref_at_0[0]:.4f}, {ref_at_0[1]:.4f}) initial_err={np.linalg.norm(ref_at_0[:2]):.4f}m")
        med, crashed, xy_errs = run_rollout(
            actor_state, actor_forward, mj_model, mj_data, drone_body_id,
            eval_traj, 0, f"polynomial_seed{seed}",
        )
        poly_meds.append(med)
        poly_crashed += int(crashed)
        print(f"    MED = {med:.4f} m | {'CRASHED' if crashed else 'clean'}")
    poly_mean = float(np.mean(poly_meds))
    poly_std  = float(np.std(poly_meds))
    results["polynomial"] = {"med": poly_mean, "med_std": poly_std, "crashed": poly_crashed > 0, "all_meds": poly_meds}
    print(f"  Polynomial mean MED = {poly_mean:.4f} ± {poly_std:.4f} m | crashes={poly_crashed}/{N_POLY_SEEDS} | paper={PAPER_NUMBERS['polynomial']:.3f} m")

    # ── Zigzag held-out — first waypoint fixed to origin ──────────────────────
    print(f"\n[zigzag] offset=0, first waypoint fixed to (0,0,1)")
    zigzag_meds = []
    zigzag_crashed = 0
    for seed in range(42, 42 + N_ZIGZAG_SEEDS):
        eval_traj = make_zigzag_traj_fixed_origin(seed, DT, EPISODE_STEPS + lookahead_steps + 5, lookahead_steps)
        ref_at_0 = eval_traj_np(eval_traj, np.array([0.0]))[0]
        print(f"  seed={seed}: ref at t=0 = ({ref_at_0[0]:.4f}, {ref_at_0[1]:.4f}) initial_err={np.linalg.norm(ref_at_0[:2]):.4f}m")
        med, crashed, xy_errs = run_rollout(
            actor_state, actor_forward, mj_model, mj_data, drone_body_id,
            eval_traj, 0, f"zigzag_seed{seed}",
        )
        zigzag_meds.append(med)
        zigzag_crashed += int(crashed)
        print(f"    MED = {med:.4f} m | {'CRASHED' if crashed else 'clean'}")
    zigzag_mean = float(np.mean(zigzag_meds))
    zigzag_std  = float(np.std(zigzag_meds))
    results["zigzag"] = {"med": zigzag_mean, "med_std": zigzag_std, "crashed": zigzag_crashed > 0, "all_meds": zigzag_meds}
    print(f"  Zigzag mean MED = {zigzag_mean:.4f} ± {zigzag_std:.4f} m | crashes={zigzag_crashed}/{N_ZIGZAG_SEEDS} | paper={PAPER_NUMBERS['zigzag']:.3f} m")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  FULL EVAL SUMMARY — M1.3 epoch_013000 (corrected initialization)")
    print("="*72)
    print(f"  {'Trajectory':<26} {'MED (m)':>10}  {'Threshold':>10}  {'Pass/Fail':>10}  {'Paper':>8}")
    print("  " + "-"*68)

    all_pass = True
    for key in ["figure_eight_slow", "figure_eight_normal", "figure_eight_fast",
                "pentagram_slow", "pentagram_fast", "polynomial", "zigzag"]:
        r = results[key]
        med = r["med"]
        crashed = r["crashed"]
        thr = THRESHOLDS[key]
        paper = PAPER_NUMBERS[key]
        if thr is not None:
            pf = "PASS" if med < thr and not crashed else "FAIL"
            if pf == "FAIL":
                all_pass = False
            thr_str = f"{thr:.3f} m"
        else:
            pf = "CRASH" if crashed else "—"
            thr_str = "—"
        std_str = f" ±{r['med_std']:.4f}" if "med_std" in r else ""
        crashed_str = " (CRASH)" if crashed else ""
        print(f"  {key:<26} {med:>8.4f} m{std_str:<10} {thr_str:>10}  {pf:>10}  {paper:>6.3f} m{crashed_str}")

    print("="*72)
    if all_pass:
        print("  OVERALL: PASS — all thresholded trajectories within target")
    else:
        print("  OVERALL: FAIL — one or more thresholded trajectories above target")
    print("="*72)

    # ── Write results markdown ────────────────────────────────────────────────
    write_results_md(output_path, results, checkpoint)
    print(f"\nResults written to: {output_path}")


def write_results_md(output_path, results, checkpoint):
    from pathlib import Path
    import datetime

    checkpoint_name = Path(checkpoint).name

    # M1 baseline reference numbers (from training diagnostics — old broken eval)
    M1_BASELINE_OLD_EVAL = {
        "figure_eight_normal": 0.066,  # inline eval mean with cold-start
    }

    lines = [
        f"# M1.3 Full Eval Results — {checkpoint_name}",
        f"",
        f"**Date:** {datetime.date.today().isoformat()}  ",
        f"**Checkpoint:** `{checkpoint}`  ",
        f"**Eval methodology:** Corrected initialization matching SimpleFlight (arXiv 2412.11764).  ",
        f"Figure-eight: T/4 phase offset applied. Pentagram/poly/zigzag: traj_t0=0, poly/zigzag first waypoint fixed to (0,0,1).",
        f"",
        f"---",
        f"",
        f"## Results Table",
        f"",
        f"| Trajectory | M1.3 MED (corrected eval) | M1 Threshold | Pass/Fail | Paper (SimpleFlight 100Hz) |",
        f"|---|---|---|---|---|",
    ]

    for key in ["figure_eight_slow", "figure_eight_normal", "figure_eight_fast",
                "pentagram_slow", "pentagram_fast", "polynomial", "zigzag"]:
        r = results[key]
        med = r["med"]
        std_str = f" ± {r['med_std']:.4f}" if "med_std" in r else ""
        crashed = r["crashed"]
        thr = THRESHOLDS[key]
        paper = PAPER_NUMBERS[key]

        if thr is not None:
            pf = "**PASS**" if med < thr and not crashed else "**FAIL**"
            thr_str = f"{thr:.3f} m"
        else:
            pf = "CRASH" if crashed else "—"
            thr_str = "—"

        crashed_str = " *(CRASH)*" if crashed else ""
        lines.append(f"| {key} | {med:.4f}{std_str} m{crashed_str} | {thr_str} | {pf} | {paper:.3f} m |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Comparison: Old Eval vs Corrected Eval vs Paper",
        f"",
        f"| Metric | Old eval (cold-start) | Corrected eval (T/4 offset) | Paper target |",
        f"|---|---|---|---|",
        f"| figure_eight_normal MED | ~0.069 m (mean w/ 1m cold-start) | {results['figure_eight_normal']['med']:.4f} m | 0.028 m |",
        f"| figure_eight_normal steady-state | ~0.025 m | — | — |",
        f"",
        f"**Root cause of old eval gap:** Our `_run_med_eval` initialized the trajectory at t=0,",
        f"placing the reference at (1,0,1) while the drone spawned at (0,0,1). SimpleFlight uses",
        f"`traj_t0 = T/4`, placing the reference at (0,0,1) = drone spawn → zero initial error.",
        f"Confirmed from SimpleFlight `track.py`. Corrected eval applies the same T/4 offset.",
        f"",
        f"---",
        f"",
        f"## M1 Ship Decision",
        f"",
    ]

    all_pass = all(
        results[k]["med"] < THRESHOLDS[k] and not results[k]["crashed"]
        for k in ["figure_eight_slow", "figure_eight_normal", "figure_eight_fast"]
        if THRESHOLDS[k] is not None
    )

    if all_pass:
        lines += [
            f"**VERDICT: M1 PASSES.** All thresholded trajectories below target.",
            f"",
            f"Next steps:",
            f"1. `git tag m1-baseline` on this commit",
            f"2. Begin M2 planning",
        ]
    else:
        failed = [
            k for k in ["figure_eight_slow", "figure_eight_normal", "figure_eight_fast"]
            if THRESHOLDS[k] is not None and (results[k]["med"] >= THRESHOLDS[k] or results[k]["crashed"])
        ]
        lines += [
            f"**VERDICT: M1 FAILS on:** {', '.join(failed)}",
            f"",
            f"Gap analysis required before M1.4 planning.",
        ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
