"""
Trajectory diagnosis on best checkpoint (epoch_006000).
Plots flown vs reference figure-eight, computes error breakdown:
  - Lag: cross-correlation of position vs shifted reference
  - Overshoot: error peaks at curve apex vs straights
  - Noise: high-frequency component of tracking error
  - Offset: mean bias in x/y

Usage:
  python scripts/diagnose_trajectory.py --checkpoint experiments/m1_baseline/phase2_validation/checkpoints/epoch_006000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "experiments/m1_baseline/diagnosis"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint — provide full target structure so orbax doesn't hang
    # inferring types from metadata.
    import orbax.checkpoint as ocp
    import yaml
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    cfg_path = REPO_ROOT / "experiments/m1_baseline/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = PPOConfig(
        actor_obs_dim=cfg["observation"]["actor_dim"],
        critic_obs_dim=cfg["observation"]["critic_dim"],
        action_dim=cfg["action"]["dim"],
        hidden_dim=cfg["network"]["hidden_dim"],
        num_layers=cfg["network"]["num_layers"],
    )

    key = jax.random.PRNGKey(0)
    _, _, actor_state_init, critic_state_init = create_train_states(key, ppo_cfg)
    target = {"actor": actor_state_init, "critic": critic_state_init}

    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_path = str(Path(args.checkpoint).expanduser().resolve())
    restored = checkpointer.restore(ckpt_path, item=target)
    actor_state = restored["actor"]
    print(f"Checkpoint loaded: {args.checkpoint}")

    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, DT, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, _build_obs,
    )
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, get_reference_pos, eval_trajectory_position,
    )

    class EvalCfg:
        num_envs = 1

    env = VecEnv(EvalCfg())
    drone_id = env.mj_model.body("drone").id
    ls = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    eval_traj = make_figure_eight_trajectory(DT, EPISODE_STEPS, ls, speed="normal")

    key = jax.random.PRNGKey(42)
    eval_reset = jax.jit(env._reset_fn)
    eval_step  = jax.jit(env._step_fn)

    # Run deterministic episode
    key, rk = jax.random.split(key)
    state, a_obs, _ = eval_reset(rk, jnp.ones(()))
    state = state._replace(traj=eval_traj)
    a_obs, _ = _build_obs(state.mjx_data, eval_traj, state.step, drone_id)

    positions, refs, errors, actions_log = [], [], [], []

    for si in range(EPISODE_STEPS):
        mean, _ = actor_state.apply_fn(actor_state.params, a_obs[None])
        action = mean[0]
        actions_log.append(np.array(action))

        state, a_obs, _, _, done = eval_step(state, action, jnp.ones(()))
        state = state._replace(traj=eval_traj)

        pos = np.array(state.mjx_data.xpos[drone_id])        # (3,)
        ref = np.array(get_reference_pos(eval_traj, jnp.int32(si)))  # (3,)
        positions.append(pos)
        refs.append(ref)
        errors.append(pos[:2] - ref[:2])                     # xy error vector

        if bool(done):
            print(f"  Episode terminated at step {si}")
            break

    positions = np.array(positions)   # (T, 3)
    refs      = np.array(refs)        # (T, 3)
    errors    = np.array(errors)      # (T, 2)
    actions_log = np.array(actions_log)
    T = len(positions)
    t_axis = np.arange(T) * DT

    xy_err_mag = np.linalg.norm(errors, axis=1)  # (T,)
    med = xy_err_mag.mean()
    print(f"\nMED: {med:.4f} m")

    # ── Error breakdown ──────────────────────────────────────────────────────

    # 1. Offset (mean bias)
    mean_offset = errors.mean(axis=0)
    print(f"\n[1] Offset (mean xy bias): x={mean_offset[0]:+.4f} m, y={mean_offset[1]:+.4f} m")
    print(f"    Offset magnitude: {np.linalg.norm(mean_offset):.4f} m  "
          f"({'SIGNIFICANT — likely a bug' if np.linalg.norm(mean_offset) > 0.02 else 'small, not a constant bug'})")

    # 2. Lag — cross-correlate drone x-position with reference x-position
    drone_x = positions[:, 0] - positions[:, 0].mean()
    ref_x   = refs[:, 0] - refs[:, 0].mean()
    corr = np.correlate(drone_x, ref_x, mode='full')
    lags = np.arange(-(T-1), T)
    best_lag = lags[np.argmax(corr)]
    best_lag_s = best_lag * DT
    print(f"\n[2] Lag (cross-corr peak): {best_lag} steps = {best_lag_s:.3f} s")
    print(f"    {'SIGNIFICANT lag — controller too slow' if abs(best_lag) > 5 else 'small lag, not the main issue'}")

    # 3. Noise — std of error after subtracting a 10-step moving average
    from numpy.lib.stride_tricks import sliding_window_view
    win = 10
    if T > win:
        smooth = np.array([errors[max(0,i-win//2):min(T,i+win//2)].mean(axis=0)
                           for i in range(T)])
        residual = errors - smooth
        noise_rms = np.sqrt((residual**2).sum(axis=1)).mean()
        print(f"\n[3] Noise (RMS of high-freq residual): {noise_rms:.4f} m")
        print(f"    Noise / MED ratio: {noise_rms/med*100:.1f}%  "
              f"({'DOMINANT — jittery flight' if noise_rms/med > 0.4 else 'not dominant'})")
    else:
        noise_rms = 0.0

    # 4. Overshoot — compare error at curve apices vs straight sections
    # Figure-eight apices are at t = 0, T/2 (x=+/-1, y=0)
    # Straights are mid-half-period
    T_period_steps = int(5.5 / DT)  # normal speed period = 5.5s
    # Curve apices: every T_period_steps/2 steps
    apex_indices   = [i for i in range(T) if (i % (T_period_steps // 2)) < 5]
    mid_indices    = [i for i in range(T) if (T_period_steps // 4 - 5 < (i % (T_period_steps // 2)) < T_period_steps // 4 + 5)]
    if apex_indices and mid_indices:
        apex_err = xy_err_mag[apex_indices].mean()
        mid_err  = xy_err_mag[mid_indices].mean()
        print(f"\n[4] Overshoot (apex vs mid-arc error):")
        print(f"    Apex (tight turn) error:  {apex_err:.4f} m")
        print(f"    Mid-arc (straight) error: {mid_err:.4f} m")
        print(f"    Ratio: {apex_err/mid_err:.2f}×  "
              f"({'OVERSHOOT on curves' if apex_err/mid_err > 1.5 else 'uniform — not overshoot-dominated'})")

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"M1 Trajectory Diagnosis — epoch_006000  (MED={med:.4f} m)", fontsize=13)

        # Plot 1: XY trajectory overlay
        ax = axes[0, 0]
        ax.plot(refs[:, 0],      refs[:, 1],      "b-",  lw=1.5, alpha=0.6, label="Reference")
        ax.plot(positions[:, 0], positions[:, 1], "r-",  lw=1.0, alpha=0.8, label="Drone")
        ax.plot(positions[0, 0], positions[0, 1], "go",  ms=8,  label="Start")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("XY Trajectory"); ax.legend(); ax.set_aspect("equal"); ax.grid(True)

        # Plot 2: Error magnitude over time
        ax = axes[0, 1]
        ax.plot(t_axis, xy_err_mag, "r-", lw=0.8, label="‖e_xy‖")
        ax.axhline(med, color="k", ls="--", lw=1, label=f"MED={med:.3f}m")
        ax.axhline(0.056, color="g", ls="--", lw=1, label="Threshold 0.056m")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("XY error (m)")
        ax.set_title("Tracking Error over Time"); ax.legend(); ax.grid(True)

        # Plot 3: Error vectors coloured by time (to see lag/offset pattern)
        ax = axes[1, 0]
        sc = ax.scatter(errors[:, 0], errors[:, 1],
                        c=t_axis, cmap="viridis", s=2, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Time (s)")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.plot(*mean_offset, "r*", ms=12, label=f"Mean offset ({mean_offset[0]:+.3f}, {mean_offset[1]:+.3f})")
        ax.set_xlabel("x error (m)"); ax.set_ylabel("y error (m)")
        ax.set_title("Error Vectors (coloured by time)"); ax.legend(); ax.grid(True)

        # Plot 4: Z altitude
        ax = axes[1, 1]
        ax.plot(t_axis, positions[:, 2], "b-", lw=1, label="Drone z")
        ax.plot(t_axis, refs[:, 2],      "k--", lw=1, label="Reference z=1m")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("z (m)")
        ax.set_title("Altitude"); ax.legend(); ax.grid(True)

        plt.tight_layout()
        out_path = out_dir / "trajectory_diagnosis.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved: {out_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available — skipping plots. pip install matplotlib")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ERROR BREAKDOWN SUMMARY")
    print("="*60)
    components = {
        "Offset (constant bias)": np.linalg.norm(mean_offset),
        "Noise (high-freq jitter)": noise_rms,
    }
    for name, val in components.items():
        pct = val / med * 100
        print(f"  {name:<30}: {val:.4f} m  ({pct:.0f}% of MED)")
    print(f"  Lag: {abs(best_lag)} steps ({best_lag_s:.3f} s)")
    print("="*60)

    # Save raw data for further analysis
    np.save(out_dir / "positions.npy", positions)
    np.save(out_dir / "refs.npy", refs)
    np.save(out_dir / "errors.npy", errors)
    np.save(out_dir / "actions.npy", actions_log)
    print(f"Raw arrays saved to {out_dir}/")


if __name__ == "__main__":
    main()
