"""
Check 3: Plot error vs time for first 200 steps with current eval init.
Drone starts at (0,0,1), figure_eight_normal reference starts at (1,0,1).
This visualizes the 1m cold-start acquisition artifact.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import mujoco
import orbax.checkpoint as ocp
import yaml

REPO_ROOT = Path(__file__).parent.parent
CHECKPOINT = REPO_ROOT / "experiments/m1_3_polynomial_fix/m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000"
OUTPUT_PLOT = REPO_ROOT / "notes/m1_3_acq_error.png"
N_STEPS = 200


def main():
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    config_path = CHECKPOINT.parent.parent / "config_frozen.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = PPOConfig(
        actor_obs_dim=cfg["observation"]["actor_dim"],
        critic_obs_dim=cfg["observation"]["critic_dim"],
        action_dim=cfg["action"]["dim"],
        hidden_dim=cfg["network"]["hidden_dim"],
        num_layers=cfg["network"]["num_layers"],
    )

    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(CHECKPOINT.resolve()))
    actor_state = actor_state.replace(params=ckpt["actor"]["params"])

    @jax.jit
    def actor_forward(params, obs):
        return actor_state.apply_fn(params, obs)

    dummy = jnp.zeros((1, 42))
    actor_forward(actor_state.params, dummy)

    xml_path = str(REPO_ROOT / "src/iron_man_drone/envs/crazyflie.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    drone_body_id = mj_model.body("drone").id

    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, get_reference_window, get_reference_pos,
    )
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT
    from iron_man_drone.control.ctbr_controller import (
        ctbr_to_rotor_speeds, compute_wrench, MASS, GRAVITY, KF,
    )

    eval_traj = make_figure_eight_trajectory(
        DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal"
    )

    # Reference position at t=0 (trajectory start)
    ref_t0 = np.array(get_reference_pos(eval_traj, jnp.int32(0)))
    print(f"Trajectory start (t=0): ({ref_t0[0]:.3f}, {ref_t0[1]:.3f}, {ref_t0[2]:.3f})")
    print(f"Drone init position:    (0.000, 0.000, 1.000)")
    print(f"Initial XY offset:      {np.linalg.norm(ref_t0[:2]):.3f} m")

    # SimpleFlight uses traj_t0 = T/4 = 5.5/4 = 1.375s
    T_f8 = 5.5
    traj_t0_steps = int(round((T_f8 / 4.0) / DT))
    ref_tquarter = np.array(get_reference_pos(eval_traj, jnp.int32(traj_t0_steps)))
    print(f"\nSimpleFlight uses traj_t0 = T/4 = {T_f8/4:.3f}s = step {traj_t0_steps}")
    print(f"Reference at t=T/4:     ({ref_tquarter[0]:.3f}, {ref_tquarter[1]:.3f}, {ref_tquarter[2]:.3f})")
    print(f"Initial XY offset at T/4: {np.linalg.norm(ref_tquarter[:2]):.4f} m")

    # Initialize drone at origin (current eval)
    mj_data.qpos[:3] = [0.0, 0.0, 1.0]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    hover_omega = np.sqrt(float(MASS * GRAVITY) / (4.0 * float(KF))) * np.ones(4)
    rotor_speeds = hover_omega.copy()

    positions = []
    ref_positions = []
    errors_3d = []
    errors_xy = []

    print(f"\nRunning {N_STEPS}-step rollout ...")
    for step_i in range(N_STEPS):
        pos    = np.array(mj_data.xpos[drone_body_id])
        R_flat = np.array(mj_data.xmat[drone_body_id]).reshape(-1)
        vel    = np.array(mj_data.qvel[:3])

        ref_win = np.array(
            get_reference_window(eval_traj, jnp.int32(step_i), LOOKAHEAD_N, LOOKAHEAD_DT_STEPS)
        )
        e_W = (ref_win - pos[None, :]).reshape(-1)
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
        ref_pos   = np.array(get_reference_pos(eval_traj, jnp.int32(step_i + 1)))
        positions.append(pos_after.copy())
        ref_positions.append(ref_pos.copy())
        errors_3d.append(float(np.linalg.norm(pos_after - ref_pos)))
        errors_xy.append(float(np.linalg.norm(pos_after[:2] - ref_pos[:2])))

    positions     = np.array(positions)
    ref_positions = np.array(ref_positions)
    errors_xy     = np.array(errors_xy)
    t_axis        = np.arange(N_STEPS) * DT

    # Key stats
    print(f"\nFirst 10 steps XY error: {errors_xy[:10]}")
    print(f"Step 50  XY error: {errors_xy[49]:.4f} m")
    print(f"Step 100 XY error: {errors_xy[99]:.4f} m")
    print(f"Step 150 XY error: {errors_xy[149]:.4f} m")
    print(f"Step 199 XY error: {errors_xy[199]:.4f} m")
    acq_end = np.argmax(errors_xy < 0.05)
    print(f"First step below 0.05m: step {acq_end} (t={acq_end*DT:.2f}s)")
    print(f"Mean over all {N_STEPS} steps: {errors_xy.mean():.4f} m")
    print(f"Mean over steps 100-200: {errors_xy[100:].mean():.4f} m")

    # SimpleFlight comparison: if they use T/4 phase offset, what would our error be?
    # Compute error vs the T/4-shifted reference
    shifted_errors_xy = []
    for i in range(N_STEPS):
        ref_shifted = np.array(get_reference_pos(eval_traj, jnp.int32(i + 1 + traj_t0_steps)))
        shifted_errors_xy.append(float(np.linalg.norm(positions[i, :2] - ref_shifted[:2])))
    shifted_errors_xy = np.array(shifted_errors_xy)
    print(f"\nWith T/4 phase shift (SimpleFlight convention):")
    print(f"  Initial error at step 0: {shifted_errors_xy[0]:.4f} m (drone at origin, ref also near origin)")
    print(f"  Mean over all {N_STEPS} steps: {shifted_errors_xy.mean():.4f} m")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("M1.3 epoch_013000 — Acquisition error (first 200 steps)", fontsize=13)

        # Panel 1: XY error vs time, first 200 steps
        ax = axes[0]
        ax.plot(t_axis, errors_xy, "b-", lw=1.5, label="XY error (current eval: drone@origin)")
        ax.axhline(0.05, color="orange", ls="--", lw=1.2, label="0.05m threshold")
        ax.axhline(0.056, color="purple", ls="--", lw=1.2, label="M1 target (0.056m)")
        ax.axhline(errors_xy[100:].mean(), color="green", ls="-.", lw=1.2,
                   label=f"Mean steps 100-200 ({errors_xy[100:].mean():.3f}m)")

        if acq_end > 0:
            ax.axvline(acq_end * DT, color="red", ls=":", lw=1, label=f"First <0.05m (t={acq_end*DT:.1f}s)")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("XY error (m)")
        ax.set_title("Current eval: drone at (0,0,1), ref starts at (1,0,1)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(1.2, errors_xy.max() * 1.1))

        # Panel 2: Trajectory phase comparison
        ax2 = axes[1]
        ax2.plot(t_axis, errors_xy, "b-", lw=1.5, label="Current eval (t=0 phase, 1m offset)")
        ax2.plot(t_axis, shifted_errors_xy, "g-", lw=1.5, label="SimpleFlight convention (T/4 phase, 0m offset)")
        ax2.axhline(0.028, color="red", ls="--", lw=1.2, label="Paper target (0.028m)")
        ax2.axhline(0.056, color="purple", ls="--", lw=1.2, label="M1 target (0.056m)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("XY error (m)")
        ax2.set_title("Phase offset comparison\n(current t=0 vs SimpleFlight T/4 phase)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(1.2, errors_xy.max() * 1.1))

        plt.tight_layout()
        OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(OUTPUT_PLOT), dpi=130)
        print(f"\nPlot saved: {OUTPUT_PLOT}")

    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
