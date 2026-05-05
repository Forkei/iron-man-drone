"""
Diagnostic 1+2: Apex-vs-straight error decomposition + flight path visualization.
Checkpoint: epoch_013000 (best inline MED during M1.3 run 2).

Uses CPU mujoco.mj_step — no MJX JIT compilation required.
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
OUTPUT_PLOT = REPO_ROOT / "notes/m1_3_flight_path.png"


def main():
    # ── Load actor ────────────────────────────────────────────────────────────
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
    print("Actor loaded from epoch_013000.")

    @jax.jit
    def actor_forward(params, obs):
        return actor_state.apply_fn(params, obs)

    # Warm up JIT (small MLP, fast)
    dummy = jnp.zeros((1, 42))
    actor_forward(actor_state.params, dummy)
    print("Actor JIT warm-up done.")

    # ── Load MuJoCo model ─────────────────────────────────────────────────────
    xml_path = str(REPO_ROOT / "src/iron_man_drone/envs/crazyflie.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    drone_body_id = mj_model.body("drone").id
    print(f"MuJoCo model loaded. drone_body_id={drone_body_id}")

    # ── Trajectory ───────────────────────────────────────────────────────────
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

    # ── Precompute figure_eight_normal curvature at each timestep ─────────────
    # x(t) = cos(ω*t), y(t) = sin(2ω*t)/2, ω = 2π/T
    T_f8 = 5.5
    omega = 2.0 * np.pi / T_f8
    t_vals = np.arange(EPISODE_STEPS) * DT

    s = np.sin(omega * t_vals)
    c = np.cos(omega * t_vals)
    # x' = -ω*s,  y' = ω*(1-2s²)  [derivative of sin(2ωt)/2 = ω*cos(2ωt) = ω*(1-2s²)]
    vx_ref = -omega * s
    vy_ref = omega * (1.0 - 2.0 * s**2)
    # x'' = -ω²*c, y'' = -4ω²*s*c  [d/dt of ω*(1-2s²) = -4ω²*sin*cos]
    ax_ref = -omega**2 * c
    ay_ref = -4.0 * omega**2 * s * c
    speed_sq = vx_ref**2 + vy_ref**2
    kappa_ref = np.abs(vx_ref * ay_ref - vy_ref * ax_ref) / np.maximum(speed_sq**1.5, 1e-10)

    # ── Initialize simulation ─────────────────────────────────────────────────
    mj_data.qpos[:3] = [0.0, 0.0, 1.0]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # w,x,y,z upright quaternion
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    hover_omega = np.sqrt(float(MASS * GRAVITY) / (4.0 * float(KF))) * np.ones(4)
    rotor_speeds = hover_omega.copy()

    # ── Rollout ───────────────────────────────────────────────────────────────
    print(f"\nRunning {EPISODE_STEPS}-step rollout on figure_eight_normal ...")
    positions = []
    ref_positions = []
    kappa_steps = []
    crashed = False

    for step_i in range(EPISODE_STEPS):
        pos     = np.array(mj_data.xpos[drone_body_id])
        R_flat  = np.array(mj_data.xmat[drone_body_id]).reshape(-1)
        vel     = np.array(mj_data.qvel[:3])

        ref_win = np.array(
            get_reference_window(eval_traj, jnp.int32(step_i), LOOKAHEAD_N, LOOKAHEAD_DT_STEPS)
        )
        e_W = (ref_win - pos[None, :]).reshape(-1)

        actor_obs = jnp.array(np.concatenate([e_W, vel, R_flat])[None])
        mean, _   = actor_forward(actor_state.params, actor_obs)
        action    = np.array(mean[0])

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
        kappa_steps.append(kappa_ref[step_i])

        if step_i % 100 == 0:
            err = np.linalg.norm(pos_after[:2] - ref_pos[:2])
            print(f"  step {step_i:4d}: pos=({pos_after[0]:.3f},{pos_after[1]:.3f},{pos_after[2]:.3f})"
                  f"  ref=({ref_pos[0]:.3f},{ref_pos[1]:.3f})  xy_err={err:.4f}m")

        if pos_after[2] < 0.05 or np.linalg.norm(pos_after) > 8.0:
            print(f"  >> Crash/diverge at step {step_i}")
            crashed = True
            break

    # ── Apex / straight decomposition ─────────────────────────────────────────
    positions     = np.array(positions)
    ref_positions = np.array(ref_positions)
    kappa_arr     = np.array(kappa_steps)
    xy_err        = np.linalg.norm(positions[:, :2] - ref_positions[:, :2], axis=1)

    kappa_p33 = np.percentile(kappa_arr, 33)
    kappa_p67 = np.percentile(kappa_arr, 67)
    straight_mask = kappa_arr <= kappa_p33
    apex_mask     = kappa_arr >= kappa_p67

    straight_med  = float(np.median(xy_err[straight_mask])) if straight_mask.any() else float("nan")
    apex_med      = float(np.median(xy_err[apex_mask]))     if apex_mask.any()     else float("nan")
    ratio         = apex_med / straight_med if straight_med > 0 else float("inf")
    overall_med   = float(np.median(xy_err))
    overall_mean  = float(np.mean(xy_err))
    # Match inline eval convention: ref at si (1 step behind drone at si+1).
    # ref_positions[i] = ref at step (i+1), so ref at step i = ref_positions[i-1].
    # Prepend ref at step 0 to get the full shifted array.
    ref0_xy = np.array(get_reference_pos(eval_traj, jnp.int32(0))[:2])
    refs_inline_xy = np.vstack([ref0_xy[None], ref_positions[:-1, :2]])
    xy_err_inline = np.linalg.norm(positions[:, :2] - refs_inline_xy, axis=1)
    mean_inline_convention = float(xy_err_inline.mean())

    print(f"\n{'='*55}")
    print(f"  APEX vs STRAIGHT — epoch_013000 figure_eight_normal")
    print(f"{'='*55}")
    print(f"  Steps completed:          {len(positions):4d}/1000  {'(CRASHED)' if crashed else '(CLEAN)'}")
    print(f"  Overall MED  (median):    {overall_med:.4f} m")
    print(f"  Overall MEAN (diag conv): {overall_mean:.4f} m")
    print(f"  Overall MEAN (inline conv, ref=si): {mean_inline_convention:.4f} m  [compare to training 0.066m]")
    print(f"  κ thresholds (p33/p67):   {kappa_p33:.3f} / {kappa_p67:.3f} m^-1")
    print(f"  κ_max (figure-eight):     {kappa_arr.max():.3f} m^-1")
    print(f"  Straight (κ ≤ p33):  n={straight_mask.sum():3d},  median_err={straight_med:.4f} m")
    print(f"  Apex    (κ ≥ p67):  n={apex_mask.sum():3d},  median_err={apex_med:.4f} m")
    print(f"  Apex:Straight ratio:      {ratio:.1f}x  (M1 baseline: 11x)")
    print(f"{'='*55}\n")

    # Error along trajectory
    error_by_position = []
    for i in range(len(positions)):
        ref = ref_positions[i]
        err = xy_err[i]
        angle = np.arctan2(ref[1], ref[0])
        error_by_position.append((angle, kappa_arr[i], err))

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("M1.3 epoch_013000 — figure_eight_normal Diagnostic", fontsize=13)

        # Panel 1: Flight path (xy)
        ax = axes[0]
        ref_x = ref_positions[:, 0]
        ref_y = ref_positions[:, 1]
        drone_x = positions[:, 0]
        drone_y = positions[:, 1]

        # Color drone path by XY error
        norm = mcolors.Normalize(vmin=0, vmax=0.20)
        cmap = plt.cm.RdYlGn_r
        for i in range(len(drone_x) - 1):
            ax.plot(drone_x[i:i+2], drone_y[i:i+2], color=cmap(norm(xy_err[i])), lw=1.5)

        ax.plot(ref_x, ref_y, "b--", lw=1, alpha=0.6, label="Reference")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="XY error (m)")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("Flight path (color = XY error)")
        ax.set_aspect("equal"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: XY error over time, colored by apex/straight
        ax2 = axes[1]
        t_axis = np.arange(len(xy_err)) * DT
        ax2.plot(t_axis, xy_err, "k-", lw=0.8, alpha=0.4, label="_nolegend_")
        ax2.fill_between(t_axis, 0, xy_err, where=apex_mask[:len(xy_err)],
                         alpha=0.4, color="red",   label=f"Apex (κ≥p67={kappa_p67:.2f})")
        ax2.fill_between(t_axis, 0, xy_err, where=straight_mask[:len(xy_err)],
                         alpha=0.4, color="green", label=f"Straight (κ≤p33={kappa_p33:.2f})")
        ax2.axhline(0.056, color="purple", ls="--", lw=1.5, label="Target (0.056m)")
        ax2.axhline(overall_med, color="black", ls="-.", lw=1, label=f"Overall MED ({overall_med:.3f}m)")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("XY error (m)")
        ax2.set_title(f"Error over time  |  Apex:Straight = {ratio:.1f}x")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(0.25, xy_err.max() * 1.1))

        OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_PLOT), dpi=130)
        print(f"Plot saved: {OUTPUT_PLOT}")

    except ImportError:
        print("matplotlib not available — skipping plot.")

    # Print summary line for the markdown report
    print(f"\nSUMMARY_LINE: epoch=013000 | overall_MED={overall_med:.4f} "
          f"| straight_med={straight_med:.4f} | apex_med={apex_med:.4f} "
          f"| ratio={ratio:.1f}x | crashed={crashed}")


if __name__ == "__main__":
    main()
