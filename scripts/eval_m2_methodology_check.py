"""
M2 eval methodology comparison — T/4 offset vs t=0.

Runs a single M2 checkpoint on figure_eight_normal with BOTH:
  (a) t=0 start (current eval_m2_full.py methodology — cold-start inflated)
  (b) T/4 offset (M1.3 correct methodology — matches SimpleFlight)

Purpose: quantify the exact systematic offset between M2's current eval
methodology and the corrected methodology used to produce M1.3's 0.037m number.

Usage:
  python scripts/eval_m2_methodology_check.py --checkpoint PATH/checkpoints/epoch_007000
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

F8_NORMAL_T = 5.5   # figure_eight_normal period (seconds)


def load_actor(checkpoint_path):
    # Find config — may be in parent.parent (ablation run) or use m2 config directly
    ckpt_path = Path(checkpoint_path).resolve()
    config_path = ckpt_path.parent.parent / "config_frozen.yaml"
    if not config_path.exists():
        config_path = REPO_ROOT / "experiments/m2_no_dr_ablation/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from iron_man_drone.policy.ppo import PPOConfig, create_train_states
    ppo_cfg = PPOConfig(
        actor_obs_dim=cfg["observation"]["actor_dim"],   # 50
        critic_obs_dim=cfg["observation"]["critic_dim"], # 51
        action_dim=cfg["action"]["dim"],
        hidden_dim=cfg["network"]["hidden_dim"],
        num_layers=cfg["network"]["num_layers"],
    )
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(ckpt_path))
    actor_state = actor_state.replace(params=ckpt["actor"]["params"])

    @jax.jit
    def actor_forward(params, obs):
        return actor_state.apply_fn(params, obs)

    dummy = jnp.zeros((1, cfg["observation"]["actor_dim"]))
    actor_forward(actor_state.params, dummy)
    return actor_state, actor_forward, cfg


def _f8_pos(t, T=F8_NORMAL_T):
    """Lemniscate position at time t (numpy)."""
    t = np.asarray(t, dtype=float)
    return np.stack([
        np.cos(2*np.pi*t/T),
        np.sin(4*np.pi*t/T)/2,
        np.ones_like(t),
    ], axis=-1)


def run_rollout_cpu(actor_state, actor_forward, cfg, mj_model, mj_data, drone_body_id,
                    offset_steps, label, priv_state_np):
    """
    CPU mujoco rollout with precomputed reference positions.
    offset_steps: how many trajectory steps to offset reference (T/4 → 137 steps; 0 → no offset).
    priv_state_np: (8,) numpy array appended to 42-dim base obs → 50-dim M2 obs.
    """
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT
    from iron_man_drone.control.ctbr_controller import (
        ctbr_to_rotor_speeds, compute_wrench, MASS, GRAVITY, KF,
    )

    # Precompute reference positions and lookahead windows (plain numpy)
    ref_pos_err = np.zeros((EPISODE_STEPS, 3))   # reference at each step (for error)
    ref_windows = np.zeros((EPISODE_STEPS, LOOKAHEAD_N, 3))  # lookahead for obs

    for si in range(EPISODE_STEPS):
        t_base = (si + offset_steps) * DT
        ref_pos_err[si] = _f8_pos(t_base)
        for k in range(LOOKAHEAD_N):
            t_look = t_base + (k + 1) * LOOKAHEAD_DT_STEPS * DT
            ref_windows[si, k] = _f8_pos(t_look)

    # Log initial reference position
    ref0 = ref_pos_err[0]
    init_err = float(np.linalg.norm(ref0[:2]))

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

        ref_win = ref_windows[step_i]          # (10, 3)
        e_W = (ref_win - pos[None, :]).reshape(-1)  # (30,)

        # 50-dim M2 obs: [e_W (30), vel (3), R_flat (9), priv_state (8)]
        base_obs = np.concatenate([e_W, vel, R_flat])  # 42-dim
        actor_obs = jnp.array(np.concatenate([base_obs, priv_state_np])[None])  # (1, 50)

        mean, _ = actor_forward(actor_state.params, actor_obs)
        action = np.array(mean[0])

        omega_current = np.array(mj_data.qvel[3:6])
        new_rs = np.array(ctbr_to_rotor_speeds(
            jnp.array(action), jnp.array(rotor_speeds),
            jnp.array(omega_current), DT,
        ))
        force_body_j, torque_body_j = compute_wrench(jnp.array(new_rs))
        R_mat = np.array(mj_data.xmat[drone_body_id]).reshape(3, 3)
        mj_data.xfrc_applied[:] = 0.0
        mj_data.xfrc_applied[drone_body_id, :3] = R_mat @ np.array(force_body_j)
        mj_data.xfrc_applied[drone_body_id, 3:]  = R_mat @ np.array(torque_body_j)
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
    med = float(xy_errs.mean())

    print(f"  [{label}] offset={offset_steps} steps | init_ref_xy_err={init_err:.4f}m | "
          f"MED={med:.4f}m | {'CRASHED' if crashed else 'clean'}")
    return med, crashed, xy_errs, init_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Checkpoint:  {args.checkpoint}")
    print()

    actor_state, actor_forward, cfg = load_actor(args.checkpoint)
    print(f"Actor loaded. obs_dim={cfg['observation']['actor_dim']}")
    print()

    xml_path = str(REPO_ROOT / "src/iron_man_drone/envs/crazyflie.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)
    drone_body_id = mj_model.body("drone").id

    from iron_man_drone.envs.quadrotor_env import DT

    # Nominal priv_state: [eta1,eta2,eta3,eta4, mass_scale, 0, 0, 0] = [1,1,1,1,1,0,0,0]
    priv_nominal = np.array([1., 1., 1., 1., 1., 0., 0., 0.], dtype=np.float32)

    # T/4 offset for figure_eight_normal (T=5.5s)
    t4_offset = int(round(F8_NORMAL_T / 4.0 / DT))  # 137 steps for DT=0.01
    t4_time   = t4_offset * DT

    print(f"figure_eight_normal: T={F8_NORMAL_T}s, T/4 offset = {t4_offset} steps ({t4_time:.3f}s)")
    ref_at_t4 = _f8_pos(t4_time)
    print(f"Reference at T/4: ({ref_at_t4[0]:.4f}, {ref_at_t4[1]:.4f}, {ref_at_t4[2]:.4f})")
    print(f"Reference at t=0: ({_f8_pos(0.0)[0]:.4f}, {_f8_pos(0.0)[1]:.4f}, {_f8_pos(0.0)[2]:.4f})")
    print()

    print("Running figure_eight_normal — t=0 (current M2 eval methodology, cold-start):")
    med_t0, crash_t0, errs_t0, init0 = run_rollout_cpu(
        actor_state, actor_forward, cfg, mj_model, mj_data, drone_body_id,
        offset_steps=0, label="t=0", priv_state_np=priv_nominal,
    )

    print()
    print("Running figure_eight_normal — T/4 offset (M1.3 correct methodology):")
    med_t4, crash_t4, errs_t4, init4 = run_rollout_cpu(
        actor_state, actor_forward, cfg, mj_model, mj_data, drone_body_id,
        offset_steps=t4_offset, label="T/4", priv_state_np=priv_nominal,
    )

    print()
    print("=" * 72)
    print("  METHODOLOGY COMPARISON — figure_eight_normal")
    print("=" * 72)
    print(f"  t=0  (current M2 methodology):  {med_t0:.4f} m  "
          f"({'CRASH' if crash_t0 else 'clean'})  init_err={init0:.4f}m")
    print(f"  T/4  (M1.3 correct methodology): {med_t4:.4f} m  "
          f"({'CRASH' if crash_t4 else 'clean'})  init_err={init4:.4f}m")
    print(f"  Offset (t=0 − T/4):              {med_t0 - med_t4:.4f} m")
    print(f"  M1.3 reference:  t=0 → 0.0690m,  T/4 → 0.0369m  (offset = 0.0321m)")
    print(f"  M2 spec target (T/4 basis):      0.0370 m")
    print(f"  M2 no-DR epoch_007000 (T/4):     {med_t4:.4f} m  "
          f"({'PASS' if med_t4 <= 0.037 else 'FAIL: ×'+f'{med_t4/0.037:.2f}'})")
    print("=" * 72)


if __name__ == "__main__":
    main()
