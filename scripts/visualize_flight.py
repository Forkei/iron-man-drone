"""
Live 3D flight visualization — M1.3 policy on figure-eight.

Shows:
  - Drone body flying in real-time (blue cylinder + colored propeller discs)
  - White dotted reference path (the full figure-eight)
  - Green moving sphere = current reference target
  - Red sphere = drone's actual position marker (for error visibility)

Usage:
  python scripts/visualize_flight.py                    # figure-eight normal, real-time
  python scripts/visualize_flight.py --speed 0.3        # slow-motion
  python scripts/visualize_flight.py --traj fast        # fast figure-eight
  python scripts/visualize_flight.py --loop             # repeat after each episode

Controls (MuJoCo viewer):
  Left-drag     rotate camera
  Right-drag    translate camera
  Scroll        zoom
  Space         pause/unpause
  Esc           quit
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import orbax.checkpoint as ocp
import yaml

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "experiments/m1_3_polynomial_fix"
    / "m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000"
)

# figure-eight periods
F8_PERIODS = {"slow": 15.0, "normal": 5.5, "fast": 3.5}


# ---------------------------------------------------------------------------
# Pure-numpy trajectory eval (no JAX in render loop)
# ---------------------------------------------------------------------------

def f8_positions(T, n_points=300):
    """Returns (n_points, 3) array tracing the figure-eight."""
    t = np.linspace(0, T, n_points, endpoint=False)
    x = np.cos(2 * np.pi * t / T)
    y = np.sin(4 * np.pi * t / T) / 2
    z = np.ones_like(t)
    return np.stack([x, y, z], axis=-1)


def f8_at_t(t, T):
    return np.array([
        np.cos(2 * np.pi * t / T),
        np.sin(4 * np.pi * t / T) / 2,
        1.0,
    ])


# ---------------------------------------------------------------------------
# Reference window precomputation (pure numpy, avoids JAX recompilation)
# ---------------------------------------------------------------------------

def precompute_refs(T, offset_steps, episode_steps, lookahead_n, lookahead_dt_steps, dt):
    ref_windows = np.zeros((episode_steps, lookahead_n, 3))
    ref_targets = np.zeros((episode_steps, 3))
    for si in range(episode_steps):
        t_base = (si + offset_steps) * dt
        ref_targets[si] = f8_at_t(t_base, T)
        for k in range(lookahead_n):
            t_look = t_base + (k + 1) * lookahead_dt_steps * dt
            ref_windows[si, k] = f8_at_t(t_look, T)
    return ref_windows, ref_targets


# ---------------------------------------------------------------------------
# MuJoCo viewer custom geometry helpers
# ---------------------------------------------------------------------------

def _mat_identity():
    return np.eye(3).flatten()


def draw_path_geoms(viewer, path_pts, rgba=(1.0, 1.0, 1.0, 0.35), radius=0.008):
    """Draw the reference trajectory as small spheres in viewer.user_scn."""
    scn = viewer.user_scn
    n = len(path_pts)
    # Reserve geoms 0..n-1 for path, geom n for target marker, n+1 for drone ghost
    scn.ngeom = 0
    mat = _mat_identity()
    for i, pt in enumerate(path_pts):
        if scn.ngeom >= scn.maxgeom - 4:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius]),
            pt.astype(np.float64),
            mat,
            np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1
    return scn.ngeom  # index where dynamic geoms start


def update_dynamic_geoms(viewer, path_geom_end, ref_pos, drone_pos, error):
    """Update the moving reference target and error indicator each step."""
    scn = viewer.user_scn
    mat = _mat_identity()
    scn.ngeom = path_geom_end

    # Green sphere = current reference target
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.025, 0.025, 0.025]),
        ref_pos.astype(np.float64),
        mat,
        np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32),
    )
    scn.ngeom += 1

    # Error line from drone to reference (red cylinder connecting them)
    if error > 0.005:
        mid = ((drone_pos + ref_pos) / 2).astype(np.float64)
        diff = ref_pos - drone_pos
        length = float(np.linalg.norm(diff))
        # Use mjv_makeConnector to draw a line geom
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([0.003, length / 2, 0.003]),
            mid,
            mat,
            np.array([1.0, 0.2, 0.2, 0.6], dtype=np.float32),
        )
        # Orient capsule along diff direction
        if length > 1e-6:
            z = diff / length
            x = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(z, x)) > 0.99:
                x = np.array([0.0, 1.0, 0.0])
            y = np.cross(z, x)
            y /= np.linalg.norm(y)
            x = np.cross(y, z)
            g.mat[:] = np.stack([x, y, z], axis=1)  # (3,3)
        scn.ngeom += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (0.3 = slow-mo, 1.0 = real-time)")
    parser.add_argument("--traj", default="normal", choices=["slow", "normal", "fast"],
                        help="Figure-eight variant")
    parser.add_argument("--loop", action="store_true",
                        help="Restart episode after completion")
    args = parser.parse_args()

    T = F8_PERIODS[args.traj]
    DT = 0.01
    EPISODE_STEPS = 1000
    LOOKAHEAD_N = 10
    LOOKAHEAD_DT_STEPS = 5
    offset_steps = int(round(T / 4.0 / DT))

    # ── Load actor ─────────────────────────────────────────────────────────
    checkpoint = str(Path(args.checkpoint).resolve())
    config_path = Path(checkpoint).parent.parent / "config_frozen.yaml"
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
    _, _, actor_state, critic_state = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(checkpoint, item={"actor": actor_state, "critic": critic_state})
    actor_state = ckpt["actor"]

    @jax.jit
    def actor_forward(params, obs):
        return actor_state.apply_fn(params, obs)

    print("Warming up actor JIT...")
    actor_forward(actor_state.params, jnp.zeros((1, cfg["observation"]["actor_dim"])))
    print("Done.")

    # ── Load MuJoCo ────────────────────────────────────────────────────────
    xml_path = str(REPO_ROOT / "src/iron_man_drone/envs/crazyflie.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)
    drone_body_id = mj_model.body("drone").id

    from iron_man_drone.control.ctbr_controller import (
        ctbr_to_rotor_speeds, compute_wrench, MASS, GRAVITY, KF,
    )
    hover_omega = np.sqrt(float(MASS * GRAVITY) / (4.0 * float(KF))) * np.ones(4)

    # ── Precompute references ───────────────────────────────────────────────
    print(f"Precomputing references for figure-eight {args.traj} (T={T}s, offset={offset_steps} steps)...")
    ref_windows, ref_targets = precompute_refs(
        T, offset_steps, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, DT
    )
    path_pts = f8_positions(T, n_points=250)
    print("Done.")

    def reset_sim():
        mj_data.qpos[:3] = [0.0, 0.0, 1.0]
        mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        return hover_omega.copy()

    # ── Launch viewer ───────────────────────────────────────────────────────
    print(f"\nLaunching viewer — figure-eight {args.traj}, speed {args.speed}x")
    print("Controls: left-drag=rotate  right-drag=pan  scroll=zoom  Space=pause  Esc=quit\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Set a nice initial camera angle
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 45

        # Draw static reference path once
        path_geom_end = draw_path_geoms(viewer, path_pts)

        rotor_speeds = reset_sim()
        step_i = 0
        episode = 1

        print(f"Episode {episode} — watching drone fly figure-eight {args.traj} ({T}s period)")

        while viewer.is_running():
            t_step_start = time.perf_counter()

            if step_i >= EPISODE_STEPS:
                if not args.loop:
                    print("Episode complete. Close the window or run with --loop to repeat.")
                    # Keep viewer open but freeze simulation
                    viewer.sync()
                    time.sleep(0.05)
                    continue
                episode += 1
                rotor_speeds = reset_sim()
                step_i = 0
                print(f"Episode {episode}")

            # ── Policy step ──────────────────────────────────────────────────
            pos    = np.array(mj_data.xpos[drone_body_id])
            R_flat = np.array(mj_data.xmat[drone_body_id]).reshape(-1)
            vel    = np.array(mj_data.qvel[:3])

            ref_win = ref_windows[step_i]                    # (10, 3)
            e_W = (ref_win - pos[None, :]).reshape(-1)       # (30,)
            actor_obs = jnp.array(np.concatenate([e_W, vel, R_flat])[None])
            mean, _ = actor_forward(actor_state.params, actor_obs)
            action = np.array(mean[0])

            omega_body = np.array(mj_data.qvel[3:6])
            new_rs = np.array(ctbr_to_rotor_speeds(
                jnp.array(action), jnp.array(rotor_speeds),
                jnp.array(omega_body), DT,
            ))
            force_body, torque_body = compute_wrench(jnp.array(new_rs))
            R_mat = np.array(mj_data.xmat[drone_body_id]).reshape(3, 3)

            mj_data.xfrc_applied[:] = 0.0
            mj_data.xfrc_applied[drone_body_id, :3] = R_mat @ np.array(force_body)
            mj_data.xfrc_applied[drone_body_id, 3:]  = R_mat @ np.array(torque_body)
            mujoco.mj_step(mj_model, mj_data)
            rotor_speeds = new_rs

            pos_after = np.array(mj_data.xpos[drone_body_id])
            ref_pos   = ref_targets[step_i]
            xy_err    = float(np.linalg.norm(pos_after[:2] - ref_pos[:2]))

            # ── Update viewer ────────────────────────────────────────────────
            update_dynamic_geoms(viewer, path_geom_end, ref_pos, pos_after, xy_err)
            viewer.sync()

            step_i += 1

            # ── Real-time pacing ─────────────────────────────────────────────
            elapsed = time.perf_counter() - t_step_start
            remaining = (DT / args.speed) - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
