"""
Render a third-person clip of the M3 policy flying one episode, for human review.

Picks an episode matching a target difficulty bucket (by ideal-path clearance),
replays the greedy policy, and records a tracking camera view with an overlay
showing mode, step, ideal-path clearance, live obstacle distance, and outcome.

Two-pass design: pass 1 runs the full policy rollout using MJWarp depth render
for the obs and records the per-step drone/obstacle state; pass 2 renders the
recorded trajectory with MuJoCo's EGL renderer. The passes are separated because
interleaving MJWarp and EGL GPU render contexts per-step corrupts the output.

Buckets (ideal-path clearance = min dist from reference trajectory to any obstacle):
  unfair    < 0.15 m   ideal line passes through crash zone (guaranteed crash)
  tight     0.15-0.35  real avoidance required
  moderate  0.35-0.50
  clear     >= 0.50    obstacles never near the path

Usage (inside WSL / jax_env):
  /home/forke/jax_env/bin/python scripts/render_m3_clip.py \
      --mode forest --bucket tight --out experiments/m3_run1/clips/forest_tight.mp4
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iron_man_drone.envs.quadrotor_env_m3 import (
    M3VecEnv, D_CRASH, D_SAFE, EPISODE_STEPS, N_OBSTACLE_SLOTS,
)
from eval_m3 import build_states, make_eval_fns
from train_m3 import load_checkpoint, find_latest_checkpoint

BUCKETS = {
    "unfair":   (-np.inf, D_CRASH),
    "tight":    (D_CRASH, 0.35),
    "moderate": (0.35,    D_SAFE),
    "clear":    (D_SAFE,  np.inf),
    "any":      (-np.inf, np.inf),
}


def _add_spheres(scn, pts, rgba, r):
    """Append sphere geoms (a path) into the live MjvScene after update_scene."""
    rgba = np.array(rgba, dtype=np.float32)
    size = np.array([r, 0.0, 0.0])
    eye  = np.eye(3).flatten()
    for p in pts:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            size, np.asarray(p, dtype=np.float64), eye, rgba)
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mode", default="forest")
    ap.add_argument("--bucket", default="tight", choices=list(BUCKETS))
    ap.add_argument("--n_search", type=int, default=48, help="batch to search for a match")
    ap.add_argument("--pick", type=int, default=0, help="which matching episode (0=first)")
    ap.add_argument("--stride", type=int, default=2, help="render every Nth sim step")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(
        Path("/home/forke/m3_checkpoints/m3_run1"))
    print(f"Checkpoint: {ckpt}")
    actor, critic, encoder, actor_state, critic_state, enc_state = build_states()
    actor_state, critic_state, enc_state, epoch = load_checkpoint(
        ckpt, actor_state, critic_state, enc_state)
    print(f"Loaded epoch {epoch}")

    class _Cfg:
        num_envs = args.n_search
    env = M3VecEnv(_Cfg(), fault_prob=0.0)
    encode, assemble, greedy_action, ref_pos_batch, ref_clearance = make_eval_fns(actor, encoder)
    drone_id = env.drone_body_id

    keys = jax.random.split(jax.random.PRNGKey(7), args.n_search)
    states, base_obs, priv = env.batch_reset(
        keys, modes=[args.mode] * args.n_search, density_mult=1.0)

    clearance = np.array(ref_clearance(
        states.traj, states.obstacle_positions,
        states.obstacle_half_extents, states.n_obstacles))
    lo, hi = BUCKETS[args.bucket]
    matches = np.where((clearance >= lo) & (clearance < hi))[0]
    if len(matches) == 0:
        print(f"No episode in bucket '{args.bucket}' (clearances: "
              f"{np.sort(clearance)[:8].round(2)} ...). Try --bucket any or larger --n_search.")
        sys.exit(1)
    idx = int(matches[min(args.pick, len(matches) - 1)])
    print(f"Episode {idx}: mode={args.mode} clearance={clearance[idx]:.3f}m "
          f"({len(matches)} match bucket '{args.bucket}')")

    obstacles = np.array(states.obstacle_positions[idx])    # (16,3) fixed for episode
    n_obs = int(np.array(states.n_obstacles[idx]))

    # ── Pass 1: policy rollout (MJWarp for obs only); record per-step state ───
    depth_bins = env.compute_depth_bins(env.batch_render(states))
    z_hat      = encode(enc_state.params, states.obs_base_buf, states.action_buf)
    k_norm     = states.step.astype(jnp.float32) / EPISODE_STEPS
    obs_dists  = env.compute_k_nearest_batch(states)
    actor_obs  = assemble(base_obs, z_hat, depth_bins, k_norm, obs_dists)

    rec = []   # (qpos(7), pos(3), ref(3), dmin, step)
    outcome, crash_step = "completed", None
    for t in range(EPISODE_STEPS):
        action = greedy_action(actor_state.params, actor_obs)
        states, base_obs, _, _, done = env.batch_step(states, action)

        od     = env.compute_k_nearest_batch(states)
        dmin   = float(np.array(jnp.min(od[idx])))
        pos    = np.array(states.mjx_data.xpos[idx, drone_id])
        ref    = np.array(ref_pos_batch(states.traj, states.step))[idx]
        step_i = int(np.array(states.step[idx]))
        done_i = bool(np.array(done[idx]))

        if dmin < D_CRASH and crash_step is None:
            outcome, crash_step = "CRASH", step_i
        elif done_i and dmin >= D_CRASH and step_i < EPISODE_STEPS - 5 and crash_step is None:
            outcome, crash_step = "OUT OF BOUNDS", step_i

        rec.append((np.array(states.mjx_data.qpos[idx]), pos, ref, dmin, step_i))

        depth_bins = env.compute_depth_bins(env.batch_render(states))
        z_hat      = encode(enc_state.params, states.obs_base_buf, states.action_buf)
        k_norm     = states.step.astype(jnp.float32) / EPISODE_STEPS
        actor_obs  = assemble(base_obs, z_hat, depth_bins, k_norm, od)
        if done_i:
            break

    # Deviation-vs-proximity: does the drone leave the reference line near obstacles?
    devs  = np.array([np.linalg.norm(p - r) for _, p, r, _, _ in rec])
    dmins = np.array([d for _, _, _, d, _ in rec])
    near  = dmins < D_SAFE
    print(f"Episode summary: {len(rec)} steps, outcome={outcome}"
          + (f" @step {crash_step}" if crash_step else ""))
    if near.any():
        away = devs[~near].mean() if (~near).any() else float("nan")
        print(f"  track-deviation near obstacles (d<{D_SAFE}m): {devs[near].mean():.3f}m "
              f"| away from obstacles: {away:.3f}m | min nearest_obs reached: {dmins.min():.3f}m")
        print("  (deviation HIGHER near obstacles => policy leaves the line to avoid)")

    # ── Pass 2: EGL render of the recorded trajectory (MJWarp idle now) ───────
    # Separate model copy with a wide view frustum. The env XML sets zfar=5m for
    # the depth camera; reusing it for a third-person view collapses the depth
    # buffer over the 20m floor -> heavy z-fighting speckle. Widen it here.
    from iron_man_drone.envs.quadrotor_env_depth import DEPTH_XML
    rmodel = mujoco.MjModel.from_xml_path(DEPTH_XML)
    rmodel.vis.map.znear = 0.05
    rmodel.vis.map.zfar  = 60.0
    mj_model = rmodel
    mj_data  = mujoco.MjData(rmodel)
    renderer = mujoco.Renderer(rmodel, height=args.h, width=args.w, max_geom=20000)
    # Reference path (blue) and drone's flown path (yellow) for overlay.
    ref_path    = np.array([r for _, _, r, _, _ in rec])    # (T,3) commanded
    actual_path = np.array([p for _, p, _, _, _ in rec])    # (T,3) flown
    ref_draw    = ref_path[::4]

    # Fixed high-angle camera framing the WHOLE scene (path + all active
    # obstacles), so the close-encounter region is always in view instead of
    # tracking the drone into an empty patch of arena.
    active_obs = obstacles[np.linalg.norm(obstacles, axis=1) < 50.0]
    focus = np.vstack([ref_path, active_obs]) if len(active_obs) else ref_path
    center = focus.mean(axis=0); center[2] = 0.5
    extent = float(np.linalg.norm(focus[:, :2] - center[:2], axis=1).max())
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 90.0, -55.0
    cam.distance = max(4.0, extent * 2.4)
    cam.lookat[:] = center

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1.0 / (0.01 * args.stride))))
    vw  = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.w, args.h))
    if not vw.isOpened():
        print("mp4 writer failed — falling back to .gif")
        args.out = str(Path(args.out).with_suffix(".gif")); vw = None
    gif_frames = []

    def emit(img):
        if vw is not None:
            vw.write(img)
        else:
            gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    frames_written, img = 0, None
    for fi, (qpos, pos, ref, dmin, step_i) in enumerate(rec):
        is_last = (fi == len(rec) - 1)
        if (fi % args.stride) and not is_last:
            continue
        mj_data.qpos[:len(qpos)] = qpos
        mj_data.mocap_pos[:obstacles.shape[0]] = obstacles
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        _add_spheres(renderer.scene, ref_draw,             (0.1, 0.4, 1.0, 1.0), 0.025)  # blue = commanded
        _add_spheres(renderer.scene, actual_path[:fi+1:2], (1.0, 0.85, 0.1, 1.0), 0.018) # yellow = flown
        _add_spheres(renderer.scene, pos[None],            (1.0, 0.1, 0.9, 1.0), 0.11)   # magenta = drone now
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        reached = crash_step is not None and step_i >= crash_step
        status  = outcome if reached else "tracking"
        color   = (0, 0, 255) if reached else (0, 200, 0)
        lines = [
            f"mode={args.mode}  clearance={clearance[idx]:.2f}m  obstacles={n_obs}",
            f"step={step_i:4d}  track_err={np.linalg.norm(pos-ref):.3f}m  nearest_obs={dmin:.2f}m",
            status + (f" @step {crash_step}" if reached else ""),
        ]
        for j, ln in enumerate(lines):
            cv2.putText(img, ln, (10, 24 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
        if reached:
            cv2.rectangle(img, (2, 2), (args.w - 3, args.h - 3), (0, 0, 255), 4)
        emit(img); frames_written += 1

    if img is not None:                      # hold final frame ~0.5s
        for _ in range(fps // 2):
            emit(img); frames_written += 1

    if vw is not None:
        vw.release()
    else:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in gif_frames]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)

    print(f"Wrote {frames_written} frames -> {args.out}  (outcome: {outcome}"
          + (f", crash @step {crash_step}" if crash_step else "") + ")")


if __name__ == "__main__":
    main()
