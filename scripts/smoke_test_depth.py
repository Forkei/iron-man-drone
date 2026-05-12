"""
SC-3 — Depth render sanity check.

Protocol:
  - Spawn drone at z=0.5, facing +x (forward).
  - Place one obstacle at (1.5, 0, 1.0) in world-0; other worlds clear.
  - Render one depth frame.
  - Assert obstacle is visible: at least one pixel depth < 0.5
    (obstacle at 1.5 m, max range 5 m → normalized depth ≈ 0.30).
  - Save frame to notes/figures/m2_5_depth_smoke.png.

Usage:
  python scripts/smoke_test_depth.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import types
import numpy as np
import jax
import jax.numpy as jnp

REPO_ROOT   = Path(__file__).parent.parent
FIGURES_DIR = REPO_ROOT / "notes" / "figures"
PNG_PATH    = FIGURES_DIR / "m2_5_depth_smoke.png"


def _gate(label, passed, detail=""):
    mark = "✓" if passed else "✗"
    print(f"  {mark}  {label}")
    if detail:
        print(f"      {detail}")
    if not passed:
        raise AssertionError(f"FAILED: {label}  {detail}")
    return True


def main():
    print(f"\n{'='*60}")
    print(f"  SC-3 — Depth render smoke test")
    print(f"{'='*60}\n")

    from iron_man_drone.envs.quadrotor_env_depth import DepthVecEnv, N_OBSTACLE_SLOTS
    from iron_man_drone.utils.obstacle_randomization import sample_obstacle_configs

    # N=2: world-0 has obstacle, world-1 is clear (for contrast check)
    cfg = types.SimpleNamespace(num_envs=2, max_episode_steps=1000)
    env = DepthVecEnv(cfg, n_obstacles=0)   # start with n_obstacles=0

    # Build per-env obstacle configs manually
    import numpy as np
    centers_all = np.full((2, N_OBSTACLE_SLOTS, 3), 100.0, dtype=np.float32)
    he_all      = np.zeros((2, N_OBSTACLE_SLOTS, 3), dtype=np.float32)

    # World-0: one obstacle at (1.5, 0, 1.0)
    centers_all[0, 0] = [1.5, 0.0, 1.0]
    he_all[0, 0]      = [0.05, 0.05, 0.5]

    obs_pos = jnp.array(centers_all)
    obs_he  = jnp.array(he_all)

    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    states, _, _ = env._reset_jit(keys, obs_pos, obs_he)

    # Render
    depth = env.batch_render(states)   # (2, 64, 64) float32
    depth_np = np.array(depth)

    # ── Assertions ────────────────────────────────────────────────────────────
    _gate("depth shape (2, 64, 64)", depth_np.shape == (2, 64, 64),
          f"got {depth_np.shape}")

    _gate("depth dtype float32", depth_np.dtype == np.float32)

    _gate("depth range [0, 1]",
          0.0 <= float(depth_np.min()) and float(depth_np.max()) <= 1.0,
          f"min={depth_np.min():.3f}  max={depth_np.max():.3f}")

    # Obstacle at 1.5 m / 5 m = 0.30 normalized depth
    obstacle_visible = bool((depth_np[0] < 0.5).any())
    _gate("obstacle visible in world-0 (pixel < 0.5)",
          obstacle_visible,
          f"world-0 min depth={depth_np[0].min():.3f}  "
          f"world-1 min depth={depth_np[1].min():.3f}")

    # World-1 has no obstacle — should have more bright (far) pixels on average
    mean_0 = float(depth_np[0].mean())
    mean_1 = float(depth_np[1].mean())
    _gate("world-0 (obstacle) darker on average than world-1 (clear)",
          mean_0 < mean_1,
          f"world-0 mean={mean_0:.3f}  world-1 mean={mean_1:.3f}")

    # ── Save PNG ──────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, (ax, title) in enumerate(zip(axes, ["world-0 (obstacle at 1.5 m)", "world-1 (clear)"])):
        im = ax.imshow(depth_np[i], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="normalized depth (0=near, 1=5 m)")
    fig.suptitle("M2.5 depth smoke test — crazyflie_depth.xml, 64×64, fovy=60°, max=5 m")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)

    print(f"\n  Depth frame saved to {PNG_PATH}")
    print(f"\n{'='*60}")
    print(f"  ALL PASS — SC-3 complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
