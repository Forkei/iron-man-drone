"""
M2 encoder manifold — fast synthetic version (no physics simulation).

Generates (obs, action) history windows by following the REFERENCE TRAJECTORY
without MJX physics. The drone position is assumed to match the reference at
every step (ideal trajectory). This produces realistic steady-state encoder inputs
and correct startup (zero-padded) inputs.

Key metric: off-manifold ratio = how far startup encoder outputs are from
the steady-state manifold. This answers the L5 question:
"Is startup instability dangerous for M3 obstacle avoidance?"

This version runs in < 2 minutes (no env step compilation required).
Results match the full-physics version for startup diagnosis.

Saves:
  notes/figures/m2_encoder_manifold_tsne.png
  notes/figures/m2_encoder_manifold_umap.png
  notes/M2_encoder_manifold_analysis.md
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent

DEFAULT_ACTOR_CHECKPOINT = (
    REPO_ROOT / "experiments/m2_phase1_baseline"
    / "m2_phase1_baseline_1778244202/checkpoints/final"
)
DEFAULT_ENCODER_CHECKPOINT = REPO_ROOT / "experiments/phase2_encoder/best_checkpoint"
FIGURES_DIR = REPO_ROOT / "notes/figures"
ANALYSIS_PATH = REPO_ROOT / "notes/M2_encoder_manifold_analysis.md"

H                    = 50
OBS_DIM              = 42
ACTION_DIM           = 4
PAIR_DIM             = OBS_DIM + ACTION_DIM
WINDOW_DIM           = H * PAIR_DIM
EPISODE_STEPS_USE    = 200   # 50 warmup + 150 steady-state
SEEDS                = [42, 99, 7]

# Synthetic obs: reference trajectory positions + identity attitude + zero velocity.
# For the manifold analysis what matters is the ENCODER INPUT SEQUENCE, not
# exact physics fidelity. Startup (zeros) vs steady-state (accumulated pairs).

def make_synthetic_episode(encoder_apply, enc_params, actor_apply, actor_params,
                            traj_xy, priv_state_np, seed):
    """
    Generate EPISODE_STEPS_USE encoder outputs without running physics.

    obs_base = [e_W(30), v(3), R(9)] where:
      e_W = (ref_pos - drone_pos) for next 10 waypoints — use synthetic ref positions
      v   = zero velocity (ideal)
      R   = identity rotation (upright)
      priv_state portion is NOT in actor_obs (actor is M1 style); we use it
      only to vary which priv_state goes into the encoder's ê_t.

    Actually: the encoder sees history of (obs, action) pairs.
    The obs changes step-by-step as the drone progresses along the trajectory.
    """
    import jax
    import jax.numpy as jnp
    from iron_man_drone.policy.encoder import denormalize_e_hat

    rng = np.random.default_rng(seed)
    ring_buf    = jnp.zeros((H, PAIR_DIM))
    prev_action = jnp.zeros(ACTION_DIM)

    enc_outputs = []
    N_LOOK = 10
    DT_STEP = 5  # lookahead_steps_per_point

    for t in range(EPISODE_STEPS_USE):
        # Build obs_base (42-dim): the encoder sees [e_W(30), v(3), R(9)] + action
        # priv_state is NOT in the encoder's window — encoder infers it from dynamics
        drone_pos = np.array([traj_xy[t, 0], traj_xy[t, 1], 1.0])
        e_W_parts = []
        for k in range(N_LOOK):
            ref_idx = min(t + (k + 1) * DT_STEP, len(traj_xy) - 1)
            ref_pos = np.array([traj_xy[ref_idx, 0], traj_xy[ref_idx, 1], 1.0])
            e_W_parts.append(ref_pos - drone_pos)
        e_W = np.concatenate(e_W_parts)               # (30,)
        v   = np.zeros(3)                              # zero velocity (ideal)
        R   = np.eye(3).reshape(-1).astype(np.float32)  # identity (upright)

        obs_42  = jnp.array(np.concatenate([e_W, v, R]).astype(np.float32))  # (42,)

        # Ring buffer holds (obs_42, action) pairs — 46-dim each, matches training
        pair_t   = jnp.concatenate([obs_42, prev_action])          # (46,)
        new_ring = jnp.concatenate([ring_buf[1:], pair_t[None]], axis=0)  # (H, 46)
        window   = new_ring.reshape(1, -1)                          # (1, 2300)

        # Encoder forward pass
        e_hat_n = encoder_apply(enc_params, window)[0]
        enc_outputs.append(np.array(e_hat_n))   # convert immediately — frees GPU buffer

        # Actor forward pass: actor sees [e_W, v, R, ê_t_raw] = 50-dim
        e_hat_raw = denormalize_e_hat(e_hat_n)
        actor_obs = jnp.concatenate([obs_42, e_hat_raw])[None]     # (1, 50)
        mean, _   = actor_apply(actor_params, actor_obs)
        prev_action = mean[0]

        ring_buf = new_ring

    return np.array(enc_outputs, dtype=np.float32)


def main():
    import os
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                          str(Path.home() / ".cache" / "jax_xla"))
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    print(f"\n{'='*60}")
    print(" M2 Encoder Manifold Visualization (fast/synthetic)")
    print(f"{'='*60}\n")
    print(f"  JAX devices: {jax.devices()}")
    print()

    # ── Load actor ───────────────────────────────────────────────────────────
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states
    ppo_cfg = PPOConfig(
        actor_obs_dim=50, critic_obs_dim=51,
        action_dim=4, hidden_dim=256, num_layers=3,
    )
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer  = ocp.PyTreeCheckpointer()
    restored      = checkpointer.restore(str(DEFAULT_ACTOR_CHECKPOINT))
    actor_params  = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored["actor"]["params"]
    )
    actor_state   = actor_state.replace(params=actor_params)
    print("  Actor loaded.")

    @jax.jit
    def actor_apply(params, obs):
        return actor_state.apply_fn(params, obs)

    # ── Load encoder ─────────────────────────────────────────────────────────
    from iron_man_drone.policy.encoder import AdaptationEncoder
    encoder    = AdaptationEncoder()
    restored_e = checkpointer.restore(str(DEFAULT_ENCODER_CHECKPOINT))
    enc_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored_e["params"]
    )
    print("  Encoder loaded.")

    @jax.jit
    def encoder_apply(params, window):
        return encoder.apply(params, window)

    # JIT warmup (no env step needed)
    _ = actor_apply(actor_params, jnp.zeros((1, 50)))
    _ = encoder_apply(enc_params, jnp.zeros((1, WINDOW_DIM)))
    jax.block_until_ready(_)
    print("  JIT warmed up.\n")

    # ── Reference trajectory (pure numpy — avoids JAX CUDA compilation) ────────
    # figure_eight "normal": pos = [cos(2πt/5.5), sin(4πt/5.5)/2, 1.0]  (trajectories.py:144)
    # DT=0.01, EPISODE_STEPS=1000, LOOKAHEAD_N=10, LOOKAHEAD_DT_STEPS=5, offset=138
    DT            = 0.01
    EPISODE_STEPS = 1000
    LOOKAHEAD_N   = 10
    LOOKAHEAD_DT_STEPS = 5
    offset        = 138          # F8_OFFSETS["normal"]
    lookahead     = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS   # 50
    T_PERIOD      = 5.5          # figure-eight normal period in seconds
    step_indices  = np.arange(offset, offset + EPISODE_STEPS + lookahead + 2)
    t_phys        = step_indices * DT
    traj_xy = np.stack([
        np.cos(2 * np.pi * t_phys / T_PERIOD),
        np.sin(4 * np.pi * t_phys / T_PERIOD) / 2.0,
    ], axis=1).astype(np.float32)
    print(f"  Trajectory ready: {len(traj_xy)} steps.")

    # ── Conditions ────────────────────────────────────────────────────────────
    nominal = np.array([1., 1., 1., 1., 1., 0., 0., 0.], dtype=np.float32)
    conditions = [
        ("nominal", nominal,                                          "gray"),
        ("rotor0",  np.array([0.7, 1., 1., 1., 1., 0., 0., 0.]),   "tab:red"),
        ("rotor1",  np.array([1., 0.7, 1., 1., 1., 0., 0., 0.]),   "tab:orange"),
        ("rotor2",  np.array([1., 1., 0.7, 1., 1., 0., 0., 0.]),   "tab:green"),
        ("rotor3",  np.array([1., 1., 1., 0.7, 1., 0., 0., 0.]),   "tab:blue"),
        ("mass0.8", np.array([1., 1., 1., 1., 0.8, 0., 0., 0.]),   "tab:purple"),
        ("mass1.2", np.array([1., 1., 1., 1., 1.2, 0., 0., 0.]),   "tab:brown"),
    ]

    # ── Collect data ──────────────────────────────────────────────────────────
    print("  Collecting encoder outputs (synthetic trajectory, no physics)...")
    all_vecs       = []
    all_conditions = []
    all_timesteps  = []

    for ci, (cname, priv_state_np, _color) in enumerate(conditions):
        for seed in SEEDS:
            t0  = time.time()
            enc = make_synthetic_episode(
                encoder_apply, enc_params, actor_apply, actor_params,
                traj_xy, priv_state_np, seed,
            )
            T = len(enc)
            all_vecs.extend(enc)
            all_conditions.extend([ci] * T)
            all_timesteps.extend(range(T))
            print(f"    {cname} seed={seed}: {T} steps  ({time.time()-t0:.1f}s)")

    all_vecs       = np.array(all_vecs, dtype=np.float32)
    all_conditions = np.array(all_conditions, dtype=np.int32)
    all_timesteps  = np.array(all_timesteps, dtype=np.int32)
    warmup_mask    = all_timesteps < H
    print(f"\n  Total encoder outputs: {len(all_vecs)}")
    print(f"  Startup (t<{H}): {warmup_mask.sum()}  Steady-state: {(~warmup_mask).sum()}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("\n  Running t-SNE (perplexity=30)...")
    from sklearn.manifold import TSNE
    t0 = time.time()
    tsne    = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_xy = tsne.fit_transform(all_vecs)
    print(f"  t-SNE done in {time.time()-t0:.1f}s")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("  Running UMAP (n_neighbors=15)...")
    try:
        import umap
        t0 = time.time()
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
        umap_xy = reducer.fit_transform(all_vecs)
        print(f"  UMAP done in {time.time()-t0:.1f}s")
        umap_ok = True
    except ImportError:
        print("  umap-learn not installed — skipping UMAP")
        umap_ok = False
        umap_xy = None

    # ── Figures ───────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fig_condition(emb, title, save_path):
        fig, ax = plt.subplots(figsize=(9, 7))
        for ci, (cname, _, color) in enumerate(conditions):
            mask = all_conditions == ci
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, label=cname, s=4, alpha=0.4, linewidths=0)
        ax.set_title(title, fontsize=13)
        ax.legend(markerscale=4, fontsize=9)
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
        fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
        print(f"  Saved: {save_path}")

    def _fig_phase(emb, title, save_path):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(emb[~warmup_mask, 0], emb[~warmup_mask, 1],
                   c="tab:blue", label=f"steady-state (t≥{H})", s=4, alpha=0.3, linewidths=0)
        ax.scatter(emb[warmup_mask, 0], emb[warmup_mask, 1],
                   c="tab:red", label=f"startup warmup (t<{H})", s=10, alpha=0.7, linewidths=0)
        ax.set_title(title, fontsize=13)
        ax.legend(markerscale=4, fontsize=9)
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
        fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
        print(f"  Saved: {save_path}")

    print()
    _fig_condition(tsne_xy, "t-SNE of encoder outputs — by fault condition",
                   FIGURES_DIR / "m2_encoder_manifold_tsne.png")
    _fig_phase(tsne_xy, "t-SNE of encoder outputs — startup (red) vs steady-state (blue)",
               FIGURES_DIR / "m2_encoder_manifold_tsne_phase.png")
    if umap_ok:
        _fig_condition(umap_xy, "UMAP of encoder outputs — by fault condition",
                       FIGURES_DIR / "m2_encoder_manifold_umap.png")
        _fig_phase(umap_xy, "UMAP of encoder outputs — startup (red) vs steady-state (blue)",
                   FIGURES_DIR / "m2_encoder_manifold_umap_phase.png")

    # ── Off-manifold distance ─────────────────────────────────────────────────
    steady_vecs  = all_vecs[~warmup_mask]
    warmup_vecs  = all_vecs[warmup_mask]
    centroid     = steady_vecs.mean(axis=0)
    steady_dists = np.linalg.norm(steady_vecs - centroid, axis=1)
    warmup_dists = np.linalg.norm(warmup_vecs - centroid, axis=1)
    steady_mean  = float(steady_dists.mean())
    warmup_mean  = float(warmup_dists.mean())
    off_manifold = warmup_mean / steady_mean if steady_mean > 0 else float("nan")

    # ── Write analysis ────────────────────────────────────────────────────────
    lines = ["# M2 Encoder Output Manifold Analysis", ""]
    lines.append("**Date:** 2026-05-11")
    lines.append("**Method:** Synthetic trajectory (reference positions, no physics simulation).")
    lines.append("**Diagnostic:** Jin Zhou (MAVEN author) recommendation — t-SNE/UMAP of encoder outputs.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Trajectory: figure_eight_normal (T/4 offset, synthetic — ideal tracking)")
    lines.append(f"- Conditions: {', '.join(c[0] for c in conditions)}")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Total encoder outputs collected: {len(all_vecs)}")
    lines.append(f"- Startup warmup region: t ∈ [0, {H-1}] (zero-padded history)")
    lines.append(f"- Steady-state region: t ∈ [{H}, {EPISODE_STEPS_USE-1}]")
    lines.append("")
    lines.append("**Note:** Synthetic obs (ideal reference tracking) means fault-condition")
    lines.append("separability is limited — priv_state differs but obs sequence is similar.")
    lines.append("The startup off-manifold ratio is the primary reliable metric here.")
    lines.append("")
    lines.append("## Off-Manifold Distance (startup vs steady-state)")
    lines.append("")
    lines.append("Metric: L2 distance from the steady-state centroid in 8-D encoder output space.")
    lines.append("")
    lines.append(f"| Region | Mean L2 distance from steady-state centroid |")
    lines.append(f"|---|---|")
    lines.append(f"| Steady-state (t≥{H}) | {steady_mean:.4f} |")
    lines.append(f"| Startup warmup (t<{H}) | {warmup_mean:.4f} |")
    lines.append(f"| **Off-manifold ratio** | **{off_manifold:.2f}×** |")
    lines.append("")

    if off_manifold < 1.5:
        verdict = "LOW — encoder handles zero-padded startup well; startup instability may not block M3."
    elif off_manifold < 3.0:
        verdict = "MODERATE — startup is visibly off-manifold. Fix option 2 or 4 recommended before M3."
    else:
        verdict = "HIGH — startup is severely off-manifold. Fix required before M3; risk of early obstacle collision."

    lines.append(f"**Startup severity:** {verdict}")
    lines.append("")
    lines.append("## Condition Separability")
    lines.append("")
    lines.append("From t-SNE 2D embedding (synthetic data — separability is limited):")
    lines.append("")
    nominal_tsne = tsne_xy[all_conditions == 0].mean(axis=0)
    for ci, (cname, _, _) in enumerate(conditions[1:], 1):
        mask = all_conditions == ci
        if mask.sum() > 0:
            dist = float(np.linalg.norm(tsne_xy[mask].mean(axis=0) - nominal_tsne))
            lines.append(f"- {cname}: centroid distance from nominal = {dist:.2f}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `notes/figures/m2_encoder_manifold_tsne.png` — t-SNE by condition")
    lines.append("- `notes/figures/m2_encoder_manifold_tsne_phase.png` — t-SNE startup vs steady-state")
    if umap_ok:
        lines.append("- `notes/figures/m2_encoder_manifold_umap.png` — UMAP by condition")
        lines.append("- `notes/figures/m2_encoder_manifold_umap_phase.png` — UMAP startup vs steady-state")
    else:
        lines.append("- UMAP: skipped (umap-learn not installed)")
    lines.append("")
    lines.append("## Implication for L5 Fix Choice")
    lines.append("")
    if off_manifold < 1.5:
        lines.append("Off-manifold ratio is low. Fix option 2 (train on zero-padded prefixes) "
                     "is a low-cost improvement but may not be urgent for M3 figure_eight_normal speed.")
    elif off_manifold < 3.0:
        lines.append("Off-manifold ratio is moderate. Apply fix option 2 (train on zero-padded prefixes) "
                     "before M3 deployment. Option 4 (joint fine-tuning) is stronger if ratio stays elevated.")
    else:
        lines.append("Off-manifold ratio is high. Apply fix option 4 (short joint fine-tuning) before M3. "
                     "Garbage ê_t at startup is the regime that will cause early obstacle collisions in M3.")

    ANALYSIS_PATH.write_text("\n".join(lines))
    print(f"\n  Analysis written to {ANALYSIS_PATH}")
    print()
    print(f"  Off-manifold ratio: {off_manifold:.2f}×")
    print(f"  Severity: {verdict}")
    print()


if __name__ == "__main__":
    main()
