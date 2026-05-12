"""
M2 encoder output manifold visualization — 2026-05-11.

Per Jin Zhou's suggestion (MAVEN author, personal comm.): visualize the Phase 2
encoder's output manifold to quantify startup instability.

Procedure:
  1. Load Phase 1 actor + Phase 2 encoder.
  2. Roll out the figure_eight_normal trajectory under 7 conditions:
       nominal, rotor0/1/2/3 fault (η=0.7), mass=0.8, mass=1.2
  3. At every timestep, record the encoder's 8-dim output (ê_t_norm).
  4. Apply t-SNE (perplexity=30) and UMAP (n_neighbors=15) on the full set.
  5. Figure 1: colored by condition (fault rotor / mass).
     Figure 2: colored by phase (t<50 = startup warmup, t≥50 = steady-state).

Saves:
  notes/figures/m2_encoder_manifold_tsne.png
  notes/figures/m2_encoder_manifold_umap.png

Writes:
  notes/M2_encoder_manifold_analysis.md

Usage:
  python scripts/visualize_encoder_manifold.py
"""

import sys
import time
from pathlib import Path

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
MAX_STEPS_PER_EPISODE = 200   # 50 warmup + 150 steady-state is enough

SEEDS = [42, 99, 7]


def collect_episode_encodings(
    actor_apply, actor_params, encoder_apply, enc_params,
    env, drone_id, cos_max_tilt,
    priv_state_override, traj, ref_xy, offset_steps,
    seed
):
    """Run one episode and collect (timestep, enc_output) at every step."""
    import jax
    import jax.numpy as jnp
    from iron_man_drone.envs.quadrotor_env import (
        EPISODE_STEPS, _build_obs, MIN_HEIGHT, MAX_HEIGHT_ABOVE_REF, MAX_TILT_RAD,
    )
    from iron_man_drone.policy.encoder import denormalize_e_hat

    OBS_DIM_LOCAL = OBS_DIM

    state, _, _ = env._reset_fn(jax.random.PRNGKey(seed))
    state = state._replace(
        traj=traj,
        priv_state=priv_state_override,
        rotor_efficiency=priv_state_override[:4],
        mass_scale=priv_state_override[4],
        kf_multiplier=jnp.ones(()),
        step=jnp.int32(offset_steps),
    )
    full_obs, _ = _build_obs(
        state.mjx_data, traj, state.step, drone_id, priv_state_override
    )
    obs_base    = full_obs[:OBS_DIM_LOCAL]
    ring_buf    = jnp.zeros((H, PAIR_DIM))
    prev_action = jnp.zeros(ACTION_DIM)

    import numpy as np
    enc_outputs = []  # list of (8,) numpy arrays — convert immediately to free GPU memory

    for t in range(min(EPISODE_STEPS, MAX_STEPS_PER_EPISODE)):
        pair_t   = jnp.concatenate([obs_base, prev_action])
        new_ring = jnp.concatenate([ring_buf[1:], pair_t[None]], axis=0)
        window   = new_ring.reshape(1, -1)

        e_hat_n   = encoder_apply(enc_params, window)[0]
        enc_outputs.append(np.array(e_hat_n))   # convert immediately — frees GPU buffer

        e_hat_raw = denormalize_e_hat(e_hat_n)
        actor_obs = jnp.concatenate([obs_base, e_hat_raw])[None]
        mean, _   = actor_apply(actor_params, actor_obs)
        action    = mean[0]

        new_state, new_full_obs, _, _, _ = env._step_fn(state, action)
        new_state = new_state._replace(traj=traj)

        # crash check
        pos      = new_state.mjx_data.xpos[drone_id]
        body_z_z = new_state.mjx_data.xmat[drone_id].reshape(-1)[8]
        crashed  = (
            (float(pos[2]) < float(MIN_HEIGHT))
            or (float(jnp.abs(pos[2] - 1.0)) > float(MAX_HEIGHT_ABOVE_REF))
            or (float(body_z_z) < cos_max_tilt)
        )
        if crashed:
            break

        state       = new_state
        obs_base    = new_full_obs[:OBS_DIM_LOCAL]
        ring_buf    = new_ring
        prev_action = action

    return np.array(enc_outputs, dtype=np.float32)


def main():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    print(f"\n{'='*60}")
    print(" M2 Encoder Manifold Visualization")
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

    # JIT warmup — include env step so compilation doesn't hit inside the timed loop
    _ = actor_apply(actor_params, jnp.zeros((1, 50)))
    _ = encoder_apply(enc_params, jnp.zeros((1, WINDOW_DIM)))
    print("  Actor + encoder JIT warmed up.")

    # ── Environment ───────────────────────────────────────────────────────────
    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, DT, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS, MAX_TILT_RAD,
    )

    class _Cfg:
        num_envs = 1

    env      = VecEnv(_Cfg(), fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)
    drone_id = env.mj_model.body("drone").id
    cos_max_tilt = float(jnp.cos(MAX_TILT_RAD))

    # ── Trajectory (create BEFORE warmup — shapes must match) ─────────────────
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, get_reference_pos,
    )
    from iron_man_drone.evaluation.eval_suite import F8_OFFSETS

    lookahead  = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    offset     = F8_OFFSETS["normal"]
    traj       = make_figure_eight_trajectory(
        DT, EPISODE_STEPS + offset + lookahead + 5, lookahead, speed="normal"
    )
    ref_xy = None  # passed to collect_episode_encodings but not used inside

    # ── Warm up env JIT with the SAME trajectory shape used in episodes ────────
    # Critical: if warmup uses a different traj shape, a second (slow) recompilation
    # happens on the first real episode call. Always warmup with the actual traj.
    print("  Warming up env step JIT (may take a few minutes)...")
    _state0, _, _ = env._reset_fn(jax.random.PRNGKey(0))
    _state_w = _state0._replace(traj=traj, step=jnp.int32(0))
    _warmup_result = env._step_fn(_state_w, jnp.zeros(4))
    jax.block_until_ready(_warmup_result)
    print("  Env step JIT warmed up.\n")

    # ── Conditions ────────────────────────────────────────────────────────────
    # priv_state = [η0, η1, η2, η3, mass_scale, Fx, Fy, Fz]
    nominal = jnp.array([1., 1., 1., 1., 1., 0., 0., 0.])
    conditions = [
        ("nominal",  nominal, "gray"),
        ("rotor0",   jnp.array([0.7, 1., 1., 1., 1., 0., 0., 0.]), "tab:red"),
        ("rotor1",   jnp.array([1., 0.7, 1., 1., 1., 0., 0., 0.]), "tab:orange"),
        ("rotor2",   jnp.array([1., 1., 0.7, 1., 1., 0., 0., 0.]), "tab:green"),
        ("rotor3",   jnp.array([1., 1., 1., 0.7, 1., 0., 0., 0.]), "tab:blue"),
        ("mass0.8",  jnp.array([1., 1., 1., 1., 0.8, 0., 0., 0.]), "tab:purple"),
        ("mass1.2",  jnp.array([1., 1., 1., 1., 1.2, 0., 0., 0.]), "tab:brown"),
    ]

    # ── Collect data ──────────────────────────────────────────────────────────
    print("  Collecting encoder outputs...")
    all_vecs       = []   # (N, 8) float32
    all_conditions = []   # (N,) int — condition index
    all_timesteps  = []   # (N,) int

    for ci, (cname, priv_state, _color) in enumerate(conditions):
        for seed in SEEDS:
            t0  = time.time()
            enc = collect_episode_encodings(
                actor_apply, actor_params, encoder_apply, enc_params,
                env, drone_id, cos_max_tilt,
                priv_state, traj, ref_xy, offset, seed,
            )
            T = len(enc)
            all_vecs.extend(enc)
            all_conditions.extend([ci] * T)
            all_timesteps.extend(range(T))
            print(f"    {cname} seed={seed}: {T} steps  ({time.time()-t0:.1f}s)")

    all_vecs       = np.array(all_vecs, dtype=np.float32)
    all_conditions = np.array(all_conditions, dtype=np.int32)
    all_timesteps  = np.array(all_timesteps, dtype=np.int32)
    warmup_mask    = all_timesteps < H   # t < 50 = startup warmup
    print(f"\n  Total encoder outputs: {len(all_vecs)}")
    print(f"  Startup (t<50): {warmup_mask.sum()}  Steady-state: {(~warmup_mask).sum()}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("\n  Running t-SNE (perplexity=30)...")
    from sklearn.manifold import TSNE
    t0 = time.time()
    tsne  = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_xy = tsne.fit_transform(all_vecs)
    print(f"  t-SNE done in {time.time()-t0:.1f}s")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("  Running UMAP (n_neighbors=15)...")
    try:
        import umap
        t0 = time.time()
        reducer  = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
        umap_xy  = reducer.fit_transform(all_vecs)
        print(f"  UMAP done in {time.time()-t0:.1f}s")
        umap_ok = True
    except ImportError:
        print("  umap-learn not installed — skipping UMAP (pip install umap-learn)")
        umap_ok  = False
        umap_xy  = None

    # ── Figures ───────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fig_condition(emb, title, save_path):
        fig, ax = plt.subplots(figsize=(9, 7))
        for ci, (cname, _, color) in enumerate(conditions):
            mask = all_conditions == ci
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=color, label=cname, s=4, alpha=0.4, linewidths=0,
            )
        ax.set_title(title, fontsize=13)
        ax.legend(markerscale=4, fontsize=9)
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    def _fig_phase(emb, title, save_path):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            emb[~warmup_mask, 0], emb[~warmup_mask, 1],
            c="tab:blue", label=f"steady-state (t≥{H})", s=4, alpha=0.3, linewidths=0,
        )
        ax.scatter(
            emb[warmup_mask, 0], emb[warmup_mask, 1],
            c="tab:red", label=f"startup warmup (t<{H})", s=10, alpha=0.7, linewidths=0,
        )
        ax.set_title(title, fontsize=13)
        ax.legend(markerscale=4, fontsize=9)
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print()
    _fig_condition(
        tsne_xy,
        "t-SNE of encoder outputs — by fault condition",
        FIGURES_DIR / "m2_encoder_manifold_tsne.png",
    )
    _fig_phase(
        tsne_xy,
        "t-SNE of encoder outputs — startup (red) vs steady-state (blue)",
        FIGURES_DIR / "m2_encoder_manifold_tsne_phase.png",
    )

    if umap_ok:
        _fig_condition(
            umap_xy,
            "UMAP of encoder outputs — by fault condition",
            FIGURES_DIR / "m2_encoder_manifold_umap.png",
        )
        _fig_phase(
            umap_xy,
            "UMAP of encoder outputs — startup (red) vs steady-state (blue)",
            FIGURES_DIR / "m2_encoder_manifold_umap_phase.png",
        )

    # ── Quantify off-manifold distance ───────────────────────────────────────
    # Measure: mean L2 distance of startup points from steady-state centroid in 8-D
    steady_vecs  = all_vecs[~warmup_mask]
    warmup_vecs  = all_vecs[warmup_mask]
    centroid     = steady_vecs.mean(axis=0)
    steady_dists = np.linalg.norm(steady_vecs - centroid, axis=1)
    warmup_dists = np.linalg.norm(warmup_vecs - centroid, axis=1)
    steady_mean  = float(steady_dists.mean())
    warmup_mean  = float(warmup_dists.mean())
    off_manifold = warmup_mean / steady_mean if steady_mean > 0 else float("nan")

    # Per-condition cluster analysis in t-SNE space
    cond_centroids = []
    for ci, (cname, _, _) in enumerate(conditions):
        mask = all_conditions == ci
        if mask.sum() > 0:
            cent = tsne_xy[mask].mean(axis=0)
            cond_centroids.append((cname, cent))

    # ── Write analysis ────────────────────────────────────────────────────────
    lines = ["# M2 Encoder Output Manifold Analysis", ""]
    lines.append("**Date:** 2026-05-11")
    lines.append(f"**Diagnostic:** Jin Zhou (MAVEN author) recommendation — t-SNE/UMAP of encoder outputs.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Trajectory: figure_eight_normal (T=5.5s, T/4 offset)")
    lines.append(f"- Conditions: {', '.join(c[0] for c in conditions)}")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Total encoder outputs collected: {len(all_vecs)}")
    lines.append(f"- Startup warmup region: t ∈ [0, {H-1}] (zero-padded history)")
    lines.append(f"- Steady-state region: t ∈ [{H}, episode_end]")
    lines.append("")
    lines.append("## Off-Manifold Distance (startup vs steady-state)")
    lines.append("")
    lines.append(f"Metric: L2 distance from the steady-state centroid in 8-D encoder output space.")
    lines.append("")
    lines.append(f"| Region | Mean L2 distance from steady-state centroid |")
    lines.append(f"|---|---|")
    lines.append(f"| Steady-state (t≥{H}) | {steady_mean:.4f} |")
    lines.append(f"| Startup warmup (t<{H}) | {warmup_mean:.4f} |")
    lines.append(f"| **Off-manifold ratio** | **{off_manifold:.2f}×** |")
    lines.append("")

    if off_manifold < 1.5:
        startup_verdict = "LOW — encoder handles zero-padded startup well; fix may not be urgent for M3."
    elif off_manifold < 3.0:
        startup_verdict = "MODERATE — startup is visibly off-manifold. Fix option 2 or 4 recommended before M3 deployment."
    else:
        startup_verdict = "HIGH — startup is severely off-manifold. Fix required before M3; startup instability is dangerous near obstacles."

    lines.append(f"**Startup severity:** {startup_verdict}")
    lines.append("")
    lines.append("## Condition Separability")
    lines.append("")
    lines.append("From t-SNE 2D embedding:")
    lines.append("")

    nominal_tsne = tsne_xy[all_conditions == 0].mean(axis=0)
    for ci, (cname, _, _) in enumerate(conditions[1:], 1):
        mask = all_conditions == ci
        if mask.sum() > 0:
            dist = float(np.linalg.norm(tsne_xy[mask].mean(axis=0) - nominal_tsne))
            lines.append(f"- {cname}: centroid distance from nominal = {dist:.2f}")
    lines.append("")
    lines.append("(Larger centroid distance = more separable from nominal = encoder encodes this condition more distinctly.)")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `notes/figures/m2_encoder_manifold_tsne.png` — t-SNE by condition")
    lines.append("- `notes/figures/m2_encoder_manifold_tsne_phase.png` — t-SNE startup (red) vs steady-state (blue)")
    if umap_ok:
        lines.append("- `notes/figures/m2_encoder_manifold_umap.png` — UMAP by condition")
        lines.append("- `notes/figures/m2_encoder_manifold_umap_phase.png` — UMAP startup vs steady-state")
    else:
        lines.append("- UMAP: skipped (umap-learn not installed)")
    lines.append("")
    lines.append("## Implication for L5 Fix Choice")
    lines.append("")
    if off_manifold < 1.5:
        lines.append("Off-manifold ratio is low. Fix option 2 (train on true zero-padded prefixes) "
                     "is a low-cost one-line improvement, but the issue may not block M3 on figure_eight_normal speed. "
                     "Proceed with M3 and monitor for startup crashes.")
    elif off_manifold < 3.0:
        lines.append("Off-manifold ratio is moderate. Apply fix option 2 (train on zero-padded prefixes) "
                     "before M3 deployment to close the gap. Option 4 (joint fine-tuning) is the strongest fix "
                     "if off-manifold ratio remains elevated after retraining.")
    else:
        lines.append("Off-manifold ratio is high. Apply fix option 4 (short joint fine-tuning) before M3. "
                     "The policy has no learned robustness to garbage ê_t during startup, and this is "
                     "exactly the regime that will cause early collisions in cluttered M3 environments.")

    ANALYSIS_PATH.write_text("\n".join(lines))
    print(f"\n  Analysis written to {ANALYSIS_PATH}")
    print()
    print(f"  Off-manifold ratio: {off_manifold:.2f}×  ({startup_verdict})")
    print()


if __name__ == "__main__":
    main()
