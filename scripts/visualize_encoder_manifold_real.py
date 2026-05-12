"""
M2 encoder manifold — REAL physics rollout version.

Addresses three clarifications:
1. Uses real MJX rollouts (not synthetic ideal tracking) for both condition
   separability and the startup off-manifold ratio.
2. Reports raw off-manifold ratios with no invented threshold labels.
3. Predictive validity: compares figure_eight_normal vs figure_eight_fast+fault
   using the Phase 2 COMBINED policy (actor + encoder), which is what crashed
   in deployment.

Phase 1 policy (ground truth priv_state from env) is used for Part 1 to get
a realistic training-distribution manifold. The combined Phase 2 policy is
used for Part 2 to expose startup instability in the crash scenario.

Saves:
  notes/figures/m2_encoder_manifold_real_tsne.png
  notes/figures/m2_encoder_manifold_real_tsne_phase.png
  notes/figures/m2_encoder_manifold_real_umap.png
  notes/figures/m2_encoder_manifold_real_umap_phase.png
  notes/figures/m2_encoder_manifold_real_validity.png
  notes/M2_encoder_manifold_analysis.md   (full overwrite)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# XLA compilation cache — makes re-runs fast after first cold start.
_XLA_CACHE = str(Path.home() / ".cache" / "jax_xla")
os.makedirs(_XLA_CACHE, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _XLA_CACHE)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent

DEFAULT_ACTOR_CHECKPOINT = (
    REPO_ROOT / "experiments/m2_phase1_baseline"
    / "m2_phase1_baseline_1778244202/checkpoints/final"
)
DEFAULT_ENCODER_CHECKPOINT = REPO_ROOT / "experiments/phase2_encoder/best_checkpoint"
FIGURES_DIR = REPO_ROOT / "notes/figures"
ANALYSIS_PATH = REPO_ROOT / "notes/M2_encoder_manifold_analysis.md"

H           = 50   # ring buffer length
OBS_DIM     = 42   # encoder sees [e_W(30), v(3), R(9)] — no priv_state
ACTION_DIM  = 4
PAIR_DIM    = OBS_DIM + ACTION_DIM   # 46
WINDOW_DIM  = H * PAIR_DIM          # 2300

N_STEPS   = 200   # steps per episode
SEEDS     = [42, 99, 7]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _priv_nominal():
    return np.array([1., 1., 1., 1., 1., 0., 0., 0.], dtype=np.float32)


def _priv_rotor(idx, eta=0.70):
    p = _priv_nominal()
    p[idx] = eta
    return p


def _priv_mass(scale):
    p = _priv_nominal()
    p[4] = scale
    return p


CONDITIONS_P1 = [
    ("nominal",  _priv_nominal(),   "gray"),
    ("rotor0",   _priv_rotor(0),    "tab:red"),
    ("rotor1",   _priv_rotor(1),    "tab:orange"),
    ("rotor2",   _priv_rotor(2),    "tab:green"),
    ("rotor3",   _priv_rotor(3),    "tab:blue"),
    ("mass0.8",  _priv_mass(0.8),   "tab:purple"),
    ("mass1.2",  _priv_mass(1.2),   "tab:brown"),
]


def run_episode_p1(step_jit, reset_jit, actor_apply, actor_params,
                   encoder_apply, enc_params,
                   priv_np, seed, drone_id, traj=None):
    """
    Phase 1 policy (ground truth priv_state).
    Collects encoder outputs from real MJX rollout.
    Returns enc_outputs (N_STEPS, 8), n_crashed_at (None or int).
    step_jit / reset_jit must be jax.jit(env._step_fn / env._reset_fn).
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(seed)
    state, actor_obs, _ = reset_jit(key)

    priv_jnp = jnp.array(priv_np)
    override = dict(
        priv_state=priv_jnp,
        rotor_efficiency=priv_jnp[:4],
        mass_scale=priv_jnp[4],
        kf_multiplier=jnp.ones(()),
    )
    if traj is not None:
        override["traj"] = traj
    state = state._replace(**override)

    # Rebuild initial obs with the overridden state
    from iron_man_drone.envs.quadrotor_env import _build_obs
    actor_obs, _ = _build_obs(
        state.mjx_data, state.traj, state.step, drone_id, priv_jnp
    )

    ring_buf    = jnp.zeros((H, PAIR_DIM))
    prev_action = jnp.zeros(ACTION_DIM)
    enc_outputs = []
    crashed_at  = None

    for t in range(N_STEPS):
        obs_42 = actor_obs[:OBS_DIM]     # strip priv_state from actor obs

        # Encoder ring buffer update and forward pass
        pair_t    = jnp.concatenate([obs_42, prev_action])
        new_ring  = jnp.concatenate([ring_buf[1:], pair_t[None]], axis=0)
        window    = new_ring.reshape(1, -1)
        e_hat_n   = encoder_apply(enc_params, window)[0]
        enc_outputs.append(np.array(e_hat_n))

        # Phase 1 actor uses ground-truth priv_state (actor_obs is 50-dim)
        mean, _ = actor_apply(actor_params, actor_obs[None])
        action   = mean[0]
        prev_action = action

        # Physics step (jitted single-env)
        new_state, new_actor_obs, _, _, done = step_jit(state, action)
        if traj is not None:
            new_state = new_state._replace(traj=traj)

        ring_buf   = new_ring
        actor_obs  = new_actor_obs
        state      = new_state

        if bool(np.array(done)) and crashed_at is None:
            crashed_at = t + 1

    return np.array(enc_outputs, dtype=np.float32), crashed_at


def run_episode_p2(step_jit, reset_jit, actor_apply, actor_params,
                   encoder_apply, enc_params,
                   priv_np, seed, drone_id, traj, offset_steps=0):
    """
    Phase 2 COMBINED policy (encoder estimate replaces priv_state for actor).
    Used for predictive validity — this is the policy that crashed in deployment.
    Returns enc_outputs (≤N_STEPS, 8), crashed_at (None or int).
    step_jit / reset_jit must be jax.jit(env._step_fn / env._reset_fn).
    """
    import jax
    import jax.numpy as jnp
    from iron_man_drone.policy.encoder import denormalize_e_hat
    from iron_man_drone.envs.quadrotor_env import _build_obs

    key       = jax.random.PRNGKey(seed)
    state, _, _ = reset_jit(key)

    priv_jnp = jnp.array(priv_np)
    state = state._replace(
        traj=traj,
        priv_state=priv_jnp,
        rotor_efficiency=priv_jnp[:4],
        mass_scale=priv_jnp[4],
        kf_multiplier=jnp.ones(()),
        step=jnp.int32(offset_steps),
    )
    actor_obs, _ = _build_obs(
        state.mjx_data, traj, state.step, drone_id, priv_jnp
    )

    ring_buf    = jnp.zeros((H, PAIR_DIM))
    prev_action = jnp.zeros(ACTION_DIM)
    enc_outputs = []
    crashed_at  = None

    for t in range(N_STEPS):
        obs_42 = actor_obs[:OBS_DIM]

        # Encoder ring buffer update and forward pass
        pair_t   = jnp.concatenate([obs_42, prev_action])
        new_ring = jnp.concatenate([ring_buf[1:], pair_t[None]], axis=0)
        window   = new_ring.reshape(1, -1)
        e_hat_n  = encoder_apply(enc_params, window)[0]
        enc_outputs.append(np.array(e_hat_n))

        # Phase 2 actor uses encoder estimate, NOT ground truth priv_state
        e_hat_raw = denormalize_e_hat(e_hat_n)
        actor_input = jnp.concatenate([obs_42, e_hat_raw])[None]
        mean, _ = actor_apply(actor_params, actor_input)
        action   = mean[0]
        prev_action = action

        # Physics step (jitted single-env)
        new_state, new_actor_obs, _, _, done = step_jit(state, action)
        new_state = new_state._replace(traj=traj)

        ring_buf   = new_ring
        actor_obs  = new_actor_obs
        state      = new_state

        if bool(np.array(done)) and crashed_at is None:
            crashed_at = t + 1
            break   # stop collection after crash — no meaningful post-crash data

    return np.array(enc_outputs, dtype=np.float32), crashed_at


def main():
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    print(f"\n{'='*60}")
    print(" M2 Encoder Manifold — Real Physics Rollout")
    print(f"{'='*60}\n")
    print(f"  JAX devices: {jax.devices()}\n")

    # ── Load actor ────────────────────────────────────────────────────────────
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    ppo_cfg = PPOConfig(actor_obs_dim=50, critic_obs_dim=51,
                        action_dim=4, hidden_dim=256, num_layers=3)
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)
    checkpointer = ocp.PyTreeCheckpointer()
    restored     = checkpointer.restore(str(DEFAULT_ACTOR_CHECKPOINT))
    actor_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored["actor"]["params"])
    actor_state  = actor_state.replace(params=actor_params)
    print("  Actor loaded (Phase 1, 50-dim).")

    @jax.jit
    def actor_apply(params, obs):
        return actor_state.apply_fn(params, obs)

    # ── Load encoder ──────────────────────────────────────────────────────────
    from iron_man_drone.policy.encoder import AdaptationEncoder
    encoder      = AdaptationEncoder()
    restored_e   = checkpointer.restore(str(DEFAULT_ENCODER_CHECKPOINT))
    enc_params   = jax.tree_util.tree_map(
        lambda x: jnp.array(x), restored_e["params"])
    print("  Encoder loaded (Phase 2).")

    @jax.jit
    def encoder_apply(params, window):
        return encoder.apply(params, window)

    # JIT warmup (actor + encoder only — no env step yet)
    _ = actor_apply(actor_params, jnp.zeros((1, 50)))
    _ = encoder_apply(enc_params, jnp.zeros((1, WINDOW_DIM)))
    jax.block_until_ready(_)
    print("  Actor + encoder JIT warmed up.\n")

    # ── Create VecEnv (num_envs=1) ────────────────────────────────────────────
    from iron_man_drone.envs.quadrotor_env import VecEnv, EPISODE_STEPS, DT
    from iron_man_drone.policy.ppo import PPOConfig as _PC

    env_cfg = _PC(num_envs=1, actor_obs_dim=50, critic_obs_dim=51,
                  action_dim=4, hidden_dim=256, num_layers=3)
    print("  Creating VecEnv (num_envs=1, fault_prob=0.7)...")
    env = VecEnv(env_cfg, fault_prob=0.7, eta_min=0.5, mass_lo=0.8, mass_hi=1.2)
    print("  VecEnv created.\n")

    # Warm up env._step_fn — triggers MJX CUDA kernel compilation.
    # JIT the single-env functions — env.step/env.reset are vmap'd over num_envs
    # and expect batched inputs; step_jit/reset_jit work with single states.
    step_jit  = jax.jit(env._step_fn)
    reset_jit = jax.jit(env._reset_fn)
    drone_id  = env.mj_model.body("drone").id

    print("  Warming up env step JIT (first call compiles MJX kernels)...")
    t_wup = time.time()
    _state0, _, _ = reset_jit(jax.random.PRNGKey(0))
    _r = step_jit(_state0, jnp.zeros(4))
    jax.block_until_ready(_r)
    print(f"  Env step JIT warmed up in {time.time()-t_wup:.1f}s.\n")

    # ── Trajectories for Part 2 ───────────────────────────────────────────────
    from iron_man_drone.envs.trajectories import make_figure_eight_trajectory
    from iron_man_drone.envs.quadrotor_env import LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
    from iron_man_drone.evaluation.eval_suite import F8_OFFSETS

    lookahead = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS

    def _make_f8(speed):
        off   = F8_OFFSETS[speed]
        total = EPISODE_STEPS + off + lookahead + 5
        traj  = make_figure_eight_trajectory(DT, total, lookahead, speed=speed)
        return traj, off

    print("  Building figure_eight trajectories...")
    f8_normal_traj, f8_normal_off = _make_f8("normal")
    f8_fast_traj,   f8_fast_off   = _make_f8("fast")
    print("  Trajectories ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1 — Training distribution (Phase 1 policy, poly/zigzag from reset)
    # ──────────────────────────────────────────────────────────────────────────
    print("  === Part 1: training distribution (Phase 1 policy) ===")
    all_vecs       = []
    all_conditions = []
    all_timesteps  = []

    for ci, (cname, priv_np, _color) in enumerate(CONDITIONS_P1):
        for seed in SEEDS:
            t0 = time.time()
            enc, crashed = run_episode_p1(
                step_jit, reset_jit, actor_apply, actor_params,
                encoder_apply, enc_params,
                priv_np, seed, drone_id,
            )
            T = len(enc)
            all_vecs.extend(enc)
            all_conditions.extend([ci] * T)
            all_timesteps.extend(range(T))
            c_str = f" (crashed@{crashed})" if crashed else ""
            print(f"    {cname} seed={seed}: {T} steps  "
                  f"({time.time()-t0:.1f}s){c_str}")

    all_vecs       = np.array(all_vecs, dtype=np.float32)
    all_conditions = np.array(all_conditions, dtype=np.int32)
    all_timesteps  = np.array(all_timesteps, dtype=np.int32)
    warmup_mask    = all_timesteps < H

    n_steady = (~warmup_mask).sum()
    if n_steady == 0:
        print("  ERROR: no steady-state steps collected")
        return

    steady_centroid = all_vecs[~warmup_mask].mean(axis=0)  # (8,)
    dist_steady = np.linalg.norm(all_vecs[~warmup_mask] - steady_centroid, axis=1)
    dist_startup = np.linalg.norm(all_vecs[warmup_mask]  - steady_centroid, axis=1)

    mean_dist_steady  = dist_steady.mean()
    mean_dist_startup = dist_startup.mean()
    ratio_p1 = mean_dist_startup / mean_dist_steady if mean_dist_steady > 0 else float("nan")

    print(f"\n  Part 1 summary:")
    print(f"    Total outputs: {len(all_vecs)}")
    print(f"    Startup (t<{H}): {warmup_mask.sum()}  "
          f"Steady-state: {n_steady}")
    print(f"    Mean L2 steady-state:   {mean_dist_steady:.4f}")
    print(f"    Mean L2 startup:        {mean_dist_startup:.4f}")
    print(f"    Off-manifold ratio:     {ratio_p1:.2f}×\n")

    # Condition separability (centroid distances from nominal steady-state)
    nom_mask    = (all_conditions == 0) & ~warmup_mask
    nom_centroid = all_vecs[nom_mask].mean(axis=0) if nom_mask.sum() > 0 else steady_centroid

    sep_stats = {}
    for ci, (cname, _, _) in enumerate(CONDITIONS_P1):
        mask = (all_conditions == ci) & ~warmup_mask
        if mask.sum() == 0:
            sep_stats[cname] = float("nan")
        else:
            sep_stats[cname] = float(np.linalg.norm(
                all_vecs[mask].mean(axis=0) - nom_centroid))

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2 — Predictive validity (Phase 2 combined policy)
    # ──────────────────────────────────────────────────────────────────────────
    print("  === Part 2: predictive validity (Phase 2 combined policy) ===")

    p2_configs = [
        ("figure_eight_normal / nominal",      f8_normal_traj, f8_normal_off, _priv_nominal()),
        ("figure_eight_normal / fault_eta70",  f8_normal_traj, f8_normal_off, _priv_rotor(0, 0.70)),
        ("figure_eight_fast   / fault_eta70",  f8_fast_traj,   f8_fast_off,   _priv_rotor(0, 0.70)),
    ]

    validity_results = {}
    for label, traj, off, priv_np in p2_configs:
        ep_ratios  = []
        ep_crashes = []
        ep_n_steady = []
        for seed in SEEDS:
            t0  = time.time()
            enc, crashed = run_episode_p2(
                step_jit, reset_jit, actor_apply, actor_params,
                encoder_apply, enc_params,
                priv_np, seed, drone_id, traj, offset_steps=off,
            )
            T = len(enc)
            su_mask = np.arange(T) < H
            st_mask = ~su_mask

            d_all    = np.linalg.norm(enc - steady_centroid, axis=1)
            d_su     = d_all[su_mask].mean() if su_mask.sum() > 0 else float("nan")
            d_st     = d_all[st_mask].mean() if st_mask.sum() > 0 else float("nan")
            ratio    = d_su / d_st if (not np.isnan(d_st) and d_st > 0) else float("nan")
            ep_ratios.append(ratio)
            ep_crashes.append(crashed)
            ep_n_steady.append(int(st_mask.sum()))

            c_str = f" crashed@{crashed}" if crashed else " no crash"
            print(f"    {label}  seed={seed}: {T} steps  "
                  f"ratio={ratio:.2f}×  {c_str}  ({time.time()-t0:.1f}s)")

        validity_results[label] = {
            "ratios":   ep_ratios,
            "crashes":  ep_crashes,
            "n_steady": ep_n_steady,
        }
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # DIMENSIONALITY REDUCTION (Part 1 data only)
    # ──────────────────────────────────────────────────────────────────────────
    print("  Running t-SNE (perplexity=30)...")
    from sklearn.manifold import TSNE
    t0 = time.time()
    tsne_xy = TSNE(n_components=2, perplexity=30, random_state=42,
                   max_iter=1000).fit_transform(all_vecs)
    print(f"  t-SNE done in {time.time()-t0:.1f}s")

    print("  Running UMAP (n_neighbors=15)...")
    try:
        import umap
        t0 = time.time()
        umap_xy = umap.UMAP(n_components=2, n_neighbors=15,
                             random_state=42).fit_transform(all_vecs)
        print(f"  UMAP done in {time.time()-t0:.1f}s")
        umap_ok = True
    except ImportError:
        print("  umap-learn not found — skipping UMAP")
        umap_xy = None
        umap_ok = False

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURES
    # ──────────────────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [c for _, _, c in CONDITIONS_P1]
    labels = [n for n, _, _ in CONDITIONS_P1]

    def _fig_condition(emb, title, path):
        fig, ax = plt.subplots(figsize=(9, 7))
        for ci, (cname, _, color) in enumerate(CONDITIONS_P1):
            m = all_conditions == ci
            ax.scatter(emb[m, 0], emb[m, 1],
                       c=color, label=cname, s=4, alpha=0.4, linewidths=0)
        ax.set_title(title, fontsize=13)
        ax.legend(markerscale=4, fontsize=9)
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Saved: {path}")

    def _fig_phase(emb, title, path):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(emb[~warmup_mask, 0], emb[~warmup_mask, 1],
                   c="steelblue", label=f"steady-state (t≥{H})",
                   s=4, alpha=0.3, linewidths=0)
        ax.scatter(emb[warmup_mask, 0], emb[warmup_mask, 1],
                   c="tomato", label=f"startup (t<{H})",
                   s=8, alpha=0.6, linewidths=0)
        ax.set_title(title, fontsize=13)
        ax.legend(markerscale=4, fontsize=9)
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Saved: {path}")

    _fig_condition(tsne_xy,
                   "M2 Encoder — t-SNE by condition (real rollouts, Phase 1 policy)",
                   FIGURES_DIR / "m2_encoder_manifold_real_tsne.png")
    _fig_phase(tsne_xy,
               "M2 Encoder — t-SNE startup vs steady-state (real rollouts)",
               FIGURES_DIR / "m2_encoder_manifold_real_tsne_phase.png")

    if umap_ok:
        _fig_condition(umap_xy,
                       "M2 Encoder — UMAP by condition (real rollouts, Phase 1 policy)",
                       FIGURES_DIR / "m2_encoder_manifold_real_umap.png")
        _fig_phase(umap_xy,
                   "M2 Encoder — UMAP startup vs steady-state (real rollouts)",
                   FIGURES_DIR / "m2_encoder_manifold_real_umap_phase.png")

    # Validity figure: per-condition off-manifold ratio bar chart
    v_labels = [lbl.replace("figure_eight_", "f8_") for lbl in validity_results]
    v_means  = [np.nanmean(r["ratios"]) for r in validity_results.values()]
    v_stds   = [np.nanstd(r["ratios"])  for r in validity_results.values()]
    v_colors = ["steelblue", "darkorange", "crimson"]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(v_labels))
    ax.bar(xs, v_means, yerr=v_stds, color=v_colors, alpha=0.8,
           capsize=6, width=0.5)
    ax.axhline(ratio_p1, color="k", linestyle="--", linewidth=1,
               label=f"Part 1 training-dist ratio ({ratio_p1:.2f}×)")
    ax.set_xticks(xs); ax.set_xticklabels(v_labels, fontsize=10)
    ax.set_ylabel("startup off-manifold ratio (startup / steady-state)")
    ax.set_title("Predictive validity: does off-manifold ratio predict crash?",
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "m2_encoder_manifold_real_validity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'm2_encoder_manifold_real_validity.png'}")

    # ──────────────────────────────────────────────────────────────────────────
    # ANALYSIS MARKDOWN
    # ──────────────────────────────────────────────────────────────────────────
    now = "2026-05-11"
    crash_summary = {}
    for label, res in validity_results.items():
        n_crash  = sum(1 for c in res["crashes"] if c is not None)
        crash_at = [c for c in res["crashes"] if c is not None]
        mean_ratio = np.nanmean(res["ratios"])
        crash_summary[label] = {
            "n_crash": n_crash, "crash_at": crash_at,
            "mean_ratio": mean_ratio,
        }

    def _fmt_ratio(r):
        return f"{r:.2f}×" if not np.isnan(r) else "N/A (insufficient steady-state)"

    lines = [
        "# M2 Encoder Output Manifold Analysis",
        "",
        f"**Date:** {now}  ",
        "**Method:** Real MJX physics rollouts.  ",
        "**Policy:** Phase 1 (ground-truth priv_state) for Part 1; Phase 2 combined "
        "(actor + encoder) for Part 2 predictive validity.",
        "",
        "## Clarification notes",
        "",
        "1. **Real vs synthetic**: This analysis uses actual MJX physics rollouts, not "
        "synthetic ideal-tracking obs. The Phase 1 policy runs under real DR (fault, mass, k_f) "
        "and the drone's actual tracking errors appear in e_W.",
        "",
        "2. **No threshold labels**: Off-manifold ratios are reported as raw numbers. "
        "The 1.5×/3.0× thresholds from the earlier synthetic analysis had no published "
        "justification and have been removed.",
        "",
        "3. **Predictive validity**: Part 2 runs the Phase 2 combined policy (the one that "
        "actually crashed in deployment) on three trajectory+condition combinations.",
        "",
        "## Part 1 — Training distribution manifold (Phase 1 policy)",
        "",
        "Trajectory: polynomial/zigzag (from reset_fn, matching Phase 1 training distribution).  ",
        "Conditions: 7 (nominal + 4 rotor faults η=0.70 + mass 0.8 + mass 1.2).  ",
        f"Seeds: {SEEDS}.",
        "",
        "### Off-manifold distance (startup vs steady-state)",
        "",
        "| Region | Mean L2 from steady-state centroid |",
        "|---|---|",
        f"| Steady-state (t≥{H}) | {mean_dist_steady:.4f} |",
        f"| Startup warmup (t<{H}) | {mean_dist_startup:.4f} |",
        f"| **Off-manifold ratio** | **{ratio_p1:.2f}×** |",
        "",
        "### Condition separability (centroid distance from nominal, steady-state only)",
        "",
        "| Condition | Centroid distance from nominal |",
        "|---|---|",
    ]
    for cname, d in sep_stats.items():
        lines.append(f"| {cname} | {d:.3f} |")

    lines += [
        "",
        "Separability reflects real trajectory tracking errors under each fault condition, "
        "unlike the synthetic analysis where all conditions had identical e_W (ideal tracking).",
        "",
        "## Part 2 — Predictive validity (Phase 2 combined policy)",
        "",
        "Phase 2 combined policy: actor sees encoder estimate ê_t instead of ground-truth "
        "priv_state. Startup instability (first H=50 steps with zero-padded ring buffer) "
        "may produce wrong ê_t → actor cannot compensate fault → crash.",
        "",
        "| Trajectory / condition | Off-manifold ratio (mean±std) | Crashes (of 3 seeds) |",
        "|---|---|---|",
    ]
    for label, res in validity_results.items():
        mean_r = np.nanmean(res["ratios"])
        std_r  = np.nanstd(res["ratios"])
        n_c    = sum(1 for c in res["crashes"] if c is not None)
        crash_steps = [str(c) for c in res["crashes"] if c is not None]
        c_str  = f"{n_c}/3" + (f" (steps {', '.join(crash_steps)})" if crash_steps else "")
        lines.append(f"| {label} | {_fmt_ratio(mean_r)} ± {std_r:.2f} | {c_str} |")

    normal_nominal_ratio = np.nanmean(validity_results[
        "figure_eight_normal / nominal"]["ratios"])
    fast_fault_ratio = np.nanmean(validity_results[
        "figure_eight_fast   / fault_eta70"]["ratios"])
    ratio_delta = fast_fault_ratio - normal_nominal_ratio

    lines += [
        "",
        "### Interpretation",
        "",
        f"- figure_eight_normal / nominal off-manifold ratio: **{_fmt_ratio(normal_nominal_ratio)}**  ",
        f"- figure_eight_fast / fault_eta70 off-manifold ratio: **{_fmt_ratio(fast_fault_ratio)}**  ",
        f"- Difference: **{ratio_delta:+.2f}×**  ",
        "",
    ]

    if np.isnan(fast_fault_ratio) or np.isnan(normal_nominal_ratio):
        interp = ("Insufficient data (crashes too early for steady-state measurement). "
                  "Off-manifold ratio cannot be computed for the crash scenario — "
                  "the crash happened before H=50 steps of steady-state were available. "
                  "This itself indicates severe startup failure.")
    elif ratio_delta > 0.5:
        interp = (
            f"figure_eight_fast+fault has meaningfully higher off-manifold ratio "
            f"({_fmt_ratio(fast_fault_ratio)}) than figure_eight_normal/nominal "
            f"({_fmt_ratio(normal_nominal_ratio)}). Off-manifold distance is a valid "
            f"predictor of deployment failures. **Fix Option 2 targets the right mechanism.**"
        )
    elif ratio_delta < 0.2:
        interp = (
            f"Both trajectories have similar off-manifold ratios "
            f"(Δ = {ratio_delta:+.2f}×). Off-manifold distance does NOT clearly predict "
            f"the crash. Startup encoder behavior may not be the root cause — "
            f"consider whether the figure_eight_fast+fault task is simply beyond "
            f"the Phase 1 policy's control authority, independent of encoder quality. "
            f"**Fix Option 2 may not be sufficient; investigate task difficulty first.**"
        )
    else:
        interp = (
            f"Marginal difference (Δ = {ratio_delta:+.2f}×). Off-manifold distance is "
            f"a weak predictor. Fix Option 2 may help but is not clearly sufficient alone."
        )

    lines += [
        interp,
        "",
        "## Figures",
        "",
        "- `notes/figures/m2_encoder_manifold_real_tsne.png` — t-SNE by condition (real rollouts)",
        "- `notes/figures/m2_encoder_manifold_real_tsne_phase.png` — t-SNE startup vs steady-state",
        "- `notes/figures/m2_encoder_manifold_real_umap.png` — UMAP by condition" + (" (real rollouts)" if umap_ok else " (skipped)"),
        "- `notes/figures/m2_encoder_manifold_real_umap_phase.png` — UMAP phase" + ("" if umap_ok else " (skipped)"),
        "- `notes/figures/m2_encoder_manifold_real_validity.png` — predictive validity bar chart",
        "",
        "*(Synthetic figures preserved at `m2_encoder_manifold_tsne.png` etc.)*",
    ]

    ANALYSIS_PATH.write_text("\n".join(lines))
    print(f"\n  Analysis written to {ANALYSIS_PATH}")
    print(f"\n  Part 1 off-manifold ratio: {ratio_p1:.2f}×")
    for label, res in validity_results.items():
        mean_r = np.nanmean(res["ratios"])
        n_c    = sum(1 for c in res["crashes"] if c is not None)
        print(f"  {label}: ratio={_fmt_ratio(mean_r)}  crashes={n_c}/3")
    print()


if __name__ == "__main__":
    main()
