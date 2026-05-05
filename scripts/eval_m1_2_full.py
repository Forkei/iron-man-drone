"""
Full evaluation of an M1 checkpoint.

Runs all 7 benchmark trajectories + apex/straight error decomposition on
figure_eight_normal.

Usage:
  python scripts/eval_m1_2_full.py --checkpoint PATH [--num_episodes 3]
  python scripts/eval_m1_2_full.py \\
      --checkpoint experiments/m1_2_entropy/m1_2_entropy_1777808012/checkpoints/epoch_001000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent

PAPER_MED = {
    "figure_eight_slow":   0.020,
    "figure_eight_normal": 0.028,
    "figure_eight_fast":   0.050,
    "pentagram_slow":      0.030,
    "pentagram_fast":      0.060,
    "random_polynomial":   0.030,
    "random_zigzag":       0.050,
}
M1_BASELINE_MED = {
    "figure_eight_slow":   0.086,
    "figure_eight_normal": 0.105,
    "figure_eight_fast":   0.157,
    "pentagram_slow":      0.090,
    "pentagram_fast":      0.130,
    "random_polynomial":   0.065,
    "random_zigzag":       0.078,
}
THRESHOLD = {k: 2 * v for k, v in PAPER_MED.items()}


def ref_curvature_figure_eight_normal(total_steps, dt):
    """
    κ(t) for x = cos(2πt/T), y = sin(4πt/T)/2 at T=5.5s.
    Returns array (total_steps,) in m^-1.
    """
    T = 5.5
    w = 2 * np.pi / T
    t = np.arange(total_steps) * dt
    xp  = -w * np.sin(w * t)
    yp  =  w * np.cos(2 * w * t)
    xpp = -(w**2) * np.cos(w * t)
    ypp = -2 * (w**2) * np.sin(2 * w * t)
    speed2 = xp**2 + yp**2
    return np.abs(xp * ypp - yp * xpp) / (speed2**1.5 + 1e-9)


def run_episode(actor_net, actor_params, env, eval_traj, key):
    """
    Run one deterministic episode. Returns arrays (T,3) pos and (T,3) refs.
    """
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, _build_obs
    from iron_man_drone.envs.trajectories import get_reference_pos

    drone_body_id = env.mj_model.body("drone").id

    # Reset with 1 env
    keys      = jax.random.split(key, 1)
    kf_mults  = jnp.ones(1)
    state, a_obs, c_obs = env.batch_reset(keys, kf_mults)

    # Inject the fixed eval trajectory (batched: leading dim 1)
    batched_traj = jax.tree_util.tree_map(lambda x: x[None], eval_traj)
    state = state._replace(traj=batched_traj)

    # Recompute obs with injected trajectory (reset used a random traj)
    mjx_single = jax.tree_util.tree_map(lambda x: x[0], state.mjx_data)
    a_obs_new, _ = _build_obs(
        mjx_single, eval_traj,
        jnp.zeros((), dtype=jnp.int32), drone_body_id
    )
    a_obs = a_obs_new[None]  # (1, 42)

    positions     = []
    ref_positions = []

    for step_i in range(EPISODE_STEPS):
        mean, _ = actor_net.apply(actor_params, a_obs)  # (1, 4)
        action  = mean  # deterministic

        state, a_obs, c_obs, reward, done = env.batch_step(state, action, kf_mults)
        state = state._replace(traj=batched_traj)   # re-inject after auto-reset guard

        pos = np.array(state.mjx_data.xpos[0, drone_body_id, :])   # (3,)
        ref = np.array(get_reference_pos(eval_traj, jnp.int32(step_i + 1)))  # (3,)

        positions.append(pos)
        ref_positions.append(ref)

        if bool(done[0]):
            print(f"      terminated at step {step_i}")
            break

    return np.array(positions), np.array(ref_positions)


def evaluate(actor_net, actor_params, env, eval_traj, num_episodes):
    """Run num_episodes, return (mean_MED, per_ep_MEDs, last_pos, last_refs)."""
    meds = []
    last_pos = last_refs = None
    for ep in range(num_episodes):
        key = jax.random.PRNGKey(ep * 7919)
        pos, refs = run_episode(actor_net, actor_params, env, eval_traj, key)
        xy_err = np.linalg.norm(pos[:, :2] - refs[:, :2], axis=1)
        med = float(xy_err.mean())
        meds.append(med)
        print(f"      ep {ep}: MED={med:.4f}m")
        last_pos, last_refs = pos, refs
    return float(np.mean(meds)), meds, last_pos, last_refs


def apex_straight_breakdown(pos, refs, kappa, apex_thresh=2.5, straight_thresh=0.8):
    n = min(len(pos), len(refs), len(kappa))
    xy_err = np.linalg.norm(pos[:n, :2] - refs[:n, :2], axis=1)
    k = kappa[:n]
    apex_mask     = k >= apex_thresh
    straight_mask = k <= straight_thresh

    def safe_mean(arr, m):
        return float(arr[m].mean()) if m.sum() > 0 else float("nan")

    return {
        "apex":     safe_mean(xy_err, apex_mask),
        "straight": safe_mean(xy_err, straight_mask),
        "transit":  safe_mean(xy_err, ~apex_mask & ~straight_mask),
        "apex_n":   int(apex_mask.sum()),
        "straight_n": int(straight_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    print(f"Loading checkpoint: {ckpt_path}")

    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(ckpt_path))
    # actor TrainState saves: step, params, opt_state
    # ckpt["actor"]["params"] = Flax variable dict {"params": {...}}
    actor_params = ckpt["actor"]["params"]
    print(f"Checkpoint loaded (step={int(ckpt['actor']['step'])})")

    from iron_man_drone.policy.networks import Actor
    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, DT, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS,
    )
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory,
        make_pentagram_trajectory,
        sample_polynomial_trajectory,
        sample_zigzag_trajectory,
    )

    actor_net = Actor(hidden_dim=256, num_layers=3, action_dim=4)

    class EvalCfg:
        num_envs = 1

    env = VecEnv(EvalCfg())
    ls  = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS

    eval_trajs = {
        "figure_eight_slow":   make_figure_eight_trajectory(DT, EPISODE_STEPS, ls, speed="slow"),
        "figure_eight_normal": make_figure_eight_trajectory(DT, EPISODE_STEPS, ls, speed="normal"),
        "figure_eight_fast":   make_figure_eight_trajectory(DT, EPISODE_STEPS, ls, speed="fast"),
        "pentagram_slow":      make_pentagram_trajectory(DT, EPISODE_STEPS, ls, speed="slow"),
        "pentagram_fast":      make_pentagram_trajectory(DT, EPISODE_STEPS, ls, speed="fast"),
    }

    rng = np.random.default_rng(42)
    poly_seeds = [int(rng.integers(0, 100000)) for _ in range(args.num_episodes)]
    zig_seeds  = [int(rng.integers(0, 100000)) for _ in range(args.num_episodes)]

    results = {}
    last_f8n_pos = last_f8n_refs = None

    print()
    print("=" * 70)
    print(f"  M1.2 epoch_{int(ckpt['actor']['step'])} — Full Evaluation")
    print("=" * 70)

    # Fixed trajectories
    for name, traj in eval_trajs.items():
        print(f"\n  [{name}]")
        med, meds, lpos, lrefs = evaluate(actor_net, actor_params, env, traj, args.num_episodes)
        results[name] = {"med": med, "meds": meds}
        if name == "figure_eight_normal":
            last_f8n_pos, last_f8n_refs = lpos, lrefs

    # Random polynomial
    print("\n  [random_polynomial]")
    poly_meds = []
    for seed in poly_seeds:
        traj = sample_polynomial_trajectory(jax.random.PRNGKey(seed), DT, EPISODE_STEPS, ls)
        pos, refs = run_episode(actor_net, actor_params, env, traj, jax.random.PRNGKey(seed + 1))
        med = float(np.linalg.norm(pos[:, :2] - refs[:, :2], axis=1).mean())
        poly_meds.append(med)
        print(f"      seed {seed}: MED={med:.4f}m")
    results["random_polynomial"] = {"med": float(np.mean(poly_meds)), "meds": poly_meds}

    # Random zigzag
    print("\n  [random_zigzag]")
    zig_meds = []
    for seed in zig_seeds:
        traj = sample_zigzag_trajectory(jax.random.PRNGKey(seed), DT, EPISODE_STEPS, ls)
        pos, refs = run_episode(actor_net, actor_params, env, traj, jax.random.PRNGKey(seed + 1))
        med = float(np.linalg.norm(pos[:, :2] - refs[:, :2], axis=1).mean())
        zig_meds.append(med)
        print(f"      seed {seed}: MED={med:.4f}m")
    results["random_zigzag"] = {"med": float(np.mean(zig_meds)), "meds": zig_meds}

    # ── MED table ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    hdr = f"  {'Trajectory':<25} {'M1.2':>8} {'M1 base':>8} {'Paper':>7} {'2×Paper':>8} {'Pass?':>6}"
    print(hdr)
    print("  " + "-" * 65)

    overall_pass = True
    for name in PAPER_MED:
        if name not in results:
            continue
        med    = results[name]["med"]
        paper  = PAPER_MED[name]
        m1base = M1_BASELINE_MED.get(name, float("nan"))
        thresh = THRESHOLD[name]
        passed = med < thresh
        if not passed:
            overall_pass = False
        delta = med - m1base
        print(f"  {name:<25} {med:8.4f} {m1base:8.4f} {paper:7.4f} {thresh:8.4f}  "
              f"{'PASS' if passed else 'FAIL':>4}  Δ{delta:+.4f}")

    print()
    print(f"  Overall: {'ALL PASS' if overall_pass else 'SOME FAIL'}")

    # ── Apex / straight breakdown ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  APEX / STRAIGHT BREAKDOWN — figure_eight_normal")
    print("=" * 70)

    kappa = ref_curvature_figure_eight_normal(EPISODE_STEPS, DT)
    print(f"  κ_max={kappa.max():.3f} m⁻¹  κ_mean={kappa.mean():.3f} m⁻¹")
    print(f"  Apex zone  (κ ≥ 2.5 m⁻¹): {(kappa >= 2.5).sum()} / {len(kappa)} steps")
    print(f"  Straight   (κ ≤ 0.8 m⁻¹): {(kappa <= 0.8).sum()} / {len(kappa)} steps")
    print()

    if last_f8n_pos is not None:
        bd = apex_straight_breakdown(last_f8n_pos, last_f8n_refs, kappa)
        ratio = bd["apex"] / (bd["straight"] + 1e-9)
        print(f"  Apex    (κ≥2.5): {bd['apex']:.4f} m  [{bd['apex_n']} steps]")
        print(f"  Straight(κ≤0.8): {bd['straight']:.4f} m  [{bd['straight_n']} steps]")
        print(f"  Transit:         {bd['transit']:.4f} m")
        print()
        print(f"  Apex : straight = {ratio:.1f}×   (M1 baseline was 11.3×)")
        if ratio < 11.3:
            print(f"  → IMPROVED by {(11.3 - ratio):.1f}× — entropy bump reduced apex overshoot")
        else:
            print(f"  → NO IMPROVEMENT — entropy had no effect on apex tracking")

    # ── Write results file ───────────────────────────────────────────────────
    out = Path(args.output or str(REPO_ROOT / "experiments/m1_2_entropy/M1_2_eval_results.md"))
    with open(out, "w") as f:
        f.write("# M1.2 Evaluation Results\n\n")
        f.write(f"Checkpoint: `{ckpt_path}`\n\n")
        f.write("## MED results\n\n")
        f.write("| Trajectory | M1.2 MED | M1 baseline | Paper | 2×Paper | Pass? |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name in PAPER_MED:
            if name not in results:
                continue
            med = results[name]["med"]
            f.write(f"| {name} | {med:.4f} | {M1_BASELINE_MED.get(name, 0):.4f} "
                    f"| {PAPER_MED[name]:.4f} | {THRESHOLD[name]:.4f} | "
                    f"{'✓' if med < THRESHOLD[name] else '✗'} |\n")
        f.write(f"\n**Overall: {'PASS' if overall_pass else 'FAIL'}**\n\n")

        if last_f8n_pos is not None:
            bd = apex_straight_breakdown(last_f8n_pos, last_f8n_refs, kappa)
            ratio = bd["apex"] / (bd["straight"] + 1e-9)
            f.write("## Apex/straight breakdown — figure_eight_normal\n\n")
            f.write("| Zone | κ threshold | Steps | Mean XY error |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| Apex     | ≥ 2.5 m⁻¹ | {bd['apex_n']}  | {bd['apex']:.4f} m |\n")
            f.write(f"| Straight | ≤ 0.8 m⁻¹ | {bd['straight_n']} | {bd['straight']:.4f} m |\n")
            f.write(f"\nApex:straight = **{ratio:.1f}×** (M1 baseline was 11.3×)\n")

    print(f"\n  Written: {out}")


if __name__ == "__main__":
    main()
