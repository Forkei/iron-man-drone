"""
Task 4 — MJX→MJWarp state-transfer overhead benchmark.

Measures:
  - Total batch_render latency per call (ms) at N = 1, 64, 256, 1024
  - Breakdown: state transfer vs mjw.forward vs mjw.render+get_depth
  - Effective env-steps/sec = N / render_latency
  - State-transfer fraction of total render time (decision gate: flag if > 20%)

Protocol:
  1. Create DepthVecEnv at each N.
  2. Run 200 batch_step calls to produce realistic MJX state.
  3. Warm up batch_render for WARMUP calls.
  4. Time batch_render over BENCH calls.
  5. Time each sub-phase separately.

Results written to notes/M2_5_benchmark_results.md.

Usage:
  python scripts/benchmark_mjwarp_transfer.py
  python scripts/benchmark_mjwarp_transfer.py --nworld 1024 --bench 50
"""

import sys, time, argparse, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax, jax.numpy as jnp
import warp as wp
import mujoco_warp as mjw
import mujoco

REPO_ROOT = Path(__file__).parent.parent


def _gate(n, label, passed, detail=""):
    mark = "✓" if passed else "✗"
    print(f"  G{n} {mark}  {label}")
    if detail:
        print(f"      {detail}")
    return passed


def bench_n(N, steps, warmup, bench_reps):
    from iron_man_drone.envs.quadrotor_env_depth import DepthVecEnv

    cfg = types.SimpleNamespace(num_envs=N, max_episode_steps=1000)
    env = DepthVecEnv(cfg, n_obstacles=4, fault_prob=0.7)

    # Produce realistic MJX state via batch_step
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    states, _, _ = env.batch_reset(keys)

    rng_act = np.random.default_rng(1)
    for _ in range(steps):
        actions = jnp.array(rng_act.uniform(-1, 1, (N, 4)).astype(np.float32))
        states, _, _, _, _ = env.batch_step(states, actions)
    jax.block_until_ready(states.mjx_data.qpos)

    # ── Warm up ──────────────────────────────────────────────────────────────
    for _ in range(warmup):
        _ = env.batch_render(states)

    # ── Total render latency ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(bench_reps):
        _ = env.batch_render(states)
    dt_total = (time.perf_counter() - t0) / bench_reps
    tp = N / dt_total

    # ── Sub-phase breakdown ───────────────────────────────────────────────────
    # Phase A: state transfer (qpos + mocap_pos assign)
    qpos_np  = np.array(states.mjx_data.qpos)
    mocap_np = np.array(states.obstacle_positions)

    t0 = time.perf_counter()
    for _ in range(bench_reps):
        env._mjw_data.qpos.assign(qpos_np)
        env._mjw_data.mocap_pos.assign(
            wp.from_numpy(mocap_np, dtype=wp.vec3f, device="cuda:0")
        )
        wp.synchronize()
    dt_transfer = (time.perf_counter() - t0) / bench_reps

    # Phase B: mjw.forward
    t0 = time.perf_counter()
    for _ in range(bench_reps):
        mjw.forward(env._mjw_model, env._mjw_data)
        wp.synchronize()
    dt_forward = (time.perf_counter() - t0) / bench_reps

    # Phase C: render + get_depth + numpy copy
    t0 = time.perf_counter()
    for _ in range(bench_reps):
        mjw.render(env._mjw_model, env._mjw_data, env._rc)
        mjw.get_depth(env._rc, env.cam_id, 5.0, env._depth_buf)
        wp.synchronize()
        _ = env._depth_buf.numpy()
    dt_render = (time.perf_counter() - t0) / bench_reps

    transfer_frac = dt_transfer / dt_total

    return {
        "N":              N,
        "dt_total_ms":    dt_total * 1e3,
        "dt_transfer_ms": dt_transfer * 1e3,
        "dt_forward_ms":  dt_forward * 1e3,
        "dt_render_ms":   dt_render * 1e3,
        "tp_ksteps":      tp / 1e3,
        "transfer_frac":  transfer_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=200,
                        help="MJX steps before benchmarking")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bench",  type=int, default=50,
                        help="Timed repetitions per N")
    parser.add_argument("--nworld", type=int, default=None,
                        help="Single N to benchmark (default: [1, 64, 256, 1024])")
    args = parser.parse_args()

    N_list = [args.nworld] if args.nworld else [1, 64, 256, 1024]

    print(f"\n{'='*70}")
    print(f"  Task 4 — MJX→MJWarp state-transfer overhead benchmark")
    print(f"  steps={args.steps}  warmup={args.warmup}  bench={args.bench}")
    print(f"{'='*70}\n")

    wp.init()
    results = []
    for N in N_list:
        print(f"  Benchmarking N={N} ...", flush=True)
        r = bench_n(N, args.steps, args.warmup, args.bench)
        results.append(r)
        print(f"    total={r['dt_total_ms']:.1f}ms  "
              f"transfer={r['dt_transfer_ms']:.1f}ms  "
              f"forward={r['dt_forward_ms']:.1f}ms  "
              f"render={r['dt_render_ms']:.1f}ms  "
              f"tp={r['tp_ksteps']:.1f}k steps/sec  "
              f"transfer_frac={r['transfer_frac']:.1%}")

    # ── SC-4 gate: ≥ 25k env-steps/sec at N=1024 ─────────────────────────────
    print()
    r1024 = next((r for r in results if r["N"] == 1024), None)
    gates = {}
    if r1024:
        gates["SC4"] = _gate(
            "SC-4", "Throughput ≥ 25k env-steps/sec at N=1024",
            r1024["tp_ksteps"] >= 25.0,
            f"{r1024['tp_ksteps']:.1f}k env-steps/sec",
        )

    # Transfer overhead gate (informational, not a hard gate)
    if r1024:
        flag = r1024["transfer_frac"] > 0.20
        _gate(
            "F3",
            "State-transfer overhead ≤ 20% of total render time",
            not flag,
            f"transfer_frac={r1024['transfer_frac']:.1%}"
            + (" — flag for M3 arch review" if flag else ""),
        )

    # ── Write results to notes/ ───────────────────────────────────────────────
    md_path = REPO_ROOT / "notes" / "M2_5_benchmark_results.md"
    with open(md_path, "w") as f:
        f.write("# M2.5 Task 4 — MJWarp State-Transfer Benchmark\n\n")
        f.write(f"Date: 2026-05-12  |  n_obstacles=4  |  steps={args.steps}  "
                f"|  warmup={args.warmup}  |  bench={args.bench}\n\n")
        f.write("## Latency breakdown (ms per batch_render call)\n\n")
        f.write("| N | Total (ms) | Transfer (ms) | Forward (ms) | Render+copy (ms) | "
                "Throughput (k steps/s) | Transfer% |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['N']} "
                f"| {r['dt_total_ms']:.1f} "
                f"| {r['dt_transfer_ms']:.1f} "
                f"| {r['dt_forward_ms']:.1f} "
                f"| {r['dt_render_ms']:.1f} "
                f"| {r['tp_ksteps']:.1f} "
                f"| {r['transfer_frac']:.1%} |\n"
            )
        f.write("\n## SC-4 gate\n\n")
        if r1024:
            sc4_pass = r1024["tp_ksteps"] >= 25.0
            f.write(
                f"N=1024 throughput: **{r1024['tp_ksteps']:.1f}k env-steps/sec** — "
                f"{'PASS' if sc4_pass else 'FAIL'} (gate: ≥ 25k)\n\n"
            )
        f.write("## M3 architectural note\n\n")
        if r1024 and r1024["transfer_frac"] > 0.20:
            f.write(
                f"State-transfer overhead is {r1024['transfer_frac']:.1%} of total render time "
                f"(> 20% threshold). Flag for M3 architectural decision: "
                f"consider Option B (full MJWarp) or render subsampling.\n"
            )
        else:
            frac = r1024["transfer_frac"] if r1024 else 0
            f.write(
                f"State-transfer overhead is {frac:.1%} of total render time "
                f"(≤ 20% threshold). Option A (MJX physics + MJWarp render) "
                f"remains viable for M3 training.\n"
            )

    print(f"\n  Results written to {md_path}")
    print(f"\n{'='*70}")
    all_pass = all(gates.values())
    print(f"  {'ALL PASS' if all_pass else 'FAIL — see gates above'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
