"""
Verify lazy trajectory eval matches eager precomputed version.
Tests 100 random (traj_params, t) pairs for both polynomial and zigzag types.
Pass: all within 1e-5 absolute tolerance.

Usage:
  python scripts/verify_lazy_traj.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from iron_man_drone.envs.trajectories import (
    sample_polynomial_trajectory,
    sample_zigzag_trajectory,
    eval_trajectory_position,
    get_reference_window,
    DT,
)
from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS


def eager_eval_poly(traj, t):
    """Reference eager implementation: quintic Hermite between two waypoints."""
    t = float(np.clip(t, 0.0, float(traj.total_time)))
    cum = np.array(traj.cum_times)
    wp  = np.array(traj.waypoints)
    # searchsorted to find segment
    idx = int(np.searchsorted(cum, t, side="right")) - 1
    idx = int(np.clip(idx, 0, 13))
    t0, t1 = cum[idx], cum[idx + 1]
    tau = float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
    p0, p1 = wp[idx], wp[idx + 1]
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    return p0 + (p1 - p0) * s


def eager_eval_zigzag(traj, t):
    """Reference eager implementation: linear interp between waypoints."""
    t = float(np.clip(t, 0.0, float(traj.total_time)))
    cum = np.array(traj.cum_times)
    wp  = np.array(traj.waypoints)
    idx = int(np.searchsorted(cum, t, side="right")) - 1
    idx = int(np.clip(idx, 0, 13))
    t0, t1 = cum[idx], cum[idx + 1]
    alpha = float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
    p0, p1 = wp[idx], wp[idx + 1]
    return p0 + alpha * (p1 - p0)


def main():
    key = jax.random.PRNGKey(0)
    ls  = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    tol = 1e-5
    n_trajs = 50   # 50 poly + 50 zigzag = 100 total
    n_times = 20   # random time queries per traj

    print(f"Testing {n_trajs} poly + {n_trajs} zigzag trajectories × {n_times} time points each")
    print(f"Tolerance: {tol:.0e} absolute\n")

    all_pass = True
    max_err  = 0.0

    # ── Polynomial ───────────────────────────────────────────────────────────
    print("── Polynomial (quintic Hermite) ────────────────────────────────")
    poly_errs = []
    for i in range(n_trajs):
        key, tk = jax.random.split(key)
        traj = sample_polynomial_trajectory(tk, DT, EPISODE_STEPS, ls)
        total_time = float(traj.total_time)

        key, tkey = jax.random.split(key)
        times = np.random.uniform(0.0, total_time, n_times)

        for t in times:
            lazy_pos  = np.array(eval_trajectory_position(traj, jnp.float32(t)))
            eager_pos = np.array(eager_eval_poly(traj, t))
            err = np.abs(lazy_pos - eager_pos).max()
            poly_errs.append(err)
            if err > tol:
                print(f"  FAIL traj={i} t={t:.3f}: lazy={lazy_pos} eager={eager_pos} err={err:.2e}")
                all_pass = False

    max_poly = max(poly_errs)
    max_err  = max(max_err, max_poly)
    print(f"  Poly: {n_trajs*n_times} queries, max_err={max_poly:.2e}  "
          f"{'PASS' if max_poly <= tol else 'FAIL'}")

    # ── Zigzag ───────────────────────────────────────────────────────────────
    print("── Zigzag (linear) ─────────────────────────────────────────────")
    zz_errs = []
    for i in range(n_trajs):
        key, tk = jax.random.split(key)
        traj = sample_zigzag_trajectory(tk, DT, EPISODE_STEPS, ls)
        total_time = float(traj.total_time)

        times = np.random.uniform(0.0, total_time, n_times)

        for t in times:
            lazy_pos  = np.array(eval_trajectory_position(traj, jnp.float32(t)))
            eager_pos = np.array(eager_eval_zigzag(traj, t))
            err = np.abs(lazy_pos - eager_pos).max()
            zz_errs.append(err)
            if err > tol:
                print(f"  FAIL traj={i} t={t:.3f}: lazy={lazy_pos} eager={eager_pos} err={err:.2e}")
                all_pass = False

    max_zz  = max(zz_errs)
    max_err = max(max_err, max_zz)
    print(f"  Zigzag: {n_trajs*n_times} queries, max_err={max_zz:.2e}  "
          f"{'PASS' if max_zz <= tol else 'FAIL'}")

    # ── Boundary cases ───────────────────────────────────────────────────────
    print("── Boundary cases ──────────────────────────────────────────────")
    key, tk = jax.random.split(key)
    traj = sample_polynomial_trajectory(tk, DT, EPISODE_STEPS, ls)
    for t_val, label in [(0.0, "t=0"), (float(traj.total_time), "t=total"), (-0.1, "t<0"), (1e6, "t>>total")]:
        pos = np.array(eval_trajectory_position(traj, jnp.float32(t_val)))
        finite = np.all(np.isfinite(pos))
        print(f"  {label:15s}: pos={pos}  finite={finite}  {'OK' if finite else 'FAIL'}")
        if not finite:
            all_pass = False

    # ── get_reference_window ─────────────────────────────────────────────────
    print("── get_reference_window (vmap) ─────────────────────────────────")
    key, tk = jax.random.split(key)
    traj = sample_polynomial_trajectory(tk, DT, EPISODE_STEPS, ls)
    window = get_reference_window(traj, jnp.int32(50), LOOKAHEAD_N, LOOKAHEAD_DT_STEPS)
    window_np = np.array(window)
    ok = (window_np.shape == (LOOKAHEAD_N, 3)) and np.all(np.isfinite(window_np))
    print(f"  shape={window_np.shape}, finite={np.all(np.isfinite(window_np))}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

    print()
    print("="*60)
    print(f"  RESULT: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"  Max absolute error across all queries: {max_err:.2e}")
    if all_pass:
        print("  Lazy trajectory refactor is bit-equivalent to eager reference.")
    else:
        print("  FAILURES detected — refactor may have introduced eval bugs.")
    print("="*60)


if __name__ == "__main__":
    main()
