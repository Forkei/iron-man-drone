"""
Diagnostic 3: Curvature distribution of the M1.3 polynomial generator.

Samples 1000 random training trajectories and reports κ_max statistics.
Compares to figure_eight_normal apex requirement (~4.79 m^-1).
No simulation required — pure trajectory math.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).parent.parent
N_TRAJ = 1000
N_SAMPLES_PER_SEG = 30   # curvature sample points per segment


def compute_seg_kappa_max(coeffs_np, T_seg):
    """
    Compute max curvature over a quintic segment.
    coeffs_np: (6, 3) — polynomial coefficients [c0..c5] for x, y, z
    T_seg: segment duration in seconds
    """
    taus = np.linspace(0.0, T_seg, N_SAMPLES_PER_SEG, endpoint=False)

    # Velocity: p'(tau) = c1 + tau*(2c2 + tau*(3c3 + tau*(4c4 + 5c5*tau)))
    vx = coeffs_np[1, 0] + taus * (
         2*coeffs_np[2, 0] + taus * (
         3*coeffs_np[3, 0] + taus * (
         4*coeffs_np[4, 0] + 5*coeffs_np[5, 0]*taus)))
    vy = coeffs_np[1, 1] + taus * (
         2*coeffs_np[2, 1] + taus * (
         3*coeffs_np[3, 1] + taus * (
         4*coeffs_np[4, 1] + 5*coeffs_np[5, 1]*taus)))

    # Acceleration: p''(tau) = 2c2 + tau*(6c3 + tau*(12c4 + 20c5*tau))
    ax = 2*coeffs_np[2, 0] + taus * (
         6*coeffs_np[3, 0] + taus * (
         12*coeffs_np[4, 0] + 20*coeffs_np[5, 0]*taus))
    ay = 2*coeffs_np[2, 1] + taus * (
         6*coeffs_np[3, 1] + taus * (
         12*coeffs_np[4, 1] + 20*coeffs_np[5, 1]*taus))

    speed_sq = vx**2 + vy**2
    valid = speed_sq > 1e-8
    kappa = np.zeros(len(taus))
    kappa[valid] = np.abs(vx[valid]*ay[valid] - vy[valid]*ax[valid]) / speed_sq[valid]**1.5
    return kappa.max() if valid.any() else 0.0


def figure8_normal_kappa_stats():
    """Compute κ statistics for figure_eight_normal analytically."""
    T = 5.5
    omega = 2.0 * np.pi / T
    t_vals = np.linspace(0, T, 2000, endpoint=False)
    s = np.sin(omega * t_vals)
    c = np.cos(omega * t_vals)
    vx = -omega * s
    vy = omega * (1.0 - 2.0*s**2)
    ax = -omega**2 * c
    ay = -4.0 * omega**2 * s * c
    speed_sq = vx**2 + vy**2
    kappa = np.abs(vx*ay - vy*ax) / np.maximum(speed_sq**1.5, 1e-10)
    return kappa.max(), np.percentile(kappa, 90), np.median(kappa)


def main():
    from iron_man_drone.envs.trajectories import (
        sample_polynomial_trajectory, MAX_SEGS, DT,
    )
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS

    lookahead_steps = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    jit_sample = jax.jit(
        lambda k: sample_polynomial_trajectory(k, DT, EPISODE_STEPS, lookahead_steps)
    )

    # Warm up
    key = jax.random.PRNGKey(0)
    key, sk = jax.random.split(key)
    jit_sample(sk)
    print(f"Sampling {N_TRAJ} polynomial trajectories ...")

    kappa_maxes = []
    for i in range(N_TRAJ):
        key, sk = jax.random.split(key)
        traj = jit_sample(sk)

        cum_times = np.array(traj.cum_times)
        coeffs    = np.array(traj.poly_coeffs)   # (MAX_SEGS, 6, 3)

        seg_kappas = []
        for seg_i in range(MAX_SEGS):
            T_seg = float(cum_times[seg_i + 1] - cum_times[seg_i])
            if not np.isfinite(T_seg) or T_seg < 1e-6:
                break
            k_max = compute_seg_kappa_max(coeffs[seg_i], T_seg)
            seg_kappas.append(k_max)

        if seg_kappas:
            kappa_maxes.append(max(seg_kappas))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_TRAJ} sampled ...")

    kappa_maxes = np.array(kappa_maxes)

    # Figure-eight reference
    f8_max, f8_p90, f8_med = figure8_normal_kappa_stats()

    print(f"\n{'='*58}")
    print(f"  CURVATURE DISTRIBUTION — M1.3 polynomial generator")
    print(f"  (n={len(kappa_maxes)} trajectories, {N_SAMPLES_PER_SEG} pts/seg)")
    print(f"{'='*58}")
    print(f"  κ_max per trajectory:")
    print(f"    Median:   {np.median(kappa_maxes):.3f} m^-1")
    print(f"    p75:      {np.percentile(kappa_maxes, 75):.3f} m^-1")
    print(f"    p90:      {np.percentile(kappa_maxes, 90):.3f} m^-1")
    print(f"    p99:      {np.percentile(kappa_maxes, 99):.3f} m^-1")
    print(f"    Max:      {kappa_maxes.max():.3f} m^-1")
    print(f"")
    print(f"  Figure-eight normal reference:")
    print(f"    κ_max apex:  {f8_max:.3f} m^-1")
    print(f"    κ_p90:       {f8_p90:.3f} m^-1")
    print(f"    κ_median:    {f8_med:.3f} m^-1")
    print(f"")

    pct_above_apex = float((kappa_maxes >= f8_max).mean()) * 100
    pct_above_3   = float((kappa_maxes >= 3.0).mean()) * 100
    pct_above_4   = float((kappa_maxes >= 4.0).mean()) * 100

    print(f"  Training coverage of figure-eight:")
    print(f"    Fraction with κ_max >= 3.0 m^-1:  {pct_above_3:.1f}%")
    print(f"    Fraction with κ_max >= 4.0 m^-1:  {pct_above_4:.1f}%")
    print(f"    Fraction with κ_max >= {f8_max:.2f} m^-1 (apex): {pct_above_apex:.1f}%")
    print(f"")

    if pct_above_apex >= 80:
        coverage = "GOOD — figure-eight apex is in-distribution"
    elif pct_above_apex >= 40:
        coverage = "PARTIAL — apex borderline in-distribution"
    else:
        coverage = "POOR — apex is extrapolation (curvature gap remains)"

    print(f"  Coverage assessment: {coverage}")
    print(f"{'='*58}\n")

    print(f"SUMMARY_LINE: n={len(kappa_maxes)} | "
          f"median={np.median(kappa_maxes):.3f} | "
          f"p90={np.percentile(kappa_maxes,90):.3f} | "
          f"p99={np.percentile(kappa_maxes,99):.3f} | "
          f"max={kappa_maxes.max():.3f} | "
          f"pct_above_apex={pct_above_apex:.1f}%")


if __name__ == "__main__":
    main()
