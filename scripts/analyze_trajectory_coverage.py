"""
Trajectory coverage analysis: training distribution vs figure-eight eval.

Pure numpy implementation — no JAX, runs in ~30 seconds for N=1000.

Computes curvature, angular velocity, and turn radius distribution for:
  - 1000 polynomial training trajectories
  - 1000 zigzag training trajectories
  - Figure-eight (slow / normal / fast)

Answers: Is figure-eight apex curvature in-distribution for our training mix?

Usage:
  python scripts/analyze_trajectory_coverage.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

REPO_ROOT = Path(__file__).parent.parent

# Match env constants
DT           = 0.01   # 100 Hz
EPISODE_STEPS = 1000
LOOKAHEAD_STEPS = 50   # LOOKAHEAD_N * LOOKAHEAD_DT_STEPS = 10 * 5
POLY_SEG_DUR  = (1.5, 4.0)
POLY_BOUNDS   = 1.5    # ±m
ZZ_SEG_DUR    = (1.0, 1.5)
ZZ_BOUNDS     = 1.0    # ±m


# ── Trajectory samplers (pure numpy) ─────────────────────────────────────────

MAX_VEL = 0.8   # m/s at interior waypoints
MAX_ACC = 2.0   # m/s² at interior waypoints


def _solve_quintic(p0, v0, a0, p1, v1, a1, T):
    """Closed-form quintic coefficients for one coordinate (scalars or 1-D arrays)."""
    T  = max(T, 1e-6)
    c0 = p0
    c1 = v0
    c2 = a0 * 0.5
    d1 = p1 - p0 - v0*T - a0*T**2*0.5
    d2 = v1 - v0 - a0*T
    d3 = a1 - a0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    c3 =  10*d1/T3 -  4*d2/T2 + d3/(2*T)
    c4 = -15*d1/T4 +  7*d2/T3 - d3/T2
    c5 =   6*d1/T5 -  3*d2/T4 + d3/(2*T3)
    return np.array([c0, c1, c2, c3, c4, c5])  # (6,) or (6, 2)


def sample_poly_positions(rng, n_steps: int) -> np.ndarray:
    """
    C2-continuous quintic polynomial trajectory (M1.3 implementation).
    Interior waypoints have random nonzero velocity and acceleration.
    """
    total_time = (n_steps + LOOKAHEAD_STEPS) * DT
    min_dur, max_dur = POLY_SEG_DUR
    n_seg = int(total_time / min_dur) + 2

    durs = rng.uniform(min_dur, max_dur, size=n_seg)
    cum  = np.concatenate([[0.0], np.cumsum(durs)])
    wps  = rng.uniform(-POLY_BOUNDS, POLY_BOUNDS, size=(n_seg + 1, 2))
    vels = rng.uniform(-MAX_VEL, MAX_VEL, size=(n_seg + 1, 2))
    accs = rng.uniform(-MAX_ACC, MAX_ACC, size=(n_seg + 1, 2))
    # Hover at episode start and end
    vels[0] = 0.0;  vels[-1] = 0.0
    accs[0] = 0.0;  accs[-1] = 0.0

    # Precompute per-segment coefficients: list of (6, 2) arrays
    segs = []
    for i in range(n_seg):
        # _solve_quintic with (2,) inputs returns (6, 2)
        c = _solve_quintic(wps[i], vels[i], accs[i],
                           wps[i+1], vels[i+1], accs[i+1], durs[i])
        segs.append(c)  # (6, 2)

    times = np.arange(n_steps) * DT
    pos   = np.zeros((n_steps, 2))
    for j, t in enumerate(times):
        idx = int(np.searchsorted(cum, t, side="right")) - 1
        idx = np.clip(idx, 0, n_seg - 1)
        tau = t - cum[idx]
        c = segs[idx]  # (6, 2)
        # Horner evaluation
        p = c[5]
        p = c[4] + tau * p
        p = c[3] + tau * p
        p = c[2] + tau * p
        p = c[1] + tau * p
        p = c[0] + tau * p
        pos[j] = p
    return pos


def sample_zz_positions(rng, n_steps: int) -> np.ndarray:
    """Sample one zigzag trajectory, return (n_steps, 2) XY positions."""
    total_time = (n_steps + LOOKAHEAD_STEPS) * DT
    min_dur, max_dur = ZZ_SEG_DUR
    n_seg = int(total_time / min_dur) + 2

    durs = rng.uniform(min_dur, max_dur, size=n_seg)
    cum  = np.concatenate([[0.0], np.cumsum(durs)])
    wps  = rng.uniform(-ZZ_BOUNDS, ZZ_BOUNDS, size=(n_seg + 1, 2))

    times = np.arange(n_steps) * DT
    pos   = np.zeros((n_steps, 2))
    for i, t in enumerate(times):
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = np.clip(idx, 0, n_seg - 1)
        t0, t1 = cum[idx], cum[idx + 1]
        alpha = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
        pos[i] = wps[idx] + alpha * (wps[idx + 1] - wps[idx])
    return pos


def figure_eight_positions(n_steps: int, T: float) -> np.ndarray:
    """Lemniscate figure-eight: x=cos(2πt/T), y=0.5*sin(4πt/T)."""
    times = np.arange(n_steps) * DT
    x = np.cos(2 * np.pi * times / T)
    y = 0.5 * np.sin(4 * np.pi * times / T)
    return np.stack([x, y], axis=1)  # (n_steps, 2)


# ── Curvature / angular velocity helpers ─────────────────────────────────────

def curvature_stats(pos: np.ndarray) -> dict:
    """
    Given (T, 2) XY positions sampled at uniform DT, compute 2D curvature
    and angular velocity of heading via central finite differences.
    """
    v = np.gradient(pos, DT, axis=0)   # (T, 2) velocity
    a = np.gradient(v,   DT, axis=0)   # (T, 2) acceleration

    vx, vy = v[:, 0], v[:, 1]
    ax, ay = a[:, 0], a[:, 1]
    speed  = np.sqrt(vx**2 + vy**2)

    mask = speed > 0.02   # ignore near-stopped samples

    cross = vx * ay - vy * ax                          # 2D "cross product"
    kappa = np.abs(cross[mask]) / (speed[mask]**3 + 1e-12)  # |κ| [m^-1]
    omega = np.abs(cross[mask]) / (speed[mask]**2 + 1e-12)  # |dθ/dt| [rad/s]
    spd_m = speed[mask]

    def pstats(arr):
        if len(arr) == 0:
            return dict(max=0., p99=0., p95=0., mean=0.)
        return dict(max=float(arr.max()), p99=float(np.percentile(arr, 99)),
                    p95=float(np.percentile(arr, 95)), mean=float(arr.mean()))

    return dict(
        kappa=pstats(kappa),
        omega=pstats(omega),
        speed=pstats(spd_m),
        n_valid=int(mask.sum()),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N = 1000
    rng = np.random.default_rng(42)

    print(f"Analyzing {N} polynomial + {N} zigzag training trajectories "
          f"+ 3 figure-eight variants ({EPISODE_STEPS} steps each)...")

    # ── Polynomial ────────────────────────────────────────────────────────────
    poly_kappa_max = np.zeros(N)
    poly_omega_max = np.zeros(N)
    for i in range(N):
        pos = sample_poly_positions(rng, EPISODE_STEPS)
        s   = curvature_stats(pos)
        poly_kappa_max[i] = s["kappa"]["max"]
        poly_omega_max[i] = s["omega"]["max"]
        if (i + 1) % 250 == 0:
            print(f"  Polynomial {i+1}/{N}...")

    # ── Zigzag ────────────────────────────────────────────────────────────────
    zz_kappa_max = np.zeros(N)
    zz_omega_max = np.zeros(N)
    for i in range(N):
        pos = sample_zz_positions(rng, EPISODE_STEPS)
        s   = curvature_stats(pos)
        zz_kappa_max[i] = s["kappa"]["max"]
        zz_omega_max[i] = s["omega"]["max"]
        if (i + 1) % 250 == 0:
            print(f"  Zigzag {i+1}/{N}...")

    # ── Figure-eight ──────────────────────────────────────────────────────────
    f8_configs = {"slow": 15.0, "normal": 5.5, "fast": 3.5}
    f8_stats = {}
    for name, T in f8_configs.items():
        pos = figure_eight_positions(EPISODE_STEPS, T)
        f8_stats[name] = curvature_stats(pos)
        f8_stats[name]["T"] = T

    # ── OOD assessment ────────────────────────────────────────────────────────
    f8n_kappa = f8_stats["normal"]["kappa"]["max"]
    f8n_omega = f8_stats["normal"]["omega"]["max"]

    all_kappa = np.concatenate([poly_kappa_max, zz_kappa_max])
    all_omega = np.concatenate([poly_omega_max, zz_omega_max])

    poly_cov_k = float(np.mean(poly_kappa_max >= f8n_kappa) * 100)
    poly_cov_w = float(np.mean(poly_omega_max >= f8n_omega) * 100)
    zz_cov_k   = float(np.mean(zz_kappa_max   >= f8n_kappa) * 100)
    zz_cov_w   = float(np.mean(zz_omega_max   >= f8n_omega) * 100)
    mix_cov_k  = float(np.mean(all_kappa       >= f8n_kappa) * 100)
    mix_cov_w  = float(np.mean(all_omega        >= f8n_omega) * 100)

    # ── Console output ────────────────────────────────────────────────────────
    pcts = [10, 25, 50, 75, 90, 95, 99]

    def print_dist(label, arr):
        vals = np.percentile(arr, pcts)
        row = "  ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, vals))
        print(f"    {label}: {row}")

    print("\n" + "="*70)
    print("  TRAJECTORY COVERAGE ANALYSIS")
    print("="*70)

    print("\n── POLYNOMIAL (N=1000, ±1.5m, seg 1.5–4.0s, quintic Hermite) ──")
    print_dist("κ_max [m⁻¹]  ", poly_kappa_max)
    print_dist("ω_max [rad/s]", poly_omega_max)

    print("\n── ZIGZAG (N=1000, ±1.0m, seg 1.0–1.5s, linear) ──")
    print_dist("κ_max [m⁻¹]  ", zz_kappa_max)
    print_dist("ω_max [rad/s]", zz_omega_max)

    print("\n── FIGURE-EIGHT ──")
    for name, T in f8_configs.items():
        s = f8_stats[name]
        print(f"  {name:8s} T={T:4.1f}s | κ_max={s['kappa']['max']:.3f} m⁻¹ | "
              f"ω_max={s['omega']['max']:.3f} rad/s | v_max={s['speed']['max']:.3f} m/s")

    print("\n" + "="*70)
    print("  OOD ASSESSMENT — figure-eight (normal) apex")
    print("="*70)
    print(f"  Target: κ={f8n_kappa:.3f} m⁻¹, ω={f8n_omega:.3f} rad/s")
    print(f"  Polynomial trajs exceeding target:  κ={poly_cov_k:.1f}%  ω={poly_cov_w:.1f}%")
    print(f"  Zigzag trajs exceeding target:      κ={zz_cov_k:.1f}%    ω={zz_cov_w:.1f}%")
    print(f"  Mixed training (50/50) exceeding:   κ={mix_cov_k:.1f}%  ω={mix_cov_w:.1f}%")

    ood = mix_cov_k < 10 or mix_cov_w < 10
    verdict = "OOD" if ood else "IN-DISTRIBUTION"
    print(f"\n  VERDICT: Figure-eight apex is {verdict}")
    print("="*70)

    # ── Save arrays ───────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "experiments" / "m1_baseline" / "diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "poly_max_kappa.npy", poly_kappa_max)
    np.save(out_dir / "poly_max_omega.npy", poly_omega_max)
    np.save(out_dir / "zz_max_kappa.npy",   zz_kappa_max)
    np.save(out_dir / "zz_max_omega.npy",   zz_omega_max)

    # ── Write markdown ────────────────────────────────────────────────────────
    write_markdown(
        poly_kappa_max, poly_omega_max,
        zz_kappa_max, zz_omega_max,
        f8_stats, f8_configs,
        poly_cov_k, poly_cov_w,
        zz_cov_k, zz_cov_w,
        mix_cov_k, mix_cov_w,
        verdict,
    )
    print(f"\nRaw arrays saved to {out_dir}/")
    print(f"Markdown report: notes/trajectory_coverage_analysis.md")


def write_markdown(
    poly_kappa, poly_omega,
    zz_kappa, zz_omega,
    f8_stats, f8_configs,
    poly_cov_k, poly_cov_w,
    zz_cov_k, zz_cov_w,
    mix_cov_k, mix_cov_w,
    verdict,
):
    pcts = [10, 25, 50, 75, 90, 95, 99]

    def ptable(arr):
        vals = np.percentile(arr, pcts)
        h = "| " + " | ".join(f"p{p}" for p in pcts) + " |"
        d = "|" + "|".join(" --- " for _ in pcts) + "|"
        r = "| " + " | ".join(f"{v:.3f}" for v in vals) + " |"
        return "\n".join([h, d, r])

    f8n = f8_stats["normal"]
    f8n_kappa = f8n["kappa"]["max"]
    f8n_omega = f8n["omega"]["max"]

    lines = [
        "# Trajectory Coverage Analysis",
        "",
        "**Date**: 2026-05-03  ",
        "**N**: 1000 polynomial + 1000 zigzag + 3 figure-eight variants  ",
        "**Resolution**: 10ms (100Hz), EPISODE_STEPS=1000 (10s)  ",
        "",
        "## Summary answer",
        "",
    ]

    if "OOD" in verdict:
        lines += [
            f"**Figure-eight apex is OUT-OF-DISTRIBUTION for the training mix.**",
            "",
            f"Figure-eight (normal) apex: κ = {f8n_kappa:.3f} m⁻¹, ω = {f8n_omega:.3f} rad/s.  ",
            f"Only **{mix_cov_k:.1f}%** of mixed training trajectories exceed this curvature; "
            f"**{mix_cov_w:.1f}%** exceed this angular velocity.  ",
            "The policy was never rewarded for executing tight-radius turns at speed.",
        ]
    else:
        lines += [
            f"**Figure-eight apex is IN-DISTRIBUTION for the training mix.**",
            "",
            f"Figure-eight (normal) apex: κ = {f8n_kappa:.3f} m⁻¹, ω = {f8n_omega:.3f} rad/s.  ",
            f"**{mix_cov_k:.1f}%** of training trajectories exceed this curvature. "
            "If apex overshoot persists after entropy fix, the cause is not distribution coverage.",
        ]

    lines += [
        "",
        "---",
        "",
        "## Figure-eight reference stats",
        "",
        "| Variant | Period T | κ_max (m⁻¹) | ω_max (rad/s) | v_max (m/s) |",
        "|---|---|---|---|---|",
    ]
    for name, T in f8_configs.items():
        s = f8_stats[name]
        lines.append(f"| {name} | {T}s | {s['kappa']['max']:.3f} | "
                     f"{s['omega']['max']:.3f} | {s['speed']['max']:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## Polynomial training trajectories (N=1000)",
        "",
        "Spatial bounds: ±1.5m XY. Seg duration: 1.5–4.0s. C2-continuous quintic polynomial (nonzero vel/acc at interior waypoints).",
        "",
        "**Max curvature per trajectory κ_max (m⁻¹):**",
        "",
        ptable(poly_kappa),
        "",
        "**Max angular velocity per trajectory ω_max (rad/s):**",
        "",
        ptable(poly_omega),
        "",
        f"Coverage of fig-8 normal apex: **{poly_cov_k:.1f}%** exceed "
        f"κ={f8n_kappa:.3f} m⁻¹; **{poly_cov_w:.1f}%** exceed ω={f8n_omega:.3f} rad/s",
        "",
        "---",
        "",
        "## Zigzag training trajectories (N=1000)",
        "",
        "Spatial bounds: ±1.0m XY. Seg duration: 1.0–1.5s. Linear segments "
        "(theoretically infinite curvature at waypoints — sampled at 10ms resolution).",
        "",
        "**Max curvature per trajectory κ_max (m⁻¹):**",
        "",
        ptable(zz_kappa),
        "",
        "**Max angular velocity per trajectory ω_max (rad/s):**",
        "",
        ptable(zz_omega),
        "",
        f"Coverage of fig-8 normal apex: **{zz_cov_k:.1f}%** exceed "
        f"κ={f8n_kappa:.3f} m⁻¹; **{zz_cov_w:.1f}%** exceed ω={f8n_omega:.3f} rad/s",
        "",
        "---",
        "",
        "## OOD verdict and proposed M1.3 fixes",
        "",
        f"Mixed training (50/50 poly/zigzag): **{mix_cov_k:.1f}%** of trajectories exceed "
        f"fig-8 normal apex curvature, **{mix_cov_w:.1f}%** exceed apex angular velocity.",
        "",
        f"**VERDICT: {verdict}**",
        "",
    ]

    if "OOD" in verdict:
        lines += [
            "### Proposed fixes for M1.3 (if M1.2 fails threshold)",
            "",
            "**Option A — Tighten zigzag segment duration** ← recommended first",
            "- Change: `seg_duration=(1.0, 1.5)` → `(0.5, 1.0)` in `sample_zigzag_trajectory`",
            "- Effect: 2× more direction reversals per episode → exposes apex-like heading changes",
            "- Why it's safe: zigzag is already 'infeasible'; tighter segments are more infeasible, not less",
            "- Not training on the test: exposes similar *curvature magnitude*, different *geometry* (random vs lemniscate)",
            "- Risk: verify `MAX_SEGS` is large enough with shorter segments",
            "",
            "**Option B — Reduce zigzag spatial bounds** ± 1.0m → ±0.5m",
            "- Smaller area + same duration = higher angular velocity per waypoint transition",
            "- Can combine with Option A for a stronger effect",
            "",
            "**Option C — Add curvature-rich midpoint to polynomial segments**",
            "- Insert intermediate waypoints at segment midpoints with random offsets",
            "- Forces the quintic Hermite to curve sharply through the midpoint",
            "- More complex code change; not in SimpleFlight recipe",
            "",
            "**Option D — Add analytic figure-eight to training mix (flag: trains on test geometry)**",
            "- Add random-period figure-eight (T ~ Uniform[3s, 20s]) at e.g. 20% of training mix",
            "- Directly exposes apex dynamics",
            "- ⚠️ Compromises held-out OOD test — the eval figure-eight is the same geometry",
            "- Only consider if Options A/B/C fail",
            "",
            "**Recommendation**: Option A (zigzag duration 0.5–1.0s). "
            "One parameter, one line, within the recipe.",
        ]
    else:
        lines += [
            "### Implication for M1.3",
            "",
            "Apex curvature IS in the training distribution. If apex overshoot persists "
            "after M1.2 (entropy fix), investigate:",
            "1. **Policy capacity**: 256-hidden 3-layer may not represent sharp-turn dynamics",
            "2. **Reward shaping**: exp(-d²) gives weak gradient far from reference; "
            "consider exp(-d) or negative L2",
            "3. **Observation horizon**: 10×50ms = 500ms lookahead; figure-eight apex "
            "turn window is ~200ms — policy may not see it coming early enough",
        ]

    lines.append("")
    out_path = REPO_ROOT / "notes" / "trajectory_coverage_analysis.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
