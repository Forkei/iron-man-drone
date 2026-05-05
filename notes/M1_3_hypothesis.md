# M1.3 Hypothesis — Polynomial Generator Fix

**Date**: 2026-05-03  
**Status**: Pre-training  
**Experiment**: `experiments/m1_3_polynomial_fix/`

---

## Root cause confirmed

The polynomial trajectory generator in M1/M1.1/M1.2 was structurally wrong. It applied a quintic scalar function `h(τ) = 10τ³ - 15τ⁴ + 6τ⁵` to straight-line direction vectors:

```python
# OLD (broken): straight-line path, velocity=0 at every junction
return p0 + (p1 - p0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
```

This produced paths with:
- κ = 0 **everywhere** (analytically exact, confirmed by 1000-trajectory empirical analysis)
- Velocity = 0 at every waypoint (stop-pivot-go)
- The "polynomial" described only the *speed profile*, not the *path geometry*

The paper (arXiv 2412.11764 §V-A1-b) specifies *"continuity of the first, second, and third derivatives at the junctions between consecutive segments"* — meaning nonzero velocity at junctions, genuinely curved path within each segment.

## What M1.3 changes

Single fix: replace the scalar-blend polynomial with a true C2-continuous piecewise quintic:

```python
# NEW: true 5th-degree polynomial, random nonzero vel/acc at interior waypoints
# 6 boundary conditions per segment (pos, vel, acc at start/end) → closed-form solve
coeffs = _solve_quintic_coeffs(p0, v0, a0, p1, v1, a1, T)
p(tau) = coeffs[0] + coeffs[1]*tau + ... + coeffs[5]*tau^5
```

Interior waypoints: `vel ~ Uniform[-0.8, 0.8] m/s`, `acc ~ Uniform[-2.0, 2.0] m/s²`.  
Start/end: vel=0, acc=0 (hover transitions).

**Everything else identical to M1.1** (reward fix kept, entropy_coeff=1e-3 unchanged).

## Validation (pre-training)

From `analyze_trajectory_coverage.py` on the new generator (N=1000):

| | M1/M1.1/M1.2 | M1.3 |
|---|---|---|
| Polynomial κ_max p50 | **0.000 m⁻¹** | **293 m⁻¹** |
| Polynomial κ_max p10 | 0.000 m⁻¹ | 37 m⁻¹ |
| Coverage of fig-8 apex (κ=4.789 m⁻¹) | **0.0%** | **100.0%** |
| Coverage of fig-8 apex ω | 0.0% | 99.6% |

The figure-eight apex is now **in-distribution** for polynomial training trajectories. 100% of trajectories expose curvature at or above the 4.789 m⁻¹ target.

---

## Predictions

### My prediction (Claude)

**Expected best MED on figure_eight_normal: 0.035–0.065m**

The polynomial generator being broken was the structural root cause behind the 0.105m plateau. With it fixed:

**Strong improvement factors:**
- The drone will train on genuinely curved paths for the first time
- κ > 4.789 m⁻¹ exposure = 100% vs 0% — the policy now sees the type of maneuver the eval requires
- C2 continuity means transitions are smooth, giving the policy useful experience with sustained lateral acceleration at speed

**Why I might not clear 0.056m immediately:**
- Entropy will still collapse (~epoch 150 with 4096 envs, based on M1.2 observation)
- The entropy collapse limits how much exploration happens in the first 150 epochs
- New training trajectories have κ_max much higher (293 m⁻¹ median) than the eval target (4.789 m⁻¹) — the policy is still being asked to generalize from very-tight-turn training to moderate-turn eval
- 15k epochs at 4096 envs should give enough training signal to overcome entropy collapse

**Most likely outcome: 0.040–0.065m** (improvement of 40–60% over 0.105m baseline).  
**30% chance of clearing 0.056m threshold** — depends on whether the policy generalizes well from κ~293 training distribution to κ~5 eval target.

### User's prediction

MED 0.07–0.09m — meaningful improvement from the polynomial fix, but still above threshold because:
- Training curvatures (κ_max median ~293 m⁻¹) are still far higher than eval curvatures (~5 m⁻¹)
- The policy will learn to handle *some* curvature but may not generalize to the specific curvature magnitude of the figure-eight apex
- Entropy collapse still limits exploration

---

## What to watch in the wake-up report

**(a) Entropy zero-crossing epoch**  
- M1.2 (4096 envs, 3e-3): collapsed epoch ~149
- M1.3 (4096 envs, 1e-3): expect collapse ~epoch 130-160 (similar pattern)
- If collapse is significantly earlier, the larger batch is overcoming entropy faster

**(b) Reward curve shape**  
- M1/M1.1/M1.2: reward plateau ~1.33–1.35
- M1.3: if polynomial fix works, reward should climb higher (better tracking = higher exp(-d²))
- Watch for reward still climbing at epoch 10k+ (sign the policy is learning better maneuvers)

**(c) MED at epoch 1000, 5000, 10000, 15000**  
- If MED is still ~0.10m at epoch 1000: the fix isn't helping, investigate
- If MED drops below 0.08m by epoch 5000: the fix is working, entropy is the remaining blocker
- If MED drops below 0.056m: M1 passes

**(d) Apex:straight ratio**  
- M1/M1.2 baseline: 11.3× (apex error much worse than straight)
- M1.3 prediction: <5× (the new training distribution teaches sustained turning)
- If ratio stays at 11×: the curvature magnitude mismatch (training κ >> eval κ) is the remaining problem

---

## Decision tree after M1.3

| MED | Apex:straight | Interpretation | Next |
|---|---|---|---|
| < 0.056m | Any | M1 passes | Ship M1, proceed to M2 |
| 0.056–0.080m | < 5× | Good tracking, entropy is blocker | M1.4: entropy scheduling |
| 0.056–0.080m | > 8× | Still apex-specific problem | M1.4: tighten poly curvature distribution |
| > 0.080m | > 8× | Polynomial fix insufficient | Diagnose: check training reward curve slope |
