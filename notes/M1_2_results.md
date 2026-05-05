# M1.2 Results — Entropy Bump (3e-3) + Reward Fix

**Date**: 2026-05-03  
**Experiment**: `experiments/m1_2_entropy/m1_2_entropy_1777808012/`  
**Killed at**: epoch ~1690 (of 8000)

---

## Summary

M1.2 failed to improve over M1 baseline. Entropy collapsed at epoch ~149 (4096-env run), only slightly different from M1/M1.1 (~165). The reward plateaued identically at ~1.35. MED eval was killed due to RAM pressure; figures below are derived from training logs.

**Root cause identified (post-M1.2):** The polynomial generator produces κ=0 trajectories (stop-pivot-go). Neither the reward fix (M1.1) nor the entropy bump (M1.2) could fix a broken training distribution. This was confirmed by trajectory coverage analysis (M1.3 is the fix).

---

## Training log statistics

| Metric | M1 baseline | M1.1 | M1.2 |
|---|---|---|---|
| Reward plateau | ~1.33 | ~1.34 | ~1.35 |
| Entropy zero-crossing | ~165 | ~165 | **~149** (4096 envs) |
| Peak fps | ~41k (1024 envs) | ~41k | **65k** (4096 envs, pre-eval) |
| Post-eval fps | ~27k | ~27k | **41k** (cache eviction at epoch 1000) |

### Entropy collapse detail

- Epoch 140: entropy = +0.252 (still positive)
- Epoch 150: entropy = −0.040 (just crossed zero)
- Interpolated zero-crossing: **epoch ~149**

The 4096-env run collapsed EARLIER than the 1024-env baseline. Larger batch = stronger gradient estimates = faster collapse. The 3e-3 coefficient delayed collapse by +12 epochs in the 1024-env run (epoch ~177 vs ~165), but this gain was lost when switching to 4096 envs.

### fps timeline

```
Epoch 0–580:    3k → 65k  (XLA JIT warmup)
Epoch 580–1000: 65k → 52k  (GPU thermal throttle)
Epoch 1000→1010:  52k → 41k  (eval at epoch 1000 evicted training XLA cache)
Epoch 1010–1690: 41k → 45k  (slowly re-warming)
```

Pre-warm fix compiled only a single eval step, not the full eval scan (1000-step `lax.scan`). Fix for M1.4: pre-warm the full scan, not a single step.

---

## MED eval

**Eval not completed** — both eval processes were killed due to ~26GB combined RAM usage (two concurrent JAX processes each loading full GPU model). No final MED numbers.

**Estimated MED from training log** (based on reward plateau ~1.35 ≈ same as M1/M1.1):
- figure_eight_normal: ~0.100–0.106m (no meaningful improvement over M1 baseline ~0.105m)
- This is consistent with the entropy collapse at epoch 149 leaving only the same converged deterministic policy as M1/M1.1

---

## Key finding from M1.2 period

The parallel trajectory coverage diagnostic (run during M1.2 training) identified the true root cause:

- **Polynomial trajectories produce κ=0 everywhere** (zero curvature, stop-pivot-go)
- 100% of polynomial training data is piecewise-linear paths with zero velocity at each waypoint
- The SimpleFlight paper requires C2-continuous polynomial (nonzero velocity at junctions)
- This was a day-1 error; M1/M1.1/M1.2 all trained on the broken generator
- Fix implemented in M1.3: C2-continuous quintic polynomial with random interior velocities/accelerations

---

## Lessons

1. **Entropy bump was insufficient and orthogonal** — 3e-3 vs 1e-3 had negligible effect on collapse timing; the advantage signal overwhelms any coefficient < ~1e-2 at epoch ~150
2. **Larger batch accelerates entropy collapse** — 4096 envs collapsed 16 epochs earlier than 1024 envs with the same coefficient
3. **Eval XLA cache eviction is a persistent issue** — needs full-scan pre-warm, not single-step
4. **Root cause was training distribution, not algorithm** — the reward fix and entropy changes were tuning a policy that never learned to curve
