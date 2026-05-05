# M1.3 Final Results — epoch_013000

**Date:** 2026-05-05  
**Checkpoint:** `experiments/m1_3_polynomial_fix/m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000`  
**Eval methodology:** Corrected initialization matching SimpleFlight (arXiv 2412.11764).  
Figure-eight: T/4 phase offset applied (traj starts at drone spawn position).  
Pentagram/poly/zigzag: traj_t0=0, poly/zigzag first waypoint fixed to (0,0,1).  
Aggregation: arithmetic mean over full 1000-step episode (no exclusion window).

---

## Full Results Table

| Trajectory | M1 Target | M1 Baseline (old) | M1.3 (old eval, broken) | **M1.3 (corrected eval)** | Paper (SimpleFlight 100Hz) |
|---|---|---|---|---|---|
| figure_eight_slow | < 0.050 m | — | — | **0.0170 m ✓ PASS** | 0.016 m |
| figure_eight_normal | < 0.056 m | ~0.105 m | 0.069 m (cold-start inflated) | **0.0369 m ✓ PASS** | 0.028 m |
| figure_eight_fast | < 0.150 m | — | — | **0.0895 m ✓ PASS** | 0.051 m |
| pentagram_slow | — | — | — | 0.0544 m | 0.024 m |
| pentagram_fast | — | — | — | 0.0639 m | 0.045 m |
| polynomial (3 seeds) | — | — | — | 0.0162 ± 0.0018 m | 0.032 m |
| zigzag (3 seeds) | — | — | — | 0.0265 ± 0.0014 m | 0.052 m |

**Crashes:** 0 / 9 rollouts. All trajectories completed clean 1000 steps.

---

## OVERALL: PASS

All three thresholded trajectories (figure-eight slow, normal, fast) are within M1 targets.  
Zero crashes on all trajectory types including OOD held-outs.

---

## Comparison: Old Eval vs Corrected Eval vs Paper

| | Old inline eval | Corrected eval | Paper target |
|---|---|---|---|
| figure_eight_normal | 0.069 m | **0.037 m** | 0.028 m |
| figure_eight_slow | — | **0.017 m** | 0.016 m |
| figure_eight_fast | — | **0.090 m** | 0.051 m |

**Why the old eval was wrong:** Our `_run_med_eval` used `traj_t0=0`, placing the figure-eight reference at (1,0,1) while the drone spawned at (0,0,1). The 1m cold-start inflated the mean by ~0.044m. SimpleFlight uses `traj_t0 = T/4` (confirmed in `track.py`), which places the reference at (0,0,1) = drone spawn → zero initial error. The corrected eval applies the same T/4 offset.

**Gap vs paper:** Figure-eight normal 0.037m vs paper 0.028m — a real 1.3× gap. Figure-eight slow essentially matches the paper (0.017m vs 0.016m). Figure-eight fast is 1.75× above paper (0.090m vs 0.051m). These gaps likely reflect that the paper trained at 100Hz with more aggressive DR; our implementation is at 100Hz but without all paper tuning details. The M1 targets (which we set conservatively) are all met.

---

## M1 Ship Decision

**M1 is complete.** The policy meets all M1 success criteria:
- figure_eight_normal MED < 0.056 m ✓ (0.037 m)
- figure_eight_slow MED < 0.050 m ✓ (0.017 m)
- figure_eight_fast MED < 0.150 m ✓ (0.090 m)
- All benchmark trajectories complete without crash ✓
- Stable reward/loss curves, polynomial curvature fix confirmed ✓

**Next steps:**
1. `git init` + `git tag m1-baseline` (repo not yet initialized)
2. Begin M2 planning
