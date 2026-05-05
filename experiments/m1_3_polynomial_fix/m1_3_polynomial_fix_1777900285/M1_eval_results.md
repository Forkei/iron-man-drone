# M1.3 Full Eval Results — epoch_013000

**Date:** 2026-05-05  
**Checkpoint:** `/mnt/c/Users/forke/Documents/Drone/iron-man-drone/experiments/m1_3_polynomial_fix/m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000`  
**Eval methodology:** Corrected initialization matching SimpleFlight (arXiv 2412.11764).  
Figure-eight: T/4 phase offset applied. Pentagram/poly/zigzag: traj_t0=0, poly/zigzag first waypoint fixed to (0,0,1).

---

## Results Table

| Trajectory | M1.3 MED (corrected eval) | M1 Threshold | Pass/Fail | Paper (SimpleFlight 100Hz) |
|---|---|---|---|---|
| figure_eight_slow | 0.0170 m | 0.050 m | **PASS** | 0.016 m |
| figure_eight_normal | 0.0369 m | 0.056 m | **PASS** | 0.028 m |
| figure_eight_fast | 0.0895 m | 0.150 m | **PASS** | 0.051 m |
| pentagram_slow | 0.0544 m | — | — | 0.024 m |
| pentagram_fast | 0.0639 m | — | — | 0.045 m |
| polynomial | 0.0162 ± 0.0018 m | — | — | 0.032 m |
| zigzag | 0.0265 ± 0.0014 m | — | — | 0.052 m |

---

## Comparison: Old Eval vs Corrected Eval vs Paper

| Metric | Old eval (cold-start) | Corrected eval (T/4 offset) | Paper target |
|---|---|---|---|
| figure_eight_normal MED | ~0.069 m (mean w/ 1m cold-start) | 0.0369 m | 0.028 m |
| figure_eight_normal steady-state | ~0.025 m | — | — |

**Root cause of old eval gap:** Our `_run_med_eval` initialized the trajectory at t=0,
placing the reference at (1,0,1) while the drone spawned at (0,0,1). SimpleFlight uses
`traj_t0 = T/4`, placing the reference at (0,0,1) = drone spawn → zero initial error.
Confirmed from SimpleFlight `track.py`. Corrected eval applies the same T/4 offset.

---

## M1 Ship Decision

**VERDICT: M1 PASSES.** All thresholded trajectories below target.

Next steps:
1. `git tag m1-baseline` on this commit
2. Begin M2 planning