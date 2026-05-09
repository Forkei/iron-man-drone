# M1.3 Final Results — epoch_013000

**These numbers are canonical as of 2026-05-09 (eval_suite.py, GPU MJX backend, seeds [42, 99, 7]).
Previous numbers from eval_m1_full.py / CPU mujoco are preserved in git history and explained below.**

**Checkpoint:** `experiments/m1_3_polynomial_fix/m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000`  
**Eval script (canonical):** `scripts/eval_m1_suite.py` → `src/iron_man_drone/evaluation/eval_suite.py`  
**Results file:** `experiments/m1_suite_results.json`

---

## Canonical Results (eval_suite.py, 2026-05-09)

| Trajectory | MED (3-seed mean) | Seed range | Gate | Status |
|---|---|---|---|---|
| figure_eight_slow (T=15s) | **0.0200 m** | [0.0193, 0.0205] | ≤ 0.050 m | ✓ PASS |
| figure_eight_normal (T=5.5s) | **0.0402 m** | [0.0387, 0.0414] | ≤ 0.056 m | ✓ PASS |
| figure_eight_fast (T=3.5s) | **0.0938 m** | [0.0922, 0.0950] | ≤ 0.150 m | ✓ PASS |
| pentagram_slow | 0.0581 m | [0.0575, 0.0590] | — | — |
| pentagram_fast | 0.0675 m | [0.0672, 0.0679] | — | — |
| polynomial | 0.0652 m | [0.0645, 0.0662] | — | — |
| zigzag | 0.0434 m | [0.0429, 0.0441] | — | — |

**Crashes:** 0 / 21 rollouts (7 traj × 3 seeds). All trajectories completed all 1000 steps.

**Eval methodology:** T/4 phase offset for figure-eight (offset=375/138/88 steps for slow/normal/fast).  
Pentagram/polynomial/zigzag at t=0. Crash-only termination (no step timeout). GPU MJX lax.scan.

---

## OVERALL: PASS

All three gated trajectories pass M1 thresholds under the canonical methodology.

---

## Previous Results (eval_m1_full.py, CPU mujoco, ~2026-05-05)

| Trajectory | Old MED | New MED | Delta | Why it changed |
|---|---|---|---|---|
| figure_eight_slow | 0.0170 m | 0.0200 m | +0.003 m | GPU vs CPU backend |
| figure_eight_normal | 0.0369 m | 0.0402 m | +0.003 m | GPU vs CPU backend |
| figure_eight_fast | 0.0895 m | 0.0938 m | +0.004 m | GPU vs CPU backend |
| pentagram_slow | 0.0544 m | 0.0581 m | +0.004 m | GPU vs CPU backend |
| pentagram_fast | 0.0639 m | 0.0675 m | +0.004 m | GPU vs CPU backend |
| polynomial | 0.0162 m | 0.0652 m | **+0.049 m** | Methodology (see below) |
| zigzag | 0.0265 m | 0.0434 m | **+0.017 m** | Methodology (see below) |

### Backend shift (figure-eight and pentagram): systematic +0.003–0.004 m

The GPU MJX lax.scan backend and CPU mujoco backend produce systematically different numbers for the same policy on the same trajectory. The difference is ~0.003 m on figure-eight normal, consistent across runs. This is documented in the `eval_suite.py` smoke test (expected_med=0.040, not 0.037). Both numbers are reproducible; the difference is not measurement noise.

The GPU backend is now the reference. All M2 numbers use GPU MJX eval_suite.py, so M1 canonical numbers must also come from the same backend for valid comparison.

### Polynomial and zigzag: methodology difference, not just backend

The old eval fixed the first waypoint of random polynomial/zigzag trajectories to the drone's spawn position (0, 0, 1), eliminating cold-start tracking error. The eval_suite.py uses seeds [42, 99, 7] with random trajectories whose first waypoints are not constrained to spawn — initial tracking error is non-zero.

This is the correct methodology for measuring trajectory-following capability. The old 0.016 m polynomial number was artificially low because the trajectory started exactly at the drone. The new 0.065 m is the honest number.

Note: the SimpleFlight paper reports 0.032 m for polynomial — between old (0.016) and new (0.065). Different seeds, possibly different trajectory generator. The comparison is informational only; no polynomial gate exists.

---

## M1 Gate Outcome (canonical numbers)

| Gate | Threshold | Result | Status |
|---|---|---|---|
| figure_eight_slow | ≤ 0.050 m | 0.020 m | ✓ PASS |
| figure_eight_normal | ≤ 0.056 m | 0.040 m | ✓ PASS |
| figure_eight_fast | ≤ 0.150 m | 0.094 m | ✓ PASS |
| Zero crashes | 0 crashes | 0 / 21 | ✓ PASS |

**M1 is complete and ships. git tag: m1-baseline.**

---

## Comparison to Paper (SimpleFlight)

| Trajectory | Ours (canonical) | Paper | Ratio |
|---|---|---|---|
| figure_eight_slow | 0.020 m | 0.016 m | 1.25× |
| figure_eight_normal | 0.040 m | 0.028 m | 1.43× |
| figure_eight_fast | 0.094 m | 0.051 m | 1.84× |

The gap is real — SimpleFlight used longer training and more aggressive tuning. The M1 targets (conservative) are all met. M2 trade-off: DR costs ~+0.017 m on figure_eight_normal (see M2_results.md), which is acceptable for fault tolerance.
