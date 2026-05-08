# M2 Ablation Eval Clarification
**Date:** 2026-05-08  
**Status:** Blocking — Phase 2 hold pending spec-target resolution

---

## The Question

After the M2-no-DR ablation (same architecture, DR disabled) showed 0.0800m inline MED at epoch 7000, the claim was made that "the architecture ceiling is ~0.080m and the 0.037m target is unreachable." The user correctly flagged that this conclusion skipped a measurement question: the 0.037m M1.3 number was produced by the **corrected eval with T/4 phase offset**, while the M2 inline eval and `eval_m2_full.py` both use **t=0** — the same broken methodology that inflated M1.3's training eval to 0.069m.

Two checks were required:
1. Verify the systematic offset between M2's current eval methodology and M1.3's correct eval methodology.
2. Run M2-no-DR epoch_007000 (best inline checkpoint) through the correct methodology. If it lands near 0.037m, the architecture is fine and the gap is DR-attributable. If it lands at ~0.080m anyway, the architecture itself is the problem.

---

## Check 1 — Methodology Comparison

### The T/4 phase offset issue (recap from M1.3)

SimpleFlight initializes the figure-eight trajectory at `traj_t0 = T/4`, placing the first reference point at (0, 0, 1) — exactly where the drone spawns. Initial XY error: **0.0 m**.

Our implementation (M1 inline eval, M2 inline eval, `eval_m2_full.py`) uses `traj_t0 = 0`, placing the first reference at (1, 0, 1). Initial XY error: **1.0 m**.

The policy acquires the trajectory within ~100 steps (error decays from 1.0m to ~0.025m), but the mean over the full 1000 steps includes this acquisition phase, inflating the MED by ~**0.031–0.032 m** for figure_eight_normal.

### Which evals have the bug?

| Script | T/4 offset applied? | Notes |
|---|---|---|
| `scripts/eval_m1_full.py` (M1.3 final) | **YES** | Correct. Used to produce M1.3's 0.037m. |
| `scripts/train_m1.py` `_run_med_eval` | NO | Produced inflated 0.069m during training. |
| `scripts/train_m2.py` `_eval_episode_jit` | NO | Inline training eval — same bug. |
| `scripts/eval_m2_full.py` | NO | Full eval suite used for M2 Phase 1 — same bug. |

**`eval_m2_full.py` has the same methodology bug as the old M1 training eval.** All M2 figure-eight MED numbers reported to date are cold-start inflated by ~0.031m.

---

## Check 2 — M2 No-DR Epoch 7000 on Correct Methodology

Script: `scripts/eval_m2_methodology_check.py`  
Checkpoint: `experiments/m2_no_dr_ablation/m2_no_dr_ablation_1778262087/checkpoints/epoch_007000`  
Methodology: CPU MuJoCo (identical to `eval_m1_full.py`), 50-dim M2 obs, priv_state = [1,1,1,1,1,0,0,0] (nominal).

| Eval | Initial XY error | figure_eight_normal MED | Status |
|---|---|---|---|
| t=0 (current M2 methodology) | 1.0000 m | **0.0748 m** | inflated |
| T/4 (M1.3 correct methodology) | 0.0081 m | **0.0437 m** | corrected |
| Systematic offset | — | **0.0310 m** | (matches M1.3's 0.0321m offset) |

### Reference: M1.3 epoch_013000 (from `notes/M1_3_eval_methodology.md`)

| Eval | MED |
|---|---|
| t=0 (old training eval) | 0.0690 m |
| T/4 (corrected eval) | 0.0369 m |
| Offset | 0.0321 m |

---

## Answers to the Two Questions

### Does the methodology mismatch explain the apparent gap?

**Partially, but not fully.** The mismatch inflates all M2 figures by ~0.031m, which changes the picture significantly:

| Metric | Previously reported (t=0) | Corrected (T/4) | Spec target |
|---|---|---|---|
| M2+DR final nominal | 0.0908 m | **~0.060 m** (extrapolated) | 0.037 m |
| M2 no-DR epoch_007000 | 0.0800 m | **0.0437 m** (measured) | 0.037 m |
| M1.3 epoch_013000 | 0.0690 m | **0.0369 m** (measured) | 0.037 m |

The reported "2.5× gap" (0.091m vs 0.037m) was largely a methodology artifact. The actual gap for M2+DR is approximately **1.6×** (0.060m vs 0.037m). The no-DR gap is approximately **1.18×** (0.044m vs 0.037m).

### Is the architecture preserving M1.3's nominal capability?

**Mostly, but not fully.** M2 no-DR with correct methodology lands at 0.0437m vs M1.3's 0.0369m — a **0.007m real gap** (about 18% above M1.3). This is not the 2.5× regression we inferred from the inline eval numbers. It is, however, a real and detectable gap.

Possible sources of the remaining 0.007m:
1. **Epoch**: M2 no-DR was stopped at epoch 7000 by the trend gate; it may not have fully converged. The inline eval showed improvement from 5k→7k (0.0828→0.0800), so more training might close this.
2. **Architecture overhead**: The extra 8 priv_state dims (always [1,1,1,1,1,0,0,0] in no-DR mode) add constant-value inputs that weren't present in M1.3. The network may need more epochs to learn to ignore them.
3. **Single eval variance**: Both numbers are single-seed single-rollout eval points; ±0.005m variance is plausible.

The gap is **not catastrophic** and almost certainly not caused by a fundamental architecture flaw.

---

## Implications for M2 Phase 1

The M2+DR final nominal MED, corrected to the proper methodology, is approximately **0.060m** — not 0.091m.

The spec target (0.037m) was correctly set against M1.3's actual 0.037m result (T/4-corrected). M2+DR is 1.6× over that target on the corrected basis. This is a real gap, driven by:
- DR (adds ~0.016m: 0.044m no-DR → ~0.060m with-DR, both corrected)
- Possible residual architecture effect (~0.007m no-DR gap vs M1.3)

**eval_m2_full.py must be corrected** (T/4 offset for figure-eight trajectories) before any final M2 Phase 1 go/no-go can be made against spec. The current full-eval numbers are not comparable to M1.3's.

---

## Required Fix to `eval_m2_full.py`

Apply T/4 offset for figure-eight trajectories, matching `eval_m1_full.py`. For figure_eight_normal (T=5.5s): offset_steps = round(5.5/4 / DT) = 138 steps. Similar for slow (T=15s → 375 steps) and fast (T=3.5s → 88 steps).

The `eval_episode` lax.scan function needs to accept an `offset_steps` argument and shift both:
1. The `ref_xy` precomputed array (start from step `offset_steps`, not step 0)
2. The initial `_build_obs` call (use `state.step + offset_steps` for the trajectory reference)

This is a ~10-line change. Do not change pentagram, polynomial, or zigzag — those use t=0 correctly (as in `eval_m1_full.py`).

---

## Decision Gate

The following must happen before Phase 2 begins:

1. Fix `eval_m2_full.py` with the T/4 offset for figure-eight trajectories.
2. Re-run the full eval suite against the M2+DR 15k checkpoint with the corrected methodology.
3. Report the corrected MED table and compare to spec.
4. Decide: accept the corrected gap and revise spec targets, or extend M2 Phase 1 training.

**Do not proceed to Phase 2 until step 4 is resolved.**
