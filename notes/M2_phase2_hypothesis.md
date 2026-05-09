# M2 Phase 2 Hypothesis — Causal Adaptation Encoder
**Date:** 2026-05-09  
**Spec:** `notes/M2_phase2_spec.md`  
**Status:** COMPLETE — all gates passed 2026-05-09

**GATE: This document must be complete before any Phase 2 script is run.**

---

## What this run tests

Whether a flat MLP encoder ϕ (2300→256→128→8→tanh) can predict the privileged physical
state e_t = [η₁, η₂, η₃, η₄, m_scale, F_x, F_y, F_z] from 0.5 s of (obs, action) history,
and whether deploying ϕ in place of ground-truth e_t preserves Phase 1's fault-tolerant
tracking performance within the ≤14%/≤27% degradation budget.

---

## Phase 1 reference numbers (corrected eval, T/4-corrected, 3-seed mean)

| Condition | figure_eight_normal MED |
|---|---|
| Nominal (ground-truth e_t) | 0.057 m |
| Fault η=0.70 (ground-truth e_t) | 0.079 m |

Phase 2 pass criteria:
- Nominal (ê_t deployed): ≤ 0.065 m
- Fault η=0.70 (ê_t deployed): ≤ 0.100 m

---

## Pre-implementation checklist

- [ ] Phase 1 gate passed: `notes/M2_phase1_corrected_results.md` exists, nominal=0.057m, gate PASS
- [ ] Spec reviewed: `notes/M2_phase2_spec.md` references corrected Phase 1 numbers
- [ ] Checkpoint verified: `m2_phase1_baseline_1778244202/checkpoints/final` loads correctly
- [ ] DR setup confirmed: fault_prob=0.70, η∈[0.5,1.0), mass±20%, kf±30%, wind=0
- [ ] obs_base stripping confirmed: encoder input uses 42-dim obs (no priv_state), not 50-dim
- [ ] Normalization confirmed: η→(x−0.75)/0.25, m_scale→(x−1.0)/0.20, wind→0

---

## Predictions

### Data collection (Step 1)

Expected episode count: 20480 (10 batches × 2048 envs).  
Expected fault episode fraction: ~70% (matches fault_prob=0.70).  
Expected crash rate: <5% (Phase 1 policy handles DR well under training distribution).  
Expected collection time: ~30 min.

### Offline MSE (Step 3)

**User prediction: normalized MSE ≈ 0.35.**

Note: this is much higher than the spec gate (≤ 0.02). Either the prediction is on the
wrong scale (raw vs normalized η), or the user expects the flat MLP to underperform.
The spec's 0.02 target corresponds to RMSE ≈ 0.14 per normalized dim (≈ 0.035 on raw η
scale), which should be achievable if the history contains enough information to identify
the fault. If the user's prediction is correct, the encoder fails the offline gate and
Phase 2 is blocked until failure mode F4 is diagnosed.

**Per-channel prediction:**
- η dims (0–3): user predicts EASIER to predict than mass scale.
  Rationale: a rotor fault causes a clear physical asymmetry observable from
  (vel, R) history — the drone must compensate with roll/pitch moments that show
  up in the obs. Mass scale is harder because it scales all forces uniformly
  and is harder to decouple from kf variation.
  Counter-argument: fault is a step change at episode start, so the encoder
  has to identify it from the very first steps — no dynamics transient to read.
  Call: prediction is that rotors are easier, but this is low-confidence.

- Wind dims (5–7): always-zero targets. Encoder will learn to output near-constant
  values regardless of history. Predicted effective MSE: ~0 (trivial to predict).

**Confidence:** Low — no prior Phase 2 runs to calibrate against.

### Closed-loop eval (Step 4)

**Prediction:** if offline MSE < 0.02, closed-loop will pass. If MSE ≈ 0.05–0.10,
expect ~10–20% degradation above Phase 1 (may still pass gate). If MSE > 0.20, expect
instability at episode start from zero-padding artifacts.

**Most likely failure mode:** F5 — zero-padding instability in the first 50 steps. The
encoder hasn't seen the trajectory yet and outputs a bad ê_t, which could push the actor
toward an unstable region briefly before the history fills in.

---

## Time-box rules

| Step | Time limit | Action if exceeded |
|---|---|---|
| Data collection | 1.5 hours | Stop, check GPU util, reduce N_ENVS if OOM |
| Encoder training | 2 hours for 2000 epochs | Stop at plateau; check MSE curve before continuing |
| Closed-loop eval | 20 min | If crashes detected mid-eval, stop and diagnose before all seeds |

---

## Discipline rules

- **Do not touch Phase 1 actor weights.** If closed-loop fails badly, the fix is in encoder
  or data collection, not the actor. If the actor seems unstable to ê_t noise → F7 protocol.
- **One variable at a time.** History window size, architecture, LR, data collection changes
  are each separate experiments. Do not compound changes.
- **Do not proceed to M3** until Phase 2 closed-loop gate is confirmed. The gate is
  ≤0.065m nominal / ≤0.100m fault on figure_eight_normal with ê_t deployed.

---

## Notes (filled in during implementation, 2026-05-09)

### Data collection (Step 1)
- 20,480 episodes collected in **3.0 min** (not 30 min — JIT warmup was 28 s, then 270–340k fps)
- Fault rate: 68.9% (expected 70%) ✓
- Sanity: η mean=0.750 ✓, mass=0.999±0.115 ✓, wind=0.0 ✓
- Disk: 2.89 GB, 10 chunks × 2048 episodes

### Offline training (Step 2)
- Training time: **0.9 min** for 2000 epochs (Adam lr=5e-4, batch=4096)
- Best val MSE (epoch 1850): **0.01561** — gate ≤0.020 PASS ✓
- Per-channel MSE (best checkpoint):
  - η mean: 0.01145 — gate ≤0.030 PASS ✓
  - m_scale: 0.07576 — bottleneck (as predicted)
  - wind: 0.00109 — near-zero ✓
- Hypothesis check:
  - User predicted MSE ≈ 0.35 — actual 0.0156. WAY over (encoder learned far better)
  - η easier than mass: CONFIRMED (η=0.011 vs mass=0.076)
  - Wind near-zero: confirmed (MSE=0.001)

### Closed-loop eval (Step 3)
- JIT compile: 17.5 s, total eval time: 2.2 min
- **figure_eight_normal gate: PASS ✓**
  - Nominal: 0.0569 m (−0.9% vs Phase 1's 0.0574 m — essentially zero degradation)
  - Fault η=0.70: 0.0807 m (+2.2% vs Phase 1's 0.0790 m — well within ≤27% budget)
- Notable: figure_eight_fast + fault crashes at 2/3 seeds (not gated; high-agility + fault edge case)
- polynomial fault: 0.141 m (no crash)
- zigzag: minimal degradation between nominal and fault (0.053 vs 0.056 m)

### Conclusion
Phase 2 gate PASSED. Causal encoder ϕ(history) → ê_t enables the Phase 1 actor to operate
with near-zero nominal degradation and <3% fault degradation on the primary eval trajectory.
The m_scale channel (MSE=0.076) is the main uncertainty; it doesn't appear to harm closed-loop
performance because the actor is relatively robust to mass estimation error.
