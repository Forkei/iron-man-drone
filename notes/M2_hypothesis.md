# M2 Hypothesis — Phase 1 Baseline

**Date:** 2026-05-05  
**Experiment dir:** `experiments/m2_phase1_baseline/`  
**Status:** Pre-training — hypothesis written, awaiting implementation

**GATE: This document must be complete and reviewed before any training run starts.**

---

## What this run tests

Whether the RMA Phase 1 setup (single-rotor DR at 70% episode rate, mass ±20%, z-augmented actor) converges to a working fault-tolerant policy, and whether nominal tracking performance survives the z augmentation or regresses.

---

## Setup checklist (verify before launching)

- [ ] Single-rotor DR wired: log e_t at episode start for 10 episodes, confirm η_j ≠ 1.0 in ~70% of episodes, exactly one rotor degraded per fault episode
- [ ] No wind in Phase 1 training: confirm F_wind = 0 in env reset (wind is OOD eval only)
- [ ] μ sanity: run 100 random e_t samples, variance(z) > 0.01 per dimension (encoder not collapsed)
- [ ] Entropy vs reward: at random init, entropy contribution < 10% of reward magnitude
- [ ] Actor input 50-dim (not 42): print actor_obs.shape, confirm [batch, 50]
- [ ] Critic input 51-dim (not 43): print critic_obs.shape, confirm [batch, 51]
- [ ] Crash rate baseline: random policy crash rate on nominal conditions (for reference)

---

## Predictions

### User prediction (gut feel, stated before training)

**Expected nominal MED at epoch 5,000:** ~0.065m  
(Slower to converge than M1.3 due to expanded input space and noisier training distribution; likely misses the ≤0.055m epoch-5k gate)

**Expected nominal MED at epoch 13,000:** ~0.045–0.050m  
(Converges but doesn't reach M1.3's 0.037m — z augmentation costs some nominal precision. Misses the ≤0.037m hard requirement.)

**Predicted fault performance (one rotor at η=0.70, epoch 13,000):** Unknown — lower confidence. If the encoder is working, possibly in range; if z is underused, performance ≈ RL-DR baseline.

**Expected crash rate in epochs 1–500:** ~20–30% (elevated but not catastrophic; training proceeds, just noisier early on)

**Most likely failure mode:** F1 — nominal regression. The z augmentation adds noise to the actor's input space that the policy doesn't fully learn to filter out under nominal conditions, costing 20–35% on nominal MED.

**Predicted Phase 2 outcome:** Fine. Once Phase 1 converges, the adaptation encoder should train cleanly to MSE ≤ 0.02 with no surprises.

**Confidence:** Medium — gut feeling. The overall shape of training (runs to completion, learns to fly, misses the hard nominal gate) feels right; the specific numbers are soft.

---

## Success signal to watch at each checkpoint

| Epoch | Nominal MED threshold | Crash rate | Action if exceeded |
|---|---|---|---|
| 1,000 | ≤ 0.090 m | < 20% | Stop and diagnose (is z collapsing? is DR too aggressive? check crash rate vs M1.3 epoch 1k) |
| 5,000 | ≤ 0.055 m | < 10% | If yes: continue. If no: stop and re-read spec §F1 before touching any hyperparameter. |
| 10,000 | ≤ 0.045 m | < 3% | If yes: continue. If no: abort — do not train to 15k hoping it improves. |
| 13,000 | ≤ **0.037 m** | 0% | M2 Phase 1 nominal pass. This is the M1.3 target, not a relaxed version. |

**Time-box rule:** If nominal MED is not below 0.060m at epoch 5,000, stop the run. Do not tweak and continue. Re-read M2_spec §F1 and diagnose before launching the next run.

**If prediction is correct (nominal ~0.045–0.050m at epoch 13k):** The epoch-10k gate (≤0.045m) may fire or nearly fire. Either way, diagnosis before the next run — do not just relax the target. Per spec §F1: check z variance (is μ collapsed?), check whether actor gradient is flowing through z (log ∂L/∂z), check crash rate history. Most likely fix: curriculum fault ramp (0→0.70 over first 5k epochs) to give the policy a cleaner signal early.

---

## What to do on success

- [ ] Archive Phase 1 checkpoint to `experiments/m2_[name]/checkpoints/epoch_[N]/`
- [ ] Run full M2 eval suite (nominal + all fault scenarios from spec)
- [ ] Collect Phase 2 training data (20,000 rollouts with full DR)
- [ ] Train adaptation encoder ϕ
- [ ] Verify ϕ MSE ≤ 0.02 on held-out set
- [ ] Run M2 eval suite again with ϕ (replacing μ)
- [ ] Write M2_results.md with all numbers vs targets from spec

---

## What to do on failure

1. **Do not** change more than one variable before the next run.
2. Identify which failure mode (F1–F6) from the spec matches the symptom.
3. State the diagnosis in one sentence. State the fix in one sentence.
4. Write a new hypothesis doc (M2_hypothesis_v2.md) before the next run.
5. If three consecutive runs fail on the same failure mode: stop, re-read MAVEN paper, re-read M2_spec, do not iterate blindly.

---

## Notes (fill in during training)

[Anything unusual observed — entropy curve shape, reward plateau, crash episodes, z variance logs, etc.]
