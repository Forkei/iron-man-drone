# M3 Hypothesis — [name of this run]

**Date:** [YYYY-MM-DD]
**Experiment dir:** `experiments/m3_[name]/`
**Status:** Pre-training — hypothesis written, awaiting implementation

**GATE: This document must be complete and reviewed before any training run starts.**

---

## What this run tests

[One paragraph. What specifically is this run testing? Reference §A/§B/§C of M3_spec.md. Is this the headline M3 attempt, or a recovery run after a failed F-mode? If recovery, which F-mode and what changed?]

---

## Setup checklist (verify before launching)

- [ ] M2.5 baseline checkpoint loaded as initialization? (yes/no — note: project rule says no fine-tuning new capabilities, so this is normally NO; obstacles are baked from epoch 0 onto an M2.5-arch policy with random init or matched-arch init)
- [ ] Depth obs wired: log d_bins for 10 random states, confirm bins are not all NaN, values in [0, d_max]
- [ ] Depth noise active: log d_bins σ across 100 samples of same state, confirm σ ≈ 0.02·d
- [ ] Empty-scene probability: confirm `p_empty = 0.20` (or document deviation) — sample 100 episodes, count ones with no obstacles
- [ ] Clearance reward: at d_min = d_safe, `r_clear` is zero (boundary); at d_min = 0, |r_clear| ≤ 10% of |r_track| (entropy/balance rule)
- [ ] Collision termination: drone in contact with cylinder → terminal reward = -100, episode ends
- [ ] Critic privileged obstacle summary: confirm 8 dims are populated (not all zero); confirm actor does NOT see it (project rule)
- [ ] Trajectory placement respects clearance buffer: 100 random episodes, verify no obstacle within `clearance_buffer = drone_radius + 0.15` of reference
- [ ] Actor input 58-dim, critic input 67-dim (or document deviation): print shapes
- [ ] Sanity gates pass: `python scripts/sanity_check_m3.py`
- [ ] Pinned simulator stack: `mujoco-mjx>=3.8.0`, `warp-lang>=1.11`. Same as M2.5 (MJWarp camera integration).
- [ ] Physics backend confirmed: XLA, not MJWarp's warp backend (project rule, per M3_spec §F5b)
- [ ] Throughput baseline: pre-training warm-up confirms ≥ 10k env-steps/sec with rendering

---

## Predictions

### User prediction (gut feel, stated before training)

**Expected obstacle-free MED at epoch 10,000 (figure_eight_normal):** [your number]
[Rationale.]

**Expected obstacle-free MED at epoch 20,000:** [your number]
[Rationale.]

**Predicted cluttered nominal (3 m/s) success rate at epoch 20,000:** [%]
[Rationale.]

**Predicted cluttered stretch (4 m/s) success rate at epoch 20,000:** [%]
[Rationale.]

**Expected crash rate in epochs 1–500:** [%]
[Rationale.]

**Most likely failure mode:** [F1, F2, F3, F4, F5, F6, F7 or "none"]
[Why.]

**Predicted entropy trajectory:** [Where do you expect entropy to land at epochs 1k, 5k, 10k, 20k? Will it collapse before convergence?]

**Confidence:** [Low / Medium / High]

---

## Success signal to watch at each checkpoint

| Epoch | Obstacle-free MED | Cluttered success rate | Action if exceeded |
|---|---|---|---|
| 1,000 | ≤ 0.200 m | — (untrained on cluttered yet) | If above: stop, check obs pipeline. |
| 5,000 | ≤ 0.150 m | ≥ 30% | If either fails: stop, re-read M3_spec §F before tweaking. |
| 10,000 | ≤ 0.100 m | ≥ 60% | If either fails: stop. Diagnose one failure mode. Do not train to 15k hoping. |
| 15,000 | ≤ 0.070 m | ≥ 75% | Trend check: improvement vs epoch 10k ≥ 5% on both metrics. |
| 20,000 | **≤ 0.060 m** | **≥ 85%** | M3 pass. Run full eval suite, write results. |

**Time-box rule:** If obstacle-free MED is not below 0.100 m at epoch 10,000, stop. Do not tweak and continue.

---

## What to do on success

- [ ] Archive checkpoint to `experiments/m3_[name]/checkpoints/epoch_20000/`
- [ ] Run full M3 eval suite (Tier 1 obstacle-free + Tier 2 cluttered + Tier 3 figure-eight-with-obstacles + Tier 4 demo)
- [ ] Build demo scene video at `experiments/m3_[name]/demo_scene.mp4`
- [ ] Write `experiments/m3_[name]/M3_results.md` with all numbers vs targets from spec
- [ ] If stretch eval (4 m/s) ≥ 70%: log as bonus pass. If below: log as known limit.
- [ ] If wall-clock allows, attempt M3-with-RMA ablation (re-introduce M2 fault DR on top of converged M3 — see M3_spec "Open design decision")
- [ ] `git tag m3-baseline`

---

## What to do on failure

1. **Do not** change more than one variable before the next run.
2. Identify which failure mode (F1–F7) from `M3_spec.md` matches the symptom.
3. State the diagnosis in one sentence. State the fix in one sentence.
4. Write a new hypothesis doc (`M3_hypothesis_v2.md` or similar) before the next run.
5. If three consecutive runs fail on the same failure mode: stop, re-read the relevant primary reference (Loquercio for depth obs / scene gen; Swift for obs density; M3_spec for the rule that failed), do not iterate blindly.

---

## Notes (fill in during training)

[Anything unusual observed — entropy curve shape, reward plateau, crash episodes, depth-bin distribution drift, clearance-vs-tracking weighting in practice, etc.]
