# M1.2 Hypothesis — Entropy Coefficient Bump

**Date**: 2026-05-03  
**Status**: Pre-training (gating artifact)  
**Experiment**: `experiments/m1_2_entropy/`

---

## Changes from M1 baseline

Two cumulative changes (M1.1 reward fix carried forward + new entropy change):

| Parameter | M1 baseline | M1.1 | M1.2 |
|---|---|---|---|
| Reward ref index | `state.step` (stale) | `new_step` (fixed) | `new_step` (fixed) |
| `entropy_coeff` | 1e-3 | 1e-3 (unchanged) | **3e-3** |
| Poly bounds | ±1.5m | ±1.5m | ±1.5m |

Everything else identical to M1 baseline.

---

## Rationale

**Why 3e-3?**  
At init, sanity check showed: reward=0.478, entropy=5.68 nats.  
- With 1e-3: entropy term = 0.0057, ~1.2% of reward signal → collapsed at epoch ~165 in both M1 and M1.1
- With 3e-3: entropy term = 0.017, ~3.6% of reward signal → meaningful resistance to early collapse

The entropy collapse in M1/M1.1 was total and irreversible. Once the policy goes deterministic, all future gradient steps just reinforce the current deterministic behavior — the policy can never rediscover the exploration needed to handle OOD figure-eight apices.

With 3e-3, the entropy bonus is large enough to resist advantage spikes that cause collapse, but small enough not to dominate the reward signal.

---

## Predictions

### My prediction (Claude)

**Expected best MED: 0.065–0.085m** — improvement over M1/M1.1 (~0.106m), probably still above the 0.056m threshold.

I'm slightly more optimistic than the stated 0.07–0.09m range. Here's why I split the difference:

**Factors that will improve:**
- Entropy should stay above 0.5 nats through at least epoch 400–600, possibly longer
- During the additional ~300–400 epochs of meaningful exploration, the policy will encounter more varied acceleration profiles from the training trajectories
- The constant offset (~5cm in M1) should largely disappear — it was an artifact of the deterministic policy converging to a biased equilibrium, which better entropy prevents
- The apex overshoot ratio should drop from 11× to ~5–7× — the policy will have seen more direction changes, even if not at figure-eight apex curvature magnitude

**Why I don't expect to clear 0.056m:**
- The training trajectory curvature distribution likely doesn't cover the figure-eight apex curvature (this is under analysis in the concurrent diagnostic). Even with good entropy, the policy cannot generalize to dynamics it was never trained on.
- The polynomial trajectories (50% of training) have zero-velocity junctions — the drone has never trained on maintaining speed through a sharp turn.
- Expected reward plateau: ~1.38–1.42 (vs 1.33–1.36 in M1/M1.1), but still plateauing.

**Where I might be wrong:**
- If zigzag direction changes (at every waypoint) are close enough to figure-eight apex curvature, the policy may generalize well → MED drops toward 0.05–0.07m. This is the optimistic case.
- If entropy collapses faster than expected (e.g., still around epoch 200 with 3e-3), results will look similar to M1.1 → 0.09–0.11m.

**The ~30% chance of passing 0.056m**: If it happens, expect it at a late epoch (>5000) after the policy has been compounding small trajectory improvements. Would be surprising given the training distribution gap, but not impossible.

### User's prediction
MED 0.07–0.09m — better but failing threshold because entropy bump addresses exploration, not training distribution coverage.

I agree with this framing. The training distribution gap (OOD apex curvature) is the structural issue. Entropy just determines whether the policy ever gets a chance to generalize; it doesn't add new dynamics to the training set.

---

## What to watch in the wake-up report

**(a) When does entropy cross zero?**  
- M1/M1.1: crossed zero at epoch ~165
- M1.2 actual (already observed): crossed zero at epoch **~177** — only 12 epochs later
- 3e-3 barely moved the collapse. The advantage signal overwhelms the entropy gradient at epoch 165–180 regardless of 1e-3 vs 3e-3
- This means M1.2 result will likely match M1.1 closely — the extra exploration window was ~12 epochs

**(b) Apex overshoot ratio**  
- M1 baseline: 11.3× (0.302m apex vs 0.027m straight)
- M1.2 prediction: 5–8× improvement
- How to check: run `scripts/diagnose_trajectory.py` on best checkpoint

**(c) MED on figure-eight normal/slow/fast**  
- Check eval epochs 3000, 6000, 9000, final
- Normal: target <0.056m
- Slow: should be easier (lower angular velocity) — if slow is bad, it's not a curvature problem
- Fast: will definitely be harder; expect 0.12–0.20m range

---

## Decision tree after M1.2

| Result | Interpretation | Next step |
|---|---|---|
| MED < 0.056m | Entropy was the only real blocker | Ship M1, entropy fix for M2 |
| MED 0.06–0.08m, apex ratio < 5× | Both entropy and distribution | M1.3: fix training distribution coverage |
| MED 0.08–0.10m, apex ratio 7–10× | Distribution gap dominates | M1.3: add curvature-rich training trajectories |
| MED ~0.10m (no improvement) | Entropy 3e-3 not enough | M1.3: try 5e-3 or entropy scheduling |
