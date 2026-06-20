# M3 Hypothesis — Joint Fault + Obstacle Training, Run 1

**Date:** 2026-05-15
**Experiment dir:** `experiments/m3_run1/`
**Status:** Pre-training — hypothesis written, Gate 4 validated (MED=0.0557m, 2026-05-15)
**Config:** `experiments/m3_run1/config.yaml` (frozen)

**GATE: This document must be complete and reviewed before any training run starts.**

---

## What this run tests

Whether the M3 joint training setup — simultaneous fault-tolerance DR (η ∈ [0.5,1.0], 70% episode rate) and procedural obstacle avoidance (4 scene modes, density curriculum) — can produce a policy that both tracks figure-eight trajectories under rotor faults and avoids pillars, starting from random weights. This is the first run using the MJX+Warp depth-render pipeline (batch_render). The key unknowns are whether the obstacle reward and crash reward are balanced well enough to prevent two failure modes: the policy ignoring obstacles to track better, or the policy hovering away from trajectory to avoid obstacles. The encoder (z_fault latent) is trained jointly, not in a separate Phase 2 — this collapses the two-phase M2 approach into a single run and raises the question of whether the policy can simultaneously learn obstacle geometry and rotor fault estimation from the same gradient signal.

---

## Setup checklist (completed before launch)

- [x] Gate 1: Scene validity (5 modes × 20 trials) — PASS
- [x] Gate 2: Depth bins sensible (3 modes) — PASS
- [x] Gate 3: Random policy no crash (32 envs × 100 steps) — PASS
- [x] Gate 4: Frozen M2 policy — MED=0.0557m ≤ 0.075m — PASS (2026-05-15)
- [x] Gate 5: Throughput ≥ 20k fps — confirmed ~20k fps (benchmark script)
- [x] train_m3.py import bug fixed (EPISODE_STEPS from quadrotor_env_m3)
- [x] Actor obs 66-dim: 42 base + 8 encoder latent + 16 depth bins
- [x] Critic obs 72-dim: 66 + 1 step-k + 5 obstacle distances
- [x] Curriculum: density_mult=0.5 until 10M steps, ramp to 1.0 by 50M steps

---

## Predictions

### User prediction (gut feel, stated before training)

**Crash rate at H1 gate (epoch 916, ~30M steps):** ~50–65%
The policy is learning two new signals simultaneously (depth bins + joint encoder). Crash rate will be elevated early. With the sparse curriculum (density_mult=0.5) the H1 gate at 0.70 should hold, but it'll be close.

**Nominal MED (no faults, no obstacles) at H1 gate:** ~0.10–0.15m
Early training is noisy. The policy won't have converged on trajectory tracking by 30M steps — it's still learning to not crash.

**Nominal MED at end of training (500M steps, ~7h):** ~0.07–0.10m
Obstacle avoidance adds load to the observation space and gradient signal. Expect nominal tracking to be modestly worse than M2's 0.057m even on clear-air figure-eight. This is acceptable — M3 is harder.

**Fault MED (η=0.70, no obstacles) at end of training:** ~0.09–0.12m
Joint training should produce working fault tolerance since the encoder sees the same DR distribution as M2 Phase 1. Slightly worse than M2's 0.081m because the actor also processes depth bins.

**Obstacle avoidance (hallway holdout, full density):** Pass in principle, some crashes expected at high speed. The density curriculum should handle gradual exposure. Hallway is the hardest mode — expect 5–15% crash rate at holdout eval.

**Encoder quality vs M2 Phase 2:** Worse. M2 Phase 2 trained the encoder on 20k offline rollouts with a frozen policy. M3 trains jointly — the encoder loss is entangled with the policy gradient. Expect z_fault latent to be noisier early, settle by ~200M steps.

**Most likely failure mode:** F-obstacle — the policy learns to slow down or hover rather than track, using the obstacle term as an excuse. Watch w_obstacle × proximity_term vs w_track × tracking_error in the reward decomposition.

**Confidence:** Low — this is the first M3 run, many unknowns. The predictions are reference points, not expectations.

---

## Success signals at each checkpoint

| Gate | Epoch | Steps | Criterion | Action if missed |
|---|---|---|---|---|
| H1 | 916 | 30M | crash_rate ≤ 0.70 AND trending down | Abort — diagnose curriculum or reward balance |
| Mid | 1525 | 50M | crash_rate ≤ 0.30 | Stop and check if obstacle reward is dominating |
| 1-hour pause | 3576 | 117M | User confirmation | Manual review of reward curve shape |
| Late | 7629 | 250M | nominal MED ≤ 0.12m | If not, check encoder z_fault variance |
| Final | 15259 | 500M | nominal MED ≤ 0.10m, crash_rate ≤ 0.05, hallway holdout crash ≤ 0.15 | See failure protocol below |

**Time-box rule:** If crash_rate > 0.50 at epoch 1525 (~50M steps), stop. The curriculum or reward weights are wrong. Do not continue hoping it improves.

---

## What to do on success

- [ ] Archive best checkpoint to `experiments/m3_run1/checkpoints/final/`
- [ ] Run M3 eval suite: figure_eight_normal (nominal + fault), hallway holdout, random holdout
- [ ] Compare nominal MED vs M2 Phase 1 baseline (0.057m) — regression ≤ 2× is acceptable
- [ ] Write `experiments/m3_run1/M3_results.md` with all numbers vs targets
- [ ] Run Gate 4 one final time with the M3 policy instead of M2 policy (optional — curiosity)

---

## What to do on failure

1. Do not change more than one variable before the next run.
2. Identify failure mode: F-tracking (MED never improves), F-obstacle (policy avoids obstacles by hovering), F-encoder (z_fault collapsed or noisy), F-crash (crash rate never drops below 0.50).
3. Most likely fixes by failure mode:
   - F-tracking: reduce w_obstacle from 0.5 to 0.2; check depth bins are actually varying
   - F-obstacle: check depth bins are plugged into actor obs (not zero); verify render pipeline
   - F-encoder: add encoder MSE auxiliary loss; or revert to two-phase approach
   - F-crash: reduce fault_prob to 0.3 for first 50M steps; tighten d_crash from 0.15 to 0.20
4. Write M3_hypothesis_v2.md before next run.

---

## Notes (fill in during training)

[H1 gate result, crash rate curve shape, reward decomposition at epoch 916]
