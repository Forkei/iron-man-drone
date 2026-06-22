# M3 v2 Hypothesis — recover tracking precision lost to obstacle-avoidance conservatism

**Date:** 2026-06-22
**Status:** Pre-training. Gating doc for the M3 v2 reward-rebalance experiment.
**Predecessor:** m3_run1 (500M, epoch 15258) — avoidance OK (fair CF 92%) but tracking
MED ~0.17m (eval) / 0.29m (figure-eight), missing the ≤0.10m ship gate.

---

## Root cause (data-driven, from scripts/diag_m3_policy.py + diag_fig8.py)

The final M3 policy tracks **globally loose**, and we ruled the causes out one by one:

| Candidate | Test | Verdict |
|---|---|---|
| Reward SHAPE saturates (`exp(−d²)`) | M1/M2 used the SAME shape → 0.037/0.057m | **RULED OUT** |
| Encoder adds noise | oracle-z 0.44m ≈ encoder-z 0.42m | **RULED OUT** |
| Oscillation / chatter | action |du| = 0.008 (smooth) | **RULED OUT** |
| Control authority | error flat vs speed (1.13×) | **RULED OUT** |
| Infeasible zigzag trajectories | figure-eight (feasible) STILL 0.287m | **RULED OUT** |

**Figure-eight, final policy, M2 methodology: XY MED = 0.287m, 0% divergence, all 16
envs in 0.28–0.30m** — a deterministic, converged loose orbit (5× M2's 0.057m on the
identical trajectory; the M3 env itself is fine — Gate 4 confirmed M2 policy → 0.056m here).

**Conclusion:** the only thing new in M3 vs M2 is the **obstacle-avoidance objective**
(16 depth bins in the actor, proximity penalty `W_OBSTACLE=0.5`, terminal crash penalty
`W_CRASH=10.0`, obstacle scenes). Training to avoid obstacles produced a **permanently
conservative policy** — cautious, loose-tracking flight applied everywhere, even in clear
air. The −10 crash penalty (≈5 steps of perfect tracking) is the prime conservatism driver:
the policy massively over-prioritizes not-crashing over tracking tight.

This is a track-vs-avoid **balance** problem, not a reward-shape, encoder, control, or
trajectory problem.

---

## The change (single variable)

**Raise the tracking weight `W_TRACK` 2.0 → 6.0** in `quadrotor_env_m3.py`. Everything else
unchanged. This makes per-step tracking the dominant signal (≈12 at perfect track vs the
−10 one-time crash), testing whether more tracking pressure pulls the converged orbit tight
without collapsing avoidance. One constant changed — keeps the experiment clean.

(Secondary levers held in reserve if this fails: reduce `W_CRASH` 10→4, or test whether the
depth-bin input disrupts the actor — FM2 ablation.)

---

## Prediction

- **Clear figure-eight XY MED drops from 0.287m toward ≤0.12m** by the validation checkpoint.
  Falsified if it stays > 0.20m → conservatism is structural (crash penalty or depth
  pathway), not a tracking-weight issue → pursue secondary levers.
- Collision-free rate (fair episodes) stays ≥ 80% — i.e., we trade *some* caution for
  precision but don't break avoidance. Falsified if fair CF drops < 70%.
- The 12% clear-air divergence shrinks (a less-conservative, more-committed tracker is also
  more stable on the feasible trajectories).

---

## Validation plan (cheap first, full only if it works)

1. **50M-step validation run** (`m3_v2_val`, ~epoch 1525, ~1.5h clean GPU). NOT the full 500M.
2. At 50M: eval clear figure-eight MED (diag_fig8.py) + Tier-1 collision-free (eval_m3.py).
3. **Decision gate:** fig8 MED < 0.15m AND fair CF ≥ 80% → reward balance confirmed, launch
   full run with **early-stop on plateau** (stop when 30-epoch MED/reward flatten — m3_run1
   plateaued ~200M, so do NOT run 500M again). Else → falsified, go to secondary levers.

---

## Notes / discipline
- Do NOT add a tracking-then-obstacle curriculum (violates the no-fine-tuning-new-capability
  rule that killed the prior project — bake via DR from epoch 0).
- Keep checkpoints under a NEW run_name (`m3_v2_val`) — never overwrite m3_run1/final.
- train_m3.py's hypothesis gate currently reads notes/M3_hypothesis.md; point it at this doc
  (or copy) before the v2 run.
