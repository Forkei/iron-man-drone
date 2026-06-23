# M3 v2 — reward-lever sweep results (cloud, 2026-06-24)

Ran a 2x2-ish reward sweep on a rented Vast.ai RTX 4090 (~$1.5–2 total, box torn
down after) to test the diagnostic hypothesis: the v1 policy tracked 5x looser than
M2 on the identical figure-eight (0.287m vs 0.057m) — is it the reward balance?

## Sweep results (early checkpoints, ~epoch 450–600 / 15–20M steps, done-masked)

| Config | W_TRACK | W_CRASH | figure-8 XY MED | fair collision-free |
|---|---|---|---|---|
| v1 baseline | 2 | 10 | 0.287 m | 92% |
| wtrack6 | **6** | 10 | 0.279 m | — |
| crash4 | 2 | **4** | **0.215 m** | **91.5%** |

(W_CRASH=2 / "both" points not completed — abandoned at the project pivot, see below.)

## Verdict — the crash penalty is the lever, not the tracking weight

- **Up-weighting tracking (W_TRACK 2→6) is INERT** — 0.279 ≈ 0.287, no effect.
- **Cutting the crash penalty (W_CRASH 10→4) is a GENUINE win** — figure-8 tracking
  tightened **25% (0.287→0.215m)** while avoidance held **flat (91.5% ≈ 92%)**. No
  collision tradeoff. The terminal −10 penalty bred excessive caution; easing it let
  the policy commit to the line without crashing more.

This validates the root cause: the obstacle-avoidance objective made the policy
conservative, specifically via the catastrophic crash penalty (not a tracking-weight
imbalance, not reward-shape, not encoder/control — all ruled out by diagnostics).

Reward-weight tuning alone likely plateaus ~0.18m (won't reach the 0.10m gate); the
full precision fix would be the penalty SHAPE (smooth proximity penalty replacing the
terminal −10).

## BUT — we are NOT pursuing this further (project pivot, 2026-06-24)

The 0.10m tracking gate was gold-plating: it's a self-imposed number, not a capability
the actual vision needs. M3's avoidance works (92%); 0.21m figure-8 tracking is fine
for an autonomous agile drone. The project is re-aiming away from from-scratch
RL-control precision toward the real differentiators (robust agile control + anomaly
recovery + perception/mapping + risk-aware planning). See [[project-vision-pivot]].

M3 is declared **good enough**. This sweep stands as the validated answer to "why was
tracking loose" + the lever if we ever want it (cut W_CRASH / reshape the penalty).
