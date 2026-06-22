# M3 Final Eval — epoch 15258 (500M steps, training COMPLETE) vs baseline

Run `m3_run1` completed the full 500M-step schedule (epoch 15258) on 2026-06-22.
Final checkpoint: `/home/forke/m3_checkpoints/m3_run1/final`.
Eval: `eval_m3.py --n 128 --only random,fault70,clear` (same seed as baseline →
same episodes, so this is an apples-to-apples policy comparison).

## Final vs baseline (epoch 5899, 193M, 38%)

| Metric | Baseline 193M | Final 500M | Δ |
|---|---|---|---|
| random fair CF (clearance≥0.15m) | 93.4% | 92.3% | flat |
| random tracking MED(cf) | 0.158 m | 0.174 m | worse |
| random unfair save-rate | 21.6% | 16.2% | worse |
| fault70 fair CF | 53.7% | 52.4% | flat |
| fault70 OOB | 39.8% | 38.3% | flat |
| fault70 MED(cf) | 0.173 m | 0.194 m | worse |
| clear (no-obstacle) MED | 0.180 m | 0.207 m | worse |

Final per-bucket (random nominal): unfair n=37 crash 81% CF 16% · tight n=20 CF 75% ·
moderate n=19 CF 95% (0% crash) · clear n=52 CF 98% (0% crash).

## Verdict: PLATEAUED — extra 307M steps wasted, slight tracking regression

The policy did not improve from 193M→500M; collision-free rates are flat and
tracking MED drifted ~0.02–0.03 m **worse** everywhere. Training reward sat at
~1.9 the whole time; only enc_loss kept falling (encoder fit, not policy skill).

**Root cause (predicted on the very first eval):** the tracking reward
`r_track = exp(−‖pos_err‖²)` is near-saturated at ~0.15 m (0.978 vs 0.997 at
0.055 m), so there's negligible gradient to sharpen tracking. The policy settled
into a ~1.9 reward basin around 150–200M steps and stopped improving.

## M3 ship-gate status (Gate 2)

| Target | Result | Pass? |
|---|---|---|
| Collision-free, fair training modes ≥80% | 92.3% | ✓ |
| **Tracking MED ≤ 0.10 m** | **0.174 m** | **✗ (big miss)** |
| Fault CF ≥50% | 52.4% | ~✓ (marginal) |
| Fault OOB | 38.3% | weak |

**M3 does NOT pass the ship gate as-is.** Obstacle avoidance is genuinely good;
tracking precision and fault tolerance fell short — and won't be fixed by more
training.

## Next iteration (reward redesign, NOT more compute)

1. **Sharpen the tracking reward** — replace saturated `exp(−d²)` with a term that
   keeps gradient near zero error (higher α, or linear/`exp(−d)` near zero, or a
   staged tight-tracking bonus). This is the #1 lever for the MED gate.
2. **Fault-OOB (~38%)** — the fault encoder/recovery path needs work; consider the
   M2.5-style attention to startup/fault states.
3. **Stop earlier** — the plateau was visible in the flat reward by ~200M; future
   runs should early-stop on a reward/MED plateau rather than run the full schedule.
4. Deferred ideas still open: actor-side danger signal (M3.1), batched save-rate
   metric, render clips of the final policy.

**Discipline lesson:** "training completed" ≠ "milestone achieved." The run
finished but the science says: plateau + reward-saturation → redesign reward,
re-run shorter. The day's infra firefighting (CS2 GPU contention, SIGTERM→wedged-
GPU, WSL restarts) was real but orthogonal to this outcome.
