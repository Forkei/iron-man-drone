# M1 Hypothesis Document

**Written**: 2026-05-02  
**Spec**: MILESTONE_1_SPEC_v2.md  
**Gate**: This file must exist before `scripts/train_m1.py` will run.

---

## What this run tests

Whether the five SimpleFlight factors — rotation matrix obs, time-vector-in-critic-only, smoothness reward, selective DR, large parallel envs — reproduce acceptable trajectory-tracking behavior when re-implemented on MuJoCo MJX + JAX rather than the original Isaac Sim stack.

We are not testing whether we can improve on SimpleFlight. We are testing whether we understand the recipe well enough to replicate it on a different simulator.

---

## Pre-training expected values (from theory, before seeing any data)

**Initial reward per step.** At reset, the drone spawns near `[0, 0, 1]` m and the reference trajectory starts at a random point. Expected initial position error: ~0.3–0.8 m. So:
- `r_task = exp(-d²)`: for d=0.5m → 0.78, for d=1m → 0.37. Call it ~0.6 average.
- `r_smooth = exp(-||u_t - u_{t-1}||²)` = exp(0) = 1.0 at step 0 (prev_action = zeros).
- `r_total = r_task + 0.4 * r_smooth` → ~1.0 at step 0, settling to ~0.7 once the policy starts moving.

**Entropy sanity ratio at init.** Actor is a 4-dim diagonal Gaussian with `log_std` initialized to 0 (std=1). Entropy of N(0, I₄) = 4 × ½(1 + ln2π) ≈ 5.7 nats. Entropy contribution to loss = 1e-3 × 5.7 = 0.006. At mean reward ~0.7, ratio = 0.006/0.7 ≈ 0.8%. Well below 10% — this should pass immediately.

**Initial value estimates.** With γ=0.99 and mean reward per step ~0.7, the geometric-series expected return is roughly 0.7/(1-0.99) = 70. The critic starts random so will predict something else entirely. Value loss will start high (order 1000–5000) and should fall within a few hundred epochs as the critic learns. If it goes up instead of down, something is wrong with the critic optimizer or the returns computation.

---

## Expected training trajectory

**Epochs 0–200**: Mostly noise. Reward may dip slightly as the policy starts moving (worse than random hover) before recovering. Value loss should peak early and start falling.

**Epochs 200–1000**: Reward trend becomes visibly positive. Value loss falls steadily. Entropy starts slow decline from ~5.7 nats. This is the window where structural bugs (wrong obs, broken mixer) would show up as flat or declining reward.

**Epochs 1000–5000**: Policy learns to roughly track polynomial trajectories. MED on polynomial eval should drop below 0.3m. Zigzag tracking will lag behind since it requires sharper responses.

**Epochs 5000–15000**: Fine-grained tracking. Paper's curve shows most of the gain happens in this range. Entropy converges to a low positive value (not zero — the policy should stay stochastic). Figure-eight MED should cross below 0.1m somewhere in this range.

**By epoch 15000**: Target MED on figure-eight (normal) < 0.056m.

---

## Success criteria

All of the following must hold:

| Check | Target | Hard limit |
|---|---|---|
| Entropy / reward ratio at init | < 10% | — |
| Reward trending up by epoch 200 (smoothed) | Yes | — |
| Value loss direction by epoch 500 | Decreasing | Not diverging |
| Figure-eight normal MED at final eval | < 0.056 m | < 0.10 m |
| Figure-eight slow MED | < 0.040 m | — |
| Pentagram slow: completes without crash | Yes | — |
| Zigzag: completes without crash | Yes | — |
| Entropy at epoch 15000 | > 0.3 nats | > 0.0 |

---

## Failure modes and what each means

**Entropy collapses within 500 epochs** (drops to < 0.1 nats and stays there):
Policy committed to a near-deterministic action early. Most likely cause: reward signal is much larger than expected, overwhelming the entropy term. Check that r_task and r_smooth are both ∈ [0, 1] — if the reward isn't normalized, the entropy coefficient is effectively zero. Do not continue training.

**Value loss explodes** (grows past 1e5 and keeps going):
Critic learning is unstable. Check that the critic optimizer is `optax.adam(lr=1e-4)`, not sharing state with the actor. Check that returns are computed correctly in GAE — if they're in the thousands due to a γ or normalization bug, the MSE loss will be proportionally enormous.

**Reward flat or declining past epoch 1000**:
This is the structural bug signal. Debug in this order:
1. Print `e_W` at step 0 — are the relative reference positions in meters and the right sign? (`ref - pos`, not `pos - ref`)
2. Print `det(R)` — should be ~1.0. If it's -1 or wildly off, rotation matrix is being read incorrectly from `xmat`.
3. Test CTBR controller in isolation: does `ctbr_to_rotor_speeds` with a zero-body-rate, half-throttle command produce near-hover rotor speeds? (~1500 rad/s for Crazyflie hover)
4. Check that `actor_obs.shape[-1] == 42` and `critic_obs.shape[-1] == 43`. If 45/46, body rates snuck back in.

**Policy oscillates after initial improvement** (reward goes up to ~0.9, then bounces between 0.7–0.9):
Usually GAE or PPO clip. Verify γ=0.99, λ=0.95, clip=0.2. Check horizon=32 — too short a horizon with γ=0.99 means the critic can't see far enough ahead. If everything checks out, this may just be normal PPO variance; look at the smoothed 100-epoch curve before panicking.

**Training speed < 5k steps/sec** (caught by sanity_check gate 4):
Something is on CPU. Do not start training until fixed.

---

## Wall-clock budget

- **First run cap**: 12 hours. If no clear convergence signal (reward up, value loss down, entropy slowly falling) within this window, stop. Re-read SimpleFlight Sections III-D and IV-B before the second run. Do not tweak hyperparameters and rerun — find the structural reason first.
- **Estimated time**: at 50k+ steps/sec, ~2.7h for 15k epochs. At the WARN threshold (~20k), ~7h. Both within budget.

---

## If it works

1. Archive the final checkpoint.
2. Run `python scripts/eval_m1.py --checkpoint PATH` on all benchmark trajectories.
3. Write `experiments/m1_baseline/M1_results.md`: numbers vs paper Table III, what we deviated from, what we learned.
4. `git tag m1-baseline`.
5. Come back to the project plan and spec M2 (MAVEN-style fault tolerance).

## If it fails

Re-read the paper before touching any code. Specifically: if reward doesn't improve, re-read Section III (architecture + reward). If it improves but plateaus far from target, re-read Section IV-B (ablation). The answer to "what's wrong" is almost always in those two sections.
