# M1 Hypothesis Document

**Date**: 2026-05-01  
**Status**: DRAFT — must be finalized and committed before any training run starts  
**Gate**: No training without this document signed off.

---

## What this run tests

Full reproduction of SimpleFlight (Chen et al., RAL 2025) using their published hyperparameters and architecture, trained in the Omnidrones simulator on an RTX 4070 laptop. We are not innovating. The sole question is: does our toolchain + implementation faithfully reproduce their recipe?

---

## What we expect

**Training dynamics**:
- Reward starts near 0 and climbs steadily. Paper reaches ~15,000 epochs. We expect visible improvement within the first 1,000 epochs.
- Value loss should decrease within ~500 epochs. If it doesn't, actor/critic separation is broken.
- Entropy should start moderate (~2-4 nats for a 4-dim Gaussian) and gradually decrease as the policy sharpens. It must NOT collapse to near-zero within the first few hundred epochs — that means the policy found a degenerate solution.
- At init, we will manually log: reward magnitude, entropy, smoothness reward, task reward. **Entropy must be < 10% of total reward magnitude.** If it isn't, reduce entropy coefficient before starting.

**Convergence targets** (from paper Table III, with 2× buffer):
| Trajectory | Paper MED (m) | Our target (m) |
|---|---|---|
| Figure-eight normal | 0.028 | < 0.056 |
| Figure-eight slow | ~0.02 | < 0.04 |
| Figure-eight fast | ~0.05 | < 0.10 |
| Random polynomial | ~0.03 | < 0.06 |
| Pentagram slow | ~0.03 | < 0.06 |
| Pentagram fast | ~0.06 | < 0.12 |
| Random zigzag | ~0.05 | < 0.10 |

*Paper numbers from Table III; "~" means read from figures, not directly tabulated.*

**Wall-clock estimate**: Paper trained for 15,000 epochs. At 128 parallel envs on a 4070, we estimate ~4-8 hours for full training. Set hard wall-clock limit at 12 hours for the first run.

---

## What success looks like

Concrete, binary checks:

- [ ] Training reward curve is monotonically increasing (smoothed over 100-epoch window) by epoch 2,000
- [ ] Value loss is decreasing by epoch 1,000
- [ ] Entropy does not collapse (stays > 0.5 nats) through epoch 5,000
- [ ] Policy completes all four benchmark trajectories without crashing at epoch 10,000 eval
- [ ] Figure-eight (normal) MED < 0.056 m at final eval
- [ ] No oscillation visible in velocity profile plots

---

## What failure looks like

**Entropy collapse** (entropy → ~0 within first few hundred epochs):
- *Means*: entropy coefficient too high, or reward scale too large, or shared actor/critic. 
- *Action*: Stop. Check entropy coeff is ≤ 1e-3. Verify actor and critic are separate nn.Module instances with separate optimizers. Do NOT continue training.

**Value loss exploding** (diverges to > 1e4):
- *Means*: critic learning rate too high, or critic is seeing actor's gradient.
- *Action*: Stop. Verify separate optimizers. Try critic LR 1e-4 → 5e-5.

**Reward oscillates** (goes up then down then up, not monotone):
- *Means*: PPO clip too tight, or horizon too short, or trajectory distribution is unbalanced.
- *Action*: Check training trajectory mix is 50/50 polynomial/zigzag. Check GAE λ = 0.95. Check horizon = 32. Do NOT increase LR.

**Tracking error doesn't drop below 0.5m** by epoch 3,000:
- *Means*: most likely the rotation matrix is being passed as quaternion, or CTBR action space isn't being correctly converted to motor commands, or time vector is in actor (not critic only).
- *Action*: Stop. Re-read Section III of SimpleFlight paper. Check observation assembly code line by line against the paper spec.

**OOM on 4070** (CUDA out of memory):
- *Action*: Reduce batch size per env first (rollout buffer size). Only reduce num_envs as last resort — parallel envs matter more for transfer.

**Training too slow** (< 100 epochs/hour):
- *Action*: Profile. Check that simulation is running on GPU (not CPU). Reduce eval frequency to every 1,000 epochs. This is acceptable for M1 since we aren't deploying to real hardware.

---

## Time budget

- **First run**: 12 hours wall-clock maximum
- **If first run fails**: stop, re-read the paper, write a one-paragraph diff between what we implemented and what the paper says. No hyperparameter tweaking without reading first.
- **Total M1 budget**: 2 weeks. If environment setup consumes > 1 week, escalate — that's a toolchain problem, not a training problem.

---

## What we do if it works

1. Checkpoint the final policy to `experiments/m1_baseline/checkpoints/`
2. Run full eval on all four benchmark trajectories, record MED per trajectory
3. Write `M1_results.md` comparing our numbers to paper Table III
4. Tag git release `m1-baseline`
5. Review PROJECT_PLAN.md together to decide M2 specifics

---

## What we do if it fails

**Do not** tweak hyperparameters in a doom loop. The debugging protocol is:

1. **First**: check if the failure mode matches one of the named patterns above
2. **Second**: re-read the relevant section of the SimpleFlight paper
3. **Third**: write a one-paragraph hypothesis about what's different between our implementation and theirs
4. **Fourth**: make exactly one targeted change, re-run
5. **If two targeted changes don't fix it**: pause, read the paper again from scratch

The previous project's failure was running five iterations of fault-tolerance training without reading. This document exists to prevent that.

---

## Architecture checklist (verify before first run)

- [ ] Actor input: `[e^W (30 values), v (3 values), R (9 values)]` = 42-dim total
- [ ] Critic input: same 42-dim + `f_t = [k]` (1-dim timestep) = 43-dim total
- [ ] Actor network: Linear(42, 256) → ELU → LayerNorm → Linear(256, 256) → ELU → LayerNorm → Linear(256, 256) → ELU → LayerNorm → Linear(256, 8) [mean + log_std for 4-dim CTBR]
- [ ] Critic network: same but input 43, output 1 (scalar)
- [ ] Separate `torch.optim.Adam` instances for actor (lr=3e-4) and critic (lr=1e-4)
- [ ] Reward: `r_total = r_task + 0.4 * r_smooth`, both normalized to [0,1]
- [ ] Smoothness: `exp(-||u_t - u_{t-1}||²)`, NOT `||u_t||²`
- [ ] Previous action `u_{t-1}` is NOT in actor observation
- [ ] Time step `k` is ONLY in critic observation
- [ ] Rotation matrix (not quaternion) in both actor and critic input
- [ ] Thrust coefficient `k_f` is randomized ±30%; mass and inertia are NOT randomized in M1
