# M1 Hypothesis Document — MJX Stack

**Date**: 2026-05-02  
**Spec**: MILESTONE_1_SPEC_v2.md (MuJoCo MJX path)  
**Status**: WRITTEN — gate satisfied for first training run  
**Gate**: No training without this document signed off.

---

## What this run tests

Re-implementation of the SimpleFlight recipe (Chen et al., RAL 2025) on MuJoCo MJX / JAX instead of the original Isaac Sim + thu-uav/SimpleFlight codebase. Isaac Sim was abandoned because it doesn't run cleanly on WSL2 and pins to EOL Python 3.7.

The five SimpleFlight factors transfer 1:1. This run tests whether our MJX implementation of those factors — and our PPO training loop — produces a policy that tracks arbitrary trajectories in simulation.

---

## Setup checklist (verify before epoch 1)

- [ ] `conda activate drone && python -c "import jax; print(jax.devices())"` shows CUDA device
- [ ] `python scripts/train_m1.py --total_epochs 1` completes without error
- [ ] MJX sim throughput > 50k steps/sec at 1024 envs (log during first epoch)
- [ ] Random policy obs: no NaNs, no Infs (check by printing obs stats at init)
- [ ] Random policy reward: mean per step near 0.4–0.6 (both r_task and r_smooth near exp(-r²) for some small r)
- [ ] Entropy < 10% of reward magnitude at init (sanity check printed by train_m1.py before epoch 1)

---

## Expected convergence signals

**By epoch 200**:
- Mean reward should be increasing (smoothed over 20-epoch window)
- Value loss should be finite and not exploding

**By epoch 1,000**:
- Clear positive reward trend (not just noise)
- Value loss consistently decreasing
- Entropy slowly decreasing but not collapsed (> 1.0 nats)

**By epoch 5,000**:
- Policy should begin tracking easy trajectories (polynomial)
- MED on figure-eight normal should drop below 0.5m if we evaluate at this point

**By epoch 15,000**:
- Figure-eight normal MED < 0.056 m (M1 target)
- Zigzag trajectories completed without crash

---

## Success criteria (concrete, binary)

| Criterion | Target | Hard limit |
|---|---|---|
| Sim throughput | > 50k steps/sec @ 1024 envs | > 10k (minimum for reasonable training) |
| Figure-eight normal MED | < 0.056 m | < 0.10 m |
| Figure-eight slow MED | < 0.040 m | < 0.08 m |
| Figure-eight fast MED | < 0.100 m | < 0.20 m |
| Pentagram slow MED | < 0.060 m | < 0.12 m |
| Zigzag: completes without crash | Yes | — |
| Entropy at final eval | > 0.3 nats (not collapsed) | > 0.0 |
| Value loss at epoch 15k | Decreasing trend | Not diverging |

---

## What failure looks like (and what to do)

**Entropy collapse** (→ ~0 within first 500 epochs):
- *Means*: entropy_coeff too high or reward too small, causing policy to commit to degenerate solution.
- *Action*: Stop. Re-check entropy/reward ratio print at epoch 0. Verify `entropy_coeff = 1e-3` in config. Do not continue training.

**Value loss exploding** (> 1e4):
- *Means*: critic LR too high, or actor and critic sharing state somehow.
- *Action*: Stop. Verify `create_train_states` returns separate actor and critic `TrainState` instances with separate `optax.adam` optimizers. Try critic LR → 5e-5.

**Reward stuck near zero** past epoch 2,000:
- *Means*: observation pipeline is wrong (most likely R or e^W computed incorrectly), or CTBR mixer is inverted/broken.
- *Debug order*:
  1. Print obs at step 0: verify e^W is in meters and the right sign (ref - pos, not pos - ref)
  2. Verify `xmat[drone_body_id]` is the body→world rotation (not transpose)
  3. Check `qvel[:3]` is linear velocity (world frame), `qvel[3:6]` is angular velocity (body frame)
  4. Test the CTBR controller in isolation: does `ctbr_to_rotor_speeds` with a hover command produce near-hover motor speeds?

**Oscillating policy** (reward goes up then oscillates):
- *Means*: GAE or PPO clip not set correctly, or trajectory distribution imbalanced.
- *Action*: Verify γ=0.99, λ=0.95, clip=0.2, horizon=32. Verify training mix is truly 50/50 polynomial/zigzag.

**OOM at 1024 envs**:
- Reduce to 512. Document in results.

**MJX sim < 10k steps/sec**:
- Verify everything is inside `jax.jit`. No Python in inner loop.
- Verify rollout uses `jax.lax.scan`, not a Python for-loop.
- If using Python for-loop accidentally: rewrite with scan.

---

## Wall-clock budget

- **First run**: 12 hours max, then stop and inspect
  - Expected at 1024 envs on 4070: ~6-8 hours for 15k epochs
  - If signals are good at 12h (reward increasing, loss decreasing): continue
  - If signals are bad at 12h: stop, re-read SimpleFlight paper Section III before next run
- **Total M1 budget**: 2 weeks

**Do not** extend training hoping it converges. If reward hasn't improved meaningfully by epoch 3,000 (first 3h), there's a structural bug. Fix it.

---

## Debugging order (if training fails)

1. Does the CTBR controller hover correctly (zero-policy test)?
2. Are the observations finite and plausibly-ranged (no NaNs)?
3. Is the reward formula correct (check r_task and r_smooth separately)?
4. Is the actor NOT receiving u_{t-1} or timestep k? (Verify obs dims: actor=45, critic=46)
5. Is R the rotation matrix (not quaternion)?
6. Only after all of the above check out: look at hyperparameters.

**Do not tweak hyperparameters before fixing structural issues.**

---

## What to do if it works

1. Save final checkpoint to `experiments/m1_baseline/{run_name}/checkpoints/final`
2. Run `python scripts/eval_m1.py --checkpoint PATH`
3. Write `experiments/m1_baseline/M1_results.md` comparing to paper Table III
4. `git tag m1-baseline`
5. Bring results back here to plan M2 (fault tolerance via MAVEN-style DR)

---

## What to do if it fails after two targeted attempts

Re-read SimpleFlight paper Sections III-D (reward) and IV-B (ablation) before a third run. The paper is detailed about what breaks and why. Trust the paper over intuition.
