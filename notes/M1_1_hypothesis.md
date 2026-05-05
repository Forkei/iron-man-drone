# M1.1 Hypothesis — Reward Reference Off-by-One Fix

**Date**: 2026-05-02  
**Status**: Pre-training (gating artifact)  
**Experiment**: `experiments/m1_1_reward_fix/`

---

## Change from M1 baseline

**Single change — isolated bug fix:**

`quadrotor_env.py:296` — reward reference step index:
```python
# Before (M1 baseline — BUG):
ref_pos = get_reference_pos(state.traj, state.step)   # r(t): stale by one step

# After (M1.1 fix):
ref_pos = get_reference_pos(state.traj, new_step)     # r(t+1): matches post-physics position
```

Everything else is identical to M1 baseline:
- `entropy_coeff = 1e-3` (unchanged — entropy collapse is expected and unfixed in this run)
- Polynomial bounds ±1.5m (unchanged — the ±2.5m change was unjustified, reverted)
- Same network, LRs, PPO hyperparams, trajectory mix, 15k epochs

---

## What the bug was

In `step()`, physics runs first → drone moves to position `p(t+1)`. Then:

```
new_step = state.step + 1              # = t+1

pos     = mjx_data.xpos[drone_body_id] # post-physics: p(t+1)
ref_pos = get_reference_pos(traj, state.step)  # r(t) — ONE STEP STALE
```

The reward trained the policy to minimize `||p(t+1) − r(t)||` instead of `||p(t+1) − r(t+1)||`.

For figure-eight normal speed (T=5.5s, ~0.36 m/s average), the reference moves ~3.6mm per step. Over 1000 steps, the systematic stale reference biases the gradient on every update. The policy learns to track where the reference WAS, not where it IS — encoding a structural 1-step lag into the trained weights.

The bug interacts with the figure-eight geometry: during the −x arc (leftward return), `r(t)` is more negative-x than `r(t+1)`, so the policy is consistently trained to target the more-negative-x position. This matches the observed −x bias of −0.046m in M1 baseline.

The obs (`e^W`) is computed correctly using `new_step` (line 313) — the lag is purely in the reward signal, not the observation.

---

## Predictions

### If the off-by-one was material (expected):

| Metric | M1 baseline | M1.1 prediction |
|---|---|---|
| MED (figure-eight normal) | 0.105 m | 0.07 – 0.09 m |
| Constant offset (mean xy bias) | 0.050 m | ≤ 0.025 m, direction may flip |
| Apex overshoot ratio | 11× | 6–9× (partial improvement) |
| Lag (cross-corr peak) | 5 steps | 3–4 steps |

The overshoot ratio should improve because the reward now points at the correct target position — during apex traversal the reference position at t+1 is already rounding the corner, giving a cleaner gradient direction. The constant offset should shrink and may flip sign temporarily as the policy unlearns the stale-reference behavior.

Entropy will still collapse around epoch 165–200 (unfixed). Reward plateau will occur. The policy will still fail to generalize to apex dynamics. MED improvement from bug fix alone is likely partial — enough to see clearly in the metrics, not enough to pass 0.056m.

**Expected MED: 0.07–0.09m** (improvement, but still above threshold).

### If the off-by-one was minor (alternate):

- MED stays in 0.10–0.12m range, comparable to baseline
- Offset direction and magnitude similar
- This would mean the bias is purely from entropy collapse, not the reward index

If this outcome occurs, the off-by-one fix was correct but had negligible effect. M1.2 would then bump `entropy_coeff` to 3e-3 as the primary fix.

---

## Decision tree after M1.1

| Result | Next step |
|---|---|
| MED < 0.056m | **Ship M1.** Run entropy investigation separately as diagnostic only. |
| MED 0.07–0.09m (meaningful improvement) | **M1.2**: bump `entropy_coeff` 1e-3 → 3e-3, all else identical. |
| MED 0.10–0.12m (no improvement) | Bug fix was minor. **M1.2**: bump `entropy_coeff` to 3e-3. |

---

## What is not changed and why

**entropy_coeff stays at 1e-3**: The previous session confirmed entropy collapse at epoch ~165. Bumping entropy is a known improvement. It is deliberately held back to isolate the reward fix effect. If we changed both simultaneously and MED improved, we could not attribute the improvement to either change specifically.

**Polynomial bounds stay at ±1.5m**: The paper (arXiv 2412.11764 Section V.A.1.b) specifies velocity range (0–1 m/s) and segment duration (1.5–4s) for polynomial trajectories — no spatial bounds stated. The claim that "paper uses ±2.5m" was not supported by the paper text. Reverted to original.
