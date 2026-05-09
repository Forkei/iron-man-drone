# M2 Phase 2 Spec — Causal Adaptation Encoder
**Date:** 2026-05-09  
**Prerequisite:** `notes/M2_phase1_corrected_results.md` — gate PASS required before implementing  
**Status:** Ready to implement. Phase 1 gate passed 2026-05-09.

---

## One-line goal

Train a causal MLP encoder ϕ(history) → ê_t that predicts the privileged physical state e_t from 0.5 s of observable (obs, action) history, then deploy it in place of the ground-truth e_t — giving the policy fault awareness without privileged access.

---

## Background: what Phase 1 left us

Phase 1 trained a privileged actor that receives e_t = [η₁, η₂, η₃, η₄, m_scale, F_x, F_y, F_z] directly as the last 8 dims of its 50-dim observation. At eval time the actor shows:

| Condition | figure_eight_normal MED |
|---|---|
| Nominal (ground-truth e_t) | 0.057 m |
| Fault η=0.70 (ground-truth e_t) | 0.079 m |

Phase 2 replaces ground-truth e_t with ê_t = ϕ(history). The actor weights are **frozen**. The only trainable object in Phase 2 is ϕ.

This directly follows **RMA §IV-B** (Kumar et al., RSS 2021, arXiv:2107.04034): "the adaptation module is trained in a supervised manner using the collected dataset, where the inputs are the history of observations and actions, and the labels are the environmental factors estimated by the base policy."

---

## Implementation note: no intermediate μ encoder

The original M2 design spec (Section B) proposed a small encoder μ: e_t → z before passing z to the actor. The Phase 1 implementation skipped μ and passed e_t directly. Phase 2 therefore predicts e_t directly (not z = μ(e_t)). This is cleaner and matches RMA's formulation exactly — RMA's base policy receives the environment factor directly as input.

**Consequence for Phase 2 loss:** MSE(ê_t, e_t), not MSE(ê_t, μ(e_t)).

---

## Encoder architecture

**Reference:** RMA §IV-B uses a 1D CNN over the (obs, action) history. For our Flax/JAX stack, a flat MLP is simpler to implement and matches M2 spec Section C. Upgrade to 1D CNN only if MLP MSE stalls above threshold (see Failure Modes §F4).

```
Input:  H × (o_dim + a_dim) = 50 × (42 + 4) = 50 × 46 = 2300 dims
         (flattened; chronologically ordered, oldest first)

ϕ: Linear(2300 → 256) → ELU → Linear(256 → 128) → ELU → Linear(128 → 8) → tanh

Output: ê_t ∈ [−1, 1]^8  (clamped, same bounds as e_t after normalization)
```

Parameter count: ~650k — negligible.

**Why flat MLP over RNN/CNN here:** H=50 is a fixed window, not variable-length. A flat MLP can learn temporal correlations within the 0.5s window without sequence processing overhead. RMA uses 1D CNN but notes the flat encoder also works for short windows.

---

## Input dimensions (per step)

Each step contributes a 46-dim vector:
- `o_t` (42-dim): observable actor obs = [e_W(30), v(3), R(9)] — **without** the 8-dim priv_state slot
- `a_{t-1}` (4-dim): action taken at step t (CTBR output of actor)

History buffer at step t: `[o_{t-H}, a_{t-H-1}, ..., o_{t-1}, a_{t-2}, o_t, a_{t-1}]` flattened = 2300-dim.  
At episode start (t < H): zero-pad missing history. The encoder must be robust to zero-padded prefixes — this is the expected deployment condition.

---

## Training data collection

**Procedure (RMA §IV-A):** Roll out the frozen Phase 1 actor under full DR. Record (history, e_t) pairs for supervised training.

**Collection script:** `scripts/collect_phase2_data.py` (to be written)

```
Rollouts:       20 000 episodes × 1000 steps
DR:             Full Phase 1 DR (fault_prob=0.70, η~U(0.50,1.00), mass~U(0.80,1.20), kf±30%)
Actor:          Frozen Phase 1 params (m2_phase1_baseline_1778244202/checkpoints/final)
GPU:            Use VecEnv(num_envs=2048) for throughput; ~10 episodes/second → ~2000s ≈ 30 min
```

**Per-step record:**
- `obs_base_t`: 42-dim observable obs (strip priv_state before saving)
- `action_t`: 4-dim CTBR action
- `e_t`: 8-dim privileged state (constant within episode, varies across episodes)
- `episode_id`: for building history windows offline

**File format:** NumPy `.npz` per 1000 episodes. Expected total: ~20GB uncompressed (manageable with 32-bit float). If disk is a concern, subsample to 5000 episodes — minimum viable dataset.

**Train/val split:** 80/20 by episode (not by step — leaking episode e_t across split would inflate val MSE).

---

## Loss function

```
L(ϕ) = mean over training windows:
    MSE(ê_t, e_t_normalized)
  = (1/8) Σ_i (ê_{t,i} - e_{t,i}^norm)²
```

**Normalization of e_t before training:**

| Dim | Physical meaning | Range in training | Normalize to [−1, 1] by |
|---|---|---|---|
| η₁–η₄ | Rotor efficiency | [0.50, 1.00] | (x − 0.75) / 0.25 |
| m_scale | Mass multiplier | [0.80, 1.20] | (x − 1.00) / 0.20 |
| F_x, F_y, F_z | Wind force (N) | [0, 0] in Phase 1 | zeros (all episodes) |

Since wind was excluded from Phase 1 training, F_x/F_y/F_z are always 0 in the training data. The encoder will output near-zero for those dims — acceptable, the policy ignores them in Phase 1 anyway.

**Normalization rationale:** Without normalization, the η dims (range 0.50–1.00) dominate the MSE over m_scale (range 0.80–1.20), making scale sensitivity unpredictable. Normalized targets put all dims on equal footing.

**No auxiliary losses.** Do not add a trajectory-tracking reward term — the actor is frozen, ϕ only needs to predict e_t accurately. Downstream tracking quality is validation, not a loss term.

---

## Training schedule

```
Optimizer:   Adam, β₁=0.9, β₂=0.999, ε=1e-8
LR:          5e-4 (no schedule; reduce by 0.5× if val MSE stops improving for 200 epochs)
Batch size:  1024 (windows, not episodes)
Epochs:      2 000
Val check:   every 50 epochs
Early stop:  if val MSE < 0.005 for 3 consecutive checks
```

**Window construction at training time:** For each episode, generate 1000 sliding-window samples (with zero-padding for t < H=50). Do NOT precompute all 20M windows to disk — generate on-the-fly per batch for memory efficiency.

**Estimated wall-clock:** ~30–60 minutes on 4070 (small network, 20M gradient steps).

---

## Validation

### Phase A — Offline MSE (necessary, not sufficient)

Run ϕ on held-out episodes. Compute MSE per e_t dimension:

| Gate | Value | Meaning |
|---|---|---|
| Overall MSE (normalized) | ≤ 0.02 | Encoder is predicting all dims reasonably |
| η MSE (dims 0–3) | ≤ 0.03 | Rotor fault is recoverable from history |
| m_scale MSE (dim 4) | ≤ 0.01 | Mass variation is easier — should be very low |

**What MSE = 0.02 means physically:** RMSE ≈ 0.14 per normalized dim → ≈ 0.035 on raw η scale. The actor was trained with e_t = ground truth; ê_t with 0.035 error is a small perturbation relative to the η range [0.50, 1.00]. From Phase 1 ablation, the architecture tolerates ±0.007 m MED perturbation at a 0.007-unit error in the observation — extrapolating, ê_t error of 0.035 may cost ≤ 0.005 m MED. Empirical closed-loop eval is required to confirm.

### Phase B — Closed-loop eval (go/no-go)

Replace ground-truth e_t with ê_t = ϕ(history) in actor obs. Run `scripts/eval_m2_full.py` (T/4-corrected) against Phase 1 checkpoint with ϕ active.

**Eval script modification needed:** Add `--encoder_checkpoint PATH` flag that loads ϕ and inserts a history buffer into the lax.scan loop.

Success criteria (Phase 2 go/no-go for M3):

| Metric | Target | Rationale |
|---|---|---|
| figure_eight_normal nominal (ϕ deployed) | ≤ 0.065 m | 14% degradation from Phase 1's 0.057 m. Phase 1 actor with slightly wrong ê_t should remain stable. |
| figure_eight_normal fault η=0.70 (ϕ deployed) | ≤ 0.100 m | Same gate as Phase 1 — encoder must maintain fault tolerance |
| Zero crashes on nominal (ϕ deployed) | required | ê_t errors should not destabilize hover |
| Deploying ϕ does not worsen fault vs nominal ratio | F/N ≤ 1.6× | Check encoder is differentiating fault from nominal, not averaging |

If Phase 1 with ground-truth e_t scored 0.057/0.079 m, deploying ϕ should stay within 14%/27% of those numbers respectively.

---

## Deployment

At deployment (eval or real hardware):

1. Maintain a ring buffer of the last H=50 (obs_base, action) pairs. Initialize to zeros.
2. At each step t:
   - Build `obs_base_t` = [e_W, v, R] (42-dim, no priv_state)
   - Compute `ê_t = ϕ(buffer)` (8-dim, via forward pass of ϕ)
   - Build actor input = [e_W, v, R, ê_t] (50-dim)
   - Run actor: action_t = μ_actor(actor_input)
   - Append (obs_base_t, action_t) to buffer (FIFO, drop oldest)

ê_t feeds into the same 8-dim priv_state slot in the actor observation — actor shape is unchanged.

**Latency note:** ϕ forward pass is ~0.14ms on GPU (2300→256→128→8, JIT-compiled). At 100Hz control loop this is 1.4% of budget — negligible.

---

## Success criteria (go/no-go for M3)

**Phase 2 ships if ALL of the following pass:**

- [ ] Phase 2 offline MSE ≤ 0.02 on held-out episodes (necessary prerequisite)
- [ ] η dims MSE ≤ 0.03 on held-out episodes
- [ ] figure_eight_normal nominal (ϕ deployed) ≤ **0.065 m**, zero crashes
- [ ] figure_eight_normal fault η=0.70 (ϕ deployed) ≤ **0.100 m**
- [ ] F/N ratio (ϕ deployed, figure_eight_normal) ≤ **1.6×**

**Partial pass / iterate rule:**
- Offline MSE fails → extend data collection to 50k episodes before changing architecture
- Offline MSE passes but closed-loop degrades > 14% → check zero-padding behavior at episode start; verify history buffer FIFO is correct; try H=100 if systematic
- Nominal passes, fault fails → ê_t is averaging away the fault signal; check MSE per dim — if η MSE high, check data distribution (is p=0.70 fault in rollouts?)
- F/N > 1.6× but both thresholds met → encoder is not discriminating fault from nominal; ê_t may be near-constant → check ê_t variance across conditions

**Single-variable rule:** Do not change both encoder architecture AND data collection simultaneously. Diagnose offline MSE first, then closed-loop.

---

## Failure modes

### F4 — MSE stalls above 0.05

**Symptom:** Val MSE > 0.05 after 500 epochs with no improvement.  
**Cause:** H=50 history insufficient to disambiguate fault type (similar obs/action patterns for different η configs in short windows), OR flat MLP insufficient for temporal structure.  
**Diagnostic:** Plot ê_t vs e_t scatter per dim. If η dims are all near 0.75 (the normalization midpoint) regardless of true η, the encoder is averaging. Try H=100 (double window). If still stuck, upgrade ϕ to RMA's 1D CNN: per-timestep embedding Linear(46→32) → ELU across H steps, then 1D CNN with 3 filters, flatten → Linear(8) → tanh.  
**Single-variable rule:** Try H first, then architecture.

### F5 — Zero-padding instability

**Symptom:** Nominal eval with ϕ crashes at episode start (steps 0–50), recovers thereafter.  
**Cause:** ϕ trained on zero-padded prefixes but ê_t output during zero-padding phase pushes actor into unstable region.  
**Fix:** In data collection, oversample early-episode windows (steps 0–50) to improve coverage of zero-padded inputs. Alternatively, linearly interpolate ê_t from zeros to predicted value over steps 0–H.

### F6 — Encoder sees ground-truth e_t during training

**Symptom:** Val MSE is very low (< 0.001), but closed-loop completely fails.  
**Cause:** obs_base_t was accidentally built from the 50-dim actor obs (including priv_state), leaking e_t into encoder input.  
**Prevention:** Strip priv_state before saving obs_base_t during data collection. Add assertion: `obs_base.shape[-1] == 42`.

### F7 — Phase 1 actor instability when ê_t ≠ e_t

**Symptom:** Small offline MSE but large closed-loop degradation (e.g., MED doubles).  
**Cause:** Actor is unexpectedly sensitive to ê_t errors — the 8-dim priv_state slot may not generalize to imperfect predictions.  
**Diagnostic:** Run closed-loop with e_t + Gaussian noise σ=0.05 as a sanity check. If this also degrades badly, the Phase 1 actor overfit to exact e_t and Phase 2 may not be feasible without Phase 1 retraining.  
**Fix (if needed):** Retrain Phase 1 with label noise on e_t (e_t + N(0, 0.05)) to force robustness to prediction error. This is a Phase 1 change, not a Phase 2 change.

---

## Out of scope for Phase 2

- Wind estimation: F_x/F_y/F_z were zero in Phase 1 training. ê_t dims 5-7 will be near-zero for all episodes. Wind handling is M2.x or M3.
- Two-rotor faults: not in Phase 1 DR, so not in Phase 2 data. OOD diagnostic only.
- Fine-tuning Phase 1 actor: frozen in Phase 2. Any change to actor = new Phase 1 run.

---

## Implementation checklist

- [ ] Write `scripts/collect_phase2_data.py` — VecEnv rollout, saves (obs_base_t, action_t, e_t) per step
- [ ] Write `src/iron_man_drone/policy/encoder.py` — ϕ network (Flax module, 2300→256→128→8→tanh)
- [ ] Write `scripts/train_phase2_encoder.py` — supervised training loop, Adam, MSE loss, val MSE logging
- [ ] Modify `scripts/eval_m2_full.py` — add `--encoder_checkpoint` flag + history buffer in scan loop
- [ ] Write `notes/M2_phase2_results.md` — offline MSE table + closed-loop eval table + go/no-go
