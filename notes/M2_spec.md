# M2 Spec — Fault-Tolerant Flight via RMA + MAVEN DR

**Date:** 2026-05-05  
**Prerequisite reading:** notes/M2_research_summary.md, notes/M1_results.md, notes/lessons.md  
**Status:** Draft — requires hypothesis doc approval before any code is written

---

## One-line goal

Same trajectory-tracking performance as M1.3 (figure_eight_normal MED ≤ 0.037m, no regression), plus graceful degradation under rotor faults and mass variation — achieved via RMA two-phase training with DR ranges informed by MAVEN (arXiv 2603.10714).

---

## What carries over from M1.3 (unchanged)

Every element of M1.3 is preserved:

- C2-continuous quintic polynomial trajectory generator (`envs/trajectories.py`)
- Reward off-by-one fix (`envs/quadrotor_env.py`)
- T/4 phase offset for figure-eight eval (`scripts/eval_m1_full.py`)
- Asymmetric actor/critic (3-layer MLP, 256 hidden, ELU + LayerNorm)
- Separate actor/critic Flax modules with separate Optax optimizers
- PPO hyperparameters (γ=0.99, λ=0.95, clip=0.2, entropy_coeff=1e-3, actor_lr=3e-4, critic_lr=1e-4)
- Existing k_f ± 30% thrust coefficient DR (kept, now one of several DR axes)
- All eval scripts, diagnostic scripts, notes structure
- No u_{t-1} in actor. No k in actor. CTBR action space. Rotation matrix in obs.

---

## What changes for M2

Three additions, introduced together in Phase 1 (no sequential fine-tuning):

### A. Expanded Domain Randomization

Per episode, sample the physical perturbation vector e_t ∈ ℝ^8 (constant for the entire episode):

```
e_t = [η_1, η_2, η_3, η_4,   # per-rotor efficiency (4-dim)
       m_scale,                # mass multiplier (1-dim)
       F_x, F_y, F_z]         # constant wind force in world frame (3-dim)
```

**Per-rotor efficiency sampling (Phase 1 training):**
- With probability 0.30: all η_i = 1.0 (nominal episode)
- With probability 0.70: select one rotor j ∈ {1,2,3,4} uniformly at random; set η_j ~ U(0.50, 1.00); all others = 1.0
- Rationale: MAVEN's convention, prevents total-thrust collapse (see research_summary §3.3). With one rotor at η_min=0.50, total thrust = 3.5/4.0 = 87.5% of nominal → T/W ≈ 1.57 → can hover and maneuver.
- Multi-rotor faults and η=0.30 OOD: held-out eval only, not Phase 1 training.

**Mass scale sampling:**
- m_scale ~ U(0.80, 1.20) (±20% of nominal 0.0321 kg)
- Applied as: m_sim = m_scale × m_nominal in MuJoCo dynamics

**Wind force: excluded from Phase 1 training.** Added to held-out eval at F = [0.05, 0.05, 0.02] N. Rationale: wind + rotor fault + mass variation together may destabilize early training. Smaller scope = better debugging. If Phase 2 ϕ generalizes to wind in OOD eval, we get it free. If eval fails, M2.x adds wind to Phase 1.

**k_f DR (carried from M1):** k_f_actual = k_f_nominal × U(0.70, 1.30)

---

### B. Privileged Encoder μ (Phase 1 only)

A small MLP that maps the privileged state e_t to a latent z, trained jointly with the actor via PPO gradients.

**Architecture:**

```
μ: [8] → Linear(64) → ELU → Linear(64) → ELU → Linear(8) → tanh → z ∈ [-1, 1]^8
```

- Input: e_t (8-dim privileged physical state, constant per episode)
- Output: z ∈ ℝ^8, tanh-bounded to [-1, 1] for stable integration with actor
- Parameters: ~1,100 weights (negligible training cost)
- Gradient source: PPO actor loss flows backward through z = μ(e_t) into μ weights

**Integration into Phase 1 policy:**

```
# Actor (50-dim input):
actor_obs = [e^W (30), v (3), R (9), z (8)]   # z = μ(e_t)

# Critic (51-dim input):
critic_obs = [e^W (30), v (3), R (9), e_t (8), k (1)]
             # critic gets raw privileged state, not z — better value estimates
```

Actor architecture: Linear(50→256) → ELU → LayerNorm → Linear(256→256) → ELU → LayerNorm → Linear(256→4)  
Critic architecture: Linear(51→256) → ELU → LayerNorm → Linear(256→256) → ELU → LayerNorm → Linear(256→1)

The 8-dim expansion of the actor input (42 → 50) and critic input (43 → 51) are the only network architecture changes from M1.3.

**Pre-training sanity check for z:** Before training, verify μ is not producing z ≈ 0 for all inputs. Run 100 random e_t samples and confirm variance(z) > 0.01 per dimension. If the encoder collapses, the actor never learns to use z.

---

### C. Adaptation Encoder ϕ (Phase 2 only)

Trained after Phase 1 completes, fully supervised, with actor + μ frozen.

**Purpose:** Learn to predict z = μ(e_t) from observable history alone (no privileged state at deploy time).

**Input:**
- History of K = 50 (observation, action) pairs from the last 0.5 seconds (50 steps × 10ms)
- Per step: o_t (42-dim) + a_t (4-dim) = 46-dim
- Total input: 50 × 46 = 2300-dim (flattened)

**Architecture:**

```
ϕ: flatten(2300) → Linear(256) → ELU → Linear(128) → ELU → Linear(8) → tanh → ẑ ∈ [-1, 1]^8
```

Simple flat MLP. Chosen over RMA's 1D CNN for implementation simplicity in Flax/JAX. If Phase 2 MSE stalls above threshold, upgrade to per-timestep MLP embedding + 1D CNN (see research_summary §2.1).

**Phase 2 training procedure:**

1. Collect data: roll out Phase 1 policy (actor + μ, parameters frozen) for 20,000 episodes with full DR active. Record (history_{1:50}, z_t) pairs where z_t = μ(e_t) with frozen μ.
2. Train ϕ: minimize MSE(ẑ, z). Adam, lr=5e-4, batch=1024, 2,000 epochs.
3. Validate: compute MSE on held-out 20% of episodes.
4. Gate: MSE ≤ 0.02 before deploying ϕ.

**Deployment:** Replace z = μ(e_t) in actor input with ẑ = ϕ(history). Actor weights unchanged.

**History buffer implementation note:** During eval, maintain a ring buffer of the last K=50 (o_t, a_t) pairs. Initialize with zeros (zero history at episode start). At each step: append (o_t, a_t) to buffer, compute ẑ = ϕ(buffer), pass ẑ to actor alongside o_t.

---

## Training Schedule

### Phase 1: Privileged Policy Training

- Algorithm: PPO (identical to M1.3)
- Duration: 15,000 epochs (same as M1.3; ~4–5 hours on 4070)
- Gating checkpoint: evaluate at epoch 5,000 and 10,000. If nominal figure_eight_normal MED > 0.060m at epoch 5,000, stop and diagnose (see Failure Modes §F1).
- Convergence signal: same as M1.3 — reward trending up, entropy slow-decreasing, MED dropping
- Do NOT pause and resume (L1 from lessons.md). Run uninterrupted.

**First training run must validate nominal before trusting robustness numbers:**
The very first eval of Phase 1 must include the nominal scenario (η = [1,1,1,1], m_scale = 1.0, F_wind = 0). If nominal MED regresses by > 20% vs M1.3, abort and diagnose before running fault-scenario evals.

### Phase 2: Adaptation Encoder Training

- Duration: 2,000 supervised epochs (~30–60 minutes)
- Data collection: 20,000 Phase 1 rollouts (1,000 per fault condition × 20 conditions)
- Single-variable rule: Phase 2 changes only ϕ — do not alter actor, μ, or DR parameters between Phase 1 and Phase 2

---

## Eval Suite

Phase 1 eval (runs at training checkpoints, same as M1.3):

| Trajectory | Target | Notes |
|---|---|---|
| figure_eight_slow | MED ≤ 0.020 m | M1.3 achieved 0.017m |
| figure_eight_normal | MED ≤ 0.042 m | M1.3 achieved 0.037m (15% margin for M2 overhead) |
| figure_eight_fast | MED ≤ 0.100 m | M1.3 achieved 0.090m |
| pentagram_slow | — (monitor) | M1.3 was 0.054m |

M2 extended eval (after Phase 2, using ϕ):

In-distribution fault scenarios (trained on):

| Scenario | Parameter | Target | Justification |
|---|---|---|---|
| Nominal | η=[1,1,1,1], m=1.0×, wind=0 | MED ≤ 0.037m, zero crashes | Must match M1.3 exactly |
| Mild rotor fault | One rotor at η=0.70 | MED ≤ 0.060m, zero crashes | ≤1.6× nominal; MAVEN reports near-full performance at 30% loss |
| Moderate rotor fault | One rotor at η=0.50 | MED ≤ 0.100m, zero crashes | ≤2.7× nominal; at the edge of training distribution |
| Mass + | m_scale = 1.20 | MED ≤ 1.5× nominal | Upper training bound |
| Mass − | m_scale = 0.80 | MED ≤ 1.5× nominal | Lower training bound |

OOD eval (not trained on — diagnostic, not go/no-go):

| Scenario | Parameter | Target |
|---|---|---|
| Heavy rotor fault (OOD) | One rotor at η=0.30 (70% loss) | Flight sustained ≥ 500 steps, no crash-at-start |
| Two-rotor fault (OOD) | Two rotors at η=0.70 | Flight sustained ≥ 300 steps |
| Wind (OOD) | F = [0.05, 0.05, 0.02] N constant | MED ≤ 2.0× nominal (pass would be a free win) |
| Combined (OOD) | One rotor η=0.70 + m=1.2× | Flight sustained ≥ 500 steps |

**Threshold justification for η=0.70 (≤0.060m):** MAVEN reports that at 30% single-rotor thrust loss, their RL-DR baseline completes all tracks with only small velocity reductions. 0.060m = 1.6× our nominal target of 0.037m. With the adaptation encoder correctly identifying the fault and pre-compensating the torque asymmetry, 1.6× degradation is achievable. If we miss this threshold, the encoder is not effectively encoding the fault.

**Threshold justification for η=0.50 (≤0.100m):** MAVEN's RL-DR baseline shows more degradation at 50% loss (lower success rates). 0.100m = 2.7× nominal. At η=0.50 single rotor, the torque imbalance requires sustained attitude correction; degradation to 2.7× is realistic without active compensation overfit to this specific fault level.

All eval uses corrected initialization (T/4 offset for figure-eight). MED = arithmetic mean over full 1000-step episode.

---

## Success Criteria (Go/No-Go for M3)

**M2 ships if and only if ALL of the following pass:**

(a) Nominal performance — hard requirement, no regression from M1.3:
- [ ] figure_eight_normal MED ≤ **0.037 m** (M1.3 level — not 1.15×, not any margin)
- [ ] figure_eight_slow MED ≤ 0.020 m
- [ ] figure_eight_fast MED ≤ 0.100 m
- [ ] Zero crashes on nominal eval suite

Rationale: z augmentation should be additive. Under nominal conditions (η=[1,1,1,1], m=1.0×), the encoder should produce a latent z the policy can effectively treat as "no compensation needed," recovering M1.3 behavior. Nominal regression is a bug in z integration, not an accepted cost.

(b) Graceful degradation under in-distribution faults:
- [ ] Single rotor η=0.70: figure_eight_normal MED ≤ **0.060 m**, zero crashes
- [ ] Single rotor η=0.50: figure_eight_normal MED ≤ **0.100 m**, zero crashes
- [ ] Mass ±20%: MED ≤ 1.5× nominal (≤ 0.056 m)

(c) Adaptation encoder:
- [ ] Phase 2 MSE(ẑ, z) ≤ 0.02 on held-out eval
- [ ] Deploying ϕ (replacing μ) does not degrade nominal MED by > 10% (≤ 0.041 m)

**Partial pass / iterate rule:**
- Nominal passes, fault robustness fails → diagnose encoder quality (is z variance high enough?), do not widen DR yet
- Fault robustness passes, nominal regresses → z integration is the bug; try reducing z dim or checking μ init; do NOT accept nominal regression
- Encoder MSE high → check μ output variance before changing ϕ architecture

---

## Failure Modes to Watch

### F1 — Nominal regression (most likely)
**Symptom:** Phase 1 nominal MED > 0.050m at epoch 10,000 (worse than M1 baseline).  
**Cause candidates:** z = 0 collapse in μ (policy ignores z), z variance is too high (noisy signal confuses actor), DR so aggressive that most rollouts crash and gradient is degenerate.  
**Diagnostic:** Check z variance across episodes. Plot crash rate per epoch. Compare reward curve slope to M1.3.  
**Fix:** If z variance < 0.01: add a diversity penalty or check μ init. If crash rate > 20% in epoch 1-500: reduce fault probability from 0.70 to 0.30 for first 5k epochs.  
**Single-variable rule:** Do not change both z_dim AND DR range in the same recovery run.

### F2 — μ encoder collapse (z ≈ 0 for all inputs)
**Symptom:** variance(z) ≈ 0 across different e_t samples. Policy tracks trajectories but does so identically regardless of rotor condition.  
**Cause:** PPO gradient through z is too small to overcome initialization; μ stays near zero.  
**Diagnostic:** Log mean and variance of z at each epoch. If variance < 0.001 after epoch 1000, μ is collapsed.  
**Fix:** Increase μ learning rate (separate optimizer from actor, lr=1e-3). Add L2 norm regularization to z to prevent trivial zero.

### F3 — Phase 1 instability from aggressive DR
**Symptom:** Crash rate > 30% in epochs 1-500. Reward does not climb past early baseline.  
**Cause:** η=0.50 single-rotor fault + random wind + mass variation together create environments the untrained policy can never escape. No positive reward signal.  
**Fix:** Curriculum in a single run — linear ramp of fault probability from 0.0 to 0.70 over epochs 0–5,000. This is not sequential fine-tuning (single training run, not two phases), so it doesn't violate the anti-pattern from PROJECT_PLAN. Flag this as a departure from "full DR from epoch 0" if it's needed.

### F4 — Phase 2 MSE stuck high (> 0.10)
**Symptom:** Adaptation encoder cannot predict z from history.  
**Cause candidates:** (a) μ output has low variance (not much to predict); (b) K=50 history is too short to disambiguate fault type; (c) flat MLP insufficient for temporal structure.  
**Diagnostic:** Check μ output variance (if this is also F2, fix F2 first). Try increasing K to 100. If still stuck, upgrade ϕ to RMA's 1D CNN architecture.  
**Single-variable rule:** Try one fix at a time.

### F5 — Resume-after-pause (L1 from lessons.md)
**Prevention:** Do not interrupt Phase 1. If a pause is unavoidable, save full optimizer state (Adam mu, nu, step count) — not just params. Do not rely on Orbax default checkpointing for optimizer state without verification. Prefer restarting from epoch 0 to resuming at epoch > 10,000.

### F6 — Rotor-fault eval signal inflated by non-fault comparison
**Symptom:** MED at η=0.70 looks better than nominal (the fault actually helps?).  
**Cause:** Eval randomization is seeded wrong, or η is not actually applied in env dynamics.  
**Diagnostic:** Log η_applied at each eval step. Verify MuJoCo thrust model respects η scaling. Run sanity: a random policy at η=0.30 should crash quickly; if it doesn't, DR is not wired.

---

## Out of Scope for M2

- Obstacle avoidance (M3)
- Visual perception (M3)
- Two-rotor simultaneous faults (added if M2 passes with margin)
- Real hardware deployment
- MAVEN's off-policy encoder (too much infrastructure for the gain)

---

## Estimated Time Budget

| Phase | Task | Wall-clock |
|---|---|---|
| Pre-training | DR implementation, μ/ϕ architecture, obs pipeline | 1 day |
| Phase 1 | 15k PPO epochs | ~5 hours |
| Analysis | Checkpoint evals, failure diagnosis | ~2 hours |
| Phase 2 | Data collection + supervised encoder training | ~2 hours |
| Eval | Full M2 eval suite with ϕ | ~1 hour |
| Write-up | M2_results.md | ~1 hour |
| **Total** | | **~2 days of focused work** |

3-week calendar budget (from PROJECT_PLAN) — most buffer is debugging time.
