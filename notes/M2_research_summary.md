# M2 Research Summary — MAVEN and RMA

**Date:** 2026-05-05  
**Purpose:** Pre-spec extraction of concrete numbers from MAVEN (arXiv 2603.10714) and RMA (arXiv 2107.04034). Every number in M2_spec.md must trace back to a row in this document or a conscious deviation from it.

---

## Part 1 — MAVEN (arXiv 2603.10714)

### 1.1 What MAVEN is

MAVEN is a meta-RL framework for agile quadrotor flight under mass variation and rotor thrust degradation. It runs a probabilistic context encoder (off-policy) concurrently with a PPO agent (on-policy). Evaluated in sim and transferred zero-shot to real hardware.

**Most important fact for M2 planning:** MAVEN does NOT use RMA-style two-phase training. It uses a single concurrent training loop. This distinction matters for our implementation strategy (see §Part 3).

### 1.2 DR Ranges (exact)

| Parameter | Training range | OOD test |
|---|---|---|
| Quadrotor mass | [0.25 kg, 0.50 kg] | 0.55 kg |
| Mass as ratio of nominal (330g) | [0.76×, 1.52×] | 1.67× |
| Per-rotor thrust efficiency | η ∈ [0.50, 1.00] | η = 0.30 |
| Thrust loss | [0%, 50%] | 70% |
| Rotor fault mode | Single rotor per trial | Single rotor |
| Wind | Not mentioned | Not tested |

Key: MAVEN perturbs **one rotor at a time**, not all four independently. This prevents the total-thrust collapse that occurs when all rotors are simultaneously degraded (if all η = 0.5, max thrust = 50% of nominal → below T/W = 1 → can't hover).

Tested fault levels in paper: 0%, 15%, 30%, 45%, 60% thrust loss (in training), 70% (OOD).  
Tested masses: 260g, 330g (nominal), 440g, 550g. 550g is OOD.

No sensor noise specs. No wind disturbances. Wind is NOT part of MAVEN's DR.

### 1.3 Training Architecture

**Type: Single-phase concurrent training (off-policy encoder + on-policy PPO)**

Two learned components trained simultaneously:

1. **Probabilistic context encoder qφ(z | c_{1:N})** (off-policy)
   - Input: context window of N = 128 transition tuples (o_n, a_n, r_n, o'_n)
   - Each transition: o ∈ ℝ^27 (velocity 3 + rotation matrix 9 + waypoint deltas 3×W, W=2), a ∈ ℝ^2, r ∈ ℝ^1
   - Architecture: 2-layer MLP, 64 units each (activation not specified)
   - Output: Gaussian posterior parameters (mean, variance) over latent z ∈ ℝ^6 (D=6)
   - Loss: L_encoder = ω_KL × L_KL + L_pred + ω_spec × L_spec
     - L_KL: KL divergence regularization
     - L_pred: MSE on next-state prediction + Huber loss on reward prediction
     - L_spec: specialization loss (prevents encoder collapse)
   - Encoder update frequency: every Nenc = 3 PPO steps

2. **PPO policy πθ(a | o, z)** (on-policy)
   - Actor takes [observation, latent z] as input
   - Standard PPO training

Critic Qψ(o, a, z) is also trained.

**Key implication:** MAVEN requires an off-policy replay buffer per task and a Bayesian context encoder (ELBO loss). This is substantially more infrastructure than a PPO loop.

### 1.4 Evaluation Protocol

Metrics: completion time, average velocity, max velocity, success rate, % flight with throttle > 0.8.

Trajectories: switchback, "8-3laps" (24 waypoints), "butterfly" (23 waypoints), 5-waypoint random (100 trials per scenario), M-shaped (5 waypoints), A-shaped (13 waypoints).

Real hardware: 330g nominal quadrotor with Betaflight autopilot, motion-capture feedback, 100 Hz control loop. Prop-swap to induce 30%, 45%, 70% thrust loss.

### 1.5 Training Compute

- GPU: NVIDIA RTX 5090 D
- Simulator: Genesis (GPU-vectorized Python simulator, much faster than MJX)
- Parallel envs: 4,096 envs × 16–20 concurrent tasks
- Training time: 35 min (mass variation, 4.92B timesteps) / 53 min (thrust loss, 7.37B timesteps)
- Control loop: 100 Hz

**Compute gap warning:** MAVEN uses an RTX 5090 with Genesis running 4K envs × 20 tasks concurrently. Our setup is an RTX 4070 with MJX at ~65k env-steps/sec. We cannot replicate MAVEN's timestep count in the same wall-clock time. Our M2 training will run ~15k PPO epochs at MJX throughput, equivalent to roughly the same schedule as M1.3 (~4–5 hrs).

---

## Part 2 — RMA (arXiv 2107.04034)

### 2.1 Two-Phase Training Scheme

RMA was designed for legged locomotion, but its training architecture is the standard reference for adaptation encoder design.

**Phase 1 — Privileged policy training (PPO):**

- Base policy π: 3-layer MLP, 128 hidden units. Input: [nominal_obs (30-dim), z (8-dim)] where z = μ(e_t).
- Privileged environment encoder μ: 3-layer MLP, hidden [256, 128]. Input: e_t (17-dim privileged state = mass, motor strengths, friction, terrain height). Output: z ∈ ℝ^8.
- μ is trained jointly with π via PPO gradients (end-to-end differentiable).
- Duration: 15,000 PPO iterations, ~1.2B timesteps, ~24 hrs on their hardware.

**Phase 2 — Adaptation encoder training (supervised):**

- Input to encoder: history of K = 50 (state, action) pairs over a 0.5-second window (50 steps at 100 Hz).
- Architecture: 2-layer MLP embedding per timestep (input 46-dim per timestep, output 32-dim) → reshape to [50, 32] → 3-layer 1D CNN:
  - Conv1: [in=32, out=32, kernel=8, stride=4]
  - Conv2: [in=32, out=5, kernel=1, stride=1]
  - Conv3: [in=5, out=1, kernel=1, stride=1]
  - → flatten → 8-dim output
- Loss: MSE(ẑ_t, z_t) where z_t = μ(e_t) from **frozen Phase 1 encoder μ**. Policy π also frozen.
- Training: Adam lr=5e-4, 1,000 iterations, batch 80,000.
- Duration: ~3 hrs, ~80M timesteps.

### 2.2 Key RMA Properties

- Phase 1 policy uses privileged z during training → learns to use the latent to adapt behavior.
- Phase 2 encoder learns to reconstruct z from observable history (no access to e_t at deploy time).
- Deployment: actor uses ẑ from ϕ(history) instead of z from μ(e_t). Policy weights unchanged from Phase 1.
- Previous Claude session validated this architecture on our MJX stack: MSE 0.000267 on adaptation encoder output.

---

## Part 3 — M2 Implementation Strategy

### 3.1 Why RMA-style, not MAVEN-style

MAVEN's joint training requires:
1. Off-policy experience buffer per task (stores (o, a, r, o') tuples for 128-step context windows)
2. ELBO loss with KL, next-state prediction, reward prediction, specialization term
3. Genesis simulator's throughput to make this feasible at reasonable wall-clock time
4. Multi-task batch structure (16–20 concurrent tasks)

Our stack is: MJX + Flax + single-task PPO. Adding off-policy infrastructure would be a multi-week infrastructure task orthogonal to the goal of M2.

**Decision: Use RMA two-phase. Borrow MAVEN's DR ranges. This gives us the fault tolerance goal without rewriting the training framework.**

Justification:
- RMA two-phase is proven on legged robots with similar perturbation types (mass, motor strength)
- Our own previous session validated Phase 2 on this stack (MSE 0.000267)
- MAVEN's core contribution is the meta-RL framing and the empirical result that adaptation works for quadrotors; the DR ranges are the part directly transferable to us

### 3.2 DR Range Adaptation for Crazyflie (32.1 g nominal)

MAVEN's drone is 330g nominal; ours is 32.1g. The dimensionless perturbation ratios transfer; the absolute values do not.

| Parameter | MAVEN ratio | M2 training range | M2 OOD test |
|---|---|---|---|
| Per-rotor efficiency (single rotor) | [0.50, 1.00] | [0.50, 1.00] | 0.30 |
| Rotor fault mode | 1 rotor/episode | 1 rotor/episode | 1 rotor |
| Mass scale (relative to nominal) | [0.76, 1.52] | [0.80, 1.20] | 0.70, 1.30 |
| Wind force (per axis) | N/A | [-0.05, 0.05] N | ±0.07 N |

Mass range [0.80, 1.20] is conservative relative to MAVEN — we start narrower because the Crazyflie is a smaller, less stable platform and we want to preserve nominal performance. Expand to [0.70, 1.30] if nominal holds.

Wind range 0.05 N/axis is ~16% of Crazyflie weight (0.315 N), chosen to be physically meaningful but not dominating.

### 3.3 Rotor-Fault Physical Constraint

**Critical:** If all 4 rotors are independently randomized to η ∈ [0.5, 1.0], worst-case total thrust = 4 × 0.5 = 2.0 units (50% of nominal). For Crazyflie T/W ≈ 1.8, this means max thrust = 0.9 × weight → can't hover. Training on unhoverably degraded states produces degenerate gradients.

**Fix:** Match MAVEN's convention — perturb exactly one randomly selected rotor per episode. The other three hold η = 1.0. With one rotor at η_min = 0.5, total thrust = 3.5/4.0 = 87.5% of nominal → T/W ≈ 1.57 → can hover and maneuver.

The privileged state still encodes all four rotor efficiencies [η_1, η_2, η_3, η_4]. In nominal episodes (30% of training), all are 1.0. In fault episodes (70%), three are 1.0 and one is sampled from [0.5, 1.0].

### 3.4 Latent z Dimension

MAVEN uses D = 6. RMA uses D = 8. Our privileged state is 8-dim [η_1–4, m_scale, F_wind_xyz]. We use D = 8, matching the state dimensionality.

---

## Summary Table

| Source | Phase | Architecture | Latent dim | History window | Loss |
|---|---|---|---|---|---|
| MAVEN (2603.10714) | Joint | MLP 2L×64 + PPO | 6 | 128 transitions | ELBO (KL + pred + spec) |
| RMA (2107.04034) Ph1 | PPO | MLP [256,128] priv enc | 8 | N/A (priv state) | PPO |
| RMA (2107.04034) Ph2 | Supervised | MLP embed + 1D CNN | 8 | 50 steps | MSE |
| **M2 (our plan)** | **Two-phase** | **MLP [64,64] priv enc; MLP [256,128,8] adapt** | **8** | **50 steps** | **Ph1: PPO; Ph2: MSE** |

The M2 plan uses RMA's training scheme, MAVEN's DR ranges, and an MLP adaptation encoder (simpler than RMA's CNN, easier to implement in Flax/JAX, can be replaced with CNN if MSE is too high in Phase 2).
