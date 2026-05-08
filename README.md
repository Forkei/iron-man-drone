# Iron Man Drone

Progressive reinforcement learning project for agile quadrotor flight. Five milestones from a clean SimpleFlight reproduction to GRaD-Nav++ language-commanded flight. Built on MuJoCo MJX + JAX.

---

## Milestones

| # | Goal | Status |
|---|---|---|
| M1 | SimpleFlight recipe on MuJoCo MJX — figure-eight tracking | **DONE** |
| M2 | Fault tolerance via RMA two-phase training + MAVEN-style DR | Phase 1 done, Phase 2 pending eval fix |
| M3 | Visual obstacle avoidance | Pending M2 |
| M4 | 3DGS-SLAM mapping + landmark return | Pending M3 |
| M5 | GRaD-Nav++ — natural language flight commands | Pending M4 |

---

## M1 Results

SimpleFlight (Chen et al., RAL 2025) on MuJoCo MJX with the same Crazyflie 2.1 parameters and PPO hyperparameters from their Table VI. All evals use the corrected T/4 phase offset (see [Eval methodology](#eval-methodology)).

| Trajectory | Ours (M1.3) | Paper (SimpleFlight) | Pass |
|---|---|---|---|
| figure_eight_slow (T=15s) | **0.017 m** | 0.016 m | ✓ |
| figure_eight_normal (T=5.5s) | **0.037 m** | 0.028 m | ✓ |
| figure_eight_fast (T=3.5s) | **0.090 m** | 0.051 m | ✓ |
| pentagram_slow | 0.054 m | 0.024 m | — |
| polynomial (random, 3-seed) | 0.016 ± 0.002 m | 0.032 m | — |
| zigzag (random, 3-seed) | 0.027 ± 0.001 m | 0.052 m | — |

Thresholds from SimpleFlight Table III: figure_eight_slow ≤ 0.050 m, figure_eight_normal ≤ 0.056 m, figure_eight_fast ≤ 0.150 m. All pass. Pentagram and random-trajectory numbers are OOD diagnostics, not gated.

The gap on figure_eight_fast and pentagram is real — the policy lacks rotational inertia compensation for aggressive trajectories. This is an acceptable M1 limitation; M2's task is fault tolerance, not closing that gap.

---

## M2 Results (Phase 1 — in progress)

M2 extends M1.3 with RMA two-phase training: Phase 1 trains a privileged policy that observes the physical perturbation vector directly; Phase 2 (pending) trains a causal encoder to predict it from observable history.

**Domain randomization active in Phase 1 training:** one-rotor efficiency fault (η ∈ [0.5, 1.0], probability 0.70 per episode), mass variation ±20%, k_f ±30%.

All numbers use the corrected T/4 methodology (see below). Previously reported numbers (t=0 baseline) were inflated by ~0.031 m.

| Condition | figure_eight_normal MED (T/4-corrected) |
|---|---|
| M1.3 baseline (no DR) | 0.037 m |
| M2 no-DR ablation, epoch 7k | 0.044 m |
| M2 + full DR, epoch 15k (extrapolated) | ~0.060 m |
| M2 spec target | 0.037 m |

The architecture itself is not the bottleneck: M2 no-DR at 0.044 m is only 1.18× above M1.3, not the 2.5× apparent from raw inline eval numbers. The remaining gap under full DR (~0.060 m) is driven by the added difficulty of adapting to per-episode rotor faults.

**Phase 2 is on hold** until `eval_m2_full.py` is corrected with the T/4 offset and re-run against the 15k checkpoint. The current full-eval numbers are not comparable to M1.3.

---

## Architecture

### Action space

CTBR — collective thrust + body rates (ω_x, ω_y, ω_z). The policy outputs a 4-dim Gaussian; a rate-PD controller with motor mixing and a first-order lag (τ = 25 ms) converts CTBR to per-rotor speeds. Direct motor command outputs were not explored; the paper uses CTBR and it matters.

### Observations

```
Actor  (50-dim, M2): [e^W (30), v (3), R (9), z (8)]
Critic (51-dim, M2): [e^W (30), v (3), R (9), e_t (8), k (1)]
```

- **e^W**: relative positions to the next 10 reference waypoints in world frame, spaced 50 ms ahead (10-point lookahead × 3 dims = 30).
- **v**: linear velocity in world frame from MuJoCo `qvel[:3]`.
- **R**: rotation matrix body→world, flattened (9-dim). Never quaternion — empirically ~63% performance drop with quaternion on figure-eight.
- **z (M2)**: 8-dim privileged latent. In Phase 1: z = e_t passed directly. At deployment: z = ϕ(history) from the causal encoder.
- **e_t**: raw privileged state [η₁, η₂, η₃, η₄, m_scale, F_x, F_y, F_z]. Critic only — better value estimates with ground-truth physical params.
- **k**: normalized episode step ∈ [0, 1]. Critic only — putting k in actor observations causes OOD failures on long flights.

Body rates ω and previous action u_{t−1} are excluded from the actor. Both cause non-stationarity or redundancy with R.

### Policy network

3-layer MLP, 256 hidden units, ELU activations + LayerNorm after each layer. Asymmetric: actor and critic are separate Flax modules with separate Optax optimizers.

```
Actor:  50 → 256 → 256 → 256 → 4   (CTBR mean + log_std)
Critic: 51 → 256 → 256 → 256 → 1   (scalar value)
```

### PPO hyperparameters

From SimpleFlight Table VI, unchanged:

| Parameter | Value |
|---|---|
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Clip ε | 0.20 |
| Actor LR | 3e-4 |
| Critic LR | 1e-4 (separate, lower) |
| Critic updates per actor update | 16 |
| Entropy coefficient | 1e-3 |
| Horizon | 32 steps/env |
| Minibatches | 8 |
| PPO epochs | 5 |
| Envs | 2048 |

16 critic updates per actor update is unusual but follows the paper. Entropy coefficient of 1e-3 keeps entropy below 10% of reward at initialization — a hard requirement; higher values prevent convergence.

### Training infrastructure

Rollout is a single `jax.lax.scan` over (state, action) — fully JIT-compiled, no Python loop overhead. GAE computed in a second reverse scan. 2048 environments via `jax.vmap` over a shared `mjx.Model`. Throughput: ~70k steps/sec on an RTX 4070. 15k epochs at horizon=32 take approximately 3.75 hours.

### Trajectory system

**Lazy design**: trajectories store coefficients and evaluate positions on demand. Precomputing a 1000-step position array per env at reset caused a 17× throughput loss from the vmap cost — abandoned.

**Training distribution** (50/50 per episode):
- C2-continuous quintic polynomial: random 5th-degree polynomial per segment, with nonzero interior velocities/accelerations (C2 continuity). Early attempts used a scalar h(τ) applied to straight-line segments, producing κ=0 everywhere — the drone learned to stop-and-pivot rather than curve, causing persistent failures on figure-eight apices.
- Random zigzag: waypoints in [−1,1]², connected by straight segments. Intentionally infeasible (waypoint corners require infinite acceleration). Included for OOD generalization.

**Eval trajectories** (held-out, never seen during training):
- Figure-eight: analytic Lemniscate at three speeds (slow/normal/fast).
- Pentagram: star-pentagon at two speeds, traversed in star-skipping order.

### Domain randomization (M2)

Sampled at episode reset, constant within episode, stored in `EnvState`:

- **Rotor efficiency** η_i: with p=0.70, one rotor is faulted at η_j ~ U(0.50, 1.00); others at 1.0. With p=0.30, all nominal. At η=0.50, total thrust is 87.5% of nominal → T/W ≈ 1.57, still controllable.
- **Mass scale**: m_scale ~ U(0.80, 1.20), applied as extra external force.
- **Thrust coefficient**: k_f ± 30% (carried from M1).
- **Wind**: excluded from Phase 1 training; OOD-only in Phase 2 eval.

Applied via: `effective_speeds = rotor_speeds × sqrt(kf_mult × η_i)`.

---

## Eval methodology

SimpleFlight evaluates figure-eight trajectories by initializing the reference at t = T/4, not t = 0. At T/4, the Lemniscate evaluates to (0, 0, 1) — exactly where the drone spawns — so initial XY error is zero. This is not stated in the paper; it is code-only behavior in `omni_drones/envs/single/track.py`:

```python
self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 5.5 / 4
```

At t = 0, the reference is at (1, 0, 1) while the drone is at (0, 0, 1): 1 m initial error. The policy acquires the trajectory within ~100 steps, but the mean over 1000 steps includes this acquisition phase, inflating MED by ~0.031 m. All M1.3 results use the corrected T/4 offset. M2 inline training evals do not yet have this fix (documented; required before Phase 2 go/no-go).

---

## Lessons from failed attempts

**L1 — Resume-after-pause kills near-converged PPO.** Restoring params while reinitializing Adam momentum caused a jump from 0.085 m to 0.215 m MED immediately post-resume, followed by 11,000 epochs of unrecoverable crash-at-step-8 behavior. The policy entered a feedback loop where short rollouts (crash at step 8) provided gradient signal only for the initial observation, reinforcing the crash. Run uninterrupted; if forced to resume, restore full optimizer state including Adam μ, ν, and step count.

**L2 — Training reward is not a proxy for eval MED.** Training on 2048 random trajectories allows high mean reward even when the held-out figure-eight eval fails completely. A policy crashing on every figure-eight episode can maintain training reward of 1.33 because randomly-initialized polynomial episodes provide positive signal. Use held-out MED as the primary convergence signal.

**L3 — Polynomial curvature coverage matters.** The original polynomial generator applied a scalar quintic profile to straight-line segments (κ=0 everywhere). The figure-eight apex has κ=4.789 m⁻¹. With 0% coverage of this curvature in training, the policy plateaued at 0.105 m on figure-eight. Replacing with C2-continuous piecewise quintic polynomials (random nonzero interior velocities) immediately dropped MED to 0.085 m in 2000 epochs.

---

## Setup (WSL2)

```bash
# Requires WSL2 Ubuntu 22.04+ with NVIDIA GPU passthrough

# 1. Environment setup (one-time)
bash scripts/setup_env.sh
source ~/.bashrc

# 2. Activate JAX environment
source /home/$USER/jax_env/bin/activate

# 3. Verify GPU
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]

# 4. Train M1
python scripts/train_m1.py

# 5. Evaluate M1
python scripts/eval_m1_full.py \
  --checkpoint experiments/m1_3_polynomial_fix/RUN/checkpoints/epoch_013000

# 6. Train M2 Phase 1
python scripts/train_m2.py --config experiments/m2_phase1_baseline/config.yaml --num_envs 2048
```

**Stack:** Python 3.12, JAX with CUDA 12 (`pip install "jax[cuda12]"`), MuJoCo ≥ 3.1.6 (MJX bundled), Flax, Optax, Distrax, Orbax.

---

## Repo layout

```
src/iron_man_drone/
  envs/
    crazyflie.xml             MuJoCo MJCF — Crazyflie 2.1 dynamics, forces applied programmatically
    quadrotor_env.py          MJX env: vmap over 2048 envs, M2 DR, priv_state
    trajectories.py           Lazy trajectory system: quintic poly, zigzag, figure-eight, pentagram
  control/
    ctbr_controller.py        CTBR → rotor speeds (rate PD + X-config mixer + motor lag)
  policy/
    networks.py               Flax Actor (50→4) + Critic (51→1), 3×256 ELU+LayerNorm
    ppo.py                    PPO trainer: lax.scan rollout, separate actor/critic TrainStates
  utils/
    domain_randomization.py   M2 DR parameter sampling
experiments/
  m1_3_polynomial_fix/        M1.3 final run (config + eval results; checkpoints gitignored)
  m2_phase1_baseline/         M2 Phase 1 run
notes/
  M1_hypothesis.md            Required gating artifact before any M1 training
  M1_3_eval_methodology.md    T/4 phase offset discovery and justification
  M1_3_results.md             M1.3 final eval table
  M2_spec.md                  M2 design spec (RMA architecture, DR ranges, success criteria)
  m2_ablation_eval_clarification.md  Methodology mismatch analysis; Phase 2 decision gate
  lessons.md                  L1–L4: failure post-mortems
scripts/
  train_m1.py                 M1 training entry point
  train_m2.py                 M2 training entry point
  eval_m1_full.py             Full M1 eval with T/4 offset (source of M1.3 0.037 m number)
  eval_m2_full.py             Full M2 eval suite (T/4 fix pending)
  eval_m2_methodology_check.py  M2 vs M1.3 methodology comparison script
```

---

## Dynamics (Crazyflie 2.1)

| Parameter | Value |
|---|---|
| Mass | 0.0321 kg |
| Inertia (Ixx, Iyy) | 1.4 × 10⁻⁵ kg·m² |
| Inertia (Izz) | 2.17 × 10⁻⁵ kg·m² |
| Thrust coefficient k_f | 2.35 × 10⁻⁸ N/(rad/s)² |
| Torque coefficient k_m | 7.24 × 10⁻¹⁰ N·m/(rad/s)² |
| Motor time constant τ | 25 ms |
| Arm length | 46 mm (35.4 mm to rotor center, X-config) |
| Max rotor speed | 2315 rad/s |
| Sim frequency | 100 Hz |
| Episode length | 10 s (1000 steps) |

---

## Papers

- **SimpleFlight** — Chen et al., RAL 2025. arXiv:2412.11764. Source of M1 architecture, hyperparameters, and eval methodology.
- **RMA** — Kumar et al., RSS 2021. arXiv:2107.04034. Two-phase privileged training used in M2.
- **MAVEN** — arXiv:2603.10714. Domain randomization ranges for rotor faults and mass variation.
- **GRaD-Nav++** — Chen et al., RAL 2025. arXiv:2506.14009. Language-commanded flight; target for M5.
- **PureJaxRL** — Lu et al., 2022. Fully-JIT-compiled PPO training pattern used throughout.
