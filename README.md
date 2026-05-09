# Agile quadrotor RL on consumer hardware

I'm a CS student on a break from school. I've been working through the literature on agile quadrotor reinforcement learning — SimpleFlight, RMA, MAVEN, GRaD-Nav++ — and building toward each one on a laptop with a 4070. The end goal is something close to GRaD-Nav++: a drone that you can command in natural language, that maps its environment in 3D, and that can recover from motor failures mid-flight. Five milestones.

This is a personal project. It's also a dream I'm working toward.

Built on MuJoCo MJX + JAX. Codenamed *Iron Man Drone*.

---

## Milestones

| # | Goal | Status |
|---|---|---|
| M1 | SimpleFlight recipe on MuJoCo MJX — figure-eight tracking | **DONE** |
| M2 | Fault tolerance via RMA two-phase training + MAVEN-style DR | **DONE** |
| M3 | Visual obstacle avoidance | Pending M2 |
| M4 | 3DGS-SLAM mapping + landmark return | Pending M3 |
| M5 | GRaD-Nav++ — natural language flight commands | Pending M4 |

---

## M1 Results

SimpleFlight (Chen et al., RAL 2025) on MuJoCo MJX with the same Crazyflie 2.1 parameters and PPO hyperparameters from their Table VI. All evals use the corrected T/4 phase offset (see [Eval methodology](#eval-methodology)).

| Trajectory | Ours (M1.3) | Paper (SimpleFlight) | Pass |
|---|---|---|---|
| figure_eight_slow (T=15s) | **0.020 m** | 0.016 m | ✓ |
| figure_eight_normal (T=5.5s) | **0.040 m** | 0.028 m | ✓ |
| figure_eight_fast (T=3.5s) | **0.094 m** | 0.051 m | ✓ |
| pentagram_slow | 0.058 m | 0.024 m | — |
| pentagram_fast | 0.068 m | 0.045 m | — |
| polynomial (random, 3-seed) | 0.065 m | 0.032 m | — |
| zigzag (random, 3-seed) | 0.043 m | 0.052 m | — |

Thresholds from SimpleFlight Table III: figure_eight_slow ≤ 0.050 m, figure_eight_normal ≤ 0.056 m, figure_eight_fast ≤ 0.150 m. All pass. Pentagram and random-trajectory numbers are OOD diagnostics, not gated.

Numbers are from `eval_suite.py` (GPU MJX lax.scan, seeds [42, 99, 7], 2026-05-09). Previous numbers from `eval_m1_full.py` (CPU mujoco) were ~0.003 m lower for figure-eight; see `notes/M1_3_results.md` for reconciliation.

The gap on figure_eight_fast and pentagram is real — the policy lacks rotational inertia compensation for aggressive trajectories. This is an acceptable M1 limitation; M2's task is fault tolerance, not closing that gap.

---

## M2 Results

M2 extends M1.3 with RMA two-phase training (Kumar et al., RSS 2021):
- **Phase 1:** PPO with privileged access to physical state e_t = [η₁–η₄, m_scale, F_x, F_y, F_z]. Actor input extended to 50-dim. 15k epochs, ~3.75 hours.
- **Phase 2:** Causal encoder ϕ: history → ê_t trained supervised on 20k frozen-actor rollouts. Deployed in place of ground-truth e_t. Training: 2000 epochs Adam lr=5e-4, ~1 minute.

**Domain randomization:** single-rotor η ∈ [0.5, 1.0] (fault_prob=0.70), mass ±20%, k_f ±30%.

All numbers use the corrected T/4 methodology and GPU MJX lax.scan backend (3 seeds).

### Phase 1 (privileged e_t) vs Phase 2 (encoder ê_t)

| Trajectory | M1.3 | M2 P1 Nom | M2 P2 Nom | M2 P2 Fault η=0.70 |
|---|---|---|---|---|
| figure_eight_slow (T=15s) | 0.017 m | 0.024 m | 0.024 m | 0.034 m |
| **figure_eight_normal (T=5.5s)** | **0.037 m** | **0.057 m** | **0.057 m** | **0.081 m** |
| figure_eight_fast (T=3.5s) | 0.090 m | 0.138 m | 0.133 m | 0.605 m† |
| pentagram_slow | 0.054 m | 0.067 m | 0.067 m | 0.083 m |
| pentagram_fast | — | 0.079 m | 0.079 m | 0.090 m |
| polynomial | 0.016 m | 0.088 m | 0.087 m | 0.140 m |
| zigzag | 0.027 m | 0.053 m | 0.053 m | 0.056 m |

† Crashes at 2/3 seeds — encoder startup instability under high agility + fault (see below).

**Phase 2 gate (figure_eight_normal):** nominal ≤ 0.065 m → **PASS** | fault ≤ 0.100 m → **PASS**

The encoder adds effectively zero overhead on nominal performance (Phase 2 within ±0.001 m of Phase 1 on all non-crash trajectories). The DR penalty vs M1.3 is real — 0.020 m on figure_eight_normal — and is a feature: the policy is tolerating single-rotor faults at p=0.70 while maintaining nominal performance within spec.

**Offline encoder:** best val MSE 0.0156 (η-channel mean 0.011, m_scale 0.076 — mass is harder to predict but doesn't visibly hurt closed-loop performance). 

**Encoder startup caveat:** The ring buffer initializes to zeros at episode start. During the first H=50 steps the encoder sees a zero-padded history and produces unreliable ê_t estimates. On figure_eight_normal the policy recovers quickly; on figure_eight_fast + fault the reduced thrust margin amplifies the startup perturbation to crashes. This is documented in `lessons.md` (L5) with fix options and will be addressed before M3.

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

**L1 — I paused a training run and lost two weeks of progress.** Restoring params while reinitializing Adam momentum caused a jump from 0.085 m to 0.215 m MED immediately post-resume, followed by 11,000 epochs of unrecoverable crash-at-step-8 behavior. The policy entered a feedback loop where short rollouts (crash at step 8) provided gradient signal only for the initial observation, reinforcing the crash. Run uninterrupted; if forced to resume, restore full optimizer state including Adam μ, ν, and step count.

**L2 — I spent three weeks confused about why a "working" policy kept failing on figure-eight.** Training on 2048 random trajectories allows high mean reward even when the held-out figure-eight eval fails completely. A policy crashing on every figure-eight episode can maintain training reward of 1.33 because randomly-initialized polynomial episodes provide positive signal. Use held-out MED as the primary convergence signal.

**L3 — The polynomial generator was wrong for a month before I noticed. It produced straight-line segments with a smooth speed profile — κ=0 everywhere. The figure-eight apex needs κ=4.8 m⁻¹. The drone wasn't undertrained; it was a perfect student of the wrong curriculum.** With 0% curvature coverage in training, the policy plateaued at 0.105 m on figure-eight. Replacing with C2-continuous piecewise quintic polynomials (random nonzero interior velocities) immediately dropped MED to 0.085 m in 2000 epochs.

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
  policy/
    encoder.py                Phase 2 causal encoder ϕ (AdaptationEncoder, 2300→256→128→8→tanh)
  evaluation/
    eval_suite.py             Unified eval module — T/4-corrected, crash-only, GPU MJX lax.scan
notes/
  M1_hypothesis.md            Required gating artifact before any M1 training
  M1_3_eval_methodology.md    T/4 phase offset discovery and justification
  M1_3_results.md             M1.3 canonical eval table (eval_suite.py, GPU MJX, 2026-05-09)
  M2_spec.md                  M2 design spec (RMA architecture, DR ranges, success criteria)
  M2_results.md               M2 final results — Phase 1 + Phase 2 side-by-side
  M2_phase2_hypothesis.md     Phase 2 hypothesis + actuals comparison
  lessons.md                  L1–L7: failure post-mortems and key findings
scripts/
  train_m1.py                 M1 training entry point
  train_m2.py                 M2 Phase 1 training entry point
  eval_m1_suite.py            M1 canonical eval via eval_suite.py (GPU MJX)
  eval_m1_full.py             Legacy M1 eval (CPU mujoco; superseded by eval_m1_suite.py)
  eval_m2_full.py             M2 Phase 1 full eval (T/4-corrected)
  collect_phase2_data.py      Phase 2 data collection (frozen actor rollouts)
  train_phase2_encoder.py     Phase 2 encoder supervised training
  eval_m2_phase2.py           Phase 2 closed-loop eval (encoder deployed)
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

---

## Contact

Olivier Couthaud — olivier.couthaud@gmail.com — [@olivier_couth](https://x.com/olivier_couth)

If you work on agile drone RL, sim-to-real, 3DGS-SLAM, or vision-language robotics and want to compare notes, I'd like to hear from you.
