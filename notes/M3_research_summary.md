# M3 Research Summary — Visual Obstacle Avoidance

**Date:** 2026-05-10
**Purpose:** Pre-spec extraction from three primary references on vision-conditioned agile flight. Every number in `M3_spec.md` must trace back to a row in this document or a conscious deviation.

**References:**
1. **Loquercio et al. 2021** — "Learning High-Speed Flight in the Wild," Science Robotics 6(59):eabg5810. [Paper](https://www.science.org/doi/10.1126/scirobotics.abg5810), [arXiv 2110.05113](https://arxiv.org/abs/2110.05113), [code: uzh-rpg/agile_autonomy](https://github.com/uzh-rpg/agile_autonomy).
2. **Kaufmann et al. 2023** — "Champion-level drone racing using deep reinforcement learning" (Swift), Nature 620:982–987. [Paper](https://www.nature.com/articles/s41586-023-06419-4), [PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/). No public training code.
3. **Chen et al. 2025** — GRaD-Nav ([arXiv 2503.03984](https://arxiv.org/abs/2503.03984)) and GRaD-Nav++ ([arXiv 2506.14009](https://arxiv.org/abs/2506.14009)). [Code: Qianzhong-Chen/grad_nav](https://github.com/Qianzhong-Chen/grad_nav).

**Sourcing caveat:** arXiv and Science Robotics PDFs returned HTTP 403 from research sandbox. Loquercio numerics are reconstructed from the `agile_autonomy` repo configs (verified line-by-line). Swift numerics from PMC mirror + Nature page. GRaD-Nav numerics from author project page + alphaXiv/ResearchGate snippets — several fine-grained details are flagged as "not specified in available sources."

---

## Part 1 — Observation architectures

### 1.1 Loquercio (depth, raw)

| Field | Value | Source |
|---|---|---|
| Modality | Single forward-facing depth image | `train_settings.yaml: use_depth: True, use_rgb: False` |
| Raw resolution | 640×480, HFOV 91°, stereo baseline 0.1 m | `flightmare.yaml` |
| Network input | **224×224** (MobileNet-native) | `train_settings.yaml: img_width/img_height: 224` |
| Update rate | **15 Hz** | `dagger_settings.yaml: network_frequency: 15.0` |
| Pre-processing | MobileNet `preprocess_input`; no min-pool, no embedding | `nets.py: PlaNet` |
| Encoder | **ImageNet-pretrained MobileNet V1** → Conv1D fusion | `nets.py` |
| State branch | Attitude + body rates + linear velocity, body frame | `train_settings.yaml: inputs:` block |
| Goal in obs | Single relative reference point (next state on unobstructed ref) in body frame | `train_settings.yaml` |
| Temporal stack | **`seq_len = 1`** — reactive, no history | `train_settings.yaml` |
| Output head | **M=3 candidate trajectories × 10 waypoints × 3 dims** + per-mode confidence (31 outputs) | `nets.py: out_dim = state_dim * out_seq_len + 1` |
| Action horizon | 1 second (10 steps × 0.1 s) | `traj_dt: 0.1, out_seq_len: 10` |

### 1.2 Swift (low-D state, no pixels)

Swift's policy **does not consume images.** Upstream perception (VIO + CNN gate-corner detector + EKF fusion) produces a low-D state.

| Field | Value | Source |
|---|---|---|
| Modality | **Low-D state vector, not pixels** | Methods §"Observation and action space" |
| Position | 3 | |
| Velocity | 3 | |
| Attitude | **Rotation matrix flattened (9)** — matches our M1/M2 convention | |
| Next gate | **4 corner positions × 3 = 12** in body frame | |
| Previous action `u_{t-1}` | 4 | **Note:** Swift includes it; our project rule excludes it from actor. |
| Total obs dim | **~31** | |
| Policy | **2-layer MLP, 128 hidden, LeakyReLU(0.2)** | Extended Data Table 2 |
| Value net | Same shape | |
| Body rates `ω` | **Not in actor obs** | Matches our M1/M2 |

### 1.3 GRaD-Nav / GRaD-Nav++ (RGB, not depth)

| Field | Value | Source |
|---|---|---|
| Modality v1 | **RGB**, not depth | Project page + abstract |
| Modality v++ | RGB + natural-language instruction (frozen CLIP for both) | |
| Resolution / FOV / rate | Not specified in available snippets | |
| Encoder v1 | **Pretrained SqueezeNet** (~1.2 M params) | |
| Encoder v++ | Frozen CLIP image+text encoder | |
| Goal in obs | Next waypoint vector | |
| Context encoder | **CENet-style GRU/recurrent** over (state, action, vision) history — architecturally equivalent to our M2 RMA encoder | |
| Action head v1 | MLP | |
| Action head v++ | Mixture-of-Experts | |
| Action rate | 30 Hz onboard | Project page |

### 1.4 Convergent patterns across the three

- **Body-frame goal, not world-frame.** All three encode the goal as a relative target in body frame (next gate corners, next reference point, next waypoint). Matches our existing `e^W` convention (30-dim relative positions to next 10 reference points).
- **Rotation matrix, not quaternion** for attitude in the obs. Swift agrees with our M1 rule explicitly.
- **Body rates `ω` not in the actor.** Loquercio and Swift exclude. Matches our M1 rule.
- **Reactive, no temporal stack on the perception side.** Loquercio runs `seq_len=1`. Swift has no image input at all. GRaD-Nav uses a context encoder over short history, but the visual feature itself is reactive.
- **Perception update rate &lt; control rate.** Loquercio: 15 Hz network, kHz controller. Swift: ~100 Hz policy with kHz inner loop. GRaD-Nav: 30 Hz onboard. Our M1/M2 runs the policy at 100 Hz; for M3 the depth-feature update rate is a design lever (see §5).

### 1.5 Observation density spectrum

The three papers span a wide range of perception bandwidth:

| Approach | Bits of perception per step (order of magnitude) |
|---|---|
| Swift (4 gate corners × 3 floats) | ~10² — extremely thin, task-specific summary |
| Loquercio (224×224 depth) | ~10⁵ — full image |
| GRaD-Nav (RGB through SqueezeNet) | ~10⁵ input, ~10² latent after backbone |

**Implication for M3:** the design lever is *how much we summarize depth before feeding the actor*. Swift's result — that a 12-number summary is enough for champion-level reactive flight — is the strongest single piece of evidence that we should consider min-pooled or binned depth, not raw depth, as the actor input. See §6.

---

## Part 2 — Methods

### 2.1 Loquercio — privileged-expert DAgger

- **Imitation learning, NOT RL.** A sampling-based motion planner with full obstacle-geometry access generates labels; a student depth-CNN policy learns to imitate.
- **Expert:** Metropolis-Hastings sampling, cost = progress along reference + clearance from obstacles. Returns top-K trajectories; K=3 become labels.
- **Loss:** `MixtureSpaceLoss` (winner-takes-all regression on M=3 modes + confidence classification) + `TrajectoryCostLoss` (re-evaluates predicted trajectories against the GT point cloud, penalizing collisions even on modes the expert never showed).
- **Training procedure:** DAgger. `max_rollouts: 100`, `train_every_n_rollouts: 10`, `increase_net_usage_every_n_rollouts: 10`.
- **Optimizer:** Adam, lr=1e-3 cosine-decayed, batch=8, max 150 epochs.
- **Simulator:** Flightmare (Unity-based, with simulated SGM stereo to produce noisy depth — not GT depth).

### 2.2 Swift — PPO on low-D state

- **PPO, on-policy, model-free.** Adam lr=3e-4, 100 parallel agents, episodes of 1500 steps.
- **Total experience:** ~1e8 env steps, ~50 min on i9-12900K + RTX 3090.
- **Reward (sum per step):**
  1. **Progress** — change in distance toward next gate centre along racing line.
  2. **Perception** — penalty on angle between camera axis and direction to next gate (keeps gate detectable).
  3. **Smoothness** — quadratic on body-rate command and its time derivative.
  4. Sparse gate-pass bonus; crash penalty + termination.
- **Action space:** **CTBR (collective thrust + body rates), 4-dim continuous.** Same as our project.
- **Simulator:** custom rigid-body sim on the Agilicious dynamics stack (BEM-based aero). Flightmare is the renderer for evaluation, not for training (Swift training never renders — its observation is state, not pixels).

### 2.3 GRaD-Nav — differentiable RL through renderer + dynamics

- **SHAC-style differentiable RL** (Short-Horizon Actor-Critic, Xu et al. 2022). Episodes sliced into short sub-windows; gradients flow through dynamics + renderer + encoder.
- **Renderer:** 3D Gaussian Splatting (gsplat). Differentiable rasterization w.r.t. camera pose.
- **Physics:** Custom PyTorch differentiable quadrotor sim, not Genesis.
- **Reward:** waypoint progress + collision penalty + smoothness. Intentionally reused unchanged across 8 training tasks.
- **Why this matters for us:** We **cannot reproduce this approach** — PPO is not differentiable through the env, JAX doesn't bridge to a Torch 3DGS renderer, and Genesis has no differentiable rendering. The GRaD-Nav method is out of scope for M3. The *observation architecture* (small CNN over visual obs + waypoint vector + context encoder → MLP) is PPO-compatible.

---

## Part 3 — Training environment generation

### 3.1 Loquercio — procedural forests

| Parameter | Value | Source |
|---|---|---|
| Tree placement | Regular grid, **4 m average spacing** | `flightmare.yaml: avg_tree_spacing: 4.0` |
| Jitter | **5 m uniform perturbation per cell** | `rand_width: 5.0` |
| Spawn bounding box | **235 × 235 × 15 m** | `bounding_box: [235, 235, 15]` |
| Generic obstacles | Cylinders/cubes, scale ∈ [0.5, 5.0] m per axis, random orientation | `spawn_objects: true` block |
| Reference trajectory | **Straight 40 m line** | `length_straight: 40.0` |
| Training velocity (default) | **7 m/s** | `maneuver_velocity: 7` |
| Deployed velocity range | 1–10 m/s | Released checkpoint description |
| Curriculum | None explicit; DAgger's expert-fallback radius (10 m) acts as a soft curriculum | `fallback_radius_expert: 10` |
| Point cloud resolution | 0.2 m | `pointcloud_resolution: 0.2` |

### 3.2 GRaD-Nav — real-scene 3DGS reconstructions

- 3DGS scenes captured by hand/phone scans of real indoor environments.
- **Multi-task:** v++ trains 8 scenes, holds out 4.
- Obstacles: gates + distractor objects placed inside captured scenes.
- Reference trajectory: hand-authored per task.
- No documented speed or density curriculum.

### 3.3 Convergent guidance for M3

- **Procedural clutter is enough for training** (Loquercio). Real-scene capture (GRaD-Nav) is for demo, not for the bulk of training.
- **Tree-as-cylinder approximation works.** Don't model branches, leaves, bark. Loquercio's deployed real-world environments include dense forests, snow, ruined buildings — generalization came from the procedural diversity, not from photorealistic training scenes.
- **Reference trajectory can be simple** (straight line for Loquercio, hand-authored for GRaD-Nav). Trajectory complexity is decoupled from obstacle complexity.
- **No explicit speed curriculum in Loquercio.** The fixed 7 m/s training velocity generalized to 1–10 m/s deployment. Argues against expensive speed scheduling for M3.

---

## Part 4 — Reward shaping (RL approaches only)

### 4.1 Swift — three terms

1. **Progress reward** — Δ(distance to next gate centre along racing line). Continuous, dense.
2. **Perception reward** — penalty proportional to (angle between camera axis, direction to next gate). Keeps the gate in view so the detector keeps working.
3. **Smoothness penalty** — quadratic on body-rate command and on its time derivative.
4. **Sparse gate-pass bonus** on success.
5. **Crash penalty + episode termination** on collision or out-of-bounds.

**Exact numerical coefficients:** in Extended Data Table 1, not retrievable from accessible mirrors. Read directly from the Nature PDF before transcribing.

### 4.2 GRaD-Nav — minimalism

- Waypoint progress + collision penalty + smoothness + attitude/action-rate penalty.
- "Same reward across tasks" stated as design discipline.
- Soft vs hard collision: not specified in available sources.

### 4.3 Convergent reward shape for M3

| Term | Function | Status |
|---|---|---|
| Trajectory tracking | Match M1's existing tracking reward (negative L2 on `e^W[0]`) | Already in our reward |
| Collision avoidance — soft | Penalty inside a safety margin (e.g., `−w_clear × max(0, d_safe − d_min)²`) | New |
| Collision — hard | Termination + large negative reward on actual contact | New |
| Smoothness | Already in our reward (action rate penalty) | Already present |
| Perception/visibility | **Skip for M3.** Swift's term keeps a specific gate detectable; M3 is reactive depth, no specific feature to look at. Reconsider for M5 if depth FOV becomes a limit. | — |

**Single-term rule for M3 reward:** add *only* the soft + hard collision terms. Do not add energy, yaw, attitude, or stability terms beyond what M1+M2 already had. Each new reward term doubles the failure-mode debugging surface.

---

## Part 5 — Evaluation protocols

### 5.1 Loquercio

- **Real-world envs:** Central-European forests, snow, derailed trains, partially-collapsed building. **31 real experiments.**
- **Maneuvers:** 40 m straight line, 6 m radius circle.
- **Speed × reliability:**
  - 3 m/s, 5 m/s: 0 crashes.
  - **7 m/s: 8/10 successful.**
  - 10 m/s: reduced reliability; widely cited as ~70% success in forest.
- **Sim-only:** speed × tree-density sweep with per-cell success rate. Compared to MPC+planner baseline; learned policy dominates at high speed.
- **Sim-to-real:** the team attributes the small gap to **depth (appearance-invariant) + SGM-noisy depth simulation + multi-modal expert labels.**

### 5.2 Swift

- **Track:** 75 m indoor course, 7 gates including split-S. Speeds up to ~22 m/s, accel ~5g.
- **Head-to-head vs 3 pilots** (DRL 2019 world champion + others): Swift won majority of races; fastest 3-lap time **17.465 s vs human best 17.956 s**.
- **No general obstacle eval** — Swift is racing-specific.

### 5.3 GRaD-Nav++

- **Sim:** 83% success on 8 trained tasks, 75% on 4 held-out.
- **Real:** 67% trained, 50% unseen.
- **Sim-to-real gap:** ~16–25 pp drop.

### 5.4 Convergent eval design for M3

- **Procedural clutter test set, held out from training seeds.** Mirror Loquercio's speed × density sweep.
- **Per-speed success rate**, not just mean tracking error. M3 has two failure modes (crash + tracking error), each needs its own metric.
- **Nominal-obstacle-free preservation test.** Run full M1 figure-eight eval; confirm no regression from M2-baseline MED. This is the hard requirement that obstacles-when-absent don't degrade tracking.
- **Speed regime:** train at 3 m/s (our cap), evaluate at 3 m/s + a 4 m/s and 5 m/s stretch. Loquercio at 7 m/s with no curriculum suggests our 3 m/s cap is conservative; we can add headroom in eval cheaply.

---

## Part 6 — What we'd actually steal for M3

The single most useful pattern across all three papers, and the most important architectural decision for M3:

### 6.1 Perception summary, not raw pixels (Swift's lesson)

Swift proves a champion-level vision policy can be a small MLP over a low-D state plus a **thin perception summary** (12 numbers for the next gate's corners). For M3 the analogous causal-sufficient summary is the local free-space frontier in the direction of motion.

**Concrete proposal for M3 obs (subject to design in `M3_spec.md`):**

- **Min-pooled depth panorama:** divide the front-facing FOV into K angular bins (K=16 or 32); per bin, the **nearest depth** clipped at some `d_max` (e.g., 5 m). Yields a 16-dim or 32-dim vector. Body-frame, reactive.
- Append to the existing M2 actor obs: `[e^W (30), v (3), R (9), z (8), depth_bins (16–32)]` → 66–82 dim.
- Critic gets the same plus `k` (1) and the privileged obstacle map summary (TBD).

This sits between Swift (12-dim) and Loquercio (50k-dim raw depth). Swift's evidence is that ~10²-dim works; Loquercio shows ~10⁵ also works but needs ImageNet pretraining and an imitation-learning workaround. Min-pooled is the cheap middle ground that keeps PPO viable on our budget.

### 6.2 Body-frame relative reference point (Loquercio)

Already in our M1 obs (`e^W`). No change needed.

### 6.3 Single-policy with shaped reward (GRaD-Nav, Loquercio)

Both papers fuse tracking and avoidance into a single policy with a single loss/reward. Matches our M3 design decision (single-policy reactive avoidance, no planner). Plan: M1's tracking reward + soft clearance + hard collision termination.

### 6.4 Procedural training, structured demo (Loquercio)

- Training: procedural cylinder forests (4 m grid, 5 m jitter) — adapt to Genesis.
- Demo: structured scene with hand-placed obstacles for "show what it does." No need to train on the demo scene.

### 6.5 What we explicitly do NOT steal

- **Loquercio's DAgger + privileged sampling-based teacher.** PPO with shaped reward is simpler and our existing M1+M2 stack is PPO. Adopt DAgger only if PPO fails.
- **GRaD-Nav's differentiable rendering / SHAC.** Out of scope (no diff renderer in Genesis on JAX). Defer to v3 research arc, not M3.
- **Swift's GP + KNN sim-to-real residuals.** Heavy (mocap data, two non-parametric models, fine-tune loop). Defer to M5+. When we get there, start with a single dynamics-residual MLP on (state, action) → acceleration error from ~1 minute of real data. Don't reach for GPs.
- **GRaD-Nav++ CLIP / MoE / language conditioning.** M3 is single-task; no multi-task signal yet.
- **Raw 224×224 depth tensors as actor input.** Min-pooled summary first. Upgrade to raw if min-pooled stalls.

---

## Part 7 — Real-world deployment notes (M5+ reference)

| Platform element | Loquercio | Swift | GRaD-Nav |
|---|---|---|---|
| Depth sensor | Intel RealSense **D435** active stereo | n/a (VIO + gate detector) | Intel RealSense **D435** RGB |
| Pose sensor | Intel RealSense **T265** VIO | Onboard VIO | Onboard VIO |
| Onboard compute | NVIDIA **Jetson TX2** | i9 not specified for onboard; offboard for training | NVIDIA **Jetson Orin Nano** |
| Flight controller | (not specified) | Agilicious | **Pixracer** (PX4) |
| Perception rate | 15 Hz | ~100 Hz policy | 30 Hz |
| Control rate | kHz inner loop | kHz inner loop | kHz inner loop |

**Convergent stack for our future M5+ hardware:** RealSense D435 (depth) + Orin Nano (more headroom than TX2 for newer policies) + Pixracer or similar PX4 board running the rate loop. **Not a Crazyflie** — D435 + Orin Nano won't fit a 32 g airframe. Hardware platform choice belongs in the M5 spec, not M3.

---

## Part 8 — Open questions for the M3 spec

These don't have clean answers from the literature; they need designer decisions in `M3_spec.md`:

1. **Depth resolution.** 64×64 (cheap) or 96×96 (Madrona-comfortable)? Loquercio used 224×224 with a pretrained backbone we won't have for depth. With min-pooled bins as the actor input, raw resolution matters less — pick the lowest that doesn't alias on small obstacles at our `d_max`.
2. **Min-pooled bin count K.** 16 (Swift-style thin) or 32 (Loquercio-style richer)? Both feasible.
3. **Depth update rate.** Match policy rate (100 Hz) or sub-sample like Loquercio (15 Hz)? Sub-sampling is cheaper and matches deployable sensor rates; 100 Hz update would burn render budget for marginal information gain.
4. **Soft clearance coefficient `w_clear`.** Tune to satisfy "entropy contribution still &lt; 10% of reward at init" rule from project lessons.
5. **Procedural obstacle parameters.** Loquercio's 4 m grid + 5 m jitter is for forest at 7 m/s. At our 3 m/s cap, we can tolerate denser (closer-spaced) obstacles. Likely halve the spacing.
6. **Fault tolerance preservation.** M2 shipped RMA with single-rotor faults + mass DR. Does M3 keep the M2 encoder `z` in obs, or rebuild from scratch for vision? See §F1 in `M3_spec.md`.
7. **Eval scene definition.** A held-out procedural seed range + a hand-built "demo scene" for video. Specify exact scene.
