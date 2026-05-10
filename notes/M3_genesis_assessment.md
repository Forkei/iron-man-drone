# M3 Genesis Assessment

**Date:** 2026-05-10
**Purpose:** Decide whether to migrate from MuJoCo MJX to Genesis for M3 (visual obstacle avoidance). M3 needs GPU-batched depth rendering across thousands of envs; MJX has no native depth path.
**Source caveat:** Some Genesis primary docs returned HTTP 403 during research. Items reconstructed from GitHub issues, release notes, and cached snippets are marked accordingly. Verify before committing the port.

---

## 1. Project maturity and activity

- **Repo:** [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis), Apache-2.0.
- **Latest release:** **v0.4.6**, 2026-04-11 ([PyPI: genesis-world](https://pypi.org/project/genesis-world/)).
- **Stars:** ~28.7k (latest crawl). Older snapshots ~14k. Active community.
- **Open issues:** ~104 — small relative to star count, triage is current.
- **Contributors:** "20+ research labs over two years"; exact number not verified.
- **Backend:** **Single backend.** Built on **GsTaichi** (forked Taichi, renamed "Quadrants" internally since v0.4.0). **No JAX backend.** Outputs are **PyTorch CUDA tensors.** GPU batched envs (`n_envs=...`) is the default path.
- **API stability:** Mid-stabilization. v0.4.x landed several silent breaking changes in 2026:
  - v0.4.0 (2026-02-18): Quadrants compiler migration; AMD ROCm experimental.
  - v0.4.4 (2026-03-29): `set_quat` semantics changed to relative; rigid material density default changed.
  - v0.3.12 (2026-01-17): Y-up/Z-up mesh import logic changed.
  - `rsl-rl` compat broke and was repinned to 2.2.4 ([#1035](https://github.com/Genesis-Embodied-AI/Genesis/issues/1035)).
  - Recurring "breaking change in main" thread ([#2285](https://github.com/Genesis-Embodied-AI/Genesis/issues/2285)).
- **Implication:** Pin to `genesis-world==0.4.6` (or equivalent). Budget breakage time on each minor bump.

---

## 2. Drone / quadrotor support

- **Built-in URDF:** `genesis/assets/urdf/drones/cf2x.urdf` (Crazyflie 2.X, X-config). Also CF2P and RACE.
- **`gs.morphs.Drone`** loads URDF, reads `kf`/`km` as XML attributes, exposes `set_propellers_rpm(rpm)`. Force/torque applied at propeller link via rigid solver.
- **Modeled dynamics (out of the box):**
  - Thrust = `kf × ω²` at each propeller link.
  - Torque = `km × ω²`.
  - **No motor lag** — RPM is set instantly. Our MJX env models first-order motor dynamics with `τ = 0.025 s` per the SimpleFlight recipe; this is **not portable** without manual implementation.
  - **No aero drag, no prop wash, no ground effect.**
- **Examples:** `examples/drone/` ships:
  - `hover_env.py` — RL env for hovering, **direct RPM action** (`(1 + a*0.8) * 14468.43 RPM`), not CTBR.
  - `hover_train.py` — `rsl-rl` PPO trainer (Torch-only).
  - `quadcopter_controller.py` — cascaded PID over RPM, not CTBR.
  - `fly.py`, `fly_route.py`, `interactive_drone.py` — replay / teleop demos.
- **CTBR controller: not provided.** We port our `control/ctbr_controller.py` to sit between policy output and `set_propellers_rpm`, including:
  1. Rate-PD on body rates (in Torch).
  2. X-config mixer (collective thrust + 3 body-rate moments → 4 desired thrusts).
  3. First-order motor lag on Torch state buffer (`τ = 0.025 s`).
  4. `ω = √(T / kf)` per rotor → `set_propellers_rpm`.

**Implication:** Genesis is a less complete out-of-the-box quadrotor sim than MJX-with-our-custom-env. We re-implement motor lag and CTBR; we lose the asymmetric site-force trick currently used in MJX.

---

## 3. Depth camera / rendering pipeline

**This is the only reason to consider the migration. Genesis is competitive here.**

- **Two render paths:**
  - `gs.renderers.Rasterizer()` and `gs.renderers.RayTracer()`: per-camera, OpenGL/PBR or vendored LuisaRender. Not designed for batched RL.
  - `gs.renderers.BatchRenderer(use_rasterizer=True|False)`: backed by **gs-madrona** (the [Madrona batch renderer](https://madrona-engine.github.io/renderer.html), SIGGRAPH Asia 2024). GPU-batched, purpose-built for embodied-AI training. Vulkan rasterizer or CUDA single-bounce ray tracer.
- **Outputs:** `camera.render(rgb=True, depth=True, segmentation=True, normals=True)` returns `(n_envs, H, W, C)` tensors. Depth and segmentation are first-class.
- **GPU-resident.** No PCI roundtrip on the batched path.
- **Throughput (Madrona paper, RTX 4090 / A100):**
  - &gt;100k FPS at 128×128 ray-traced simple scenes.
  - &gt;10k FPS on complex scenes.
  - **No published 4070-class benchmark.** Scaling 4090 numbers by ~0.55× is a rough estimate.
- **Known limitations / open bugs:**
  - Only basic materials, lighting, shadows. No PBR refraction, no transparent surfaces, no heightfields, no moving lights.
  - **[#1648](https://github.com/Genesis-Embodied-AI/Genesis/issues/1648)** — depth values from `BatchRenderer(use_rasterizer=True)` disagree with the standard rasterizer; cross-camera point-cloud fusion misaligns. **Action: validate depth values against ground truth before trusting.**
  - **[#1591](https://github.com/Genesis-Embodied-AI/Genesis/issues/1591)** — `BatchRenderer` install/Vulkan setup friction.
  - Resolution caps not documented. Optimize for 64×64 to 128×128 depth observations, not 480p.
- **Stone Tao independent benchmark** ([Substack](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)): Genesis FPS does **not** scale beyond ~8192 envs; headline FPS numbers use 1 substep where the project's own RL scripts use 2–4. Critique is about **rigid-body** sim throughput, not rendering. Reduces our throughput expectations from "10× MJX" to "comparable to MJX or slower at fair substep settings."

---

## 4. Community drone RL baselines on Genesis

The honest finding: **almost no public depth-camera drone RL on Genesis.**

- **MAVEN (arXiv:2603.10714)** — paper abstract names Genesis as the simulator, 4096 envs/task, converges in &lt;1 hour on Ryzen 9-9950. The "20 tasks × RTX 5090" detail in our M2 research summary was paraphrased and may overstate what the abstract actually says. **MAVEN code release: not verified.** Searches surfaced no public repo.
- **GRaD-Nav (arXiv:2503.03984) / GRaD-Nav++ (arXiv:2506.14009):** [Qianzhong-Chen/grad_nav](https://github.com/Qianzhong-Chen/grad_nav). **Does NOT use Genesis.** Uses a custom PyTorch differentiable sim with 3D Gaussian Splatting for rendering. Not a Genesis precedent.
- **GenesisEnvs ([RochelleNi/GenesisEnvs](https://github.com/RochelleNi/GenesisEnvs)):** community RL collection. No drone env present.
- **Aerial Gym ([NTNU](https://ntnu-arl.github.io/aerial_gym_simulator/), arXiv:2503.01471):** parallel drone sim with GPU ray-cast depth on **Isaac Gym, not Genesis**. The closest mature alternative if Genesis fails.
- **madrona_mjx ([shacklettbp/madrona_mjx](https://github.com/shacklettbp/madrona_mjx)):** the same Madrona renderer wired into MJX. **Keeps our JAX/Flax stack.** Less mature stitch, but the natural fallback that doesn't require a full sim migration.

**Implication:** We'd be the first public depth-camera drone RL on Genesis. No reference implementation to mimic. Higher project risk; ground-truth signal is correspondingly weaker.

---

## 5. MJX → Genesis migration cost

### Ports cleanly
- `envs/trajectories.py` — pure NumPy.
- `scripts/eval_*.py` — pure Python.
- PPO algorithm logic (rollout-update loop, GAE, clip).
- Reward function shape (Torch ↔ JAX op-by-op).

### Needs rewrite (significant)
- **`envs/quadrotor_env.py`** — Genesis is stateful and imperative. No `jax.lax.scan`, no `vmap`. Rewrite as a step/reset class wrapping `scene.step()`.
- **Observation extraction** — `xmat[drone_body_id]` and `qvel[:3]` (MJX) become Genesis Torch tensor accessors. Layout is `(n_envs, ...)` on CUDA.
- **CTBR controller** — port rate-PD + mixer + motor lag from MJX site forces to Torch tensor ops. Validate that motor lag dynamics match (`τ=0.025 s` discrete update at 100 Hz).
- **Domain randomization** — Genesis supports per-env mass/inertia/friction setters (v0.4.6 release notes call out "batching for `RigidLink.set_mass`"). API is there but **undocumented** ([#70](https://github.com/Genesis-Embodied-AI/Genesis/issues/70), [#107](https://github.com/Genesis-Embodied-AI/Genesis/issues/107)). Read source.

### The big one — Flax/JAX policy
Genesis returns Torch tensors; our actor/critic is Flax. Three options:

| Option | Cost | Risk |
|---|---|---|
| **A. DLPack bridge** every step (`torch.utils.dlpack.to_dlpack` → `jax.dlpack.from_dlpack`) | Low (zero-copy on CUDA) | Brittle on each JAX/Torch upgrade; debug determinism becomes harder |
| **B. Rewrite policy in PyTorch + rsl-rl** | High (re-validate the entire M1+M2 PPO recipe) | Loses the asymmetric actor/critic + separate Optax optimizers — exactly the architecture our project rules call load-bearing |
| **C. Genesis as a separate process; RPC tensor pipe** | Medium-high (operational pain) | Latency on rollouts; PPO throughput drops |

**Recommended:** Option A (DLPack bridge), validated in a 3-day spike before committing. Falling back to Option B means re-doing M1.3 and M2 in Torch — a project-month, not a port.

---

## 6. Known gotchas

- **Determinism: not guaranteed.** No public statement that `gs.init(seed=...)` gives bit-exact rollouts. GPU-parallel reductions are non-deterministic by default in Taichi/Quadrants. Plan: log as risk, not guarantee. Verify with a 100-step rollout pair under fixed seed; require &lt;1e-5 max abs difference.
- **Contact handling:** Genesis uses MuJoCo-style soft contact (depends on `mujoco>=3.2.5`). Defaults tuned for manipulation, not high-velocity wall flight. [#1139](https://github.com/Genesis-Embodied-AI/Genesis/issues/1139) reports friction setters being quirky. Plan: bump constraint stiffness, validate vs MJX on a flyby-near-cylinder test.
- **Small timestep:** `dt=0.01` (100 Hz, our target) works with `substeps=1` per Genesis defaults, but this is the configuration Stone Tao calls accuracy-compromised. Validate vs MJX integrator output for a 1-second free fall and a hover under disturbance before trusting downstream eval.
- **WSL2 + 4070:** **This is the highest install risk.** Multiple open issues:
  - [#187](https://github.com/Genesis-Embodied-AI/Genesis/issues/187) "no CUDA-capable device" on WSL2 + CUDA 11.8 (our exact platform).
  - [#352](https://github.com/Genesis-Embodied-AI/Genesis/issues/352), [#343](https://github.com/Genesis-Embodied-AI/Genesis/issues/343), [#370](https://github.com/Genesis-Embodied-AI/Genesis/issues/370) WSL2 install errors.
  - [#486](https://github.com/Genesis-Embodied-AI/Genesis/issues/486) LuisaRender build fails on WSL.
  - [#2039](https://github.com/Genesis-Embodied-AI/Genesis/issues/2039) `libGLU.so.1` missing on WSL2.
  - The Madrona/Vulkan path on WSL2 is the riskiest component — and it's exactly the part needed for depth rendering.
  - **Budget 2–3 days for environment setup before any RL code.**
- **Memory:** Madrona depth at 64×64×4 bytes × 4096 envs ≈ 64 MB just for depth. Fine on 4070 (12 GB). RGB + segmentation + multiple cameras compound. Profile early.

---

## 7. Compute and throughput

- **No public 4070 benchmark.** Genesis team headlines are RTX 4090. Madrona paper benchmarks are 4090 / A100. Conservative: scale 4090 by ~0.55× (4070 has ~7,168 CUDA cores, ~half the memory bandwidth).
- **MAVEN claim:** 4096 envs/task, &lt;1 hour to converge on Ryzen 9-9950 workstation (abstract). Specific GPU not in abstract; verify from paper body.
- **Stone Tao's correction:** in fair (substep-matched) comparisons, Genesis rigid-body sim is **slower than MJX**, not faster. The win is **batched depth rendering**, where MJX has no native equivalent.

**Realistic expectation:** On a 4070, M3 training throughput will be **MJX-comparable or slower** for the physics step, **plus** the additional cost of depth rendering at 64×64. Estimate 15–30k env-steps/sec with rendering active, down from M1.3's 65k env-steps/sec without rendering. Wall-clock per 15k epochs: 8–16 hours, vs M1.3's 4–5 hours. Live with it; depth was always going to cost.

---

## Bottom line

A 2-week M2.5 Genesis port is **risky**. Realistic estimate: 2–4 weeks, with a ~30% chance of slipping past 4 weeks into a project-month if WSL2/Vulkan friction compounds.

### Strongest evidence for Genesis
- The Madrona-backed `BatchRenderer` is a real GPU-batched depth path. MJX has no native equivalent. For visual obstacle avoidance training across 1024+ envs, this is currently one of two credible options.
- The `examples/drone/hover_env.py` exists and trains end-to-end — Genesis can do quadrotor RL.

### Strongest evidence against Genesis (or for delay)
- **PyTorch-tensor native, no JAX backend.** Our entire M1+M2 stack (Flax actor/critic, Optax separate optimizers, asymmetric architecture, RMA encoder) was built on JAX. A clean migration requires either a brittle DLPack bridge or a full rewrite to Torch+rsl-rl.
- **No CTBR or motor lag out of the box.** We re-implement the controller stack from scratch in Torch.
- **WSL2 + Vulkan/Madrona** has multiple open issues, on our exact platform.
- **API churn:** v0.4.x still landing silent breaking changes every 2–4 weeks.
- **No public Genesis drone RL precedent** beyond the bundled hover env. MAVEN code is not released. We'd be debugging without a reference.

### Recommended path

1. **3-day spike before commit.** Install Genesis on WSL2, run `hover_train.py`, render one batched depth tensor at 64×64 from 1024 envs, DLPack-bridge it into a dummy Flax forward pass, verify a 100-step rollout is deterministic under fixed seed. **If any of these fail, abort the migration.**
2. **If the spike passes:** proceed to M2.5 port (see `notes/M2_5_genesis_port_spec.md`).
3. **If the spike fails:** fall back to **madrona_mjx** (keeps the JAX stack, adds depth) or Aerial Gym (Isaac Gym, requires Torch policy rewrite but mature drone+depth).

### Spike acceptance gates

| Check | Pass criterion |
|---|---|
| Install on WSL2 Ubuntu 22.04 + CUDA 12 + 4070 | `python -c "import genesis as gs; gs.init(); print(gs.device)"` returns CUDA in &lt; 1 hour of setup |
| Run `examples/drone/hover_train.py` | First 100 PPO iterations complete, reward trends up |
| Batched depth render at 64×64, n_envs=1024 | `BatchRenderer` returns finite depth tensor; no NaNs; values within [0, max_depth] |
| DLPack Torch → JAX bridge | A Flax module forward pass on the depth tensor produces finite output, zero-copy on CUDA |
| Determinism | Two rollouts under same seed differ by &lt;1e-5 on final state |
| Throughput | Sim+render &gt; 10k env-steps/sec on the 4070 |

**Time-box: 3 working days.** Past day 3 with gates unmet → fall back, do not extend.
