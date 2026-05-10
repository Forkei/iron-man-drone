# M3 Sim Choice Decision

**Date:** 2026-05-10
**Purpose:** Peer-comparison of simulator candidates for the M3 visual obstacle avoidance milestone. The prior plan (`M3_genesis_assessment.md`, `M2_5_genesis_port_spec.md`) treated Genesis as the goal with madrona_mjx as a fallback. This document reframes: evaluate candidates as peers, recommend the one with better risk-reward for our specific setup.
**Status:** Decision deliverable. Recommendation actionable.

---

## Reframing

The original Genesis vs madrona_mjx comparison was based on an outdated map of the landscape. Two things have changed since the project's previous research pass:

1. **madrona_mjx was deprecated on 2026-02-18.** Its README explicitly points users to MJWarp as the successor: "High-throughput batch rendering is now natively supported within the MuJoCo ecosystem via MJWarp... We recommend all users migrate to the official MJX renderer for better long-term support, performance, and feature parity."
2. **MJWarp is the actively-maintained successor**, jointly developed by Google DeepMind and NVIDIA, integrated into MuJoCo via the `mujoco.mjx.warp` namespace, and is the default vision-environment backend in MuJoCo Playground v0.2.0 (March 2026).

So the real peer evaluation is:

- **Candidate A: Genesis** (separate sim, PyTorch-tensor native, Madrona renderer wrapped in `BatchRenderer`).
- **Candidate B: MJX + MJWarp** (our existing sim, JAX-native, NVIDIA-Warp-based ray-traced batch renderer, official Google DeepMind path).
- **Candidate C (longshot): Aerial Gym** (Isaac Gym, PyTorch, ships pre-built vision-based drone obstacle-avoidance examples).

This document compares all three honestly. The recommendation is at the bottom — read the reasoning first.

---

## Hard data

### Sandbox-collected ground truth

- `pip index versions genesis-world` → 0.4.6 (April 2026). On PyPI.
- `pip index versions madrona-mjx` → NOT on PyPI. Build-from-source only. (Independently confirms low-maturity packaging.)
- `pip index versions mujoco-mjx` → 3.8.0 (April 2026). On PyPI.
- madrona_mjx GitHub: last commit `1505699`, 2026-02-18, message "Add deprecation notice for Madrona MJX." 160 stars, 6 open issues.
- mujoco_warp GitHub: v3.8.0.3 (2026-05-08), 1.2k stars, 30 open issues, 1894 commits, jointly maintained by Google DeepMind + NVIDIA.
- mujoco_playground GitHub: v0.2.0 (2026-03-16), 1.9k stars, 62 open issues. v0.2.0 release notes: "Vision notebooks now utilize MuJoCo Warp" and "MuJoCo Warp became the default implementation across all environments."
- MJWarp render example at `mjx/mujoco/mjx/warp/visualize_render.py` confirmed: render call is `jax.jit`'d and `jax.vmap`'d, returns JAX arrays of shape `(n_envs, H, W, C)`. Supports RGB + depth + segmentation natively.
- Aerial Gym v2.0.0 (April 2025, latest release), 708 stars, 11 open issues. Has vision-based navigation examples. Built on Isaac Gym (not Isaac Lab).

### What the sandbox could NOT validate

- **No GPU.** Cannot run any of the three candidates here. All install-experience claims below are based on documented prerequisites, not hands-on attempts.
- **No WSL2.** Sandbox is plain Linux 6.18.5, no Windows host.
- The actual install timing on the target platform (WSL2 + 4070 mobile) requires the user to run the 3-day spike.

---

## Candidate A — Genesis

**Strengths:**
- Has a bundled drone example (`examples/drone/hover_env.py`) and a `gs.morphs.Drone` morph with Crazyflie URDFs.
- Active community, ~28k stars, batched DR APIs landing in recent releases.
- Pipelines RGB + depth + segmentation through the Madrona-backed `BatchRenderer`.

**Costs:**
- **PyTorch-tensor native. No JAX backend.** Outputs Torch CUDA tensors. Our entire M1/M2 stack (Flax actor/critic, Optax separate optimizers, PureJaxRL-style `lax.scan` training loop) cannot consume them without either a DLPack bridge (brittle, debug-determinism risk) or a full rewrite to Torch + rsl-rl (kills the asymmetric Flax architecture).
- **No CTBR controller, no motor lag.** The bundled controller is cascaded PID over raw RPM. Our `motor_tau = 0.025 s` first-order rotor dynamics must be reimplemented as a Torch tensor buffer.
- **No fault-tolerance precedent.** Genesis has no published RMA / privileged-encoder examples for drones.
- **WSL2 + Vulkan friction** — multiple open issues (#187, #486, #2039, #1591) on our exact platform. The Madrona Vulkan path is the highest-risk install component, and it's exactly the part we need for depth rendering.
- **API churn:** v0.4.x landed silent breaking changes every 2-4 weeks. `set_quat` semantics changed in v0.4.4. We'd have to pin and budget breakage time on each bump.
- **Open depth bug** ([#1648](https://github.com/Genesis-Embodied-AI/Genesis/issues/1648)): depth values from `BatchRenderer(use_rasterizer=True)` disagree with the standard rasterizer. Cross-camera fusion misaligns.
- **Migration cost:** estimated 1000+ LOC — full env rewrite, Torch policy bridge, controller port, DR reimplementation.

**Verdict:** Genuinely capable for visual drone RL, but for *us specifically* the migration cost is large and the architectural alignment is poor.

---

## Candidate B — MJX + MJWarp

**Strengths:**
- **JAX-native.** Render call is `jax.jit`-compatible, `jax.vmap`-friendly, `lax.scan`-threadable via a `render_token`. Returns JAX arrays of shape `(n_envs, H, W, C)`. **No DLPack bridge.**
- **CTBR controller, motor lag, asymmetric Flax actor/critic, separate Optax optimizers, RMA encoder `μ`, adaptation encoder `ϕ` — all preserved unchanged.** MJWarp adds rendering alongside MJX; it does not touch the physics layer where these live.
- **M1.3 and M2 numbers are preserved** as long as we keep physics on XLA (the existing backend). MJWarp also offers a physics backend, but that's optional and orthogonal to its renderer — we can adopt the renderer without switching physics.
- **CUDA-only, no Vulkan.** NVIDIA Warp uses CUDA kernels. The WSL2 Vulkan friction that plagues Madrona/Genesis is irrelevant. Our WSL2 + CUDA 12 + 4070 mobile platform is in the documented happy path.
- **Official Google DeepMind direction.** MJWarp was explicitly designated as the post-madrona_mjx successor. The render API is in the main `mujoco` repo. v0.2.0 of Playground (March 2026) makes it default.
- **Natively supports RGB + depth + segmentation.** Confirmed in `mjx/mujoco/mjx/warp/io.py` (`ctx.depth_data_shape`, `ctx.seg_data_shape`, `ctx.rgb_data_shape`).
- **Migration cost: ~50–150 LOC**, additive. Add a `<camera>` to `crazyflie.xml`, build a `RenderContext` in env `__init__`, thread it through `step`/`reset`, call `bvh.refit_bvh` + `render.render` + `get_depth`, append depth to obs dict, add a small CNN encoder (or min-pooled summary per `M3_spec.md`) before the existing actor MLP.

**Costs:**
- **No bundled drone example.** Playground has no aerial env. We bring our own MJCF (already have it: `crazyflie.xml`).
- **No public drone-with-depth RL precedent.** Same as Genesis on this dimension — both are first-mover for this exact task.
- **Newer than madrona_mjx**, in alpha/early-stable phase. Bugs are being found and fixed (issue #299, #317). But the upstream commit cadence is weekly+ and the issues are open with maintainer attention.
- **Vision Colab tutorial broken** (Playground issue #268, open Jan 2026) — minor friction for self-teaching, not a blocker.
- **One nuance: BVH refit between physics step and render step.** When bodies move (every step, for our drone), the BVH must be refit before the render call. This is an explicit one-line call in the example. No actual cost beyond awareness.
- **Backend-switch trap:** If we accidentally adopt the MJWarp *physics* backend along with the renderer (e.g., via Playground's `--impl warp` flag), our M1.3 / M2 numbers could shift due to different float reductions and solver ordering on GPU. **Mitigation: explicitly keep physics on XLA**, only switch the renderer. See "Implementation discipline" below.

**Verdict:** This is the obvious choice. It preserves everything we have, costs ~50–150 LOC, and removes the WSL2 Vulkan + Torch bridge + JAX-version-pin tax that comes with Genesis and madrona_mjx respectively.

---

## Candidate C — Aerial Gym (Isaac Gym)

**Strengths:**
- **Ships pre-built vision-based drone obstacle-avoidance examples.** This is the *most ready-out-of-the-box* of the three.
- Custom rendering framework based on NVIDIA Warp (same kernel family as MJWarp), supports depth + segmentation across parallel envs.
- v2.0.0 (April 2025), 708 stars — modest but real community.

**Costs:**
- **Built on Isaac Gym, which itself is deprecated** by NVIDIA, replaced by Isaac Lab. Aerial Gym docs note "Support for Isaac Lab and Isaac Sim is currently under development" — i.e., not yet. Adopting Aerial Gym today means adopting two layers of upcoming-deprecation risk: Isaac Gym → Isaac Lab.
- **PyTorch only.** Full rewrite of our policy stack to Torch — same cost dimension as Genesis.
- **Isaac Gym install on WSL2 + 4070 mobile:** non-trivial, requires NVIDIA OmniIsaac packaging, multi-GB downloads, signed Isaac Sim binaries.
- **The team's existing M1.3 + M2 numbers are invalidated** by sim swap.

**Verdict:** Has the right out-of-the-box example, but the deprecation overhang on Isaac Gym plus the Torch rewrite cost make this a worse trade than MJWarp.

---

## Side-by-side comparison

| Axis | Genesis | MJX + MJWarp | Aerial Gym |
|---|---|---|---|
| **Active maintenance** | Yes (community + labs) | **Yes — Google DeepMind + NVIDIA, official** | Yes (NTNU lab, slower cadence) |
| **Project age signal** | v0.4.6 (Apr 2026), ~28k stars, weekly churn | mujoco_warp v3.8.0.3 (May 2026), 1.2k stars; mujoco_playground v0.2.0 (Mar 2026), 1.9k stars | v2.0.0 (Apr 2025), 708 stars |
| **JAX native** | No (Torch + DLPack bridge required) | **Yes** — `jax.jit` / `jax.vmap` / `lax.scan` | No (PyTorch) |
| **Drone example shipped** | Yes (`hover_env`) | No (we own `crazyflie.xml`) | **Yes (vision-based obstacle avoidance)** |
| **Depth API** | `gs.renderers.BatchRenderer` (Madrona-backed, Vulkan or CUDA) | `mjx.warp.render.render` + `render_util.get_depth` (NVIDIA Warp BVH ray-trace) | Custom Warp-based renderer on Isaac Gym |
| **CTBR controller** | Must port (PID-over-RPM only) | **Carries over unchanged** | Must port |
| **Motor lag** | Must reimplement (Torch buffer) | **Carries over unchanged** | Must port |
| **Asymmetric Flax actor/critic** | Lost or risky (bridge) | **Preserved** | Lost (full Torch rewrite) |
| **M1.3 + M2 result validity** | Invalidated (different sim) | **Preserved** (if XLA physics kept) | Invalidated |
| **WSL2 + 4070 install risk** | High (Vulkan path, multiple open issues on our platform) | **Low** (CUDA-only, no Vulkan; in documented happy path) | High (Isaac Gym install is heavy) |
| **WSL2 Vulkan dependency** | Yes (Madrona rasterizer) | **No** | No (Warp-only) |
| **Migration LOC** | 1000+ (env + bridge + controller) | **50–150** (camera + render call + obs) | 1000+ (Torch rewrite) |
| **Drone-with-depth public precedent** | None (`hover_env` has no obstacles) | None | **Yes** |
| **API churn risk** | High (silent breaking changes every 2-4 weeks) | Low (tracks MuJoCo release train) | Medium (Isaac Gym deprecation overhang) |
| **Future deprecation risk** | Unlikely (active investment) | **Lowest** (DeepMind strategic direction) | High (Isaac Gym → Isaac Lab migration pending) |
| **Open depth bug** | Yes (Genesis #1648 cross-rasterizer discrepancy) | None reported | None reported |
| **Pinned tooling required** | `genesis-world==0.4.6` (until next breaking change) | `mujoco-mjx>=3.8.0`, `warp-lang>=1.11` | Isaac Gym + specific PyTorch version |

---

## Spike protocol (3 days)

The original spike protocol in `M2_5_genesis_port_spec.md` was Genesis-only. Replace with the following peer protocol.

### Day 1 — Both candidates: install

Run two parallel installs on the same WSL2 + 4070 mobile machine, with separate conda envs to prevent CUDA driver fights.

**Candidate B — MJX + MJWarp (do this first, low risk):**
- Existing `drone` env already has `jax[cuda12]` and `mujoco-mjx`. Upgrade to `mujoco-mjx>=3.8.0`.
- `pip install warp-lang>=1.11`.
- `pip install playground` (Playground v0.2.0, used for vision-env example reference).
- Smoke test: clone `google-deepmind/mujoco`, run `python mjx/mujoco/mjx/warp/visualize_render.py` if it accepts CLI args, OR run the equivalent code snippet from the example. Confirm a single batched depth tensor renders.

**Candidate A — Genesis (do this second, in a separate conda env):**
- `conda create -n genesis_spike python=3.11`
- `pip install torch genesis-world==0.4.6`
- WSL2 env vars per Genesis docs: `LD_LIBRARY_PATH=/usr/lib/wsl/lib`, `LIBGL_ALWAYS_INDIRECT=0`.
- Run `python examples/drone/hover_train.py` first 100 PPO iterations.
- Try `BatchRenderer` with `use_rasterizer=False` (CUDA path; avoids Vulkan).

**If MJWarp installs cleanly and Genesis hits Vulkan/CUDA/build errors in the first hour, stop installing Genesis. Allocate Days 2–3 to MJWarp validation only.** The opposite is unlikely but should be respected if it happens.

### Day 2 — Validation gates

For whichever candidate(s) installed:

| Gate | Candidate A (Genesis) | Candidate B (MJWarp) |
|---|---|---|
| G1. Drone scene loads | Load `cf2x.urdf` via `gs.morphs.Drone`, drop test | Load `crazyflie.xml` in MJX, single `mjx.step` |
| G2. Zero-action gravity | Drone falls under gravity, position matches free-fall analytic | Same in MJX (already known to work — sanity check) |
| G3. Match MJX baseline (Genesis only) | 100-step zero-action rollout matches MJX to <1e-3 m position | N/A — same sim |
| G4. Batched depth render | `BatchRenderer.render()` at n_envs=1024, 64×64. No NaN. Values in [0, max_depth]. | `render.render` + `get_depth` at n_envs=1024, 64×64. JAX array. |
| G5. Bridge round-trip (Genesis only) | DLPack Torch→JAX bridge: Flax forward on depth produces finite output | N/A — already JAX |
| G6. Determinism | Two rollouts under same seed differ <1e-5 on final state | Same |
| G7. Throughput | ≥10k env-steps/sec (sim + render) | ≥10k env-steps/sec (sim + render); ideally near M1.3's 65k for no-render |
| G8. Visual DR works | Genesis batched mass setter works on per-env basis | `MadronaWrapper(randomization_fn=...)` or equivalent in MJWarp; check for batched DR support |

### Day 3 — Decision write-up

Fill in the empirical results table and post-spike addendum to this document.

If only one candidate passed installation: that's the choice, full stop.

If both passed: **default to MJWarp** unless Genesis-specific evidence reveals a blocker for MJWarp that wasn't visible from desk research.

---

## Recommendation

**Use MJX + MJWarp for M3.** Do not migrate to Genesis. Do not adopt Aerial Gym.

### Why

1. **Preserves the M1.3 + M2 results.** We've spent months getting figure_eight_normal MED to 0.037 m on MJX-XLA with the asymmetric Flax actor/critic, separate Optax optimizers, RMA encoder `μ`, and adaptation encoder `ϕ`. MJWarp adds a renderer alongside our existing physics — none of those numbers are at risk.
2. **JAX-native, in-graph rendering.** The render call returns JAX arrays inside `jax.jit` and `jax.vmap`. The PureJaxRL `lax.scan` rollout pattern works with a single `render_token` thread. No DLPack bridge, no Torch policy rewrite, no rsl-rl.
3. **WSL2-friendly.** NVIDIA Warp is CUDA-only. The WSL2 Vulkan friction that affects Genesis is irrelevant. Our exact platform (Ubuntu 22.04 + CUDA 12 + 4070 mobile) is in the documented happy path for both Warp and MJX.
4. **Migration cost is small.** ~50–150 LOC, all additive: a `<camera>` element in MJCF, a render-context constructor, a render call in the env step, depth into the observation. No rewrites.
5. **Official strategic direction.** Google DeepMind explicitly deprecated madrona_mjx and pointed users at MJWarp. The renderer lives in the main `mujoco` repo. Long-term support is in the strongest possible posture.
6. **Future-proof for M5 sim-to-real.** Same MJX physics our existing M1/M2 deploys against. No "different sim during M3, return to MJX for sim-to-real" awkwardness.

### Why not Genesis

The "Genesis has a drone example" argument is real but small. We already have a better Crazyflie 2.1 MJX model than `hover_env` (motor lag, kf±30% DR, validated trajectory tracking, fault tolerance). We'd port nothing from `hover_env`; we'd start from our own MJCF either way.

The "Genesis is Madrona under the hood, MJWarp is also Madrona under the hood" claim from prior research was **wrong**. MJWarp is a fresh NVIDIA-Warp ray-traced renderer, not a Madrona wrapper. They're independent rasterizer-class projects. So "moving from Genesis to MJWarp loses Madrona-specific features" is not a real cost — neither project shares Madrona internals visible to us.

### Why not Aerial Gym

The pre-built obstacle-avoidance example would save us env design time. But it carries Isaac Gym (deprecated by NVIDIA), forces a Torch rewrite, and invalidates the M1/M2 results. Net cost is higher than MJWarp by a wide margin. Reserve Aerial Gym as a reference for *task design* — how they generate procedural clutter, what their reward looks like — without adopting it as our simulator.

### What this means for the prior documents

- **`M2_5_genesis_port_spec.md`** is superseded. The Genesis port doesn't happen. The "M2.5 milestone" reframes to "M2.5 = add MJWarp depth rendering to the existing MJX env and validate that M1.3 / M2 numbers survive the camera attachment."
- **`M3_genesis_assessment.md`** stays as a record of why Genesis was evaluated and rejected. Cross-reference it from this doc.
- **`M3_spec.md`** needs minor edits — references to Genesis as the sim become references to MJX + MJWarp; the spike gates section is replaced with a pointer to this doc. The actual M3 design (16-bin min-pooled depth, procedural cylinder clutter, soft+hard collision reward, etc.) is sim-agnostic and survives unchanged.
- **`M3_research_summary.md`** — independently updated by the primary-source verification pass (separate commit / section).

### Honest unknowns

- Throughput at 1024 envs on a 4070 mobile is not published for either candidate. The 3-day spike measures this empirically.
- MJWarp v3.8.0.3 is recent (May 8 2026); fewer in-the-wild stress reports than older alternatives. Risk of finding new bugs is real but bounded by the time-box.
- The MJWarp render call uses BVH ray tracing, which is deterministic given fixed scene state. No published numbers on render-time variance from background CUDA load. Validate in the spike.
- No published MJWarp benchmark of "depth rendering at 64×64, 1024 envs on a 4070-class GPU." We're flying somewhat blind on the throughput-vs-resolution tradeoff. Default to 64×64; raise only if throughput is comfortably above 30k env-steps/sec.

---

## Implementation discipline (post-decision)

If the spike validates MJWarp:

1. **Pin tooling explicitly.** Add to `scripts/setup_env.sh`:
   - `mujoco-mjx>=3.8.0,<4.0`
   - `mujoco>=3.8.0`
   - `warp-lang>=1.11,<2.0`
   - Keep `jax[cuda12]` unchanged (no version pin needed since MJWarp doesn't have the JAX-version ceiling madrona_mjx had).
2. **Keep physics on XLA, not MJWarp physics.** MJWarp offers a physics backend (different from its renderer). We're adopting the renderer only. Physics stays on the existing XLA `mjx.step` that produced M1.3 and M2. The MJWarp physics backend is a separate experiment to consider after M3 ships.
3. **BVH refit discipline.** Every step: physics → BVH refit → render. Document this in the env step function.
4. **Camera lives in MJCF, not in Python.** Add `<camera name="fpv" pos="0.02 0 0" xyaxes="0 -1 0 0 0 1" fovy="90"/>` to `crazyflie.xml` as a child of the drone body. Single source of truth.
5. **Single renderer instance per training process.** Don't accidentally create a second renderer for eval — that path is fragile in batch-render systems generally. Use the same renderer for training and in-loop eval.
6. **Render token threading.** The render call is stateful w.r.t. an opaque `render_token` JAX array. Thread it through the `lax.scan` carry like an RNG key. Document explicitly in the env class.

---

## What the user should do next

1. Read this document.
2. If you agree with the recommendation: skip the Genesis spike entirely. Run the 1-day MJWarp validation spike instead (gates G1, G2, G4, G6, G7, G8 against MJWarp; about half a day if MJX is already installed).
3. If you want to verify Genesis is genuinely worse before committing: run the 3-day peer spike as specified above. Either you'll confirm MJWarp wins, or you'll find a Genesis-specific surprise that changes the call.
4. After the spike: update `notes/M3_spec.md` to drop the Genesis-specific language, replace "M2.5 Genesis port" with "M2.5 MJWarp camera integration," and proceed.
5. **Do not start any porting work before the spike completes** — even with the strong recommendation here, the empirical install / throughput / determinism gates are what authorize the milestone, not desk research.

---

## Sourcing caveat

This document is desk research. The sandbox where it was written has no GPU, no WSL2, and could not run any candidate. All "install works" / "doesn't work" / "throughput is X" claims are flagged as "unverified" or "documented in [URL]." The 3-day spike on the actual target hardware is what turns these claims into knowledge.

References:
- [madrona_mjx deprecation README](https://github.com/shacklettbp/madrona_mjx/blob/main/README.md)
- [MJWarp render example](https://github.com/google-deepmind/mujoco/blob/main/mjx/mujoco/mjx/warp/visualize_render.py)
- [mujoco_warp repo](https://github.com/google-deepmind/mujoco_warp)
- [mujoco_playground v0.2.0 release notes](https://github.com/google-deepmind/mujoco_playground/releases)
- [Playground vision-PPO entry script](https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py)
- [NVIDIA Warp + WSL2 install discussion](https://github.com/NVIDIA/warp/discussions/149)
- [Genesis BatchRenderer docs](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/batch_renderer.html)
- [Genesis #1648 — depth value discrepancy](https://github.com/Genesis-Embodied-AI/Genesis/issues/1648)
- [Aerial Gym project page](https://ntnu-arl.github.io/aerial_gym_simulator/)
