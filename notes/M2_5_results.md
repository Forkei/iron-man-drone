# M2.5 Results — Depth Rendering + Obstacle Infrastructure

**Date:** 2026-05-12  
**Status:** Complete. All smoke-check gates pass.  
**Scope:** Rendering and obstacle infrastructure only — no policy changes, no new training objective.

---

## Summary

M2.5 adds MJWarp-based depth camera rendering and a 16-slot obstacle randomization system to the existing MJX environment. The M1–M2 physics stack, controller, trajectory system, RMA architecture, and PPO training loop are unchanged. Depth frames and obstacle privileged state are computed and exposed in `DepthEnvState` but not consumed by any policy — that is M3's job.

The central question was whether to switch the physics engine to Genesis (full migration) or bolt MJWarp's renderer onto MJX (additive port). After an honest spike test, the additive port was chosen. Reasons documented in the [decision log](#decision-log).

---

## Implementation — 5 tasks

### Task 1 — Regression validation after mujoco 3.8.1 upgrade

Installing `mujoco-warp==3.8.0.3` transiently upgraded `mujoco` from 3.6.0 to 3.8.1. Before any new code, M1.3 and M2 Phase 2 checkpoints were re-evaluated on the new stack. Numbers were bit-for-bit identical to canonical baselines (see L8 in `lessons.md`).

### Task 2 — `obstacle_randomization.py`

`sample_obstacle_configs(rng, n_obstacles)` → `(centers, half_extents)`, both `(N_OBSTACLE_SLOTS=16, 3)` float32.

- Active rows: xy from U(-3, 3), z=1.0, half_extents=[0.05, 0.05, 0.5].
- Spawn exclusion: ≥ 0.5 m from (0, 0) in XY (rejection sampling, max 50 retries).
- Inter-obstacle exclusion: ≥ 0.4 m between any two active centers in XY.
- Inactive rows: centers=[100, 100, 100], half_extents=zeros (parked far outside arena).

`min_distance_to_obstacle(drone_pos, centers, half_extents, n_obstacles)` — L∞ box SDF over active slots. Returns `jnp.inf` when `n_obstacles=0`.

### Task 3 — `quadrotor_env_depth.py`

`DepthEnvState` extends M2's `EnvState` with four fields: `obstacle_positions (16, 3)`, `obstacle_half_extents (16, 3)`, `n_obstacles (scalar int32)`, `depth (64, 64) float32`.

`make_depth_reset_fn` / `make_depth_step_fn` wrap M2's reset/step exactly. Obstacle fields are carried through each step unchanged; depth is zeroed at step time and populated externally by `batch_render`.

`DepthVecEnv.__init__` initializes MJWarp lazily on construction (not on module import — satisfies F-E). `batch_render(states)` does the MJX→MJWarp state transfer and returns `(N, 64, 64)` float32 in [0, 1].

State-transfer path: `qpos (N, 7)` and `obstacle_positions (N, 16, 3)` (as `wp.vec3f`) are written into MJWarp data, then `mjw.forward` → `mjw.render` → `mjw.get_depth`. DLPack zero-copy was considered but has a stale-buffer aliasing hazard in warp 1.13.0; the implementation uses `depth_buf.numpy()` → `jnp.array()` (two copies, GPU→CPU→GPU) for safety.

`DepthVecEnv` exposes `step` / `reset` attributes (JAX-traceable, no-obstacle) for `collect_rollout` compatibility, alongside `batch_reset` (Python-level obstacle sampling) for episode-start resets.

### Task 4 — MJX→MJWarp state-transfer benchmark

| N | Total (ms) | Transfer (ms) | Forward (ms) | Render+copy (ms) | Throughput (k steps/s) | Transfer% |
|---|---|---|---|---|---|---|
| 1 | 18.7 | 0.1 | 2.5 | 17.4 | 0.1 | 0.7% |
| 64 | 19.1 | 0.1 | 2.9 | 19.3 | 3.3 | 0.7% |
| 256 | 20.1 | 0.1 | 3.0 | 15.3 | 12.7 | 0.6% |
| **1024** | **30.6** | **0.2** | **2.8** | **18.7** | **33.5** | **0.6%** |

SC-4 gate (≥ 25k steps/sec at N=1024): **PASS — 33.5k**.  
Transfer overhead gate (≤ 20%): **PASS — 0.6%**.

Option A (MJX physics + MJWarp render) confirmed viable for M3. Full benchmark in `notes/M2_5_benchmark_results.md`.

### Task 5 — Regression validation (all smoke checks)

| SC | Description | Result | Gate | Status |
|---|---|---|---|---|
| SC-1 | M2 Phase 1 actor on DepthVecEnv, figure_eight_normal nominal | 0.0574 m | ≤ 0.065 m | **PASS** |
| SC-2 | M1.3 actor on DepthVecEnv, figure_eight_normal nominal | 0.0402 m | ≤ 0.042 m | **PASS** |
| SC-3 | Depth render: shape (N,64,64), dtype, range [0,1], obstacle visible, world contrast | 5/5 | all pass | **PASS** |
| SC-5 | Obstacle geometry (100 samples): spawn excl, inter excl, z=1.0, he, parking | 8/8 | all pass | **PASS** |
| SC-6 | DepthVecEnv PPO training, 200 epochs, N=512 | reward 1.12 at ep.200 | ≥ 0.5 | **PASS** |

SC-1 and SC-2 match their VecEnv canonical baselines exactly (Δ = 0.0000 m). The 16 mocap bodies in `crazyflie_depth.xml` have zero impact on drone dynamics.

---

## Regression results — dynamics parity

| Policy | Env | figure_eight_normal nominal | Δ |
|---|---|---|---|
| M1.3 epoch_013000 | VecEnv (canonical, 2026-05-09) | 0.0402 m | — |
| M1.3 epoch_013000 | DepthVecEnv (SC-2, 2026-05-12) | 0.0402 m | 0.0000 m |
| M2 Phase 1 final | VecEnv (canonical, 2026-05-11) | 0.0574 m | — |
| M2 Phase 1 final | DepthVecEnv (SC-1, 2026-05-12) | 0.0574 m | 0.0000 m |

---

## Decision log — MJWarp over Genesis

**Option evaluated:** Genesis (full physics + rendering migration).  
**Concerns:** Full stack rewrite mid-project, loss of all MJX JAX infrastructure (`lax.scan` rollout, vmapped reset/step, existing checkpoint compatibility), unclear Genesis API stability, install complexity on WSL2.

**Option evaluated:** MJWarp (rendering only, additive to MJX).  
**Spike results (2026-05-11, `notes/M3_mjwarp_spike_results.md`):**
- 6/6 gates pass: install, JAX array compatibility, N=1024 parallel render, throughput, no Vulkan/EGL needed on WSL2, dynamics parity (L2 < 5e-4 over 50 steps, zero action).
- Throughput with rendering: 32.6k env-steps/sec (pre-M2.5 spike), 33.5k env-steps/sec (M2.5 final with 18-body model including obstacle slots).
- CUDA-native: works without Vulkan, EGL, or Mesa in WSL2 — eliminates a common failure mode.

**Decision:** MJWarp (Option A — MJX physics + MJWarp render only). Rationale:
1. Additive port preserves all MJX investment. The `lax.scan` training loop and vmap infrastructure are untouched.
2. State-transfer overhead is 0.6% — effectively zero cost for the Option A bridge.
3. M3 can wire depth into the policy without any simulator migration.

M3 architectural decision (Option A vs Option B — full MJWarp physics) deferred until M3 determines whether render-every-step is required. If render frequency is low (e.g., every 32 steps), Option A remains optimal. If render-every-step is needed, Option B costs 23× throughput; that tradeoff is M3's call.

---

## References to lessons.md

- **L8** — mujoco 3.6→3.8.1 upgrade is safe; regression validation confirmed bit-identical results. Rule: any physics-stack dependency upgrade requires full regression eval before new code lands on top of it.
- **L9** — MJX throughput: GPU occupancy effect (4–5× ratio between 2-body and 18-body models) and warm-GPU session effect (~2× for small models). Rule: never compare throughput across sessions without noting warm vs cold GPU state.
