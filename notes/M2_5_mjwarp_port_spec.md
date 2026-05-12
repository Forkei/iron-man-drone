# M2.5 Spec — MJWarp Depth Rendering Integration

**Date:** 2026-05-11
**Status:** Draft — pending user review. Implementation begins after approval.
**Supersedes:** `notes/M2_5_genesis_port_spec.md` (sim migration path, abandoned after MJWarp spike passed)
**Prerequisite:** `notes/M3_mjwarp_spike_results.md` — all 6 gates PASS (2026-05-11)
**Time-box:** 3–5 days of focused work. Escalate if exceeded.

---

## One-line goal

Add a MJWarp depth camera channel to the existing MJX environment — preserving all M1–M2
dynamics, controller, trajectory, and RMA infrastructure exactly — so that M3 can consume
depth observations from day one without a mid-training sim migration.

---

## What M2.5 is NOT

M2.5 is an **additive port**, not a sim migration. The existing code is the ground truth;
M2.5 only adds the rendering path alongside it. Specifically:

- No change to dynamics model (MJX physics stays as the physics engine)
- No change to controller (`ctbr_controller.py`), trajectory system, or reward
- No change to actor/critic architecture or observation dimensions (still 50-dim actor)
- No change to PPO training hyperparameters
- No change to RMA two-phase infrastructure
- Depth tensor is **computed but not consumed by the policy** — that is M3's job
- Obstacle privileged state (positions, half-extents, min-distance function) is **exposed in `EnvState` but not wired to actor or critic** — M3 plugs in from this infrastructure rather than adding it mid-training
- No obstacle avoidance reward, no changes to training distribution

---

## Architecture decision: Option A (MJX physics + MJWarp render-only)

The spike evaluated full MJWarp (physics + render, Option B, 32.6k env-steps/sec) and
partial MJWarp (MJX physics + MJWarp render-only, Option A, state-transfer overhead TBD).

**M2.5 adopts Option A** (additive to MJX) for the following reasons:

1. "Additive port" framing: existing lax.scan training loop stays intact. MJWarp render
   is a side-call outside the JAX execution graph — no scan rewrite required.
2. MJX physics is 23× faster than MJWarp physics (748.9k vs 32.6k env-steps/sec);
   preserving MJX physics keeps training speed near current levels if renders are infrequent.
3. State-transfer overhead (MJX xpos/xmat → MJWarp) is unknown and must be measured
   before committing to Option B. M2.5 includes a benchmark task for this.

**M3 architectural decision is deferred until M2.5 measures state-transfer overhead.**
If transfer cost is low (< 5% of step time), Option A remains optimal for M3 too.
If transfer cost is high, M3 may switch to Option B (full MJWarp); this will not require
re-doing M2.5 because the rendering infrastructure (MJCF, camera, obstacle randomization)
is shared between both options.

### Render call placement in training vs eval

**During training:** MJWarp render is called at the end of each rollout horizon (every
`horizon = 32` steps), not every step. This amortizes the MJX→MJWarp state transfer cost
and keeps the hot lax.scan loop pure JAX. The depth tensor is stored but not included in
the PPO update — verifying that compute overhead does not destabilize training.

**During eval:** MJWarp render is called every step. The eval loop is already a Python
for-loop (not lax.scan), so per-step render calls are natural. Depth frames can be saved
for visualization and sanity-checking.

---

## Implementation tasks

Tasks are ordered; each is a prerequisite for the next.

### Task 1 — Extend MJCF for multiple randomizable box obstacles

**File:** `src/iron_man_drone/envs/crazyflie_depth.xml`

Current state: one fixed box pillar at `pos="2.0 0.0 1.0"` (from spike). This is the
depth scene MJCF; the original `crazyflie.xml` stays untouched.

Changes required:
- Use a single constant `N_OBSTACLE_SLOTS = 16` everywhere: MJCF body count, EnvState
  tensor shape, and sampling code. The MJCF, EnvState, and sampling code must never
  disagree on this number.
- Replace the single fixed pillar with 16 named mocap box bodies (`obstacle_0` through
  `obstacle_15`). Use `mocap=true` bodies so positions can be updated via
  `data.mocap_pos` at reset without rebuilding the model.
- All 16 bodies exist in the MJCF regardless of how many are active per episode; inactive
  bodies are parked at a far-away position (e.g. `pos="100 100 100"`) at reset.
- Define obstacle size in the MJCF as a default: box `0.1 × 0.1 × 1.0 m` (same as spike
  pillar). Size can be parameterized later for M3.
- Ensure all obstacle geoms have `contype=1 conaffinity=1` (solid for collision with
  ground/walls) but are out of contact range with the drone during training
  (drone geoms remain `contype=0 conaffinity=0` as in spike). Drone–obstacle collision
  is not modeled in M2.5; M3 will add collision reward and termination.
- Depth camera: `name="depth_cam"`, `mode="fixed"` body-mounted on drone, `fovy="60"`,
  resolution `64×64`, **`clipfar="5.0"`** (covers M3 flight envelope; up from 1 m in spike).

No changes to `crazyflie.xml` (used by M1/M2 training — must remain untouched).

### Task 2 — Obstacle randomization at episode reset

**New file:** `src/iron_man_drone/utils/obstacle_randomization.py`

Implement two functions:

**`sample_obstacle_configs(rng: np.random.Generator, n_obstacles: int) → ((N_OBSTACLE_SLOTS, 3), (N_OBSTACLE_SLOTS, 3))`**
Pure numpy function — called at the Python level before JIT, not inside traced code.
Returns `(centers, half_extents)`. Both arrays are `(N_OBSTACLE_SLOTS=16, 3)` numpy
float32; active obstacles occupy the first `n_obstacles` rows, inactive rows parked at
`[100, 100, 100]` (out of scene). Results are passed as static arguments into the JIT'd
reset function. JAX static shapes are preserved — no dynamic resizing inside JIT.

Rules for `centers`:
- Obstacle xy sampled from `U(-3.0, 3.0) × U(-3.0, 3.0)` independently per obstacle.
- Obstacle z fixed at `1.0` (pillar base at floor, top at 2 m, consistent with spike).
- **Spawn exclusion zone:** any obstacle whose xy distance to `(0, 0)` is less than
  `0.5 m` is resampled. This prevents obstacle-at-spawn crashes.
- **Inter-obstacle exclusion:** obstacles must be `≥ 0.4 m` apart (center-to-center).
  Rejection-sample with a max-retry limit of 50; accept with warning if unmet (should
  be rare at N_OBSTACLES ≤ 4 in a 6×6 m space).
- Padded (inactive) rows set to zeros.

Rules for `half_extents`:
- Default pillar: `half_extents = [0.05, 0.05, 0.5]` (0.1 × 0.1 × 1.0 m box, matching
  spike geometry). Parameterizable for M3 if needed, but fixed for M2.5.
- Padded rows set to zeros.

**`min_distance_to_obstacle(drone_pos: jnp.ndarray, centers: jnp.ndarray, half_extents: jnp.ndarray, n_obstacles: int) → float`**
Returns the minimum L∞ surface-to-surface distance from `drone_pos` to the nearest
active obstacle (first `n_obstacles` rows). Computed on demand — not stored per step.
Available for M3 collision reward and for smoke-test validation. Use box SDF:
`dist_to_box = max(0, max(|drone_pos - center| - half_extents))` per axis.

**Integration point:** call `sample_obstacle_configs` inside `make_reset_fn` (depth
variant) before creating the initial `EnvState`. Store in `EnvState` as:
```
obstacle_positions:    jnp.ndarray  # (N_OBSTACLE_SLOTS=16, 3) — centers, inactive parked at [100,100,100]
obstacle_half_extents: jnp.ndarray  # (N_OBSTACLE_SLOTS=16, 3) — half-extents, inactive rows zero
n_obstacles:           int           # number of active obstacles ≤ N_OBSTACLE_SLOTS (for M3 masking)
```
These fields are accessible to the env and critic (privileged info) but not wired to
either in M2.5. `n_obstacles = 0` with all-zero tensors is a valid state (M1/M2 mode).

**Sanity test (gate for Task 2 completion):** sample 100 obstacle configurations, assert
no active obstacle within 0.5 m of spawn, assert no two active obstacles within 0.4 m
of each other, assert all active z values = 1.0, assert padded rows are zeros in both
tensors, assert `min_distance_to_obstacle` returns a positive float for a drone at origin.

### Task 3 — Depth VecEnv (new env file, does not replace existing)

**New file:** `src/iron_man_drone/envs/quadrotor_env_depth.py`

Do NOT modify `quadrotor_env.py`. Create a parallel file that:

1. Imports and re-uses `make_step_fn` and reward/termination logic from `quadrotor_env.py`
   without duplication — only the reset function and the top-level `DepthVecEnv` class are new.

2. `make_depth_reset_fn`: identical to `make_reset_fn` in `quadrotor_env.py` except:
   - Loads `crazyflie_depth.xml` (not `crazyflie.xml`)
   - Calls `sample_obstacle_configs` to fill obstacle fields in `EnvState`
   - Initializes `mujoco_warp` model at module import time (one-time cost)

3. `DepthVecEnv`: mirrors the `VecEnv` interface but adds:
   ```
   def batch_render(self, states: EnvState) -> jnp.ndarray:
       """Returns depth arrays (N, 64, 64) as a JAX float32 tensor."""
   ```
   Internal implementation:
   - Write `states.mjx_data.xpos` and `states.mjx_data.xmat` into the MJWarp model
     (state transfer from MJX to MJWarp).
   - Call `mjw.render(cam_id=0)` → Warp CUDA array.
   - `wp.to_jax(depth_wp)` → JAX array (zero-copy DLPack).
   - Return shape `(N_ENVS, 64, 64)` float32, values in `[0, 1]` (normalized depth,
     max range 1.0 m from spike; adjust if needed for M3 flight envelope).

4. `EnvState` in this file extends the M2 `EnvState` with:
   ```
   obstacle_positions:    jnp.ndarray  # (MAX_OBSTACLES, 3) — centers, zero-padded
   obstacle_half_extents: jnp.ndarray  # (MAX_OBSTACLES, 3) — half-extents, zero-padded
   n_obstacles:           int           # number of active obstacles
   depth: jnp.ndarray                  # (64, 64) — last rendered depth frame
   ```
   `depth` starts as zeros; populated by `batch_render`. All obstacle fields start
   populated by `make_depth_reset_fn`. They are carried in state every step but not
   consumed by the actor, critic, or training update in M2.5 — M3 wires them in.

5. Obs dict from step: return a plain Python dict alongside the existing actor/critic
   obs tensors:
   ```python
   obs_dict = {
       "actor_obs":  actor_obs,    # (N, 50) — unchanged
       "critic_obs": critic_obs,   # (N, 51) — unchanged
       "depth":      states.depth, # (N, 64, 64) — new, zeros if not yet rendered
   }
   ```
   The existing `actor_obs` and `critic_obs` are identical to M2; nothing in the
   training update path changes.

### Task 4 — Measure MJX→MJWarp state-transfer overhead

**New script:** `scripts/benchmark_mjwarp_transfer.py`

Measure the cost of transferring MJX state to MJWarp before each render call.

Protocol:
1. Create `DepthVecEnv` with `N = [1, 64, 256, 1024]` envs.
2. Run 200 `env.batch_step` calls (MJX physics, lax.scan).
3. For each N: time `env.batch_render(states)` over 50 calls after warm-up.
4. Report: render latency (ms), state-transfer latency (ms, measured separately if
   possible), and effective env-steps/sec = N / render_latency.

**Decision gate:** If state-transfer overhead > 20% of MJWarp render time, document
in results and flag for M3 architectural decision (may favour Option B for M3).

Write results to `notes/M2_5_benchmark_results.md`.

### Task 5 — Eval suite extension (depth-compatible eval)

**File to extend:** `src/iron_man_drone/evaluation/eval_suite.py`

Add an optional `render_depth: bool = False` parameter to `run_eval_suite`. When enabled:
- After each step, call `env.batch_render(states)` to collect depth frames.
- Accumulate depth frames per episode (shape `(T, 64, 64)`).
- After eval completes, save a sample depth frame per trajectory as PNG to `notes/figures/`.

This must be a no-op when `render_depth=False` — existing M1/M2 eval scripts are unchanged
and pass exactly the same `VecEnv`, not `DepthVecEnv`.

---

## Success criteria

All must pass before M2.5 is tagged and M3 begins.

### SC-1 — No regression: M2 policy with render enabled

Run `python scripts/eval_m2_full.py --env depth --render_depth` on the frozen M2 Phase 2
checkpoint (`experiments/phase2_encoder/best_checkpoint`).

**Gate:** figure_eight_normal nominal MED ≤ 0.065 m (same as M2 Phase 2 gate). The
render path must not perturb dynamics.

Implementation note: the eval loop runs `DepthVecEnv._step_fn` (MJX physics, same code
path as `VecEnv._step_fn`). If physics code is correctly shared, this gate passes by
construction. It exists to catch accidental divergence (e.g., wrong xml, different timestep).

### SC-2 — M1.3 tracking regression check

Run the unified eval suite on the M1.3 canonical checkpoint
(`experiments/m1_3_polynomial_fix/.../epoch_013000`) using `DepthVecEnv`.

**Gate:** figure_eight_normal MED ≤ 0.037 m (M1.3 canonical number).

This verifies the depth MJCF (`crazyflie_depth.xml`) does not accidentally alter dynamics
relative to `crazyflie.xml`. The physics XML sections must be byte-identical between the
two files (only additions: camera, obstacle bodies).

### SC-3 — Depth renders are sensible

Run `scripts/smoke_test_depth.py` (new script, written as part of M2.5):
- Spawn drone at origin, place a box obstacle at `(1.5, 0, 1.0)`.
- Render one depth frame.
- Assert obstacle is visible: at least one pixel with depth < 0.8 (obstacle at 1.5 m,
  normalized to 1.0 m max → obstacle appears at depth ≈ 1.0 in a 1 m-range camera, so
  adjust camera far-plane to 5 m and verify obstacle pixel depth < 0.5).
- Save frame to `notes/figures/m2_5_depth_smoke.png` for visual inspection.

### SC-4 — Throughput with rendering

Run `scripts/benchmark_mjwarp_transfer.py` (Task 4) at N=1024 envs.

**Gate:** effective throughput ≥ 25k env-steps/sec at 1024 envs with one render per step.

The spike achieved 32.6k env-steps/sec (MJWarp physics + render). Option A (MJX physics
+ MJWarp render-only) will have different characteristics; 25k is a conservative floor
that allows for state-transfer overhead. If below 25k, diagnose the bottleneck before
proceeding to M3.

### SC-5 — Obstacle randomization validation

Run `scripts/smoke_test_obstacles.py` (new script):
- Sample 100 obstacle configurations.
- Assert: no obstacle within 0.5 m of `(0, 0)`, no pair within 0.4 m, z = 1.0 for all.
- Assert: episode reset with random obstacles runs 1000 steps without MJX/MJWarp error.

### SC-7 — Obstacle privileged state correctness

Run `scripts/smoke_test_obstacle_state.py` (new script):

- Create `DepthVecEnv` with `n_obstacles = 4`.
- After reset: assert `state.obstacle_positions.shape == (MAX_OBSTACLES, 3)` and
  `state.obstacle_half_extents.shape == (MAX_OBSTACLES, 3)`.
- Assert first 4 rows of `obstacle_positions` are non-zero; rows 4–15 are zeros.
- Assert first 4 rows of `obstacle_half_extents` match the default pillar geometry
  `[0.05, 0.05, 0.5]`; rows 4–15 are zeros.
- Call `min_distance_to_obstacle(drone_pos=(0,0,0), ...)` and assert result > 0.4
  (spawn exclusion is 0.5 m center-to-center; surface distance = 0.5 - 0.05 half-extent = 0.45 m min; 0.4 is a conservative safe floor).
- Create `DepthVecEnv` with `n_obstacles = 0`: assert all obstacle fields are zero-tensors,
  assert `min_distance_to_obstacle` returns `inf` or a large sentinel value.
- Run the existing M1/M2 eval scripts (which use plain `VecEnv`, not `DepthVecEnv`):
  assert they complete without error — verifying zero-obstacle compatibility is
  automatic when those scripts never touch `DepthVecEnv`.

**Gate:** all assertions pass; M1.3 and M2 eval scripts run without modification.

---

### SC-6 — M2 Phase 1 training still converges (abbreviated run)

Run `python scripts/train_m2.py --env depth --total_epochs 200`.

**Gate:** reward curve does not collapse by epoch 200 (mean reward ≥ 0.5; same threshold
used for M2 Phase 1 training health checks). This is a smoke test, not a convergence
check — full convergence is not expected at 200 epochs. The goal is to confirm the training
loop does not error, gradient flow is healthy, and entropy does not collapse.

---

## Out of scope for M2.5

- Any policy architecture change (M3 territory)
- Depth in actor/critic observation or training loss
- Obstacle avoidance reward or curriculum
- Trajectory generator changes
- Multi-obstacle collision avoidance during flight
- Dynamic obstacles
- Any camera other than the 64×64 depth camera defined in the spike
- Wind force (already out of scope for M2 Phase 1; still out of scope here)

---

## Failure modes to watch for

### F1 — Render path silently corrupts dynamics
**Symptom:** M2 figure_eight MED regresses vs. canonical numbers when `DepthVecEnv` is used.
**Guard:** SC-1 and SC-2 catch this.
**Most likely cause:** `crazyflie_depth.xml` has different physics parameters from
`crazyflie.xml` (timestep, inertia, etc.). Fix by diffing XML physics sections.

### F2 — OOM at 1024 envs with rendering
**Symptom:** CUDA OOM error in `batch_render` at N=1024.
**Guard:** SC-4 benchmark will reveal this before M3.
**Most likely cause:** depth buffer is 64 × 64 × 4 bytes = 16 KB per env; at 1024 envs
that is 16 MB total — not itself a problem. The actual OOM risk is the MJWarp render
pipeline allocating intermediate CUDA buffers per-env. If OOM manifests, chunk renders
(e.g., 128 envs per call, 8 calls per step) or reduce to N=512 envs for M3 training.
Document the maximum stable N and set M3's `num_envs` accordingly.

### F3 — MJX→MJWarp state transfer too slow
**Symptom:** SC-4 throughput < 25k env-steps/sec.
**Guard:** Task 4 benchmark measures this explicitly.
**Response if slow:** consider render subsampling (render every 5 steps instead of every
step); depth observation would then be 5 steps stale. Document the staleness assumption in
the M3 spec and measure whether it degrades obstacle avoidance performance.

### F4 — WSL2/CUDA driver issues (unlikely after spike)
**Symptom:** `import warp` fails or MJWarp render crashes, despite spike passing.
**Context:** spike ran successfully on 2026-05-11 with warp 1.13.0 + mujoco-warp 3.8.0.3.
Driver version must remain pinned. `pip install warp-lang==1.13.0 mujoco-warp==3.8.0.3`.
If WSL2 kernel update changes CUDA ABI, uninstall/reinstall both packages.

---

## Repo changes summary

| Path | Change |
|---|---|
| `src/iron_man_drone/envs/crazyflie_depth.xml` | Extend: add N randomizable mocap obstacle bodies |
| `src/iron_man_drone/envs/quadrotor_env_depth.py` | New: DepthVecEnv + make_depth_reset_fn |
| `src/iron_man_drone/utils/obstacle_randomization.py` | New: sample_obstacle_configs + min_distance_to_obstacle |
| `src/iron_man_drone/evaluation/eval_suite.py` | Extend: optional render_depth param |
| `scripts/benchmark_mjwarp_transfer.py` | New: state-transfer benchmark |
| `scripts/smoke_test_depth.py` | New: SC-3 depth sanity test |
| `scripts/smoke_test_obstacles.py` | New: SC-5 obstacle randomization test |
| `scripts/smoke_test_obstacle_state.py` | New: SC-7 obstacle privileged state test |
| `experiments/m2_5_baseline/` | New: abbreviated training run artifact |
| `notes/M2_5_benchmark_results.md` | New: Task 4 output |
| `notes/figures/m2_5_depth_smoke.png` | New: SC-3 visual output |

Files that must NOT change:
- `src/iron_man_drone/envs/crazyflie.xml` (M1/M2 training XML)
- `src/iron_man_drone/envs/quadrotor_env.py` (M1/M2 training env)
- `src/iron_man_drone/policy/networks.py`, `ppo.py`, `encoder.py`
- Any frozen checkpoints under `experiments/`

---

## Flag for M3 spec

M3 spec does not yet exist. When it is written, note the following decisions made in M2.5:

1. **Obstacle type is box-only.** Cylinder-box collision pairs are unsupported in MJWarp.
   M3 obstacle design (gates, walls, pillars) must use box geometry exclusively.
2. **Depth camera parameters are fixed at 64×64, fovy=60.** Change requires re-running SC-3/SC-4.
3. **Option A vs Option B decision for M3 training throughput** is deferred to M2.5 Task 4
   benchmark. M3 spec should include the benchmark result and justify the final choice.
4. **N_OBSTACLES** for M3 training is TBD from SC-4/SC-5; M2.5 validates N=4 with SC-5 only.
   M3 may need a different value depending on scene complexity and OOM behavior (F2).
5. If MJX→MJWarp state transfer overhead is high (F3), M3 must decide on render subsampling
   frequency and document the resulting staleness in the depth observation.
