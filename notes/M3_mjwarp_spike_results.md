# M3 MJWarp Spike Results

**Date:** 2026-05-11
**nworld:** 1024  **cam_res:** 64×64  **steps:** 200

## Verdict: **PASS** — proceed with MJWarp as M2.5/M3 depth-render backend

## Gate Results

- Gate 1: **PASS** — Install + GPU reachable
- Gate 2: **PASS** — JAX-compatible depth arrays
- Gate 3: **PASS** — Depth render at 1024 parallel envs
- Gate 4: **PASS** — Throughput > 10k env-steps/sec (with render)
- Gate 5: **PASS** — No Vulkan/driver issues on WSL2
- Gate 6: **PASS** — Zero-action dynamics match reference (L2 < 5e-4, float32 tolerance)

## Throughput

| Condition | Throughput |
|---|---|
| MJX-only (JAX vmap, 1024 envs) | 748.9k env-steps/sec |
| MJWarp + depth render (1024 envs) | 32.6k env-steps/sec |
| Render overhead | 23.0× slowdown |

## Install Notes

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- warp 1.13.0, mujoco-warp 3.8.0.3
- Scene: crazyflie_depth.xml, ngeom=9, cam_id=0
- Depth shape: (1, 64, 64), max=1.000m
- Pillar visible in single-world render: True
- 1024-world depth shape: (1024, 64, 64)
- MJX-only throughput : 748.9k env-steps/sec
- MJWarp+render       : 32.6k env-steps/sec
- Render slowdown     : 23.0×
- MJWarp rendering is CUDA-only, no Vulkan requirement
- Dynamics diff (50 steps, zero action): 2.46e-07

## Integration Notes

- MJWarp uses CUDA (not Vulkan): no Vulkan/EGL dependencies in WSL2.
- `wp.to_jax(depth_wp)` converts Warp CUDA arrays to JAX arrays (zero-copy DLPack).
- The depth arrays are JAX-consumable and can be passed to jit-compiled policy networks.
- MJWarp render is NOT JAX-jit compiled (it's a CUDA kernel launched by Warp),
  but the output is compatible with JAX's JIT via DLPack interop.
- For M2.5: run MJX for physics, call `mjw.render` + `wp.to_jax` for each policy step.
- Alternatively: switch physics to MJWarp (its own `step`) and keep JAX for policy.

## Architectural Decision Points for M2.5

**Throughput concern (23× slowdown):** MJX-only physics reaches 748k env-steps/sec; MJWarp
physics+render reaches only 32.6k. At 10Hz policy frequency, 32.6k env-steps/sec = 3.26k
parallel envs per second of wall-clock time — still 32 environments running in real time.
For training, the relevant number is training throughput in env-steps/sec across all envs,
not real-time factor. 32.6k env-steps/sec at 1024 envs means 32 real-time seconds of
simulation per wall-clock second — adequate for RL training but ~23× slower than M2.

**Key decision before M2.5:** Choose one of:
- **Option A**: MJX physics + MJWarp render-only. Physics stays fast (748k); state must be
  transferred MJX→MJWarp on each render call. State transfer overhead TBD (not measured here).
  Renders every N steps (not every step) to amortize cost.
- **Option B**: MJWarp for everything (physics + render). Simpler pipeline; 32.6k throughput.
  Requires all obstacle geoms to be box type (cylinder-box collision not implemented).

**Geom type limitation:** MJWarp does NOT support cylinder-box contact pairs. If using MJWarp
physics, all obstacles must be box type (not cylinders). The Crazyflie drone geoms are disabled
for collision (contype=0) in the depth scene, which sidesteps this for training, but the real
M3 obstacle geometry must be designed accordingly (box pillars, box gates, etc.).

**Recommendation for M2.5:** Start with Option B (MJWarp for everything). Simpler to implement;
32.6k env-steps/sec is sufficient for M3 training. Design obstacles as boxes. If training
speed becomes a bottleneck, revisit Option A with measured state-transfer cost.
