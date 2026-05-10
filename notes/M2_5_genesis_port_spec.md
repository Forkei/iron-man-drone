# M2.5 Spec — Genesis Port

**Date:** 2026-05-10
**Prerequisite reading:** `notes/M3_genesis_assessment.md`, `notes/M2_spec.md`, `notes/lessons.md`
**Status:** Draft — awaiting user approval before any code or environment changes.

---

## One-line goal

Reproduce M1.3-equivalent trajectory tracking on Genesis (Crazyflie 2.1 dynamics, CTBR action space, asymmetric actor/critic via PPO) within a 2-week time-box, as a prerequisite to M3 visual obstacle avoidance. **If not converging by the time-box, fall back to MJX + a depth-rendering bridge (madrona_mjx) and revise M3 plan.**

---

## Why M2.5 exists

M3 requires GPU-batched depth rendering across many envs. MJX has no native depth path. Genesis has `BatchRenderer` (Madrona-backed) which provides this. But Genesis is **PyTorch-tensor native, has no JAX backend, no CTBR controller, no motor lag, and no public drone RL precedent beyond `examples/drone/hover_env.py`**. Before betting M3 on Genesis, we confirm Genesis can train the dynamics we already understand. M2.5 = "Genesis can do what MJX did for M1.3."

**M2.5 is NOT shipped under the M2 fault-tolerance umbrella.** It deliberately drops RMA (the `μ` encoder, the 70%-fault DR, the adaptation encoder `ϕ`) to isolate the simulator-port question from the algorithm-port question. M2.5 is a clean M1.3-equivalent re-baseline on a new simulator. The fault-tolerance question for M3 is in `M3_spec.md` §F1.

---

## Pre-port spike (gate before committing 2 weeks)

**3-day time-box.** All six gates must pass:

| Gate | Pass criterion | Estimated time |
|---|---|---|
| G1. Install | `python -c "import genesis as gs; gs.init(backend=gs.cuda); print(gs.device)"` returns CUDA on WSL2 + 4070 | 4–8 h (multiple known issues — see `M3_genesis_assessment.md` §6) |
| G2. Drone example runs | `python examples/drone/hover_train.py` first 100 PPO iterations complete, reward trends up | 2 h |
| G3. Batched depth renders | `BatchRenderer` returns finite depth tensor at 64×64, n_envs=1024, no NaNs, values within [0, max_depth] | 4 h |
| G4. DLPack Torch→JAX bridge | Zero-copy CUDA bridge: Flax module forward on the Genesis depth tensor produces finite output. `torch.utils.dlpack` + `jax.dlpack` round-trip is bit-exact | 4 h |
| G5. Determinism | Two rollouts under same seed differ by &lt;1e-5 on final state across 100 steps | 2 h |
| G6. Throughput | Sim + render &gt; 10k env-steps/sec on the 4070 at n_envs=1024, dt=0.01, depth=64×64 | 2 h |

**Past day 3 with any gate unmet → abort migration.** Fall back paths (in order of preference):
1. **madrona_mjx** — same Madrona renderer wired into MJX, keeps the JAX stack. Less mature; depth-only paths may need patching.
2. **Aerial Gym + Isaac Gym** — mature drone-with-depth sim, but requires Torch policy rewrite (kills our Flax + asymmetric optimizers).
3. **Hand-roll depth via JAX raycasting** — last-resort, slow.

---

## Port checklist (after spike passes)

Pin `genesis-world==0.4.6` exactly. Pin the JAX, Torch, CUDA versions in `scripts/setup_env.sh`.

### A. Environment setup

- [ ] **A1.** Add Genesis branch of `scripts/setup_env.sh` — installs `genesis-world==0.4.6`, `torch` (CUDA 12), keeps `jax[cuda12]` and `flax` and `optax`. Document the WSL2 environment variables (`LD_LIBRARY_PATH=/usr/lib/wsl/lib`, `LIBGL_ALWAYS_INDIRECT=0`).
- [ ] **A2.** Verify CUDA device visible: `gs.init(backend=gs.cuda)`.
- [ ] **A3.** Smoke test: load `cf2x.urdf` via `gs.morphs.Drone(model='CF2X')`, take a single `scene.step()` with all RPM=0, confirm drone falls under gravity.

### B. Dynamics — port `envs/quadrotor_env.py`

The existing MJX env is structured for `jax.lax.scan`-friendly stateless step functions. Genesis is imperative and stateful. Rewrite, not migrate.

- [ ] **B1.** New module `src/iron_man_drone/envs/quadrotor_genesis_env.py`. Same public API (`reset`, `step`, observation/reward functions) but Torch-tensor-backed.
- [ ] **B2.** Load drone: `drone = scene.add_entity(gs.morphs.Drone(model='CF2X', pos=(0,0,0.5)))`.
- [ ] **B3.** Set `kf`, `km`, mass, inertia in URDF or via post-load `set_mass` / `set_inertia` (verify the v0.4.6 batched setters work). Match our M1 constants exactly:
  - mass: 0.0321 kg
  - inertia: diag(1.4e-5, 1.4e-5, 2.17e-5) kg·m²
  - kf: 2.350347298350041e-08 N/(rad/s)²
  - km: 7.24e-10 N·m/(rad/s)²
  - arm_length: 0.046 m → d = 0.0325 m offset
  - max_rotor_speed: 2315 rad/s
- [ ] **B4.** Disable Genesis's built-in PID controller. Action goes directly through our ported CTBR layer (next section).
- [ ] **B5.** Observation extraction:
  - `e^W` (30): same as MJX, computed from drone position vs trajectory in `trajectories.py`. World-frame.
  - `v` (3): `drone.get_vel()`. Verify world-frame (Genesis convention) matches our MJX convention.
  - `R` (9): `quat_to_R(drone.get_quat()).reshape(-1)`. Body→world rotation, row-major.
  - `k` (1, critic-only): episode step / max_steps.
- [ ] **B6.** Termination: same as M1 — bounds check on position, attitude angle, episode length.
- [ ] **B7.** **Validation against MJX:** With matched initial state, matched constants, zero action, run 100 steps in both sims. Final position should agree to &lt; 1e-3 m, final velocity &lt; 1e-3 m/s. Hover under constant thrust for 1 second — height drift should match within 1%.

### C. Control — port `control/ctbr_controller.py`

- [ ] **C1.** New module `src/iron_man_drone/control/ctbr_controller_genesis.py`. Same public API but Torch-tensor-backed.
- [ ] **C2.** Rate-PD on body rates. Use the same gains as M1 (transcribe from current MJX implementation — single source of truth).
- [ ] **C3.** X-config mixer: `(T_total, τ_x, τ_y, τ_z) → (T1, T2, T3, T4)`. Numeric inversion of the same 4×4 matrix from M1.
- [ ] **C4.** **First-order motor lag.** Maintain a per-env Torch tensor `omega_state` of shape `(n_envs, 4)`. Per step: `omega_state += (omega_cmd - omega_state) * (dt / motor_tau)`, with `motor_tau = 0.025 s`. Genesis does not do this internally; the asymmetric site-force trick from MJX does not exist in Genesis. Without this we lose sim-to-real validity for the recipe.
- [ ] **C5.** `ω = √(T / kf)` per rotor, clipped to `max_rotor_speed`.
- [ ] **C6.** `drone.set_propellers_rpm(omega_state * 60 / (2π))` (Genesis takes RPM, not rad/s).
- [ ] **C7.** **Validation:** Send unit body-rate command. Verify drone rotates at that rate after rate-PD settles. Compare against MJX response curve.

### D. Trajectories — keep `envs/trajectories.py`

Pure NumPy/JAX, no sim coupling. **No changes required.** The trajectory generator outputs world-frame reference points; the env consumes them.

- [ ] **D1.** Confirm import path still works under the Genesis env. NumPy operations may need to switch to Torch for tensor concat with the rest of the obs.

### E. PPO trainer — keep the Flax stack

This is the load-bearing decision: **do not rewrite the PPO trainer in Torch.** Reasons:
- M1.3 and M2 results depend on the Flax asymmetric actor/critic + separate Optax optimizers. The current `policy/networks.py` and `policy/ppo.py` are validated.
- A Torch rewrite re-introduces all the bugs that M1.1–M1.3 found and fixed in the Flax version.
- The DLPack bridge keeps the policy on JAX, the sim on Torch.

- [ ] **E1.** New module `src/iron_man_drone/policy/ppo_genesis.py`. Wraps existing Flax PPO loop. Per step:
  1. `obs_torch = env.get_obs()` — Torch CUDA tensor.
  2. `obs_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(obs_torch))` — zero-copy.
  3. `action_jax, value_jax = actor_critic(obs_jax)` — existing Flax forward.
  4. `action_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(action_jax))` — back to Torch.
  5. `env.step(action_torch)`.
- [ ] **E2.** Rollout storage in JAX (existing `PureJaxRL` pattern) — pull obs/action/reward batches into JAX arrays via DLPack and `lax.scan` over training updates.
- [ ] **E3.** **Validation:** A 100-step rollout on a hover task should match wall-clock-per-step to within 2× of pure-MJX PPO. If 5× slower, something is wrong with the bridge.

### F. Domain randomization — minimal port

M2.5 deliberately drops M2's fault-tolerance DR. We only carry the M1 baseline DR.

- [ ] **F1.** Per-episode `kf` scaling: `kf_actual = kf_nominal × U(0.70, 1.30)`. Same range as M1.
- [ ] **F2.** Genesis batched mass setter exists in v0.4.6 but is undocumented. Read `genesis/engine/entities/rigid_entity/rigid_link.py` to confirm. Validate by setting one env's mass to 2× and confirming its drop-rate matches.
- [ ] **F3.** Skip per-rotor fault DR. Skip wind. Skip mass DR. M2.5 is a baseline, not a robustness milestone.

### G. Eval — keep `scripts/eval_*.py`

- [ ] **G1.** New `scripts/eval_m25.py` — same trajectory suite as M1 (figure_eight_slow/normal/fast, pentagram_slow, etc.). Reuses `envs/trajectories.py` directly.
- [ ] **G2.** **Use T/4 phase offset for figure-eight** (M1.3 eval methodology fix). Do not relitigate.
- [ ] **G3.** MED computed as arithmetic mean over full 1000-step episode. Same definition as M1.3.

### H. Sanity checks (pre-training, gating)

These mirror M1's `sanity_check.py` gates. **None may be skipped.**

- [ ] **H1.** Entropy &lt; 10% of reward magnitude at init. Project rule.
- [ ] **H2.** Reward at random init within 2× of M1.3's random-init reward (no off-by-one bugs).
- [ ] **H3.** Action distribution: PPO actor output mean ≈ 0, std ≈ 1 at init (no NaN, no saturation).
- [ ] **H4.** Trajectory coverage on training distribution: at least 95% of figure-eight apex curvatures (κ ≈ 4.789 m⁻¹) covered by training samples (per M1.3 polynomial fix).
- [ ] **H5.** Throughput: training (sim only, no render) ≥ 30k env-steps/sec at n_envs=1024.

---

## Success criteria — Go/No-Go for M3

M2.5 ships if and only if **all** of the following pass:

| Metric | Target | Justification |
|---|---|---|
| figure_eight_normal MED | **≤ 0.060 m** | M1.3 hit 0.037 m, but M2.5 is a port not an optimization. ~1.6× M1.3 budget for port-introduced variance. **Below 0.060 m is the threshold.** |
| figure_eight_slow MED | ≤ 0.030 m | M1.3 hit 0.017 m; same 1.6× budget |
| figure_eight_fast MED | ≤ 0.130 m | M1.3 hit 0.090 m |
| Pentagram_slow MED | ≤ 0.080 m | M1.3 hit 0.054 m |
| Zero crashes on full eval suite | hard | Crashes = failed port |
| Training throughput (no render) | ≥ 30k env-steps/sec @ n_envs=1024 | Below this, M3 wall-clock exceeds budget |
| MJX dynamics agreement | 100-step zero-action rollout matches MJX to &lt; 1e-3 m position, &lt; 1e-3 m/s velocity | Validation gate, runs before any PPO training |

**Why not "match M1.3 exactly (0.037 m)":** A new simulator port introduces dynamics variance (contact stiffness, integrator differences, motor-lag implementation). Demanding M1.3-exact would over-constrain the port; demanding M1.3 + 60% provides a clear "did it work" signal while tolerating expected port noise. If M2.5 matches 0.037 m, even better — but it's not the gate.

---

## Failure modes

### F1 — Spike fails on G1 (install) or G4 (DLPack bridge)
**Action:** Abort migration. Fall back to **madrona_mjx**. Keep MJX, add depth via the same Madrona renderer Genesis uses, no sim migration. Less mature, but the Madrona kernel is the same.

### F2 — Spike passes but MJX-vs-Genesis dynamics disagreement > 1%
**Likely cause:** Motor lag implementation mismatch, or Genesis's rigid solver using different damping. **Action:** Stop and diagnose. Do not train. Walk through `omega_state` evolution step-by-step under a constant command and verify the discrete update.

### F3 — Training throughput < 10k env-steps/sec
**Cause candidates:** DLPack bridge has overhead, Genesis substep too small, n_envs too large for memory. **Action:** Profile each. If bridge is the culprit, try reducing rollout batch size; if substep, validate with substep=2 and accept the throughput hit; if memory, drop n_envs to 512.

### F4 — Training diverges (MED &gt; 1.0 m at epoch 5k)
**Likely cause:** Reward off-by-one bug re-introduced in port, or observation tensor layout mismatch (Genesis batches differently than MJX). **Action:** Apply M1.3 reward fix manually — verify `state.step` vs `new_step` ordering matches. Verify observation tensor shapes log-by-log.

### F5 — Genesis v0.4.6 has an undocumented breaking change vs our pinned config
**Symptom:** Code that worked yesterday breaks today after `pip install`. **Cause:** Someone unpinned the version, or a Genesis dep pulled a newer release.
**Action:** Re-pin everything in `requirements.txt`. Log the breakage in `notes/lessons.md` for L8.

### F6 — Time-box hit (2 weeks elapsed, M2.5 not converged)
**Action:** Stop. Do not extend. Fall back to madrona_mjx or stay on MJX without depth (and rescope M3 to use a low-dim observation stand-in for depth — e.g., a 16-bin obstacle-distance vector computed analytically from a parametric obstacle world, no rendering required). **Re-write M3_spec.md to reflect the fallback.**

---

## Time-box

| Phase | Wall-clock | Cumulative |
|---|---|---|
| Spike (G1–G6) | 3 days | 3 d |
| Env port (B + C) | 3 days | 6 d |
| Trainer bridge (E) | 2 days | 8 d |
| Sanity (H) | 1 day | 9 d |
| Training run (15k epochs at lower throughput) | 1–2 days | 10–11 d |
| Eval + write-up | 1 day | 11–12 d |
| Buffer | 2 days | 13–14 d |

**Hard time-box: 14 calendar days from the day the spike passes.** Past 14 days, the project switches to a fallback path. This protects M3's calendar budget.

---

## What we explicitly do NOT do in M2.5

- **Visual perception.** That's M3.
- **Obstacles.** That's M3.
- **RMA encoder `μ` / `ϕ` port.** Robustness lives in M2; M2.5 is just M1.3-equivalent. Re-introducing RMA on Genesis is its own port, deferred to "post-M2.5 evaluation" — likely rolled into M3 directly.
- **Fault tolerance DR.** Drop, by design. M3 spec answers whether to keep it.
- **Real hardware deployment.** M5+.
- **Performance optimization beyond throughput gate.** Don't tune Genesis for speed beyond 30k env-steps/sec; that buys nothing in the port-validation context.

---

## Fallback decision tree

```
Spike (G1–G6 in 3 days)
├── Pass → Proceed with Genesis port (14-day time-box)
│   ├── Port converges → M2.5 ships, M3 proceeds on Genesis
│   └── Port fails (time-box hit or F1–F5) → Fall back to madrona_mjx
└── Fail → Choose fallback:
    ├── madrona_mjx — same renderer, JAX stack preserved. Try first.
    ├── Aerial Gym (Isaac Gym) — full Torch rewrite of policy. Only if madrona_mjx fails too.
    └── Analytic depth (no rendering) — last resort. M3 scope shrinks: obstacles are parametric shapes with analytic distance, no renderer at all. Defers visual perception to M4.
```

---

## Estimated time budget

| Item | Estimate |
|---|---|
| Genesis spike (gating) | 3 days |
| Full port if spike passes | 11 days |
| **Total (best case)** | **14 days** |
| Fallback to madrona_mjx (if spike fails) | 7–10 days |
| Fallback to analytic-depth (last resort) | 5 days, scope reduction |

**This is the upper-bound budget. The spike is the cheapest insurance against the worst case** — if Genesis-on-WSL2-on-4070 doesn't work, we know in 3 days, not 3 weeks.
