# M3 Spec — Visual Reactive Obstacle Avoidance

**Date:** 2026-05-10
**Prerequisite reading:** `notes/M3_research_summary.md`, `notes/M3_genesis_assessment.md`, `notes/M2_5_genesis_port_spec.md`, `notes/M2_spec.md`, `notes/lessons.md`
**Status:** Draft — requires hypothesis doc approval and M2.5 baseline pass before any code is written.

---

## One-line goal

A single PPO policy that, given a forward-facing depth observation, **avoids static obstacles** while **preserving M1.3-equivalent trajectory tracking** when the world is obstacle-free. Trained on procedural cluttered scenes in Genesis. Reactive, no planning, no backtracking. Trained speed cap 3 m/s; eval stretch to 4–5 m/s.

---

## Pre-conditions (gates before any M3 code)

- [ ] M2.5 (Genesis port) ships: figure_eight_normal MED ≤ 0.060 m on Genesis, zero crashes on full M1 eval suite. See `M2_5_genesis_port_spec.md` for full criteria.
- [ ] `notes/M3_hypothesis.md` written by hand (gating artifact, by project rule).
- [ ] Sanity check `scripts/sanity_check_m3.py` passes — including the new depth-obs-not-NaN gate and entropy &lt; 10% reward gate.

**No M3 training kicks off until all three pre-conditions are green.**

---

## What carries over from M2.5 baseline

- C2-continuous quintic polynomial trajectory generator (`envs/trajectories.py`)
- Reward off-by-one fix (verified on Genesis port)
- T/4 phase offset for figure-eight eval
- Asymmetric actor/critic (3-layer MLP, 256 hidden, ELU + LayerNorm)
- Separate Flax modules with separate Optax optimizers (preserved via DLPack bridge from M2.5)
- PPO hyperparameters (γ=0.99, λ=0.95, clip=0.2, entropy_coeff=1e-3, actor_lr=3e-4, critic_lr=1e-4) as starting point
- CTBR action space — **never motor commands**, project rule
- Rotation matrix in obs — never quaternion, project rule
- No u_{t-1} in actor (project rule; Swift includes it, we don't)
- No k in actor — critic only (project rule)
- Existing k_f ± 30% DR (carried from M1, kept in M2, kept in M3)
- All eval scripts and notes structure

---

## What changes for M3

Three additions, all introduced together at epoch 0 — no fine-tuning a converged M2.5 policy with obstacles (this is the rule that killed v1). Per project rule, bake via DR from epoch 0.

### A. Depth observation

**Sensor model in Genesis:**

- Forward-facing depth camera, mounted at drone CoM, **0° pitch** (matches Loquercio).
- Resolution: **64×64** (Madrona-comfortable, low-bandwidth).
- HFOV: **90°** (matches Loquercio's 91° within rounding).
- Max range: **5.0 m** (depth clipped beyond).
- Update rate: **30 Hz** (every 3 policy steps, since policy runs at 100 Hz). Matches GRaD-Nav onboard rate; cheaper than rendering every 10 ms.
- Render path: `gs.renderers.BatchRenderer(use_rasterizer=False)` (Madrona CUDA single-bounce). Verify depth values via the spike per `M3_genesis_assessment.md` §3.
- **Noise injection:** add Gaussian noise σ=0.02·d per pixel, plus 1% dropout (set to max_range, "no return"). Mirrors Loquercio's SGM-noisy depth — the key sim-to-real lever. **Do not train on perfect depth.**

**Actor input: min-pooled depth bins, not raw image.**

Following Swift's evidence (12-number perception summary is sufficient for champion-level reactive flight), the actor consumes a low-D summary of the depth image, not the raw tensor:

- Divide horizontal FOV into **K=16 angular bins** (each ~5.6° wide).
- Per bin, the **minimum depth** across the bin's pixel column, **min-pooled vertically** as well (single nearest-distance per bin).
- Yields a **16-dim depth feature `d_bins`**, body-frame.
- Normalize: `d_bins_normalized = d_bins / d_max` ∈ [0, 1], with `d_max = 5.0`.

**Why not raw 64×64 depth into a CNN:**
1. Swift's perception-summary lesson: thin causal-sufficient summaries work.
2. Min-pooled depth is causal-sufficient for reactive obstacle avoidance (the policy needs to know "how close is the nearest obstacle in each direction," not the obstacle's shape).
3. PPO is sample-inefficient with high-D inputs. Loquercio needed imitation learning to make 224×224 tractable. Our budget doesn't support that pivot.
4. Reusable for sim-to-real: D435 → min-pool externally → policy. No CNN weights to worry about transferring.
5. If min-pooled stalls below the success threshold (see §F2), we upgrade to a depth-CNN — but that's a contingency, not the default.

**Actor obs dim:**

```
actor_obs = [e^W (30), v (3), R (9), d_bins (16)] = 58-dim
```

The M2 RMA `z` (8-dim privileged encoder output) is **dropped** for M3. Rationale: see §F1.

**Critic obs dim:**

```
critic_obs = [actor_obs (58), k (1), priv_obstacle_summary (8)] = 67-dim
```

Critic gets an **asymmetric privileged obstacle summary** (analogous to M2's privileged `e_t`):
- Distance to nearest obstacle (1)
- Direction to nearest obstacle in body frame (3)
- Number of obstacles within 3 m (1)
- Mean obstacle radius within 3 m (1)
- Time-to-collision under current velocity (clipped at 5 s) (1)
- Free-space margin in current velocity direction (1)

8-dim. Per project rule, this stays asymmetric — actor never sees it.

### B. Procedural obstacle environment

**Training scene generator** (`envs/obstacles.py`, new):

- Obstacles are vertical cylinders. Single shape primitive. Adapted from Loquercio's "tree" approach.
- Per episode:
  - **Density:** Poisson-distributed count over a 20×20 m flat region, with average **0.4 cylinders/m² × 20² m² = 160 cylinders**. Loquercio at 7 m/s used 4 m grid spacing → 0.0625 cyl/m²; at our 3 m/s cap, denser is fine. **6.4× Loquercio's density** at half the speed gives similar dwell-time per obstacle.
  - **Radius:** U(0.1, 0.4) m per cylinder. Loquercio used U(0.5, 5.0) m for generic obstacles — we go smaller to keep the figure-eight footprint (radius ~1 m) reachable.
  - **Height:** 3.0 m (vertical, full ceiling-to-floor). Drone z-range is ~0–2 m, height is non-issue.
  - **Trajectory placement:** generate reference trajectory first (figure-eight or polynomial), **then** reject cylinder placements within `clearance_buffer = drone_radius + 0.15 m` of the reference. Guarantees the nominal trajectory is feasible.
- **Per-episode randomization:** new seed per episode; obstacle layout differs every episode. 1024 envs × 15k epochs = 1024 simultaneous diverse layouts.

**Curriculum: no obstacle density curriculum.** Loquercio's evidence: fixed-density training generalized 1–10 m/s. We default to no curriculum and watch for the failure mode.

**Empty-scene episodes (preservation training):** With probability **p_empty = 0.20**, sample an empty scene (no obstacles). Forces the policy to still see "no-obstacle" data after introducing the depth obs, preserving the M2.5 figure-eight ability. Without this, the depth bin distribution shifts so far from "all bins at max range" that obstacle-free flight regresses.

**Demo scene capability:** `scripts/build_demo_scene.py` constructs a hand-placed obstacle field for video. Same env code, different obstacle initializer. **Not trained on; eval only.**

### C. Reward shaping (additive to M1's tracking reward)

Current M1 reward (kept unchanged):
- Tracking: `-||e^W[0]||₂` per step, plus action smoothness penalty.

**Two new terms only.** Single-term-at-a-time rule: don't introduce energy, attitude, perception-visibility, or other shaping at M3. Two terms are the minimum needed; more is debugging surface.

**C1. Soft clearance penalty:**

```
r_clear = -w_clear × max(0, d_safe - d_min)²
```

Where `d_min = min(d_bins)` (the closest obstacle in any direction), `d_safe = 0.5 m` (the soft margin), `w_clear` is the coefficient.

- Quadratic to give a smooth gradient near the boundary (PPO friendly).
- Zero contribution when no obstacle is within `d_safe` — preserves M1 reward shape in obstacle-free flight.
- **Coefficient `w_clear` tuning:** target ≤ 10% of the tracking reward at peak (when drone is at d_min = 0 from an obstacle, just before crash). Tracking reward magnitude ~1.0 at peak under nominal flight; clearance at d=0 = `w_clear × 0.25`. So `w_clear ≈ 0.4` puts clearance at 10% of tracking at the worst case. **Pre-training check:** verify on a 1000-random-sample distribution that `r_clear / r_track < 0.10` in 95% of cases.

**C2. Hard collision termination:**

- On contact with any obstacle (or floor/ceiling): terminal reward `-100`, episode ends.
- Same magnitude as M1's existing crash penalty.

**No perception-visibility term.** Swift used one because their gate detector needs the gate in FOV; M3 is reactive depth, no specific feature to maintain. If FOV becomes a limit (drone consistently flies into obstacles outside the 90° cone), reconsider in M3.x.

---

## Action space and policy

**Action space: unchanged from M1/M2/M2.5.** CTBR 4-dim.

**Policy architecture:** unchanged from M2.5. Linear(58→256) → ELU → LayerNorm → Linear(256→256) → ELU → LayerNorm → Linear(256→4). Critic equivalent shape with 67-dim input and scalar output.

**Why no perception encoder:** the 16-dim depth bins are already low-D. A CNN-encoded latent is unnecessary. If min-pooled stalls, the contingency is a small 1-D CNN over the K bins (treating them as a sequence) — see §F2.

---

## Training schedule

- Algorithm: PPO (identical to M2.5).
- Duration: **20,000 epochs** (vs M2.5's 15k). The +5k buffer accounts for the harder optimization landscape from obstacle DR.
- Eval every 1000 epochs (denser than M2's 500 is unnecessary; eval is more expensive with rendering).
- Convergence signal: tracking reward trending up, entropy slow-decreasing, MED in obstacle-free figure-eight stable, **success rate on cluttered eval rising**.
- **Do NOT pause and resume** (L1, lessons.md). Run uninterrupted.

**Abort gates (cluttered + nominal):**

| Epoch | Nominal MED (figure_eight_normal) | Success rate (cluttered, 3 m/s) | Action if exceeded |
|---|---|---|---|
| 5,000 | > 0.150 m | < 30% | Stop and diagnose. Most likely: obstacle DR too aggressive, or depth obs not wired correctly. Re-read §F. |
| 10,000 | > 0.100 m | < 60% | Stop. The policy is not converging on the joint task. |
| 15,000 | > 0.070 m | < 75% | Stop. Trajectory tracking is regressing; obstacle avoidance is plateauing. |
| 20,000 | **≤ 0.060 m** | **≥ 85%** | Pass — eval and ship. |

All gates are on **obstacle-free** MED for nominal tracking. Cluttered eval is a fresh hold-out seed range.

**Trend gate:** improvement between 10k and 15k must be ≥ 5% on both metrics. A policy that is flat over 5,000 epochs has stopped learning.

---

## Eval suite

### Tier 1: Tracking preservation (M2.5 carry-over)

**Hard requirement: no regression from M2.5 baseline.**

| Trajectory | Target | Notes |
|---|---|---|
| figure_eight_slow | MED ≤ 0.030 m | M2.5 baseline |
| figure_eight_normal | MED ≤ 0.060 m | M2.5 baseline |
| figure_eight_fast | MED ≤ 0.130 m | M2.5 baseline |
| pentagram_slow | MED ≤ 0.080 m | M2.5 baseline |

Run on a fully obstacle-free scene. Confirms that adding depth observation + clearance reward does not break clean trajectory following.

### Tier 2: Cluttered avoidance

| Scenario | Parameter | Target |
|---|---|---|
| Cluttered nominal | 3 m/s straight line, 50 held-out seeds, default density | ≥ 85% success, ≥ 0.3 m mean min-clearance |
| Cluttered slow | 1.5 m/s, 50 seeds | ≥ 95% success |
| Cluttered stretch | 4 m/s, 50 seeds | ≥ 70% success (stretch target) |
| Cluttered far stretch | 5 m/s, 50 seeds | ≥ 50% success (diagnostic only, OOD) |

**Success** = drone travels ≥ 15 m along reference direction without collision and without exceeding 2× nominal MED (tracking is still happening, not just survival).

### Tier 3: Combined task

| Scenario | Description | Target |
|---|---|---|
| Figure-eight + obstacles | Standard figure-eight but with 30 obstacles inside the 4×2 m envelope, none within `clearance_buffer` of reference | MED ≤ 0.080 m, success rate ≥ 75% |

### Tier 4: Demo

Hand-built structured scene for video. **Not a go/no-go**, but expected to look qualitatively reasonable.

---

## Success criteria — Go/No-Go for M4

M3 ships if and only if **all** of the following pass:

(a) **Tracking preservation** — hard requirement, no regression from M2.5:
- [ ] figure_eight_normal MED ≤ 0.060 m on obstacle-free scene
- [ ] Zero crashes on Tier 1 eval

(b) **Cluttered nominal** — primary M3 goal:
- [ ] Cluttered nominal (3 m/s) ≥ **85% success**
- [ ] Mean min-clearance during successful traversals ≥ 0.30 m

(c) **Stretch (not gating, diagnostic):**
- [ ] Cluttered stretch (4 m/s) ≥ 70%

**Partial pass rules:**
- Tracking preserves, cluttered fails → fail M3, do not ship. Diagnose §F2 / §F3 before iterating.
- Cluttered passes, tracking regresses → fail M3. The depth obs is not "absent" enough in obstacle-free scenes; check `p_empty`, check the min-pooled distribution.
- Both pass at 3 m/s, stretch fails → ship M3, log stretch as known limit, move M4 forward.

---

## Open design decision — fault tolerance preservation

**The question:** M2 shipped RMA (single-rotor faults + mass DR + privileged encoder `μ` + adaptation encoder `ϕ`). M3 drops this. Should we keep it?

**Argument for keeping RMA in M3:**
- It's already trained. The `μ` encoder weights exist.
- Real-hardware deployment will have rotor and mass variance.
- Dropping it means a future M3.x or M4 has to retrain RMA *with* obstacles, which is exactly the "fine-tune new capability onto converged policy" anti-pattern we reject.

**Argument for dropping RMA in M3 (the proposed plan):**
- Adding depth observation + obstacle DR + clearance reward is already three changes. Adding fault-DR is a fourth, and increases the optimization surface to something we may not be able to debug.
- The M2 fault-tolerance encoder was **tuned for fault DR**. Mixing depth obs with `z` adds two interacting representation spaces in the actor; risk of `z` collapse (M2's failure mode F2) is higher.
- Iron Man's M3 demo task is obstacle avoidance, not "obstacle avoidance under rotor faults." Keep scope minimal.

**Resolved (this spec):** **Drop M2 RMA for M3.** Train M3 obstacles-only on top of M2.5. Plan an M3.5 or M4 milestone that re-introduces RMA via a single co-trained run (all DR axes — k_f, mass, rotor faults, obstacles — bake from epoch 0, no sequential fine-tune). The M2 encoder weights are archived but not used by M3.

**This decision is reversible.** If M3 converges quickly and we have wall-clock left, attempt an M3-with-RMA run as an ablation. If it converges, great. If it doesn't, the unfusedM3 still ships.

---

## Failure modes to watch

### F1 — Obstacle-free tracking regression

**Symptom:** Tier 1 MED > 0.080 m (M2.5 baseline + 33%) at any checkpoint.
**Cause candidates:** (a) `p_empty` too low — depth obs distribution shift breaks tracking; (b) `w_clear` too high — clearance penalty dominates even when no obstacles present; (c) actor over-relying on noisy depth bins instead of `e^W`.
**Diagnostic:**
- Log distribution of `d_bins` per training batch — at `p_empty = 0.20`, ~20% of samples should have all bins at `d_max`.
- Log `r_track` vs `r_clear` per step during eval — `r_clear` should be 0 in obstacle-free scenes.
- Mask depth bins to constant `d_max` at eval time and re-run Tier 1; if MED recovers, the depth obs is the culprit.
**Fix:** raise `p_empty` to 0.30. Reduce `w_clear` by 50%. Single-variable rule.

### F2 — Cluttered avoidance plateau (min-pooled too thin)

**Symptom:** Cluttered nominal success rate stalls at 50–70%, doesn't reach 85%.
**Cause candidates:** (a) 16 bins is too coarse — small obstacles vanish in min-pool; (b) max range 5 m is too short for 3 m/s closing; (c) policy lacks temporal context to anticipate.
**Diagnostic:**
- Crash cases: dump depth obs + action + clearance for 20 crashes. Is the depth obs showing the obstacle in advance, or only at the last bin?
- Try K=32 bins (2× resolution).
- Try `d_max = 8 m`.
**Fix path:** raise K, raise `d_max`. If still stalled, **upgrade to a 1-D CNN over depth bins** (treats bins as a sequence). If still stalled, upgrade to raw 64×64 depth + small CNN encoder. **Each upgrade is a separate run with a new hypothesis doc.**

### F3 — Crash during early training (depth obs not converged)

**Symptom:** Crash rate > 50% in epochs 1–500.
**Cause:** untrained policy + dense obstacles + no curriculum = no positive reward signal.
**Diagnostic:** crash rate per epoch curve. If it's monotonically declining, ignore; if flat, intervene.
**Fix:** **add a density curriculum** — linear ramp from 0.0 → full density over epochs 0–3000. This violates "no curriculum" only if we declare it as a deviation; flag in hypothesis doc.

### F4 — Depth obs is noisy in a way that breaks the policy

**Symptom:** Policy works fine in sim with `noise=0` ablation, fails with the planned σ=0.02·d + 1% dropout.
**Cause:** noise model isn't matched to Genesis's depth output distribution.
**Diagnostic:** plot histograms of `d_bins` with noise on vs off.
**Fix:** retune noise model. Should not be a blocker but is a known sim-to-real lever to revisit before M5.

### F5 — Genesis BatchRenderer depth values inconsistent (issue #1648)

**Symptom:** Depth values disagree with computed ground-truth ranges. Open Genesis bug.
**Diagnostic:** sanity check at every Genesis version bump — render a known-distance plane, verify depth value.
**Fix:** stay on the validated Genesis version (pinned in M2.5 spec). Do not upgrade Genesis during M3 training.

### F6 — Resume-after-pause (L1 lessons.md)

**Prevention:** same as M2. Do not pause; if unavoidable, save full optimizer state, prefer restart over resume past 30% of training.

### F7 — Eval methodology bug (L4, M1.3 lesson)

**Watchpoint:** any new eval scenario inherits the T/4 phase offset for figure-eight. New cluttered eval scenarios use a defined random seed range, distinct from training seeds. Document this in the eval script header.

---

## Out of scope for M3

- Dynamic obstacles (moving cylinders, other drones). M4+.
- RGB observation. Env supports it (per M3 design decision), but training is depth-only. M5+.
- Outdoor / GPS-denied scenes. Sim-only, controlled lighting. M5+.
- Real hardware deployment. M5+.
- Multi-task / language conditioning (GRaD-Nav++ style). Defer.
- Sim-to-real residual model (Swift style). Plan for M5; not built in M3.
- Fault tolerance during obstacle flight (see "Open design decision" above). M3.5 or M4.

---

## Estimated time budget

| Phase | Task | Wall-clock |
|---|---|---|
| Pre-training | Obstacle env code, depth obs pipeline, sanity check | 3 days |
| Pre-training validation | Spike: render depth → forward through actor → reward computed correctly on 100 random states | 1 day |
| Training (15k → 20k PPO epochs at depth-rendering throughput) | | 2–3 days |
| Eval suite | Full 4-tier eval | 1 day |
| Diagnostics + iterate | Buffer for at least one F-mode debug cycle | 3 days |
| Write-up | M3_results.md | 1 day |
| **Total** | | **~10–12 days of focused work** |

3-week calendar budget. Most of the buffer is debugging time.

---

## Closing reminders

- Project rule: **no training without a written `M3_hypothesis.md`**. The template is at `notes/M3_hypothesis_template.md`. The user fills it in by hand.
- Project rule: **one variable changed per run**. If F-modes fire, fix one thing per recovery run, write a new hypothesis doc, re-train.
- Project rule: **eval MED on held-out trajectories is the primary success signal** (L2). Training reward is supporting context only.
- Project rule: **lessons L1–L7 apply** — resume risks, training reward ≠ eval, polynomial coverage, eval methodology bugs, encoder startup, verify-before-react, privileged-state tractability. The privileged obstacle summary in the critic must be tractable: each of its 8 dimensions has a clear analytic value at any sim step, not a learned summary.
